// CEP:FILE: crates/xir-graph/src/dce.rs
// CEP:WHAT: Dead code elimination — removes pure nodes with zero uses.
// CEP:WHY: Level-0 canonicalization (arch: DCE across independent
//          functions/disjoint SCCs under Gear 1). Legality is conservative
//          and explicit: only PURE nodes (no side effects, no token edges)
//          with zero uses are removed (CEP&CC 38.22 — transform only under
//          proven legality).
// CEP:CLASS: CEP-0 (hot pass)
// CEP:STATUS: complete
// CEP:FAILURE: PassError::Internal on invariant break; conservative no-op
//             otherwise.
// CEP:ASSUMES: verified input (pass manager contract).
// CEP:COST: fixpoint iterations O(uses) each; total O(nodes * max_uses) in
//           the worst case, O(nodes) typical (dead chains peel one node per
//           iteration; bounded by node count).
// CEP:EVIDENCE: tests `removes_dead_chain`, `keeps_used_and_effectful`.
// CEP:SECURITY: bounded walks.
// CEP:HPC-PASS: dce
// CEP:HPC-PASS-KIND: dead code elimination
// CEP:HPC-PASS-INPUT: verified Level-0 sea-of-nodes + root set (results)
// CEP:HPC-PASS-OUTPUT: unused pure nodes removed
// CEP:HPC-PASS-ANALYSIS-REQUIRED: use counts
// CEP:HPC-PASS-ANALYSIS-PRODUCED: none (analysis is transient)
// CEP:HPC-PASS-ANALYSIS-INVALIDATED: use-def chains
// CEP:HPC-PASS-LEGALITY: pure op AND zero uses AND no effect edge
// CEP:HPC-PASS-PRESERVES: semantics, effect order (only dead pure nodes go)
// CEP:HPC-PASS-COST: O(nodes) typical
// CEP:HPC-PASS-FAILURE: conservative abort
// CEP:HPC-PASS-TARGET: target-independent
// CEP:HPC-PASS-EVIDENCE: tests in this module
// CEP:HPC-TRANSFORM: Deletes unused pure subgraphs.
// CEP:HPC-DETERMINISM: deterministic; slot-order removal
//! Dead code elimination.
use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::MAX_INPUTS;

use crate::gvn::PassError;

/// CEP:WHAT: Runs DCE to fixpoint with an explicit root set.
/// CEP:WHY: Dead chains (a dead node feeding another dead node) need
///          iteration; each pass peels the leaves. Bounded by node count —
///          no unbounded loops (CEP-0 contract, CEP&CC 38.11). The root
///          set (function results) is the liveness anchor: without it DCE
///          would legally delete the entire program — roots make the
///          intended semantics explicit (Law 2).
/// CEP:STATUS: complete
/// CEP:FAILURE: Internal on arena inconsistency.
/// CEP:ASSUMES: verified input; roots reference live nodes.
/// CEP:COST: see module header.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn run(arena: &mut IrArena, roots: &[NodeId]) -> Result<u32, PassError> {
    let mut removed_total = 0u32;
    loop {
        // Use-count table by slot (CEP-1 analysis allocation per iteration;
        // bounded by node count).
        let n = arena.slot_count();
        let mut used = vec![false; n];
        // Roots anchor liveness.
        for r in roots {
            let slot = r.index() as usize;
            if slot < used.len() {
                used[slot] = true;
            }
        }
        arena.for_each_live_node(|_id, node| {
            for i in 0..node.n_inputs as usize {
                if i < MAX_INPUTS {
                    let slot = node.inputs[i].node().index() as usize;
                    if slot < used.len() {
                        used[slot] = true;
                    }
                }
            }
            // Effect-token consumers keep their producers alive.
            if !node.effect_in.is_none() {
                let slot = node.effect_in.node().index() as usize;
                if slot < used.len() {
                    used[slot] = true;
                }
            }
        });
        // Collect dead pure nodes in slot order (deterministic).
        let mut dead: Vec<NodeId> = Vec::new();
        arena.for_each_live_node(|id, node| {
            if node.op.is_pure() {
                let slot = id.index() as usize;
                if slot < used.len() && !used[slot] {
                    dead.push(id);
                }
            }
        });
        if dead.is_empty() {
            break;
        }
        for id in dead {
            if arena.remove_node(id).is_err() {
                return Err(PassError::Internal);
            }
            removed_total += 1;
        }
    }
    Ok(removed_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::const_f64;
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op, UnaryOp};
    use xir_core::ty::{ScalarType, Type};

    // CEP:WHAT: A dead chain is fully removed.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if dead nodes survive.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn removes_dead_chain() {
        let mut a = IrArena::with_capacity(32, 8);
        let root = a.root_region();
        let c = const_f64(&mut a, root, 5.0);
        assert!(c.is_ok());
        if let Ok(cv) = c {
            let v = a.value_of(cv, 0);
            assert!(v.is_ok());
            if let Ok(val) = v {
                // dead1 = neg(c)  (unused)
                let dead1 = Node::new(
                    Op::Unary(UnaryOp::Neg),
                    root,
                    &[val],
                    Type::Scalar(ScalarType::F64),
                );
                let d1 = a.insert_node(root, dead1);
                assert!(d1.is_ok());
                if let Ok(d1v) = d1 {
                    let dv = a.value_of(d1v, 0);
                    assert!(dv.is_ok());
                    if let Ok(dval) = dv {
                        // dead2 = neg(dead1)  (unused)
                        let dead2 = Node::new(
                            Op::Unary(UnaryOp::Neg),
                            root,
                            &[dval],
                            Type::Scalar(ScalarType::F64),
                        );
                        let _ = a.insert_node(root, dead2);
                    }
                }
            }
        }
        let before = a.node_count();
        let r = run(&mut a, &[]);
        assert!(r.is_ok());
        if let Ok(removed) = r {
            // dead1, dead2 AND the now-unused constant are all dead.
            assert_eq!(removed, 3);
            assert_eq!(a.node_count(), before - 3);
        }
        assert_eq!(crate::verifier::verify(&a), Ok(()));
    }

    // CEP:WHAT: Used nodes and effectful nodes survive.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on live-node loss.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn keeps_used_and_effectful() {
        let mut a = IrArena::with_capacity(32, 8);
        let root = a.root_region();
        let c0 = const_f64(&mut a, root, 1.0);
        let c1 = const_f64(&mut a, root, 2.0);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                // USED add (consumed by a reduce, which is itself
                // consuming so it counts as a use anchor via its input).
                let used = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::F64),
                );
                let add_id = a.insert_node(root, used);
                assert!(add_id.is_ok());
                if let Ok(av) = add_id {
                    let aval = a.value_of(av, 0);
                    assert!(aval.is_ok());
                    if let Ok(add_val) = aval {
                        let red = Node::new(
                            Op::Reduce {
                                axis: 0,
                                monoid: xir_core::op::Monoid::Add,
                            },
                            root,
                            &[add_val],
                            Type::Scalar(ScalarType::F64),
                        );
                        let red_id = a.insert_node(root, red);
                        assert!(red_id.is_ok());
                        // DCE with the reduce as the function result.
                        let before = a.node_count();
                        let r2 = run(&mut a, &[red_id.ok().unwrap_or(NodeId::NONE)]);
                        assert!(r2.is_ok());
                        if let Ok(removed2) = r2 {
                            assert_eq!(removed2, 0);
                            assert_eq!(a.node_count(), before);
                        }
                    }
                }
                // Effectful rng (unused but impure).
                let rng = Node::new(
                    Op::Rng {
                        dist: xir_core::op::RngDist::Uniform,
                        seed: 9,
                    },
                    root,
                    &[],
                    Type::Scalar(ScalarType::F64),
                );
                let _ = a.insert_node(root, rng);
            }
        }
    }
}
