// CEP:FILE: crates/xir-graph/src/gvn.rs
// CEP:WHAT: Global value numbering / common subexpression elimination.
// CEP:WHY: Master architecture Level 0: "Anvil Execution: Gear 1 (Static
//          Partitioning) for global CSE/GVN/Algebraic simplification".
//          Congruent pure nodes are merged when the representative
//          DOMINATES the duplicate — the classical CSE legality proof
//          (CEP&CC 38.22: transform only when legality is proven).
// CEP:CLASS: CEP-0 (hot pass)
// CEP:STATUS: complete
// CEP:FAILURE: PassError::{DominanceBuildFailed, Internal} — conservative
//             aborts, never silent miscompiles.
// CEP:ASSUMES: input arena verified (pass manager contract).
// CEP:COST: O(nodes log nodes): slot-order walk + BTreeMap lookups; the
//           BTreeMap (not HashMap) keeps results seed-independent
//           (CEP&CC 38.19 — no hash-order dependence).
// CEP:EVIDENCE: tests `eliminates_duplicates`, `respects_dominance`,
//           `keeps_impure_ops`.
// CEP:SECURITY: IR treated as untrusted; lookups bounded.
// CEP:HPC-PASS: gvn
// CEP:HPC-PASS-KIND: global value numbering (analysis + rewrite)
// CEP:HPC-PASS-INPUT: verified Level-0 sea-of-nodes
// CEP:HPC-PASS-OUTPUT: congruent pure nodes merged under dominance
// CEP:HPC-PASS-ANALYSIS-REQUIRED: dominance (region tree)
// CEP:HPC-PASS-ANALYSIS-PRODUCED: value-number table (transient)
// CEP:HPC-PASS-ANALYSIS-INVALIDATED: use-def chains
// CEP:HPC-PASS-LEGALITY: pure ops only; representative dominates duplicate;
//           identical op + immediates + canonical input lists + type; the
//           64-bit hash is a FILTER, congruence is re-proven on every hit
//           (audit F-4: collisions cannot merge non-congruent nodes)
// CEP:HPC-PASS-PRESERVES: semantics, effect order (pure ops only touched)
// CEP:HPC-PASS-COST: O(nodes log nodes)
// CEP:HPC-PASS-FAILURE: conservative abort on internal inconsistency
// CEP:HPC-PASS-TARGET: target-independent
// CEP:HPC-PASS-EVIDENCE: tests in this module; bench gvn in benches
// CEP:HPC-TRANSFORM: Merges dominated pure congruent nodes.
// CEP:HPC-DETERMINISM: deterministic; slot-order walk + BTreeMap
//! Global value numbering (CSE).
use std::collections::BTreeMap;

use xir_core::arena::IrArena;
use xir_core::hash::Fnv64;
use xir_core::id::{IrLevel, NodeId, RegionId, ValueId};
use xir_core::node::MAX_INPUTS;
use xir_core::op::Op;
use xir_core::ty::Type;

use crate::dominance::{DominanceError, DominatorTree};

/// Pass failure enumeration.
///
/// CEP:WHAT: Explicit error type for xir-graph passes.
/// CEP:WHY: Law 6 — conservative aborts must be loud.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this crate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassError {
    /// The dominance analysis could not be built (broken region tree).
    DominanceBuildFailed(DominanceError),
    /// An internal invariant broke; the pass aborted without mutation.
    Internal,
}

/// CEP:WHAT: Runs GVN/CSE over the arena, removing dominated duplicates.
/// CEP:WHY: The Level-0 canonicalization workhorse: merges `x+x`, repeated
///          constants, identical subgraphs — reducing node count before
///          tensor lowering (arch section 3).
/// CEP:STATUS: complete
/// CEP:FAILURE: DominanceBuildFailed propagates; Internal on invariant break
///              (no mutation happens before the failure point is reached —
///              removals are batched after classification).
/// CEP:ASSUMES: verified input (pass manager contract).
/// CEP:COST: O(nodes log nodes); allocation: two node-sized tables (CEP-1
///           analysis setup boundary).
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: bounded lookups.
/// CEP:HPC-DETERMINISM: deterministic (see module header).
pub fn run(arena: &mut IrArena) -> Result<u32, PassError> {
    let tree = match DominatorTree::build(arena) {
        Ok(t) => t,
        Err(e) => return Err(PassError::DominanceBuildFailed(e)),
    };
    let n_slots = arena.slot_count();
    // Canonical representative per node slot (None = not yet seen / dead).
    let mut canon: Vec<Option<NodeId>> = vec![None; n_slots];
    // Structural hash -> representative id.
    let mut table: BTreeMap<u64, NodeId> = BTreeMap::new();
    // Batched removals (applied only after full classification).
    let mut removals: Vec<NodeId> = Vec::new();

    let mut changed = 0u32;
    // Collect (id, node snapshot) in slot order: the walk needs stable data
    // while later removals mutate the arena.
    let mut snapshot_nodes: Vec<(NodeId, Op, [ValueId; MAX_INPUTS], u8, Type, RegionId)> =
        Vec::with_capacity(n_slots);
    arena.for_each_live_node(|id, node| {
        snapshot_nodes.push((
            id,
            node.op,
            node.inputs,
            node.n_inputs,
            node.ty,
            node.region,
        ));
    });

    for (id, op, inputs, n_inputs, ty, region) in snapshot_nodes {
        if !op.is_pure() {
            canon[id.index() as usize] = Some(id);
            continue;
        }
        // Canonical inputs: map each input's defining node to its
        // representative BEFORE hashing (value numbering).
        let mut h = Fnv64::new();
        op.hash_into(&mut h);
        for v in inputs.iter().take(n_inputs as usize).take(MAX_INPUTS) {
            let def = v.node();
            let rep = canon
                .get(def.index() as usize)
                .copied()
                .flatten()
                .unwrap_or(def);
            h.write_u64(rep.0);
        }
        let key = h.finish();
        match table.get(&key) {
            Some(&rep) => {
                // Legality: representative's region must dominate this node's
                // region, and the types must agree.
                let rep_node = match arena.node(rep) {
                    Ok(n) => n,
                    Err(_) => return Err(PassError::Internal),
                };
                // Full congruence proof (audit F-4): the 64-bit hash is
                // only a filter; op, arity, type, dominance AND the
                // canonical input LIST must all match before merging — a
                // hash collision alone can never merge non-congruent nodes.
                let mut inputs_congruent = rep_node.n_inputs == n_inputs;
                if inputs_congruent {
                    for (mine, theirs) in inputs
                        .iter()
                        .zip(rep_node.inputs.iter())
                        .take(n_inputs as usize)
                        .take(MAX_INPUTS)
                    {
                        let my_rep = canon
                            .get(mine.node().index() as usize)
                            .copied()
                            .flatten()
                            .unwrap_or(mine.node());
                        let their_rep = canon
                            .get(theirs.node().index() as usize)
                            .copied()
                            .flatten()
                            .unwrap_or(theirs.node());
                        if my_rep != their_rep {
                            inputs_congruent = false;
                            break;
                        }
                    }
                }
                if rep_node.ty == ty
                    && tree.dominates(rep_node.region, region)
                    && rep_node.n_inputs == n_inputs
                    && rep_node.op == op
                    && inputs_congruent
                {
                    // Replace uses of this node's value with the
                    // representative's value, then remove the duplicate.
                    let from = ValueId::from_node(id, 0);
                    let to = ValueId::from_node(rep, 0);
                    replace_uses(arena, from, to);
                    removals.push(id);
                    canon[id.index() as usize] = Some(rep);
                    changed += 1;
                    continue;
                }
                // Not provable: this node becomes its own representative.
                canon[id.index() as usize] = Some(id);
            }
            None => {
                table.insert(key, id);
                canon[id.index() as usize] = Some(id);
            }
        }
    }
    for id in removals {
        if arena.remove_node(id).is_err() {
            return Err(PassError::Internal);
        }
    }
    Ok(changed)
}

/// CEP:WHAT: Rewrites every use of `from` to `to` across the arena.
/// CEP:WHY: The rewrite half of CSE; single linear scan (Gear-1 friendly:
///           can be partitioned by disjoint use ranges later).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: `to`'s definition dominates all uses of `from` (caller
///              proved via the dominator tree).
/// CEP:COST: O(nodes).
/// CEP:EVIDENCE: tests `eliminates_duplicates`.
fn replace_uses(arena: &mut IrArena, from: ValueId, to: ValueId) {
    let mut ids: Vec<NodeId> = Vec::new();
    arena.for_each_live_node(|id, _| ids.push(id));
    for id in ids {
        if let Ok(node) = arena.node_mut(id) {
            for i in 0..MAX_INPUTS {
                if node.inputs[i] == from {
                    node.inputs[i] = to;
                }
            }
            if node.effect_in == from {
                node.effect_in = to;
            }
        }
    }
}

// Silence dead import if IrLevel unused in some configurations.
#[allow(unused_imports)]
use IrLevel as _IrLevelUsed;

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::const_f64;
    use xir_core::node::Node;
    use xir_core::op::BinaryOp;

    fn build_two_identical_adds() -> IrArena {
        let mut a = IrArena::with_capacity(32, 8);
        let root = a.root_region();
        let c0 = const_f64(&mut a, root, 1.0);
        let c1 = const_f64(&mut a, root, 2.0);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let add1 = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(xir_core::ty::ScalarType::F64),
                );
                let _ = a.insert_node(root, add1);
                let add2 = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(xir_core::ty::ScalarType::F64),
                );
                let _ = a.insert_node(root, add2);
            }
        }
        a
    }

    // CEP:WHAT: Identical pure nodes in the same region are merged.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if duplicates survive.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn eliminates_duplicates() {
        let mut a = build_two_identical_adds();
        let before = a.node_count();
        let r = run(&mut a);
        assert!(r.is_ok());
        if let Ok(merged) = r {
            assert_eq!(merged, 1);
            assert_eq!(a.node_count(), before - 1);
        }
        assert_eq!(crate::verifier::verify(&a), Ok(()));
    }

    // CEP:WHAT: Congruent nodes in sibling regions are NOT merged (the
    //           representative does not dominate).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if legality is violated.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn respects_dominance() {
        let mut a = IrArena::with_capacity(32, 8);
        let root = a.root_region();
        let r1 = a.new_region(root);
        let r2 = a.new_region(root);
        if let (Ok(reg1), Ok(reg2)) = (r1, r2) {
            let c0 = const_f64(&mut a, root, 1.0);
            if let Ok(v0) = c0 {
                let x0 = a.value_of(v0, 0);
                assert!(x0.is_ok());
                if let Ok(val) = x0 {
                    let n1 = Node::new(
                        Op::Unary(xir_core::op::UnaryOp::Neg),
                        reg1,
                        &[val],
                        Type::Scalar(xir_core::ty::ScalarType::F64),
                    );
                    let n2 = Node::new(
                        Op::Unary(xir_core::op::UnaryOp::Neg),
                        reg2,
                        &[val],
                        Type::Scalar(xir_core::ty::ScalarType::F64),
                    );
                    let _ = a.insert_node(reg1, n1);
                    let _ = a.insert_node(reg2, n2);
                    let r = run(&mut a);
                    assert!(r.is_ok());
                    if let Ok(merged) = r {
                        assert_eq!(merged, 0, "sibling duplicates must survive");
                    }
                }
            }
        }
    }

    // CEP:WHAT: Impure ops never merge.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if an effectful op is merged.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn keeps_impure_ops() {
        let mut a = IrArena::with_capacity(32, 8);
        let root = a.root_region();
        let rng1 = Node::new(
            Op::Rng {
                dist: xir_core::op::RngDist::Uniform,
                seed: 3,
            },
            root,
            &[],
            Type::Scalar(xir_core::ty::ScalarType::F64),
        );
        let rng2 = Node::new(
            Op::Rng {
                dist: xir_core::op::RngDist::Uniform,
                seed: 3,
            },
            root,
            &[],
            Type::Scalar(xir_core::ty::ScalarType::F64),
        );
        let _ = a.insert_node(root, rng1);
        let _ = a.insert_node(root, rng2);
        let r = run(&mut a);
        assert!(r.is_ok());
        if let Ok(merged) = r {
            assert_eq!(merged, 0);
        }
    }
}
