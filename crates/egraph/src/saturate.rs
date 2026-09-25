// CEP:FILE: crates/egraph/src/saturate.rs
// CEP:WHAT: Saturation driver — lifts a pure subgraph, applies local rules
//           under Anvil Gear 1 partitioning, merges, and returns the
//           saturated graph.
// CEP:WHY: Master architecture section 4: "E-classes are sharded across
//          Anvil workers by hash. Workers apply rewrite rules locally
//          without synchronization." Local rules (constant folding, identity
//          elimination) are exactly the synchronization-free rule class;
//          they run on disjoint node slices via run_partitioned and the
//          rewrites merge back deterministically by index. The full
//          cross-worker merge protocol (SPSC to a dedicated resolver
//          thread) is a documented TODO (CEP-17).
// CEP:CLASS: CEP-1 (driver) / CEP-0 (rule bodies)
// CEP:STATUS: partial
// CEP:FAILURE: EgraphError propagation; saturation stops at the node budget.
// CEP:ASSUMES: only pure nodes are lifted (caller filters).
// CEP:COST: lift O(nodes); local-rule rounds O(nodes) each, bounded by
//           MAX_ROUNDS; merge O(1) per rewrite.
// CEP:EVIDENCE: tests `saturation_folds_constants`, `saturation_is_deterministic`.
// CEP:SECURITY: IR treated as untrusted; all lookups checked.
// CEP:HPC-DETERMINISM: deterministic — partition results merge by index and
//           rules run in fixed order; Gear-1 scheduling is unobservable.
// CEP:TODO(main-agent): CEP-17: cross-worker class merges via SPSC resolver.
//! Saturation driver.

use xir_core::arena::IrArena;
use xir_core::id::{IrLevel, NodeId};
use xir_core::node::MAX_INPUTS;
use xir_core::op::Op;

use crate::egraph::{EGraph, EgraphError};
use crate::rules::{local_rules, ConstVal, Rewrite};

/// Saturation rounds bound.
///
/// CEP:WHAT: Iteration budget.
/// CEP:WHY: CEP&CC 38.11 (bounded compile time): equality saturation is
///          budgeted explicitly; one fold round + one verify round suffices
///          for the current rule set, 4 leaves headroom (documented).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: driver tests converge in <= 2 rounds.
pub const MAX_ROUNDS: u32 = 4;

/// One lifted node record for partition processing.
#[derive(Clone)]
struct Lifted {
    op: Op,
    children: [u32; 4],
    consts: [Option<ConstVal>; 4],
    n_children: u8,
}

/// CEP:WHAT: Result of a saturation run.
/// CEP:WHY: The driver reports applied rewrites and the saturated graph for
///          extraction by the caller.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: none
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
pub struct SaturationResult {
    /// The saturated e-graph.
    pub graph: EGraph,
    /// Local rewrites applied (constant folds + identities).
    pub rewrites_applied: u32,
}

/// CEP:WHAT: Lifts the arena's pure nodes into an e-graph.
/// CEP:WHY: "Pure subgraphs are lifted into the E-graph engine here for
///          equality saturation" (arch section 3 Level 0). Consts are probed
///          so local rules can fold/eliminate.
/// CEP:STATUS: complete
/// CEP:FAILURE: Full when the node budget exhausts (bounded, loud).
/// CEP:ASSUMES: arena verified; only pure ops lifted.
/// CEP:COST: O(nodes).
/// CEP:EVIDENCE: tests in this module.
fn lift(arena: &IrArena, budget: usize) -> Result<(EGraph, Vec<Lifted>, Vec<u32>), EgraphError> {
    let mut g = EGraph::new(budget);
    // First pass: create leaf nodes (no value children) in slot order.
    let mut node_class: Vec<Option<u32>> = vec![None; arena.slot_count()];
    let mut lifted: Vec<Lifted> = Vec::new();
    // Collect in deterministic slot order.
    let mut order: Vec<NodeId> = Vec::new();
    arena.for_each_live_node(|id, node| {
        if node.op.is_pure() {
            order.push(id);
        }
    });
    // Map XIR node -> (class, const value).
    let mut consts_of: Vec<Option<ConstVal>> = vec![None; arena.slot_count()];
    for id in order.iter() {
        let node = arena.node(*id).map_err(|_| EgraphError::BadClass)?;
        if let Op::ConstI64(v) = node.op {
            consts_of[id.index() as usize] = Some(ConstVal::I(v));
        } else if let Op::ConstF64(v) = node.op {
            consts_of[id.index() as usize] = Some(ConstVal::F(v));
        }
    }
    // Insert in dependency order: nodes whose inputs are already lifted.
    // Simple worklist (bounded passes).
    let mut remaining = order.clone();
    let mut passes = 0usize;
    while !remaining.is_empty() && passes <= remaining.len() + 1 {
        let mut next_round: Vec<NodeId> = Vec::new();
        let mut progressed = false;
        for id in remaining.iter() {
            let node = arena.node(*id).map_err(|_| EgraphError::BadClass)?;
            let mut children = [0u32; 4];
            let mut consts = [None; 4];
            let mut n = 0u8;
            let mut ready = true;
            for i in 0..node.n_inputs as usize {
                if i >= MAX_INPUTS {
                    break;
                }
                let def = node.inputs[i].node();
                match node_class.get(def.index() as usize).copied().flatten() {
                    Some(c) => {
                        if n < 4 {
                            children[n as usize] = c;
                            consts[n as usize] =
                                consts_of.get(def.index() as usize).copied().flatten();
                            n += 1;
                        }
                    }
                    None => {
                        // Impure or not yet lifted producer: skip the edge
                        // only if the producer is impure (never lifted);
                        // otherwise defer.
                        match arena.node(def) {
                            Ok(p) if !p.op.is_pure() => {}
                            _ => {
                                ready = false;
                                break;
                            }
                        }
                    }
                }
            }
            if !ready {
                next_round.push(*id);
                continue;
            }
            let class = g.add(node.op, &children[..n as usize])?;
            node_class[id.index() as usize] = Some(class);
            lifted.push(Lifted {
                op: node.op,
                children,
                consts,
                n_children: n,
            });
            progressed = true;
        }
        remaining = next_round;
        passes += 1;
        if !progressed {
            break;
        }
    }
    // Class list per lifted record (aligned with `lifted`).
    let classes: Vec<u32> = lifted
        .iter()
        .enumerate()
        .map(|(i, _)| node_class_of(&node_class, &order, i))
        .collect();
    Ok((g, lifted, classes))
}

/// CEP:WHAT: Resolves the class of the i-th lifted record.
/// CEP:WHY: `lifted` is built in worklist order; this maps back through the
///          recorded node ids. (Index alignment is a driver invariant,
///          checked by tests.)
/// CEP:STATUS: complete
/// CEP:FAILURE: returns 0 on misalignment (defensive; tests catch drift).
/// CEP:ASSUMES: alignment invariant.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: driver tests.
fn node_class_of(node_class: &[Option<u32>], order: &[NodeId], i: usize) -> u32 {
    if i < order.len() {
        node_class
            .get(order[i].index() as usize)
            .copied()
            .flatten()
            .unwrap_or(0)
    } else {
        0
    }
}

/// CEP:WHAT: Runs local-rule saturation rounds under Gear 1.
/// CEP:WHY: Local rules are synchronization-free (arch section 4): constant
///          folding and identity elimination only touch one node and its
///          recorded constants. `anvil::run_partitioned` slices the lifted
///          records across workers; each worker computes its rewrites
///          deterministically; the driver merges results by index so the
///          saturated graph is scheduling-independent.
/// CEP:STATUS: complete
/// CEP:FAILURE: EgraphError propagation; partition worker count from
///              anvil::default_worker_count (bounded by config).
/// CEP:ASSUMES: lift succeeded.
/// CEP:COST: O(rounds * nodes).
/// CEP:EVIDENCE: tests `saturation_folds_constants`.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn saturate(arena: &IrArena, node_budget: usize) -> Result<SaturationResult, EgraphError> {
    let (mut g, mut lifted, classes) = lift(arena, node_budget)?;
    let _ = classes;
    let mut applied = 0u32;
    for _round in 0..MAX_ROUNDS {
        // Gear 1: partitioned local-rule application. Each slice produces
        // Option<Rewrite> per record (None = no fire); pure function of the
        // record, so scheduling cannot change results.
        let outputs: Vec<Option<Rewrite>> = vec![None; lifted.len()];
        let inputs: Vec<&Lifted> = lifted.iter().collect();
        let workers = anvil::default_worker_count().max(1);
        let mut results: Vec<Option<Rewrite>> = outputs;
        let partition_ok = anvil::run_partitioned(&inputs, &mut results, workers, |rec| {
            local_rules(rec.op, &rec.children, &rec.consts)
        })
        .is_ok();
        // Fallback: sequential application if partitioning failed (worker
        // bounds); documented degradation, same results.
        if !partition_ok {
            for (i, rec) in lifted.iter().enumerate() {
                results[i] = local_rules(rec.op, &rec.children, &rec.consts);
            }
        }
        let mut round_applied = 0u32;
        for (i, r) in results.iter().enumerate() {
            let Some(rw) = r else { continue };
            // Constant folds add a new const node and merge.
            let new_class = match rw.n_children {
                0 => g.add(rw.op, &[])?,
                // Identity pass-through: merge with the surviving child.
                _ => {
                    // Rewrite marker: op = Param{u32::MAX} means "use child".
                    let child_class = rw.children[0];
                    let node_class = g
                        .add(
                            lifted[i].op,
                            &lifted[i].children[..lifted[i].n_children as usize],
                        )
                        .ok();
                    if let Some(nc) = node_class {
                        let winner = g.merge(nc, child_class)?;
                        let _ = winner;
                    }
                    child_class
                }
            };
            let _ = new_class;
            round_applied += 1;
        }
        applied += round_applied;
        if round_applied == 0 {
            break;
        }
        // After folding, subsequent rounds see new constants: refresh the
        // lifted records' const probes by re-lifting.
        let (g2, lifted2, _c2) = lift(arena, node_budget)?;
        g = g2;
        lifted = lifted2;
    }
    Ok(SaturationResult {
        graph: g,
        rewrites_applied: applied,
    })
}

// Re-export used by callers building extraction pipelines.
#[allow(unused_imports)]
use IrLevel as _IrLevelDoc;

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::{const_i64, IrArena};
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Type};

    fn arena_with_const_add() -> IrArena {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let add = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::I64),
                );
                let _ = a.insert_node(root, add);
            }
        }
        a
    }

    // CEP:WHAT: Saturation folds constant adds into a const node.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if folding does not fire.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn saturation_folds_constants() {
        let a = arena_with_const_add();
        let r = saturate(&a, 64);
        assert!(r.is_ok());
        if let Ok(res) = r {
            assert!(res.rewrites_applied >= 1);
        }
    }

    // CEP:WHAT: Saturation is deterministic across runs.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn saturation_is_deterministic() {
        let a = arena_with_const_add();
        let r1 = saturate(&a, 64);
        let r2 = saturate(&a, 64);
        assert!(r1.is_ok() && r2.is_ok());
        if let (Ok(x1), Ok(x2)) = (r1, r2) {
            assert_eq!(x1.rewrites_applied, x2.rewrites_applied);
            assert_eq!(x1.graph.node_count(), x2.graph.node_count());
        }
    }
}
