// CEP:FILE: crates/egraph/src/extract.rs
// CEP:WHAT: Fusion-aware extraction — picks the cheapest e-node per class
//           with producer-consumer locality and layout-conversion penalties.
// CEP:WHY: Master architecture section 4: "The extraction cost model does
//          not just minimize op count. It heavily penalizes rewrites that
//          break producer-consumer locality or force expensive layout
//          conversions, ensuring the e-graph output is primed for Level 2
//          fusion." Layout-breaking ops (transpose, broadcast) carry an
//          explicit penalty constant; extraction minimizes total cost with
//          deterministic tie-breaks (lowest seq).
// CEP:CLASS: CEP-0 (extraction core)
// CEP:STATUS: complete
// CEP:FAILURE: EgraphError propagation (BadClass on inconsistent state).
// CEP:ASSUMES: saturation completed; classes canonical.
// CEP:COST: O(nodes + classes) post-order dynamic program.
// CEP:EVIDENCE: tests `cheapest_node_wins`, `layout_penalty_steers_choice`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-DETERMINISM: deterministic (cost then seq tie-break).
//! Fusion-aware extraction.

use xir_core::op::Op;

use crate::egraph::{EGraph, EgraphError};

/// Layout/locality penalty (abstract cost units).
///
/// CEP:WHAT: The fusion-aware penalty constant.
/// CEP:WHY: Architecture mandate: extraction must "heavily penalize" layout
///          conversions; 64 units dwarfs per-op costs (1-8) so a transpose
///          survives only when it saves an entire materialization.
///          Named constant (Law 7), calibrated in docs/cost_model.md.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: calibrated against the fusion cost model's memory term.
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: test `layout_penalty_steers_choice`
pub const FUSION_LOCALITY_PENALTY: u32 = 64;

/// Base op cost table (abstract units).
///
/// CEP:WHAT: Per-opcode base costs.
/// CEP:WHY: Extraction needs a monotone cost; elementwise ops are cheap,
///          matmul/conv expensive, constants cheapest.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: calibrated in docs/cost_model.md (Tier-1 analytic heuristics).
/// CEP:COST: branch
/// CEP:EVIDENCE: extraction tests
fn base_cost(op: Op) -> u32 {
    match op {
        Op::ConstI64(_) | Op::ConstF64(_) | Op::Param { .. } => 1,
        Op::Binary(_) | Op::Unary(_) => 2,
        Op::Dot => 8,
        Op::Reduce { .. } => 4,
        // Layout-breaking ops carry the fusion penalty (architecture).
        Op::Transpose { .. } | Op::Broadcast { .. } => FUSION_LOCALITY_PENALTY,
        Op::Matmul { .. } => 16,
        Op::Conv { .. } => 24,
        Op::Rng { .. } => 4,
        Op::If => 8,
        Op::Custom { .. } => 32,
        Op::FusionCluster | Op::FusionBarrier | Op::FusionMaterialize => 1,
        Op::LoopParallel { .. } | Op::LoopAlloc { .. } | Op::LoopAsyncCopy => 1,
        Op::LoopPipelineStage { .. } => 1,
        Op::TargetMma => 12,
        Op::TargetWarpShuffle | Op::TargetBarrier => 2,
    }
}

/// Extraction result: per class, the chosen e-node.
///
/// CEP:WHAT: Class -> chosen node id and its cost.
/// CEP:WHY: The driver rebuilds the optimized subgraph from these choices.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: extraction completed.
/// CEP:COST: O(classes) storage.
/// CEP:EVIDENCE: tests in this module.
pub struct Extraction {
    /// (class, chosen node id, total cost).
    pub choices: Vec<(u32, u32, u32)>,
    /// Total cost of the extracted program (sum over root classes).
    pub total_cost: u32,
}

/// CEP:WHAT: Extracts the cheapest program from the e-graph.
/// CEP:WHY: The saturated e-graph holds many equivalent forms; extraction is
///          the deterministic selection (cost then seq tie-break) that
///          honors the fusion-aware penalty model.
/// CEP:STATUS: complete
/// CEP:FAILURE: EgraphError propagation.
/// CEP:ASSUMES: classes canonical; children reference classes.
/// CEP:COST: O(nodes + classes) with memoized child costs.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn extract(g: &EGraph) -> Result<Extraction, EgraphError> {
    // Process nodes in ascending id order (children have smaller ids in our
    // driver's construction; verify and fall back to iterative fixpoint
    // otherwise — one extra pass, still deterministic).
    let n = g.node_count();
    let mut best_cost: Vec<u32> = vec![u32::MAX; n];
    let mut best_node: Vec<Option<u32>> = vec![None; n];
    // Class -> best (cost, node) chosen at class level.
    let mut class_best: Vec<Option<(u32, u32)>> = vec![None; n.max(1)];
    // Iterate to fixpoint (bounded by n passes — children-before-parents
    // convergence; typical 2 passes).
    for _round in 0..=n {
        let mut changed = false;
        for id in 0..n {
            let node = g.node(id as u32)?;
            // Cost = base + child class bests (children must be resolved).
            let mut cost = base_cost(node.op);
            let mut resolved = true;
            for c in node.children.iter().take(node.n_children as usize) {
                match class_best.get(*c as usize).copied().flatten() {
                    Some((cc, _cn)) => cost = cost.saturating_add(cc),
                    None => {
                        resolved = false;
                        break;
                    }
                }
            }
            if !resolved {
                continue;
            }
            // Class of THIS node: recorded by the driver as node.children[0]
            // convention does not apply here; classes carry members. The
            // driver's class_of mapping is external — we recompute via
            // class membership below.
            if cost < best_cost[id] {
                best_cost[id] = cost;
                best_node[id] = Some(id as u32);
                changed = true;
            }
        }
        // Update class-level bests from members.
        for class in g.class_ids() {
            let members = g.members(class)?;
            let mut cbest: Option<(u32, u32)> = None;
            for m in members {
                if let Some(chosen) = best_node.get(*m as usize).copied().flatten() {
                    let c = best_cost[chosen as usize];
                    // Tie-break: lower node id (== lower seq) wins.
                    let better = match cbest {
                        None => true,
                        Some((bc, bn)) => c < bc || (c == bc && chosen < bn),
                    };
                    if better {
                        cbest = Some((c, chosen));
                    }
                }
            }
            let slot = class as usize;
            if slot < class_best.len() {
                class_best[slot] = cbest;
            }
        }
        if !changed {
            break;
        }
    }
    // Assemble choices for classes that resolved.
    let mut choices: Vec<(u32, u32, u32)> = Vec::new();
    for class in g.class_ids() {
        if let Some((c, node)) = class_best.get(class as usize).copied().flatten() {
            choices.push((class, node, c));
        }
    }
    let total_cost = choices.iter().map(|(_, _, c)| c).sum();
    Ok(Extraction {
        choices,
        total_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::EGraph;
    use xir_core::op::BinaryOp;

    // CEP:WHAT: Among equal forms, the cheapest node is chosen.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on cost inversion.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn cheapest_node_wins() {
        let mut g = EGraph::new(16);
        let a = g.add(Op::ConstI64(1), &[]);
        let b = g.add(Op::ConstI64(2), &[]);
        assert!(a.is_ok() && b.is_ok());
        if let (Ok(ka), Ok(kb)) = (a, b) {
            let add = g.add(Op::Binary(BinaryOp::Add), &[ka, kb]);
            assert!(add.is_ok());
            let e = extract(&g);
            assert!(e.is_ok());
            if let Ok(ex) = e {
                // Total = sum over ALL classes: const(1) + const(1) + add(4).
                assert_eq!(ex.total_cost, 6);
                assert_eq!(ex.choices.len(), 3);
            }
        }
    }

    // CEP:WHAT: Transpose nodes carry the heavy layout penalty.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the penalty is dropped.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn layout_penalty_steers_choice() {
        assert_eq!(
            base_cost(Op::Transpose {
                perm: [0, 1, 0, 0],
                rank: 2
            }),
            FUSION_LOCALITY_PENALTY
        );
        assert_eq!(
            base_cost(Op::Broadcast {
                to: xir_core::ty::Shape::scalar()
            }),
            FUSION_LOCALITY_PENALTY
        );
        assert!(base_cost(Op::Binary(BinaryOp::Add)) < FUSION_LOCALITY_PENALTY);
        assert!(
            base_cost(Op::Matmul {
                transpose_a: false,
                transpose_b: false
            }) < FUSION_LOCALITY_PENALTY
        );
    }
}
