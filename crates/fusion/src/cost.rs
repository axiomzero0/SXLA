// CEP:FILE: crates/fusion/src/cost.rs
// CEP:WHAT: The tiered fusion cost model — Tier-1 analytic heuristics,
//           Tier-2 ML-guided ranking (placeholder), Tier-3 autotuning (stub).
// CEP:WHY: Master architecture section 5 mandates the tiered approach.
//          Tier 1 is complete and deterministic (analytic memory-traffic +
//          arithmetic model). Tier 2 is a loud PLACEHOLDER (no model
//          weights in this release — returning Tier-1 results with a
//          documented status). Tier 3 is a STUB that fails loudly when
//          invoked without a benchmark harness (CEP&CC 10.5: stubs must
//          fail loudly, never silently return numbers).
// CEP:CLASS: CEP-0 (Tier-1 scoring) / CEP-2 (Tier-2/3)
// CEP:STATUS: partial
// CEP:FAILURE: Tier3Unavailable when autotuning is requested without a
//             harness; Tier2 falls back to Tier 1 (documented, not silent:
//             the tier report records the fallback).
// CEP:ASSUMES: Tier-1 costs are abstract units calibrated in
//           docs/cost_model.md.
// CEP:COST: Tier-1 scoring O(cluster size + edges).
// CEP:EVIDENCE: tests `fusion_reduces_edge_cost`, `tier3_is_loud`,
//           `tier2_reports_fallback`.
// CEP:SECURITY: no untrusted input.
// CEP:HPC-DETERMINISM: Tier 1 deterministic; Tier 2 fallback deterministic.
// CEP:TODO(main-agent): CEP-20: trained Tier-2 ranker; CEP-21: autotune
//           harness with bounded micro-benchmarks.
//! Tiered fusion cost model.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::MAX_INPUTS;
use xir_core::ty::Type;

/// Cost-model tier selector.
///
/// CEP:WHAT: The three architecture tiers.
/// CEP:WHY: The JIT picks tiers per compilation budget (Tier-1 fast path,
///          Tier-2 background); the selection is explicit, never implicit.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Fast analytical heuristics (complete).
    Tier1,
    /// ML-guided ranking (placeholder; falls back with report).
    Tier2,
    /// Autotuning micro-benchmarks (stub; loud failure).
    Tier3,
}

/// Cost-model failure enumeration.
///
/// CEP:WHAT: Explicit error type for tier requests.
/// CEP:WHY: Law 6 + CEP&CC 10.5 — stubs fail loudly.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: test `tier3_is_loud`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostError {
    /// Tier 3 was requested but no autotune harness is linked.
    Tier3Unavailable,
    /// Tier 2 has no model weights in this release; the caller must read
    /// the tier report instead of treating the score as ML-derived.
    Tier2Fallback,
}

/// The cost model handle.
///
/// CEP:WHAT: Tier selection + scoring entry points.
/// CEP:WHY: One object owns the tier policy so the JIT and the search share
///          identical calibration (no per-site constants — Law 7).
/// CEP:STATUS: partial
/// CEP:FAILURE: see CostError.
/// CEP:ASSUMES: none
/// CEP:COST: see per-method fields.
/// CEP:EVIDENCE: tests in this module.
pub struct CostModel {
    /// The selected tier.
    pub tier: Tier,
}

/// A cluster scoring input.
///
/// CEP:WHAT: Cluster members + the inter-cluster edge byte counts.
/// CEP:WHY: Fusion removes intra-cluster edges from memory traffic: the
///          score prices remaining (inter-cluster) edge bytes plus
///          arithmetic (arch section 5 cost model).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: edge bytes measured on the ORIGINAL (unfused) graph.
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
pub struct ClusterScoreInput<'a> {
    /// The arena the members live in.
    pub arena: &'a IrArena,
    /// Cluster member nodes.
    pub members: &'a [NodeId],
    /// Edge byte counts that remain OUTSIDE the cluster (inter-cluster).
    pub external_edge_bytes: u64,
}

/// A scored cluster plan.
///
/// CEP:WHAT: Total cost + tier provenance.
/// CEP:WHY: The search compares plans by (cost, priority); the report field
///          records whether Tier 2 actually fired or fell back (no silent
///          degradation — Law 1).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreReport {
    /// Total analytic cost (abstract units).
    pub cost: u64,
    /// The tier that actually produced the score.
    pub effective_tier: Tier,
}

impl CostModel {
    /// CEP:WHAT: Creates a cost model for a tier.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn new(tier: Tier) -> CostModel {
        CostModel { tier }
    }

    /// CEP:WHAT: Scores a cluster plan.
    /// CEP:WHY: Tier 1: cost = arithmetic(units per op) + external edge
    ///          bytes (1 unit/byte) + spill penalty when register pressure
    ///          exceeds budget (8 units per excess unit). Fusion lowers
    ///          cost by internalizing edges — the architecture's memory
    ///          traffic model.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Tier3Unavailable when tier == Tier3 (loud stub);
    ///              Tier2 reports the fallback in the report (not an Err —
    ///              the score is still usable, and the provenance is
    ///              explicit).
    /// CEP:ASSUMES: members live in the arena.
    /// CEP:COST: O(members).
    /// CEP:EVIDENCE: tests in this module.
    /// CEP:HPC-DETERMINISM: deterministic.
    pub fn score(&self, input: &ClusterScoreInput<'_>) -> Result<ScoreReport, CostError> {
        if self.tier == Tier::Tier3 {
            // CEP:STATUS: stub — loud failure per CEP&CC 10.5.
            return Err(CostError::Tier3Unavailable);
        }
        let mut cost: u64 = input.external_edge_bytes;
        let mut register_units: u32 = 0;
        for m in input.members {
            let node = match input.arena.node(*m) {
                Ok(n) => n,
                Err(_) => continue,
            };
            cost += u64::from(op_arithmetic(node.op));
            register_units = register_units.saturating_add(8);
        }
        // Spill penalty.
        let over = register_units.saturating_sub(crate::resource::REGISTER_BUDGET_UNITS);
        cost += u64::from(over) * 8;
        let effective_tier = if self.tier == Tier::Tier2 {
            // CEP:STATUS: placeholder — documented fallback to Tier 1.
            Tier::Tier1
        } else {
            Tier::Tier1
        };
        Ok(ScoreReport {
            cost,
            effective_tier,
        })
    }
}

/// CEP:WHAT: Arithmetic cost table (abstract units per op).
/// CEP:WHY: Tier-1 heuristic calibration point (docs/cost_model.md);
///          memory-bound ops are cheap per element, compute-bound ops
///          carry FLOP-ish weights.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: calibration documented.
/// CEP:COST: branch
/// CEP:EVIDENCE: tests in this module
pub(crate) fn op_arithmetic(op: xir_core::op::Op) -> u32 {
    use xir_core::op::Op as O;
    match op {
        O::ConstI64(_) | O::ConstF64(_) | O::Param { .. } => 1,
        O::Binary(_) | O::Unary(_) => 2,
        O::Dot => 64,
        O::Reduce { .. } => 8,
        O::Matmul { .. } => 128,
        O::Conv { .. } => 256,
        O::Transpose { .. } | O::Broadcast { .. } => 16,
        O::Rng { .. } => 8,
        O::If => 16,
        O::Custom { .. } => 64,
        O::FusionCluster | O::FusionBarrier | O::FusionMaterialize => 1,
        O::LoopParallel { .. } | O::LoopAlloc { .. } | O::LoopAsyncCopy => 1,
        O::LoopPipelineStage { .. } => 1,
        O::TargetMma => 96,
        O::TargetWarpShuffle | O::TargetBarrier => 4,
    }
}

/// CEP:WHAT: External edge bytes for a candidate clustering.
/// CEP:WHY: Counts the value edges that CROSS cluster boundaries — the
///          traffic fusion eliminates (intra-cluster edges become
///          registers).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: cluster_of returns the plan's assignment.
/// CEP:COST: O(nodes).
/// CEP:EVIDENCE: test `fusion_reduces_edge_cost`.
pub fn external_edge_bytes<F>(arena: &IrArena, cluster_of: F) -> u64
where
    F: Fn(NodeId) -> Option<u32>,
{
    let mut total: u64 = 0;
    arena.for_each_live_node(|id, node| {
        let consumer_cluster = cluster_of(id);
        for i in 0..node.n_inputs as usize {
            if i >= MAX_INPUTS {
                break;
            }
            let producer = node.inputs[i].node();
            let producer_cluster = cluster_of(producer);
            let crosses = match (producer_cluster, consumer_cluster) {
                (Some(p), Some(c)) => p != c,
                _ => true,
            };
            if crosses {
                // Bytes of the edge = producer's result size.
                if let Ok(pn) = arena.node(producer) {
                    let b = pn.ty.byte_size();
                    if b > 0 {
                        total += b as u64;
                    }
                }
            }
        }
    });
    total
}

// Type import kept for the byte_size helper's users.
#[allow(unused_imports)]
use Type as _TypeDoc;

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::IrArena;
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{Layout, ScalarType, Shape, TensorType, Type};

    fn tensor(shape: &[i64]) -> Type {
        Type::Tensor(TensorType {
            elem: ScalarType::F64,
            shape: Shape::from_dims(shape).ok().unwrap_or(Shape::scalar()),
            layout: Layout::RowMajor,
        })
    }

    fn chain_arena() -> (IrArena, NodeId, NodeId, NodeId) {
        // param -> neg -> relu (a 3-node chain).
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let p = Node::new(Op::Param { index: 0 }, root, &[], tensor(&[64, 64]));
        let pid = a.insert_node(root, p);
        let mut ids = (NodeId::NONE, NodeId::NONE, NodeId::NONE);
        if let Ok(pv) = pid {
            ids.0 = pv;
            let v = a.value_of(pv, 0);
            if let Ok(val) = v {
                let neg = Node::new(
                    Op::Unary(xir_core::op::UnaryOp::Neg),
                    root,
                    &[val],
                    tensor(&[64, 64]),
                );
                if let Ok(nv) = a.insert_node(root, neg) {
                    ids.1 = nv;
                    if let Ok(nval) = a.value_of(nv, 0) {
                        let relu = Node::new(
                            Op::Unary(xir_core::op::UnaryOp::Relu),
                            root,
                            &[nval],
                            tensor(&[64, 64]),
                        );
                        if let Ok(rv) = a.insert_node(root, relu) {
                            ids.2 = rv;
                        }
                    }
                }
            }
        }
        (a, ids.0, ids.1, ids.2)
    }

    // CEP:WHAT: Fusing a chain removes its edges from external traffic.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if fusion does not reduce cost.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fusion_reduces_edge_cost() {
        let (a, p, n, r) = chain_arena();
        let chain_ok = !r.is_none();
        assert!(chain_ok, "chain construction failed");
        if r.is_none() {
            return;
        }
        // Unfused: every edge is external.
        let unfused = external_edge_bytes(&a, |_id| None);
        // Fused: one cluster for neg+relu.
        let fused = external_edge_bytes(&a, |id| {
            if id == n || id == r {
                Some(0)
            } else if id == p {
                Some(1)
            } else {
                None
            }
        });
        assert!(fused < unfused);
        // Scoring prefers the fused plan.
        let cm = CostModel::new(Tier::Tier1);
        let s_unfused = cm.score(&ClusterScoreInput {
            arena: &a,
            members: &[n],
            external_edge_bytes: unfused,
        });
        let s_fused = cm.score(&ClusterScoreInput {
            arena: &a,
            members: &[n, r],
            external_edge_bytes: fused,
        });
        if let (Ok(u), Ok(f)) = (s_unfused, s_fused) {
            assert!(f.cost < u.cost);
            assert_eq!(f.effective_tier, Tier::Tier1);
        }
    }

    // CEP:WHAT: Tier 3 is a loud stub.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if Tier 3 returns a score.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tier3_is_loud() {
        let (a, _p, n, _r) = chain_arena();
        let cm = CostModel::new(Tier::Tier3);
        let r = cm.score(&ClusterScoreInput {
            arena: &a,
            members: &[n],
            external_edge_bytes: 0,
        });
        assert_eq!(r, Err(CostError::Tier3Unavailable));
    }

    // CEP:WHAT: Tier 2 reports its Tier-1 fallback explicitly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on silent degradation.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tier2_reports_fallback() {
        let (a, _p, n, _r) = chain_arena();
        let cm = CostModel::new(Tier::Tier2);
        let r = cm.score(&ClusterScoreInput {
            arena: &a,
            members: &[n],
            external_edge_bytes: 0,
        });
        assert!(r.is_ok());
        if let Ok(rep) = r {
            assert_eq!(rep.effective_tier, Tier::Tier1);
        }
        let _ = Op::Binary(BinaryOp::Add);
    }
}
