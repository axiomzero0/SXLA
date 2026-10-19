// CEP:FILE: crates/supertensor/src/lib.rs
// CEP:WHAT: supertensor — the deterministic, constrained, tensor-native
//           superoptimizer: WPA extracts the Schedule Polytope (Gear-0),
//           workers enumerate the feasible region through typed SPSC
//           pipelines (Anvil v2 Upgrade 3) into a Pareto frontier, and the
//           final gate constructively verifies the Top-K winner before its
//           fusion decisions land as a ClusterSet for the existing pipeline.
// CEP:WHY: Dual-engine architecture: SuperTensor-WPA owns XIR Level-2 tensor
//          scheduling ONLY (scalar/control code is opaque — axiom 1); the
//          stochastic engine (crates/mss) owns the SoN scalar domain. Zero
//          core modification: this crate consumes/produces XIR artifacts via
//          public APIs (axiom 5); the jit driver integrates it as a Tier-2
//          accelerator with fallback to the existing fusion search.
// CEP:CLASS: CEP-29
// CEP:STATUS: partial (see CEP:TODOs; the pipeline end-to-end is complete —
//             SMT escalation and Level-3 tile consumption are follow-ups).
// CEP:FAILURE: SuperOptError variants; the driver falls back to the
//              existing Tier-2 path on ANY error (never a wrong result).
// CEP:ASSUMES: verified arena input; HardwareModel::reference (documented).
// CEP:COST: WPA O(nodes + edges) Gear-0; search budget-capped (65_536
//           candidates + 262_144 expansions per worker); gate O(top-k × ops).
// CEP:EVIDENCE: crate test suite (polytope soundness, WPA rules, enumerator
//           determinism + budgets, verifier gate, cluster bridge);
//           tests/constraint_soundness.rs; tests/concurrency_stress.rs;
//           jit driver differential test (values equal Tier-1).
// CEP:SECURITY: repository-trusted XIR only; no subprocesses, no unsafe.
// CEP:HPC-DETERMINISM: end-to-end deterministic result (see enumerator
//           module-header proof): the frontier content and the verified
//           winner are functions of the polytope, never of interleaving.
// CEP:TODO(main-agent): CEP-29: (a) SMT escalation via CVC5 subprocess
//                       (needs a timeout-capable zero-dep subprocess facility);
//                       (b) Level-3 tiling consumption of TileHint;
//                       (c) interprocedural liveness ranges (wpa.rs note).
//! # supertensor
//!
//! IPO/WPA tensor superoptimizer: polytope → concurrent search → gate.

pub mod cost;
pub mod enumerator;
pub mod pareto;
pub mod polytope;
pub mod verifier;
pub mod wpa;
pub mod xir_bridge;

use xir_core::arena::IrArena;

pub use cost::CostPoint;
pub use enumerator::{CandidateRecord, MAX_SCHED_OPS};
pub use polytope::{HardwareModel, PackedOpSched, SchedulePolytope};
pub use verifier::{SuperOptError, TopCandidate};
pub use xir_bridge::TileHint;

/// The outcome consumed by the jit driver.
///
/// CEP:WHAT: Verified winner's fusion decisions (ClusterSet), its cost,
///           advisory search counters, and the observable tile hints.
/// CEP:WHY: The driver needs clusters + a reportable result; nothing else
///          crosses the crate boundary (Law 3: one authority per artifact).
/// CEP:STATUS: complete
/// CEP:FAILURE: produced only on a fully verified winner.
/// CEP:ASSUMES: arena unchanged since extraction.
/// CEP:COST: owned cluster set + hints.
/// CEP:EVIDENCE: driver differential test.
/// CEP:SECURITY: no untrusted input.
/// CEP:NOTE: no Debug/Clone derives — ClusterSet is neither; callers report
///            through best_cost/hints (the observable summary).
pub struct SuperOutcome {
    /// The winner's fusion decisions (pipeline-native artifact).
    pub clusters: xir_levels::level2::ClusterSet,
    /// The winner's evaluated cost.
    pub best_cost: CostPoint,
    /// Advisory: candidates evaluated (nondeterministic under pruning).
    pub evaluated: u64,
    /// Advisory: candidates skipped by prune hints.
    pub pruned: u64,
    /// The winner's per-op placement decisions (diagnostic).
    pub tile_hints: Vec<TileHint>,
}

/// CEP:WHAT: Tensor-heaviness predicate (the accelerator's dispatch guard).
/// CEP:WHY: "Non-tensor programs bypass it entirely" — the guard is cheap,
///          deterministic, and lives with the engine (one authority).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (predicate).
/// CEP:ASSUMES: none.
/// CEP:COST: O(nodes).
/// CEP:EVIDENCE: driver dispatch test.
/// CEP:SECURITY: no untrusted input.
pub fn is_tensor_heavy(arena: &IrArena, min_ops: usize) -> bool {
    let mut count: usize = 0;
    arena.for_each_live_node(|_, node| {
        if wpa::is_scheduleable(node) {
            count += 1;
        }
    });
    count >= min_ops
}

/// CEP:WHAT: Truncates the polytope to the MAX_SCHED_OPS dominant ops
///           (largest flops; ties by slot order) re-sorted by slot order.
/// CEP:WHY: Bounded search arity (bounded resources); deterministic
///          selection keeps the scheduleable set stable across runs.
/// CEP:STATUS: complete
/// CEP:FAILURE: returns the original when within the cap.
/// CEP:ASSUMES: none.
/// CEP:COST: O(ops log ops).
/// CEP:EVIDENCE: entry-point test on a 14-op graph.
/// CEP:SECURITY: no untrusted input.
fn truncate_polytope(poly: SchedulePolytope) -> SchedulePolytope {
    use crate::polytope::{DivConstraint, LinearRow};
    let n = poly.op_count();
    if n <= MAX_SCHED_OPS {
        return poly;
    }
    // Rank by (flops DESC, slot ASC); take MAX_SCHED_OPS; re-sort by slot.
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| poly.op_flops[b].cmp(&poly.op_flops[a]).then(a.cmp(&b)));
    ranked.truncate(MAX_SCHED_OPS);
    let mut keep: Vec<usize> = ranked;
    keep.sort_unstable();
    let mut op_bounds = Vec::with_capacity(keep.len());
    let mut op_flops = Vec::with_capacity(keep.len());
    let mut op_out_bytes = Vec::with_capacity(keep.len());
    let mut op_in_bytes = Vec::with_capacity(keep.len());
    let mut op_min_hbm = Vec::with_capacity(keep.len());
    for &i in keep.iter() {
        op_bounds.push(poly.op_bounds[i].clone());
        op_flops.push(poly.op_flops[i]);
        op_out_bytes.push(poly.op_out_bytes[i]);
        op_in_bytes.push(poly.op_in_bytes[i]);
        op_min_hbm.push(poly.op_min_hbm[i]);
    }
    // Remap constraint indices through the kept set.
    let remap = |old: u16| -> Option<u16> {
        keep.iter()
            .position(|&k| k == old as usize)
            .map(|p| p as u16)
    };
    let rows: Vec<LinearRow> = poly
        .global_constraints
        .rows
        .iter()
        .filter_map(|row| {
            let mut entries = Vec::with_capacity(row.entries.len());
            let mut ok = true;
            for (op, param, coeff) in row.entries.iter() {
                match remap(*op) {
                    Some(new_op) => entries.push((new_op, *param, *coeff)),
                    None => {
                        ok = false;
                    }
                }
            }
            if ok {
                Some(LinearRow {
                    entries,
                    rhs: row.rhs,
                })
            } else {
                None
            }
        })
        .collect();
    let divisibility: Vec<DivConstraint> = poly
        .divisibility_constraints
        .iter()
        .filter_map(|d| {
            remap(d.op_index).map(|op| DivConstraint {
                op_index: op,
                dim: d.dim,
                divisor: d.divisor,
            })
        })
        .collect();
    SchedulePolytope {
        op_bounds,
        global_constraints: crate::polytope::SparseLinearSystem { rows },
        divisibility_constraints: divisibility,
        hw: poly.hw,
        op_flops,
        op_out_bytes,
        op_in_bytes,
        op_min_hbm,
    }
}

/// CEP:WHAT: The full pipeline over a parsed arena: extract -> truncate ->
///           search -> gate -> bridge.
/// CEP:WHY: The drop-in accelerator entry the jit driver calls at Tier 2
///           for tensor-heavy programs; every failure degrades to the
///           caller's fallback (the existing fusion search).
/// CEP:STATUS: complete
/// CEP:FAILURE: Err(SuperOptError) — NoTensorOps / NoVerifiedCandidate /
///              gate violations; NEVER a wrong result.
/// CEP:ASSUMES: verified arena (the driver verifies before Tier-2 passes).
/// CEP:COST: see module header.
/// CEP:EVIDENCE: entry-point test + jit driver differential test.
/// CEP:SECURITY: no untrusted input.
/// CEP:HPC-DETERMINISM: deterministic winner (enumerator proof).
pub fn superoptimize_arena(arena: &IrArena) -> Result<SuperOutcome, SuperOptError> {
    let hw = HardwareModel::reference();
    // Stage 1: WPA -> polytope (Gear-0).
    let poly = truncate_polytope(wpa::extract_polytope(arena, hw));
    if poly.op_count() == 0 {
        return Err(SuperOptError::NoTensorOps);
    }
    // Stage 2: concurrent enumeration (typed SPSC pipelines).
    let workers = anvil::default_worker_count().max(1);
    let outcome = enumerator::search(&poly, workers);
    if outcome.frontier.is_empty() {
        return Err(SuperOptError::NoVerifiedCandidate);
    }
    // Stage 3: the gate (Top-K, first verified wins).
    let top: Vec<TopCandidate> = outcome
        .frontier
        .top_k(10)
        .into_iter()
        .map(|p| TopCandidate {
            sched: p.sched,
            cost: p.cost,
        })
        .collect();
    let winner = verifier::verify_candidates(arena, &poly, &top)
        .ok_or(SuperOptError::NoVerifiedCandidate)?;
    // Bridge: winner -> ClusterSet (+ observable hints).
    let clusters = xir_bridge::build_clusters(arena, &poly, &winner.sched, hw)?;
    Ok(SuperOutcome {
        clusters,
        best_cost: winner.cost,
        evaluated: outcome.evaluated,
        pruned: outcome.pruned,
        tile_hints: xir_bridge::tile_hints(&winner.sched),
    })
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;
    use xir_core::text::parse_arena;

    const SRC: &str = concat!(
        "xir v1 func @main {\n",
        "  %0 = param 0 : tensor<f64>[16,16]\n",
        "  %1 = param 1 : tensor<f64>[16,16]\n",
        "  %2 = matmul ta=false tb=false %0, %1 : tensor<f64>[16,16]\n",
        "  %3 = param 2 : tensor<f64>[16,16]\n",
        "  %4 = binary.add %2, %3 : tensor<f64>[16,16]\n",
        "  %5 = unary.relu %4 : tensor<f64>[16,16]\n",
        "}\n"
    );

    // CEP:WHAT: End-to-end entry point: matmul+add+relu produces a verified
    //           winner with clusters and hints; two runs agree EXACTLY.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on pipeline failure or nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test (the end-to-end determinism witness)
    #[test]
    fn superoptimize_end_to_end_deterministic() {
        let arena = parse_arena(SRC, 64).expect("parse");
        assert!(is_tensor_heavy(&arena, 3));
        let a = superoptimize_arena(&arena).expect("outcome");
        let b = superoptimize_arena(&arena).expect("outcome");
        assert_eq!(a.best_cost, b.best_cost);
        assert_eq!(a.tile_hints, b.tile_hints);
        assert_eq!(a.clusters.len(), b.clusters.len());
        // The relu chain gives the search fusion headroom; the winner must
        // fuse at least one pair (cluster set non-empty) on this input.
        assert!(
            !a.clusters.is_empty(),
            "expected at least one fused cluster"
        );
    }

    // CEP:WHAT: Scalar-only input bypasses (NoTensorOps, loud).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a silent bypass.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn scalar_program_is_loudly_rejected() {
        let src = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : scalar<i64>\n",
            "  %1 = const.i64 7\n",
            "  %2 = binary.mul %0, %1 : scalar<i64>\n",
            "}\n"
        );
        let arena = parse_arena(src, 64).expect("parse");
        assert!(!is_tensor_heavy(&arena, 1));
        assert!(matches!(
            superoptimize_arena(&arena),
            Err(SuperOptError::NoTensorOps)
        ));
    }
}
