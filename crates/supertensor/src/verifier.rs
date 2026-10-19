// CEP:FILE: crates/supertensor/src/verifier.rs
// CEP:WHAT: Stage 3 — the final verification gate: constructive feasibility
//           proof of each Top-K candidate (every polytope constraint re-checked
//           EXACTLY, every fusion pair re-checked against fusion::legality,
//           the op multiset pinned to the polytope's op set), with the
//           SMT-subprocess escalation documented as unavailable-by-default.
// CEP:WHY: Axiom 3 — SMT at the gate ONLY, never in the hot loop. For the
//           tensor-schedule domain the equivalence obligation reduces to
//           feasibility + legality: an OpSchedule changes PLACEMENT (tiles,
//           memory level, fusion), never the computation's denotation, so a
//           candidate that satisfies every WPA constraint and every fusion
//           legality rule is provably semantics-preserving BY CONSTRUCTION.
//           That constructive proof is what this gate implements and what the
//           SMT escalation would double-check externally.
// CEP:CLASS: CEP-29
// CEP:STATUS: complete (constructive gate); SMT escalation is a loud
//             documented placeholder (see CEP:TODO below).
// CEP:FAILURE: Err(SuperOptError) naming the violated constraint; candidates
//              that fail are SKIPPED, not fatal — the gate returns the first
//              verified candidate (or None -> caller falls back to Tier-2).
// CEP:ASSUMES: Top-K from the deterministic frontier; arena is the one the
//              polytope was extracted from.
// CEP:COST: O(candidates × (ops + rows)); microseconds — the "gate only"
//           budget contract.
// CEP:EVIDENCE: tests `verifier_accepts_sound_candidate`,
//           `verifier_rejects_illegal_fuse`, `top_k_gate_skips_violators`.
// CEP:SECURITY: no untrusted input.
// CEP:TODO(main-agent): CEP-29: SMT escalation via CVC5 subprocess — blocked
//                       on a timeout-capable, zero-dependency subprocess
//                       facility (std::process has none); the escalation is
//                       env-gated when it lands and NEVER in the default path.
//! Stage 3: the verification gate.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;

use crate::cost::CostPoint;
use crate::polytope::{ConstraintViolation, PackedOpSched, SchedulePolytope};

/// Failure enumeration for the supertensor pipeline.
///
/// CEP:WHAT: Explicit error type (Law 6).
/// CEP:WHY: Callers (the JIT driver) fall back to the existing Tier-2 path —
///          failures must be diagnosable, not silent.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: small enum
/// CEP:EVIDENCE: driver fallback tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuperOptError {
    /// No tensor ops in the input (nothing to schedule).
    NoTensorOps,
    /// The search produced no verifiable candidate.
    NoVerifiedCandidate,
    /// A candidate violated the polytope (name = violation).
    ConstraintViolation(ConstraintViolation),
}

/// CEP:WHAT: Verifies ONE candidate constructively (the gate's core check).
/// CEP:WHY: The three obligations: (1) polytope feasibility — re-check every
///          bound, divisibility, linear row, SRAM pressure; (2) fusion
///          legality — every fuse pair through can_fuse, THE authority;
///          (3) domain integrity — the schedule covers exactly the
///          polytope's ops (placement-only transformation, axiom 1).
/// CEP:STATUS: complete
/// CEP:FAILURE: Err(SuperOptError::ConstraintViolation) with the exact class.
/// CEP:ASSUMES: schedule aligned with polytope op order.
/// CEP:COST: O(ops + rows + fuse pairs).
/// CEP:EVIDENCE: `verifier_accepts_sound_candidate`.
/// CEP:SECURITY: no untrusted input.
pub fn verify_candidate(
    arena: &IrArena,
    poly: &SchedulePolytope,
    sched: &[PackedOpSched],
) -> Result<(), SuperOptError> {
    // (1) Full polytope feasibility (bounds, divisibility, linear, SRAM).
    poly.check_full(sched)
        .map_err(SuperOptError::ConstraintViolation)?;
    // (2) Fusion legality re-check: every fuse pair through the authority.
    for (i, s) in sched.iter().enumerate() {
        if s.fuse_target != PackedOpSched::NO_FUSE {
            let target = NodeId(s.fuse_target);
            let producer = NodeId(s.op_id);
            let legal = fusion::legality::can_fuse(arena, producer, target).unwrap_or(false);
            if !legal {
                return Err(SuperOptError::ConstraintViolation(
                    ConstraintViolation::FuseIllegal(i as u16),
                ));
            }
        }
    }
    // (3) Domain integrity: op ids are exactly the polytope's, in order
    //     (check_full already pins this — repeated loudly here because the
    //     verifier is the trust boundary, not an internal fast path).
    if sched.len() != poly.op_count() {
        return Err(SuperOptError::ConstraintViolation(
            ConstraintViolation::UnknownOp(sched.len() as u16),
        ));
    }
    Ok(())
}

/// The gate's view of one Top-K entry.
#[derive(Debug, Clone)]
pub struct TopCandidate {
    /// The schedule under test.
    pub sched: Vec<PackedOpSched>,
    /// Its evaluated cost (reported, not trusted).
    pub cost: CostPoint,
}

/// CEP:WHAT: The Top-K gate — first VERIFIED candidate wins.
/// CEP:WHY: Budget-capped (k <= 10), sequential, post-enumeration: the
///          architecture's "SMT at the gate only" contract. Violators are
///          skipped (their violation is reportable); an empty result sends
///          the caller to the Tier-2 fallback.
/// CEP:STATUS: complete
/// CEP:FAILURE: None when no candidate verifies.
/// CEP:ASSUMES: candidates ordered by the deterministic total order.
/// CEP:COST: O(k × verify_candidate).
/// CEP:EVIDENCE: `top_k_gate_skips_violators`.
/// CEP:SECURITY: no untrusted input.
pub fn verify_candidates(
    arena: &IrArena,
    poly: &SchedulePolytope,
    candidates: &[TopCandidate],
) -> Option<TopCandidate> {
    for cand in candidates.iter().take(10) {
        if verify_candidate(arena, poly, &cand.sched).is_ok() {
            return Some(cand.clone());
        }
    }
    None
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;
    use crate::polytope::{HardwareModel, PackedOpSched};
    use crate::wpa::extract_polytope;
    use xir_core::text::parse_arena;

    const SRC: &str = concat!(
        "xir v1 func @main {\n",
        "  %0 = param 0 : tensor<f64>[8,8]\n",
        "  %1 = param 1 : tensor<f64>[8,8]\n",
        "  %2 = matmul ta=false tb=false %0, %1 : tensor<f64>[8,8]\n",
        "  %3 = param 2 : tensor<f64>[8,8]\n",
        "  %4 = binary.add %2, %3 : tensor<f64>[8,8]\n",
        "}\n"
    );

    fn poly_and_sched(arena: &IrArena) -> (SchedulePolytope, Vec<PackedOpSched>) {
        let poly = extract_polytope(arena, HardwareModel::reference());
        let sched: Vec<PackedOpSched> = poly
            .op_bounds
            .iter()
            .map(|b| PackedOpSched {
                op_id: b.op_id.0,
                tile: b.tile_max,
                fuse_target: PackedOpSched::NO_FUSE,
                mem_level: b.mem_max as u8,
                vec_width: 1,
            })
            .collect();
        (poly, sched)
    }

    // CEP:WHAT: The gate accepts a sound candidate (bounds-max schedule).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a false rejection.
    // CEP:ASSUMES: WPA bounds are satisfiable at their maxima.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn verifier_accepts_sound_candidate() {
        let arena = parse_arena(SRC, 64).expect("parse");
        let (poly, sched) = poly_and_sched(&arena);
        assert!(verify_candidate(&arena, &poly, &sched).is_ok());
        let gated = verify_candidates(
            &arena,
            &poly,
            &[TopCandidate {
                sched: sched.clone(),
                cost: CostPoint {
                    cycles: 1,
                    hbm_bytes: 1,
                    sram_usage: 1,
                },
            }],
        );
        assert!(gated.is_some());
    }

    // CEP:WHAT: The gate rejects an illegal fuse pair (a PARAM node —
    //           not a tensor op — blessed into valid_fuse so ONLY the
    //           verifier's can_fuse re-check catches it).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a false acceptance.
    // CEP:ASSUMES: params refuse fusion (NotTensorOp).
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn verifier_rejects_illegal_fuse() {
        let arena = parse_arena(SRC, 64).expect("parse");
        let (mut poly, mut sched) = poly_and_sched(&arena);
        let mut params: Vec<NodeId> = Vec::new();
        arena.for_each_live_node(|id, node| {
            if matches!(node.op, xir_core::op::Op::Param { .. }) {
                params.push(id);
            }
        });
        let param_id = params.first().copied().expect("param exists");
        poly.op_bounds[0].valid_fuse.push(param_id);
        sched[0].fuse_target = param_id.0;
        assert!(matches!(
            verify_candidate(&arena, &poly, &sched),
            Err(SuperOptError::ConstraintViolation(
                ConstraintViolation::FuseIllegal(0)
            ))
        ));
    }

    // CEP:WHAT: The Top-K gate skips violators and returns the first sound one.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a violator wins.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn top_k_gate_skips_violators() {
        let arena = parse_arena(SRC, 64).expect("parse");
        let (poly, sched) = poly_and_sched(&arena);
        let mut bad = sched.clone();
        bad[0].tile[0] = poly.op_bounds[0].tile_max[0] + 1; // bound violation
        let gated = verify_candidates(
            &arena,
            &poly,
            &[
                TopCandidate {
                    sched: bad,
                    cost: CostPoint {
                        cycles: 0,
                        hbm_bytes: 0,
                        sram_usage: 0,
                    },
                },
                TopCandidate {
                    sched: sched.clone(),
                    cost: CostPoint {
                        cycles: 9,
                        hbm_bytes: 9,
                        sram_usage: 9,
                    },
                },
            ],
        );
        assert_eq!(gated.expect("gated").sched, sched);
    }
}
