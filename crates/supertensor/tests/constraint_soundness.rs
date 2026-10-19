#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion (waiver W-5)"
)]

// CEP:FILE: crates/supertensor/tests/constraint_soundness.rs
// CEP:WHAT: Integration test — every Pareto point the search emits satisfies
//           the polytope's constraints EXACTLY (bounds, divisibility, linear
//           rows, SRAM pressure, fusion legality): the constructive
//           soundness proof of the whole Stage-2 pipeline, executed.
// CEP:WHY: The enumerator's prefix checks are fast-path filters; the gate's
//          check_full is the authority. This test pins that NOTHING the
//          search emits can violate the region — the spec's "no candidate
//          outside this region is ever enumerated", verified post-hoc over
//          the emitted frontier (the strongest executable form).
// CEP:CLASS: CEP-29
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on any out-of-region emission.
// CEP:ASSUMES: public API surface (lib.rs).
// CEP:COST: test-only.
// CEP:EVIDENCE: this file.
// CEP:SECURITY: no untrusted input.

use supertensor::enumerator::search;
use supertensor::verifier::verify_candidate;
use supertensor::wpa::extract_polytope;
use supertensor::HardwareModel;
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

// CEP:WHAT: Every frontier point passes the exact constructive check.
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on any violation (names the schedule).
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn all_frontier_points_satisfy_polytope() {
    let arena = parse_arena(SRC, 64).expect("parse");
    let poly = extract_polytope(&arena, HardwareModel::reference());
    assert!(poly.op_count() >= 3, "matmul + add + relu");
    let outcome = search(&poly, anvil::default_worker_count().max(1));
    let points = outcome.frontier.top_k(64);
    assert!(!points.is_empty(), "search must emit candidates");
    for p in points.iter() {
        assert!(
            verify_candidate(&arena, &poly, &p.sched).is_ok(),
            "frontier point violated the polytope: {:?}",
            p.sched
        );
    }
}

// CEP:WHAT: The gate accepts the search's own Top-K winner (pipeline
//           end-to-end soundness through lib::superoptimize_arena).
// CEP:STATUS: complete
// CEP:FAILURE: assert fires when the winner cannot be verified.
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn search_winner_passes_the_gate() {
    let arena = parse_arena(SRC, 64).expect("parse");
    let outcome = supertensor::superoptimize_arena(&arena);
    assert!(outcome.is_ok(), "superoptimize failed: {:?}", outcome.err());
}
