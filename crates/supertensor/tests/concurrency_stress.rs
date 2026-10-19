#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion (waiver W-5)"
)]

// CEP:FILE: crates/supertensor/tests/concurrency_stress.rs
// CEP:WHAT: Integration test — the concurrency torture: repeated searches on
//           a wide polytope across worker counts must (a) terminate under
//           the budgets (no hang, by construction and by test), (b) produce
//           IDENTICAL frontier content every run and at every worker count
//           (the module-header determinism proof, executed), and (c) never
//           emit a candidate outside the polytope.
// CEP:WHY: The no-hangs/no-nondeterminism contract: nondeterministic
//          interleaving with deterministic RESULTS (CEP&CC 38.10) — the
//          executable witness that prune hints are result-invariant.
// CEP:CLASS: CEP-29
// CEP:STATUS: complete
// CEP:FAILURE: assert (or test timeout) fires on any divergence/hang.
// CEP:ASSUMES: public API; CI box has >= 1 core.
// CEP:COST: test-only; ~10 searches on a wide domain, budget-bounded.
// CEP:EVIDENCE: this file.
// CEP:SECURITY: no untrusted input.

use supertensor::enumerator::search;
use supertensor::verifier::verify_candidate;
use supertensor::wpa::extract_polytope;
use supertensor::{HardwareModel, SchedulePolytope};
use xir_core::text::parse_arena;

const WIDE_SRC: &str = concat!(
    "xir v1 func @main {\n",
    "  %0 = param 0 : tensor<f64>[32,32]\n",
    "  %1 = param 1 : tensor<f64>[32,32]\n",
    "  %2 = matmul ta=false tb=false %0, %1 : tensor<f64>[32,32]\n",
    "  %3 = param 2 : tensor<f64>[32,32]\n",
    "  %4 = binary.add %2, %3 : tensor<f64>[32,32]\n",
    "  %5 = unary.relu %4 : tensor<f64>[32,32]\n",
    "  %6 = binary.mul %5, %5 : tensor<f64>[32,32]\n",
    "}\n"
);

fn fingerprint(poly: &SchedulePolytope, workers: usize) -> Vec<(u64, u64, u64)> {
    let out = search(poly, workers);
    out.frontier
        .top_k(64)
        .into_iter()
        .map(|p| (p.cost.cycles, p.cost.hbm_bytes, p.cost.sram_usage))
        .collect()
}

// CEP:WHAT: Repeated searches (same worker count) agree exactly.
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on any content divergence.
// CEP:ASSUMES: none
// CEP:COST: test-only; 5 runs
// CEP:EVIDENCE: this test
#[test]
fn repeated_searches_are_identical() {
    let arena = parse_arena(WIDE_SRC, 128).expect("parse");
    let poly = extract_polytope(&arena, HardwareModel::reference());
    let reference = fingerprint(&poly, 2);
    assert!(!reference.is_empty());
    for _ in 0..4 {
        assert_eq!(fingerprint(&poly, 2), reference);
    }
    // Every point stays in-region under repetition.
    for _ in 0..2 {
        let out = search(&poly, 2);
        for p in out.frontier.top_k(64).iter() {
            assert!(verify_candidate(&arena, &poly, &p.sched).is_ok());
        }
    }
}

// CEP:WHAT: Worker count does not change the frontier content (disjoint-union
//           partitioning is content-invariant).
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on any content divergence.
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn worker_count_is_content_invariant() {
    let arena = parse_arena(WIDE_SRC, 128).expect("parse");
    let poly = extract_polytope(&arena, HardwareModel::reference());
    let one = fingerprint(&poly, 1);
    let two = fingerprint(&poly, 2);
    let four = fingerprint(&poly, 4);
    assert_eq!(one, two);
    assert_eq!(two, four);
}

// CEP:WHAT: Budgets bound the work: evaluated <= workers * cap and the search
//           terminates (the test finishing IS the no-hang evidence).
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on a budget overrun; timeout on a hang.
// CEP:ASSUMES: MAX_CAND_PER_WORKER from the enumerator.
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn budgets_bound_the_search() {
    let arena = parse_arena(WIDE_SRC, 128).expect("parse");
    let poly = extract_polytope(&arena, HardwareModel::reference());
    let out = search(&poly, 2);
    assert!(out.evaluated <= 2 * supertensor::enumerator::MAX_CAND_PER_WORKER);
    assert!(!out.frontier.is_empty());
}
