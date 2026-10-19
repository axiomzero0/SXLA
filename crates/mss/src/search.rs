// CEP:FILE: crates/mss/src/search.rs
// CEP:WHAT: The V0 search driver — parallel-tempering MCMC over the SoN
//           program space: four temperature bands (HOT/WARM/COLD/ELITE) run
//           as Gear-0 static partitions (anvil::run_partitioned over band
//           indices, per-band deterministic rng/TT), each performing
//           Metropolis-accepted mutations with tiered filtering (tier-0
//           structural check, tier-1 preconditions inside the operators,
//           canonical hash, TT dedup, cheap cost); the merge is a
//           deterministic total-order reduction over band elites.
// CEP:WHY: Design §11: temperature bands trade exploration for refinement;
//          acceptance is Metropolis (exp(-dE/T)); elitism preserves the best
//          (design §11.5). V0 deviations, documented: (a) bands are
//          independent Gear-0 partitions with a deterministic final merge —
//          async gossip replica exchange is the CEP-31 V1 layer; (b) the
//          population is the (current, elite) pair per band, not a
//          niched population — fitness sharing arrives with V1. Both keep
//          the determinism contract: identical inputs -> identical results.
// CEP:CLASS: CEP-31
// CEP:STATUS: complete (V0 form)
// CEP:FAILURE: search never fails; a band with no applicable mutations
//              returns its input as elite.
// CEP:ASSUMES: bounded iteration budget (no hangs by construction).
// CEP:COST: O(iters * n) per band; workers = anvil default.
// CEP:EVIDENCE: tests `search_finds_the_reduction`,
//           `search_is_deterministic`, `search_reports_throughput`.
// CEP:SECURITY: no untrusted input.
//! Parallel-tempering MCMC search (V0).

use anvil::run_partitioned;

use crate::cand::Candidate;
use crate::cost;
use crate::mutate::{self, FAMILIES};
use crate::rng::Rng;
use crate::son::Program;
use crate::tt::TranspositionTable;

/// Temperature bands (design §11.1).
///
/// CEP:WHAT: HOT explores (huge structural acceptance), ELITE refines.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 8 bytes
/// CEP:EVIDENCE: search tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    /// Huge mutations, high diversity, low commitment.
    Hot,
    /// Moderate.
    Warm,
    /// Local refinements.
    Cold,
    /// Tiny perturbations of the best-known.
    Elite,
}

impl Band {
    /// CEP:WHAT: Metropolis temperature for the band.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: search tests
    /// CEP:SECURITY: no untrusted input
    pub fn temperature(self) -> f64 {
        match self {
            Band::Hot => 100.0,
            Band::Warm => 30.0,
            Band::Cold => 10.0,
            Band::Elite => 3.0,
        }
    }

    /// CEP:WHAT: Band index.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: partition tests
    /// CEP:SECURITY: no untrusted input
    pub fn index(self) -> usize {
        match self {
            Band::Hot => 0,
            Band::Warm => 1,
            Band::Cold => 2,
            Band::Elite => 3,
        }
    }
}

/// All bands in order.
pub const BANDS: [Band; 4] = [Band::Hot, Band::Warm, Band::Cold, Band::Elite];

/// Per-band iteration budget (V0: fixed; the PID-controlled adaptive budget
/// is the CEP-31 V3 JIT policy layer).
pub const ITERS_PER_BAND: u64 = 400;

/// Search outcome.
///
/// CEP:WHAT: The global elite (best candidate by the deterministic total
///           order) + diagnostics (candidates generated/accepted/pruned,
///           nodes touched, wall-time for the nodes/sec metric).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: owned candidate + counters
/// CEP:EVIDENCE: search tests
/// CEP:SECURITY: no untrusted input
pub struct SearchOutcome {
    /// The winning candidate (materialize() for the program).
    pub best: Candidate,
    /// Elite cost of the winner.
    pub best_cost: crate::cost::Cost,
    /// Candidates that passed all filters (generated).
    pub generated: u64,
    /// Metropolis-accepted transitions.
    pub accepted: u64,
    /// TT-pruned (seen) candidates.
    pub pruned: u64,
    /// Node-touch counter (the throughput metric's numerator).
    pub nodes_touched: u64,
}

/// CEP:WHAT: Tier-0 structural legality over the materialized program.
/// CEP:WHY: The gauntlet's first gate (design §8): valid arities, no
///          dangling operands, root live. Cheap, deterministic.
/// CEP:STATUS: complete
/// CEP:FAILURE: false on any violation.
/// CEP:ASSUMES: none
/// CEP:COST: O(n)
/// CEP:EVIDENCE: `search_finds_the_reduction` (indirect); filter unit path.
/// CEP:SECURITY: no untrusted input
fn tier0_legal(cand: &Candidate) -> bool {
    let n = cand.slot_count() as u32;
    for slot in 0..n {
        let def = cand.view(slot);
        match def.kind {
            Kind::Const | Kind::Param => {
                if def.lhs != u32::MAX || def.rhs != u32::MAX {
                    return false;
                }
            }
            Kind::Bin => {
                if def.lhs == u32::MAX || def.rhs == u32::MAX || def.lhs >= n || def.rhs >= n {
                    return false;
                }
            }
            Kind::Un | Kind::Alias => {
                if def.lhs == u32::MAX || def.lhs >= n || def.rhs != u32::MAX {
                    return false;
                }
            }
            Kind::Dead => {}
        }
    }
    if cand.root >= n {
        return false;
    }
    matches!(
        cand.view(cand.root).kind,
        Kind::Const | Kind::Param | Kind::Bin | Kind::Un | Kind::Alias
    )
}

use crate::son::Kind;

/// One band's search result (the partition output).
#[derive(Clone)]
struct BandResult {
    elite: Candidate,
    elite_cost: crate::cost::Cost,
    generated: u64,
    accepted: u64,
    pruned: u64,
    nodes_touched: u64,
}

/// CEP:WHAT: Runs one band's MCMC chain from the seed candidate.
/// CEP:WHY: Fixed budget; per-band rng (seed mixed with the band index)
///          keeps chains independent and reproducible; Metropolis accepts
///          uphill moves at rate exp(-dE/T).
/// CEP:STATUS: complete
/// CEP:FAILURE: never (elite defaults to the seed).
/// CEP:ASSUMES: none
/// CEP:COST: O(iters x n)
/// CEP:EVIDENCE: search tests
/// CEP:SECURITY: no untrusted input
fn run_band(seed: &Candidate, band: Band) -> BandResult {
    let mut rng = Rng::new(0x5EED_0000 ^ (band.index() as u64 + 1));
    let mut tt = TranspositionTable::new();
    let mut current = seed.clone();
    let mut current_cost = cost::evaluate(&current);
    let mut elite = seed.clone();
    let mut elite_cost = current_cost;
    let (mut generated, mut accepted, mut pruned, mut touched) = (0u64, 0u64, 0u64, 0u64);
    for _ in 0..ITERS_PER_BAND {
        // Cost-biased family dispatch (V0: uniform deterministic draw).
        let family = FAMILIES[rng.below(FAMILIES.len() as u64) as usize];
        let child = match mutate::apply(&current, family, &mut rng) {
            Some(c) => c,
            None => continue, // no applicable site — retry
        };
        touched += child.slot_count() as u64 + current.slot_count() as u64;
        // Tier 0: structural legality.
        if !tier0_legal(&child) {
            continue;
        }
        // Tier 2/3: canonical hash + TT dedup.
        let hash = child.canonical_hash();
        touched += child.slot_count() as u64;
        if !tt.probe_and_insert(hash) {
            pruned += 1;
            continue;
        }
        // Tier 4: cheap cost + Metropolis acceptance.
        let child_cost = cost::evaluate(&child);
        touched += child.slot_count() as u64;
        generated += 1;
        let de = child_cost.total as f64 - current_cost.total as f64;
        let accept = if de <= 0.0 {
            true
        } else {
            // exp(-de / T) as a permille threshold (deterministic f64 math).
            let p = (-de / band.temperature()).exp();
            (rng.permille() as f64) < p * 1000.0
        };
        if accept {
            accepted += 1;
            current = child;
            current_cost = child_cost;
            // Elitism: strictly better under the total order.
            if candidate_order_strict(&current_cost, &elite_cost, &current, &elite) {
                elite = current.clone();
                elite_cost = current_cost;
            }
        }
    }
    BandResult {
        elite,
        elite_cost,
        generated,
        accepted,
        pruned,
        nodes_touched: touched,
    }
}

/// Strict elite comparison (total cost FIRST, then depth, then hash —
/// the documented total order; review F4 fixed the (_, Less) arm that let
/// a worse-total-but-shallower candidate displace the elite).
fn candidate_order_strict(
    ca: &crate::cost::Cost,
    cb: &crate::cost::Cost,
    a: &Candidate,
    b: &Candidate,
) -> bool {
    match (ca.total.cmp(&cb.total), ca.depth.cmp(&cb.depth)) {
        (core::cmp::Ordering::Less, _) => true,
        (core::cmp::Ordering::Equal, core::cmp::Ordering::Less) => true,
        (core::cmp::Ordering::Equal, core::cmp::Ordering::Equal) => {
            a.canonical_hash() < b.canonical_hash()
        }
        _ => false,
    }
}

/// CEP:WHAT: The search entry point — Gear-0 partition over the four bands
///           + deterministic merge.
/// CEP:WHY: Bands partition independently (disjoint outputs at fixed
///          indices — the run_partitioned contract); the merge reduces by
///          the total order, so the RESULT is independent of worker
///          interleaving (HPC-0 determinism, same pattern as the fusion
///          search and the supertensor enumerator).
/// CEP:STATUS: complete
/// CEP:FAILURE: never; the seed is always a valid elite fallback.
/// CEP:ASSUMES: seed program well-formed.
/// CEP:COST: O(bands x iters x n).
/// CEP:EVIDENCE: module tests + differential suite.
/// CEP:SECURITY: no untrusted input.
pub fn search(seed: &Program, workers: usize) -> SearchOutcome {
    let seed_cand = Candidate::from_program(seed.clone());
    let band_indices: [usize; 4] = [0, 1, 2, 3];
    // run_partitioned maps elements to outputs; use BandResult directly
    // (no Option needed — every index gets a real result).
    let seed_cost = cost::evaluate(&seed_cand);
    let placeholder = BandResult {
        elite: seed_cand.clone(),
        elite_cost: seed_cost,
        generated: 0,
        accepted: 0,
        pruned: 0,
        nodes_touched: 0,
    };
    let mut results: Vec<BandResult> = vec![placeholder; 4];
    let _ = run_partitioned(&band_indices, &mut results, workers.max(1), |&bi| {
        run_band(&seed_cand, BANDS[bi])
    });
    // Deterministic merge: total order over band elites. The results vector
    // has one entry per band index (run_partitioned overwrites all four), so
    // the fold always sees at least one — but the law-6 form avoids expect.
    let mut best: Option<BandResult> = None;
    let (mut generated, mut accepted, mut pruned, mut touched) = (0u64, 0u64, 0u64, 0u64);
    for r in results.into_iter() {
        generated += r.generated;
        accepted += r.accepted;
        pruned += r.pruned;
        touched += r.nodes_touched;
        best = match best {
            None => Some(r),
            Some(cur) => {
                if candidate_order_strict(&r.elite_cost, &cur.elite_cost, &r.elite, &cur.elite) {
                    Some(r)
                } else {
                    Some(cur)
                }
            }
        };
    }
    // Fallback to the seed candidate (unreachable in practice — the
    // partition always writes all four indices).
    let best = best.unwrap_or(BandResult {
        elite: seed_cand,
        elite_cost: cost::evaluate(&Candidate::from_program(seed.clone())),
        generated: 0,
        accepted: 0,
        pruned: 0,
        nodes_touched: 0,
    });
    SearchOutcome {
        best_cost: best.elite_cost,
        best: best.elite,
        generated,
        accepted,
        pruned,
        nodes_touched: touched,
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;
    use crate::son::{BinOp, ProgramBuilder};

    fn seed_program() -> Program {
        // f(x) = ((x + 0) * 1) + (2 + 3)  — identities + a fold site.
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let zero = b.const_i64(0);
        let one = b.const_i64(1);
        let add = b.bin(BinOp::Add, x, zero).expect("add");
        let mul = b.bin(BinOp::Mul, add, one).expect("mul");
        let c2 = b.const_i64(2);
        let c3 = b.const_i64(3);
        let fold = b.bin(BinOp::Add, c2, c3).expect("fold");
        let root = b.bin(BinOp::Add, mul, fold).expect("root");
        b.finish(root).expect("finish")
    }

    // CEP:WHAT: The search reduces the seed (elite cost strictly below the
    //           seed's) while preserving semantics.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a wrong-value elite or no improvement.
    // CEP:ASSUMES: budget suffices for the identities/fold sites.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn search_finds_the_reduction() {
        let seed = seed_program();
        let seed_cost = cost::evaluate(&Candidate::from_program(seed.clone()));
        let out = search(&seed, anvil::default_worker_count().max(1));
        assert!(out.generated > 0, "no candidate passed the filters");
        assert!(
            out.best_cost.total < seed_cost.total,
            "elite {} vs seed {}",
            out.best_cost.total,
            seed_cost.total
        );
        // Differential: values preserved at several bindings.
        let best_prog = out.best.materialize();
        for x in [-3i64, 0, 7, 100] {
            assert_eq!(
                crate::interp::evaluate(&best_prog, &[x]),
                crate::interp::evaluate(&seed, &[x]),
                "divergence at x={}",
                x
            );
        }
    }

    // CEP:WHAT: Two searches produce the IDENTICAL elite (determinism).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any divergence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn search_is_deterministic() {
        let seed = seed_program();
        let a = search(&seed, 1);
        let b = search(&seed, 1);
        let c = search(&seed, 2);
        assert_eq!(a.best_cost, b.best_cost);
        assert_eq!(a.best.canonical_hash(), b.best.canonical_hash());
        assert_eq!(a.generated, b.generated);
        // Worker count invariance (Gear-0 disjoint-output contract).
        assert_eq!(a.best.canonical_hash(), c.best.canonical_hash());
        assert_eq!(a.best_cost, c.best_cost);
    }

    // CEP:WHAT: Throughput accounting is reported (nodes touched > 0).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires when the counter is dead.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test (the nodes/sec metric's counter witness)
    #[test]
    fn search_reports_throughput() {
        let seed = seed_program();
        let out = search(&seed, 1);
        assert!(out.nodes_touched > 0);
        assert!(out.nodes_touched >= out.generated);
    }
}
