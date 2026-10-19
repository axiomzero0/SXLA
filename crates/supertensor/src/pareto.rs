// CEP:FILE: crates/supertensor/src/pareto.rs
// CEP:WHAT: The Pareto frontier merger — single-writer dominance-filtered
//           point set with a bounded capacity, a deterministic total order
//           (cycles, then hbm_bytes, then lexicographic schedule), and the
//           packed prune-anchor publication used by the enumerators.
// CEP:WHY: Multi-objective selection must not collapse early (runtime vs
//          traffic vs footprint); the total order makes the winner a
//          deterministic function of the frontier CONTENT (never arrival
//          order — CEP&CC 38.10). The anchor packs (cycles, hbm) into ONE
//          atomic word so workers observe a tear-free snapshot.
// CEP:CLASS: CEP-29
// CEP:STATUS: complete
// CEP:FAILURE: insert returns false for dominated/duplicate points (never a
//              panic); anchor decoding saturates (sound: a saturated anchor
//              prunes LESS).
// CEP:ASSUMES: single merger thread (the enumerator's main thread).
// CEP:COST: insert = O(frontier) dominance scan (frontier <= 64 — contention
//           free by design, ~1µs); anchor publish = 1 store.
// CEP:EVIDENCE: tests `dominance_filter_admits_only_frontier`,
//           `total_order_is_deterministic`, `anchor_pack_roundtrips`.
// CEP:SECURITY: no untrusted input.
//! Pareto frontier and prune anchor.

use crate::cost::{dominates, CostPoint};
use crate::polytope::PackedOpSched;

/// One frontier point: the cost vector + the schedule that produced it.
#[derive(Debug, Clone)]
pub struct FrontierPoint {
    /// The evaluated cost.
    pub cost: CostPoint,
    /// The producing schedule.
    pub sched: Vec<PackedOpSched>,
}

/// CEP:WHAT: Total order over frontier points — cycles, then hbm_bytes, then
///           lexicographic (op_id, tile, fuse, mem, vec) schedule comparison.
/// CEP:WHY: Determinism: the winner is the frontier minimum under THIS order,
///          independent of insertion order (completion-order independence).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (Ordering).
/// CEP:ASSUMES: none
/// CEP:COST: O(schedule length) worst case.
/// CEP:EVIDENCE: `total_order_is_deterministic`
/// CEP:SECURITY: no untrusted input
pub fn total_order(a: &FrontierPoint, b: &FrontierPoint) -> core::cmp::Ordering {
    a.cost
        .cycles
        .cmp(&b.cost.cycles)
        .then(a.cost.hbm_bytes.cmp(&b.cost.hbm_bytes))
        .then(sched_lex(&a.sched, &b.sched))
}

/// CEP:WHAT: Lexicographic schedule comparison (the tie-break tail).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(len)
/// CEP:EVIDENCE: `total_order_is_deterministic`
/// CEP:SECURITY: no untrusted input
fn sched_lex(a: &[PackedOpSched], b: &[PackedOpSched]) -> core::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        let ord = x
            .op_id
            .cmp(&y.op_id)
            .then(x.tile.cmp(&y.tile))
            .then(x.fuse_target.cmp(&y.fuse_target))
            .then(x.mem_level.cmp(&y.mem_level))
            .then(x.vec_width.cmp(&y.vec_width));
        if ord != core::cmp::Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

/// The bounded Pareto frontier.
///
/// CEP:WHAT: Dominance-filtered point set, capacity-bounded by the total
///           order (worst points evicted first).
/// CEP:WHY: Bounded memory (Law: bounded resources) + deterministic content
///           regardless of arrival order.
/// CEP:STATUS: complete
/// CEP:FAILURE: insert returns false when dominated or duplicate.
/// CEP:ASSUMES: single writer thread.
/// CEP:COST: O(n) insert.
/// CEP:EVIDENCE: `dominance_filter_admits_only_frontier`
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    points: Vec<FrontierPoint>,
    cap: usize,
    /// Advisory: set when a capacity truncation evicted live points
    /// (determinism precondition surface — see enumerator header).
    evicted: bool,
}

impl ParetoFrontier {
    /// CEP:WHAT: Empty frontier with the default capacity (64).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn new() -> ParetoFrontier {
        ParetoFrontier {
            points: Vec::new(),
            cap: 64,
            evicted: false,
        }
    }

    /// CEP:WHAT: Attempts to insert a point.
    /// CEP:WHY: Admission rule (order-independent by construction):
    ///          - reject if an existing point dominates it;
    ///          - reject if an EQUAL-COST existing point has lex <= ours
    ///            (among equal-cost twins only the lex-MINIMUM survives);
    ///          - evict every point it dominates AND every equal-cost twin
    ///            with lex > ours; enforce capacity by total order.
    ///          Net effect: each equal-cost class converges to its lex-min
    ///          regardless of arrival order — the frontier CONTENT is a
    ///          deterministic function of the candidate SET (CEP&CC 38.10;
    ///          the arrival-order bug this replaces admitted a decreasing
    ///          lex CHAIN per class — caught by search_is_deterministic).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: false when the point adds nothing (dominated/out-ranked).
    /// CEP:ASSUMES: single writer.
    /// CEP:COST: O(n) scan + O(n) eviction.
    /// CEP:EVIDENCE: `dominance_filter_admits_only_frontier`,
    ///           `total_order_is_deterministic`, `search_is_deterministic`.
    /// CEP:SECURITY: no untrusted input
    pub fn insert(&mut self, point: FrontierPoint) -> bool {
        for existing in self.points.iter() {
            if dominates(&existing.cost, &point.cost) {
                return false;
            }
            if existing.cost == point.cost
                && sched_lex(&existing.sched, &point.sched) != core::cmp::Ordering::Greater
            {
                return false;
            }
        }
        self.points.retain(|p| {
            !dominates(&point.cost, &p.cost)
                && !(p.cost == point.cost
                    && sched_lex(&p.sched, &point.sched) == core::cmp::Ordering::Greater)
        });
        self.points.push(point);
        if self.points.len() > self.cap {
            self.points.sort_by(total_order);
            self.points.truncate(self.cap);
            self.evicted = true;
        }
        true
    }

    /// CEP:WHAT: The winner — frontier minimum under the total order.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None when empty.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(n)
    /// CEP:EVIDENCE: `total_order_is_deterministic`
    /// CEP:SECURITY: no untrusted input
    pub fn best(&self) -> Option<&FrontierPoint> {
        self.points.iter().min_by(|a, b| total_order(a, b))
    }

    /// CEP:WHAT: The k best points under the total order (the verifier's
    ///           Top-K input).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: empty vec when empty.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(n log n)
    /// CEP:EVIDENCE: constraint-soundness integration test
    /// CEP:SECURITY: no untrusted input
    pub fn top_k(&self, k: usize) -> Vec<FrontierPoint> {
        let mut v = self.points.clone();
        v.sort_by(total_order);
        v.truncate(k);
        v
    }

    /// CEP:WHAT: Point count.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// CEP:WHAT: Emptiness probe.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// CEP:WHAT: Advisory capacity-eviction flag (determinism surface).
    /// CEP:WHY: See the enumerator's conditioned determinism claim.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: enumerator tests
    /// CEP:SECURITY: no untrusted input
    pub fn evicted(&self) -> bool {
        self.evicted
    }
}

impl Default for ParetoFrontier {
    fn default() -> Self {
        ParetoFrontier::new()
    }
}

/// The packed prune anchor — one atomic word.
///
/// CEP:WHAT: (cycles, hbm, sram) each saturated to 21 bits, packed
///           MSB-first into a u64; published by the merger, read by
///           enumerators.
/// CEP:WHY: A tear-free three-field snapshot WITHOUT locks — and the three
///          fields EXACTLY mirror `dominates()`'s three axes, so "pruned"
///           means "dominated by the anchor point" (the frontier-invariance
///          proof's premise; the two-axis form this replaces pruned points
///          the frontier would have kept — review F1). Saturation only
///          WEAKENS pruning (larger anchor values dominate fewer
///          candidates) — sound under overflow. The sentinel u64::MAX is
///          unreachable by any packed value (max packed = 0x1F_FFFF_FFFF_FFFF).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: anchor values fit 21 bits or saturate (documented).
/// CEP:COST: pack/unpack = shifts; publish = 1 Relaxed store (the anchor is
///           advisory: soundness never depends on it — see enumerator).
/// CEP:EVIDENCE: `anchor_pack_roundtrips`
/// CEP:SECURITY: no untrusted input
pub const ANCHOR_NONE: u64 = u64::MAX;

/// 21-bit field saturation bound.
const ANCHOR_SAT: u64 = (1u64 << 21) - 1;

/// CEP:WHAT: Packs (cycles, hbm, sram) into one word.
/// CEP:STATUS: complete
/// CEP:FAILURE: saturates per field.
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: `anchor_pack_roundtrips`
/// CEP:SECURITY: no untrusted input
pub fn pack_anchor(cycles: u64, hbm: u64, sram: u64) -> u64 {
    let c = cycles.min(ANCHOR_SAT);
    let h = hbm.min(ANCHOR_SAT);
    let s = sram.min(ANCHOR_SAT);
    (c << 42) | (h << 21) | s
}

/// CEP:WHAT: Unpacks an anchor word into (cycles, hbm, sram), saturated.
/// CEP:STATUS: complete
/// CEP:FAILURE: ANCHOR_NONE decodes to all-saturated values that dominate
///              nothing — pruning stays off until the first publication.
/// CEP:ASSUMES: word from pack_anchor or ANCHOR_NONE.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: `anchor_pack_roundtrips`
/// CEP:SECURITY: no untrusted input
pub fn unpack_anchor(word: u64) -> (u64, u64, u64) {
    if word == ANCHOR_NONE {
        return (u64::MAX, u64::MAX, u64::MAX);
    }
    let c = (word >> 42) & ANCHOR_SAT;
    let h = (word >> 21) & ANCHOR_SAT;
    let s = word & ANCHOR_SAT;
    (c, h, s)
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;

    fn pt(cycles: u64, hbm: u64) -> FrontierPoint {
        FrontierPoint {
            cost: CostPoint {
                cycles,
                hbm_bytes: hbm,
                sram_usage: 0,
            },
            sched: vec![],
        }
    }

    // CEP:WHAT: Dominance filtering admits only frontier points.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a dominated admission.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn dominance_filter_admits_only_frontier() {
        let mut f = ParetoFrontier::new();
        assert!(f.insert(pt(10, 100)));
        assert!(f.insert(pt(20, 50))); // trade-off — both survive
        assert!(!f.insert(pt(15, 120))); // dominated by (10,100)
        assert!(!f.insert(pt(10, 100))); // duplicate
        assert!(f.insert(pt(5, 400))); // third trade-off point
                                       // (5,400) does not dominate the others (hbm worse).
        assert_eq!(f.len(), 3);
        // A point dominating (20,50) evicts it.
        assert!(f.insert(pt(15, 40)));
        assert_eq!(f.len(), 3);
    }

    // CEP:WHAT: The total order picks a deterministic winner among equal-cost
    //           trade-offs (ties broken by hbm).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on an order flip.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn total_order_is_deterministic() {
        let mut f = ParetoFrontier::new();
        assert!(f.insert(pt(10, 100)));
        assert!(f.insert(pt(10, 60)));
        assert!(!f.insert(pt(10, 60))); // duplicate — rejected
        let best = f.best().expect("non-empty");
        assert_eq!((best.cost.cycles, best.cost.hbm_bytes), (10, 60));
        // Same points inserted in reverse order yield the same winner: here
        // (10,100) is DOMINATED by the already-present (10,60) — rejected.
        let mut g = ParetoFrontier::new();
        assert!(g.insert(pt(10, 60)));
        assert!(!g.insert(pt(10, 100)));
        assert_eq!(g.best().map(|b| b.cost), f.best().map(|b| b.cost));
    }

    // CEP:WHAT: Anchor packing round-trips and saturates soundly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a tear or overflow.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn anchor_pack_roundtrips() {
        let w = pack_anchor(12345, 678, 9);
        assert_eq!(unpack_anchor(w), (12345, 678, 9));
        let s = pack_anchor(u64::MAX, 0, u64::MAX);
        assert_eq!(unpack_anchor(s).0, ANCHOR_SAT);
        // The NONE sentinel decodes weaker than any real saturated anchor.
        let (c, h, s) = unpack_anchor(ANCHOR_NONE);
        assert!(c >= ANCHOR_SAT && h >= ANCHOR_SAT && s >= ANCHOR_SAT);
        // No packed value collides with the sentinel.
        assert_ne!(pack_anchor(ANCHOR_SAT, ANCHOR_SAT, ANCHOR_SAT), ANCHOR_NONE);
    }
}
