// CEP:FILE: crates/mss/src/tt.rs
// CEP:WHAT: The V0 transposition table: an in-memory HashSet of canonical
//           hashes with a DETERMINISTIC FNV build hasher (std's random
//           per-process SipHash keys would break reproducibility — 38.10)
//           and per-band ownership (no cross-thread sharing in V0; the
//           Cuckoo-filter + LSM-tree two-level store of the design is the
//           CEP-31 V1 persistence layer).
// CEP:WHY: Different mutation paths converge on the same effective graph;
//           the TT prunes them (design §9). SEEN -> mark and skip.
// CEP:CLASS: CEP-31
// CEP:STATUS: complete (V0 in-memory form)
// CEP:FAILURE: none (insert is idempotent).
// CEP:ASSUMES: single-band ownership (no concurrent access).
// CEP:COST: O(1) amortized per probe.
// CEP:EVIDENCE: tests `tt_is_deterministic`, `tt_dedups`.
// CEP:SECURITY: no untrusted input.
//! V0 transposition table.

use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};

/// FNV-1a 64-bit hasher.
///
/// CEP:WHAT: Deterministic hasher (zero state, no random keys).
/// CEP:WHY: Process-stable hashes: identical inputs -> identical TT
///          behavior across machines and runs.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(bytes)
/// CEP:EVIDENCE: `tt_is_deterministic`
/// CEP:SECURITY: not cryptographic
#[derive(Debug, Clone, Copy, Default)]
pub struct FnvHasher {
    state: u64,
}

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.state == 0 {
            FNV_OFFSET
        } else {
            self.state
        };
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        self.state = h;
    }
}

/// BuildHasher for the FNV hasher.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildFnv;

impl BuildHasher for BuildFnv {
    type Hasher = FnvHasher;

    fn build_hasher(&self) -> FnvHasher {
        FnvHasher { state: FNV_OFFSET }
    }
}

/// The per-band transposition table.
///
/// CEP:WHAT: seen-set of canonical hashes.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: owned by one band (no sync needed).
/// CEP:COST: O(1) amortized probes.
/// CEP:EVIDENCE: `tt_dedups`
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Default)]
pub struct TranspositionTable {
    seen: HashSet<u64, BuildFnv>,
}

impl TranspositionTable {
    /// CEP:WHAT: Empty table.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn new() -> TranspositionTable {
        TranspositionTable {
            seen: HashSet::with_hasher(BuildFnv),
        }
    }

    /// CEP:WHAT: Probes and (on miss) records a hash.
    /// CEP:WHY: Returns true when FRESH (caller continues the pipeline);
    ///          false when SEEN (caller prunes).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1) amortized
    /// CEP:EVIDENCE: `tt_dedups`
    /// CEP:SECURITY: no untrusted input
    pub fn probe_and_insert(&mut self, hash: u64) -> bool {
        self.seen.insert(hash)
    }

    /// CEP:WHAT: Recorded-hash count.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// CEP:WHAT: Emptiness probe.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
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

    // CEP:WHAT: The build hasher is process-stable (same values, same set).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on randomized hashing.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tt_is_deterministic() {
        let mut h1 = BuildFnv.build_hasher();
        let mut h2 = BuildFnv.build_hasher();
        for v in [1u64, 2, 3, 42] {
            h1.write(&v.to_le_bytes());
            h2.write(&v.to_le_bytes());
        }
        assert_eq!(h1.finish(), h2.finish());
        assert_ne!(h1.finish(), 0);
    }

    // CEP:WHAT: Duplicate hashes are pruned (second probe returns false).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a missed dedup.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tt_dedups() {
        let mut tt = TranspositionTable::new();
        assert!(tt.probe_and_insert(0xABCD));
        assert!(!tt.probe_and_insert(0xABCD));
        assert!(tt.probe_and_insert(0x1234));
        assert_eq!(tt.len(), 2);
        assert!(!tt.is_empty());
    }
}
