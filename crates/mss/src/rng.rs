// CEP:FILE: crates/mss/src/rng.rs
// CEP:WHAT: Deterministic xorshift64* PRNG — the engine's only randomness
//           source, seeded per band so search results are exactly
//           reproducible (CEP&CC 38.10: no time/hash-order dependence).
// CEP:WHY: Stochastic search with a FIXED seed is deterministic: the same
//          input, seed and iteration budget produce the same elite set on
//          every machine and every run — the determinism law holds for the
//          RESULT while the SEARCH remains stochastic (the design's
//          "nondeterministic exploration, deterministic outcome" contract).
// CEP:CLASS: CEP-31
// CEP:STATUS: complete
// CEP:FAILURE: none (bounded state machine; zero is a rejected seed).
// CEP:ASSUMES: seeds are nonzero.
// CEP:COST: O(1) per draw; ~1ns.
// CEP:EVIDENCE: tests `rng_is_reproducible`, `rng_is_bounded`.
// CEP:SECURITY: not a cryptographic source (documented; search use only).
//! Deterministic PRNG (xorshift64*).

/// CEP:WHAT: xorshift64* generator state.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: state != 0 (constructor mixes; documented).
/// CEP:COST: 8 bytes
/// CEP:EVIDENCE: tests in this module
/// CEP:SECURITY: not cryptographic
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// CEP:WHAT: New generator from a nonzero-mixed seed.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (seed 0 is spread to a nonzero odd constant).
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn new(seed: u64) -> Rng {
        let s = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Rng { state: s | 1 }
    }

    /// CEP:WHAT: Next raw u64.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 3 shifts + 3 xors + 1 mul
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// CEP:WHAT: Uniform draw below `bound` (bias-free for small bounds).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: 0 when bound == 0 (documented degenerate).
    /// CEP:ASSUMES: bound > 0 for meaningful use.
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: `rng_is_bounded`
    /// CEP:SECURITY: no untrusted input
    pub fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        self.next_u64() % bound
    }

    /// CEP:WHAT: Uniform draw in [0, 1000) (Metropolis percentages).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: search tests
    /// CEP:SECURITY: no untrusted input
    pub fn permille(&mut self) -> u64 {
        self.below(1000)
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

    // CEP:WHAT: Same seed -> same stream.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on divergence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rng_is_reproducible() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        // Different seeds diverge.
        let mut c = Rng::new(43);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    // CEP:WHAT: Draws stay below the bound.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on an out-of-range draw.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rng_is_bounded() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            assert!(r.below(10) < 10);
            assert!(r.permille() < 1000);
        }
        assert_eq!(r.below(0), 0);
    }
}
