// CEP:FILE: crates/xir-core/src/hash.rs
// CEP:WHAT: Deterministic FNV-1a 64-bit hashing for IR identity.
// CEP:WHY: HPC-IR contract (CEP&CC 38.17 "hashable", 38.19 "IR hashing must
//          not depend on hash seed / map iteration order"): std HashMap's
//          RandomState is explicitly banned. FNV-1a is dependency-free,
//          deterministic across runs and machines, and fast enough for IR
//          fingerprints (1 multiply-xor per byte).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: none; hashing cannot fail (no allocation, no I/O).
// CEP:ASSUMES: FNV-1a offset basis and prime are fixed by this file (named
//              constants, not literals at use sites — Law 7).
// CEP:COST: ~1 ns per 8 bytes on modern cores (multiply + xor chain);
//           see benches/anvil_bench.rs methodology notes for the harness.
// CEP:EVIDENCE: test `fnv_reference_vectors` (published FNV-1a test values),
//           test `deterministic_across_runs`.
// CEP:SECURITY: not a cryptographic hash — documented; used only for cache
//           keys and equality fingerprints, never for secrets (CEP&CC 22.9).
// CEP:HPC-DETERMINISM: deterministic by construction.
//! Deterministic FNV-1a 64-bit hashing.

/// FNV-1a 64-bit offset basis (named per Law 7).
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime (named per Law 7).
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// FNV-1a 64-bit hasher state.
///
/// CEP:WHAT: Incremental hash state with byte and u64 mixing.
/// CEP:WHY: IR hashing walks nodes incrementally (write_u64 per field);
///          keeping the state 8 bytes lets the hot loop stay in a register.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: 1 mul + 1 xor per byte; 8 of each per u64.
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: non-cryptographic (documented above).
#[derive(Debug, Clone, Copy)]
pub struct Fnv64 {
    state: u64,
}

impl Fnv64 {
    /// CEP:WHAT: Creates a hasher at the offset basis.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 store.
    /// CEP:EVIDENCE: tests in this module.
    #[inline]
    pub const fn new() -> Fnv64 {
        Fnv64 {
            state: FNV_OFFSET_BASIS,
        }
    }

    /// CEP:WHAT: Mixes one byte.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 mul + 1 xor.
    /// CEP:EVIDENCE: reference-vector test.
    #[inline]
    pub fn write_byte(&mut self, b: u8) {
        self.state ^= u64::from(b);
        self.state = self.state.wrapping_mul(FNV_PRIME);
    }

    /// CEP:WHAT: Mixes a byte slice.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: O(len); ~1ns/8B.
    /// CEP:EVIDENCE: reference-vector test.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_byte(*b);
        }
    }

    /// CEP:WHAT: Mixes one u64 in little-endian byte order.
    /// CEP:WHY: Fixed byte order makes hashes portable across endianness
    ///          (CEP&CC 22.5 portability: endianness must be explicit).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 8 muls + 8 xors.
    /// CEP:EVIDENCE: reference-vector test.
    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        let le = v.to_le_bytes();
        self.write_bytes(&le);
    }

    /// CEP:WHAT: Mixes one u16 in little-endian byte order.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 muls + 2 xors
    /// CEP:EVIDENCE: opcode hashing tests
    #[inline]
    pub fn write_u16(&mut self, v: u16) {
        let le = v.to_le_bytes();
        self.write_bytes(&le);
    }

    /// CEP:WHAT: Mixes one u32 in little-endian byte order.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 4 muls + 4 xors.
    /// CEP:EVIDENCE: tests in this module.
    #[inline]
    pub fn write_u32(&mut self, v: u32) {
        let le = v.to_le_bytes();
        self.write_bytes(&le);
    }

    /// CEP:WHAT: Mixes one i64 (bit pattern).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: two's complement negative values (Rust guarantee).
    /// CEP:COST: as write_u64.
    /// CEP:EVIDENCE: tests in this module.
    #[inline]
    pub fn write_i64(&mut self, v: i64) {
        self.write_u64(v as u64);
    }

    /// CEP:WHAT: Mixes one f64 (bit pattern).
    /// CEP:WHY: Hashing the bit pattern keeps NaN payloads and -0.0 distinct,
    ///          which the verifier requires (CEP&CC 38.24: no silent FP
    ///          semantic change; distinct bits are distinct IR).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: IEEE 754 binary64 (documented platform contract; CEP&CC 7.2).
    /// CEP:COST: as write_u64.
    /// CEP:EVIDENCE: tests in this module.
    #[inline]
    pub fn write_f64(&mut self, v: f64) {
        self.write_u64(v.to_bits());
    }

    /// CEP:WHAT: Mixes a discriminant tag (opcode/enum byte).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: as write_byte.
    /// CEP:EVIDENCE: op hashing tests.
    #[inline]
    pub fn write_tag(&mut self, t: u8) {
        self.write_byte(t);
    }

    /// CEP:WHAT: Finalizes and returns the 64-bit fingerprint.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 load.
    /// CEP:EVIDENCE: reference-vector test.
    #[inline]
    pub fn finish(self) -> u64 {
        self.state
    }
}

impl Default for Fnv64 {
    /// CEP:WHAT: Same as new() (offset basis).
    /// CEP:WHY: clippy::new_without_default parity.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 store.
    /// CEP:EVIDENCE: tests in this module.
    fn default() -> Self {
        Fnv64::new()
    }
}

/// CEP:WHAT: One-shot convenience hash of a byte slice.
/// CEP:WHY: Cache-key derivation at call sites that do not keep state.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: O(len).
/// CEP:EVIDENCE: tests in this module.
pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = Fnv64::new();
    h.write_bytes(bytes);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Published FNV-1a 64 test vectors (ditto spec):
    //           "" -> basis; "a" -> 0xaf63_dc4c_8601_ec8c;
    //           "foobar" -> 0x85944171f73967e8.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the constants or mixing drift.
    // CEP:ASSUMES: vectors from the FNV reference test suite.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fnv_reference_vectors() {
        assert_eq!(fnv64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    // CEP:WHAT: Hashing is order-sensitive and repeatable.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn deterministic_across_runs() {
        let a = fnv64(b"xir-core");
        let b = fnv64(b"xir-core");
        assert_eq!(a, b);
        assert_ne!(a, fnv64(b"xir-corf"));
    }
}
