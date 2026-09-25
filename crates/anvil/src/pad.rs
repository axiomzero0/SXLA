// CEP:FILE: crates/anvil/src/pad.rs
// CEP:WHAT: CachePadded — aligns a hot field to the configured cache-line floor.
// CEP:WHY: Adjacent atomics written by different workers share a cache line and
//          false-share (arch section 2: Chase-Lev deques per core must not
//          contend). Padding isolates each per-core hot field.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: none; layout-only wrapper.
// CEP:ASSUMES: alignment floor comes from config::CACHE_LINE_BYTES (Law 7).
// CEP:COST: one extra cache line per padded field; zero instructions.
// CEP:EVIDENCE: unit test `padding_size_covers_cache_line`.
// CEP:SECURITY: no unsafe code.
// CEP:HPC-DETERMINISM: deterministic; layout only.
//! Cache-line padding wrapper for hot per-core state.

use core::mem::size_of;

/// Pads `T` so that consecutive `CachePadded<T>` values cannot share a cache line.
///
/// CEP:WHAT: Align-to-cache-line wrapper around a single value.
/// CEP:WHY: Prevents false sharing between per-worker atomics (e.g. deque
///          `top`/`bottom`, SPSC head/tail). Alternative `#[repr(align(128))]`
///          rejected: wastes 64B per field on 64B-line targets.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: config::CACHE_LINE_BYTES is the supported-target floor (config.rs).
/// CEP:COST: size grows to a multiple of CACHE_LINE_BYTES; zero runtime instructions.
/// CEP:EVIDENCE: test `padding_size_covers_cache_line`
#[repr(align(64))]
pub struct CachePadded<T> {
    /// The padded value.
    pub value: T,
}

impl<T> CachePadded<T> {
    /// CEP:WHAT: Wraps `value` with cache-line alignment.
    /// CEP:WHY: Uniform construction at init sites without exposing the field.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: zero instructions
    /// CEP:EVIDENCE: module tests
    pub const fn new(value: T) -> CachePadded<T> {
        CachePadded { value }
    }

    /// CEP:WHAT: Returns the total size of one padded value.
    /// CEP:WHY: Used by `assert_config_invariants` to prove the padding floor
    ///          is actually met (Law 3: enforced invariant).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: constant-time size query
    /// CEP:EVIDENCE: test `padding_size_covers_cache_line`
    pub const fn padding_size() -> usize {
        size_of::<CachePadded<T>>()
    }
}

impl<T> core::ops::Deref for CachePadded<T> {
    type Target = T;

    /// CEP:WHAT: Transparent access to the wrapped value.
    /// CEP:WHY: Call sites read `self.top.load(...)` on padded atomics without
    ///          a `.value` hop on every line; zero-cost (repr(transparent)-like
    ///          forwarding through the single field).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: zero instructions
    /// CEP:EVIDENCE: crate-wide compilation
    #[inline]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> core::ops::DerefMut for CachePadded<T> {
    /// CEP:WHAT: Transparent mutable access to the wrapped value.
    /// CEP:WHY: Owner-side mutation (e.g. deque stores) without `.value`.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: single-owner discipline of the inner type.
    /// CEP:COST: zero instructions
    /// CEP:EVIDENCE: crate-wide compilation
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CACHE_LINE_BYTES;

    // CEP:WHAT: Proves padded size is a multiple of CACHE_LINE_BYTES and >= it.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if alignment attribute is removed/regressed.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn padding_size_covers_cache_line() {
        let s = CachePadded::<u64>::padding_size();
        assert!(s >= CACHE_LINE_BYTES);
        assert_eq!(s % CACHE_LINE_BYTES, 0);
    }
}
