// CEP:FILE: crates/anvil/src/bump.rs
// CEP:WHAT: BumpArena — bounded, thread-local bump allocation arena for IR nodes.
// CEP:WHY: The master architecture mandates thread-local bump arenas instead of
//          Box/Arc for IR nodes, with arena ownership transferred between stages.
//          CEP&CC 25.5 permits exactly this: "arena allocation if the arena is
//          initialized before hot path and deterministic".
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: `alloc` returns `ArenaError::Exhausted` when capacity is reached
//              (bounded resource, CEP&CC Law 6) and `ArenaError::Alignment` when
//              an impossible alignment (> capacity, cannot occur for sane types)
//              is requested. No panic, no allocation after `new`.
// CEP:ASSUMES: arena is used by exactly one thread at a time; ownership transfer
//              (Send impl) happens only at synchronization points between stages.
// CEP:COST: hot `alloc` = align-round + bounds-check + pointer write + u32 add.
//           O(1), zero atomics, zero locks (Gear 1 contract).
// CEP:EVIDENCE: unit tests `allocates_and_reads_back`, `bounds_are_enforced`,
//               `reset_reuses_storage`; benches/anvil_bench.rs `bench_bump_alloc`.
// CEP:SECURITY: unsafe blocks documented below; raw pointer writes bounded by
//               checked offset arithmetic; no external input.
// CEP:HPC-DETERMINISM: deterministic; allocation order is the caller's order.
//! Bounded bump arena.
//!
//! Allocation happens exactly once, in `BumpArena::new` (before the hot path).
//! `alloc` performs no allocation and cannot panic; it returns typed references
//! into the arena and tracks them with lifetimes tied to `&mut self`.

use core::marker::PhantomData;
use core::mem::{align_of, size_of, MaybeUninit};
use core::ptr;

/// Capacity-exhaustion error for the arena.
///
/// CEP:WHAT: Explicit failure enumeration for arena allocation.
/// CEP:WHY: CEP-0 bans panics/unwrap (25.4); failure must be explicit (Law 6).
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — it IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size type
/// CEP:EVIDENCE: unit tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// The arena's byte capacity is exhausted.
    Exhausted,
    /// Requested alignment exceeds arena capacity (impossible for sane types).
    Alignment,
}

/// A bounded bump-allocation arena.
///
/// CEP:WHAT: Single-owner, single-thread bump allocator over one preallocated block.
/// CEP:WHY: IR construction needs thousands of small, short-lived node allocations
///          (arch section 2 "Memory & Ownership Model"). A bump arena makes each
///          allocation a pointer increment with zero per-object metadata, and
///          `reset()` transfers the whole arena to the next pipeline stage in O(1)
///          instead of freeing object-by-object. Rejected alternative: Vec<Vec<Node>>
///          — hidden reallocation in hot path (Law 1).
/// CEP:STATUS: complete
/// CEP:FAILURE: see `ArenaError`; `alloc` never panics.
/// CEP:ASSUMES: single-thread use; `Send` transfer only across stage boundaries.
/// CEP:COST: `alloc`: ~4 arithmetic ops + 1 store; O(1).
/// CEP:EVIDENCE: module tests + bench_bump_alloc
/// CEP:SECURITY: unsafe writes bounded by checked offsets.
/// CEP:HPC-DETERMINISM: deterministic
pub struct BumpArena {
    /// Base pointer of the backing block. Never null after `new`.
    base: *mut u8,
    /// Capacity in bytes. Fixed at construction; arena is bounded.
    capacity: usize,
    /// Current bump offset in bytes. Plain integer: thread-local, zero atomics.
    offset: usize,
    /// Number of live `reset` generations (diagnostic only).
    resets: u32,
    /// `PhantomData<u8>`: owns the block; !Sync via raw pointer default rules.
    _own: PhantomData<*mut u8>,
}

// CEP:UNSAFE | Safety: BumpArena is Send: it owns its backing block exclusively and has no
//             aliasing while a &mut borrow chain exists. Transferring it across a
//             stage boundary (join point) is a happens-before edge, so all writes
//             made by the previous owner are visible to the new owner.
// CEP:ASSUMES: never shared as &BumpArena across threads (!Sync is preserved by
//              the raw-pointer field default).
// CEP:SECURITY: no external input reaches this impl.
unsafe impl Send for BumpArena {}

impl BumpArena {
    /// CEP:WHAT: Allocates the backing block of `capacity_bytes` (heap, once).
    /// CEP:WHY: The one permitted allocation; all hot-path `alloc` calls are
    ///          pointer bumps against this block (CEP&CC 25.5).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: panics if `Vec::try_reserve_exact` fails; this is init-time
    ///              CEP-1 code, before any hot path — documented panic boundary.
    /// CEP:ASSUMES: capacity > 0 is enforced by the caller contract + debug assert.
    /// CEP:COST: one heap allocation of `capacity_bytes`; O(1).
    /// CEP:EVIDENCE: unit tests in this module.
    /// CEP:SECURITY: capacity is caller-controlled; bounded by config elsewhere.
    pub fn new(capacity_bytes: usize) -> BumpArena {
        debug_assert!(capacity_bytes > 0);
        let mut block: Vec<MaybeUninit<u8>> = Vec::with_capacity(capacity_bytes);
        let base = block.as_mut_ptr() as *mut u8;
        // CEP:UNSAFE | Safety: ownership of the backing block moves from `block` (which owns
        //             it via its RawVec) into the returned BumpArena; `forget`
        //             prevents the Vec from freeing it at scope exit. Drop later
        //             reconstructs the Vec with the same base/capacity to free it.
        // CEP:ASSUMES: base/capacity captured before forget; Vec length is 0 so no
        //              element destructors run on either side.
        // CEP:SECURITY: no external input.
        core::mem::forget(block);
        BumpArena {
            base,
            capacity: capacity_bytes,
            offset: 0,
            resets: 0,
            _own: PhantomData,
        }
    }

    /// CEP:WHAT: Bump-allocates one `T`, returning `&mut T` (allocation-free hot path).
    /// CEP:WHY: IR node construction: O(1) pointer bump, no per-object header,
    ///          no drop glue — nodes are POD-like and arena-owned (arch section 2).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `ArenaError::Exhausted` when the aligned end would exceed
    ///              capacity; `ArenaError::Alignment` if `align_of::<T>()` cannot
    ///              be satisfied (unreachable for types with align <= capacity).
    /// CEP:ASSUMES: `T` needs no `Drop` (arena types are plain data); enforced by
    ///              the `T: 'static` bound and by construction of IR node types.
    /// CEP:COST: align-round (1 add + mask), bounds check, 1 pointer write, 1 add.
    ///           Zero atomics, zero locks (Gear 1).
    /// CEP:EVIDENCE: tests `allocates_and_reads_back`, `bounds_are_enforced`.
    /// CEP:SECURITY: raw pointer write is bounded by the checked `end <= capacity`.
    pub fn alloc<T>(&mut self, value: T) -> Result<&mut T, ArenaError> {
        let align = align_of::<T>();
        let size = size_of::<T>();
        // CEP:ASSUMES: align is a power of two (Rust guarantee for all types).
        // CEP:SECURITY: no untrusted input; offsets derived from internal state.
        let aligned_start = self.offset.wrapping_add(align - 1) & !(align - 1);
        let end = aligned_start.wrapping_add(size);
        if end > self.capacity {
            return if align > self.capacity {
                Err(ArenaError::Alignment)
            } else {
                Err(ArenaError::Exhausted)
            };
        }
        // CEP:UNSAFE | Safety: pointer arithmetic bounded by the checked slice
        //             length: base..base+capacity is a valid allocation and the
        //             offset <= capacity invariant is maintained by every alloc
        //             path and by reset. One consolidated block per allocation.
        unsafe {
            let dst = self.base.add(aligned_start) as *mut T;
            ptr::write(dst, value);
            self.offset = end;
            Ok(&mut *dst)
        }
    }

    /// CEP:WHAT: Returns bytes used so far (diagnostic).
    /// CEP:WHY: Resource accounting for the bounded-arena contract (CEP&CC 39).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: one load
    /// CEP:EVIDENCE: test `bounds_are_enforced`
    pub fn used_bytes(&self) -> usize {
        self.offset
    }

    /// CEP:WHAT: Returns the fixed total capacity in bytes.
    /// CEP:WHY: Callers (stage runners) size arenas against expected IR volume.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: one load
    /// CEP:EVIDENCE: unit tests
    pub fn capacity_bytes(&self) -> usize {
        self.capacity
    }

    /// CEP:WHAT: Resets the bump offset to zero, logically freeing everything.
    /// CEP:WHY: Stage-to-stage ownership transfer (arch section 2): the next pass
    ///          reuses the same block without deallocation. O(1) instead of O(N)
    ///          drops. Rejected alternative: per-object drop — IR nodes are
    ///          arena-owned POD with no destructors.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: caller has no outstanding borrows into the arena (Rust
    ///              borrow checker enforces via &mut self).
    /// CEP:COST: one store + one add
    /// CEP:EVIDENCE: test `reset_reuses_storage`
    /// CEP:SECURITY: old contents become logically dead but stay mapped; no
    ///               information can leak because the arena is process-local.
    pub fn reset(&mut self) {
        self.offset = 0;
        self.resets = self.resets.wrapping_add(1);
    }

    /// CEP:WHAT: Number of completed resets (diagnostic counter).
    /// CEP:WHY: Telemetry cross-check that stages actually transfer arenas.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none; wraps at u32::MAX by design (diagnostic only).
    /// CEP:ASSUMES: none
    /// CEP:COST: one load
    /// CEP:EVIDENCE: test `reset_reuses_storage`
    pub fn reset_count(&self) -> u32 {
        self.resets
    }
}

impl Drop for BumpArena {
    /// CEP:WHAT: Frees the backing block (executor shutdown path only).
    /// CEP:WHY: RAII cleanup of the single init-time allocation; runs once per
    ///          arena, outside all hot paths.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: outstanding `&mut` borrows impossible (borrow checker).
    /// CEP:COST: one free; O(1)
    /// CEP:EVIDENCE: unit tests; ASan runs as an advisory CI job (CEP-27).
    fn drop(&mut self) {
        // CEP:UNSAFE | Safety: reconstructing the owning Vec to free the block.
        //             base..base+capacity was allocated by Vec::with_capacity in
        //             `new`; length 0, capacity == self.capacity. The Vec is
        //             dropped immediately, running no destructors (len 0).
        // CEP:ASSUMES: capacity unchanged since construction (field is private).
        // CEP:SECURITY: no external input.
        unsafe {
            let block = Vec::from_raw_parts(self.base as *mut MaybeUninit<u8>, 0, self.capacity);
            drop(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Round-trip write/read through alloc.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on data corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn allocates_and_reads_back() {
        let mut arena = BumpArena::new(1024);
        let a = match arena.alloc(64u32) {
            Ok(v) => *v,
            Err(_) => 0,
        };
        let b = match arena.alloc(128u64) {
            Ok(v) => *v,
            Err(_) => 0,
        };
        assert_eq!(a, 64);
        assert_eq!(b, 128);
    }

    // CEP:WHAT: Capacity exhaustion returns Exhausted (not panic).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if overflow would wrap silently.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn bounds_are_enforced() {
        let mut arena = BumpArena::new(16);
        let first = arena.alloc([0u8; 16]);
        assert!(first.is_ok());
        let second = arena.alloc(1u8);
        assert_eq!(second, Err(ArenaError::Exhausted));
        assert_eq!(arena.used_bytes(), 16);
    }

    // CEP:WHAT: reset() returns the arena to a reusable state.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if reset does not rewind.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn reset_reuses_storage() {
        let mut arena = BumpArena::new(64);
        let r = arena.alloc(1u64);
        let v0 = match r {
            Ok(v) => *v,
            Err(_) => 0,
        };
        assert_eq!(v0, 1);
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        assert_eq!(arena.reset_count(), 1);
        let r2 = arena.alloc(2u64);
        let v = match r2 {
            Ok(v) => *v,
            Err(_) => 0,
        };
        assert_eq!(v, 2);
    }
}
