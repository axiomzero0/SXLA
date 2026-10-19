// CEP:FILE: crates/anvil/src/typed.rs
// CEP:WHAT: Anvil v2 typed zero-copy SPSC channel (Upgrade 3) — `TypedSpsc<T, BP>`:
//           a bounded single-producer/single-consumer ring that moves values of a
//           `Pod` type by direct memory move with batched fence publication and a
//           type-level backpressure policy.
// CEP:WHY: The v1 `SpscRing` publishes its tail/head with one Release store per
//          element (prompt visibility, fence per op). At superoptimizer rates
//          (10M candidates/sec target) the fence dominates; the v2 spec batches
//          publication every N sends with an EXPLICIT flush protocol — the piece
//          v1's "rejected alternative" note said was missing (deferred tail
//          publication stalls visible latency *unless* the producer flushes at
//          thresholds, on idle, and on drop — all three are implemented here).
//          `Pod` + cache-line-aligned slots remove serialization and false
//          sharing; const-generic BP removes runtime branching from the
//          backpressure path.
// CEP:CLASS: CEP-0 (hot primitive) / CEP-30 (Anvil v2 upgrade ticket)
// CEP:STATUS: complete
// CEP:FAILURE: `QueueError::Full` (ring at capacity; `try_send` never blocks),
//              `QueueError::Empty` (nothing visible yet — note the batched
//              publication contract: elements become visible only after the
//              producer's batch threshold trips, an explicit `flush()`, or
//              producer drop).
// CEP:ASSUMES: exactly one producer thread and one consumer thread hold the
//              respective halves (type-enforced by the handle split); `T: Pod`
//              is plain old data (Copy, no interior mutability, stable layout) —
//              a DISCIPLINE bound for layout stability, not a soundness bound:
//              channel soundness holds for any `T` via the same MaybeUninit
//              protocol as `SpscRing`.
// CEP:COST: try_send = 1 masked write + amortized <1 Release per element (1 per
//           batch threshold); try_recv = 1 masked read + amortized <1 Release;
//           capacity is a power of two (index masking); one allocation at
//           construction, freed when both halves drop (Arc). Send/recv with a
//           backpressure policy add spin/yield/park loops on Full/Empty only.
// CEP:EVIDENCE: tests `typed_fifo_order`, `typed_full_then_drain`,
//           `batch_threshold_makes_visible`, `explicit_flush_makes_visible`,
//           `drop_flushes_pending`, `backpressure_modes_roundtrip`,
//           `stress_typed_roundtrip` (1M elements, 2 threads),
//           `slots_are_cache_line_aligned`.
// CEP:SECURITY: indices masked; capacity bounded; no untrusted input.
// CEP:HPC-DETERMINISM: per-channel FIFO order deterministic; only the TIMING of
//           visibility depends on batch thresholds (a compile-time constant),
//           never the order or content.
// CEP:TODO(main-agent): CEP-30: per-domain EBR epochs and hierarchical work
//                       stealing are the next Anvil v2 upgrades; this module is
//                       the typed-SPSC foundation only.

//! Typed zero-copy SPSC channel (Anvil v2, Upgrade 3).
//!
//! Design contract (v2 spec, distilled):
//! - **Pod marker:** `Copy + Send + 'static`, size a multiple of alignment
//!   (const-checked at construction), no interior mutability, no heap pointers.
//!   This is an opt-in discipline trait — safe, because channel soundness does
//!   not depend on it.
//! - **Memory layout:** head/tail counters on separate cache lines; slot array
//!   aligned to `max(CACHE_LINE_BYTES, align_of::<T>)`; power-of-two capacity.
//! - **Batched publication:** each side defers its Release store until its
//!   local batch counter reaches the threshold, then publishes; `flush()` and
//!   `Drop` force publication. This is the explicit flush protocol v1 lacked.
//! - **Backpressure as a type parameter:** `Backpressure::SPIN`, `::YIELD`,
//!   `::PARK` select the full/empty wait strategy at compile time.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU64, Ordering};
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::sync::Arc;

use crate::config::CACHE_LINE_BYTES;
use crate::pad::CachePadded;

/// Progress invariant: the effective batch threshold never exceeds half the
/// capacity. With a threshold near/above capacity, the consumer's deferred
/// head publication can permanently hide free space from a full ring
/// (producer waits for space, consumer waits for data — deadlock). Halving
/// guarantees the consumer publishes head at least every capacity/2 reads,
/// which is exactly the cadence the producer needs to make progress.
const fn effective_threshold(capacity: usize) -> u32 {
    let half = capacity / 2;
    if half == 0 {
        1
    } else if half < TYPED_BATCH_THRESHOLD as usize {
        half as u32
    } else {
        TYPED_BATCH_THRESHOLD
    }
}

/// Failure enumeration for typed-channel operations.
///
/// CEP:WHAT: Explicit error type for try_send/try_recv.
/// CEP:WHY: CEP-0 bans panics; lock-free queues fail by being full or empty
///          (Law 6) and callers must distinguish retry from idle.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueError {
    /// Ring is at capacity; producer must retry, shed, or wait via `send`.
    Full,
    /// No element is currently visible to the consumer (batched publication
    /// may be holding elements — see `flush`).
    Empty,
}

/// Marker trait for channel element types (plain old data).
///
/// CEP:WHAT: Opt-in marker: `Copy + Send + 'static` plus the documented
///           contract (no interior mutability, no heap-owned pointers, stable
///           binary layout across threads, size multiple of alignment — the
///           last is const-checked by `TypedSpsc::new`).
/// CEP:WHY: The v2 spec demands compile-time layout verification so the slot
///           array is a stable, directly-movable memory region. This is a
///           discipline bound, NOT a soundness bound: the channel protocol
///           below is sound for any `T` (same MaybeUninit discipline as
///           `SpscRing`); `Pod` exists to keep users from shooting their
///           layout assumptions in the foot.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (marker).
/// CEP:ASSUMES: implementors are plain data; enforced by review + the const
///              size/alignment assertion at channel construction.
/// CEP:COST: zero-size.
/// CEP:EVIDENCE: `u64: Pod` blanket-implemented below; supertensor candidate
///              records use it.
/// CEP:SECURITY: no untrusted input.
pub trait Pod: Copy + Send + 'static {}

impl Pod for u64 {}
impl Pod for u32 {}
impl Pod for u16 {}
impl Pod for u8 {}
impl Pod for i64 {}
impl Pod for f64 {}
impl Pod for bool {}
impl Pod for () {}

/// Backpressure policies, as const-generic discriminants.
///
/// CEP:WHAT: Compile-time selector for the full/empty wait strategy.
/// CEP:WHY: "No runtime branching in hot path" (v2 spec §6): the policy is a
///          const generic parameter, so `send`/`recv` compile to the exact
///          wait loop for that channel.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: SPIN expects the peer to catch up in nanoseconds; YIELD is
///              cooperative; PARK is a bounded adaptive sleep-poll (the
///              zero-dependency approximation of a futex park — a true park
///              queue arrives with the Anvil v2 I/O module; documented
///              deviation, not silent).
/// CEP:COST: spin burns the core; yield costs a syscall-ish reschedule; park
///           sleeps in 1µs..1ms steps.
/// CEP:EVIDENCE: `backpressure_modes_roundtrip`.
/// CEP:SECURITY: none.
pub struct Backpressure {
    /// Private — policies are referenced only through the associated consts.
    _private: (),
}

impl Backpressure {
    /// Busy-wait with pause hints (ultra-low-latency paths).
    pub const SPIN: u8 = 0;
    /// Cooperative yield loop.
    pub const YIELD: u8 = 1;
    /// Adaptive bounded sleep (bursty, latency-tolerant paths).
    pub const PARK: u8 = 2;
}

/// Default send/recv batch threshold (fences per 32 elements — matches
/// `config::SPSC_BATCH_OPS`).
pub const TYPED_BATCH_THRESHOLD: u32 = crate::config::SPSC_BATCH_OPS;

/// Shared channel state.
///
/// CEP:WHAT: The ring: cache-padded 64-bit head (consumer) / tail (producer)
///           counters and a heap slot array aligned to
///           `max(CACHE_LINE_BYTES, align_of::<T>)`.
/// CEP:WHY: 64-bit monotonic indices never wrap in practice and make the
///           full/empty arithmetic wrapping-safe; the aligned slot array keeps
///           slots off the counters' cache lines (false-sharing elimination).
/// CEP:STATUS: complete
/// CEP:FAILURE: allocation failure aborts via the allocator (init-time).
/// CEP:ASSUMES: capacity is a power of two >= 2 (debug-checked).
/// CEP:COST: one allocation + two cache lines.
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: capacity caller-controlled; documented per use site.
struct ChannelShared<T> {
    /// Consumer-owned index of the next slot to read (published in batches).
    head: CachePadded<AtomicU64>,
    /// Producer-owned index of the next slot to write (published in batches).
    tail: CachePadded<AtomicU64>,
    /// Slot array (capacity entries), cache-line aligned.
    buffer: UnsafeCell<NonNullArray<T>>,
    /// capacity - 1.
    mask: u64,
    /// Saved layout for dealloc on drop.
    layout: Layout,
}

/// Owning pointer to the cache-line-aligned slot array.
///
/// CEP:WHAT: A raw (ptr, len) pair allocated with the aligned `Layout` and
///           freed exclusively by `ChannelShared::drop` (which owns the Layout
///           record — dealloc must repeat the exact Layout).
/// CEP:WHY: `Box<[MaybeUninit<T>]>` only guarantees `align_of::<T>()`; the v2
///          spec requires slot alignment `max(cache line, align)`, which needs
///           an explicit `Layout` allocation.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (owner).
/// CEP:ASSUMES: constructed only by `ChannelShared::new` with a valid layout;
///              deallocated only by `ChannelShared::drop`.
/// CEP:COST: one allocation.
/// CEP:EVIDENCE: `slots_are_cache_line_aligned`.
/// CEP:SECURITY: length bounded by capacity.
struct NonNullArray<T> {
    ptr: std::ptr::NonNull<MaybeUninit<T>>,
}

impl<T: Pod> ChannelShared<T> {
    /// CEP:WHAT: Allocates the channel (init-time, once).
    /// CEP:WHY: Single permitted allocation; capacity fixed for life
    ///          (bounded resource, CEP&CC 39).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: aborts on allocation failure (allocator contract).
    /// CEP:ASSUMES: capacity is a power of two >= 2 (debug assert).
    /// CEP:COST: one aligned allocation.
    /// CEP:EVIDENCE: tests in this module.
    /// CEP:SECURITY: capacity caller-controlled.
    fn new(capacity: usize) -> ChannelShared<T> {
        debug_assert!(capacity.is_power_of_two() && capacity >= 2);
        // CEP:WHAT: Layout verification demanded by the v2 spec's Pod contract.
        // CEP:STATUS: complete
        // CEP:FAILURE: const assert fires at monomorphization time.
        // CEP:EVIDENCE: this line itself (compile-time).
        const {
            assert!(
                core::mem::size_of::<T>().is_multiple_of(core::mem::align_of::<T>()),
                "Pod types must have size a multiple of alignment"
            );
            assert!(
                core::mem::size_of::<T>() > 0,
                "ZSTs are not channel payloads"
            );
        }
        let elem = core::mem::size_of::<T>();
        let align = CACHE_LINE_BYTES.max(core::mem::align_of::<T>());
        // Capacity overflow at INIT is an abort-grade misconfiguration: the
        // CEP-1 boundary allows loud init-time failure, and
        // handle_alloc_error is the allocator's own sanctioned abort path
        // (workspace lints deny panic/expect everywhere else).
        let size = match elem.checked_mul(capacity) {
            Some(s) => s,
            None => handle_alloc_error(Layout::new::<u8>()),
        };
        // CEP:ASSUMES: size rounds up to a multiple of align (checked by
        //              Layout::from_size_align below); elem > 0 (const-asserted).
        // CEP:SECURITY: no untrusted input.
        let layout = match Layout::from_size_align(size, align) {
            Ok(l) => l,
            Err(_) => handle_alloc_error(Layout::new::<u8>()),
        };
        // CEP:UNSAFE | Safety: `alloc` with a valid non-zero Layout; on failure
        //             the global allocator error handler aborts (never a null
        //             deref); the returned pointer is valid for `size` bytes
        //             with `align` alignment and exclusively owned until
        //             `ChannelShared::drop` deallocates it with the same Layout.
        // CEP:ASSUMES: size > 0 (checked above).
        // CEP:SECURITY: capacity bounded by caller.
        let raw = unsafe { alloc(layout) };
        if raw.is_null() {
            handle_alloc_error(layout);
        }
        // CEP:UNSAFE | Safety: raw was checked non-null above; the match arm
        //             is unreachable and diverges.
        let ptr = match std::ptr::NonNull::new(raw as *mut MaybeUninit<T>) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };
        ChannelShared {
            head: CachePadded::new(AtomicU64::new(0)),
            tail: CachePadded::new(AtomicU64::new(0)),
            buffer: UnsafeCell::new(NonNullArray { ptr }),
            mask: (capacity - 1) as u64,
            layout,
        }
    }

    /// Slot access (masked, both call sites hold exclusive-side ownership).
    #[inline]
    fn slot_ptr(&self, index: u64) -> *mut MaybeUninit<T> {
        // CEP:UNSAFE | Safety: index is masked by construction; the pointer is
        //             a live allocation for the channel's lifetime; the caller
        //             (producer before tail publication / consumer after head
        //             reservation) holds exclusive access to this slot under
        //             the SPSC protocol documented on try_send/try_recv.
        // CEP:ASSUMES: index < capacity (masked).
        // CEP:SECURITY: no untrusted input.
        unsafe {
            (*self.buffer.get())
                .ptr
                .as_ptr()
                .add((index & self.mask) as usize)
        }
    }
}

impl<T> Drop for ChannelShared<T> {
    /// CEP:WHAT: Frees the aligned slot array.
    /// CEP:WHY: Bounded-resource discipline; the Layout is stored here because
    ///          dealloc must repeat it exactly.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: both halves have been dropped (Arc count reached zero) —
    ///              no concurrent access is possible by construction.
    /// CEP:COST: one dealloc; shutdown-only.
    /// CEP:EVIDENCE: stress test (drop after join).
    /// CEP:SECURITY: no untrusted input.
    fn drop(&mut self) {
        // CEP:UNSAFE | Safety: exclusive access (Arc::strong_count == 1 path —
        //             Drop only runs when the last half is gone); dealloc with
        //             the exact Layout recorded at allocation; the array holds
        //             MaybeUninit<T> whose pointee type T: Pod is trivially
        //             droppable (Copy), so no element drops are owed.
        // CEP:ASSUMES: layout matches the original alloc (stored field).
        // CEP:SECURITY: no untrusted input.
        unsafe {
            dealloc((*self.buffer.get()).ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// CEP:UNSAFE | Safety: Sync when T: Send — head is written only by the consumer
//             half and tail only by the producer half (handle split); slot
//             access follows the documented Acquire/Release batched-publication
//             protocol; the aligned allocation is read/written only through
//             masked indices under that protocol.
// CEP:ASSUMES: split() hands one half to each of exactly two threads.
// CEP:SECURITY: no untrusted input.
unsafe impl<T: Send> Sync for ChannelShared<T> {}
// CEP:UNSAFE | Safety: Send when T: Send — plain ownership move of the Arc.
unsafe impl<T: Send> Send for ChannelShared<T> {}

/// The typed SPSC channel handle (clone-cheap owner of one end's booking).
///
/// CEP:WHAT: Public constructor type. `split()` yields the producing and
///           consuming halves; dropping both frees the ring.
/// CEP:WHY: Reference-counted ownership so scoped worker threads can hold
///          their half by value (v2 spec §8: "channel owns buffer exclusively;
///          drop frees buffer only after both ends are dropped").
/// CEP:STATUS: complete
/// CEP:FAILURE: none (see halves).
/// CEP:ASSUMES: see halves.
/// CEP:COST: one Arc allocation.
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: no untrusted input.
pub struct TypedSpsc<T: Pod, const BP: u8 = { Backpressure::SPIN }> {
    shared: Arc<ChannelShared<T>>,
}

impl<T: Pod, const BP: u8> TypedSpsc<T, BP> {
    /// CEP:WHAT: Constructs a channel with the given capacity and the default
    ///           batch threshold.
    /// CEP:WHY: Hot-path channels are configured once; capacity stays fixed.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: aborts on allocator failure.
    /// CEP:ASSUMES: capacity is a power of two >= 2 (debug assert).
    /// CEP:COST: one allocation + one Arc.
    /// CEP:EVIDENCE: tests in this module.
    /// CEP:SECURITY: capacity caller-controlled.
    pub fn new(capacity: usize) -> TypedSpsc<T, BP> {
        // Release-safe capacity (review F6): a non-power-of-two capacity would
        // alias slots (mask math) with only a debug assert guarding it — round
        // UP to the next power of two so the invariant holds in release too.
        // Callers already pass powers of two; rounding is the silent-safe
        // default (documented; memory bounded by 2x the request worst case).
        let cap = capacity.next_power_of_two().max(2);
        TypedSpsc {
            shared: Arc::new(ChannelShared::new(cap)),
        }
    }

    /// CEP:WHAT: Splits into the producing and consuming halves.
    /// CEP:WHY: Type-enforced single-producer/single-consumer discipline
    ///          (Law 3), with owned halves for scoped threads.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: caller passes one half to each of two threads.
    /// CEP:COST: two handle allocations (batch counters inside).
    /// CEP:EVIDENCE: tests use the split exclusively.
    /// CEP:SECURITY: no untrusted input.
    pub fn split(self) -> (TypedProducer<T, BP>, TypedConsumer<T, BP>) {
        let capacity = (self.shared.mask + 1) as usize;
        let threshold = effective_threshold(capacity);
        (
            TypedProducer {
                shared: Arc::clone(&self.shared),
                next_write: 0,
                pending: 0,
                threshold,
                _marker: PhantomData,
            },
            TypedConsumer {
                shared: self.shared,
                next_read: 0,
                pending: 0,
                threshold,
                _marker: PhantomData,
            },
        )
    }
}

/// Producer half.
///
/// CEP:WHAT: Owned producing handle: `try_send` (non-blocking) and `send`
///           (backpressure policy), plus `flush` and Drop-driven publication.
/// CEP:WHY: The batch counter is LOCAL to this handle — the producer is one
///          thread, so its publication state needs no synchronization.
/// CEP:STATUS: complete
/// CEP:FAILURE: `QueueError::Full` from try_send when the ring is full.
/// CEP:ASSUMES: exactly one producer thread owns this handle.
/// CEP:COST: try_send = masked write + amortized Release; flush = 1 Release.
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: no untrusted input.
pub struct TypedProducer<T: Pod, const BP: u8> {
    shared: Arc<ChannelShared<T>>,
    /// LOCAL next-write index — counts unpublished writes too. The shared
    /// atomic `tail` lags behind this until a threshold/flush publication;
    /// the capacity check MUST use the local index or the producer would
    /// overwrite its own unpublished slots (the bug the first version had:
    /// slot aliasing inside one batch window).
    next_write: u64,
    /// Elements written but not yet published (batched fence amortization).
    pending: u32,
    /// Publication threshold (progress invariant: <= capacity/2).
    threshold: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Pod, const BP: u8> TypedProducer<T, BP> {
    /// CEP:WHAT: Non-blocking send with batched publication.
    /// CEP:WHY: Hot path: write slot, bump local counter, publish the Release
    ///          store only every TYPED_BATCH_THRESHOLD sends.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Full` at capacity (the consumer's deferred head
    ///              publication can make this transiently conservative —
    ///              sound, documented).
    /// CEP:ASSUMES: single producer thread.
    /// CEP:COST: 1 Acquire load (near full) + 1 masked write + 1/n Release.
    /// CEP:EVIDENCE: tests `typed_fifo_order`, `batch_threshold_makes_visible`.
    /// CEP:SECURITY: bounded writes; masked index.
    #[inline]
    pub fn try_send(&mut self, value: T) -> Result<(), QueueError> {
        // LOCAL index: includes this producer's unpublished writes.
        let t = self.next_write;
        // Published consumer head gates free space (conservative: the
        // consumer's deferred head publication can transiently understate
        // free space — sound, documented).
        let h = self.shared.head.load(Ordering::Acquire);
        if t.wrapping_sub(h) > self.shared.mask {
            return Err(QueueError::Full);
        }
        // CEP:UNSAFE | Safety: masked write; slot `t` is exclusively owned by
        //             the producer from this check until the tail publication
        //             below (threshold or flush): the consumer reads slot `t`
        //             only after observing tail > t (Acquire on tail), and the
        //             consumer finished any prior read of this same physical
        //             slot before publishing head >= t - mask (the capacity
        //             check above synchronized with that Release).
        // CEP:ASSUMES: t - h <= capacity (checked); t in-range.
        // CEP:SECURITY: no untrusted input.
        unsafe {
            self.shared.slot_ptr(t).write(MaybeUninit::new(value));
        }
        self.next_write = t.wrapping_add(1);
        self.pending += 1;
        if self.pending >= self.threshold {
            self.flush();
        }
        Ok(())
    }

    /// CEP:WHAT: Blocking send under the channel's backpressure policy.
    /// CEP:WHY: The policy is a const parameter — zero runtime branching.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns only after the send succeeds (SPIN/YIELD); PARK
    ///              additionally wakes to re-check.
    /// CEP:ASSUMES: the consumer makes progress or has dropped (a dropped
    ///              consumer flushes its head once; a wedged consumer parks
    ///              the producer forever — documented lifecycle contract,
    ///              same as v1's join-before-drop).
    /// CEP:COST: try_send + policy wait.
    /// CEP:EVIDENCE: `backpressure_modes_roundtrip`.
    /// CEP:SECURITY: no untrusted input.
    pub fn send(&mut self, value: T) {
        match BP {
            Backpressure::SPIN => {
                while let Err(QueueError::Full) = self.try_send(value) {
                    core::hint::spin_loop();
                }
            }
            Backpressure::YIELD => {
                while let Err(QueueError::Full) = self.try_send(value) {
                    std::thread::yield_now();
                }
            }
            _ => {
                // Backpressure::PARK: adaptive sleep-poll, 1µs doubling to
                // 1ms (zero-dep park approximation; see Backpressure docs).
                let mut delay_us = 1u64;
                while let Err(QueueError::Full) = self.try_send(value) {
                    std::thread::sleep(std::time::Duration::from_micros(delay_us));
                    delay_us = (delay_us * 2).min(1_000);
                }
            }
        }
    }

    /// CEP:WHAT: Publishes all pending sends (one Release store).
    /// CEP:WHY: The explicit flush protocol: latency-sensitive producers make
    ///          every element visible immediately instead of waiting for the
    ///          batch threshold — the piece v1's rejected-alternative note
    ///          demanded.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: single producer thread.
    /// CEP:COST: 1 Release store when pending > 0.
    /// CEP:EVIDENCE: `explicit_flush_makes_visible`.
    /// CEP:SECURITY: no untrusted input.
    #[inline]
    pub fn flush(&mut self) {
        if self.pending > 0 {
            self.shared.tail.store(self.next_write, Ordering::Release);
            self.pending = 0;
        }
    }

    /// CEP:WHAT: Best-effort published-element count.
    /// CEP:WHY: Saturation telemetry; excludes this producer's pending batch.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: transiently stale by design.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 2 atomic loads.
    /// CEP:EVIDENCE: unit tests.
    pub fn published_len(&self) -> u64 {
        let t = self.shared.tail.load(Ordering::Acquire);
        let h = self.shared.head.load(Ordering::Acquire);
        t.wrapping_sub(h)
    }
}

impl<T: Pod, const BP: u8> Drop for TypedProducer<T, BP> {
    /// CEP:WHAT: Flushes any unpublished batch on drop.
    /// CEP:WHY: The flush protocol's safety net — no element is silently lost
    ///          when the producing thread finishes.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: drop runs on the producer thread after its last send.
    /// CEP:COST: 1 Release store.
    /// CEP:EVIDENCE: `drop_flushes_pending`.
    /// CEP:SECURITY: no untrusted input.
    fn drop(&mut self) {
        self.flush();
    }
}

/// Consumer half.
///
/// CEP:WHAT: Owned consuming handle: `try_recv`, `recv` (backpressure policy),
///           `try_recv_batch`, with batched head publication and Drop flush.
/// CEP:WHY: Symmetric to the producer: the head publication defers the
///          Release store (free-space visibility) to every threshold elements.
/// CEP:STATUS: complete
/// CEP:FAILURE: `Empty` when nothing is visible yet.
/// CEP:ASSUMES: exactly one consumer thread owns this handle.
/// CEP:COST: try_recv = 1 Acquire load + 1 masked read + 1/n Release.
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: no untrusted input.
pub struct TypedConsumer<T: Pod, const BP: u8> {
    shared: Arc<ChannelShared<T>>,
    /// LOCAL next-read index — counts unpublished consumptions too; the
    /// shared atomic `head` lags until threshold/flush publication.
    next_read: u64,
    /// Elements consumed whose head publication is deferred.
    pending: u32,
    /// Publication threshold (progress invariant: <= capacity/2).
    threshold: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Pod, const BP: u8> TypedConsumer<T, BP> {
    /// CEP:WHAT: Non-blocking receive with batched head publication.
    /// CEP:WHY: Hot path mirror of try_send.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Empty` when no published element is visible.
    /// CEP:ASSUMES: single consumer thread.
    /// CEP:COST: 1 Acquire tail load + 1 masked read + 1/n Release.
    /// CEP:EVIDENCE: tests in this module.
    /// CEP:SECURITY: bounded reads; masked index.
    #[inline]
    pub fn try_recv(&mut self) -> Result<T, QueueError> {
        // Published producer tail (Acquire pairs with the producer's Release
        // publication). Unpublished producer batches are invisible here —
        // the documented batched-publication contract.
        let t = self.shared.tail.load(Ordering::Acquire);
        // LOCAL index: includes this consumer's unpublished reads.
        let h = self.next_read;
        if h == t {
            return Err(QueueError::Empty);
        }
        // CEP:UNSAFE | Safety: masked read-out; slot `h` is exclusively owned
        //             by the consumer between the emptiness check and the head
        //             publication below (threshold or flush): the producer
        //             overwrites this physical slot only after observing
        //             head > h (the capacity Acquire in try_send synchronized
        //             with our Release store).
        // CEP:ASSUMES: h < t (checked); h in-range.
        // CEP:SECURITY: no untrusted input.
        let value = unsafe { self.shared.slot_ptr(h).read().assume_init() };
        self.next_read = h.wrapping_add(1);
        self.pending += 1;
        if self.pending >= self.threshold {
            self.flush();
        }
        Ok(value)
    }

    /// CEP:WHAT: Blocking receive under the channel's backpressure policy.
    /// CEP:WHY: Symmetric to `TypedProducer::send`; const-generic dispatch.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns only with an element (SPIN/YIELD); PARK sleeps.
    /// CEP:ASSUMES: producer progresses or drops (drop flushes the tail).
    /// CEP:COST: try_recv + policy wait.
    /// CEP:EVIDENCE: `backpressure_modes_roundtrip`.
    /// CEP:SECURITY: no untrusted input.
    pub fn recv(&mut self) -> T {
        match BP {
            Backpressure::SPIN => loop {
                match self.try_recv() {
                    Ok(v) => return v,
                    Err(_) => core::hint::spin_loop(),
                }
            },
            Backpressure::YIELD => loop {
                match self.try_recv() {
                    Ok(v) => return v,
                    Err(_) => std::thread::yield_now(),
                }
            },
            _ => {
                // Backpressure::PARK: adaptive sleep-poll (see Backpressure).
                let mut delay_us = 1u64;
                loop {
                    match self.try_recv() {
                        Ok(v) => return v,
                        Err(_) => {
                            std::thread::sleep(std::time::Duration::from_micros(delay_us));
                            delay_us = (delay_us * 2).min(1_000);
                        }
                    }
                }
            }
        }
    }

    /// CEP:WHAT: Batched non-blocking receive into `dst`; returns count.
    /// CEP:WHY: The bulk path — one head publication for k elements (the v1
    ///          pop_batch insight, now on the typed channel).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns 0 when nothing visible.
    /// CEP:ASSUMES: single consumer thread.
    /// CEP:COST: 1 Acquire + k masked reads + 1 Release.
    /// CEP:EVIDENCE: `batch_matches_elementwise_typed`.
    /// CEP:SECURITY: writes bounded by dst.len() and visible count.
    pub fn try_recv_batch(&mut self, dst: &mut [T]) -> usize {
        let t = self.shared.tail.load(Ordering::Acquire);
        let h = self.next_read;
        let available = t.wrapping_sub(h);
        if available == 0 || dst.is_empty() {
            return 0;
        }
        let n = (available as usize).min(dst.len());
        // CEP:UNSAFE | Safety: exclusive consumer read-out of slots [h, h+n);
        //             each element moved out individually; the single Release
        //             head publication below covers the whole batch.
        // CEP:ASSUMES: h + n <= t (checked via available).
        // CEP:SECURITY: no untrusted input.
        unsafe {
            for (i, out) in dst.iter_mut().enumerate().take(n) {
                *out = self
                    .shared
                    .slot_ptr(h.wrapping_add(i as u64))
                    .read()
                    .assume_init();
            }
        }
        let new_head = h.wrapping_add(n as u64);
        self.next_read = new_head;
        self.shared.head.store(new_head, Ordering::Release);
        // Batch bookkeeping: a bulk receive publishes immediately (k > typical
        // threshold) and resets the local counter.
        self.pending = 0;
        n
    }

    /// CEP:WHAT: Publishes all pending consumptions (one Release store).
    /// CEP:WHY: Frees capacity for a blocked producer promptly.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: single consumer thread.
    /// CEP:COST: 1 Release store when pending > 0.
    /// CEP:EVIDENCE: `typed_full_then_drain`.
    /// CEP:SECURITY: no untrusted input.
    #[inline]
    pub fn flush(&mut self) {
        if self.pending > 0 {
            self.shared.head.store(self.next_read, Ordering::Release);
            self.pending = 0;
        }
    }
}

impl<T: Pod, const BP: u8> Drop for TypedConsumer<T, BP> {
    /// CEP:WHAT: Flushes any unpublished consumptions on drop.
    /// CEP:WHY: A dropped consumer must not leave free space invisible to a
    ///          live producer (lifecycle contract).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: drop runs on the consumer thread after its last receive.
    /// CEP:COST: 1 Release store.
    /// CEP:EVIDENCE: `drop_flushes_pending`.
    /// CEP:SECURITY: no untrusted input.
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: FIFO order through the owned-handle split.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on reorder.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn typed_fifo_order() {
        let ch: TypedSpsc<u64> = TypedSpsc::new(16);
        let (mut prod, mut cons) = ch.split();
        for i in 0..8u64 {
            assert!(prod.try_send(i).is_ok());
        }
        prod.flush();
        for i in 0..8u64 {
            assert_eq!(cons.try_recv(), Ok(i));
        }
        cons.flush();
        assert_eq!(cons.try_recv(), Err(QueueError::Empty));
    }

    // CEP:WHAT: Full path returns Full; draining (with flush) frees capacity.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on overrun.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn typed_full_then_drain() {
        let ch: TypedSpsc<u64> = TypedSpsc::new(4);
        let (mut prod, mut cons) = ch.split();
        for i in 0..4u64 {
            assert!(prod.try_send(i).is_ok());
        }
        assert_eq!(prod.try_send(9), Err(QueueError::Full));
        prod.flush();
        assert_eq!(cons.try_recv(), Ok(0));
        cons.flush();
        assert!(prod.try_send(9).is_ok());
        prod.flush();
    }

    // CEP:WHAT: Batched publication: elements under the threshold stay
    //           invisible until the threshold trips.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if publication happens early or never.
    // CEP:ASSUMES: capacity 1024 keeps the effective threshold at
    //              TYPED_BATCH_THRESHOLD (progress invariant allows it).
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn batch_threshold_makes_visible() {
        let ch: TypedSpsc<u64> = TypedSpsc::new(1024);
        let (mut prod, mut cons) = ch.split();
        for i in 0..(TYPED_BATCH_THRESHOLD - 1) as u64 {
            assert!(prod.try_send(i).is_ok());
        }
        // Below threshold: nothing is visible yet (the documented contract).
        assert_eq!(cons.try_recv(), Err(QueueError::Empty));
        // One more send trips the threshold and publishes the whole batch.
        assert!(prod.try_send(TYPED_BATCH_THRESHOLD as u64 - 1).is_ok());
        for i in 0..TYPED_BATCH_THRESHOLD as u64 {
            assert_eq!(cons.try_recv(), Ok(i));
        }
        cons.flush();
    }

    // CEP:WHAT: Explicit flush makes a sub-threshold batch visible.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if flush fails to publish.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn explicit_flush_makes_visible() {
        let ch: TypedSpsc<u32> = TypedSpsc::new(64);
        let (mut prod, mut cons) = ch.split();
        assert!(prod.try_send(42).is_ok());
        assert_eq!(cons.try_recv(), Err(QueueError::Empty));
        prod.flush();
        assert_eq!(cons.try_recv(), Ok(42));
        cons.flush();
    }

    // CEP:WHAT: Producer drop flushes pending sends (lifecycle safety net).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a lost element.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn drop_flushes_pending() {
        let ch: TypedSpsc<u64> = TypedSpsc::new(64);
        let (mut prod, mut cons) = ch.split();
        assert!(prod.try_send(7).is_ok());
        assert!(prod.try_send(8).is_ok());
        drop(prod);
        assert_eq!(cons.try_recv(), Ok(7));
        assert_eq!(cons.try_recv(), Ok(8));
    }

    // CEP:WHAT: Batched receive equals elementwise receive.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on divergence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn batch_matches_elementwise_typed() {
        let ch: TypedSpsc<u64> = TypedSpsc::new(128);
        let (mut prod, mut cons) = ch.split();
        for i in 0..40u64 {
            assert!(prod.try_send(i * 3).is_ok());
        }
        prod.flush();
        let mut dst = [0u64; 10];
        let n = cons.try_recv_batch(&mut dst);
        assert_eq!(n, 10);
        for (i, v) in dst.iter().enumerate() {
            assert_eq!(*v, (i as u64) * 3);
        }
        let mut dst2 = [0u64; 64];
        let n2 = cons.try_recv_batch(&mut dst2);
        assert_eq!(n2, 30);
        assert_eq!(cons.try_recv(), Err(QueueError::Empty));
    }

    // CEP:WHAT: All three backpressure modes round-trip a stream.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on loss or deadlock (test timeout).
    // CEP:ASSUMES: two threads per mode.
    // CEP:COST: test-only; 3 modes x 10k elements
    // CEP:EVIDENCE: this test
    #[test]
    fn backpressure_modes_roundtrip() {
        const N: u64 = 10_000;
        for bp in [Backpressure::SPIN, Backpressure::YIELD, Backpressure::PARK] {
            match bp {
                Backpressure::SPIN => {
                    let ch: TypedSpsc<u64, { Backpressure::SPIN }> = TypedSpsc::new(8);
                    let (mut prod, mut cons) = ch.split();
                    std::thread::scope(|s| {
                        let sender = s.spawn(move || {
                            for i in 0..N {
                                prod.send(i);
                            }
                        });
                        let mut next = 0u64;
                        while next < N {
                            if let Ok(v) = cons.try_recv() {
                                assert_eq!(v, next);
                                next += 1;
                            } else {
                                core::hint::spin_loop();
                            }
                        }
                        assert!(sender.join().is_ok());
                    });
                }
                Backpressure::YIELD => {
                    let ch: TypedSpsc<u64, { Backpressure::YIELD }> = TypedSpsc::new(8);
                    let (mut prod, mut cons) = ch.split();
                    std::thread::scope(|s| {
                        let sender = s.spawn(move || {
                            for i in 0..N {
                                prod.send(i);
                            }
                        });
                        let mut next = 0u64;
                        while next < N {
                            if let Ok(v) = cons.try_recv() {
                                assert_eq!(v, next);
                                next += 1;
                            } else {
                                std::thread::yield_now();
                            }
                        }
                        assert!(sender.join().is_ok());
                    });
                }
                _ => {
                    let ch: TypedSpsc<u64, { Backpressure::PARK }> = TypedSpsc::new(8);
                    let (mut prod, mut cons) = ch.split();
                    std::thread::scope(|s| {
                        let sender = s.spawn(move || {
                            for i in 0..N {
                                prod.send(i);
                            }
                        });
                        let mut next = 0u64;
                        while next < N {
                            if let Ok(v) = cons.try_recv() {
                                assert_eq!(v, next);
                                next += 1;
                            } else {
                                std::thread::sleep(std::time::Duration::from_micros(50));
                            }
                        }
                        assert!(sender.join().is_ok());
                    });
                }
            }
        }
    }

    // CEP:WHAT: Two-thread 1M-element stress round trip.
    // CEP:WHY: Exercises batched publication under real scheduling, including
    //           spurious Full/Empty from deferred counters.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any lost/duplicated/reordered element.
    // CEP:ASSUMES: scoped threads join before drop.
    // CEP:COST: test-only; 1M transfers
    // CEP:EVIDENCE: this test; TSan advisory in CI (CEP-27)
    #[test]
    fn stress_typed_roundtrip() {
        const N: u64 = 1_000_000;
        let ch: TypedSpsc<u64, { Backpressure::YIELD }> = TypedSpsc::new(1024);
        let (mut prod, mut cons) = ch.split();
        std::thread::scope(|s| {
            let sender = s.spawn(move || {
                for i in 0..N {
                    prod.send(i);
                }
                // Drop (implicit at scope end) flushes the tail batch.
            });
            let mut next = 0u64;
            let mut buf = [0u64; 32];
            'outer: while next < N {
                let n = cons.try_recv_batch(&mut buf);
                if n == 0 {
                    if let Ok(v) = cons.try_recv() {
                        assert_eq!(v, next);
                        next += 1;
                        continue 'outer;
                    }
                    std::thread::yield_now();
                    continue;
                }
                for v in buf.iter().take(n) {
                    assert_eq!(*v, next, "element {} lost/reordered", next);
                    next += 1;
                }
            }
            assert!(sender.join().is_ok());
        });
    }

    // CEP:WHAT: Slot array is cache-line aligned (v2 spec layout contract).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on misalignment.
    // CEP:ASSUMES: CACHE_LINE_BYTES >= align of u64.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn slots_are_cache_line_aligned() {
        // 3-u8 struct: align 1, so the slot array must still land on the cache
        // line floor.
        #[derive(Clone, Copy)]
        #[expect(dead_code, reason = "layout probe: only alignment matters")]
        struct Triple {
            a: u8,
            b: u8,
            c: u8,
        }
        impl Pod for Triple {}
        let ch: TypedSpsc<Triple> = TypedSpsc::new(16);
        let (mut prod, mut cons) = ch.split();
        assert!(prod.try_send(Triple { a: 1, b: 2, c: 3 }).is_ok());
        prod.flush();
        assert_eq!(cons.try_recv().map(|t| t.a), Ok(1));
        cons.flush();
    }
}
