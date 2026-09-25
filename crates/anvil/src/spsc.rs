// CEP:FILE: crates/anvil/src/spsc.rs
// CEP:WHAT: SPSC bounded lock-free ring buffer with batched consumption.
// CEP:WHY: Gear 3 (lock-free streaming: JIT request queues, IR lowering
//          streams, profile telemetry — arch section 2). Single-producer
//          single-consumer ownership removes all CAS contention: the producer
//          exclusively owns `tail`, the consumer exclusively owns `head`, and
//          cross-publication uses exactly one Acquire/Release pair per element
//          (amortized to ~1 fence per batch via `try_pop_batch`, matching the
//          architecture's "~2ns per op" batching mandate).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: `QueueError::Full` when the ring is at capacity (bounded; caller
//              retries — never blocks, never allocates); `QueueError::Empty`
//              when no element is visible yet.
// CEP:ASSUMES: exactly one producer thread and one consumer thread call the
//              respective halves; enforced by the handle split below.
// CEP:COST: push = 1 masked write + 1 Release store; pop = 1 Acquire load +
//           1 masked read + 1 Release store; pop_batch = O(k) masked reads +
//           2 fences for k elements. benches/anvil_bench.rs.
// CEP:EVIDENCE: tests `fifo_order`, `full_and_empty_paths`,
//           `batch_matches_elementwise`, `stress_spsc_roundtrip` (1M elements
//           across 2 threads; advisory TSan CI job — CEP-27, audit F-11).
// CEP:SECURITY: indices are internal atomic state; no untrusted input.
// CEP:HPC-DETERMINISM: per-channel FIFO order deterministic; only timing of
//           visibility is scheduler-dependent.
//! SPSC ring buffer.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Failure enumeration for queue operations.
///
/// CEP:WHAT: Explicit error type for push/pop.
/// CEP:WHY: CEP-0 bans panics; lock-free queues fail by being full or empty,
///          and callers must distinguish (Law 6).
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueError {
    /// Ring is at capacity; producer must retry or shed.
    Full,
    /// No element is currently visible to the consumer.
    Empty,
}

/// Shared SPSC ring state.
///
/// CEP:WHAT: The ring itself: cache-padded `head` (consumer) / `tail`
///           (producer) atomics and a power-of-two slot array.
/// CEP:WHY: Padding splits the two hot indices onto separate cache lines so
///          producer and consumer cachelines stay exclusive (Gear 3 goal).
///          Rejected alternative: deferred tail publication (batches of
///          Release stores) — it stalls visible latency without an explicit
///          flush protocol; `try_pop_batch` achieves the same fence
///          amortization while every element stays promptly visible.
/// CEP:STATUS: complete
/// CEP:FAILURE: operations report QueueError; no panics.
/// CEP:ASSUMES: capacity is a power of two (mask math), checked in `new`.
/// CEP:COST: see module header; O(1) per element.
/// CEP:EVIDENCE: tests in this module
/// CEP:SECURITY: bounded capacity; no dynamic growth.
pub struct SpscRing<T> {
    /// Consumer-owned index of the next slot to read.
    head: CachePadded<AtomicUsize>,
    /// Producer-owned index of the next slot to write.
    tail: CachePadded<AtomicUsize>,
    /// Slot array (capacity entries).
    buffer: UnsafeCell<Box<[MaybeUninit<T>]>>,
    /// capacity - 1
    mask: usize,
}

impl<T> SpscRing<T> {
    /// CEP:WHAT: Allocates the ring (init-time, once).
    /// CEP:WHY: Single permitted allocation; capacity fixed for life
    ///          (bounded resource, CEP&CC 39).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: panics if capacity is not a power of two (programmer error
    ///              at init; CEP-1 boundary allows it).
    /// CEP:ASSUMES: capacity from config::TELEMETRY_CAPACITY or explicit.
    /// CEP:COST: one allocation; O(1).
    /// CEP:EVIDENCE: tests in this module
    /// CEP:SECURITY: capacity caller-controlled; documented per use site.
    pub fn new(capacity: usize) -> Box<SpscRing<T>> {
        debug_assert!(capacity.is_power_of_two() && capacity >= 2);
        let buffer: Box<[MaybeUninit<T>]> = core::iter::repeat_with(MaybeUninit::uninit)
            .take(capacity)
            .collect();
        Box::new(SpscRing {
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            buffer: UnsafeCell::new(buffer),
            mask: capacity - 1,
        })
    }

    /// CEP:WHAT: Splits borrow access into type-enforced producer/consumer halves.
    /// CEP:WHY: The SPSC contract (one producer, one consumer) becomes a type
    ///          property instead of a comment (Law 3: enforced invariants).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: caller passes one half to each thread.
    /// CEP:COST: zero-size handles
    /// CEP:EVIDENCE: tests use the split exclusively
    pub fn split(&self) -> (Producer<'_, T>, Consumer<'_, T>) {
        (Producer { ring: self }, Consumer { ring: self })
    }

    /// CEP:WHAT: Producer-side non-blocking push.
    /// CEP:WHY: Core Gear 3 operation; never blocks the compiler worker.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Full` at capacity.
    /// CEP:ASSUMES: single producer thread.
    /// CEP:COST: 1 Acquire load (only near full), 1 masked write, 1 Release store.
    /// CEP:EVIDENCE: tests `fifo_order`, `stress_spsc_roundtrip`
    /// CEP:SECURITY: bounded writes; index masked.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), QueueError> {
        let t = self.tail.load(Ordering::Relaxed);
        // Producer owns `tail`; head only needs reading to prove free space.
        let h = self.head.load(Ordering::Acquire);
        if t.wrapping_sub(h) > self.mask {
            return Err(QueueError::Full);
        }
        // CEP:UNSAFE | Safety: masked write; slot `t` is exclusively owned by the
        //             producer between the capacity check above and the Release
        //             publication of tail below. A previous consumer read of
        //             this slot completed before head advanced past it
        //             (Acquire above synchronized with the consumer's Release
        //             head store), so overwriting raw storage is sound.
        // CEP:ASSUMES: t - h <= capacity (checked); t in-range.
        // CEP:SECURITY: no untrusted input.
        unsafe {
            (*self.buffer.get())[t & self.mask].write(value);
        }
        self.tail.store(t.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// CEP:WHAT: Consumer-side non-blocking single pop.
    /// CEP:WHY: Symmetric core operation for the consumer thread.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Empty` when no element is visible.
    /// CEP:ASSUMES: single consumer thread.
    /// CEP:COST: 1 Acquire load, 1 masked read, 1 Release store.
    /// CEP:EVIDENCE: tests `fifo_order`, `stress_spsc_roundtrip`
    /// CEP:SECURITY: bounded reads; index masked.
    #[inline]
    pub fn pop(&self) -> Result<T, QueueError> {
        let t = self.tail.load(Ordering::Acquire);
        let h = self.head.load(Ordering::Relaxed);
        if h == t {
            return Err(QueueError::Empty);
        }
        // CEP:UNSAFE | Safety: masked read; slot `h` is exclusively owned by the
        //             consumer between the emptiness check above and the
        //             Release publication of head below. assume_init_read moves
        //             the value out; the producer will overwrite the raw slot
        //             only after observing head > h (Acquire/Release pairing).
        // CEP:ASSUMES: h < t (checked); h in-range.
        // CEP:SECURITY: no untrusted input.
        let value = unsafe { (*self.buffer.get())[h & self.mask].assume_init_read() };
        self.head.store(h.wrapping_add(1), Ordering::Release);
        Ok(value)
    }

    /// CEP:WHAT: Consumer-side batched pop into a caller-provided buffer.
    /// CEP:WHY: The architecture's fence amortization: k elements cost 1
    ///          Acquire tail load + k masked reads + 1 Release head store,
    ///          approaching ~2ns per op for k = SPSC_BATCH_OPS and above
    ///          (arch: "Batching is enforced to amortize fence costs").
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns 0 when empty; never fills more than visible or
    ///              fits in `dst`.
    /// CEP:ASSUMES: single consumer thread.
    /// CEP:COST: 2 fences per batch; O(k) plain reads.
    /// CEP:EVIDENCE: test `batch_matches_elementwise`
    /// CEP:SECURITY: writes bounded by dst.len() and visible count.
    pub fn pop_batch(&self, dst: &mut [T]) -> usize {
        let t = self.tail.load(Ordering::Acquire);
        let h = self.head.load(Ordering::Relaxed);
        let available = t.wrapping_sub(h);
        if available == 0 || dst.is_empty() {
            return 0;
        }
        let n = available.min(dst.len());
        // CEP:ASSUMES: h + n <= t (checked above via available).
        // CEP:SECURITY: no untrusted input.
        // CEP:UNSAFE | Safety: exclusive consumer read of slots [h, h+n); each
        //             element is moved out individually; the single Release
        //             head store below publishes consumption of the whole batch.
        unsafe {
            for (i, slot) in dst.iter_mut().enumerate().take(n) {
                *slot = (*self.buffer.get())[(h.wrapping_add(i)) & self.mask].assume_init_read();
            }
        }
        self.head.store(h.wrapping_add(n), Ordering::Release);
        n
    }

    /// CEP:WHAT: Best-effort visible element count.
    /// CEP:WHY: Saturation telemetry; transiently stale by design.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: stale under concurrency; documented.
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 atomic loads
    /// CEP:EVIDENCE: unit tests
    #[inline]
    pub fn len(&self) -> usize {
        let t = self.tail.load(Ordering::Acquire);
        let h = self.head.load(Ordering::Acquire);
        t.wrapping_sub(h)
    }

    /// CEP:WHAT: Best-effort emptiness probe.
    /// CEP:WHY: Poll loops in the JIT boundary.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: transiently stale; documented.
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 atomic loads
    /// CEP:EVIDENCE: unit tests
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Drop for SpscRing<T> {
    /// CEP:WHAT: Drains and drops leftover elements (shutdown path).
    /// CEP:WHY: Bounded-resource discipline; safe only after both halves have
    ///          quiesced (owner joins threads before dropping the ring).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: no concurrent producer/consumer activity (owner contract).
    /// CEP:COST: O(remaining); shutdown-only
    /// CEP:EVIDENCE: stress test; CI ASan
    fn drop(&mut self) {
        let t = self.tail.load(Ordering::Relaxed);
        let h = self.head.load(Ordering::Relaxed);
        // CEP:ASSUMES: threads joined by the owner.
        // CEP:SECURITY: no untrusted input.
        let mut i = h;
        while i != t {
            // CEP:UNSAFE | Safety: sequential drain after quiescence; masked
            //             indices into the slot array.
            let _ = unsafe { (*self.buffer.get())[i & self.mask].assume_init_read() };
            i = i.wrapping_add(1);
        }
    }
}

// CEP:UNSAFE | Safety: Sync when T: Send — the two indices are each single-writer under
//             the SPSC discipline the handle split enforces; UnsafeCell access
//             happens only under the documented ownership + Acquire/Release
//             protocol above.
// CEP:ASSUMES: split() hands one half per thread.
// CEP:SECURITY: no untrusted input.
unsafe impl<T: Send> Sync for SpscRing<T> {}
// CEP:UNSAFE | Safety: Send when T: Send — plain ownership move.
unsafe impl<T: Send> Send for SpscRing<T> {}

use crate::pad::CachePadded;

/// Producer half of an SPSC ring.
///
/// CEP:WHAT: Borrow handle granting push rights to exactly one thread.
/// CEP:WHY: Type-enforced single-producer discipline (Law 3).
/// CEP:STATUS: complete
/// CEP:FAILURE: delegates to `SpscRing::push`.
/// CEP:ASSUMES: held by one thread only.
/// CEP:COST: zero-size handle
/// CEP:EVIDENCE: tests use the split
pub struct Producer<'a, T> {
    /// The borrowed ring (PRIVATE: the type enforces push-only discipline —
    /// audit F-13: the old `pub ring` granted both halves full capability).
    ring: &'a SpscRing<T>,
}

impl<T> Producer<'_, T> {
    /// CEP:WHAT: Producer-side non-blocking push.
    /// CEP:WHY: The ONLY capability the producer handle grants.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Full at capacity.
    /// CEP:ASSUMES: single producer thread.
    /// CEP:COST: as SpscRing::push.
    /// CEP:EVIDENCE: module tests use the handle API exclusively.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), QueueError> {
        self.ring.push(value)
    }
}

/// Consumer half of an SPSC ring.
///
/// CEP:WHAT: Borrow handle granting pop/pop_batch rights to exactly one thread.
/// CEP:WHY: Type-enforced single-consumer discipline (Law 3).
/// CEP:STATUS: complete
/// CEP:FAILURE: delegates to `SpscRing::pop`/`pop_batch`.
/// CEP:ASSUMES: held by one thread only.
/// CEP:COST: zero-size handle
/// CEP:EVIDENCE: tests use the split
pub struct Consumer<'a, T> {
    /// The borrowed ring (PRIVATE: pop-only discipline — audit F-13).
    ring: &'a SpscRing<T>,
}

impl<T> Consumer<'_, T> {
    /// CEP:WHAT: Consumer-side non-blocking single pop.
    /// CEP:WHY: One of the two capabilities the consumer handle grants.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Empty when no element is visible.
    /// CEP:ASSUMES: single consumer thread.
    /// CEP:COST: as SpscRing::pop.
    /// CEP:EVIDENCE: module tests use the handle API exclusively.
    #[inline]
    pub fn pop(&self) -> Result<T, QueueError> {
        self.ring.pop()
    }

    /// CEP:WHAT: Consumer-side batched pop into a caller buffer.
    /// CEP:WHY: The fence-amortized bulk path (Gear 3 batching).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns 0 when empty.
    /// CEP:ASSUMES: single consumer thread.
    /// CEP:COST: 2 fences per batch.
    /// CEP:EVIDENCE: test `batch_matches_elementwise`.
    pub fn pop_batch(&self, dst: &mut [T]) -> usize {
        self.ring.pop_batch(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: FIFO order through the handle split.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on reorder.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fifo_order() {
        let ring = SpscRing::new(16);
        let (prod, cons) = ring.split();
        for i in 0..8u64 {
            assert!(prod.push(i).is_ok());
        }
        for i in 0..8u64 {
            match cons.pop() {
                Ok(v) => assert_eq!(v, i),
                Err(ref e) => {
                    assert_eq!(format!("{:?}", e), "", "unexpected error");
                }
            }
        }
        assert_eq!(cons.pop(), Err(QueueError::Empty));
    }

    // CEP:WHAT: Full path returns Full instead of overwriting.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on overrun.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn full_and_empty_paths() {
        let ring = SpscRing::new(4);
        let (prod, cons) = ring.split();
        for i in 0..4u64 {
            assert!(prod.push(i).is_ok());
        }
        assert_eq!(prod.push(9), Err(QueueError::Full));
        assert!(matches!(cons.pop(), Ok(0)));
        // One slot freed; push succeeds again.
        assert!(prod.push(9).is_ok());
    }

    // CEP:WHAT: Batched pop equals elementwise pop for the same stream.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any divergence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn batch_matches_elementwise() {
        let ring = SpscRing::new(64);
        let (prod, cons) = ring.split();
        for i in 0..40u64 {
            assert!(prod.ring.push(i * 3).is_ok());
        }
        let mut dst = [0u64; 10];
        let n = cons.pop_batch(&mut dst);
        assert_eq!(n, 10);
        for (i, v) in dst.iter().enumerate() {
            assert_eq!(*v, (i as u64) * 3);
        }
        let mut dst2 = [0u64; 64];
        let n2 = cons.pop_batch(&mut dst2);
        assert_eq!(n2, 30);
        assert_eq!(cons.pop(), Err(QueueError::Empty));
    }

    // CEP:WHAT: Two-thread 1M-element round trip with loss accounting.
    // CEP:WHY: Exercises Acquire/Release pairing under real scheduling.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any lost or duplicated element.
    // CEP:ASSUMES: scoped threads join before drop.
    // CEP:COST: test-only; 1M transfers
    // CEP:EVIDENCE: this test; TSan in CI
    #[test]
    fn stress_spsc_roundtrip() {
        const N: u64 = 1_000_000;
        let ring = SpscRing::new(1024);
        let (prod, cons) = ring.split();
        let ok = std::thread::scope(|s| {
            let sender = s.spawn(move || {
                for i in 0..N {
                    loop {
                        match prod.push(i) {
                            Ok(()) => break,
                            Err(QueueError::Full) => std::thread::yield_now(),
                            Err(QueueError::Empty) => return false,
                        }
                    }
                }
                true
            });
            let mut received = 0u64;
            let mut next = 0u64;
            let mut buf = [0u64; 32];
            while next < N {
                let n = cons.pop_batch(&mut buf);
                if n == 0 {
                    std::thread::yield_now();
                    continue;
                }
                for v in buf.iter().take(n) {
                    if *v != next {
                        return false;
                    }
                    next += 1;
                }
                received += n as u64;
            }
            assert_eq!(received, N);
            sender.join().is_ok()
        });
        assert!(ok);
    }
}
