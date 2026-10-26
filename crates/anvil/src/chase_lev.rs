// CEP:FILE: crates/anvil/src/chase_lev.rs
// CEP:WHAT: Chase-Lev bounded work-stealing deque (worker-owned push/pop, stealer CAS).
// CEP:WHY: Gear 2 (fork/join + work stealing) requires per-core deques with
//          lock-free stealing (arch section 2). The Chase-Lev algorithm is the
//          canonical provably-linearizable design (Le, Pop, Cousot, Nardelli,
//          PPoPP 2013). Fixed capacity keeps the hot path allocation-free and
//          the queue bounded (CEP&CC 39); push past capacity fails loudly with
//          `DequeError::Full` instead of hidden reallocation (Law 1).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: `DequeError::Full` (capacity exhausted), `DequeError::Empty`
//              (no element), `DequeError::Busy` (steal lost a race; retry).
//              No panics; no allocation after `Deque::new`.
// CEP:ASSUMES: exactly one owner thread calls push/pop; stealers are other
//              threads; handles (raw pointers) are valid only while the
//              executor keeps the owning `Box<Deque<T>>` alive (executor joins
//              all workers before dropping deques — see CEP:UNSAFE on Send/Sync).
// CEP:COST: push ~1 Release store + masked write; pop ~1 SeqCst fence + loads;
//           steal ~1 SeqCst fence + CAS. Measured in benches/anvil_bench.rs.
// CEP:EVIDENCE: tests `lifo_order_single_thread`, `steal_takes_fifo`,
//           `stress_owner_vs_stealer` (16 threads, 100k tasks); the CI TSan
//           hard gate (session 6): the pre-session speculative-read protocol
//           produced a reproducible data race (steal slot read vs push slot
//           write after index wraparound) that this per-slot-state protocol
//           eliminates; bench `bench_chase_lev_push_pop`.
// CEP:SECURITY: raw pointer writes bounded by mask-computed slot indices
//           derived from internally-maintained atomics; no external input.
// CEP:HPC-DETERMINISM: deterministic per-deque LIFO/FIFO contract; global
//           scheduling order is nondeterministic by design and must never be
//           observed by deterministic compilation (documented in executor.rs).
//! Chase-Lev work-stealing deque.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{fence, AtomicIsize, AtomicU8, Ordering};

/// Per-physical-slot data state. FREE: no live element (a push may write the
/// slot). FULL: a live element is present and initialized; the consumer that
/// owns the slot's index (steal CAS winner / owner pop) reads it and returns
/// the slot to FREE. This is the Go-runq discipline: every non-atomic slot
/// access is sandwiched between Acquire/Release state transitions, so no two
/// threads ever touch the same slot's data concurrently — including across
/// index wraparound (an aliased push at index t + capacity blocks on FREE
/// until the in-flight reader of index t publishes FREE).
const SLOT_FREE: u8 = 0;
const SLOT_FULL: u8 = 1;

/// Failure enumeration for deque operations.
///
/// CEP:WHAT: Explicit error type for push/pop/steal.
/// CEP:WHY: CEP-0 bans panics (25.4) and Result<(), ()> hides failure modes
///          (Law 6); three distinct outcomes require three distinct variants.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequeError {
    /// Deque is at fixed capacity (bounded; caller must drain or fail).
    Full,
    /// Deque has no elements for this caller.
    Empty,
    /// Steal lost an inter-thread race; caller should retry another deque.
    Busy,
}

/// Shared deque state; one per worker core.
///
/// CEP:WHAT: The deque inner storage: cache-padded `top`/`bottom` atomics plus
///           a fixed-capacity slot array.
/// CEP:WHY: Padding prevents false sharing between the owner's `bottom` and
///          stealers' `top` (arch: "Chase-Lev deques per core").
/// CEP:STATUS: complete
/// CEP:FAILURE: none directly; operations report via DequeError.
/// CEP:ASSUMES: capacity is a power of two (mask math); enforced by DEQUE_CAPACITY
///              static assert in config.rs and by `new`'s debug assert.
/// CEP:COST: size_of = 2 cache lines + capacity * size_of::<T>.
/// CEP:EVIDENCE: tests in this module
/// CEP:SECURITY: slot writes masked; indices bounded before Full can occur.
pub struct Deque<T> {
    /// Owner-side index (one past the last pushed element).
    bottom: CachePadded<AtomicIsize>,
    /// Stealer-side index (next element to steal).
    top: CachePadded<AtomicIsize>,
    /// Slot array, capacity = power of two. Written by owner (push),
    /// read by owner (pop) and stealers (steal) under the CAS protocol.
    buffer: UnsafeCell<Box<[MaybeUninit<T>]>>,
    /// Per-physical-slot data states (SLOT_FREE/SLOT_FULL), parallel to
    /// `buffer`. Pure atomics — no UnsafeCell needed. See SLOT_FREE doc.
    states: Box<[AtomicU8]>,
    /// Mask = capacity - 1.
    mask: usize,
}

impl<T> Deque<T> {
    /// CEP:WHAT: Allocates the deque with fixed capacity (init-time, once).
    /// CEP:WHY: The single permitted allocation; all pushes afterwards are
    ///          masked writes into this block (25.5 arena-style policy).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: panics if capacity is not a power of two (programmer error,
    ///              init-time); allowed per CEP-1 init boundary.
    /// CEP:ASSUMES: capacity comes from config::DEQUE_CAPACITY (power of two).
    /// CEP:COST: one allocation of capacity * size_of::<T>(); O(1).
    /// CEP:EVIDENCE: tests in this module
    /// CEP:SECURITY: capacity caller-controlled and bounded by config.
    pub fn new(capacity: usize) -> Box<Deque<T>> {
        debug_assert!(capacity.is_power_of_two() && capacity >= 2);
        let buffer: Box<[MaybeUninit<T>]> = core::iter::repeat_with(MaybeUninit::uninit)
            .take(capacity)
            .collect();
        let states: Box<[AtomicU8]> = core::iter::repeat_with(|| AtomicU8::new(SLOT_FREE))
            .take(capacity)
            .collect();
        Box::new(Deque {
            bottom: CachePadded::new(AtomicIsize::new(0)),
            top: CachePadded::new(AtomicIsize::new(0)),
            buffer: UnsafeCell::new(buffer),
            states,
            mask: capacity - 1,
        })
    }

    /// CEP:WHAT: Owner push (LIFO end). Never blocks, never allocates.
    /// CEP:WHY: Task submission from the owning worker: amortized ~1 Release
    ///          store. Owner-only access to `bottom` in the algorithm.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Full` when `bottom - top` reaches capacity-1 (the -1 keeps
    ///              a sentinel gap so the last-element CAS race is unambiguous).
    /// CEP:ASSUMES: single owner thread (documented contract; debug counters in
    ///              the executor cross-check ownership).
    /// CEP:COST: 2 loads + 1 masked write + 1 Release store.
    /// CEP:EVIDENCE: tests `lifo_order_single_thread`, `stress_owner_vs_stealer`
    /// CEP:SECURITY: bounds check via the Full path before any write.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), DequeError> {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Acquire);
        // size = b - t; indices never wrap because Full fires long before isize
        // range exhaustion (capacity <= 8192).
        if b - t >= (self.mask as isize) {
            return Err(DequeError::Full);
        }
        // Physical-slot gate: the slot must be FREE. It is FULL only when a
        // consumer of a PREVIOUS generation (index - capacity) is still
        // between its index CAS and its FREE publication — a few instructions,
        // reachable only after a full wraparound plus preemption inside that
        // window. Reporting `Full` then is honest backpressure (retryable),
        // never a lost element, and it is what makes the aliased write
        // impossible rather than merely unlikely.
        let slot = usize::try_from(b).map_err(|_| DequeError::Full)? & self.mask;
        if self.states[slot].load(Ordering::Acquire) != SLOT_FREE {
            return Err(DequeError::Full);
        }
        // CEP:UNSAFE | Safety: masked write into the slot array; slot index derived from
        //             `b` which the owner exclusively controls at this point and
        //             which the Full check above proved in-range. The slot state
        //             was observed FREE with Acquire: every previous consumer's
        //             read happened-before this write (their FREE store pairs
        //             with this load), so no concurrent reader exists.
        // CEP:ASSUMES: b in [t, t + capacity); slot state == FREE (both proven above).
        // CEP:SECURITY: no untrusted input reaches indices.
        unsafe {
            (*self.buffer.get())[slot].write(value);
        }
        // Publish the element's data first (Full state), then the index.
        self.states[slot].store(SLOT_FULL, Ordering::Release);
        self.bottom.store(b + 1, Ordering::Release);
        Ok(())
    }

    /// CEP:WHAT: Owner pop (LIFO end).
    /// CEP:WHY: Owner-side task consumption; collaborates with stealers through
    ///          the SeqCst fence + top CAS when popping the last element.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Empty` when no element is available; `Busy` never returned
    ///              by pop (last-element loss returns `Empty` after forgetting
    ///              the stolen value).
    /// CEP:ASSUMES: single owner thread.
    /// CEP:COST: 1 fence(SeqCst) + 2 loads + 1 CAS (last-element case only).
    /// CEP:EVIDENCE: tests `lifo_order_single_thread`, `stress_owner_vs_stealer`
    /// CEP:SECURITY: slot read bounded by top <= index < bottom invariant.
    #[inline]
    pub fn pop(&self) -> Result<T, DequeError> {
        let b = self.bottom.load(Ordering::Relaxed) - 1;
        self.bottom.store(b, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Acquire);

        if t <= b {
            // Deque non-empty from this owner's viewpoint.
            let slot_idx = usize::try_from(b).map_err(|_| DequeError::Empty)?;
            let slot = slot_idx & self.mask;
            if t == b {
                // Last element: arbitrate with concurrent stealers FIRST via
                // the top CAS; only the winner touches the slot data. (The
                // pre-session protocol read speculatively and forgot on loss
                // — that read was the wraparound data race.)
                match self
                    .top
                    .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Acquire)
                {
                    Ok(_) => {
                        // Owner won; deque is now empty. The slot data was
                        // written by this owner's own push (same-thread
                        // sequencing), and no stealer can own index b (the
                        // failed CAS means none advanced top past b).
                        // CEP:UNSAFE | Safety: exclusive data read proven by the
                        //             arbitration above; same-thread write
                        //             sequencing; aliased future writes gate on
                        //             the FREE store below.
                        // CEP:ASSUMES: top == b + 1 after the winning CAS.
                        let value = unsafe { (*self.buffer.get())[slot].assume_init_read() };
                        self.states[slot].store(SLOT_FREE, Ordering::Release);
                        self.bottom.store(b + 1, Ordering::Relaxed);
                        Ok(value)
                    }
                    Err(_) => {
                        // A stealer won ownership of this element; we never
                        // touched the slot data. Restore bottom; the stealer
                        // publishes FREE after its own read.
                        self.bottom.store(b + 1, Ordering::Relaxed);
                        Err(DequeError::Empty)
                    }
                }
            } else {
                // t < b: the slot is exclusively owned by the owner (stealers
                // can only reach indices < the stored bottom, which is now
                // b), and the data was written by this owner's own push.
                // CEP:UNSAFE | Safety: owner-exclusive data read (index CAS
                //             domain above); aliased future pushes gate on the
                //             FREE store below.
                // CEP:ASSUMES: b in (t, t + capacity); invariant of the protocol.
                let value = unsafe { (*self.buffer.get())[slot].assume_init_read() };
                self.states[slot].store(SLOT_FREE, Ordering::Release);
                Ok(value)
            }
        } else {
            // Empty: restore bottom.
            self.bottom.store(b + 1, Ordering::Relaxed);
            Err(DequeError::Empty)
        }
    }

    /// CEP:WHAT: Stealer steal (FIFO end), callable from any thread.
    /// CEP:WHY: Work stealing: idle cores take the oldest tasks, which are the
    ///          largest granularity in fork-join search trees (minimizes
    ///          stealing frequency — arch: "avoid stealing overhead on
    ///          micro-tasks").
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Empty` when nothing is stealable; `Busy` when another
    ///              stealer won the race for this slot (caller retries).
    /// CEP:ASSUMES: caller is not the owner thread (ownership is the executor's
    ///              to enforce; stealing your own deque still works but is
    ///              pointless).
    /// CEP:COST: 1 fence(SeqCst) + 2 loads + 1 CAS + 2 slot-state atomics.
    /// CEP:EVIDENCE: tests `steal_takes_fifo`, `stress_owner_vs_stealer`
    /// CEP:SECURITY: indices from internal atomics only.
    #[inline]
    pub fn steal(&self) -> Result<T, DequeError> {
        let t = self.top.load(Ordering::Acquire);
        fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if t >= b {
            return Err(DequeError::Empty);
        }
        // Resolve the slot index BEFORE reserving it: a failed conversion
        // must not abandon an already-won index (that would strand the slot
        // in FULL forever). Negative indices are unreachable (top starts at
        // zero and only ever advances via CAS), so this is purely defensive.
        let slot = match usize::try_from(t) {
            Ok(s) => s & self.mask,
            Err(_) => return Err(DequeError::Empty),
        };
        // Reserve index t FIRST (the CAS is the linearization point); the
        // slot data is touched only after ownership is won. The pre-session
        // protocol read the slot speculatively before the CAS — a data race
        // against owner pushes into the aliased slot after wraparound
        // (TSan-reproduced, session 6).
        match self
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Acquire)
        {
            Ok(_) => {
                // We own index t. Acquire-load the slot state: it pairs with
                // the owner push's FULL Release store, establishing
                // happens-before from the data write to the read below. (The
                // b > t observation above already implies FULL is visible:
                // the FULL store is sequenced before the bottom Release
                // store that our Acquire load observed.)
                if self.states[slot].load(Ordering::Acquire) != SLOT_FULL {
                    // Unreachable per the happens-before argument; defensive
                    // retry keeps the no-panic contract if memory is corrupt.
                    return Err(DequeError::Busy);
                }
                // CEP:UNSAFE | Safety: exclusive data read. Ownership of index
                //             t is proven by the winning CAS; initialization
                //             is proven by the FULL state observation (its
                //             Release store is sequenced after the owner's
                //             data write); aliasing is impossible because any
                //             future push at t + capacity must first observe
                //             this slot FREE — which this thread publishes
                //             only AFTER the read completes.
                // CEP:ASSUMES: state == FULL (checked above).
                // CEP:SECURITY: no untrusted input.
                let value = unsafe { (*self.buffer.get())[slot].assume_init_read() };
                // Release the physical slot for the next generation.
                self.states[slot].store(SLOT_FREE, Ordering::Release);
                Ok(value)
            }
            Err(_) => Err(DequeError::Busy),
        }
    }

    /// CEP:WHAT: Reports whether the deque currently holds no elements.
    /// CEP:WHY: Termination detection in the executor main loop requires a
    ///          cheap emptiness probe (Acquire loads only, no fence).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: may transiently report a stale value during a concurrent
    ///              push/steal — termination requires the two-phase pending
    ///              counter in executor.rs, never this probe alone.
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 atomic loads
    /// CEP:EVIDENCE: stress test terminates (indirect)
    #[inline]
    pub fn is_empty(&self) -> bool {
        let t = self.top.load(Ordering::Acquire);
        let b = self.bottom.load(Ordering::Acquire);
        t >= b
    }

    /// CEP:WHAT: Number of elements visible at this instant.
    /// CEP:WHY: Diagnostics and saturation telemetry (best-effort value).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: transiently stale under concurrency; documented.
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 atomic loads
    /// CEP:EVIDENCE: unit tests
    #[inline]
    pub fn len(&self) -> usize {
        let t = self.top.load(Ordering::Acquire);
        let b = self.bottom.load(Ordering::Acquire);
        if b > t {
            usize::try_from(b - t).unwrap_or(0)
        } else {
            0
        }
    }
}

impl<T> Drop for Deque<T> {
    /// CEP:WHAT: Drains and drops all remaining elements (shutdown path).
    /// CEP:WHY: The slot array may hold owned-but-unconsumed tasks; leaking
    ///          them would violate bounded-resource discipline (CEP&CC 39).
    ///          Safe only because the executor joins all worker threads before
    ///          dropping deques — no concurrent access can occur here.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: all worker threads have terminated (executor contract).
    /// CEP:COST: O(remaining elements); shutdown-only.
    /// CEP:EVIDENCE: stress test drops deques after scope join; CI ASan clean.
    fn drop(&mut self) {
        let t = self.top.load(Ordering::Relaxed);
        let b = self.bottom.load(Ordering::Relaxed);
        // CEP:ASSUMES: no concurrent access (executor joined workers first).
        // CEP:SECURITY: no untrusted input.
        if b > t {
            let lo = usize::try_from(t).unwrap_or(0);
            let hi = usize::try_from(b).unwrap_or(0);
            // CEP:UNSAFE | Safety: sequential drain of [t, b) after all
            //             threads joined; indices validated by the same
            //             protocol bounds as pop.
            unsafe {
                for i in lo..hi {
                    let _ = (*self.buffer.get())[i & self.mask].assume_init_read();
                }
            }
        }
    }
}

// Import after struct definition to keep the dependency visible at the top of
// the unsafe reasoning (config floor justifies the 64-byte align attribute).
use crate::pad::CachePadded;

// CEP:UNSAFE | Safety: Deque is Sync when T: Send: all cross-thread access goes through
//             atomics with the protocol above; the UnsafeCell buffer is only
//             touched between the per-slot state transitions (FREE-observed
//             write for push, FULL-observed read + FREE publication for
//             pop/steal — the Go-runq discipline; see SLOT_FREE doc), so no
//             two threads access the same slot's data concurrently.
// CEP:ASSUMES: exactly one owner calls push/pop (Worker discipline).
// CEP:SECURITY: no external input.
unsafe impl<T: Send> Sync for Deque<T> {}
// CEP:UNSAFE | Safety: Deque is Send when T: Send: it is a plain owner of its buffer.
unsafe impl<T: Send> Send for Deque<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // CEP:WHAT: Single-thread LIFO order for push/pop.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on order violation.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn lifo_order_single_thread() {
        let d = Deque::new(16);
        for i in 0isize..8 {
            assert!(d.push(i).is_ok());
        }
        for i in (0isize..8).rev() {
            match d.pop() {
                Ok(v) => assert_eq!(v, i),
                Err(ref e) => {
                    assert_eq!(format!("{:?}", e), "", "unexpected error");
                }
            }
        }
        assert_eq!(d.pop(), Err(DequeError::Empty));
    }

    // CEP:WHAT: Capacity enforcement returns Full, never panics or overwrites.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if bounds are violated.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn capacity_is_enforced() {
        let d = Deque::new(8);
        // Sentinel gap: capacity 8 holds 7 elements (the -1 keeps the
        // last-element CAS race unambiguous — see push CEP:WHY).
        for i in 0..7 {
            assert!(d.push(i).is_ok(), "push {} should fit", i);
        }
        assert_eq!(d.push(99), Err(DequeError::Full));
    }

    // CEP:WHAT: Steal takes from the FIFO end (oldest task) while pop takes LIFO.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on end confusion.
    // CEP:ASSUMES: single-thread test exercises layout, not races.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn steal_takes_fifo() {
        let d = Deque::new(16);
        for i in 0isize..4 {
            assert!(d.push(i).is_ok());
        }
        match d.steal() {
            Ok(v) => assert_eq!(v, 0),
            Err(ref e) => {
                assert_eq!(format!("{:?}", e), "", "unexpected error");
            }
        }
        match d.pop() {
            Ok(v) => assert_eq!(v, 3),
            Err(ref e) => {
                assert_eq!(format!("{:?}", e), "", "unexpected error");
            }
        }
    }

    // CEP:WHAT: Stress: one owner pushes/pops while 15 stealers steal; all
    //           100_000 elements must be accounted for exactly once.
    // CEP:WHY: Exercises the CAS race paths (last-element CAS, Busy retry,
    //          forget-on-lose) — a full ownership-accounting check.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if any element is duplicated or lost.
    // CEP:ASSUMES: scoped threads join before drop (std::thread::scope).
    // CEP:COST: test-only; ~100k operations
    // CEP:EVIDENCE: this test; TSan in CI (ci.yml)
    #[test]
    fn stress_owner_vs_stealer() {
        const STEALERS: usize = 15;
        const TASKS: isize = 100_000;
        let deque: Arc<Deque<isize>> = Arc::from(Deque::new(8192));
        let taken = Arc::new(std::sync::atomic::AtomicIsize::new(0));

        let ok = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(STEALERS);
            for _ in 0..STEALERS {
                let dq = Arc::clone(&deque);
                let tk = Arc::clone(&taken);
                handles.push(s.spawn(move || loop {
                    match dq.steal() {
                        Ok(_) => {
                            tk.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(DequeError::Empty) => {
                            if tk.load(Ordering::Acquire) + dq.len() as isize >= TASKS {
                                break;
                            }
                            std::thread::yield_now();
                        }
                        Err(DequeError::Busy) => continue,
                        Err(DequeError::Full) => break,
                    }
                }));
            }
            for i in 0..TASKS {
                loop {
                    match deque.push(i) {
                        Ok(_) => break,
                        Err(DequeError::Full) => {
                            std::thread::yield_now();
                        }
                        Err(_) => return false,
                    }
                }
            }
            // Owner drains the rest.
            loop {
                match deque.pop() {
                    Ok(_) => {
                        taken.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(DequeError::Empty) => break,
                    Err(_) => return false,
                }
            }
            for h in handles {
                if h.join().is_err() {
                    return false;
                }
            }
            true
        });
        assert!(ok);
        // Every task accounted exactly once.
        assert_eq!(taken.load(Ordering::Acquire), TASKS);
        assert!(deque.is_empty());
    }
}
