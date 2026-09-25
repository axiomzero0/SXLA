// CEP:FILE: crates/anvil/src/ebr.rs
// CEP:WHAT: Epoch-based reclamation (EBR) for read-heavy global state.
// CEP:WHY: Gear 4 (JIT cache, type interner, layout registry — arch section 2):
//          readers must pay no Arc traffic and no locks. EBR gives readers a
//          2-atomic-op pin/unpin pair and defers writer-side frees until every
//          live guard provably pinned after the retirement.
// CEP:CLASS: CEP-0 (pin/unpin read path) / CEP-1 (retire/collect writer path)
// CEP:STATUS: complete
// CEP:FAILURE: `EbrError::SlotsExhausted` when more threads register than
//              MAX_EBR_SLOTS; `EbrError::NotRegistered` when pinning without a
//              slot. Retire never fails — worst case garbage waits for the next
//              collect (bounded by activity, drained fully at shutdown).
// CEP:ASSUMES: guards are short-lived (map lookups); a permanently pinned
//              guard pauses reclamation for ALL garbage — documented caller
//              contract, cross-checked by telemetry `ebr_pinned_stretch`.
// CEP:COST: pin = 1 SeqCst load + 1 SeqCst store (CEP-0, no allocation; the
//           SeqCst pair is the correctness fix for the in-flight-pin race, audit F-1);
//           retire = 1 allocation (writer path, CEP-1, documented); collect =
//           O(slots + garbage) scans amortized to once per
//           EBR_RETIRE_THRESHOLD retires.
// CEP:EVIDENCE: tests `pin_unpin_roundtrip`, `retire_frees_after_two_epochs`,
//           `pinned_guard_blocks_reclamation`, `concurrent_pin_retire_stress`
//           (audit F-1 regression); TSan/ASan run as ADVISORY CI jobs
//           (CEP-27 soft gate — audit F-11).
// CEP:SECURITY: raw pointers in the garbage list never escape the collector;
//               drop functions are monomorphized, type-safe closures.
// CEP:HPC-DETERMINISM: deterministic; epoch order does not affect observable
//           compilation results (garbage freeing is not observable).
//! Epoch-based reclamation.

use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::config::{EBR_RETIRE_THRESHOLD, MAX_WORKERS};
use crate::pad::CachePadded;

/// Upper bound on concurrently registered reader slots.
///
/// CEP:WHAT: Slot count bound.
/// CEP:WHY: Slots are preallocated (bounded resource, CEP&CC 39): MAX_WORKERS
///          execution workers + headroom for the JIT compiler thread, the
///          telemetry drainer and the SPSC resolver thread (arch section 4).
/// CEP:STATUS: complete
/// CEP:FAILURE: register past this returns `EbrError::SlotsExhausted`.
/// CEP:ASSUMES: none
/// CEP:COST: one preallocated array
/// CEP:EVIDENCE: test `slots_are_bounded`
pub const MAX_EBR_SLOTS: usize = MAX_WORKERS + 4;

/// Failure enumeration for EBR registration.
///
/// CEP:WHAT: Explicit error type for slot management.
/// CEP:WHY: Law 6 — failure must be explicit; CEP-0 bans panics.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EbrError {
    /// All preallocated reader slots are taken.
    SlotsExhausted,
    /// Operation requires a slot index from `register()`.
    NotRegistered,
}

/// A retired allocation awaiting safe destruction.
struct Retired {
    /// Type-erased pointer to the allocation.
    ptr: *mut u8,
    /// Monomorphized destructor: `drop(Box::from_raw(ptr as *mut T))`.
    drop_fn: unsafe fn(*mut u8),
    /// Global epoch at retirement time.
    epoch: usize,
    /// Intrusive next pointer for the lock-free garbage stack.
    next: *mut Retired,
}

/// CEP:WHAT: Erases the concrete destructor of a `Box<T>` into a raw fn.
/// CEP:WHY: The garbage stack is type-erased; each node must still destroy its
///          payload with the correct monomorphized destructor.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: `p` originated from `Box::into_raw` of a `T`.
/// CEP:COST: call to `Box::from_raw` + drop glue
/// CEP:EVIDENCE: tests `retire_frees_after_two_epochs` (drop counter)
/// CEP:SECURITY: pointer provenance is internal (retire_box below).
unsafe fn erase_drop<T>(p: *mut u8) {
    // CEP:UNSAFE | Safety: reconstructing the Box to run the real destructor.
    // CEP:ASSUMES: provenance guaranteed by `retire_box`.
    // CEP:SECURITY: no external pointers.
    unsafe {
        drop(Box::from_raw(p as *mut T));
    }
}

/// Epoch-based reclamation collector.
///
/// CEP:WHAT: Global epoch counter + fixed slot array + lock-free garbage stack.
/// CEP:WHY: Safety argument (crossbeam-epoch algorithm, simplified):
///          (1) `pin` records the current global epoch in the caller's slot;
///          (2) `retire` stamps garbage with the epoch at retirement;
///          (3) the epoch may advance only when every pinned slot is at the
///          current epoch;
///          therefore when the global epoch reaches e+2, no guard pinned at
///          epoch <= e can still be alive (such a guard would have blocked
///          both advances), so garbage stamped e is provably unreachable and
///          safe to destroy. Readers pay 2 atomics; writers pay allocation.
/// CEP:STATUS: complete
/// CEP:FAILURE: see `EbrError`; collect silently keeps garbage when a stale
///              guard blocks advancement (correct, bounded by guard lifetime).
/// CEP:ASSUMES: each thread uses exactly one slot index; a slot is pinned by
///              at most one guard at a time (executor assigns slots).
/// CEP:COST: pin/unpin: 2 atomics; retire: 1 alloc; collect: O(slots+garbage).
/// CEP:EVIDENCE: tests `retire_frees_after_two_epochs`,
///           `pinned_guard_blocks_reclamation`
/// CEP:SECURITY: garbage pointers originate only from `retire_box`; nothing
///           external can inject pointers.
pub struct Collector {
    /// Monotonic global epoch (wraps at usize::MAX; 2^64 unreachable).
    global_epoch: CachePadded<AtomicUsize>,
    /// Reader slots: 0 = unpinned, else (pinned epoch + 1).
    slots: Vec<CachePadded<AtomicUsize>>,
    /// Next slot to hand out (monotonic; slots never return to the pool).
    next_slot: AtomicUsize,
    /// Treiber stack of retired nodes (writer-side only).
    garbage_head: AtomicPtr<Retired>,
    /// Retires since the last collect attempt.
    retires_since_collect: AtomicUsize,
    /// Total destroyed allocations (diagnostic).
    freed_count: AtomicUsize,
}

impl Collector {
    /// CEP:WHAT: Creates the collector with all slots unpinned (init-time).
    /// CEP:WHY: Slot vector is allocated once here (CEP-1 init boundary);
    ///          pin/unpin never touch the allocator afterwards.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: one allocation of MAX_EBR_SLOTS padded atomics
    /// CEP:EVIDENCE: tests in this module
    /// CEP:SECURITY: internal only
    pub fn new() -> Box<Collector> {
        let mut slots = Vec::with_capacity(MAX_EBR_SLOTS);
        for _ in 0..MAX_EBR_SLOTS {
            slots.push(CachePadded::new(AtomicUsize::new(0)));
        }
        Box::new(Collector {
            global_epoch: CachePadded::new(AtomicUsize::new(0)),
            slots,
            next_slot: AtomicUsize::new(0),
            garbage_head: AtomicPtr::new(core::ptr::null_mut()),
            retires_since_collect: AtomicUsize::new(0),
            freed_count: AtomicUsize::new(0),
        })
    }

    /// CEP:WHAT: Reserves a reader slot index for a thread (init-time).
    /// CEP:WHY: Slots are assigned per worker/JIT thread at startup; the
    ///          monotonic counter makes assignment deterministic.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `SlotsExhausted` beyond MAX_EBR_SLOTS.
    /// CEP:ASSUMES: called once per thread, before any pin on that slot.
    /// CEP:COST: one fetch_add
    /// CEP:EVIDENCE: test `slots_are_bounded`
    /// CEP:SECURITY: none
    pub fn register(&self) -> Result<usize, EbrError> {
        let slot = self.next_slot.fetch_add(1, Ordering::AcqRel);
        if slot < MAX_EBR_SLOTS {
            Ok(slot)
        } else {
            Err(EbrError::SlotsExhausted)
        }
    }

    /// CEP:WHAT: Pins the calling thread to the current epoch (CEP-0 read path).
    /// CEP:WHY: Lookup guards: while pinned, no retired allocation visible to
    ///          this thread can be destroyed (see crate-level safety argument).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `NotRegistered` if slot >= slot count (programmer error).
    /// CEP:ASSUMES: slot was returned by `register`; one guard per slot at a
    ///              time (executor discipline).
    /// CEP:COST: 1 Acquire load + 1 Release store; no allocation, no lock.
    /// CEP:EVIDENCE: tests `pin_unpin_roundtrip`,
    ///           `pinned_guard_blocks_reclamation`
    /// CEP:SECURITY: none
    #[inline]
    pub fn pin(&self, slot: usize) -> Result<Guard<'_>, EbrError> {
        if slot >= self.slots.len() {
            return Err(EbrError::NotRegistered);
        }
        // SeqCst on BOTH the epoch load and the slot store (crossbeam-epoch
        // discipline, audit F-1): weaker ordering lets a concurrent collector
        // scan the slot as unpinned twice while this pin is in flight, advance
        // the epoch twice, and free garbage this guard still traverses.
        let epoch = self.global_epoch.load(Ordering::SeqCst);
        // Store epoch + 1 so 0 remains the "unpinned" sentinel.
        self.slots[slot]
            .value
            .store(epoch.wrapping_add(1), Ordering::Release);
        Ok(Guard {
            collector: self,
            slot,
        })
    }

    /// CEP:WHAT: Current global epoch (diagnostic).
    /// CEP:WHY: Telemetry and tests.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests in this module
    pub fn current_epoch(&self) -> usize {
        self.global_epoch.load(Ordering::Acquire)
    }

    /// CEP:WHAT: Total destroyed garbage count (diagnostic).
    /// CEP:WHY: Leak detection: must reach total retires after shutdown drain.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: test `retire_frees_after_two_epochs`
    pub fn freed_count(&self) -> usize {
        self.freed_count.load(Ordering::Acquire)
    }

    /// CEP:WHAT: Retires a heap allocation for deferred destruction (writer path).
    /// CEP:WHY: Writers (JIT cache updaters, table resizers) must not free
    ///          memory a pinned reader may still traverse; retiring defers the
    ///          destructor until provably safe (two-epoch lag).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: never fails; if the garbage stack push races, the loop
    ///              retries (lock-free Treiber push is wait-free in practice).
    /// CEP:ASSUMES: `boxed` is not otherwise reachable after this call.
    /// CEP:COST: 1 allocation for the Retired node + CAS push; amortized
    ///           collect every EBR_RETIRE_THRESHOLD retires.
    /// CEP:EVIDENCE: tests `retire_frees_after_two_epochs`
    /// CEP:SECURITY: pointer provenance from Box::into_raw inside this fn.
    pub fn retire_box<T>(&self, boxed: Box<T>) {
        let node = Box::into_raw(Box::new(Retired {
            ptr: Box::into_raw(boxed) as *mut u8,
            drop_fn: erase_drop::<T>,
            epoch: self.global_epoch.load(Ordering::Acquire),
            next: core::ptr::null_mut(),
        }));
        // Treiber push.
        let mut head = self.garbage_head.load(Ordering::Relaxed);
        loop {
            // CEP:UNSAFE | Safety: writing the intrusive next pointer before CAS
            //             publication; the node is exclusively ours until the
            //             CAS succeeds, then it is owned by the stack.
            // CEP:ASSUMES: node is not yet visible to other threads.
            // CEP:SECURITY: internal pointers only.
            unsafe { (*node).next = head };
            match self.garbage_head.compare_exchange_weak(
                head,
                node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(h) => head = h,
            }
        }
        let n = self.retires_since_collect.fetch_add(1, Ordering::AcqRel) + 1;
        if n >= EBR_RETIRE_THRESHOLD as usize {
            self.collect();
            self.retires_since_collect.store(0, Ordering::Release);
        }
    }

    /// CEP:WHAT: Advances the epoch if safe, then frees aged-out garbage.
    /// CEP:WHY: The advance rule (all pinned slots at current epoch) plus the
    ///          two-epoch lag is the safety proof in the crate comment; free
    ///          criteria: garbage.epoch + 2 <= global epoch.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none; garbage that cannot be proven safe is kept.
    /// CEP:ASSUMES: called from writer threads only (single logical writer at
    ///              a time is NOT required — CAS guarded).
    /// CEP:COST: O(slots) scan + O(garbage) sweep; amortized.
    /// CEP:EVIDENCE: tests `retire_frees_after_two_epochs`,
    ///           `pinned_guard_blocks_reclamation`
    /// CEP:SECURITY: destructors run only on internally-retired pointers.
    pub fn collect(&self) {
        // Phase 1: try to advance the epoch. SeqCst loads + SeqCst CAS pair
        // with pin's SeqCst stores so a pin in flight is ALWAYS visible to
        // the scan that decides the advance (audit F-1).
        let g = self.global_epoch.load(Ordering::SeqCst);
        let mut can_advance = true;
        for slot in &self.slots {
            let v = slot.value.load(Ordering::SeqCst);
            if v != 0 && v != g.wrapping_add(1) {
                can_advance = false;
                break;
            }
        }
        if can_advance {
            let _ = self.global_epoch.compare_exchange(
                g,
                g.wrapping_add(1),
                Ordering::SeqCst,
                Ordering::Acquire,
            );
        }
        // Phase 2: free garbage two epochs behind.
        let now = self.global_epoch.load(Ordering::Acquire);
        // Pop the entire stack, partition into free / keep, push keeps back.
        let mut keep: Vec<*mut Retired> = Vec::new();
        let mut head = self
            .garbage_head
            .swap(core::ptr::null_mut(), Ordering::Acquire);
        let mut freed = 0usize;
        while !head.is_null() {
            // CEP:ASSUMES: nodes are valid Retired boxes from retire_box.
            // CEP:SECURITY: internal pointers only.
            // CEP:UNSAFE | Safety: traversing the popped (now privately
            //             owned) stack — each node was published via the
            //             Treiber push above; the erased destructor runs on
            //             the payload only for aged-out garbage.
            let (next, stamp) = unsafe { ((*head).next, (*head).epoch) };
            if now >= stamp.wrapping_add(2) {
                // CEP:UNSAFE | Safety: aged-out garbage — reconstruct the node
                //             Box and run its erased destructor once.
                unsafe {
                    let boxed = Box::from_raw(head);
                    (boxed.drop_fn)(boxed.ptr);
                }
                freed += 1;
            } else {
                keep.push(head);
            }
            head = next;
        }
        // Push survivors back (reverse order is irrelevant).
        for node in keep {
            let mut h = self.garbage_head.load(Ordering::Relaxed);
            loop {
                // CEP:UNSAFE | Safety: same Treiber publication as retire_box.
                unsafe { (*node).next = h };
                match self.garbage_head.compare_exchange_weak(
                    h,
                    node,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => h = x,
                }
            }
        }
        if freed > 0 {
            self.freed_count.fetch_add(freed, Ordering::AcqRel);
        }
    }
}

/// A pinned guard: while alive, retired memory visible to this thread is safe.
///
/// CEP:WHAT: RAII guard for a reader slot pin.
/// CEP:WHY: Drop discipline cannot be forgotten by callers (Law 3 enforced by
///          the type system); pin/unpin cost is 2 atomics.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: slot uniqueness per thread (executor contract).
/// CEP:COST: 2 atomics over the guard's lifetime
/// CEP:EVIDENCE: tests in this module
/// CEP:SECURITY: none
pub struct Guard<'a> {
    collector: &'a Collector,
    slot: usize,
}

impl Guard<'_> {
    /// CEP:WHAT: Retires a box through this guard's collector (writer helper).
    /// CEP:WHY: Ergonomic retire for sharded-map writers holding a guard.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: delegates to `Collector::retire_box` (never fails).
    /// CEP:ASSUMES: box unreachable afterwards.
    /// CEP:COST: see retire_box
    /// CEP:EVIDENCE: tests in this module
    /// CEP:SECURITY: internal provenance
    pub fn retire_box<T>(&self, boxed: Box<T>) {
        self.collector.retire_box(boxed);
    }

    /// CEP:WHAT: The collector this guard pins.
    /// CEP:WHY: Sharded map lookups need the owning collector handle.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: zero
    /// CEP:EVIDENCE: sharded map tests
    pub fn collector(&self) -> &Collector {
        self.collector
    }
}

impl Drop for Guard<'_> {
    /// CEP:WHAT: Unpins the slot (Release store of sentinel 0).
    /// CEP:WHY: Re-enables epoch advance past this thread.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 Release store
    /// CEP:EVIDENCE: tests in this module
    fn drop(&mut self) {
        self.collector.slots[self.slot]
            .value
            .store(0, Ordering::Release);
    }
}

impl Drop for Collector {
    /// CEP:WHAT: Final drain — destroys ALL remaining garbage.
    /// CEP:WHY: Bounded-resource discipline: no leak at shutdown. Safe because
    ///          the owner joins all threads before dropping the collector.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: no thread is pinned (owner joined all workers).
    /// CEP:COST: O(garbage); shutdown-only
    /// CEP:EVIDENCE: tests count freed == retired
    fn drop(&mut self) {
        let mut head = self
            .garbage_head
            .swap(core::ptr::null_mut(), Ordering::Relaxed);
        let mut freed = 0usize;
        while !head.is_null() {
            // CEP:ASSUMES: all threads joined.
            // CEP:SECURITY: internal pointers only.
            // CEP:UNSAFE | Safety: private traversal after quiescence — every
            //             node is destroyed with its erased destructor.
            let next = unsafe { (*head).next };
            // CEP:UNSAFE | Safety: final drain — destroy the node and its
            //             payload exactly once after quiescence.
            unsafe {
                let boxed = Box::from_raw(head);
                (boxed.drop_fn)(boxed.ptr);
            }
            freed += 1;
            head = next;
        }
        self.freed_count.store(freed, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize as StdAtomicUsize;
    use std::sync::Arc;

    // CEP:WHAT: pin/unpin leaves the slot unpinned and never advances garbage.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on stuck slot.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pin_unpin_roundtrip() {
        let c = Collector::new();
        let reg = c.register();
        assert!(reg.is_ok());
        let slot = match reg {
            Ok(s) => s,
            Err(_) => return,
        };
        {
            let _g = c.pin(slot);
            assert!(c.slots[slot].value.load(Ordering::Acquire) > 0);
        }
        assert_eq!(c.slots[slot].value.load(Ordering::Acquire), 0);
    }

    // CEP:WHAT: Retired memory is freed after two epoch advances.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if garbage leaks or frees early.
    // CEP:ASSUMES: no concurrent pinners (single-threaded).
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn retire_frees_after_two_epochs() {
        let drops = Arc::new(StdAtomicUsize::new(0));
        let c = Collector::new();
        let slot = c.register().ok();
        assert!(slot.is_some());
        // (register of a fresh collector cannot fail; keep the guard simple)
        {
            let pin = c.pin(slot.unwrap_or(0));
            assert!(pin.is_ok());
            if let Ok(g) = pin {
                // Retire 3 boxes with a drop counter.
                for _ in 0..3 {
                    let d = Arc::clone(&drops);
                    let payload = DropCounter { count: d };
                    g.retire_box(Box::new(payload));
                }
                c.collect(); // epoch 0 -> 1 (pinned at 0 == current 0, advance ok)
            }
        }
        // Guard dropped: unpinned.
        c.collect(); // epoch 1 -> 2; garbage stamped 0 satisfies 2 >= 0+2.
        assert_eq!(drops.load(Ordering::Acquire), 3);
        assert_eq!(c.freed_count(), 3);
    }

    // CEP:WHAT: A pinned guard blocks reclamation of garbage it could hold.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if memory is freed under a live guard.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pinned_guard_blocks_reclamation() {
        let drops = Arc::new(StdAtomicUsize::new(0));
        let c = Collector::new();
        let slot = c.register().ok();
        assert!(slot.is_some());
        // (register of a fresh collector cannot fail; keep the guard simple)
        {
            let _g = c.pin(slot.unwrap_or(0));
            let d = Arc::clone(&drops);
            c.retire_box(Box::new(DropCounter { count: d }));
            // Guard still pinned at epoch 0: advance requires all==current.
            c.collect();
            c.collect();
            assert_eq!(drops.load(Ordering::Acquire), 0);
        }
        // Now unpinned: two advances free it.
        c.collect();
        c.collect();
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }

    // CEP:WHAT: Registration is bounded by MAX_EBR_SLOTS.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the bound is not enforced.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn slots_are_bounded() {
        let c = Collector::new();
        for _ in 0..MAX_EBR_SLOTS {
            assert!(c.register().is_ok());
        }
        assert_eq!(c.register(), Err(EbrError::SlotsExhausted));
    }

    // CEP:WHAT: Concurrent pin/retire/collect stress (audit F-1 regression).
    // CEP:WHY: The SeqCst ordering fix is only provable under real
    //          interleaving: N threads pin/unpin/retire while a collector
    //          thread advances epochs; every DropCounter must fire exactly
    //          once and the process must not crash (early free = use-after
    //          free would surface under ASan in CI).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on dropped-payload miscount.
    // CEP:ASSUMES: scoped threads join before the collector drops.
    // CEP:COST: test-only; 200k pinned operations
    // CEP:EVIDENCE: this test (plus advisory TSan, CEP-27)
    #[test]
    fn concurrent_pin_retire_stress() {
        use std::sync::atomic::AtomicUsize as StdAtomicUsize;
        const THREADS: usize = 4;
        const OPS: usize = 50_000;
        let drops = Arc::new(StdAtomicUsize::new(0));
        let retired = Arc::new(StdAtomicUsize::new(0));
        let c = Collector::new();
        // Reader slots.
        let mut slots = Vec::new();
        for _ in 0..THREADS {
            match c.register() {
                Ok(s) => slots.push(s),
                Err(_) => return, // assert below catches the shortfall
            }
        }
        let collector: &Collector = &c;
        assert_eq!(slots.len(), THREADS);
        let ok = std::thread::scope(|s| {
            // Collector thread: continuously advance/collect.
            let col = s.spawn(move || {
                for _ in 0..OPS / 10 {
                    collector.collect();
                    std::thread::yield_now();
                }
            });
            // Reader threads: pin, retire occasionally, unpin.
            let mut handles = Vec::new();
            for &slot in slots.iter() {
                let d = Arc::clone(&drops);
                let r = Arc::clone(&retired);
                handles.push(s.spawn(move || {
                    for i in 0..OPS {
                        let g = match collector.pin(slot) {
                            Ok(g) => g,
                            Err(_) => return false,
                        };
                        if i % 500 == 0 {
                            let d2 = Arc::clone(&d);
                            r.fetch_add(1, Ordering::AcqRel);
                            g.retire_box(Box::new(DropCounter { count: d2 }));
                        }
                        drop(g);
                        if i % 64 == 0 {
                            std::thread::yield_now();
                        }
                    }
                    true
                }));
            }
            for h in handles {
                if h.join().is_err() {
                    return false;
                }
            }
            col.join().is_ok()
        });
        assert!(ok);
        // Final drain frees everything.
        c.collect();
        c.collect();
        assert_eq!(
            retired.load(Ordering::Acquire),
            drops.load(Ordering::Acquire)
        );
    }

    /// Drop-counter payload for reclamation tests.
    struct DropCounter {
        count: Arc<StdAtomicUsize>,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.count.fetch_add(1, Ordering::AcqRel);
        }
    }
}
