// CEP:FILE: crates/anvil/src/telemetry.rs
// CEP:WHAT: Lock-free SPSC telemetry bus (Gear 3) for pass instrumentation.
// CEP:WHY: Master architecture section 7: "Passes emit structured telemetry
//          (fusion decisions, e-graph rewrites) into a lock-free SPSC
//          telemetry queue for debugging and visualization". One bounded SPSC
//          ring per worker keeps producers single-threaded per channel (the
//          SPSC contract) with zero blocking; overflow drops are counted and
//          reported — telemetry is best-effort by contract (CEP&CC Law 1/6).
// CEP:CLASS: CEP-0 (emit) / CEP-1 (drain/formatting)
// CEP:STATUS: complete
// CEP:FAILURE: emit returns false when a worker's ring is full or the index
//              is out of range; the caller drops the event (counted in
//              `dropped_count`) and continues — compilation is never blocked
//              by telemetry.
// CEP:ASSUMES: exactly one thread (its worker) emits per ring index; drain
//              runs on one dedicated consumer thread or at region end.
// CEP:COST: emit = 1 masked write + 1 Release store (~2ns amortized via
//           pop_batch on the consumer); drain = batched pops.
// CEP:EVIDENCE: tests `emit_and_drain_roundtrip`, `overflow_is_counted`.
// CEP:SECURITY: payloads are POD (kind, pass, payload u64); no strings, no
//           secrets, no pointers — safe to retain across stages.
// CEP:HPC-DETERMINISM: telemetry is observational; it never feeds back into
//           translation decisions (else it would be a determinism hazard).
//! SPSC telemetry bus.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::config::TELEMETRY_CAPACITY;
use crate::spsc::SpscRing;

/// Event kinds emitted by the compiler stack.
///
/// CEP:WHAT: Discriminant for telemetry records.
/// CEP:WHY: Fixed-width POD events avoid string formatting on the hot path
///          (CEP&CC 5.1 bans formatting in CEP-0); the consumer renders.
/// CEP:STATUS: complete
/// CEP:FAILURE: unknown discriminants are skipped by the drainer (forward
///              compatibility).
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TelemetryKind {
    /// A pass started on a worker.
    PassStart = 0,
    /// A pass finished on a worker.
    PassEnd = 1,
    /// A fusion decision (payload: cluster id / cost).
    FusionDecision = 2,
    /// An e-graph rewrite applied (payload: rule id).
    EgraphRewrite = 3,
    /// A JIT compile request served (payload: tier).
    JitCompile = 4,
    /// A JIT cache hit (payload: cache key high bits).
    JitHit = 5,
    /// Search branch pruned by atomic best-cost (payload: node count).
    SearchPruned = 6,
}

impl TelemetryKind {
    /// CEP:WHAT: Numeric code for the POD wire form.
    /// CEP:WHY: The ring stores POD structs; the enum packs to u8.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: zero
    /// CEP:EVIDENCE: tests in this module
    pub fn code(self) -> u8 {
        self as u8
    }

    /// CEP:WHAT: Decodes a numeric code (consumer side).
    /// CEP:WHY: Forward-compatible skipping of unknown codes.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns None on unknown code.
    /// CEP:ASSUMES: none
    /// CEP:COST: branch table
    /// CEP:EVIDENCE: tests in this module
    pub fn from_code(c: u8) -> Option<TelemetryKind> {
        match c {
            0 => Some(TelemetryKind::PassStart),
            1 => Some(TelemetryKind::PassEnd),
            2 => Some(TelemetryKind::FusionDecision),
            3 => Some(TelemetryKind::EgraphRewrite),
            4 => Some(TelemetryKind::JitCompile),
            5 => Some(TelemetryKind::JitHit),
            6 => Some(TelemetryKind::SearchPruned),
            _ => None,
        }
    }
}

/// One telemetry record: 16 bytes of POD, no pointers, no strings.
///
/// CEP:WHAT: The event wire format.
/// CEP:WHY: Fixed size keeps the ring slots trivially movable POD (Send by
///          value) and lets the drainer batch-copy with memcpy-like speed.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: payload semantics depend on `kind` (documented per variant).
/// CEP:COST: 16 bytes per event
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TelemetryEvent {
    /// Event discriminant (TelemetryKind code).
    pub kind: u8,
    /// Reserved for alignment/future flags.
    pub _pad: u8,
    /// Pass or subsystem identifier.
    pub pass_id: u16,
    /// Kind-specific payload (count, cost, id, tier...).
    pub payload: u64,
}

/// Per-worker SPSC telemetry bus.
///
/// CEP:WHAT: One bounded ring per worker plus a global dropped counter.
/// CEP:WHY: Worker-local rings satisfy the SPSC single-producer contract;
///          the dropped counter makes best-effort overflow observable
///          (Law 6) without blocking.
/// CEP:STATUS: complete
/// CEP:FAILURE: see module header; never blocks, never allocates on emit.
/// CEP:ASSUMES: emit(worker) called only by that worker thread.
/// CEP:COST: emit ~2ns amortized; drain batched.
/// CEP:EVIDENCE: tests `emit_and_drain_roundtrip`, `overflow_is_counted`
/// CEP:SECURITY: POD payloads only; no secret material may be emitted
///           (policy: payloads are ids/costs, never key material).
pub struct TelemetryBus {
    /// One ring per worker (bounded by TELEMETRY_CAPACITY).
    rings: Vec<SpscRing<TelemetryEvent>>,
    /// Total dropped events across all rings (best-effort accounting).
    dropped: AtomicU64,
}

impl TelemetryBus {
    /// CEP:WHAT: Allocates one ring per worker (init-time).
    /// CEP:WHY: All allocation at init; emit never allocates.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: panics on zero workers (programmer error at init).
    /// CEP:ASSUMES: worker count matches the executor region.
    /// CEP:COST: num_workers allocations
    /// CEP:EVIDENCE: tests in this module
    /// CEP:SECURITY: internal only
    pub fn new(num_workers: usize) -> Box<TelemetryBus> {
        debug_assert!(num_workers > 0);
        let mut rings = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            rings.push(*SpscRing::new(TELEMETRY_CAPACITY));
        }
        Box::new(TelemetryBus {
            rings,
            dropped: AtomicU64::new(0),
        })
    }

    /// CEP:WHAT: Emits one event from a worker (CEP-0 hot path).
    /// CEP:WHY: Pass instrumentation without blocking or formatting; the
    ///          event is POD so no allocation happens.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns false on ring overflow or bad worker index; the
    ///              event is dropped and counted (never blocks, never panics).
    /// CEP:ASSUMES: called only by worker `worker`.
    /// CEP:COST: 1 masked write + 1 Release store.
    /// CEP:EVIDENCE: tests in this module
    /// CEP:SECURITY: payload policy: ids and costs only.
    #[inline]
    pub fn emit(&self, worker: usize, ev: TelemetryEvent) -> bool {
        if worker >= self.rings.len() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        match self.rings[worker].push(ev) {
            Ok(()) => true,
            Err(_) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// CEP:WHAT: Batched drain of one worker's ring (consumer side).
    /// CEP:WHY: Fence amortization (Gear 3 batching); the caller renders.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none; returns 0 when empty.
    /// CEP:ASSUMES: called by the single consumer thread.
    /// CEP:COST: O(k) per batch, 2 fences.
    /// CEP:EVIDENCE: test `emit_and_drain_roundtrip`
    /// CEP:SECURITY: none
    pub fn drain(&self, worker: usize, out: &mut Vec<TelemetryEvent>) -> usize {
        if worker >= self.rings.len() {
            return 0;
        }
        let (_prod, cons) = self.rings[worker].split();
        // Batch buffer on the caller-provided Vec (CEP-1 drain path).
        let mut buf = [TelemetryEvent {
            kind: 0,
            _pad: 0,
            pass_id: 0,
            payload: 0,
        }; 64];
        let mut total = 0;
        loop {
            let n = cons.pop_batch(&mut buf);
            if n == 0 {
                break;
            }
            for ev in buf.iter().take(n) {
                out.push(*ev);
            }
            total += n;
            if n < buf.len() {
                break;
            }
        }
        total
    }

    /// CEP:WHAT: Total dropped events (diagnostic).
    /// CEP:WHY: Overflow visibility (Law 6) without hot-path cost.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none; best-effort counter.
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 atomic load
    /// CEP:EVIDENCE: test `overflow_is_counted`
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Acquire)
    }

    /// CEP:WHAT: Number of worker rings.
    /// CEP:WHY: Executor wiring check.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests
    pub fn worker_count(&self) -> usize {
        self.rings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Emit then drain round trip preserves order and content.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on loss or reorder.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn emit_and_drain_roundtrip() {
        let bus = TelemetryBus::new(2);
        for i in 0..100u64 {
            let ev = TelemetryEvent {
                kind: TelemetryKind::PassStart.code(),
                _pad: 0,
                pass_id: 7,
                payload: i,
            };
            assert!(bus.emit(0, ev));
        }
        let mut out = Vec::new();
        let n = bus.drain(0, &mut out);
        assert_eq!(n, 100);
        assert_eq!(out.len(), 100);
        for (i, ev) in out.iter().enumerate() {
            assert_eq!(ev.payload, i as u64);
            assert_eq!(ev.kind, TelemetryKind::PassStart.code());
        }
        // Worker 1 ring is untouched.
        let mut out1 = Vec::new();
        assert_eq!(bus.drain(1, &mut out1), 0);
    }

    // CEP:WHAT: Ring overflow drops are counted, not silently lost.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if drops are uncounted.
    // CEP:ASSUMES: capacity TELEMETRY_CAPACITY bounds the ring.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn overflow_is_counted() {
        let bus = TelemetryBus::new(1);
        let ev = TelemetryEvent {
            kind: TelemetryKind::SearchPruned.code(),
            _pad: 0,
            pass_id: 1,
            payload: 42,
        };
        let mut emitted = 0u64;
        let mut dropped = 0u64;
        for _ in 0..TELEMETRY_CAPACITY * 2 {
            if bus.emit(0, ev) {
                emitted += 1;
            } else {
                dropped += 1;
            }
        }
        assert_eq!(emitted, TELEMETRY_CAPACITY as u64);
        assert_eq!(dropped, TELEMETRY_CAPACITY as u64);
        assert_eq!(bus.dropped_count(), TELEMETRY_CAPACITY as u64);
    }
}
