// CEP:FILE: crates/jit/src/boundary.rs
// CEP:WHAT: The SPSC compilation boundary between the execution thread and
//           the compiler worker.
// CEP:WHY: Master architecture section 6: "Execution Thread hits an
//          uncached dynamic shape -> pushes (GraphHash, Shape, Layout) to
//          an SPSC Queue. Compiler Worker (Anvil): Pops request, runs the
//          5-level pipeline, pushes (KernelBinary, LaunchConfig) back via
//          a return SPSC Queue." This module implements the request and
//          response rings with the tier attached; the scoped-thread
//          harness drives the worker (persistent worker thread is CEP-3).
// CEP:CLASS: CEP-0 (ring ops) / CEP-1 (harness)
// CEP:STATUS: complete
// CEP:FAILURE: QueueError propagation (bounded rings, loud overflow).
// CEP:ASSUMES: one producer (execution thread) + one consumer (compiler
//           worker) per ring pair.
// CEP:COST: push/pop ~2ns amortized (Gear 3 batching).
// CEP:EVIDENCE: tests `boundary_roundtrip`, `worker_serves_request`.
// CEP:SECURITY: POD requests only (no pointers across the boundary).
// CEP:HPC-DETERMINISM: per-channel FIFO deterministic.
//! The SPSC JIT boundary.

use anvil::spsc::{QueueError, SpscRing};

use crate::tier::Tier;

/// One compilation request (POD: fingerprint + shape + layout + tier).
///
/// CEP:WHAT: The request wire form.
/// CEP:WHY: The architecture's (GraphHash, Shape, Layout) triple plus the
///          tier policy; fixed 24 bytes keeps the ring slots POD.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: shape/layout hash from cache_key derivation.
/// CEP:COST: plain data
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CompileRequest {
    /// Structural graph fingerprint.
    pub fingerprint: u64,
    /// Derived shape+layout hash (cache key completion).
    pub shape_layout: u64,
    /// Requested tier.
    pub tier: Tier,
    /// Reserved padding.
    pub _pad: u32,
}

/// One compilation response (kernel id + tier + status).
///
/// CEP:WHAT: The response wire form.
/// CEP:WHY: The architecture's (KernelBinary, LaunchConfig) pair reduced
///          to the kernel's cache key + tier on the CPU target; fixed POD.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: kernel present in the cache under kernel_key.
/// CEP:COST: plain data
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CompileResponse {
    /// The cache key the kernel was inserted under.
    pub kernel_key: u64,
    /// The tier that compiled it.
    pub tier: Tier,
    /// True when compilation failed (caller falls back to Tier 0).
    pub failed: bool,
    /// Reserved padding.
    pub _pad: u32,
}

/// The boundary: request ring + response ring.
///
/// CEP:WHAT: The two SPSC channels.
/// CEP:WHY: Full isolation of the compiler worker from the execution
///          thread (arch section 6 "completely isolated").
/// CEP:STATUS: complete
/// CEP:FAILURE: QueueError on ring overflow (bounded, loud).
/// CEP:ASSUMES: single producer/consumer per ring.
/// CEP:COST: see module header.
/// CEP:EVIDENCE: tests in this module.
pub struct JitBoundary {
    /// Execution thread -> compiler worker.
    requests: SpscRing<CompileRequest>,
    /// Compiler worker -> execution thread.
    responses: SpscRing<CompileResponse>,
}

impl JitBoundary {
    /// CEP:WHAT: Allocates both rings (init boundary).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: capacity power of two.
    /// CEP:COST: two allocations.
    /// CEP:EVIDENCE: tests in this module.
    pub fn new(capacity: usize) -> Box<JitBoundary> {
        Box::new(JitBoundary {
            requests: *SpscRing::new(capacity),
            responses: *SpscRing::new(capacity),
        })
    }

    /// CEP:WHAT: Pushes a request (execution-thread side).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: QueueError::Full when saturated (caller retries).
    /// CEP:ASSUMES: single producer.
    /// CEP:COST: ~2ns amortized.
    /// CEP:EVIDENCE: tests in this module.
    pub fn push_request(&self, req: CompileRequest) -> Result<(), QueueError> {
        self.requests.push(req)
    }

    /// CEP:WHAT: Pops a request (compiler-worker side, single).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: QueueError::Empty.
    /// CEP:ASSUMES: single consumer.
    /// CEP:COST: ~2ns.
    /// CEP:EVIDENCE: tests in this module.
    pub fn pop_request(&self) -> Result<CompileRequest, QueueError> {
        self.requests.pop()
    }

    /// CEP:WHAT: Pushes a response (compiler-worker side).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: QueueError::Full.
    /// CEP:ASSUMES: single producer (the worker).
    /// CEP:COST: ~2ns amortized.
    /// CEP:EVIDENCE: tests in this module.
    pub fn push_response(&self, resp: CompileResponse) -> Result<(), QueueError> {
        self.responses.push(resp)
    }

    /// CEP:WHAT: Pops a response (execution-thread side, single).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: QueueError::Empty.
    /// CEP:ASSUMES: single consumer.
    /// CEP:COST: ~2ns.
    /// CEP:EVIDENCE: tests in this module.
    pub fn pop_response(&self) -> Result<CompileResponse, QueueError> {
        self.responses.pop()
    }

    /// CEP:WHAT: Best-effort pending request count (diagnostics).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 loads
    /// CEP:EVIDENCE: tests
    pub fn pending_requests(&self) -> usize {
        self.requests.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Requests and responses round-trip in FIFO order.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on loss or reorder.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn boundary_roundtrip() {
        let b = JitBoundary::new(16);
        let req = CompileRequest {
            fingerprint: 42,
            shape_layout: 7,
            tier: Tier::Tier1,
            _pad: 0,
        };
        assert!(b.push_request(req).is_ok());
        assert_eq!(b.pending_requests(), 1);
        let got = b.pop_request();
        assert!(got.is_ok());
        if let Ok(r) = got {
            assert_eq!(r.fingerprint, 42);
            assert_eq!(r.tier, Tier::Tier1);
        }
        let resp = CompileResponse {
            kernel_key: 42,
            tier: Tier::Tier1,
            failed: false,
            _pad: 0,
        };
        assert!(b.push_response(resp).is_ok());
        let g2 = b.pop_response();
        assert!(g2.is_ok());
        if let Ok(r) = g2 {
            assert_eq!(r.kernel_key, 42);
            assert!(!r.failed);
        }
    }

    // CEP:WHAT: The worker serves a request across threads (SPSC contract).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on cross-thread loss.
    // CEP:ASSUMES: scoped threads.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn worker_serves_request() {
        let b = JitBoundary::new(64);
        let ok = std::thread::scope(|s| {
            // Compiler worker: borrows the boundary (scoped thread).
            // Serves exactly ONE request, then exits so the scope can join.
            let _worker = s.spawn(|| loop {
                match b.pop_request() {
                    Ok(req) => {
                        let resp = CompileResponse {
                            kernel_key: req.fingerprint ^ req.shape_layout,
                            tier: req.tier,
                            failed: false,
                            _pad: 0,
                        };
                        return b.push_response(resp).is_ok();
                    }
                    Err(QueueError::Empty) => std::thread::yield_now(),
                    Err(QueueError::Full) => return false,
                }
            });
            // Execution thread: request Tier-1 compilation.
            let req = CompileRequest {
                fingerprint: 11,
                shape_layout: 3,
                tier: Tier::Tier1,
                _pad: 0,
            };
            if b.push_request(req).is_err() {
                return false;
            }
            // Wait for the response (bounded spin).
            for _ in 0..100_000 {
                match b.pop_response() {
                    Ok(resp) => {
                        return resp.kernel_key == 11 ^ 3 && resp.tier == Tier::Tier1;
                    }
                    Err(QueueError::Empty) => std::thread::yield_now(),
                    Err(QueueError::Full) => return false,
                }
            }
            false
        });
        assert!(ok);
    }
}
