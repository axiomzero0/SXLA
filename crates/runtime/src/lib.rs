// CEP:FILE: crates/runtime/src/lib.rs
// CEP:WHAT: runtime — device execution, streams, buffers, and the Tier-0
//           interpreter for TargetPrograms.
// CEP:WHY: Master architecture section 8: "Device execution, Streams, SPSC
//          boundaries." The runtime owns the CPU device, executes lowered
//          programs and exposes the stream/event API the JIT's execution
//          thread drives.
// CEP:CLASS: CEP-0 (kernels) / CEP-1 (device/stream API)
// CEP:STATUS: complete
// CEP:FAILURE: RuntimeError codes (shape mismatch, unsupported value,
//             bad slot); no panics on any execution path.
// CEP:ASSUMES: verified/lowered programs; inputs match parameter types.
// CEP:COST: interpreter O(instrs * tensor elements); naive kernels
//           documented honestly (Tier-0 correctness path, not peak perf).
// CEP:EVIDENCE: per-module tests; xla-run end-to-end tests; differential
//           tests fused-vs-unfused.
// CEP:SECURITY: untrusted programs bounded by slot checks; allocation
//           bounded by type-checked tensor sizes (verified upstream).
// CEP:HPC-CLASS: HPC-0 (kernels), HPC-1 (API).
// CEP:HPC-DETERMINISM: deterministic — fixed iteration order, seeded RNG.
//! # runtime
//!
//! Device execution and the reference interpreter.

pub mod device;
pub mod interp;
pub mod stream;
pub mod value;

pub use device::CpuDevice;
pub use interp::{execute, RuntimeError};
pub use stream::{CpuStream, StreamEvent};
pub use value::Value;
