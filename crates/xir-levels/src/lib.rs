// CEP:FILE: crates/xir-levels/src/lib.rs
// CEP:WHAT: xir-levels — the five XIR levels (graph, tensor, fusion, loop,
//           target) plus the Anvil-aware Pass Manager.
// CEP:WHY: Master architecture section 7: passes declare their form and
//           concurrency requirements; the manager inserts graphify/
//           structurize conversions, verifies after every HPC-0 pass
//           (CEP&CC 38.18) and emits telemetry (PassStart/PassEnd) into the
//           lock-free SPSC bus.
// CEP:CLASS: CEP-1 (orchestration) / CEP-0 (level passes)
// CEP:STATUS: partial
// CEP:FAILURE: PipelineError codes; no panics.
// CEP:ASSUMES: pipeline manifests are versioned (38.20); pass order changes
//           are semantic events (see docs/pipeline.md).
// CEP:COST: manager overhead per pass: 1 clone-commit + verify O(nodes);
//           the passes themselves document their own costs.
// CEP:EVIDENCE: passman tests; integration tests in tests/.
// CEP:SECURITY: dyn Pass objects are first-party only (38.41 plugin policy:
//           no third-party plugins in this release).
// CEP:HPC-CLASS: HPC-1 (manager), HPC-0 (passes).
// CEP:TODO(main-agent): CEP-14: automatic form conversion insertion beyond
//           the current schedule-projection.
//! # xir-levels
//!
//! The five-level IR stack and the pass manager.

pub mod convert;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod level4;
pub mod passman;

pub use passman::{
    CanonicalizePass, HpcClass, IrFormPreference, Pass, PassConcurrency, PassManager, PassOutput,
    PipelineError, WorkerContext,
};
