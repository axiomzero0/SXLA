// CEP:FILE: crates/codegen/src/lib.rs
// CEP:WHAT: codegen — bufferization, tiling, vectorization analysis, and
//           target lowering from the structured LoopProgram.
// CEP:WHY: Master architecture section 3 Level 3: "Bufferization, tiling,
//           software pipelining, vectorization, shared-memory promotion."
// CEP:CLASS: CEP-0 (transform cores) / CEP-1 (drivers)
// CEP:STATUS: partial
// CEP:FAILURE: CodegenError codes; conservative passes.
// CEP:ASSUMES: verified LoopProgram input (level3::project output).
// CEP:COST: bufferize O(ops); tile O(buffers); vectorize O(ops).
// CEP:EVIDENCE: per-module tests; end-to-end tool tests.
// CEP:SECURITY: internal slots only.
// CEP:HPC-CLASS: HPC-0 (transforms).
// CEP:HPC-DETERMINISM: deterministic.
// CEP:TODO(main-agent): CEP-23: software pipelining stages; CEP-24:
//           shared-memory promotion beyond the resource model's hints.
//! # codegen
//!
//! Level-3/4 lowering passes.

pub mod bufferize;
pub mod lower;
pub mod tile;
pub mod vectorize;

pub use bufferize::{bufferize, BufferPlan};
pub use lower::lower_target;
pub use tile::{tile_shapes, TilePlan, TILE_CACHE_LINE};
pub use vectorize::{vectorizable_width, VectorPlan};
