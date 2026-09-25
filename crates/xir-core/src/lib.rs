// CEP:FILE: crates/xir-core/src/lib.rs
// CEP:WHAT: xir-core — XIR node/edge storage, types, attributes, arena
//           allocators, snapshots, canonical text, deterministic hashing.
// CEP:WHY: The master architecture places "Node/Edge storage, Types,
//          Attributes, Arena allocators" at the foundation of the 5-level IR
//          stack; every other crate builds on these primitives without
//          touching raw storage.
// CEP:CLASS: CEP-0 (ids, types, ops, arena) / CEP-1 (text, snapshot commit)
// CEP:STATUS: complete
// CEP:FAILURE: per-module error enums (ArenaError, TypeError, CommitError,
//              ParseError); no panics on any public path.
// CEP:ASSUMES: no_std-incompatible features are confined to snapshot (Arc)
//              and text (String) which are documented CEP-1.
// CEP:COST: see per-module CEP:COST fields.
// CEP:EVIDENCE: per-module unit tests; workspace integration tests.
// CEP:SECURITY: parser validates all external input; no unsafe code in this
//           crate (grep-verifiable; CI lint enforces).
// CEP:HPC-CLASS: HPC-0 (IR storage), HPC-1 (serialization).
// CEP:HPC-IR: Owns node/region storage and the packed id scheme.
// CEP:HPC-DETERMINISM: deterministic — ids derive from slot order, hashes
//           from structure, printing from slot order (CEP&CC 38.19).
//! # xir-core
//!
//! Foundation crate of the SXLA 5-level IR stack: deterministic packed ids,
//! bounded arenas with generation-guarded reuse, the unified opcode set,
//! the type system, transactional snapshots, and canonical text.

pub mod arena;
pub mod hash;
pub mod id;
pub mod node;
pub mod op;
pub mod snapshot;
pub mod text;
pub mod ty;

/// Semantic version of the IR contract.
///
/// CEP:WHAT: IR format version constant.
/// CEP:WHY: CEP&CC 38.17 requires a versioned IR; the text format and the
///          fingerprint scheme both embed this number, so incompatible
///          changes bump it and old artifacts fail loudly.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: text format emits "xir v1"; snapshot fingerprints hash ops.
pub const IR_VERSION: u32 = 1;

/// Compiler version embedded in pipeline manifests.
///
/// CEP:WHAT: Workspace version string.
/// CEP:WHY: CEP&CC 38.10: deterministic translation is qualified by compiler
///          version; manifests carry it.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: matches Cargo.toml workspace.version.
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: tools print it in --version.
pub const COMPILER_VERSION: &str = "sxla 0.1.0";
