// CEP:FILE: crates/egraph/src/lib.rs
// CEP:WHAT: egraph — the optional, partitioned equality-saturation engine for
//           Levels 0/1.
// CEP:WHY: Master architecture section 4: pure subgraphs are lifted into the
//          e-graph; saturation applies rewrite rules; extraction is
//          fusion-aware (penalizes rewrites that break producer-consumer
//          locality or force layout conversions). Saturation is optional and
//          tier-gated (Tier-2 only) per section 6.
// CEP:CLASS: CEP-0 (core structures) / CEP-1 (driver)
// CEP:STATUS: partial
// CEP:FAILURE: EgraphError codes; conservative aborts; no panics.
// CEP:ASSUMES: only PURE nodes are lifted (driver filters by op.is_pure()).
// CEP:COST: saturation O(iterations * classes * rules); extraction
//           O(classes + nodes).
// CEP:EVIDENCE: per-module tests; differential test against the unfused
//           interpreter in tests/.
// CEP:SECURITY: internal ids only.
// CEP:HPC-CLASS: HPC-0 (saturation core), HPC-1 (driver).
// CEP:HPC-DETERMINISM: deterministic: fixed rule order, ascending class
//           order, deterministic tie-breaks; the Gear-1 partition processes
//           disjoint slices and merges by index (scheduling-independent).
// CEP:TODO(main-agent): CEP-17: cross-worker merges via SPSC to a dedicated
//           union-find resolver thread (architecture's full sharded design);
//           the current Gear-1 partition covers local rules only.
//! # egraph
//!
//! Partitioned equality saturation with fusion-aware extraction.

pub mod apply;
pub mod egraph;
pub mod extract;
pub(crate) mod lift;
pub mod rules;
pub mod saturate;
pub mod union_find;

pub use apply::{apply, ApplyOutcome};
pub use egraph::{EGraph, EgraphError};
pub use extract::{extract, Extraction, FUSION_LOCALITY_PENALTY};
pub use saturate::saturate;
