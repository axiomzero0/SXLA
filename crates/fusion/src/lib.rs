// CEP:FILE: crates/fusion/src/lib.rs
// CEP:WHAT: fusion — the massive fusion infrastructure: legality engine,
//           resource model, tiered cost model, parallel universe search with
//           atomic pruning, and fusion repair passes.
// CEP:WHY: Master architecture section 5: "Fusion is not a single pass; it
//          is a massive, parallelized search subsystem that dictates the
//          shape of the IR." Multiple Fusion Universes run concurrently on
//          Anvil Gear 2; a global AtomicU64 tracks best_cost and workers
//          abandon branches that cannot beat it.
// CEP:CLASS: CEP-0 (search core) / CEP-1 (driver)
// CEP:STATUS: partial
// CEP:FAILURE: FusionError codes; conservative aborts; never a silent
//             illegal fusion (HPC prime law).
// CEP:ASSUMES: verified Level-1 snapshots; tensor types attached.
// CEP:COST: search O(universes * candidates) with pruning; legality O(1)
//           per candidate pair; resource estimation O(cluster size).
// CEP:EVIDENCE: per-module tests; differential integration tests.
// CEP:SECURITY: IR untrusted; all checks explicit.
// CEP:HPC-CLASS: HPC-0 (search), HPC-1 (driver).
// CEP:HPC-DETERMINISM: deterministic — universes evaluate pure scoring
//           functions; the winner is chosen by (cost, universe priority),
//           never by completion order.
// CEP:TODO(main-agent): CEP-19: full Presburger/ISL constraint engine
//           (current legality checks are the affine-compatible subset).
//! # fusion
//!
//! The fusion search subsystem.

pub mod cost;
pub mod legality;
pub mod repair;
pub mod resource;
pub mod search;

pub use cost::{CostModel, Tier};
pub use legality::{can_fuse, LegalityError};
pub use repair::{repair_cluster, RepairAction};
pub use resource::{resource_estimate, ResourceEstimate, SHARED_MEM_BUDGET_BYTES};
pub use search::{search, SearchOutcome, Universe};
