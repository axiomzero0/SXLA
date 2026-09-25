// CEP:FILE: crates/anvil/src/lib.rs
// CEP:WHAT: Anvil — the SXLA custom thread-per-core, NUMA-aware, lock-free execution engine.
// CEP:WHY: General-purpose runtimes (tokio, rayon) impose state machines, global allocator
//          contention, and work-stealing fences that are unacceptable for compiler workloads
//          (master architecture section 2). Anvil shifts gears based on pass topology:
//          Gear 1 static partitioning (zero-sync), Gear 2 fork/join + work stealing with task
//          fuel, Gear 3 SPSC lock-free pipelines, Gear 4 epoch-based reclamation.
// CEP:CLASS: CEP-0 (primitives) / CEP-1 (executor orchestration)
// CEP:STATUS: partial
// CEP:FAILURE: Every primitive returns explicit error enums (ArenaError, DequeError, QueueError,
//              ShardError); no panics, no unwinding from public hot APIs.
// CEP:ASSUMES: All assumptions centralized in `config`; cache-line floor justified there.
// CEP:COST: see per-module CEP:COST fields; primitives are allocation-free after init.
// CEP:EVIDENCE: unit tests in each module; workspace test suite; benches/anvil_bench.rs.
// CEP:SECURITY: unsafe code confined to chase_lev/spsc/ebr/sharded with CEP:UNSAFE blocks;
//               unsafe blocks are bounded by checked indices and executor lifetime proofs.
// CEP:HPC-CLASS: HPC-0 (lock-free fast paths), HPC-1 (thread orchestration)
// CEP:HPC-DETERMINISM: deterministic; no hash-order or time dependence in any primitive.
// CEP:TODO(main-agent): CEP-2: NUMA-aware allocation policies are a future
//                       extension of config (the persistent pool landed as
//                       pool.rs — CEP-3).
//! # Anvil concurrency engine
//!
//! Module layout follows CEP&CC 32.2 hot/cold separation as closely as Rust crate
//! boundaries permit: `bump`, `chase_lev`, `spsc`, `ebr`, `sharded`, `fuel` are CEP-0
//! hot primitives (import `core::` exclusively, allocation-free after initialization);
//! `executor` and `telemetry` are CEP-1 orchestration (import `std::` for threads).

pub mod bump;
pub mod chase_lev;
pub mod config;
pub mod ebr;
pub mod executor;
pub mod fuel;
pub mod pad;
pub mod pool;
pub mod sharded;
pub mod spsc;
pub mod telemetry;

/// Re-exported gear entry points and shared types.
pub use executor::{
    default_worker_count, run_fork_join, run_fork_join_scoped, run_partitioned,
    run_partitioned_scoped, Job, WorkerCtx,
};
pub use pool::{
    global_pool, in_pooled_region, run_fork_join_pooled, run_partitioned_pooled, Pool, PoolError,
};
