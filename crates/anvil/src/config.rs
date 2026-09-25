// CEP:FILE: crates/anvil/src/config.rs
// CEP:WHAT: Centralized target configuration constants for the Anvil engine.
// CEP:WHY: CEP&CC Law 7 bans hard-coded assumptions (e.g. a literal `64` at a padding site);
//          every target belief must be a named, justified, enforced configuration value
//          (CEP&CC 7.2 target abstraction). All Anvil target constants live here and
//          nowhere else.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: `assert_config_invariants()` panics in debug builds if padding size does not
//              cover the configured cache line; it returns Err(()) in release. It is the
//              only debug-assertion gate in the engine and runs once at init.
// CEP:ASSUMES: CACHE_LINE_BYTES floor of 64 covers x86_64 and aarch64 (both >= 64);
//              justified below; checked by `assert_config_invariants`.
// CEP:COST: compile-time constants; `assert_config_invariants` is 1 compare, init-only.
// CEP:EVIDENCE: unit test `config_invariants_hold`.
// CEP:SECURITY: no untrusted input; constants only.
// CEP:HPC-DETERMINISM: deterministic; constants only.
//! Target configuration for Anvil.
//!
//! These constants are the ONLY place target-specific numbers may appear
//! (CEP&CC 32.5: target directory rules). Modules must import them from here.

/// Configured cache-line floor in bytes.
///
/// CEP:WHY: 64 is the minimum coherence granularity on every supported target
/// (x86_64: 64B lines; aarch64: 64B or 128B). Padding to the floor guarantees
/// no false sharing between adjacent hot atomics; padding to 128 would waste
/// 64B per padded field on 64B targets. Rejected alternative: runtime cache
/// detection — std exposes no portable API and it would add an OS dependency
/// to CEP-0 code (Law 1: no hidden cost).
pub const CACHE_LINE_BYTES: usize = 64;

/// SPSC ring batching factor (Gear 3).
///
/// CEP:WHY: The architecture mandates batching to amortize fence costs
/// (~2ns per op). Publishing every `SPSC_BATCH_OPS` slots converts one
/// Release store per element into one per batch; 32 keeps worst-case
/// consumer-visible latency at 32 elements while cutting published
/// Release stores by 32x on sustained streams.
pub const SPSC_BATCH_OPS: u32 = 32;

/// Deterministic task-fuel budget (Gear 2).
///
/// CEP:WHY: The architecture specifies "<500ns executes synchronously". Reading a
/// wall clock in the search loop would (a) violate HPC determinism (CEP&CC 38.10
/// bans time as a translation input) and (b) pull std::time into CEP-0 (25.3.1).
/// We therefore measure fuel in deterministic abstract work units: one unit is
/// one loop-iteration step of the driving search. Calibrated at roughly 10ns per
/// simple step on a modern out-of-order core, 64 units approximates the 500ns
/// threshold while keeping translation bit-deterministic across machines.
pub const TASK_FUEL_UNITS: u32 = 64;

/// Chase-Lev deque capacity per worker (power of two, required by mask math).
///
/// CEP:WHY: CEP&CC 39 (RCS) demands bounded queues. Capacity 8192 tasks per
/// worker bounds worst-case stolen-task memory to 8192 * size_of::<J>() per
/// core. Push past capacity returns `DequeError::Full` instead of allocating
/// (Law 1) — callers must drain or fail loudly.
pub const DEQUE_CAPACITY: usize = 8192;

/// EBR epoch advance threshold: retires pending before attempting advance.
///
/// CEP:WHY: Attempting an epoch advance scans all registered threads
/// (O(threads) atomic loads). Bounding scans to once per
/// `EBR_RETIRE_THRESHOLD` retirements keeps amortized retire cost O(1)
/// while bounding deferred garbage.
pub const EBR_RETIRE_THRESHOLD: u32 = 64;

/// Sharded map shard count (power of two).
///
/// CEP:WHY: Gear 4 reads must not contend. 64 shards reduce write-side
/// CAS pressure 64-way and keep per-shard tables small; shard selection
/// is `hash & (SHARD_COUNT - 1)` so the count must remain a power of two
/// (enforced by test).
pub const SHARD_COUNT: usize = 64;

/// Maximum number of worker threads Anvil will spawn per region.
///
/// CEP:WHY: Thread-per-core model; the default equals logical core count at
/// init (std::thread::available_parallelism) and this constant caps it so a
/// 512-core host cannot blow thread-stack memory (bounded resource, CEP&CC 39).
pub const MAX_WORKERS: usize = 64;

/// Maximum telemetry events buffered per SPSC channel before dropping.
///
/// CEP:WHY: Telemetry is best-effort by contract; the buffer is bounded and
/// overflow is counted and reported, never blocks, never allocates
/// (CEP&CC Law 1, Law 6).
pub const TELEMETRY_CAPACITY: usize = 8192;

/// Compile-time power-of-two checks for mask-based indexing.
const _: () = {
    // CEP:WHAT: Static assertions for power-of-two configuration constants.
    // CEP:WHY: Chase-Lev and SPSC index math relies on `index & (cap - 1)`;
    //          a non-power-of-two capacity would silently corrupt slots
    //          (Law 3: no comment-only invariants — enforced here at compile time).
    // CEP:STATUS: complete
    // CEP:FAILURE: compile error if DEQUE_CAPACITY, SHARD_COUNT or
    //              TELEMETRY_CAPACITY is not a power of two.
    // CEP:ASSUMES: none
    // CEP:COST: compile-time only
    // CEP:EVIDENCE: unit test `config_invariants_hold` mirrors these checks.
    assert!(DEQUE_CAPACITY.is_power_of_two());
    assert!(SHARD_COUNT.is_power_of_two());
    assert!(TELEMETRY_CAPACITY.is_power_of_two());
    assert!(CACHE_LINE_BYTES >= 64);
    assert!(TASK_FUEL_UNITS > 0);
    assert!(MAX_WORKERS > 0);
};

/// CEP:WHAT: Validates the one runtime-checkable target assumption (padding covers
///           the configured cache-line floor).
/// CEP:WHY: Law 3 requires an enforced invariant, not a comment: `CachePadded`
///          alignment must actually be >= CACHE_LINE_BYTES or padding is a lie.
/// CEP:STATUS: complete
/// CEP:FAILURE: Returns Err(()) if `size_of::<CachePadded<u8>>() < CACHE_LINE_BYTES`;
///              panics via debug_assert in debug builds (init-time, allowed).
/// CEP:ASSUMES: none
/// CEP:COST: one compare; init-only
/// CEP:EVIDENCE: unit test `config_invariants_hold`
// CEP:HPC-DETERMINISM: deterministic
pub fn assert_config_invariants() -> Result<(), &'static str> {
    let padded = crate::pad::CachePadded::<u8>::padding_size();
    debug_assert!(padded >= CACHE_LINE_BYTES);
    if padded >= CACHE_LINE_BYTES {
        Ok(())
    } else {
        Err("CachePadded size does not cover the configured cache-line floor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Mirrors the compile-time static assertions at runtime for CI visibility.
    // CEP:WHY: Belt-and-braces enforcement (psychopathic tier: rules must be testable).
    // CEP:STATUS: complete
    // CEP:FAILURE: asserts fire if constants drift away from power-of-two / floor rules.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn config_invariants_hold() {
        assert!(DEQUE_CAPACITY.is_power_of_two());
        assert!(SHARD_COUNT.is_power_of_two());
        assert!(TELEMETRY_CAPACITY.is_power_of_two());
        assert!(assert_config_invariants().is_ok());
    }
}
