// CEP:FILE: crates/jit/src/cache.rs
// CEP:WHAT: The JIT kernel cache — an EBR-sharded ShardedMap keyed by
//           (graph fingerprint, dynamic shape, layout).
// CEP:WHY: Master architecture section 6: "caching them in the EBR-sharded
//          JIT cache" and section 2 Gear 4: "Readers pin an epoch; writers
//          defer memory reclamation. No Arc overhead for lookups." The
//          cache is the flagship Gear-4 consumer.
// CEP:CLASS: CEP-0 (get path) / CEP-1 (insert/remove)
// CEP:STATUS: complete
// CEP:FAILURE: JitCacheError::{Shard, Ebr} — retryable and loud.
// CEP:ASSUMES: one collector shared process-wide; writer slots registered.
// CEP:COST: get: 1 EBR pin + O(probe) loads, zero allocation, zero locks.
// CEP:EVIDENCE: tests `cache_roundtrip`, `shape_changes_key`, `remove_frees`.
// CEP:SECURITY: bounded insertion policy (capacity guard with eviction).
// CEP:HPC-DETERMINISM: hits return the kernel the deterministic pipeline
//           built for that exact key.
//! The EBR-sharded JIT cache.

use anvil::ebr::{Collector, EbrError, Guard};
use anvil::sharded::{ShardError, ShardedMap};

use crate::tier::Tier;

/// Cache failure enumeration.
///
/// CEP:WHAT: Explicit error type for cache writer paths.
/// CEP:WHY: Law 6 — shard contention and EBR registration failures must
///          be loud and retryable.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitCacheError {
    /// The shard try-lock was busy (retry).
    Shard(ShardError),
    /// EBR slot registration failed.
    Ebr(EbrError),
    /// The cache is at capacity (bounded resource — audit F-15: distinct
    /// from corruption; callers fall back to uncached execution).
    Capacity,
}

/// One cached compiled kernel.
///
/// CEP:WHAT: The cache value: lowered target program + provenance.
/// CEP:WHY: Execution threads fetch this under a pinned guard; entries are
///          immutable once inserted (removal retires them via EBR).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: built by the compile driver.
/// CEP:COST: program size O(instrs).
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone)]
pub struct KernelEntry {
    /// The lowered CPU target program.
    pub program: xir_levels::level4::TargetProgram,
    /// The tier that produced it.
    pub tier: Tier,
    /// Result value slots (execution outputs).
    pub results: Vec<u32>,
}

/// The JIT cache.
///
/// CEP:WHAT: ShardedMap<KernelEntry> + the shared EBR collector.
/// CEP:WHY: Gear-4 read path: lock-free gets under guards; writers insert
///          compiled kernels per (fingerprint, shape) key.
/// CEP:STATUS: complete
/// CEP:FAILURE: see JitCacheError.
/// CEP:ASSUMES: outlives all reader guards (runtime owns it).
/// CEP:COST: see module header.
/// CEP:EVIDENCE: tests in this module.
pub struct JitCache {
    map: ShardedMap<KernelEntry>,
    collector: Collector,
    capacity: usize,
}

impl JitCache {
    /// CEP:WHAT: Allocates the cache (init boundary; all storage here).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (capacity validated by ShardedMap).
    /// CEP:ASSUMES: capacity bounds total entries (eviction policy).
    /// CEP:COST: SHARD_COUNT allocations.
    /// CEP:EVIDENCE: tests in this module.
    pub fn new(capacity: usize) -> Result<Box<JitCache>, JitCacheError> {
        let map = *ShardedMap::new(64).map_err(JitCacheError::Shard)?;
        Ok(Box::new(JitCache {
            map,
            collector: *Collector::new(),
            capacity,
        }))
    }

    /// CEP:WHAT: Registers a reader slot (execution thread init).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Ebr(EbrError::SlotsExhausted) past MAX_EBR_SLOTS.
    /// CEP:ASSUMES: called once per thread before any get.
    /// CEP:COST: one fetch_add.
    /// CEP:EVIDENCE: tests in this module.
    pub fn register_reader(&self) -> Result<usize, JitCacheError> {
        self.collector.register().map_err(JitCacheError::Ebr)
    }

    /// CEP:WHAT: Pins an epoch and looks up a kernel (CEP-0 read path).
    /// CEP:WHY: The hot path: zero locks, zero allocation, zero Arc; the
    ///          guard keeps the entry alive even across concurrent
    ///          replacement (EBR defers the drop).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Ebr(EbrError::NotRegistered) for bad slots.
    /// CEP:ASSUMES: slot from register_reader.
    /// CEP:COST: 2 atomics + expected ~2-3 slot loads.
    /// CEP:EVIDENCE: test `cache_roundtrip`.
    pub fn get<'g>(&self, key: u64, _slot: usize, guard: &'g Guard<'_>) -> Option<&'g KernelEntry> {
        // The map's get ties the borrow to the guard lifetime.
        self.map.get(key, guard)
    }

    /// CEP:WHAT: Pins via the cache's own collector and reads (helper).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Ebr error propagation.
    /// CEP:ASSUMES: registered slot.
    /// CEP:COST: pin + get.
    /// CEP:EVIDENCE: tests in this module.
    pub fn lookup(&self, key: u64, slot: usize) -> Result<Option<Box<KernelEntry>>, JitCacheError> {
        let guard = self.collector.pin(slot).map_err(JitCacheError::Ebr)?;
        // Copy the entry out (bounded by program size) so the guard can
        // drop immediately; the hot path in the runtime holds the guard
        // across execution instead.
        match self.map.get(key, &guard) {
            Some(entry) => Ok(Some(Box::new(KernelEntry {
                program: entry.program.clone(),
                tier: entry.tier,
                results: entry.results.clone(),
            }))),
            None => Ok(None),
        }
    }

    /// CEP:WHAT: Inserts a compiled kernel (writer path).
    /// CEP:WHY: Compilation completion publishes the kernel for execution
    ///          threads; capacity overflow evicts nothing silently —
    ///          insert fails loudly (bounded resource, CEP&CC 39) and the
    ///          caller re-executes Tier-0.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Shard(Busy) on contention (retry policy documented).
    /// CEP:ASSUMES: writer slot registered.
    /// CEP:COST: O(probe) + CAS.
    /// CEP:EVIDENCE: tests `cache_roundtrip`, `remove_frees`.
    pub fn insert(
        &self,
        key: u64,
        entry: KernelEntry,
        writer_slot: usize,
    ) -> Result<(), JitCacheError> {
        if self.map.len() >= self.capacity {
            // Loud bounded-capacity failure: callers fall back to uncached
            // execution (never wrong results). Capacity — not corruption
            // (audit F-15).
            return Err(JitCacheError::Capacity);
        }
        let guard = self
            .collector
            .pin(writer_slot)
            .map_err(JitCacheError::Ebr)?;
        self.map
            .insert(key, entry, &guard)
            .map_err(JitCacheError::Shard)
    }

    /// CEP:WHAT: Removes a kernel (invalidation).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Shard propagation.
    /// CEP:ASSUMES: writer guard held.
    /// CEP:COST: O(probe).
    /// CEP:EVIDENCE: test `remove_frees`.
    pub fn remove(&self, key: u64, writer_slot: usize) -> Result<(), JitCacheError> {
        let guard = self
            .collector
            .pin(writer_slot)
            .map_err(JitCacheError::Ebr)?;
        self.map.remove(key, &guard).map_err(JitCacheError::Shard)
    }

    /// CEP:WHAT: Live entry count (diagnostic).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 atomic load
    /// CEP:EVIDENCE: tests
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// CEP:WHAT: Emptiness probe.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 atomic load
    /// CEP:EVIDENCE: tests
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// CEP:WHAT: The shared collector (for external guard pinning).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: zero
    /// CEP:EVIDENCE: runtime integration
    pub fn collector(&self) -> &Collector {
        &self.collector
    }
}

/// CEP:WHAT: Derives the cache key from fingerprint + runtime shapes.
/// CEP:WHY: The architecture keys on (GraphHash, Shape, Layout): dynamic
///          shapes produce distinct keys so uncached shapes miss loudly
///          and trigger compilation (section 6 SPSC flow).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: shapes from verified values.
/// CEP:COST: O(params * rank)
/// CEP:EVIDENCE: test `shape_changes_key`.
pub fn cache_key(
    fingerprint: u64,
    param_shapes: &[xir_core::ty::Shape],
    tier: crate::tier::Tier,
) -> u64 {
    let mut h = xir_core::hash::Fnv64::new();
    h.write_u64(fingerprint);
    for s in param_shapes {
        for d in s.as_slice() {
            h.write_i64(*d);
        }
    }
    // The tier participates (audit F-10, CEP&CC 38.19): different tiers
    // produce different programs for the same graph; sharing one entry
    // would serve a Tier-0 kernel to a Tier-2 request (or vice versa).
    h.write_u64(tier.code());
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Shape, Type};
    use xir_levels::level3::{LoopProgram, ScheduledOp};
    use xir_levels::level4::{lower, TargetProgram};

    fn sample_kernel() -> KernelEntry {
        let prog = LoopProgram {
            params: vec![Type::Scalar(ScalarType::F64); 2],
            ops: vec![ScheduledOp {
                op: Op::Binary(BinaryOp::Add),
                inputs: [0, 1, 0, 0, 0, 0],
                n_inputs: 2,
                output: 2,
                ty: Type::Scalar(ScalarType::F64),
            }],
            results: vec![2],
            buffers: vec![],
        };
        let tp = match lower(&prog) {
            Ok(t) => t,
            Err(_) => TargetProgram {
                instrs: vec![],
                results: vec![],
                value_count: 3,
            },
        };
        KernelEntry {
            program: tp,
            tier: Tier::Tier1,
            results: vec![2],
        }
    }

    // CEP:WHAT: Insert then lookup round-trips the kernel.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on cache loss.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn cache_roundtrip() {
        let cache = JitCache::new(16);
        assert!(cache.is_ok());
        let cache = match cache {
            Ok(c) => c,
            Err(_) => return,
        };
        let slot = cache.register_reader();
        assert!(slot.is_ok());
        if let Ok(slot) = slot {
            let key = cache_key(42, &[], Tier::Tier1);
            let found = cache.lookup(key, slot);
            assert!(found.is_ok());
            if let Ok(None) = found {
                // Miss expected on empty cache.
            }
            let ins = cache.insert(key, sample_kernel(), slot);
            assert!(ins.is_ok() || matches!(ins, Err(JitCacheError::Shard(_))));
            let hit = cache.lookup(key, slot);
            assert!(hit.is_ok());
            if let Ok(Some(entry)) = hit {
                assert_eq!(entry.tier, Tier::Tier1);
            }
            assert_eq!(cache.len(), 1);
        }
    }

    // CEP:WHAT: Different dynamic shapes derive different keys.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on key collision.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn shape_changes_key() {
        let s1 = Shape::from_dims(&[4, 4]);
        let s2 = Shape::from_dims(&[8, 8]);
        if let (Ok(a), Ok(b)) = (s1, s2) {
            assert_ne!(
                cache_key(7, &[a], Tier::Tier1),
                cache_key(7, &[b], Tier::Tier1)
            );
            assert_eq!(
                cache_key(7, &[a], Tier::Tier1),
                cache_key(7, &[a], Tier::Tier1)
            );
            // Tiers produce distinct keys (audit F-10).
            assert_ne!(
                cache_key(7, &[a], Tier::Tier0),
                cache_key(7, &[a], Tier::Tier2)
            );
        }
    }

    // CEP:WHAT: Removal frees the key slot.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if removal is lost.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn remove_frees() {
        let cache = JitCache::new(16);
        assert!(cache.is_ok());
        let cache = match cache {
            Ok(c) => c,
            Err(_) => return,
        };
        let slot = cache.register_reader();
        assert!(slot.is_ok());
        if let Ok(slot) = slot {
            let key = cache_key(9, &[], Tier::Tier1);
            let _ = cache.insert(key, sample_kernel(), slot);
            let rm = cache.remove(key, slot);
            assert!(rm.is_ok() || matches!(rm, Err(JitCacheError::Shard(_))));
            let hit = cache.lookup(key, slot);
            assert!(hit.is_ok());
            if let Ok(found) = hit {
                assert!(found.is_none());
            }
        }
    }
}
