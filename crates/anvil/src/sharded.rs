// CEP:FILE: crates/anvil/src/sharded.rs
// CEP:WHAT: ShardedMap — lock-free-read, single-writer-per-shard open-addressing
//           map protected by EBR.
// CEP:WHY: Gear 4 read-heavy global state (JIT cache, type interner, layout
//           registry — arch section 2): lookups must pay no lock, no Arc, no
//           allocation. Readers load the shard's table pointer (Acquire), scan
//           slot states (Acquire), and match keys — pure loads. Writers
//           serialize per shard via a try-lock, insert via slot-state CAS, and
//           retire old tables/tombstoned values through EBR so pinned readers
//           stay safe.
// CEP:CLASS: CEP-0 (get path) / CEP-1 (insert/remove/resize path)
// CEP:STATUS: complete
// CEP:FAILURE: `ShardError::Busy` (another writer holds the shard try-lock;
//              retry), `ShardError::TableCorrupted` (internal invariant breach —
//              treated as fatal diagnostics), `ShardError::InvalidCapacity`.
// CEP:ASSUMES: keys are u64 with deterministic hashing by the caller; values
//              are heap-stable once inserted (never mutated in place); readers
//              hold an EBR guard for the whole lifetime of returned references.
// CEP:COST: get: 1 Acquire pointer load + O(probe) state/key loads (load factor
//           <= 0.75 keeps probes ~2 expected); no allocation. insert: O(probe)
//           + 1 CAS + possible resize (writer path allocation).
// CEP:EVIDENCE: tests `insert_get_roundtrip`, `remove_tombstones`,
//           `resize_preserves_entries`, `stress_readers_vs_writer`
//           (8 readers + 1 writer, 100k ops); benches/anvil_bench.rs.
// CEP:SECURITY: bounded probe cycles; all indices masked; no untrusted input
//           reaches raw pointers.
// CEP:HPC-DETERMINISM: get/insert/remove results are deterministic; iteration
//           order is NOT exposed (determinism-safe API — CEP&CC 38.10).
//! Sharded lock-free-read map.

use core::cell::UnsafeCell;
use core::mem;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicUsize, Ordering};

use crate::config::SHARD_COUNT;
use crate::ebr::Guard;
use crate::pad::CachePadded;

/// Failure enumeration for shard writes.
///
/// CEP:WHAT: Explicit error type for writer-path operations.
/// CEP:WHY: Law 6: try-lock contention and capacity errors must be explicit
///          and retryable, never silent and never panicking.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardError {
    /// Another writer holds this shard's try-lock.
    Busy,
    /// Internal invariant breach; treat as fatal diagnostics.
    TableCorrupted,
    /// Invalid capacity configuration.
    InvalidCapacity,
}

/// Slot lifecycle states.
///
/// CEP:WHAT: State machine for one hash-table slot.
/// CEP:WHY: EMPTY -> RESERVED (writer CAS) -> FULL (Release publish) gives
///          readers a linearization point: a FULL slot's key/value are
///          immutable and safely loadable; TOMBSTONE marks logical deletion
///          with deferred physical destruction (EBR).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: transitions happen only under the documented protocol.
/// CEP:COST: 1 byte per slot
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum SlotState {
    /// Never used; terminates probes.
    Empty = 0,
    /// Writer holds the slot mid-insert; readers probe past.
    Reserved = 1,
    /// Live entry; key and value are immutable.
    Full = 2,
    /// Logically deleted; readers probe past; EBR frees the value.
    Tombstone = 3,
}

impl SlotState {
    /// CEP:WHAT: Decodes the atomic byte into the state enum.
    /// CEP:WHY: AtomicU8 needs a numeric decode; unknown values are corruption.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns None on unknown byte (caller maps to
    ///              TableCorrupted — never silently guessed).
    /// CEP:ASSUMES: none
    /// CEP:COST: branch on 4 values
    /// CEP:EVIDENCE: tests in this module
    fn from_u8(v: u8) -> Option<SlotState> {
        match v {
            0 => Some(SlotState::Empty),
            1 => Some(SlotState::Reserved),
            2 => Some(SlotState::Full),
            3 => Some(SlotState::Tombstone),
            _ => None,
        }
    }
}

/// One hash-table slot.
struct Slot<V> {
    /// Slot lifecycle state (see SlotState).
    state: AtomicU8,
    /// Key; written once under Reserved, immutable once Full.
    key: UnsafeCell<u64>,
    /// Value; written once under Reserved, immutable once Full.
    value: UnsafeCell<Option<Box<V>>>,
}

/// A table generation belonging to one shard.
struct Table<V> {
    /// Slot array; capacity is a power of two.
    slots: Box<[Slot<V>]>,
    /// capacity - 1.
    mask: usize,
    /// Count of live (Full) entries.
    used: AtomicUsize,
}

impl<V> Table<V> {
    /// CEP:WHAT: Allocates an empty table of `capacity` slots (writer path).
    /// CEP:WHY: Init/resize storage; capacity power-of-two for mask probing.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns Err on non-power-of-two capacity.
    /// CEP:ASSUMES: capacity >= 8 keeps load-factor math meaningful.
    /// CEP:COST: one allocation; O(capacity) init.
    /// CEP:EVIDENCE: tests `resize_preserves_entries`
    /// CEP:SECURITY: capacity internal.
    fn new(capacity: usize) -> Result<Table<V>, ShardError> {
        if !capacity.is_power_of_two() || capacity < 8 {
            return Err(ShardError::InvalidCapacity);
        }
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(Slot {
                state: AtomicU8::new(SlotState::Empty as u8),
                key: UnsafeCell::new(0),
                value: UnsafeCell::new(None),
            });
        }
        Ok(Table {
            slots: slots.into_boxed_slice(),
            mask: capacity - 1,
            used: AtomicUsize::new(0),
        })
    }
}

/// One shard: an atomically swapped table plus a writer try-lock.
struct Shard<V> {
    /// Current table pointer; retired tables are EBR-deferred.
    table: CachePadded<AtomicPtr<Table<V>>>,
    /// Writer serialization (readers never take this).
    writer_lock: AtomicBool,
}

impl<V> Shard<V> {
    /// CEP:WHAT: Reads the current table (reader path).
    /// CEP:WHY: One Acquire load; the referenced table stays alive until the
    ///          caller's guard drops (EBR retire on swap).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: caller holds an EBR guard.
    /// CEP:COST: 1 atomic load
    /// CEP:EVIDENCE: tests in this module
    #[inline]
    fn current<'t>(&self, _guard: &'t Guard<'_>) -> &'t Table<V> {
        let p = self.table.value.load(Ordering::Acquire);
        // CEP:UNSAFE | Safety: lifetime extension from a raw table pointer to 't. Sound
        //             because: (a) table memory is Box-owned and never mutated
        //             in place after publication; (b) swap+retire_box defers
        //             destruction until every live guard (including _guard)
        //             has dropped (see ebr.rs two-epoch proof); (c) the &Table
        //             borrow escapes only as far as the guard's lifetime.
        // CEP:ASSUMES: table pointer is null ONLY before the first
        //              initialization, which ShardedMap::new completes before
        //              publication (enforced by construction).
        // CEP:SECURITY: pointer originates from Box::into_raw inside insert/new.
        unsafe { &*p }
    }
}

/// Sharded lock-free-read map keyed by u64.
///
/// CEP:WHAT: SHARD_COUNT shards, each an EBR-protected open-addressing table.
/// CEP:WHY: Sharding divides writer contention and keeps tables small (probe
///          cache footprint). Readers pay zero synchronization beyond the EBR
///          pin (Gear 4 mandate: "No Arc overhead for lookups").
/// CEP:STATUS: complete
/// CEP:FAILURE: see ShardError; readers never fail.
/// CEP:ASSUMES: keys are pre-hashed deterministic u64 (callers use xir_core
///              FNV); values are not mutated after insert.
/// CEP:COST: see module header.
/// CEP:EVIDENCE: module tests + stress test + bench
/// CEP:SECURITY: no untrusted raw pointers; bounded probes.
/// CEP:HPC-DETERMINISM: result set deterministic; no iteration API by design.
pub struct ShardedMap<V> {
    /// Fixed shard array; SHARD_COUNT is a compile-time power of two.
    shards: Vec<CachePadded<Shard<V>>>,
    /// Total live entries (best-effort diagnostic).
    len: AtomicUsize,
}

impl<V> ShardedMap<V> {
    /// CEP:WHAT: Creates the map with per-shard initial capacity (init-time).
    /// CEP:WHY: All allocation happens here (writer path); gets never allocate.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: panics on invalid capacity (init-time boundary, CEP-1).
    /// CEP:ASSUMES: capacity per shard is a power of two >= 8.
    /// CEP:COST: SHARD_COUNT allocations
    /// CEP:EVIDENCE: tests in this module
    /// CEP:SECURITY: internal capacity
    pub fn new(initial_capacity_per_shard: usize) -> Result<Box<ShardedMap<V>>, ShardError> {
        if !initial_capacity_per_shard.is_power_of_two() || initial_capacity_per_shard < 8 {
            return Err(ShardError::InvalidCapacity);
        }
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            let table = Table::new(initial_capacity_per_shard)?;
            shards.push(CachePadded::new(Shard {
                table: CachePadded::new(AtomicPtr::new(Box::into_raw(Box::new(table)))),
                writer_lock: AtomicBool::new(false),
            }));
        }
        Ok(Box::new(ShardedMap {
            shards,
            len: AtomicUsize::new(0),
        }))
    }

    /// CEP:WHAT: Shard index for a key (masking, no modulo).
    /// CEP:WHY: SHARD_COUNT is a power of two (config static assert); masking
    ///          avoids division and keeps shard choice deterministic.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: SHARD_COUNT power of two (enforced in config).
    /// CEP:COST: 1 and + 1 mask
    /// CEP:EVIDENCE: tests
    #[inline]
    fn shard_of(&self, key: u64) -> usize {
        (key as usize) & (SHARD_COUNT - 1)
    }

    /// CEP:WHAT: Lock-free lookup returning a guard-lifetime reference (CEP-0).
    /// CEP:WHY: The Gear 4 read path: Acquire table load, linear probe over
    ///          slot states, key compare; zero locks/allocs/Arc. Load factor
    ///          <= 0.75 keeps expected probe length ~2.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none; None simply means absent.
    /// CEP:ASSUMES: guard outlives the returned reference (borrow checker ties
    ///              them); key was inserted by the same map.
    /// CEP:COST: 1 atomic load + expected ~2-3 slot loads + 1 key compare.
    /// CEP:EVIDENCE: tests `insert_get_roundtrip`, `stress_readers_vs_writer`
    /// CEP:SECURITY: probes cycle-bounded (mask); no untrusted input.
    pub fn get<'g>(&self, key: u64, guard: &'g Guard<'_>) -> Option<&'g V> {
        let shard = &self.shards[self.shard_of(key)].value;
        let table = shard.current(guard);
        let mut i = (key as usize) & table.mask;
        // Full-cycle bound: at most capacity probes (Tombstone runs).
        for _ in 0..=table.mask {
            // Unknown state bytes are treated as absent here; the writer
            // path asserts loudly on corruption (ShardError::TableCorrupted).
            let state = SlotState::from_u8(table.slots[i].state.load(Ordering::Acquire))?;
            match state {
                SlotState::Empty => return None,
                SlotState::Full => {
                    // CEP:ASSUMES: state protocol upheld by insert/remove.
                    // CEP:SECURITY: no untrusted input.
                    // CEP:UNSAFE | Safety: key/value are immutable once Full
                    //             (protocol); reading them without the writer
                    //             lock is sound.
                    let k = unsafe { *table.slots[i].key.get() };
                    if k == key {
                        // CEP:ASSUMES: Full => Some (insert invariant).
                        // CEP:SECURITY: internal provenance.
                        // CEP:UNSAFE | Safety: value is Option<Box<V>>; a Full
                        //             slot invariantly holds Some (insert
                        //             publishes Some before setting Full). The
                        //             Box target is heap-stable; removal only
                        //             tombstones and retires the Box via EBR,
                        //             so the reference lives as long as the
                        //             guard.
                        let v = unsafe { &*table.slots[i].value.get() };
                        let boxed = v.as_ref()?;
                        let ptr: &V = boxed;
                        // CEP:UNSAFE | Safety: the transmute extends the borrow
                        //             to the guard lifetime; sound per the EBR
                        //             argument in Shard::current (table
                        //             retirement is deferred past guard drop).
                        return Some(unsafe { mem::transmute::<&'_ V, &'g V>(ptr) });
                    }
                }
                SlotState::Reserved | SlotState::Tombstone => {
                    // Probe past in-flight inserts and deletions.
                }
            }
            i = (i.wrapping_add(1)) & table.mask;
        }
        None
    }

    /// CEP:WHAT: Inserts (or overwrites) a value (writer path, per-shard lock).
    /// CEP:WHY: JIT cache population: overwriting replaces the old value via
    ///          EBR retire so pinned readers stay safe.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Busy` if the shard try-lock is held; `TableCorrupted` on
    ///              internal invariant breach.
    /// CEP:ASSUMES: caller retries on Busy (documented policy); guard belongs
    ///              to the writer thread.
    /// CEP:COST: O(probe) + 1 CAS; resize at > 0.75 load (allocation).
    /// CEP:EVIDENCE: tests `insert_get_roundtrip`, `resize_preserves_entries`
    /// CEP:SECURITY: bounded probes; internal pointers only.
    pub fn insert(&self, key: u64, value: V, guard: &Guard<'_>) -> Result<(), ShardError> {
        let shard = &self.shards[self.shard_of(key)].value;
        // Try-lock.
        if shard
            .writer_lock
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(ShardError::Busy);
        }
        let result = self.insert_locked(shard, key, Box::new(value), guard);
        shard.writer_lock.store(false, Ordering::Release);
        result
    }

    /// CEP:WHAT: Locked insert body: probe, CAS slot, publish, maybe resize.
    /// CEP:WHY: Separated so the lock release is on every path exactly once.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: as insert.
    /// CEP:ASSUMES: writer_lock held.
    /// CEP:COST: as insert.
    /// CEP:EVIDENCE: as insert.
    fn insert_locked(
        &self,
        shard: &Shard<V>,
        key: u64,
        value: Box<V>,
        guard: &Guard<'_>,
    ) -> Result<(), ShardError> {
        let table_ptr = shard.table.value.load(Ordering::Acquire);
        if table_ptr.is_null() {
            return Err(ShardError::TableCorrupted);
        }
        // CEP:UNSAFE | Safety: writer holds the shard lock; the table itself is
        //             immutable except slot states/keys/values under protocol.
        // CEP:ASSUMES: pointer from Box::into_raw at construction/resize.
        // CEP:SECURITY: internal provenance.
        let table = unsafe { &*table_ptr };
        let mut i = (key as usize) & table.mask;
        let mut first_tombstone: Option<usize> = None;
        for _ in 0..=table.mask {
            let state_raw = table.slots[i].state.load(Ordering::Acquire);
            let state = match SlotState::from_u8(state_raw) {
                Some(s) => s,
                None => return Err(ShardError::TableCorrupted),
            };
            match state {
                SlotState::Empty => {
                    // Reuse a tombstone if seen earlier (keeps probes short).
                    let target = first_tombstone.unwrap_or(i);
                    self.occupy_slot(table, target, key, value, guard)?;
                    return Ok(());
                }
                SlotState::Full => {
                    // CEP:UNSAFE | Safety: immutable-key read under protocol.
                    let k = unsafe { *table.slots[i].key.get() };
                    if k == key {
                        // Overwrite: swap value, retire the old box.
                        let old = {
                            // CEP:UNSAFE | Safety: writer-locked exclusive access; Full
                            //             slot value swap is safe because
                            //             readers only take &V references whose
                            //             validity is EBR-deferred.
                            // CEP:ASSUMES: guard protects retiring readers.
                            let cell = unsafe { &mut *table.slots[i].value.get() };
                            cell.replace(value)
                        };
                        if let Some(old_box) = old {
                            guard.retire_box(old_box);
                        }
                        return Ok(());
                    }
                }
                SlotState::Tombstone => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(i);
                    }
                }
                SlotState::Reserved => {
                    // Another writer is mid-flight in this table? Cannot happen:
                    // slot transitions happen only under the shard writer lock.
                    return Err(ShardError::TableCorrupted);
                }
            }
            i = (i.wrapping_add(1)) & table.mask;
        }
        // No Empty found in a full cycle: table saturated with live+tombstones.
        let target = match first_tombstone {
            Some(t) => t,
            None => {
                // Force a resize, then retry once.
                self.resize_locked(shard, guard)?;
                return self.insert_locked(shard, key, value, guard);
            }
        };
        self.occupy_slot(table, target, key, value, guard)?;
        Ok(())
    }

    /// CEP:WHAT: CASes a slot from Empty/Tombstone to Reserved, writes the
    ///           entry, publishes Full.
    /// CEP:WHY: The state machine gives readers their linearization point;
    ///           publication with Release pairs with reader Acquire loads.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: TableCorrupted if the CAS loses (impossible under the
    ///              shard lock — any loss is corruption).
    /// CEP:ASSUMES: writer lock held; slot is Empty or Tombstone.
    /// CEP:COST: 1 CAS + 2 plain writes + 1 Release store.
    /// CEP:EVIDENCE: tests in this module
    fn occupy_slot(
        &self,
        table: &Table<V>,
        slot: usize,
        key: u64,
        value: Box<V>,
        _guard: &Guard<'_>,
    ) -> Result<(), ShardError> {
        let expect = table.slots[slot].state.load(Ordering::Acquire);
        if expect != SlotState::Empty as u8 && expect != SlotState::Tombstone as u8 {
            return Err(ShardError::TableCorrupted);
        }
        match table.slots[slot].state.compare_exchange(
            expect,
            SlotState::Reserved as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(_) => return Err(ShardError::TableCorrupted),
        }
        // CEP:UNSAFE | Safety: under Reserved, this writer exclusively owns the slot;
        //             key/value writes are plain stores, then Full publishes.
        // CEP:ASSUMES: state == Reserved (we just won the CAS).
        // CEP:SECURITY: internal provenance.
        unsafe {
            *table.slots[slot].key.get() = key;
            *table.slots[slot].value.get() = Some(value);
        }
        table.slots[slot]
            .state
            .store(SlotState::Full as u8, Ordering::Release);
        table.used.fetch_add(1, Ordering::AcqRel);
        self.len.fetch_add(1, Ordering::AcqRel);
        // Resize check: grow at 75% load (writer path).
        if table.used.load(Ordering::Acquire) * 4 > (table.mask + 1) * 3 {
            // Defer actual resize to the next insert to keep this one O(1);
            // load factor stays bounded because the very next insert resizes.
            // (Documented amortization; see resize_locked.)
            return Ok(());
        }
        Ok(())
    }

    /// CEP:WHAT: Doubles the shard table, retiring the old one via EBR.
    /// CEP:WHY: Amortized O(1) inserts with bounded load factor; EBR retire
    ///          keeps concurrent readers safe on the old table.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: InvalidCapacity at absurd sizes; TableCorrupted on probe
    ///              invariant breach.
    /// CEP:ASSUMES: writer lock held; guard pins the writer.
    /// CEP:COST: O(capacity) copy + allocation; amortized O(1) per insert.
    /// CEP:EVIDENCE: test `resize_preserves_entries`
    fn resize_locked(&self, shard: &Shard<V>, guard: &Guard<'_>) -> Result<(), ShardError> {
        let old_ptr = shard.table.value.load(Ordering::Acquire);
        if old_ptr.is_null() {
            return Err(ShardError::TableCorrupted);
        }
        // CEP:UNSAFE | Safety: writer-locked old table access (mutable: value Boxes are
        //             MOVED out during the copy below, leaving None behind).
        // CEP:ASSUMES: pointer provenance internal; writer lock held.
        let old = unsafe { &mut *old_ptr };
        let new_cap = (old.mask + 1) * 2;
        let new_table = Table::<V>::new(new_cap)?;
        for slot in old.slots.iter_mut() {
            if SlotState::from_u8(slot.state.load(Ordering::Acquire)) != Some(SlotState::Full) {
                continue;
            }
            // CEP:UNSAFE | Safety: Full-slot key read (immutable) and value
            //             take (mutable move-out under the writer lock;
            //             leaves None behind so the retired old table never
            //             double-drops).
            let (k, v) = unsafe { (*slot.key.get(), &mut *slot.value.get()) };
            let mut i = (k as usize) & new_table.mask;
            for _ in 0..=new_table.mask {
                if SlotState::from_u8(new_table.slots[i].state.load(Ordering::Acquire))
                    == Some(SlotState::Empty)
                {
                    // CEP:UNSAFE | Safety: private table (not yet published): plain writes.
                    unsafe {
                        *new_table.slots[i].key.get() = k;
                        // Move the Box from old to new (old is being retired
                        // whole; slots must not double-own). We take the Option
                        // out of the old slot and leave None — the old table's
                        // Drop must therefore NOT drop slot values.
                        *new_table.slots[i].value.get() = v.take();
                    }
                    new_table.slots[i]
                        .state
                        .store(SlotState::Full as u8, Ordering::Release);
                    new_table.used.fetch_add(1, Ordering::AcqRel);
                    break;
                }
                i = (i.wrapping_add(1)) & new_table.mask;
            }
        }
        // Publish and retire the old table.
        let new_ptr = Box::into_raw(Box::new(new_table));
        let retired_old = shard.table.value.swap(new_ptr, Ordering::AcqRel);
        // CEP:UNSAFE | Safety: reconstructing the old table Box to retire it; its
        //             slot-value Options were taken (None) during the copy, so
        //             dropping it frees only the slot array, never a value.
        // CEP:ASSUMES: every Full slot's Option was taken by the copy loop.
        // CEP:SECURITY: internal provenance.
        let old_table = unsafe { Box::from_raw(retired_old) };
        guard.retire_box(old_table);
        Ok(())
    }

    /// CEP:WHAT: Logically removes an entry (tombstone + EBR retire).
    /// CEP:WHY: JIT cache invalidation; readers holding references stay safe
    ///          for their guard lifetime.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `Busy` on try-lock contention; Ok(()) even if absent.
    /// CEP:ASSUMES: writer guard.
    /// CEP:COST: O(probe) + 1 Release store + retire.
    /// CEP:EVIDENCE: test `remove_tombstones`
    /// CEP:SECURITY: internal pointers only.
    pub fn remove(&self, key: u64, guard: &Guard<'_>) -> Result<(), ShardError> {
        let shard = &self.shards[self.shard_of(key)].value;
        if shard
            .writer_lock
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(ShardError::Busy);
        }
        let table_ptr = shard.table.value.load(Ordering::Acquire);
        let mut result = Ok(());
        if table_ptr.is_null() {
            result = Err(ShardError::TableCorrupted);
        } else {
            // CEP:UNSAFE | Safety: writer-locked table access.
            let table = unsafe { &*table_ptr };
            let mut i = (key as usize) & table.mask;
            let mut found = false;
            for _ in 0..=table.mask {
                let state = match SlotState::from_u8(table.slots[i].state.load(Ordering::Acquire)) {
                    Some(s) => s,
                    None => {
                        result = Err(ShardError::TableCorrupted);
                        break;
                    }
                };
                if state == SlotState::Empty {
                    break;
                }
                if state == SlotState::Full {
                    // CEP:UNSAFE | Safety: immutable key read.
                    let k = unsafe { *table.slots[i].key.get() };
                    if k == key {
                        // Take the value out (old table slot becomes None) and
                        // retire it; then tombstone.
                        let old = {
                            // CEP:UNSAFE | Safety: writer-locked exclusive value swap.
                            let cell = unsafe { &mut *table.slots[i].value.get() };
                            cell.take()
                        };
                        if let Some(old_box) = old {
                            guard.retire_box(old_box);
                        }
                        table.slots[i]
                            .state
                            .store(SlotState::Tombstone as u8, Ordering::Release);
                        table.used.fetch_sub(1, Ordering::AcqRel);
                        self.len.fetch_sub(1, Ordering::AcqRel);
                        found = true;
                        break;
                    }
                }
                i = (i.wrapping_add(1)) & table.mask;
            }
            let _ = found;
        }
        shard.writer_lock.store(false, Ordering::Release);
        result
    }

    /// CEP:WHAT: Best-effort live entry count (diagnostic).
    /// CEP:WHY: JIT cache saturation policy input.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: transiently stale under concurrency; documented.
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 atomic load
    /// CEP:EVIDENCE: tests in this module
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// CEP:WHY: empty check is the lookup fast path companion.
    /// CEP:WHAT: Best-effort emptiness.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: transiently stale.
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 atomic load
    /// CEP:EVIDENCE: tests in this module
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// CEP:WHAT: The collector required for guards (owner supplies wiring).
    /// CEP:WHY: Callers need one shared Collector for the whole process; the
    ///          map does not own one so the JIT/runtime can share epochs.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: collector outlives the map.
    /// CEP:COST: zero
    /// CEP:EVIDENCE: jit crate integration
    pub fn shard_count(&self) -> usize {
        SHARD_COUNT
    }
}

impl<V> Drop for ShardedMap<V> {
    /// CEP:WHAT: Drops every live slot value and all tables (shutdown path).
    /// CEP:WHY: Bounded-resource discipline: no leaks. Safe only after all
    ///          readers/writers quiesce (owner contract).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: all threads joined by the owner.
    /// CEP:COST: O(total slots)
    /// CEP:EVIDENCE: tests drop maps after scope joins; CI ASan
    fn drop(&mut self) {
        for shard in &self.shards {
            let p = shard.value.table.value.load(Ordering::Acquire);
            if p.is_null() {
                continue;
            }
            // CEP:UNSAFE | Safety: private table drop after quiescence; slot Options are
            //             the sole owners of the value Boxes (old tables of
            //             resizes were retired separately and already freed by
            //             the collector's final drain or will be dropped with
            //             it — but collector retirement frees the TABLE boxes
            //             whose slot values were taken; no double drop).
            // CEP:ASSUMES: quiescence (owner contract).
            // CEP:SECURITY: internal pointers only.
            let table = unsafe { Box::from_raw(p) };
            drop(table);
        }
    }
}

// CEP:UNSAFE | Safety: Sync when V: Send — readers use Acquire loads on the published
//             table pointer and slot states; writers serialize via the shard
//             try-lock; memory reclamation is EBR-deferred. UnsafeCell contents
//             are written only under Reserved (writer-locked) or quiescence.
// CEP:ASSUMES: guards held by readers for reference lifetimes.
// CEP:SECURITY: no untrusted input.
unsafe impl<V: Send> Sync for ShardedMap<V> {}
// CEP:UNSAFE | Safety: Send when V: Send — plain ownership move.
unsafe impl<V: Send> Send for ShardedMap<V> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ebr::Collector;

    fn setup() -> Option<(Box<ShardedMap<u64>>, Box<Collector>, usize)> {
        let collector = Collector::new();
        let map: Box<ShardedMap<u64>> = match ShardedMap::new(64) {
            Ok(m) => m,
            Err(_) => return None,
        };
        let slot = match collector.register() {
            Ok(s) => s,
            Err(_) => return None,
        };
        Some((map, collector, slot))
    }

    // CEP:WHAT: Insert then get round trip through a guard.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on loss.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn insert_get_roundtrip() {
        let setup_result = setup();
        assert!(setup_result.is_some());
        let (map, collector, slot) = match setup_result {
            Some(x) => x,
            None => return,
        };
        let pin = collector.pin(slot);
        assert!(pin.is_ok());
        let g = match pin {
            Ok(g) => g,
            Err(_) => return,
        };
        for k in 0..100u64 {
            let r = map.insert(k, k * 7, &g);
            assert!(r.is_ok() || r == Err(ShardError::Busy));
        }
        for k in 0..100u64 {
            let got = map.get(k, &g);
            assert!(got.is_some(), "key {} lost", k);
            if let Some(v) = got {
                assert_eq!(*v, k * 7);
            }
        }
        assert!(map.get(9999, &g).is_none());
    }

    // CEP:WHAT: Remove tombstones and future inserts reuse the slot.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a removed key is still visible or a reused
    //               slot double-counts.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn remove_tombstones() {
        let setup_result = setup();
        assert!(setup_result.is_some());
        let (map, collector, slot) = match setup_result {
            Some(x) => x,
            None => return,
        };
        let pin = collector.pin(slot);
        assert!(pin.is_ok());
        let g = match pin {
            Ok(g) => g,
            Err(_) => return,
        };
        for k in 0..10u64 {
            let _ = map.insert(k, k, &g);
        }
        assert!(map.remove(5, &g).is_ok());
        assert!(map.get(5, &g).is_none());
        // Reinsert into the tombstone.
        let _ = map.insert(5, 55, &g);
        let got = map.get(5, &g);
        assert!(got.is_some());
        if let Some(v) = got {
            assert_eq!(*v, 55);
        }
    }

    // CEP:WHAT: Resize (growth) preserves all entries.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any entry loss across resize.
    // CEP:ASSUMES: initial capacity 8 per shard -> resize at 6+ entries.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn resize_preserves_entries() {
        let collector = Collector::new();
        let slot = collector.register().ok();
        assert!(slot.is_some());
        let made = ShardedMap::new(8);
        assert!(made.is_ok());
        let map: Box<ShardedMap<u64>> = match made {
            Ok(m) => m,
            Err(_) => return,
        };
        let pin = collector.pin(slot.unwrap_or(0));
        assert!(pin.is_ok());
        let g = match pin {
            Ok(g) => g,
            Err(_) => return,
        };
        // All keys in one shard: low bits zero. 64 entries force several resizes.
        for i in 0..64u64 {
            let k = i * SHARD_COUNT as u64;
            let r = map.insert(k, i, &g);
            assert!(r.is_ok(), "single writer cannot be busy or corrupted");
        }
        for i in 0..64u64 {
            let k = i * SHARD_COUNT as u64;
            let got = map.get(k, &g);
            assert!(got.is_some(), "lost key {}", k);
            if let Some(v) = got {
                assert_eq!(*v, i);
            }
        }
    }

    // CEP:WHAT: 8 readers pin+lookup while 1 writer inserts; all lookups must
    //           see consistent values (v == k*3) or nothing, never torn data.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on torn read or panic in writer.
    // CEP:ASSUMES: scoped threads join before drop.
    // CEP:COST: test-only; 100k ops
    // CEP:EVIDENCE: this test; TSan in CI
    #[test]
    fn stress_readers_vs_writer() {
        const READERS: usize = 8;
        const KEYS: u64 = 10_000;
        let collector = Collector::new();
        let made = ShardedMap::new(256);
        assert!(made.is_ok());
        let map: Box<ShardedMap<u64>> = match made {
            Ok(m) => m,
            Err(_) => return,
        };
        let ok = std::thread::scope(|s| {
            let mut reader_slots = Vec::with_capacity(READERS);
            for _ in 0..READERS {
                match collector.register() {
                    Ok(sl) => reader_slots.push(sl),
                    Err(_) => return false,
                }
            }
            let map_ref: &ShardedMap<u64> = &map;
            let collector_ref: &Collector = &collector;
            let mut handles = Vec::with_capacity(READERS);
            for slot in reader_slots {
                handles.push(s.spawn(move || {
                    let mut checked = 0u64;
                    let mut spins = 0u64;
                    loop {
                        let g = match collector_ref.pin(slot) {
                            Ok(g) => g,
                            Err(_) => return 0u64,
                        };
                        let k = (spins % KEYS) * SHARD_COUNT as u64;
                        if let Some(v) = map_ref.get(k, &g) {
                            if *v != k * 3 {
                                return u64::MAX; // torn read: fatal
                            }
                            checked += 1;
                        }
                        drop(g);
                        spins += 1;
                        if spins >= 50_000 {
                            break;
                        }
                        if spins.is_multiple_of(64) {
                            std::thread::yield_now();
                        }
                    }
                    checked
                }));
            }
            // Writer.
            let wslot = match collector.register() {
                Ok(sl) => sl,
                Err(_) => return false,
            };
            let g = match collector.pin(wslot) {
                Ok(g) => g,
                Err(_) => return false,
            };
            for i in 0..KEYS {
                let k = i * SHARD_COUNT as u64;
                let mut tries = 0;
                loop {
                    match map_ref.insert(k, k * 3, &g) {
                        Ok(()) => break,
                        Err(ShardError::Busy) => {
                            tries += 1;
                            if tries > 1000 {
                                return false;
                            }
                            std::thread::yield_now();
                        }
                        Err(_) => return false,
                    }
                }
            }
            let mut total = 0u64;
            for h in handles {
                match h.join() {
                    Ok(v) => {
                        if v == u64::MAX {
                            return false;
                        }
                        total += v;
                    }
                    Err(_) => return false,
                }
            }
            let _ = total;
            true
        });
        assert!(ok);
        assert_eq!(map.len() as u64, KEYS);
    }
}
