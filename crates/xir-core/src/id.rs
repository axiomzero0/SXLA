// CEP:FILE: crates/xir-core/src/id.rs
// CEP:WHAT: Deterministic, packed IR identifiers (NodeId, ValueId, RegionId).
// CEP:WHY: HPC-IR contract (CEP&CC 38.17): IR nodes must have stable identity
//          within a compilation unit. Packing index + generation + level into
//          one u64 keeps handles 8 bytes (cache-friendly), makes stale-handle
//          reuse after node deletion a *detectable* error (generation bump),
//          and embeds the IR level so mis-level accesses fail loudly instead
//          of silently misinterpreting an op (Law 2: no silent assumptions).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: `IdError::GenerationMismatch` when an arena lookup hits a slot
//              whose generation differs (stale handle); `IdError::LevelMismatch`
//              when the packed level disagrees with the requested level.
// CEP:ASSUMES: index fits 32 bits (<= 4G nodes — capacity-bounded elsewhere);
//              generation fits 16 bits (wraps at 65536, documented); level fits
//              3 bits (5 levels + reserved, enforced by static assert below).
// CEP:COST: pure bit arithmetic; no memory access, no branches beyond debug
//           assertions.
// CEP:EVIDENCE: tests `roundtrip`, `generation_detects_stale`, `level_pack`.
// CEP:SECURITY: no untrusted input; ids are internal.
// CEP:HPC-DETERMINISM: deterministic; pure functions of packed bits.
//! Packed IR identifiers.

/// The five XIR levels (master architecture section 3).
///
/// CEP:WHAT: IR level discriminant.
/// CEP:WHY: Levels select lowering stages; embedding the level in every node
///          id makes cross-level confusion a loud error instead of a silent
///          miscompile (HPC prime law: never silently change meaning).
/// CEP:STATUS: complete
/// CEP:FAILURE: from_code returns None for unknown bytes.
/// CEP:ASSUMES: at most 8 values (3 bits) — static assert below.
/// CEP:COST: 3 bits.
/// CEP:EVIDENCE: test `level_pack`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum IrLevel {
    /// Level 0: functional sea-of-nodes.
    Graph = 0,
    /// Level 1: tensor algebra graph.
    Tensor = 1,
    /// Level 2: fusion cluster hypergraph.
    Fusion = 2,
    /// Level 3: loop / memory / schedule graph.
    Loop = 3,
    /// Level 4: target / machine graph.
    Target = 4,
}

impl IrLevel {
    /// CEP:WHAT: Decodes a level byte.
    /// CEP:WHY: Snapshot serialization and id unpacking.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None on unknown byte.
    /// CEP:ASSUMES: none
    /// CEP:COST: branch table
    /// CEP:EVIDENCE: test `level_pack`
    pub const fn from_code(c: u8) -> Option<IrLevel> {
        match c {
            0 => Some(IrLevel::Graph),
            1 => Some(IrLevel::Tensor),
            2 => Some(IrLevel::Fusion),
            3 => Some(IrLevel::Loop),
            4 => Some(IrLevel::Target),
            _ => None,
        }
    }
}

const _: () = {
    // CEP:WHAT: Compile-time packing bounds.
    // CEP:WHY: The bit layout below silently corrupts ids if violated
    //          (Law 3: enforced, not comment-only).
    // CEP:STATUS: complete
    // CEP:FAILURE: compile error.
    // CEP:ASSUMES: none
    // CEP:COST: compile-time only.
    // CEP:EVIDENCE: mirrored by runtime tests.
    assert!(IrLevel::Graph as u8 <= 7);
    assert!(IrLevel::Target as u8 <= 7);
};

/// Failure enumeration for id operations.
///
/// CEP:WHAT: Explicit error type for id decode/lookup mismatches.
/// CEP:WHY: Law 6 — stale or cross-level handles must fail loudly.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdError {
    /// Slot generation differs from the handle (stale id after deletion).
    GenerationMismatch,
    /// Packed level differs from the expected IR level.
    LevelMismatch,
}

/// Opaque node handle: `index | generation<<32 | level<<48`.
///
/// CEP:WHAT: 8-byte packed node identity.
/// CEP:WHY: Stability (generation) + level tagging in one cache-friendly word.
/// CEP:STATUS: complete
/// CEP:FAILURE: none in the packed type itself (lookups report IdError).
/// CEP:ASSUMES: bit layout documented here is the ONLY constructor source
///              (pack() is the single writer of these bits).
/// CEP:COST: pure arithmetic.
/// CEP:EVIDENCE: tests `roundtrip`, `level_pack`.
/// CEP:SECURITY: internal only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Sentinel "no node" (index 0 is reserved for real node 0, so this uses
    /// an impossible generation pattern instead).
    pub const NONE: NodeId = NodeId(u64::MAX);

    /// CEP:WHAT: Packs index, generation and level into one id.
    /// CEP:WHY: Single-writer of the bit layout keeps decoding unambiguous.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (bounds enforced by callers + debug asserts).
    /// CEP:ASSUMES: index < 2^32, generation < 2^16, level <= 7.
    /// CEP:COST: 3 shifts + 3 ors.
    /// CEP:EVIDENCE: test `roundtrip`.
    #[inline]
    pub const fn pack(index: u32, generation: u16, level: IrLevel) -> NodeId {
        NodeId((index as u64) | ((generation as u64) << 32) | ((level as u64) << 48))
    }

    /// CEP:WHAT: Unpacks the slot index.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: id came from pack().
    /// CEP:COST: 1 mask.
    /// CEP:EVIDENCE: test `roundtrip`.
    #[inline]
    pub const fn index(self) -> u32 {
        (self.0 & 0xFFFF_FFFF) as u32
    }

    /// CEP:WHAT: Unpacks the generation.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: id came from pack().
    /// CEP:COST: 2 shifts + 1 mask.
    /// CEP:EVIDENCE: test `roundtrip`.
    #[inline]
    pub const fn generation(self) -> u16 {
        ((self.0 >> 32) & 0xFFFF) as u16
    }

    /// CEP:WHAT: Unpacks the level tag.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: id came from pack().
    /// CEP:COST: 2 shifts + 1 mask.
    /// CEP:EVIDENCE: test `level_pack`.
    #[inline]
    pub const fn level_bits(self) -> u8 {
        ((self.0 >> 48) & 0x7) as u8
    }

    /// CEP:WHAT: Reports whether this is the NONE sentinel.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 compare.
    /// CEP:EVIDENCE: arena tests use NONE sentinels.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u64::MAX
    }
}

/// Value handle: the (node, output-slot) pair produced by a node.
///
/// CEP:WHAT: 8-byte packed value identity.
/// CEP:WHY: Sea-of-nodes data edges reference values, not nodes: one node may
///          produce several values (e.g. graph.if regions); output slot lives
///          in bits 0..7 above the node index to keep SSA use-def explicit.
/// CEP:STATUS: complete
/// CEP:FAILURE: none in the packed type (arena lookup reports IdError).
/// CEP:ASSUMES: output slot < 16 (static assert; nodes here produce at most 2).
/// CEP:COST: pure arithmetic.
/// CEP:EVIDENCE: test `value_roundtrip`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u64);

impl ValueId {
    /// Sentinel "no value".
    pub const NONE: ValueId = ValueId(u64::MAX);

    /// CEP:WHAT: Packs a node id with an output slot.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: slot < 16.
    /// CEP:COST: 1 shift + 1 or.
    /// CEP:EVIDENCE: test `value_roundtrip`.
    #[inline]
    pub const fn from_node(node: NodeId, slot: u8) -> ValueId {
        ValueId(node.0 | ((slot as u64) & 0xF) << 60)
    }

    /// CEP:WHAT: The producing node.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: from_node constructed this value.
    /// CEP:COST: 1 mask.
    /// CEP:EVIDENCE: test `value_roundtrip`.
    #[inline]
    pub const fn node(self) -> NodeId {
        NodeId(self.0 & !(0xF_u64 << 60))
    }

    /// CEP:WHAT: The output slot.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: from_node constructed this value.
    /// CEP:COST: 2 shifts.
    /// CEP:EVIDENCE: test `value_roundtrip`.
    #[inline]
    pub const fn slot(self) -> u8 {
        ((self.0 >> 60) & 0xF) as u8
    }

    /// CEP:WHAT: Sentinel check.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 compare.
    /// CEP:EVIDENCE: arena tests.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u64::MAX
    }
}

const _: () = {
    // CEP:WHAT: Value slot packing bound.
    // CEP:WHY: 4 bits of slot must hold MAX_OUTPUTS; Law 3 enforcement.
    // CEP:STATUS: complete
    // CEP:FAILURE: compile error if MAX_OUTPUTS exceeds 15.
    // CEP:ASSUMES: none
    // CEP:COST: compile-time only.
    // CEP:EVIDENCE: mirrored by runtime tests.
    assert!(crate::node::MAX_OUTPUTS <= 15);
};

/// Region handle: `index | generation<<32` (regions are level-agnostic).
///
/// CEP:WHAT: 8-byte packed region identity.
/// CEP:WHY: Regions form the control tree of the sea-of-nodes (arch section 3
///          "graph.if (region nodes)"); generation guards deletion reuse.
/// CEP:STATUS: complete
/// CEP:FAILURE: lookups report IdError on generation mismatch.
/// CEP:ASSUMES: index < 2^32, generation < 2^16.
/// CEP:COST: pure arithmetic.
/// CEP:EVIDENCE: arena region tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u64);

impl RegionId {
    /// Sentinel "no region" (root regions have parents = NONE).
    pub const NONE: RegionId = RegionId(u64::MAX);

    /// CEP:WHAT: Packs index + generation.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: bounds documented in struct.
    /// CEP:COST: 1 shift + 1 or.
    /// CEP:EVIDENCE: arena tests.
    #[inline]
    pub const fn pack(index: u32, generation: u16) -> RegionId {
        RegionId((index as u64) | ((generation as u64) << 32))
    }

    /// CEP:WHAT: Slot index.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: pack() origin.
    /// CEP:COST: 1 mask.
    /// CEP:EVIDENCE: arena tests.
    #[inline]
    pub const fn index(self) -> u32 {
        (self.0 & 0xFFFF_FFFF) as u32
    }

    /// CEP:WHAT: Generation.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: pack() origin.
    /// CEP:COST: 2 shifts + 1 mask.
    /// CEP:EVIDENCE: arena tests.
    #[inline]
    pub const fn generation(self) -> u16 {
        ((self.0 >> 32) & 0xFFFF) as u16
    }

    /// CEP:WHAT: Sentinel check.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 compare.
    /// CEP:EVIDENCE: arena tests.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u64::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: pack/unpack roundtrip for all fields.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on bit corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn roundtrip() {
        let id = NodeId::pack(0x1234_5678, 0xBEEF, IrLevel::Tensor);
        assert_eq!(id.index(), 0x1234_5678);
        assert_eq!(id.generation(), 0xBEEF);
        assert_eq!(id.level_bits(), IrLevel::Tensor as u8);
        assert!(!id.is_none());
        assert!(NodeId::NONE.is_none());
    }

    // CEP:WHAT: Generation distinguishes stale ids.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if generations collide.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn generation_detects_stale() {
        let a = NodeId::pack(1, 1, IrLevel::Graph);
        let b = NodeId::pack(1, 2, IrLevel::Graph);
        assert_ne!(a, b);
        assert_eq!(a.index(), b.index());
    }

    // CEP:WHAT: All five levels encode/decode.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on level corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn level_pack() {
        for lv in [
            IrLevel::Graph,
            IrLevel::Tensor,
            IrLevel::Fusion,
            IrLevel::Loop,
            IrLevel::Target,
        ] {
            let id = NodeId::pack(7, 3, lv);
            assert_eq!(IrLevel::from_code(id.level_bits()), Some(lv));
        }
        assert_eq!(IrLevel::from_code(5), None);
    }

    // CEP:WHAT: ValueId node/slot roundtrip.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on slot corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn value_roundtrip() {
        let n = NodeId::pack(99, 4, IrLevel::Graph);
        let v = ValueId::from_node(n, 3);
        assert_eq!(v.node(), n);
        assert_eq!(v.slot(), 3);
        assert!(!v.is_none());
        assert!(ValueId::NONE.is_none());
    }
}
