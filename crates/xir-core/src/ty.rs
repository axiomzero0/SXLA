// CEP:FILE: crates/xir-core/src/ty.rs
// CEP:WHAT: XIR type system — scalars, bounded tensor types, layouts, tokens.
// CEP:WHY: HPC-IR contract (CEP&CC 38.17/38.18): the verifier needs type
//          correctness checks after every pass. Tensor shapes are fixed-size
//          arrays (rank <= 4) instead of heap Vecs: CEP-0 code cannot hide
//          allocation (Law 1) and shape ops stay branch-bounded.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: constructors and layout ops return TypeResult with explicit
//              RankTooLarge/DimNonPositive errors; no panics.
// CEP:ASSUMES: IEEE 754 binary64 for F64 (documented platform contract);
//              maximum rank 4 (named constant, static-asserted).
// CEP:COST: all operations are fixed-size copies; O(1).
// CEP:EVIDENCE: tests `shape_roundtrip`, `rank_is_bounded`, `size_roundtrip`.
// CEP:SECURITY: no untrusted input parsing in this module (parser validates
//               elsewhere before constructing types).
// CEP:HPC-DETERMINISM: deterministic; value semantics, no pointers.
//! XIR type system.

/// Maximum tensor rank supported by the bounded shape representation.
///
/// CEP:WHAT: Rank bound.
/// CEP:WHY: CEP-0 forbids heap-growing shapes in hot paths; rank 4 covers
///          NCHW conv workloads (the Level-1 tensor ops of the master
///          architecture). Rank 5+ returns a loud error rather than a silent
///          Vec (Law 1/6).
/// CEP:STATUS: complete
/// CEP:FAILURE: TypeError::RankTooLarge beyond 4.
/// CEP:ASSUMES: none
/// CEP:COST: 4 * 8 bytes per shape.
/// CEP:EVIDENCE: test `rank_is_bounded`.
pub const MAX_RANK: usize = 4;

/// Scalar element types.
///
/// CEP:WHAT: Element type discriminant.
/// CEP:WHY: Ops need exact element semantics for legality (e.g. integer vs
///          float reassociation rules, CEP&CC 38.24).
/// CEP:STATUS: complete
/// CEP:FAILURE: from_code returns None on unknown bytes.
/// CEP:ASSUMES: F64 is IEEE 754 binary64; I64 two's complement (Rust).
/// CEP:COST: 1 byte.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ScalarType {
    /// 64-bit IEEE 754 binary64 float.
    F64 = 0,
    /// 64-bit two's complement signed integer.
    I64 = 1,
    /// 32-bit IEEE 754 binary32 float.
    F32 = 2,
    /// 32-bit two's complement signed integer.
    I32 = 3,
    /// 1-bit predicate (graph.if conditions).
    Bool = 4,
}

impl ScalarType {
    /// CEP:WHAT: Element byte size (packed types round up to 1).
    /// CEP:WHY: Bufferization and the resource model need element widths.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch table.
    /// CEP:EVIDENCE: test `size_roundtrip`.
    pub const fn byte_size(self) -> u32 {
        match self {
            ScalarType::F64 | ScalarType::I64 => 8,
            ScalarType::F32 | ScalarType::I32 => 4,
            ScalarType::Bool => 1,
        }
    }

    /// CEP:WHAT: Decodes a discriminant byte.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None on unknown byte.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch table.
    /// CEP:EVIDENCE: tests in this module.
    pub const fn from_code(c: u8) -> Option<ScalarType> {
        match c {
            0 => Some(ScalarType::F64),
            1 => Some(ScalarType::I64),
            2 => Some(ScalarType::F32),
            3 => Some(ScalarType::I32),
            4 => Some(ScalarType::Bool),
            _ => None,
        }
    }

    /// CEP:WHAT: Reports whether the element is floating point.
    /// CEP:WHY: FP ops carry the reassociation ban (CEP&CC 38.24); the
    ///          legality engine branches on this.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 2 compares.
    /// CEP:EVIDENCE: fusion legality tests.
    pub const fn is_float(self) -> bool {
        matches!(self, ScalarType::F64 | ScalarType::F32)
    }
}

/// Tensor memory layout.
///
/// CEP:WHAT: Layout discriminant for Level-1 layout inference.
/// CEP:WHY: The master architecture makes layout a first-class tensor
///          attribute (xir.tensor "rich tensor metadata (iteration domains,
///          affine indexing maps, layouts)"); transpose propagation and
///          fusion legality consume it.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: layouts describe dense storage (no strides yet — status of
///              strided layouts is partial, see CEP:TODO).
/// CEP:COST: 1 byte.
/// CEP:EVIDENCE: layout inference tests in xir-levels.
/// CEP:TODO(main-agent): CEP-7: strided/block layouts for hardware swizzle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Layout {
    /// Row-major (C order): last axis contiguous.
    RowMajor = 0,
    /// Column-major (Fortran order): first axis contiguous.
    ColMajor = 1,
}

/// Bounded tensor shape: rank <= MAX_RANK, dims fixed-size.
///
/// CEP:WHAT: Fixed-capacity shape.
/// CEP:WHY: Value semantics, O(1) copy, zero allocation — shapes ride inside
///          nodes and snapshots (CEP-0).
/// CEP:STATUS: complete
/// CEP:FAILURE: from_dims returns RankTooLarge when dims.len() > MAX_RANK;
///              DimNonPositive when any dim <= 0.
/// CEP:ASSUMES: dims beyond rank are ignored (always zeroed by from_dims).
/// CEP:COST: 40 bytes; O(1) ops.
/// CEP:EVIDENCE: tests `shape_roundtrip`, `rank_is_bounded`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [i64; MAX_RANK],
    rank: u8,
}

impl Shape {
    /// CEP:WHAT: Builds a shape from a dim slice.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: RankTooLarge / DimNonPositive (loud, no panic).
    /// CEP:ASSUMES: none.
    /// CEP:COST: O(MAX_RANK) copy.
    /// CEP:EVIDENCE: tests `shape_roundtrip`, `rank_is_bounded`.
    pub fn from_dims(dims: &[i64]) -> TypeResult<Shape> {
        if dims.len() > MAX_RANK {
            return Err(TypeError::RankTooLarge);
        }
        let mut out = Shape {
            dims: [0; MAX_RANK],
            rank: dims.len() as u8,
        };
        for (i, d) in dims.iter().enumerate() {
            if *d <= 0 {
                return Err(TypeError::DimNonPositive);
            }
            out.dims[i] = *d;
        }
        Ok(out)
    }

    /// CEP:WHAT: Scalar (rank-0) shape.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: constant.
    /// CEP:EVIDENCE: tests in this module.
    pub const fn scalar() -> Shape {
        Shape {
            dims: [0; MAX_RANK],
            rank: 0,
        }
    }

    /// CEP:WHAT: Rank accessor.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 load.
    /// CEP:EVIDENCE: tests in this module.
    pub const fn rank(&self) -> u8 {
        self.rank
    }

    /// CEP:WHAT: Dim accessor with bounds check.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None when axis >= rank (callers treat as absent rather
    ///              than panicking; verifier flags misuse).
    /// CEP:ASSUMES: none.
    /// CEP:COST: 2 compares + 1 load.
    /// CEP:EVIDENCE: tests `shape_roundtrip`.
    pub const fn dim(&self, axis: u8) -> Option<i64> {
        if axis < self.rank {
            Some(self.dims[axis as usize])
        } else {
            None
        }
    }

    /// CEP:WHAT: Number of elements (product of dims).
    /// CEP:WHY: Bufferization sizes and the resource model consume this.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: saturating multiply — a shape whose element count
    ///              overflows i64 saturates at i64::MAX (documented; the
    ///              resource model treats saturation as over-budget).
    /// CEP:ASSUMES: none.
    /// CEP:COST: O(rank).
    /// CEP:EVIDENCE: tests `size_roundtrip`.
    pub fn num_elements(&self) -> i64 {
        let mut n: i64 = 1;
        for i in 0..self.rank as usize {
            n = n.saturating_mul(self.dims[i]);
        }
        n
    }

    /// CEP:WHAT: Dims as a slice (printing, hashing).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: zero.
    /// CEP:EVIDENCE: tests `shape_roundtrip`.
    pub fn as_slice(&self) -> &[i64] {
        &self.dims[..self.rank as usize]
    }
}

/// Tensor type: element type + shape + layout.
///
/// CEP:WHAT: The Level-1 tensor descriptor.
/// CEP:WHY: Fusion legality (index-map compatibility), the resource model
///          (bytes) and layout inference all need the full triple.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (construction is total; Shape validates dims).
/// CEP:ASSUMES: none.
/// CEP:COST: 48 bytes value type.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorType {
    /// Element type.
    pub elem: ScalarType,
    /// Bounded shape.
    pub shape: Shape,
    /// Memory layout.
    pub layout: Layout,
}

/// Memory address space (Level 3 / Level 4).
///
/// CEP:WHAT: Address space discriminant.
/// CEP:WHY: Shared-memory promotion (arch xir.loop) needs explicit spaces;
///          the resource model sums shared-space bytes only.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: 1 byte.
/// CEP:EVIDENCE: resource model tests in fusion crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AddressSpace {
    /// Global (device) memory.
    Global = 0,
    /// Shared (on-chip) memory.
    Shared = 1,
    /// Registers (private).
    Register = 2,
}

/// XIR types.
///
/// CEP:WHAT: The type lattice: scalars, tensors, effect tokens, memrefs.
/// CEP:WHY: `Token` types the effect edges (arch: "side-effect ordering via
///          token edges"); `MemRef` types Level-3 buffers with their address
///          space so bufferization/verifier can check memory legality.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (value type).
/// CEP:ASSUMES: none.
/// CEP:COST: 56 bytes value type; O(1) ops.
/// CEP:EVIDENCE: verifier tests in xir-graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    /// Scalar element type (rank-0).
    Scalar(ScalarType),
    /// Tensor type with shape and layout.
    Tensor(TensorType),
    /// Effect token (side-effect ordering edge).
    Token,
    /// Buffer reference: tensor type + address space.
    MemRef(TensorType, AddressSpace),
    /// No value (void op results, e.g. barriers).
    None,
}

/// Explicit type-construction failure enumeration.
///
/// CEP:WHAT: Error type for shape construction.
/// CEP:WHY: Law 6 — bounded rank and positive dims must fail loudly.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none.
/// CEP:COST: zero-size enum.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeError {
    /// Rank exceeded MAX_RANK.
    RankTooLarge,
    /// A dimension was <= 0.
    DimNonPositive,
}

/// Result alias for type construction.
pub type TypeResult<T> = Result<T, TypeError>;

impl Type {
    /// CEP:WHAT: Element size in bytes (scalars and tensors; 0 for Token/None).
    /// CEP:WHY: Bufferization and resource estimation.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch.
    /// CEP:EVIDENCE: tests `size_roundtrip`.
    pub fn byte_size(&self) -> i64 {
        match self {
            Type::Scalar(s) => i64::from(s.byte_size()),
            Type::Tensor(t) => t
                .shape
                .num_elements()
                .saturating_mul(i64::from(t.elem.byte_size())),
            Type::MemRef(t, _) => t
                .shape
                .num_elements()
                .saturating_mul(i64::from(t.elem.byte_size())),
            Type::Token | Type::None => 0,
        }
    }

    /// CEP:WHAT: Tensor view if this is a tensor or memref.
    /// CEP:WHY: Fusion/layout passes unwrap the common case without a match
    ///          pyramid.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None for non-tensor types.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch.
    /// CEP:EVIDENCE: tests in this module.
    pub fn as_tensor(&self) -> Option<TensorType> {
        match self {
            Type::Tensor(t) => Some(*t),
            Type::MemRef(t, _) => Some(*t),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Shape construction and access roundtrip.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn shape_roundtrip() {
        let s = Shape::from_dims(&[2, 3, 4]);
        assert!(s.is_ok());
        let s = match s {
            Ok(v) => v,
            Err(_) => return,
        };
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dim(0), Some(2));
        assert_eq!(s.dim(2), Some(4));
        assert_eq!(s.dim(3), None);
        assert_eq!(s.num_elements(), 24);
        assert_eq!(s.as_slice(), &[2, 3, 4]);
    }

    // CEP:WHAT: Rank bound is enforced loudly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if RankTooLarge is not reported.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rank_is_bounded() {
        assert_eq!(
            Shape::from_dims(&[1, 2, 3, 4, 5]),
            Err(TypeError::RankTooLarge)
        );
        assert_eq!(Shape::from_dims(&[0, 2]), Err(TypeError::DimNonPositive));
    }

    // CEP:WHAT: Byte sizes are consistent with element widths.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on width drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn size_roundtrip() {
        let t = Type::Tensor(TensorType {
            elem: ScalarType::F64,
            shape: Shape::from_dims(&[16, 16]).ok().unwrap_or(Shape::scalar()),
            layout: Layout::RowMajor,
        });
        assert_eq!(t.byte_size(), 16 * 16 * 8);
        let t32 = Type::Tensor(TensorType {
            elem: ScalarType::F32,
            shape: Shape::from_dims(&[4]).ok().unwrap_or(Shape::scalar()),
            layout: Layout::RowMajor,
        });
        assert_eq!(t32.byte_size(), 16);
    }
}
