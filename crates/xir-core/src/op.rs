// CEP:FILE: crates/xir-core/src/op.rs
// CEP:WHAT: The unified XIR opcode set across all five levels.
// CEP:WHY: The master architecture enumerates per-level key ops (graph.dot /
//          graph.reduce / graph.rng / graph.custom / graph.if; tensor.matmul /
//          tensor.conv / tensor.broadcast / tensor.transpose; fusion.cluster /
//          fusion.barrier / fusion.materialize; loop.parallel / loop.alloc /
//          loop.async_copy / loop.pipeline_stage; target.mma /
//          target.warp_shuffle / target.barrier). A single enum with a level
//          tag keeps dispatch a closed match (no dyn, CEP&CC 25.6) and lets
//          the pass manager police level confusion (HPC prime law).
// CEP:CLASS: CEP-0
// CEP:STATUS: partial
// CEP:FAILURE: from_code returns None for unknown opcodes (parser reports
//              the exact byte; nothing is guessed — Law 2).
// CEP:ASSUMES: immediates are bounded POD (axis < 4, perm entries < 4) —
//              validated by constructors, static-asserted MAX_RANK.
// CEP:COST: size_of::<Op>() = 24 bytes; O(1) discriminant dispatch.
// CEP:EVIDENCE: tests `opcode_roundtrip`, `level_tag_correct`;
//           xir-graph verifier tests exercise every op's arity.
// CEP:SECURITY: parser validates before construction; no raw pointers.
// CEP:HPC-DETERMINISM: deterministic; value type with ordered fields.
// CEP:TODO(main-agent): CEP-9: window/attention ops for the FlashAttention
//           pattern matcher referenced by the architecture.
//! XIR opcode set.

use crate::id::IrLevel;
use crate::ty::{Layout, ScalarType};

/// Reduction monoids for `graph.reduce`.
///
/// CEP:WHAT: Associative reduction operator.
/// CEP:WHY: Reduce reassociation legality (arch Level 1 "reduce
///          reassociation") depends on the monoid: Add/Mul on floats may NOT
///          reassociate by default (CEP&CC 38.24); Max/Min may.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: 1 byte.
/// CEP:EVIDENCE: egraph rewrite tests (reassociation gating).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Monoid {
    /// Addition (float reassociation gated).
    Add = 0,
    /// Multiplication (float reassociation gated).
    Mul = 1,
    /// Maximum (reassociable).
    Max = 2,
    /// Minimum (reassociable).
    Min = 3,
    /// Logical AND.
    And = 4,
    /// Logical OR.
    Or = 5,
}

/// Padding mode for `tensor.conv`.
///
/// CEP:WHAT: Convolution boundary mode.
/// CEP:WHY: The legality engine needs the padding mode to compute index-map
///          compatibility; the resource model needs it for halo sizes.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: 1 byte.
/// CEP:EVIDENCE: fusion legality tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Padding {
    /// No padding (valid convolution).
    Valid = 0,
    /// Same-output padding (zero fill).
    Same = 1,
}

/// RNG distribution for `graph.rng`.
///
/// CEP:WHAT: Stochastic-op distribution tag.
/// CEP:WHY: graph.rng is in the architecture's Level-0 key ops; the tag fixes
///          the sampling semantics so the interpreter and the verifier agree.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: seeds are caller-supplied u64; sampling is deterministic
///              given the seed (documented contract — HPC determinism 38.10).
/// CEP:COST: 1 byte.
/// CEP:EVIDENCE: interpreter tests in runtime crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RngDist {
    /// Uniform over [0, 1).
    Uniform = 0,
    /// Standard normal.
    Normal = 1,
}

/// Elementwise binary ops (shared by Levels 0 and 1).
///
/// CEP:WHAT: Pure elementwise opcodes.
/// CEP:WHY: Fusion candidates are elementwise chains; the interpreter needs
///          them to execute lowered programs end-to-end.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: 1 byte.
/// CEP:EVIDENCE: interpreter round-trip tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BinaryOp {
    /// Add.
    Add = 0,
    /// Subtract.
    Sub = 1,
    /// Multiply.
    Mul = 2,
    /// Divide.
    Div = 3,
    /// Max.
    Max = 4,
    /// Min.
    Min = 5,
}

/// Unary ops.
///
/// CEP:WHAT: Pure unary opcodes.
/// CEP:WHY: Epilogue fusion and the interpreter.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: 1 byte.
/// CEP:EVIDENCE: interpreter tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UnaryOp {
    /// Rectified linear unit.
    Relu = 0,
    /// Negation.
    Neg = 1,
    /// Natural exponent.
    Exp = 2,
    /// Natural logarithm.
    Log = 3,
}

/// The unified opcode set.
///
/// CEP:WHAT: Every XIR opcode with its immediate attributes.
/// CEP:WHY: One closed enum = exhaustive match dispatch (no dyn); level()
///          lets the pass manager and verifier police level transitions.
/// CEP:STATUS: partial
/// CEP:FAILURE: see module header.
/// CEP:ASSUMES: immediate bounds validated at construction (debug asserts).
/// CEP:COST: 24 bytes; O(1) dispatch.
/// CEP:EVIDENCE: tests `opcode_roundtrip`, `level_tag_correct`.
/// CEP:TODO(main-agent): CEP-9: window/attention ops (see module header).
// CEP:WHY: Eq/Hash are intentionally NOT derived: ConstF64 carries an f64
//          payload (partial equality only). Structural hashing goes through
//          hash_into (bit-pattern based), never std::Hash.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    // ---------------- Level 0: graph ----------------
    /// Constant i64 immediate.
    ConstI64(i64),
    /// Constant f64 immediate.
    ConstF64(f64),
    /// Function parameter (index immediate).
    Param {
        /// Zero-based parameter index.
        index: u32,
    },
    /// graph.dot: inner product over the last axis.
    Dot,
    /// graph.reduce: reduction over one axis.
    Reduce {
        /// Reduction axis (< rank).
        axis: u8,
        /// Reduction monoid.
        monoid: Monoid,
    },
    /// graph.rng: deterministic stochastic sampling.
    Rng {
        /// Distribution.
        dist: RngDist,
        /// Caller-supplied seed (determinism contract).
        seed: u64,
    },
    /// graph.custom: opaque user op (index into the symbol table).
    Custom {
        /// Symbol table index.
        sym: u32,
    },
    /// graph.if: region node (condition is input 0).
    If,

    // ---------------- shared elementwise ----------------
    /// Binary elementwise op.
    Binary(BinaryOp),
    /// Unary elementwise op.
    Unary(UnaryOp),

    // ---------------- Level 1: tensor ----------------
    /// tensor.matmul with optional operand transposes.
    Matmul {
        /// Transpose the left operand (permutation applied first).
        transpose_a: bool,
        /// Transpose the right operand.
        transpose_b: bool,
    },
    /// tensor.conv with padding and stride.
    Conv {
        /// Boundary mode.
        padding: Padding,
        /// Spatial stride (>= 1).
        stride: u8,
    },
    /// tensor.broadcast to a target shape.
    Broadcast {
        /// Target shape (rank must match input).
        to: crate::ty::Shape,
    },
    /// tensor.transpose by permutation.
    Transpose {
        /// Axis permutation (rank-length).
        perm: [u8; crate::ty::MAX_RANK],
        /// Input rank (valid prefix of perm).
        rank: u8,
    },

    // ---------------- Level 2: fusion ----------------
    /// fusion.cluster: hyperedge node grouping a subgraph.
    FusionCluster,
    /// fusion.barrier: materialization fence between clusters.
    FusionBarrier,
    /// fusion.materialize: force a value to memory.
    FusionMaterialize,

    // ---------------- Level 3: loop ----------------
    /// loop.parallel: parallel iteration domain axis.
    LoopParallel {
        /// Parallel axis.
        axis: u8,
    },
    /// loop.alloc: buffer allocation.
    LoopAlloc {
        /// Buffer byte size.
        bytes: u32,
        /// Address space.
        space: crate::ty::AddressSpace,
    },
    /// loop.async_copy: asynchronous copy between spaces.
    LoopAsyncCopy,
    /// loop.pipeline_stage: software pipeline stage marker.
    LoopPipelineStage {
        /// Stage number (0-based).
        stage: u32,
    },

    // ---------------- Level 4: target ----------------
    /// target.mma: matrix multiply-accumulate.
    TargetMma,
    /// target.warp_shuffle: intra-warp exchange.
    TargetWarpShuffle,
    /// target.barrier: execution barrier.
    TargetBarrier,
}

impl Op {
    /// CEP:WHAT: The IR level this opcode belongs to.
    /// CEP:WHY: Pass-manager form policing and verifier level checks.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch.
    /// CEP:EVIDENCE: test `level_tag_correct`.
    pub const fn level(self) -> IrLevel {
        match self {
            Op::ConstI64(_)
            | Op::ConstF64(_)
            | Op::Param { .. }
            | Op::Dot
            | Op::Reduce { .. }
            | Op::Rng { .. }
            | Op::Custom { .. }
            | Op::If
            | Op::Binary(_)
            | Op::Unary(_) => IrLevel::Graph,
            Op::Matmul { .. } | Op::Conv { .. } | Op::Broadcast { .. } | Op::Transpose { .. } => {
                IrLevel::Tensor
            }
            Op::FusionCluster | Op::FusionBarrier | Op::FusionMaterialize => IrLevel::Fusion,
            Op::LoopParallel { .. }
            | Op::LoopAlloc { .. }
            | Op::LoopAsyncCopy
            | Op::LoopPipelineStage { .. } => IrLevel::Loop,
            Op::TargetMma | Op::TargetWarpShuffle | Op::TargetBarrier => IrLevel::Target,
        }
    }

    /// CEP:WHAT: Reports whether the op is pure (no side effects).
    /// CEP:WHY: GVN/CSE legality (pure ops commute), DCE legality (pure
    /// unused nodes are dead), and e-graph lifting (pure subgraphs only —
    /// arch section 3 Level 0).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch.
    /// CEP:EVIDENCE: xir-graph GVN/DCE tests.
    pub const fn is_pure(self) -> bool {
        match self {
            Op::FusionBarrier
            | Op::FusionMaterialize
            | Op::LoopAlloc { .. }
            | Op::LoopAsyncCopy
            | Op::LoopPipelineStage { .. }
            | Op::TargetBarrier
            | Op::TargetWarpShuffle
            | Op::TargetMma
            | Op::Rng { .. } => false,
            // If is control flow: not freely commutable.
            Op::If => false,
            // Custom ops have UNKNOWN semantics: conservatively impure so
            // GVN never merges two calls and DCE never deletes an observable
            // side effect (audit F-7 — purity and effect tracking must not
            // contradict each other).
            Op::Custom { .. } => false,
            _ => true,
        }
    }

    /// CEP:WHAT: Reports whether the op needs an effect-token input.
    /// CEP:WHY: Sea-of-nodes side-effect ordering (arch Level 0: "Preserves
    ///          mathematical semantics and side-effect ordering via token
    ///          edges"); the verifier threads tokens through exactly these.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch.
    /// CEP:EVIDENCE: verifier token-chain tests.
    pub const fn has_effect(self) -> bool {
        !self.is_pure() || matches!(self, Op::Custom { .. })
    }

    /// CEP:WHAT: Canonical opcode discriminant for hashing/serialization.
    /// CEP:WHY: Stable numbering is the IR-versioning anchor (CEP&CC 38.17
    ///          "versioned"); changing a number is a semantic event.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: numbering is append-only from here on (documented).
    /// CEP:COST: branch.
    /// CEP:EVIDENCE: test `opcode_roundtrip`.
    pub fn opcode(self) -> u16 {
        match self {
            Op::ConstI64(_) => 1,
            Op::ConstF64(_) => 2,
            Op::Param { .. } => 3,
            Op::Dot => 4,
            Op::Reduce { .. } => 5,
            Op::Rng { .. } => 6,
            Op::Custom { .. } => 7,
            Op::If => 8,
            Op::Binary(b) => 10 + (b as u16),
            Op::Unary(u) => 20 + (u as u16),
            Op::Matmul { .. } => 30,
            Op::Conv { .. } => 31,
            Op::Broadcast { .. } => 32,
            Op::Transpose { .. } => 33,
            Op::FusionCluster => 40,
            Op::FusionBarrier => 41,
            Op::FusionMaterialize => 42,
            Op::LoopParallel { .. } => 50,
            Op::LoopAlloc { .. } => 51,
            Op::LoopAsyncCopy => 52,
            Op::LoopPipelineStage { .. } => 53,
            Op::TargetMma => 60,
            Op::TargetWarpShuffle => 61,
            Op::TargetBarrier => 62,
        }
    }

    /// CEP:WHAT: Hashes the op canonically into an FNV state.
    /// CEP:WHY: IR fingerprints (snapshot hashes) must cover immediates in a
    ///          fixed field order (CEP&CC 38.19).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: FNV state supplied by the caller.
    /// CEP:COST: O(1) per op.
    /// CEP:EVIDENCE: snapshot hash determinism tests.
    pub fn hash_into(&self, h: &mut crate::hash::Fnv64) {
        h.write_u16(self.opcode());
        match self {
            Op::ConstI64(v) => h.write_i64(*v),
            Op::ConstF64(v) => h.write_f64(*v),
            Op::Param { index } => h.write_u32(*index),
            Op::Reduce { axis, monoid } => {
                h.write_byte(*axis);
                h.write_byte(*monoid as u8);
            }
            Op::Rng { dist, seed } => {
                h.write_byte(*dist as u8);
                h.write_u64(*seed);
            }
            Op::Custom { sym } => h.write_u32(*sym),
            Op::If => {}
            Op::Binary(b) => h.write_byte(*b as u8),
            Op::Unary(u) => h.write_byte(*u as u8),
            Op::Matmul {
                transpose_a,
                transpose_b,
            } => {
                h.write_byte(u8::from(*transpose_a));
                h.write_byte(u8::from(*transpose_b));
            }
            Op::Conv { padding, stride } => {
                h.write_byte(*padding as u8);
                h.write_byte(*stride);
            }
            Op::Broadcast { to } => {
                for d in to.as_slice() {
                    h.write_i64(*d);
                }
            }
            Op::Transpose { perm, rank } => {
                for p in perm.iter().take(*rank as usize) {
                    h.write_byte(*p);
                }
            }
            Op::FusionCluster | Op::FusionBarrier | Op::FusionMaterialize => {}
            Op::LoopParallel { axis } => h.write_byte(*axis),
            Op::LoopAlloc { bytes, space } => {
                h.write_u32(*bytes);
                h.write_byte(*space as u8);
            }
            Op::LoopAsyncCopy => {}
            Op::LoopPipelineStage { stage } => h.write_u32(*stage),
            Op::TargetMma | Op::TargetWarpShuffle | Op::TargetBarrier => {}
            Op::Dot => {}
        }
    }

    /// CEP:WHAT: Default element type an interpreter uses for this op class.
    /// CEP:WHY: Tier-0 fallback execution needs a concrete element type when
    ///          the textual IR omits it; centralizing avoids per-site guesses.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch.
    /// CEP:EVIDENCE: interpreter tests.
    pub const fn default_elem(self) -> ScalarType {
        ScalarType::F64
    }

    /// CEP:WHAT: Layout preference of Level-1 producing ops.
    /// CEP:WHY: Layout inference seeds (xir-levels pass).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: branch.
    /// CEP:EVIDENCE: layout inference tests.
    pub const fn seed_layout(self) -> Layout {
        Layout::RowMajor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Distinct ops hash distinctly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on opcode collision.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn opcode_roundtrip() {
        let mut h1 = crate::hash::Fnv64::new();
        Op::ConstI64(7).hash_into(&mut h1);
        let mut h2 = crate::hash::Fnv64::new();
        Op::ConstI64(8).hash_into(&mut h2);
        assert_ne!(h1.finish(), h2.finish());

        let mut h3 = crate::hash::Fnv64::new();
        Op::ConstI64(7).hash_into(&mut h3);
        assert_eq!(h1.finish(), h3.finish());
    }

    // CEP:WHAT: Level tags match the architecture's level assignment.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on mis-tagging.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn level_tag_correct() {
        assert_eq!(Op::Dot.level(), IrLevel::Graph);
        assert_eq!(
            Op::Matmul {
                transpose_a: false,
                transpose_b: false
            }
            .level(),
            IrLevel::Tensor
        );
        assert_eq!(Op::FusionCluster.level(), IrLevel::Fusion);
        assert_eq!(Op::LoopParallel { axis: 0 }.level(), IrLevel::Loop);
        assert_eq!(Op::TargetMma.level(), IrLevel::Target);
    }

    // CEP:WHAT: Purity classification drives GVN/DCE legality.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on misclassification.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn purity_classification() {
        assert!(Op::Binary(BinaryOp::Add).is_pure());
        assert!(!Op::TargetBarrier.is_pure());
        assert!(!Op::Rng {
            dist: RngDist::Uniform,
            seed: 1
        }
        .is_pure());
        assert!(Op::Dot.is_pure());
        // Custom ops are impure by default (unknown semantics — audit F-7).
        assert!(!Op::Custom { sym: 0 }.is_pure());
        assert!(Op::Custom { sym: 0 }.has_effect());
    }
}
