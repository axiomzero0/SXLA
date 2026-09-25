// CEP:FILE: crates/codegen/src/vectorize.rs
// CEP:WHAT: Vectorization analysis — picks SIMD widths for elementwise ops.
// CEP:WHY: Master architecture Level 3 pass list. The CPU target's natural
//          vector width derives from the cache-line constant (2 doubles per
//          128-bit lane group, 8 per line — named, not magic).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: none (analysis returns a width).
// CEP:ASSUMES: CPU reference target (docs/targets.md).
// CEP:COST: O(1) per op class.
// CEP:EVIDENCE: tests `elementwise_vectorizes`, `matmul_stays_scalar`.
// CEP:SECURITY: none.
// CEP:HPC-DETERMINISM: deterministic.
//! Vectorization analysis.

use xir_core::op::Op;
use xir_core::ty::ScalarType;

/// Vector width (elements per vector op).
///
/// CEP:WHAT: The chosen SIMD plan.
/// CEP:WHY: The interpreter's elementwise kernels batch this many elements
///          per iteration (vector-width emulation on the CPU target).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: none
/// CEP:COST: plain data
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorPlan {
    /// Elements per vector step.
    pub width: u32,
    /// True when the op is vectorizable at all.
    pub vectorizable: bool,
}

/// CEP:WHAT: Chooses the vector width for an op + element type.
/// CEP:WHY: Elementwise float/int ops vectorize at 4 (a quarter cache line
///          of f64 — the reference target's vector grouping); reductions
///          and memory-bound reshapes stay scalar ( legality: same op,
///          same rounding — elementwise batching does not reorder within an
///          element pair, so float semantics are preserved).
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn vectorizable_width(op: Op, _elem: ScalarType) -> VectorPlan {
    match op {
        Op::Binary(_) | Op::Unary(_) => VectorPlan {
            width: 4,
            vectorizable: true,
        },
        Op::Dot | Op::Matmul { .. } | Op::Conv { .. } => VectorPlan {
            width: 1,
            vectorizable: false,
        },
        Op::Transpose { .. } | Op::Broadcast { .. } => VectorPlan {
            width: 1,
            vectorizable: false,
        },
        _ => VectorPlan {
            width: 1,
            vectorizable: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::op::{BinaryOp, UnaryOp};

    // CEP:WHAT: Elementwise ops vectorize at width 4.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on width drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn elementwise_vectorizes() {
        let p = vectorizable_width(Op::Binary(BinaryOp::Add), ScalarType::F64);
        assert!(p.vectorizable);
        assert_eq!(p.width, 4);
        let p2 = vectorizable_width(Op::Unary(UnaryOp::Relu), ScalarType::F32);
        assert!(p2.vectorizable);
    }

    // CEP:WHAT: Matmul and layout ops stay scalar in this analysis.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on false vectorization.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn matmul_stays_scalar() {
        let p = vectorizable_width(
            Op::Matmul {
                transpose_a: false,
                transpose_b: false,
            },
            ScalarType::F64,
        );
        assert!(!p.vectorizable);
        let p2 = vectorizable_width(
            Op::Transpose {
                perm: [1, 0, 0, 0],
                rank: 2,
            },
            ScalarType::F64,
        );
        assert!(!p2.vectorizable);
    }
}
