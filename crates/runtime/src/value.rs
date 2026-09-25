// CEP:FILE: crates/runtime/src/value.rs
// CEP:WHAT: Runtime values — scalars and dense f64 tensors.
// CEP:WHY: The interpreter's value table needs a closed value set; dense
//          row-major f64 tensors cover the executable op set (i64 scalars
//          cover integer arithmetic). Bounds: tensor element counts are
//          verified upstream (bounded allocation, CEP&CC 39).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: constructors validate shapes (empty dims rejected).
// CEP:ASSUMES: row-major storage; shapes rank <= 4 (xir-core bound).
// CEP:COST: O(elements) construction; O(1) scalar ops.
// CEP:EVIDENCE: tests `tensor_roundtrip`, `scalar_ops`.
// CEP:SECURITY: no untrusted input parsing here (types verified upstream).
// CEP:HPC-DETERMINISM: deterministic value semantics.
//! Runtime values.

use xir_core::ty::Shape;

/// Runtime value.
///
/// CEP:WHAT: The interpreter's value lattice.
/// CEP:WHY: Closed set: f64/i64 scalars and dense f64 tensors; every op
///          documents which kinds it accepts (RuntimeError otherwise).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: tensor data length == shape element count (checked by
///           tensor constructors).
/// CEP:COST: Vec-backed tensors.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 64-bit float scalar.
    F64(f64),
    /// 64-bit integer scalar.
    I64(i64),
    /// Dense row-major f64 tensor.
    Tensor {
        /// Row-major element storage.
        data: Vec<f64>,
        /// Tensor shape.
        shape: Shape,
    },
}

/// Value construction failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueError {
    /// Data length disagrees with the shape's element count.
    LengthMismatch,
}

impl Value {
    /// CEP:WHAT: Builds a tensor value (length-checked).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Err(LengthMismatch) on disagreement.
    /// CEP:ASSUMES: shape valid.
    /// CEP:COST: O(1) move.
    /// CEP:EVIDENCE: test `tensor_roundtrip`.
    pub fn tensor(data: Vec<f64>, shape: Shape) -> Result<Value, ValueError> {
        if data.len() as i64 != shape.num_elements() {
            return Err(ValueError::LengthMismatch);
        }
        Ok(Value::Tensor { data, shape })
    }

    /// CEP:WHAT: Reports the element count for tensor values (1 for
    ///           scalars).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn element_count(&self) -> usize {
        match self {
            Value::F64(_) | Value::I64(_) => 1,
            Value::Tensor { shape, .. } => shape.num_elements().max(0) as usize,
        }
    }

    /// CEP:WHAT: Tensor shape (scalar rank-0 for scalars).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn shape(&self) -> Shape {
        match self {
            Value::F64(_) | Value::I64(_) => Shape::scalar(),
            Value::Tensor { shape, .. } => *shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Tensor construction round-trips and validates length.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on length mismatch acceptance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tensor_roundtrip() {
        let s = Shape::from_dims(&[2, 3]);
        assert!(s.is_ok());
        if let Ok(shape) = s {
            let v = Value::tensor(vec![1.0; 6], shape);
            assert!(v.is_ok());
            if let Ok(t) = v {
                assert_eq!(t.element_count(), 6);
                assert_eq!(t.shape(), shape);
            }
            let bad = Value::tensor(vec![1.0; 5], shape);
            assert_eq!(bad, Err(ValueError::LengthMismatch));
        }
    }

    // CEP:WHAT: Scalars report rank-0 shapes and one element.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on miscount.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn scalar_ops() {
        let v = Value::F64(2.5);
        assert_eq!(v.element_count(), 1);
        assert_eq!(v.shape().rank(), 0);
    }
}
