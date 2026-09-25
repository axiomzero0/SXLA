// CEP:FILE: crates/runtime/src/interp.rs
// CEP:WHAT: The Tier-0 interpreter — executes CPU TargetPrograms.
// CEP:WHY: Master architecture section 6: "Tier 0 (Fallback): Interpreter
//          or pre-compiled generic kernel. Used immediately while
//          compilation is pending." This interpreter is the semantic
//          reference for differential testing of every optimizing tier.
// CEP:CLASS: CEP-0 (kernels) / CEP-1 (dispatch loop)
// CEP:STATUS: complete
// CEP:FAILURE: RuntimeError::{BadSlot, ShapeMismatch, UnsupportedValue,
//             DivisionByZero, UnsupportedInstr} — explicit, never panics.
// CEP:ASSUMES: lowered TargetProgram; params bound in slot order.
// CEP:COST: naive kernels: elementwise O(n), dot O(n), matmul O(m*k*n),
//           conv O(hw*cf*rr) — the correctness-first Tier-0 path
//           (documented; Tier-1/2 codegen owns performance).
// CEP:EVIDENCE: tests `scalar_arithmetic`, `tensor_elementwise`, `dot`,
//           `reduce_axis`, `matmul_reference`, `rng_is_deterministic`,
//           `conv_valid_reference`, `conv_same_zero_fill`,
//           `conv_stride_reference`, `conv_multichannel_reference`.
// CEP:SECURITY: slot bounds checked per instruction; allocation sized by
//           verified types (no untrusted sizes reach here).
// CEP:HPC-CLASS: HPC-0.
// CEP:HPC-DETERMINISM: deterministic — fixed iteration order, seeded RNG.
//! The reference interpreter.

use xir_core::op::{Monoid, Padding, RngDist};
use xir_core::ty::Shape;
use xir_levels::level4::{Instr, TargetProgram};

use crate::value::Value;

/// Interpreter failure enumeration.
///
/// CEP:WHAT: Explicit execution error type.
/// CEP:WHY: Law 6 — every failure mode of the runtime must be named.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeError {
    /// A value slot was out of range.
    BadSlot,
    /// Operand shapes disagreed.
    ShapeMismatch,
    /// A slot held the wrong value kind.
    UnsupportedValue,
    /// Float or integer division by zero (loud, never Inf silently —
    /// CEP&CC 38.24: no silent FP semantic change).
    DivisionByZero,
    /// An instruction has no interpreter kernel.
    UnsupportedInstr,
}

/// CEP:WHAT: Executes a target program over bound parameter values.
/// CEP:WHY: The Tier-0 path and the semantic oracle.
/// CEP:STATUS: complete
/// CEP:FAILURE: see RuntimeError; execution stops at the first failure.
/// CEP:ASSUMES: params.len() == program param slots; results non-empty.
/// CEP:COST: see module header.
/// CEP:EVIDENCE: tests in this module + tools.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn execute(program: &TargetProgram, params: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    let table_len = program.value_count as usize;
    let mut table: Vec<Value> = Vec::with_capacity(table_len);
    // Parameter slots come first.
    for p in params.iter() {
        table.push(p.clone());
    }
    while table.len() < table_len {
        table.push(Value::F64(0.0));
    }
    for instr in program.instrs.iter() {
        run_instr(&mut table, instr)?;
    }
    let mut out = Vec::with_capacity(program.results.len());
    for r in program.results.iter() {
        let v = table.get(*r as usize).ok_or(RuntimeError::BadSlot)?;
        out.push(v.clone());
    }
    Ok(out)
}

/// CEP:WHAT: Runs one instruction against the value table.
/// CEP:STATUS: complete
/// CEP:FAILURE: see RuntimeError.
/// CEP:ASSUMES: table sized.
/// CEP:COST: kernel-dependent (module header).
/// CEP:EVIDENCE: kernel tests.
fn run_instr(table: &mut [Value], instr: &Instr) -> Result<(), RuntimeError> {
    match instr {
        Instr::Add { dst, a, b } => bin_op(table, *dst, *a, *b, |x, y| Ok(x + y)),
        Instr::Sub { dst, a, b } => bin_op(table, *dst, *a, *b, |x, y| Ok(x - y)),
        Instr::Mul { dst, a, b } => bin_op(table, *dst, *a, *b, |x, y| Ok(x * y)),
        Instr::Div { dst, a, b } => bin_op(table, *dst, *a, *b, |x, y| {
            if y == 0.0 {
                Err(RuntimeError::DivisionByZero)
            } else {
                Ok(x / y)
            }
        }),
        Instr::Max { dst, a, b } => bin_op(table, *dst, *a, *b, |x, y| Ok(x.max(y))),
        Instr::Min { dst, a, b } => bin_op(table, *dst, *a, *b, |x, y| Ok(x.min(y))),
        Instr::Neg { dst, a } => un_op(table, *dst, *a, |x| Ok(-x)),
        Instr::Relu { dst, a } => un_op(table, *dst, *a, |x| Ok(x.max(0.0))),
        Instr::Exp { dst, a } => un_op(table, *dst, *a, |x| Ok(x.exp())),
        Instr::Log { dst, a } => un_op(table, *dst, *a, |x| Ok(x.ln())),
        Instr::ConstF64 { dst, v } => store_f64(table, *dst, *v),
        Instr::ConstI64 { dst, v } => {
            let d = table.get_mut(*dst as usize).ok_or(RuntimeError::BadSlot)?;
            *d = Value::I64(*v);
            Ok(())
        }
        Instr::Mov { dst, a } => {
            let v = table.get(*a as usize).ok_or(RuntimeError::BadSlot)?;
            let v = v.clone();
            let d = table.get_mut(*dst as usize).ok_or(RuntimeError::BadSlot)?;
            *d = v;
            Ok(())
        }
        Instr::Dot { dst, a, b } => {
            let (x, y) = (load_f64s(table, *a)?, load_f64s(table, *b)?);
            if x.len() != y.len() {
                return Err(RuntimeError::ShapeMismatch);
            }
            let mut acc = 0.0;
            for i in 0..x.len() {
                acc += x[i] * y[i];
            }
            store_f64(table, *dst, acc)
        }
        Instr::Reduce {
            dst,
            a,
            axis,
            monoid,
        } => reduce(table, *dst, *a, *axis, *monoid),
        Instr::Matmul {
            dst,
            a,
            b,
            transpose_a,
            transpose_b,
        } => matmul(table, *dst, *a, *b, *transpose_a, *transpose_b),
        Instr::Broadcast { dst, a } => {
            // Broadcast a scalar or tensor to a fixed extent (shape kept).
            let v = table.get(*a as usize).ok_or(RuntimeError::BadSlot)?.clone();
            let d = table.get_mut(*dst as usize).ok_or(RuntimeError::BadSlot)?;
            *d = v;
            Ok(())
        }
        Instr::Transpose { dst, a } => transpose(table, *dst, *a),
        Instr::Rng { dst, seed, dist } => rng(table, *dst, *seed, *dist),
        Instr::Conv {
            dst,
            a,
            b,
            padding,
            stride,
        } => conv(table, *dst, *a, *b, *padding, *stride),
    }
}

/// CEP:WHAT: Loads a value's f64 elements (scalar -> 1).
/// CEP:STATUS: complete
/// CEP:FAILURE: UnsupportedValue for i64 scalars/tensors.
/// CEP:ASSUMES: none
/// CEP:COST: O(1) borrow.
/// CEP:EVIDENCE: kernel tests
fn load_f64s(table: &[Value], slot: u32) -> Result<Vec<f64>, RuntimeError> {
    match table.get(slot as usize).ok_or(RuntimeError::BadSlot)? {
        Value::F64(x) => Ok(vec![*x]),
        Value::Tensor { data, .. } => Ok(data.clone()),
        Value::I64(_) => Err(RuntimeError::UnsupportedValue),
    }
}

/// CEP:WHAT: Stores an f64 scalar into a slot.
/// CEP:STATUS: complete
/// CEP:FAILURE: BadSlot.
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: kernel tests
fn store_f64(table: &mut [Value], slot: u32, v: f64) -> Result<(), RuntimeError> {
    let d = table.get_mut(slot as usize).ok_or(RuntimeError::BadSlot)?;
    *d = Value::F64(v);
    Ok(())
}

/// CEP:WHAT: Elementwise binary kernel (scalar-scalar, scalar-tensor,
///           tensor-tensor with shape check).
/// CEP:STATUS: complete
/// CEP:FAILURE: DivisionByZero propagates from the op closure.
/// CEP:ASSUMES: none
/// CEP:COST: O(elements).
/// CEP:EVIDENCE: tests `scalar_arithmetic`, `tensor_elementwise`.
fn bin_op(
    table: &mut [Value],
    dst: u32,
    a: u32,
    b: u32,
    f: fn(f64, f64) -> Result<f64, RuntimeError>,
) -> Result<(), RuntimeError> {
    let va = table.get(a as usize).ok_or(RuntimeError::BadSlot)?;
    let vb = table.get(b as usize).ok_or(RuntimeError::BadSlot)?;
    match (va, vb) {
        (Value::F64(x), Value::F64(y)) => {
            let r = f(*x, *y)?;
            store_f64(table, dst, r)
        }
        (Value::I64(x), Value::I64(y)) => {
            // Integer path preserves exact integer semantics.
            let r = match f(*x as f64, *y as f64)? {
                v if v.fract() == 0.0 => Value::I64(v as i64),
                v => Value::F64(v),
            };
            let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
            *d = r;
            Ok(())
        }
        (Value::F64(x), Value::Tensor { data, shape }) => {
            let sv = *x;
            let mut out = Vec::with_capacity(data.len());
            for y in data.iter() {
                out.push(f(sv, *y)?);
            }
            let shape = *shape;
            let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
            *d = Value::Tensor { data: out, shape };
            Ok(())
        }
        (Value::Tensor { data, shape }, Value::F64(y)) => {
            let sv = *y;
            let mut out = Vec::with_capacity(data.len());
            for x in data.iter() {
                out.push(f(*x, sv)?);
            }
            let shape = *shape;
            let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
            *d = Value::Tensor { data: out, shape };
            Ok(())
        }
        (Value::Tensor { data: x, shape: sx }, Value::Tensor { data: y, shape: sy }) => {
            if sx != sy || x.len() != y.len() {
                return Err(RuntimeError::ShapeMismatch);
            }
            let mut out = Vec::with_capacity(x.len());
            for i in 0..x.len() {
                out.push(f(x[i], y[i])?);
            }
            let shape = *sx;
            let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
            *d = Value::Tensor { data: out, shape };
            Ok(())
        }
        _ => Err(RuntimeError::UnsupportedValue),
    }
}

/// CEP:WHAT: Elementwise unary kernel.
/// CEP:STATUS: complete
/// CEP:FAILURE: UnsupportedValue for i64.
/// CEP:ASSUMES: none
/// CEP:COST: O(elements)
/// CEP:EVIDENCE: tests
fn un_op(
    table: &mut [Value],
    dst: u32,
    a: u32,
    f: fn(f64) -> Result<f64, RuntimeError>,
) -> Result<(), RuntimeError> {
    let va = table.get(a as usize).ok_or(RuntimeError::BadSlot)?;
    match va {
        Value::F64(x) => store_f64(table, dst, f(*x)?),
        Value::Tensor { data, shape } => {
            let mut out = Vec::with_capacity(data.len());
            for x in data.iter() {
                out.push(f(*x)?);
            }
            let shape = *shape;
            let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
            *d = Value::Tensor { data: out, shape };
            Ok(())
        }
        Value::I64(_) => Err(RuntimeError::UnsupportedValue),
    }
}

/// CEP:WHAT: Axis reduction kernel.
/// CEP:WHY: Deterministic sequential fold over the axis (float order fixed
///          left-to-right — no reassociation, CEP&CC 38.24).
/// CEP:STATUS: complete
/// CEP:FAILURE: ShapeMismatch on bad axis.
/// CEP:ASSUMES: row-major storage.
/// CEP:COST: O(elements).
/// CEP:EVIDENCE: test `reduce_axis`.
fn reduce(
    table: &mut [Value],
    dst: u32,
    a: u32,
    axis: u8,
    monoid: Monoid,
) -> Result<(), RuntimeError> {
    let va = table.get(a as usize).ok_or(RuntimeError::BadSlot)?.clone();
    let (data, shape) = match va {
        Value::Tensor { data, shape } => (data, shape),
        Value::F64(_) => return store_f64(table, dst, va_scalar(&va)),
        Value::I64(_) => return Err(RuntimeError::UnsupportedValue),
    };
    let rank = shape.rank();
    if axis >= rank {
        return Err(RuntimeError::ShapeMismatch);
    }
    // Output shape without the axis.
    let mut out_dims: Vec<i64> = Vec::with_capacity(rank as usize);
    for d in 0..rank {
        if d != axis {
            out_dims.push(shape.dim(d).unwrap_or(1));
        }
    }
    let out_shape = Shape::from_dims(&out_dims).map_err(|_| RuntimeError::ShapeMismatch)?;
    let axis_len = shape.dim(axis).unwrap_or(1) as usize;
    let outer = if rank as usize > axis as usize + 1 {
        let mut o: usize = 1;
        for d in (axis as usize + 1)..rank as usize {
            o *= shape.dim(d as u8).unwrap_or(1) as usize;
        }
        o
    } else {
        1
    };
    let total_out = out_shape.num_elements().max(0) as usize;
    let mut out = vec![monoid_identity(monoid); total_out];
    // Row-major traversal: contiguous inner segment per (outer, axis) row.
    let inner = outer;
    for (ob, o) in out.iter_mut().enumerate() {
        let row = ob / inner.max(1);
        let off = ob % inner.max(1);
        let base = row * axis_len * inner + off;
        let mut acc = monoid_identity(monoid);
        for k in 0..axis_len {
            let v = data[base + k * inner];
            acc = monoid_apply(monoid, acc, v);
        }
        *o = acc;
    }
    if out_shape.rank() == 0 {
        store_f64(table, dst, out[0])
    } else {
        let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
        *d = Value::Tensor {
            data: out,
            shape: out_shape,
        };
        Ok(())
    }
}

/// CEP:WHAT: Identity of a reduction monoid.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: reduce tests
fn monoid_identity(m: Monoid) -> f64 {
    match m {
        Monoid::Add | Monoid::Or => 0.0,
        Monoid::Mul | Monoid::And => 1.0,
        Monoid::Max => f64::NEG_INFINITY,
        Monoid::Min => f64::INFINITY,
    }
}

/// CEP:WHAT: Applies one monoid step.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: reduce tests
fn monoid_apply(m: Monoid, acc: f64, v: f64) -> f64 {
    match m {
        Monoid::Add | Monoid::Or => acc + v,
        Monoid::Mul | Monoid::And => acc * v,
        Monoid::Max => acc.max(v),
        Monoid::Min => acc.min(v),
    }
}

/// CEP:WHAT: Naive matmul with optional operand transposes.
/// CEP:STATUS: complete
/// CEP:FAILURE: ShapeMismatch on non-2D operands.
/// CEP:ASSUMES: rank-2 operands.
/// CEP:COST: O(m*k*n) — Tier-0 correctness kernel.
/// CEP:EVIDENCE: test `matmul_reference`.
fn matmul(
    table: &mut [Value],
    dst: u32,
    a: u32,
    b: u32,
    ta: bool,
    tb: bool,
) -> Result<(), RuntimeError> {
    let (xa, sa) = (load_f64s(table, a)?, tensor_shape(table, a)?);
    let (xb, sb) = (load_f64s(table, b)?, tensor_shape(table, b)?);
    if sa.rank() != 2 || sb.rank() != 2 {
        return Err(RuntimeError::ShapeMismatch);
    }
    let (m0, k0) = (sa.dim(0).unwrap_or(0), sa.dim(1).unwrap_or(0));
    let (k1, n1) = (sb.dim(0).unwrap_or(0), sb.dim(1).unwrap_or(0));
    // Effective dims after transposes.
    // Effective dims after transposes: A becomes (m, ka), B becomes (kb, n).
    let (m, ka) = if ta { (k0, m0) } else { (m0, k0) };
    let (kb, n) = if tb { (n1, k1) } else { (k1, n1) };
    if ka != kb {
        return Err(RuntimeError::ShapeMismatch);
    }
    let k = ka;
    let idx = |rows: i64, cols: i64, r: i64, c: i64, tr: bool| -> usize {
        if tr {
            (c * rows + r) as usize
        } else {
            (r * cols + c) as usize
        }
    };
    let mut out = vec![0.0f64; (m * n) as usize];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0;
            for k2 in 0..k {
                let av = xa[idx(m0, k0, r, k2, ta)];
                let bv = xb[idx(k1, n1, k2, c, tb)];
                acc += av * bv;
            }
            out[(r * n + c) as usize] = acc;
        }
    }
    let shape = Shape::from_dims(&[m, n]).map_err(|_| RuntimeError::ShapeMismatch)?;
    let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
    *d = Value::Tensor { data: out, shape };
    Ok(())
}

/// CEP:WHAT: Naive 2D convolution (NCHW input, FCHW filter).
/// CEP:WHY: Tier-0 correctness kernel for tensor.conv (CEP-26 closing):
///           the reference semantics every optimized convolution lowering
///           must match differentially. Valid = no zero padding; Same =
///           symmetric zero fill so output spatial dims are ceil(H/stride)
///           and ceil(W/stride) (the extra odd pad pixel goes to the
///           bottom/right — TF discipline, documented for differential
///           agreement with any future tiled lowering).
/// CEP:STATUS: complete
/// CEP:FAILURE: ShapeMismatch for non-rank-4 operands, input/filter
///              channel disagreement, zero stride, or non-positive
///              output dims; UnsupportedValue for i64 tensors.
/// CEP:ASSUMES: f64 row-major tensors.
/// CEP:COST: O(N*F*OH*OW*C*KH*KW) — reference kernel, no tiling.
/// CEP:EVIDENCE: tests `conv_valid_reference`, `conv_same_zero_fill`,
///           `conv_stride_reference`, `conv_multichannel_reference`,
///           `conv_shape_mismatch`.
/// CEP:HPC-DETERMINISM: deterministic — fixed loop order.
fn conv(
    table: &mut [Value],
    dst: u32,
    a: u32,
    b: u32,
    padding: Padding,
    stride: u8,
) -> Result<(), RuntimeError> {
    let (xa, sa) = (load_f64s(table, a)?, tensor_shape(table, a)?);
    let (xb, sb) = (load_f64s(table, b)?, tensor_shape(table, b)?);
    if sa.rank() != 4 || sb.rank() != 4 {
        return Err(RuntimeError::ShapeMismatch);
    }
    if stride == 0 {
        return Err(RuntimeError::ShapeMismatch);
    }
    let stride = i64::from(stride);
    let (n, c, h, w) = (
        sa.dim(0).unwrap_or(0),
        sa.dim(1).unwrap_or(0),
        sa.dim(2).unwrap_or(0),
        sa.dim(3).unwrap_or(0),
    );
    let (f, fc, kh, kw) = (
        sb.dim(0).unwrap_or(0),
        sb.dim(1).unwrap_or(0),
        sb.dim(2).unwrap_or(0),
        sb.dim(3).unwrap_or(0),
    );
    if c != fc {
        return Err(RuntimeError::ShapeMismatch);
    }
    // Output spatial dims and total zero-padding per axis.
    let (oh, ow, pad_h, pad_w) = match padding {
        Padding::Valid => {
            // Oversized kernels are rejected up front (audit F-7): Rust's
            // truncating division would turn (h - kh) = -1 at stride >= 2
            // into a ZERO quotient, silently producing a partial-window
            // convolution where the reference errors.
            if kh > h || kw > w {
                return Err(RuntimeError::ShapeMismatch);
            }
            ((h - kh) / stride + 1, (w - kw) / stride + 1, 0i64, 0i64)
        }
        Padding::Same => {
            let oh = (h + stride - 1) / stride;
            let ow = (w + stride - 1) / stride;
            let pad_h = ((oh - 1) * stride + kh - h).max(0);
            let pad_w = ((ow - 1) * stride + kw - w).max(0);
            (oh, ow, pad_h, pad_w)
        }
    };
    if n <= 0 || f <= 0 || c <= 0 || oh <= 0 || ow <= 0 {
        return Err(RuntimeError::ShapeMismatch);
    }
    // Symmetric split; the odd pixel lands bottom/right (documented).
    let (pad_top, pad_left) = (pad_h / 2, pad_w / 2);
    let mut out = vec![0.0f64; (n * f * oh * ow) as usize];
    for bn in 0..n {
        for bf in 0..f {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0;
                    for cc in 0..c {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy * stride + ky - pad_top;
                                let ix = ox * stride + kx - pad_left;
                                // Zero padding: out-of-bounds taps contribute 0.
                                if iy < 0 || iy >= h || ix < 0 || ix >= w {
                                    continue;
                                }
                                let iv = xa[((((bn * c) + cc) * h + iy) * w + ix) as usize];
                                let wv = xb[((((bf * fc) + cc) * kh + ky) * kw + kx) as usize];
                                acc += iv * wv;
                            }
                        }
                    }
                    out[(((bn * f + bf) * oh + oy) * ow + ox) as usize] = acc;
                }
            }
        }
    }
    let shape = Shape::from_dims(&[n, f, oh, ow]).map_err(|_| RuntimeError::ShapeMismatch)?;
    let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
    *d = Value::Tensor { data: out, shape };
    Ok(())
}

/// CEP:WHAT: 2D transpose kernel.
/// CEP:STATUS: complete
/// CEP:FAILURE: ShapeMismatch for non-2D.
/// CEP:ASSUMES: rank-2.
/// CEP:COST: O(elements).
/// CEP:EVIDENCE: tests
fn transpose(table: &mut [Value], dst: u32, a: u32) -> Result<(), RuntimeError> {
    let va = table.get(a as usize).ok_or(RuntimeError::BadSlot)?.clone();
    let (data, shape) = match va {
        Value::Tensor { data, shape } => (data, shape),
        _ => return Err(RuntimeError::UnsupportedValue),
    };
    if shape.rank() != 2 {
        return Err(RuntimeError::ShapeMismatch);
    }
    let (r, c) = (shape.dim(0).unwrap_or(0), shape.dim(1).unwrap_or(0));
    let mut out = vec![0.0f64; data.len()];
    for i in 0..r {
        for j in 0..c {
            out[(j * r + i) as usize] = data[(i * c + j) as usize];
        }
    }
    let shape = Shape::from_dims(&[c, r]).map_err(|_| RuntimeError::ShapeMismatch)?;
    let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
    *d = Value::Tensor { data: out, shape };
    Ok(())
}

/// CEP:WHAT: Deterministic RNG kernel (splitmix64 stream per element).
/// CEP:WHY: graph.rng semantics must be reproducible (HPC determinism
///          38.10): element i derives from splitmix64(seed + i); uniform
///          in [0,1); normal via Box-Muller from two consecutive draws.
/// CEP:STATUS: complete
/// CEP:FAILURE: BadSlot.
/// CEP:ASSUMES: seeded by the op.
/// CEP:COST: O(elements).
/// CEP:EVIDENCE: test `rng_is_deterministic`.
fn rng(table: &mut [Value], dst: u32, seed: u64, dist: RngDist) -> Result<(), RuntimeError> {
    // Output shape: keep the destination's existing shape if it is a
    // tensor; scalar otherwise (1 element).
    let count = table
        .get(dst as usize)
        .ok_or(RuntimeError::BadSlot)?
        .element_count()
        .max(1);
    let shape = table
        .get(dst as usize)
        .ok_or(RuntimeError::BadSlot)?
        .shape();
    let mut out = Vec::with_capacity(count);
    for i in 0..count as u64 {
        let v = splitmix64(seed.wrapping_add(i));
        let u = (v >> 11) as f64 / (1u64 << 53) as f64;
        let sample = match dist {
            RngDist::Uniform => u,
            RngDist::Normal => {
                let v2 = splitmix64(seed.wrapping_add(i).wrapping_add(0x9E37_79B9));
                let u2 = (v2 >> 11) as f64 / (1u64 << 53) as f64;
                // Box-Muller (u, u2 in (0,1)).
                let mag = (-2.0 * u.max(1e-12).ln()).sqrt();
                mag * (core::f64::consts::TAU * u2).cos()
            }
        };
        out.push(sample);
    }
    if shape.rank() == 0 {
        store_f64(table, dst, out[0])
    } else {
        let d = table.get_mut(dst as usize).ok_or(RuntimeError::BadSlot)?;
        *d = Value::Tensor { data: out, shape };
        Ok(())
    }
}

/// CEP:WHAT: splitmix64 step (deterministic mixing).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: rng tests
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// CEP:WHAT: Scalar value as f64.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (caller matched the kind)
/// CEP:ASSUMES: va is F64.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: tests
fn va_scalar(va: &Value) -> f64 {
    match va {
        Value::F64(x) => *x,
        _ => 0.0,
    }
}

/// CEP:WHAT: Shape of a slot's tensor (rank-0 for scalars).
/// CEP:STATUS: complete
/// CEP:FAILURE: BadSlot.
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: matmul tests
fn tensor_shape(table: &[Value], slot: u32) -> Result<Shape, RuntimeError> {
    Ok(table
        .get(slot as usize)
        .ok_or(RuntimeError::BadSlot)?
        .shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::node::MAX_INPUTS;
    use xir_core::op::{BinaryOp, Monoid, Op};
    use xir_core::ty::{ScalarType, Type};
    use xir_levels::level3::LoopProgram;
    use xir_levels::level3::ScheduledOp;
    use xir_levels::level4::lower;

    fn prog_of(
        ops: Vec<(Op, [u32; MAX_INPUTS], u8, u32)>,
        results: Vec<u32>,
        n_values: u32,
    ) -> TargetProgram {
        let sched: Vec<ScheduledOp> = ops
            .into_iter()
            .map(|(op, inputs, n, out)| ScheduledOp {
                op,
                inputs,
                n_inputs: n,
                output: out,
                ty: Type::Scalar(ScalarType::F64),
                cluster: None,
            })
            .collect();
        lower(&LoopProgram {
            params: vec![],
            ops: sched,
            results,
            buffers: vec![],
        })
        .ok()
        .unwrap_or(TargetProgram {
            instrs: vec![],
            results: vec![],
            value_count: n_values,
        })
    }

    // CEP:WHAT: Scalar arithmetic executes exactly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong results.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn scalar_arithmetic() {
        let tp = prog_of(
            vec![
                (Op::Binary(BinaryOp::Add), [0, 1, 0, 0, 0, 0], 2, 2),
                (Op::Binary(BinaryOp::Mul), [0, 1, 0, 0, 0, 0], 2, 3),
            ],
            vec![3],
            4,
        );
        // Emulate param binding: slots 0,1 hold 3.0 and 4.0.
        let mut tp = tp;
        tp.value_count = 4;
        let out = execute(&tp, &[Value::F64(3.0), Value::F64(4.0)]);
        assert!(out.is_ok());
        if let Ok(vals) = out {
            assert_eq!(vals[0], Value::F64(12.0));
        }
    }

    // CEP:WHAT: Tensor elementwise executes element by element.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong elements.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tensor_elementwise() {
        let tp = prog_of(
            vec![(
                Op::Unary(xir_core::op::UnaryOp::Relu),
                [0, 0, 0, 0, 0, 0],
                1,
                1,
            )],
            vec![1],
            2,
        );
        let shape = Shape::from_dims(&[2]).ok().unwrap_or(Shape::scalar());
        let input = Value::tensor(vec![-1.0, 2.0], shape)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[input]);
        assert!(out.is_ok());
        if let Ok(vals) = out {
            assert!(matches!(vals[0], Value::Tensor { .. }), "expected tensor");
            if let Value::Tensor { data, .. } = &vals[0] {
                assert_eq!(data[0], 0.0);
                assert_eq!(data[1], 2.0);
            }
        }
    }

    // CEP:WHAT: Dot products reduce exactly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong sum.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn dot() {
        let tp = prog_of(vec![(Op::Dot, [0, 1, 0, 0, 0, 0], 2, 2)], vec![2], 3);
        let shape = Shape::from_dims(&[3]).ok().unwrap_or(Shape::scalar());
        let x = Value::tensor(vec![1.0, 2.0, 3.0], shape)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[x.clone(), x]);
        assert!(out.is_ok());
        if let Ok(vals) = out {
            assert_eq!(vals[0], Value::F64(14.0));
        }
    }

    // CEP:WHAT: Axis reductions fold correctly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong fold.
    // CEP:ASSUMES: row-major [rows, cols].
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn reduce_axis() {
        let tp = prog_of(
            vec![(
                Op::Reduce {
                    axis: 1,
                    monoid: Monoid::Add,
                },
                [0, 0, 0, 0, 0, 0],
                1,
                1,
            )],
            vec![1],
            2,
        );
        let shape = Shape::from_dims(&[2, 3]).ok().unwrap_or(Shape::scalar());
        let x = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[x]);
        assert!(out.is_ok());
        if let Ok(vals) = out {
            assert!(matches!(vals[0], Value::Tensor { .. }), "expected tensor");
            if let Value::Tensor { data, .. } = &vals[0] {
                assert_eq!(data[0], 6.0);
                assert_eq!(data[1], 15.0);
            }
        }
    }

    // CEP:WHAT: Matmul matches the reference triple loop.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong product.
    // CEP:ASSUMES: 2x3 * 3x2.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn matmul_reference() {
        let tp = prog_of(
            vec![(
                Op::Matmul {
                    transpose_a: false,
                    transpose_b: false,
                },
                [0, 1, 0, 0, 0, 0],
                2,
                2,
            )],
            vec![2],
            3,
        );
        let s23 = Shape::from_dims(&[2, 3]).ok().unwrap_or(Shape::scalar());
        let s32 = Shape::from_dims(&[3, 2]).ok().unwrap_or(Shape::scalar());
        let a = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], s23)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let b = Value::tensor(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], s32)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[a, b]);
        assert!(out.is_ok());
        if let Ok(vals) = out {
            assert!(matches!(vals[0], Value::Tensor { .. }), "expected tensor");
            if let Value::Tensor { data, .. } = &vals[0] {
                assert_eq!(data[0], 58.0);
                assert_eq!(data[1], 64.0);
                assert_eq!(data[2], 139.0);
                assert_eq!(data[3], 154.0);
            }
        }
    }

    // CEP:WHAT: RNG is deterministic for a fixed seed.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rng_is_deterministic() {
        let tp = prog_of(
            vec![(
                Op::Rng {
                    dist: RngDist::Uniform,
                    seed: 42,
                },
                [0, 0, 0, 0, 0, 0],
                0,
                1,
            )],
            vec![1],
            2,
        );
        let o1 = execute(&tp, &[]);
        let o2 = execute(&tp, &[]);
        assert!(o1.is_ok() && o2.is_ok());
        if let (Ok(a), Ok(b)) = (o1, o2) {
            assert_eq!(a[0], b[0]);
        }
    }

    // CEP:WHAT: Valid convolution matches the reference triple loop.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong taps.
    // CEP:ASSUMES: 1x1x3x3 input, 1x1x2x2 filter, stride 1.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test (CEP-26 closing regression)
    #[test]
    fn conv_valid_reference() {
        let tp = prog_of(
            vec![(
                Op::Conv {
                    padding: Padding::Valid,
                    stride: 1,
                },
                [0, 1, 0, 0, 0, 0],
                2,
                2,
            )],
            vec![2],
            3,
        );
        let s_in = Shape::from_dims(&[1, 1, 3, 3])
            .ok()
            .unwrap_or(Shape::scalar());
        let s_f = Shape::from_dims(&[1, 1, 2, 2])
            .ok()
            .unwrap_or(Shape::scalar());
        let a = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], s_in)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let b = Value::tensor(vec![1.0, 0.0, 0.0, 1.0], s_f)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[a, b]);
        assert!(out.is_ok(), "conv failed: {:?}", out.err());
        if let Ok(vals) = out {
            if let Value::Tensor { data, shape } = &vals[0] {
                assert_eq!(data, &vec![6.0, 8.0, 12.0, 14.0]);
                assert_eq!(shape.dim(0), Some(1));
                assert_eq!(shape.dim(1), Some(1));
                assert_eq!(shape.dim(2), Some(2));
                assert_eq!(shape.dim(3), Some(2));
            }
        }
    }

    // CEP:WHAT: Same padding zero-fills the halo (3x3 all-ones input and
    //           filter -> corner 4, edge 6, center 9).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if halo taps are not zero.
    // CEP:ASSUMES: 1x1x3x3 input, 1x1x3x3 filter, stride 1.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn conv_same_zero_fill() {
        let tp = prog_of(
            vec![(
                Op::Conv {
                    padding: Padding::Same,
                    stride: 1,
                },
                [0, 1, 0, 0, 0, 0],
                2,
                2,
            )],
            vec![2],
            3,
        );
        let s33 = Shape::from_dims(&[1, 1, 3, 3])
            .ok()
            .unwrap_or(Shape::scalar());
        let ones9 = Value::tensor(vec![1.0; 9], s33)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[ones9.clone(), ones9]);
        assert!(out.is_ok());
        if let Ok(vals) = out {
            if let Value::Tensor { data, .. } = &vals[0] {
                assert_eq!(data, &vec![4.0, 6.0, 4.0, 6.0, 9.0, 6.0, 4.0, 6.0, 4.0]);
            }
        }
    }

    // CEP:WHAT: Stride 2 subsamples the output grid (4x4 identity-tap case).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong subsampling.
    // CEP:ASSUMES: 1x1x4x4 input, 1x1x2x2 all-ones filter, stride 2.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn conv_stride_reference() {
        let tp = prog_of(
            vec![(
                Op::Conv {
                    padding: Padding::Valid,
                    stride: 2,
                },
                [0, 1, 0, 0, 0, 0],
                2,
                2,
            )],
            vec![2],
            3,
        );
        let s_in = Shape::from_dims(&[1, 1, 4, 4])
            .ok()
            .unwrap_or(Shape::scalar());
        let s_f = Shape::from_dims(&[1, 1, 2, 2])
            .ok()
            .unwrap_or(Shape::scalar());
        let a = Value::tensor(
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            s_in,
        )
        .ok()
        .unwrap_or(Value::F64(0.0));
        let b = Value::tensor(vec![1.0; 4], s_f)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[a, b]);
        assert!(out.is_ok());
        if let Ok(vals) = out {
            if let Value::Tensor { data, .. } = &vals[0] {
                assert_eq!(data, &vec![10.0, 18.0, 42.0, 50.0]);
            }
        }
    }

    // CEP:WHAT: Channel summation and per-filter weights are correct
    //           (C=2 input channels, F=2 output filters).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on channel mixing.
    // CEP:ASSUMES: 1x2x2x2 input, 2x2x1x1 filters, stride 1.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn conv_multichannel_reference() {
        let tp = prog_of(
            vec![(
                Op::Conv {
                    padding: Padding::Valid,
                    stride: 1,
                },
                [0, 1, 0, 0, 0, 0],
                2,
                2,
            )],
            vec![2],
            3,
        );
        let s_in = Shape::from_dims(&[1, 2, 2, 2])
            .ok()
            .unwrap_or(Shape::scalar());
        let s_f = Shape::from_dims(&[2, 2, 1, 1])
            .ok()
            .unwrap_or(Shape::scalar());
        let a = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], s_in)
            .ok()
            .unwrap_or(Value::F64(0.0));
        // f0: (ch0 w=1, ch1 w=1); f1: (ch0 w=2, ch1 w=0).
        let b = Value::tensor(vec![1.0, 1.0, 2.0, 0.0], s_f)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[a, b]);
        assert!(out.is_ok());
        if let Ok(vals) = out {
            if let Value::Tensor { data, .. } = &vals[0] {
                // f0: ch0 + ch1 elementwise; f1: 2*ch0.
                assert_eq!(data, &vec![6.0, 8.0, 10.0, 12.0, 2.0, 4.0, 6.0, 8.0]);
            }
        }
    }

    // CEP:WHAT: Oversized kernels under Valid padding are rejected (audit
    //           F-7: truncating division would silently yield a
    //           partial-window convolution at stride >= 2).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a partial-window conv executes.
    // CEP:ASSUMES: kernel larger than the input, stride 2.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn conv_valid_oversized_kernel_rejected() {
        let tp = prog_of(
            vec![(
                Op::Conv {
                    padding: Padding::Valid,
                    stride: 2,
                },
                [0, 1, 0, 0, 0, 0],
                2,
                2,
            )],
            vec![2],
            3,
        );
        let s_in = Shape::from_dims(&[1, 1, 2, 2])
            .ok()
            .unwrap_or(Shape::scalar());
        let s_f = Shape::from_dims(&[1, 1, 3, 3])
            .ok()
            .unwrap_or(Shape::scalar());
        let a = Value::tensor(vec![1.0; 4], s_in)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let b = Value::tensor(vec![1.0; 9], s_f)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let out = execute(&tp, &[a, b]);
        assert_eq!(out.err(), Some(RuntimeError::ShapeMismatch));
    }

    // CEP:WHAT: Malformed operands fail loudly (rank 2 input, channel
    //           mismatch) — never a guessed result.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if bad shapes execute.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn conv_shape_mismatch() {
        let mk = |ops| prog_of(ops, vec![2], 3);
        let s22 = Shape::from_dims(&[2, 2]).ok().unwrap_or(Shape::scalar());
        let s_f = Shape::from_dims(&[1, 1, 2, 2])
            .ok()
            .unwrap_or(Shape::scalar());
        let rank2 = Value::tensor(vec![1.0; 4], s22)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let filt = Value::tensor(vec![1.0; 4], s_f)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let t1 = mk(vec![(
            Op::Conv {
                padding: Padding::Valid,
                stride: 1,
            },
            [0, 1, 0, 0, 0, 0],
            2,
            2,
        )]);
        // Rank-2 input against a rank-4 filter: ShapeMismatch.
        let r1 = execute(&t1, &[rank2, filt.clone()]);
        assert_eq!(r1.err(), Some(RuntimeError::ShapeMismatch));
        // Channel mismatch (C=1 vs C=2).
        let s_in = Shape::from_dims(&[1, 1, 3, 3])
            .ok()
            .unwrap_or(Shape::scalar());
        let s_f2 = Shape::from_dims(&[1, 2, 2, 2])
            .ok()
            .unwrap_or(Shape::scalar());
        let inp = Value::tensor(vec![1.0; 9], s_in)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let filt2 = Value::tensor(vec![1.0; 8], s_f2)
            .ok()
            .unwrap_or(Value::F64(0.0));
        let r2 = execute(&t1, &[inp, filt2]);
        assert_eq!(r2.err(), Some(RuntimeError::ShapeMismatch));
    }
}
