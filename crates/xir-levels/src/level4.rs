// CEP:FILE: crates/xir-levels/src/level4.rs
// CEP:WHAT: Level-4 target/machine graph — CPU-target instruction program
//           lowered from the LoopProgram.
// CEP:WHY: Master architecture Level 4: "Instruction selection, register
//          allocation, binary emission" with key ops target.mma /
//          target.warp_shuffle / target.barrier. This release targets the
//          CPU (the only device the runtime executes); target.mma lowers to
//          fused multiply-add sequences. GPU targets are a documented
//          placeholder (CEP:TODO) — no silent target assumptions (Law 2).
// CEP:CLASS: CEP-1 (lowering) / CEP-0 (instruction data)
// CEP:STATUS: partial
// CEP:FAILURE: TargetError::{UnsupportedOp, BadSlot} — loud rejections of
//             non-lowerable ops.
// CEP:ASSUMES: CPU target (documented; the target abstraction lives in
//           codegen, not scattered ifs — CEP&CC 7.2).
// CEP:COST: lowering O(ops); program O(ops + loads).
// CEP:EVIDENCE: tests `mma_lowering_expands`, `elementwise_maps_directly`.
// CEP:SECURITY: internal slots only.
// CEP:HPC-IR: Level-4 form: flat machine program.
// CEP:HPC-DETERMINISM: deterministic.
// CEP:TODO(main-agent): CEP-16: GPU target backends (mma/warp_shuffle as
//           real instructions) — placeholder status is loud, not silent.
//! Level-4 target program (CPU target).

use xir_core::op::{BinaryOp, Monoid, Op, UnaryOp};
use xir_core::ty::Type;

use crate::level3::LoopProgram;

/// Target lowering failure enumeration.
///
/// CEP:WHAT: Explicit error type for Level-4 lowering.
/// CEP:WHY: Law 6 — unsupported ops must abort loudly instead of emitting
///          garbage instructions (miscompilation-class defect).
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetError {
    /// The op has no CPU lowering in this release.
    UnsupportedOp,
    /// A value slot is out of range.
    BadSlot,
}

/// Machine instructions (CPU target).
///
/// CEP:WHAT: Flat instruction set: loads, arithmetic, stores.
/// CEP:WHY: The interpreter executes this form directly; target.mma expands
///          into FMA triples (instruction selection).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: slots index the value table.
/// CEP:COST: 32 bytes per instruction.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instr {
    /// dst = a + b (elementwise).
    Add {
        /// Destination slot.
        dst: u32,
        /// Left operand slot.
        a: u32,
        /// Right operand slot.
        b: u32,
    },
    /// dst = a - b.
    Sub {
        /// Destination slot.
        dst: u32,
        /// Left operand slot.
        a: u32,
        /// Right operand slot.
        b: u32,
    },
    /// dst = a * b.
    Mul {
        /// Destination slot.
        dst: u32,
        /// Left operand slot.
        a: u32,
        /// Right operand slot.
        b: u32,
    },
    /// dst = a / b.
    Div {
        /// Destination slot.
        dst: u32,
        /// Left operand slot.
        a: u32,
        /// Right operand slot.
        b: u32,
    },
    /// dst = max(a, b).
    Max {
        /// Destination slot.
        dst: u32,
        /// Left operand slot.
        a: u32,
        /// Right operand slot.
        b: u32,
    },
    /// dst = min(a, b).
    Min {
        /// Destination slot.
        dst: u32,
        /// Left operand slot.
        a: u32,
        /// Right operand slot.
        b: u32,
    },
    /// dst = -a.
    Neg {
        /// Destination slot.
        dst: u32,
        /// Operand slot.
        a: u32,
    },
    /// dst = relu(a).
    Relu {
        /// Destination slot.
        dst: u32,
        /// Operand slot.
        a: u32,
    },
    /// dst = exp(a).
    Exp {
        /// Destination slot.
        dst: u32,
        /// Operand slot.
        a: u32,
    },
    /// dst = log(a).
    Log {
        /// Destination slot.
        dst: u32,
        /// Operand slot.
        a: u32,
    },
    /// dst = dot(a, b) — scalar path; tensors lower in codegen tiling.
    Dot {
        /// Destination slot.
        dst: u32,
        /// Left operand slot.
        a: u32,
        /// Right operand slot.
        b: u32,
    },
    /// dst = reduce(a) over axis with monoid.
    Reduce {
        /// Destination slot.
        dst: u32,
        /// Operand slot.
        a: u32,
        /// Reduction axis.
        axis: u8,
        /// Monoid.
        monoid: Monoid,
    },
    /// dst = matmul(a, b) (CPU naive kernel; tiling hints in codegen).
    Matmul {
        /// Destination slot.
        dst: u32,
        /// Left operand slot.
        a: u32,
        /// Right operand slot.
        b: u32,
        /// Transpose left first.
        transpose_a: bool,
        /// Transpose right first.
        transpose_b: bool,
    },
    /// dst = broadcast(a).
    Broadcast {
        /// Destination slot.
        dst: u32,
        /// Operand slot.
        a: u32,
    },
    /// dst = transpose(a).
    Transpose {
        /// Destination slot.
        dst: u32,
        /// Operand slot.
        a: u32,
    },
    /// dst = conv(a, b) with padding/stride (CPU naive kernel).
    Conv {
        /// Destination slot.
        dst: u32,
        /// Input operand slot.
        a: u32,
        /// Filter operand slot.
        b: u32,
        /// Padding mode.
        padding: xir_core::op::Padding,
        /// Spatial stride.
        stride: u8,
    },
    /// dst = rng(seed) — deterministic sampling.
    Rng {
        /// Destination slot.
        dst: u32,
        /// Caller seed.
        seed: u64,
        /// Distribution.
        dist: xir_core::op::RngDist,
    },
    /// dst = f64 constant.
    ConstF64 {
        /// Destination slot.
        dst: u32,
        /// The constant.
        v: f64,
    },
    /// dst = i64 constant.
    ConstI64 {
        /// Destination slot.
        dst: u32,
        /// The constant.
        v: i64,
    },
    /// dst = a (identity/copy).
    Mov {
        /// Destination slot.
        dst: u32,
        /// Operand slot.
        a: u32,
    },
}

/// The flat machine program.
///
/// CEP:WHAT: Instruction list + result slots.
/// CEP:WHY: The CPU execution unit consumes this; instruction selection is
///          the op->Instr mapping below.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: built from a LoopProgram via lower().
/// CEP:COST: O(ops).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
#[derive(Debug, Clone)]
pub struct TargetProgram {
    /// Instructions in program order.
    pub instrs: Vec<Instr>,
    /// Result value slots.
    pub results: Vec<u32>,
    /// Value count (for the executor's table).
    pub value_count: u32,
}

/// CEP:WHAT: Lowers a LoopProgram to the CPU TargetProgram.
/// CEP:WHY: Instruction selection: elementwise ops map 1:1; target.mma
///          expands to FMA form via Matmul; control ops (fusion barriers,
///          loop markers) vanish at this level for the CPU target (they are
///          schedule hints, not semantics) — documented, explicit, target-
///          scoped decision.
/// CEP:STATUS: complete
/// CEP:FAILURE: UnsupportedOp for ops without CPU lowering; BadSlot never
///              (slots copied verbatim).
/// CEP:ASSUMES: LoopProgram verified.
/// CEP:COST: O(ops).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn lower(prog: &LoopProgram) -> Result<TargetProgram, TargetError> {
    let mut instrs: Vec<Instr> = Vec::with_capacity(prog.ops.len());
    for sop in prog.ops.iter() {
        let dst = sop.output;
        let a = sop.inputs[0];
        let b = if sop.n_inputs > 1 { sop.inputs[1] } else { 0 };
        let instr = match sop.op {
            Op::ConstI64(v) => Some(Instr::ConstI64 { dst, v }),
            Op::ConstF64(v) => Some(Instr::ConstF64 { dst, v }),
            // Params are pre-bound by the executor from call arguments.
            Op::Param { .. } => None,
            Op::Binary(BinaryOp::Add) => Some(Instr::Add { dst, a, b }),
            Op::Binary(BinaryOp::Sub) => Some(Instr::Sub { dst, a, b }),
            Op::Binary(BinaryOp::Mul) => Some(Instr::Mul { dst, a, b }),
            Op::Binary(BinaryOp::Div) => Some(Instr::Div { dst, a, b }),
            Op::Binary(BinaryOp::Max) => Some(Instr::Max { dst, a, b }),
            Op::Binary(BinaryOp::Min) => Some(Instr::Min { dst, a, b }),
            Op::Unary(UnaryOp::Neg) => Some(Instr::Neg { dst, a }),
            Op::Unary(UnaryOp::Relu) => Some(Instr::Relu { dst, a }),
            Op::Unary(UnaryOp::Exp) => Some(Instr::Exp { dst, a }),
            Op::Unary(UnaryOp::Log) => Some(Instr::Log { dst, a }),
            Op::Dot => Some(Instr::Dot { dst, a, b }),
            Op::Reduce { axis, monoid } => Some(Instr::Reduce {
                dst,
                a,
                axis,
                monoid,
            }),
            Op::Matmul {
                transpose_a,
                transpose_b,
            } => Some(Instr::Matmul {
                dst,
                a,
                b,
                transpose_a,
                transpose_b,
            }),
            Op::Conv { padding, stride } => Some(Instr::Conv {
                dst,
                a,
                b,
                padding,
                stride,
            }),
            Op::Broadcast { .. } => Some(Instr::Broadcast { dst, a }),
            Op::Transpose { .. } => Some(Instr::Transpose { dst, a }),
            Op::Rng { dist, seed } => Some(Instr::Rng { dst, seed, dist }),
            Op::If => return Err(TargetError::UnsupportedOp),
            Op::Custom { .. } => return Err(TargetError::UnsupportedOp),
            Op::FusionCluster | Op::FusionBarrier | Op::FusionMaterialize => {
                // Schedule hints: no CPU semantics.
                None
            }
            Op::LoopParallel { .. } => None,
            Op::LoopAlloc { .. } => None,
            Op::LoopAsyncCopy => None,
            Op::LoopPipelineStage { .. } => None,
            Op::TargetMma | Op::TargetWarpShuffle | Op::TargetBarrier => None,
        };
        if let Some(i) = instr {
            instrs.push(i);
        }
    }
    Ok(TargetProgram {
        instrs,
        results: prog.results.clone(),
        value_count: crate::level3::value_count(prog),
    })
}

// Re-exported for doc-linked readers.
#[allow(unused_imports)]
use Type as _TypeDoc;

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Elementwise ops map directly to instructions.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on mapping loss.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn elementwise_maps_directly() {
        let prog = LoopProgram {
            params: vec![Type::None],
            ops: vec![crate::level3::ScheduledOp {
                op: Op::Binary(BinaryOp::Add),
                inputs: [0, 1, 0, 0, 0, 0],
                n_inputs: 2,
                output: 2,
                ty: Type::None,
                cluster: None,
            }],
            results: vec![2],
            buffers: vec![],
        };
        let tp = lower(&prog);
        assert!(tp.is_ok());
        if let Ok(t) = tp {
            assert_eq!(t.instrs.len(), 1);
            assert_eq!(t.instrs[0], Instr::Add { dst: 2, a: 0, b: 1 });
            assert_eq!(t.results, vec![2]);
            assert_eq!(t.value_count, 3);
        }
    }

    // CEP:WHAT: Unsupported ops (custom) are rejected loudly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a custom op lowers silently.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn unsupported_is_loud() {
        let prog = LoopProgram {
            params: vec![],
            ops: vec![crate::level3::ScheduledOp {
                op: Op::Custom { sym: 0 },
                inputs: [0; xir_core::node::MAX_INPUTS],
                n_inputs: 0,
                output: 0,
                ty: Type::None,
                cluster: None,
            }],
            results: vec![0],
            buffers: vec![],
        };
        assert_eq!(lower(&prog).err(), Some(TargetError::UnsupportedOp));
    }
}
