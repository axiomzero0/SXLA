// CEP:FILE: crates/codegen/src/lower.rs
// CEP:WHAT: Target lowering — LoopProgram to CPU TargetProgram via the
//           xir-levels Level-4 lowering, with tiling and bufferization
//           metadata attached.
// CEP:WHY: The driver that sequences Level-3 passes (bufferize, tile) and
//           the Level-4 instruction selection. Executed by the runtime and
//           the JIT tiers.
// CEP:CLASS: CEP-1 (driver) / CEP-0 (pass cores)
// CEP:STATUS: complete
// CEP:FAILURE: TargetError propagation from Level-4 lowering.
// CEP:ASSUMES: verified LoopProgram.
// CEP:COST: O(ops).
// CEP:EVIDENCE: tests `lower_end_to_end`.
// CEP:SECURITY: internal slots only.
// CEP:HPC-DETERMINISM: deterministic.
//! Target lowering driver.

use xir_levels::level3::LoopProgram;
use xir_levels::level4::{lower, TargetError, TargetProgram};

use crate::bufferize::{bufferize, BufferPlan};

/// The full lowering output.
///
/// CEP:WHAT: Target program + buffer plan + pass provenance.
/// CEP:WHY: The runtime executes the target program; the buffer plan drives
///          allocation; provenance feeds the pipeline manifest.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: built via lower_target.
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
pub struct LoweredProgram {
    /// The CPU target program.
    pub target: TargetProgram,
    /// Bufferization decisions.
    pub buffers: BufferPlan,
}

/// CEP:WHAT: Runs bufferization then Level-4 lowering.
/// CEP:WHY: One entry point for the JIT tiers; pass order is the manifest
///          contract (38.20).
/// CEP:STATUS: complete
/// CEP:FAILURE: TargetError propagation (unsupported ops are loud).
/// CEP:ASSUMES: program verified.
/// CEP:COST: O(ops).
/// CEP:EVIDENCE: test `lower_end_to_end`.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn lower_target(prog: &LoopProgram) -> Result<LoweredProgram, TargetError> {
    let buffers = bufferize(prog);
    let target = lower(prog)?;
    Ok(LoweredProgram { target, buffers })
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: A flat program lowers end-to-end.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on lowering failure.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn lower_end_to_end() {
        let prog = LoopProgram {
            params: vec![xir_core::ty::Type::Scalar(xir_core::ty::ScalarType::F64)],
            ops: vec![xir_levels::level3::ScheduledOp {
                op: xir_core::op::Op::Unary(xir_core::op::UnaryOp::Relu),
                inputs: [0, 0, 0, 0, 0, 0],
                n_inputs: 1,
                output: 1,
                ty: xir_core::ty::Type::Scalar(xir_core::ty::ScalarType::F64),
                cluster: None,
            }],
            results: vec![1],
            buffers: vec![],
        };
        let out = lower_target(&prog);
        assert!(out.is_ok());
        if let Ok(l) = out {
            assert_eq!(l.target.instrs.len(), 1);
            assert_eq!(l.target.results, vec![1]);
        }
    }
}
