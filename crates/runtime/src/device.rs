// CEP:FILE: crates/runtime/src/device.rs
// CEP:WHAT: The CPU device — program launch over the reference interpreter.
// CEP:WHY: Master architecture section 8: "Device execution." The device
//          abstracts execution for the JIT tiers; the CPU device executes
//          TargetPrograms directly (single-device release; GPU devices are
//          the documented CEP-16 placeholder).
// CEP:CLASS: CEP-1 (API) / CEP-0 (execution)
// CEP:STATUS: complete
// CEP:FAILURE: RuntimeError propagation.
// CEP:ASSUMES: lowered programs.
// CEP:COST: interpreter cost (interp.rs module header).
// CEP:EVIDENCE: tests `device_executes`.
// CEP:SECURITY: bounded execution (slot checks).
// CEP:HPC-DETERMINISM: deterministic.
//! CPU device.

use xir_levels::level4::TargetProgram;

use crate::interp::{execute, RuntimeError};
use crate::value::Value;

/// The CPU device handle.
///
/// CEP:WHAT: Execution entry for lowered programs.
/// CEP:WHY: One object owns the target policy (CPU) so callers never
///          scatter target switches (CEP&CC 7.2 target abstraction).
/// CEP:STATUS: complete
/// CEP:FAILURE: see RuntimeError.
/// CEP:ASSUMES: none
/// CEP:COST: see interp.rs
/// CEP:EVIDENCE: tests in this module
pub struct CpuDevice;

impl CpuDevice {
    /// CEP:WHAT: Executes a target program with bound arguments.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: RuntimeError propagation.
    /// CEP:ASSUMES: params match the program signature.
    /// CEP:COST: interpreter cost.
    /// CEP:EVIDENCE: tests in this module.
    pub fn launch(
        &self,
        program: &TargetProgram,
        args: &[Value],
    ) -> Result<Vec<Value>, RuntimeError> {
        execute(program, args)
    }

    /// CEP:WHAT: Device name (diagnostics).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: constant
    /// CEP:EVIDENCE: tests
    pub fn name(&self) -> &'static str {
        "cpu-generic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Type};
    use xir_levels::level3::{LoopProgram, ScheduledOp};
    use xir_levels::level4::lower;

    // CEP:WHAT: The device executes lowered programs.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on execution failure.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn device_executes() {
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
        let tp = lower(&prog);
        assert!(tp.is_ok());
        if let Ok(t) = tp {
            let dev = CpuDevice;
            assert_eq!(dev.name(), "cpu-generic");
            let out = dev.launch(&t, &[Value::F64(20.0), Value::F64(22.0)]);
            assert!(out.is_ok());
            if let Ok(vals) = out {
                assert_eq!(vals[0], Value::F64(42.0));
            }
        }
    }
}
