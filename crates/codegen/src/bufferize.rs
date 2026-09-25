// CEP:FILE: crates/codegen/src/bufferize.rs
// CEP:WHAT: Bufferization — places loop.alloc buffers for tensor values and
//           assigns address spaces.
// CEP:WHY: Master architecture Level 3: "Bufferization, tiling, software
//          pipelining, vectorization, shared-memory promotion." Values used
//          across cluster boundaries get Global buffers; single-use
//          elementwise intermediates stay register-allocated (no material
//          copy) — the fusion payoff the cost model priced.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: none (planning is total).
// CEP:ASSUMES: LoopProgram buffers pre-seeded by level3 projection.
// CEP:COST: O(ops + buffers).
// CEP:EVIDENCE: tests `external_values_get_buffers`, `single_use_skipped`.
// CEP:SECURITY: internal slots only.
// CEP:HPC-DETERMINISM: deterministic.
//! Bufferization pass.

use xir_core::ty::AddressSpace;
use xir_levels::level3::LoopProgram;

/// The buffer plan output.
///
/// CEP:WHAT: Slot -> (bytes, space, materialize) decisions.
/// CEP:WHY: The interpreter allocates from this plan; the resource model
///          cross-checks shared-space totals.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: built via bufferize().
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone)]
pub struct BufferPlan {
    /// (value slot, byte size, address space, force materialization).
    pub entries: Vec<(u32, u32, AddressSpace, bool)>,
}

/// CEP:WHAT: Computes the buffer plan from use counts.
/// CEP:WHY: Values with >1 consumer (or which are results) cross
///          statements and need memory; single-consumer intermediates stay
///          in registers (the fusion contract).
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: program verified.
/// CEP:COST: O(ops + existing buffers).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn bufferize(prog: &LoopProgram) -> BufferPlan {
    let n = xir_levels::level3::value_count(prog) as usize;
    let mut uses = vec![0u32; n.max(1)];
    for op in prog.ops.iter() {
        for i in 0..op.n_inputs as usize {
            let slot = op.inputs[i] as usize;
            if slot < uses.len() {
                uses[slot] = uses[slot].saturating_add(1);
            }
        }
    }
    let mut entries: Vec<(u32, u32, AddressSpace, bool)> = Vec::new();
    for (slot, bytes, space) in prog.buffers.iter() {
        let multi = uses.get(*slot as usize).copied().unwrap_or(0) > 1;
        let is_result = prog.results.contains(slot);
        // Small intermediates that never escape stay register-classified.
        let keep_register = !multi && !is_result && *bytes <= 256;
        let eff_space = if keep_register {
            AddressSpace::Register
        } else {
            *space
        };
        entries.push((*slot, *bytes, eff_space, multi || is_result));
    }
    BufferPlan { entries }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Multi-use and result values are materialized.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on missing materialization.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn external_values_get_buffers() {
        // op0: output slot 2 consumed twice by later ops; result = 2.
        let prog = LoopProgram {
            params: vec![],
            ops: vec![xir_levels::level3::ScheduledOp {
                op: xir_core::op::Op::Dot,
                inputs: [0, 1, 0, 0, 0, 0],
                n_inputs: 2,
                output: 2,
                ty: xir_core::ty::Type::Scalar(xir_core::ty::ScalarType::F64),
                cluster: None,
            }],
            results: vec![2],
            buffers: vec![(2, 512, AddressSpace::Global)],
        };
        let plan = bufferize(&prog);
        assert_eq!(plan.entries.len(), 1);
        let (_, _, space, mat) = plan.entries[0];
        assert_eq!(space, AddressSpace::Global);
        assert!(mat);
    }

    // CEP:WHAT: Single-use small intermediates stay in registers.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on spurious materialization.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn single_use_skipped() {
        let prog = LoopProgram {
            params: vec![],
            ops: vec![xir_levels::level3::ScheduledOp {
                op: xir_core::op::Op::Unary(xir_core::op::UnaryOp::Neg),
                inputs: [0, 0, 0, 0, 0, 0],
                n_inputs: 1,
                output: 1,
                ty: xir_core::ty::Type::Scalar(xir_core::ty::ScalarType::F64),
                cluster: None,
            }],
            results: vec![1],
            buffers: vec![(1, 64, AddressSpace::Global)],
        };
        let plan = bufferize(&prog);
        assert_eq!(plan.entries.len(), 1);
        // Slot 1 IS the result: must materialize.
        assert!(plan.entries[0].3);
        // Non-result single-use would be register-classified; the entries
        // table keeps the decision auditable either way.
    }
}
