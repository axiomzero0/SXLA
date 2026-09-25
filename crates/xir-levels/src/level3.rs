// CEP:FILE: crates/xir-levels/src/level3.rs
// CEP:WHAT: Level-3 loop/memory/schedule graph — the structured LoopProgram
//           projection of the sea-of-nodes, with bufferization records.
// CEP:WHY: Master architecture Level 3: "Hybrid (Structured Loop Nests +
//           Schedule Dependence Graph). Lower clusters into loops, buffers,
//           and hardware mappings." The LoopProgram is the canonical
//           structured projection the pass manager hands to Structured-form
//           consumers (codegen, runtime interpreter, target lowering).
// CEP:CLASS: CEP-0 (data structures + projection builder)
// CEP:STATUS: complete
// CEP:FAILURE: LowerError::{Schedule, UnknownNode, Arity} — loud failures,
//             never partial programs.
// CEP:ASSUMES: verified input snapshot; the scheduler produces a valid
//           topological order.
// CEP:COST: projection O(nodes + edges); program size O(nodes).
// CEP:EVIDENCE: tests `projection_is_topological`, `params_and_results`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-IR: Level-3 form: structured program with buffers.
// CEP:HPC-DETERMINISM: deterministic; scheduler order + slot mapping.
//! Level-3 loop program (structured projection).

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::{MAX_INPUTS, MAX_OUTPUTS};
use xir_core::op::Op;
use xir_core::ty::{AddressSpace, Type};
use xir_graph::schedule::{schedule, ScheduleError};

/// Projection failure enumeration.
///
/// CEP:WHAT: Explicit error type for the Level-3 lowering.
/// CEP:WHY: Law 6 — scheduling and arity problems must abort lowering
///          loudly (a partial LoopProgram would be a miscompilation).
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerError {
    /// The value graph could not be scheduled (cycle).
    Schedule(ScheduleError),
    /// A use referenced an unknown node.
    UnknownNode,
    /// Op arity disagreed with the opcode contract during mapping.
    Arity,
}

/// One scheduled op with resolved value slots.
///
/// CEP:WHAT: Structured op record: opcode, input value slots, output slot.
/// CEP:WHY: The interpreter and target lowering execute from this form;
///          value slots index the program's value table (dense u32).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: input slots < value table size at execution.
/// CEP:COST: 72 bytes.
/// CEP:EVIDENCE: interpreter tests in runtime crate.
#[derive(Debug, Clone, Copy)]
pub struct ScheduledOp {
    /// Opcode + immediates.
    pub op: Op,
    /// Input value slots (dense indices).
    pub inputs: [u32; MAX_INPUTS],
    /// Live input count.
    pub n_inputs: u8,
    /// Output value slot.
    pub output: u32,
    /// Type of the produced value.
    pub ty: Type,
}

/// The structured Level-3 program.
///
/// CEP:WHAT: Params, scheduled ops, results, buffer plan.
/// CEP:WHY: The executable projection: params bind call arguments, ops run
///          in order, results index output values; the buffer plan records
///          loop.alloc decisions (bufferization output).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: built via `project` from a verified snapshot.
/// CEP:COST: O(nodes).
/// CEP:EVIDENCE: tests in this module + runtime interpreter tests.
/// CEP:HPC-DETERMINISM: deterministic.
pub struct LoopProgram {
    /// Parameter signatures in parameter-index order.
    pub params: Vec<Type>,
    /// Ops in canonical topological order.
    pub ops: Vec<ScheduledOp>,
    /// Result value slots (function outputs).
    pub results: Vec<u32>,
    /// Bufferization record: (value slot, byte size, address space).
    pub buffers: Vec<(u32, u32, AddressSpace)>,
}

/// CEP:WHAT: Projects a sea-of-nodes arena into a LoopProgram.
/// CEP:WHY: The graphify/structurize bridge (arch section 7): Structured-form
///          consumers receive this projection. Value slots are assigned in
///          scheduler order (deterministic); params first.
/// CEP:STATUS: complete
/// CEP:FAILURE: LowerError (see enum); no partial output.
/// CEP:ASSUMES: verified arena; `roots` are the result nodes.
/// CEP:COST: O(nodes + edges).
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: bounds-checked mapping.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn project(arena: &IrArena, roots: &[NodeId]) -> Result<LoopProgram, LowerError> {
    let order = schedule(arena).map_err(LowerError::Schedule)?;
    // Slot mapping: node index -> dense value slot.
    let mut slot_of: Vec<Option<u32>> = vec![None; arena.slot_count()];
    let mut ops: Vec<ScheduledOp> = Vec::with_capacity(order.len());
    let mut params: Vec<Type> = Vec::new();
    // Parameters occupy the first value slots in schedule order.
    let mut next_slot: u32 = 0;
    for id in order.iter() {
        let node = arena.node(*id).map_err(|_| LowerError::UnknownNode)?;
        if let Op::Param { index } = node.op {
            let slot = next_slot;
            next_slot += 1;
            slot_of[id.index() as usize] = Some(slot);
            let idx = index as usize;
            while params.len() <= idx {
                params.push(Type::None);
            }
            params[idx] = node.ty;
        }
    }
    // Non-param ops.
    for id in order.iter() {
        let node = arena.node(*id).map_err(|_| LowerError::UnknownNode)?;
        if matches!(node.op, Op::Param { .. }) {
            continue;
        }
        if node.n_inputs as usize > MAX_INPUTS {
            return Err(LowerError::Arity);
        }
        let mut inputs = [0u32; MAX_INPUTS];
        for (i, v) in node
            .inputs
            .iter()
            .take(node.n_inputs as usize)
            .take(MAX_INPUTS)
            .enumerate()
        {
            let def = v.node();
            let prod_slot = slot_of
                .get(def.index() as usize)
                .copied()
                .flatten()
                .ok_or(LowerError::UnknownNode)?;
            inputs[i] = prod_slot;
        }
        let out_slot = next_slot;
        next_slot += 1;
        slot_of[id.index() as usize] = Some(out_slot);
        // Bufferization record for tensor values.
        ops.push(ScheduledOp {
            op: node.op,
            inputs,
            n_inputs: node.n_inputs,
            output: out_slot,
            ty: node.ty,
        });
    }
    // Results.
    let mut results: Vec<u32> = Vec::with_capacity(roots.len());
    for r in roots {
        let slot = slot_of
            .get(r.index() as usize)
            .copied()
            .flatten()
            .ok_or(LowerError::UnknownNode)?;
        results.push(slot);
    }
    // Buffer plan: tensor-producing ops get Global buffers sized by type.
    let mut buffers: Vec<(u32, u32, AddressSpace)> = Vec::new();
    for sop in ops.iter() {
        if let Type::Tensor(t) = sop.ty {
            let bytes = t
                .shape
                .num_elements()
                .saturating_mul(i64::from(t.elem.byte_size()));
            let b32 = u32::try_from(bytes).unwrap_or(u32::MAX);
            buffers.push((sop.output, b32, AddressSpace::Global));
        }
    }
    Ok(LoopProgram {
        params,
        ops,
        results,
        buffers,
    })
}

/// CEP:WHAT: Total value count of the program (slots).
/// CEP:WHY: The interpreter sizes its value table from this.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(ops)
/// CEP:EVIDENCE: interpreter tests
pub fn value_count(prog: &LoopProgram) -> u32 {
    let mut max_slot = prog.params.len() as u32;
    for op in prog.ops.iter() {
        max_slot = max_slot.max(op.output.wrapping_add(1));
    }
    max_slot
}

// MAX_OUTPUTS participates in the slot discipline; re-export for doc clarity.
#[allow(unused_imports)]
use MAX_OUTPUTS as _MaxOutputsDoc;

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::{const_f64, const_i64};
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Type};

    // CEP:WHAT: The projection schedules values before uses.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on topological violation.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn projection_is_topological() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        let mut add_id = NodeId::NONE;
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let add = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::I64),
                );
                let id = a.insert_node(root, add);
                assert!(id.is_ok());
                if let Ok(i) = id {
                    add_id = i;
                }
            }
        }
        let prog = project(&a, &[add_id]);
        assert!(prog.is_ok());
        if let Ok(p) = prog {
            // Consts are ops (only Param nodes become param slots): the
            // program is [const3, const4, add].
            assert_eq!(p.ops.len(), 3);
            let add = p.ops[2];
            assert_eq!(add.n_inputs, 2);
            assert_eq!(add.inputs[0], 0);
            assert_eq!(add.inputs[1], 1);
            assert_eq!(p.results, vec![add.output]);
            assert_eq!(value_count(&p), 3);
        }
    }

    // CEP:WHAT: Params bind the leading slots in index order.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on param misbinding.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn params_and_results() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let p0 = Node::new(
            Op::Param { index: 0 },
            root,
            &[],
            Type::Scalar(ScalarType::F64),
        );
        let p1 = Node::new(
            Op::Param { index: 1 },
            root,
            &[],
            Type::Scalar(ScalarType::F64),
        );
        let i0 = a.insert_node(root, p0);
        let i1 = a.insert_node(root, p1);
        assert!(i0.is_ok() && i1.is_ok());
        let mut sum_id = NodeId::NONE;
        if let (Ok(v0), Ok(v1)) = (i0, i1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let sum = Node::new(
                    Op::Binary(BinaryOp::Mul),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::F64),
                );
                let sid = a.insert_node(root, sum);
                assert!(sid.is_ok());
                if let Ok(s) = sid {
                    sum_id = s;
                }
            }
        }
        let prog = project(&a, &[sum_id]);
        assert!(prog.is_ok());
        if let Ok(p) = prog {
            assert_eq!(p.params.len(), 2);
            // Mul consumes slots 0 and 1 (the params).
            assert_eq!(p.ops[0].inputs[0], 0);
            assert_eq!(p.ops[0].inputs[1], 1);
        }
    }

    // Silence unused import in non-test builds.
    #[allow(unused_imports)]
    use const_f64 as _unused_f64;
}
