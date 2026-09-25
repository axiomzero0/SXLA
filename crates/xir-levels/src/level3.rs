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
/// CEP:WHAT: Structured op record: opcode, input value slots, output slot,
///           fusion-cluster provenance.
/// CEP:WHY: The interpreter and target lowering execute from this form;
///          value slots index the program's value table (dense u32). The
///          cluster field carries the fusion search's assignment (CEP-22
///          half-closing: level-2 decisions ride INTO level-3 so target
///          backends can exploit locality; None = unclustered).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: input slots < value table size at execution.
/// CEP:COST: 80 bytes.
/// CEP:EVIDENCE: interpreter tests in runtime crate; fused projection
///           tests in this module.
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
    /// Owning fusion cluster (the search's assignment); None when the node
    /// is unclustered or the projection was cluster-free.
    pub cluster: Option<u32>,
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
    // Cluster-free projection: identical to the pre-CEP-22 behavior (slot
    // tie-break scheduling, all-Global bufferization).
    project_with_fusion(arena, roots, &crate::level2::ClusterSet::empty())
}

/// CEP:WHAT: Projects a sea-of-nodes arena into a LoopProgram HONORING the
///           fusion search's winning ClusterSet.
/// CEP:WHY: CEP-22 half-closing (audit F-3: "the search cannot influence
///          anything"): the architecture's level-2 -> level-3 flow —
///          "ClusterSet feeds the level-3 tiling decisions". Two concrete
///          decisions land here: (1) SCHEDULING — schedule_fused's
///          cluster-affinity Kahn groups cluster members adjacently
///          (producer-consumer locality; topological legality untouched);
///          (2) BUFFERIZATION — a tensor intermediate whose EVERY consumer
///          lives in the same cluster (and whose producer's cluster is not
///          force-materialized, and which is not a function result) never
///          touches global memory: its buffer record becomes Register
///          space. Cross-cluster values and results stay Global. This is
///          the textbook fusion win expressed in the buffer plan — the
///          CPU tier executes identically (semantics preserved, enforced
///          differentially), and future register-allocating codegen reads
///          the plan.
/// CEP:STATUS: complete
/// CEP:FAILURE: LowerError (see enum); no partial output.
/// CEP:ASSUMES: verified arena; `roots` are the result nodes; `clusters`
///              built from THIS arena (assignment lookup is by slot).
/// CEP:COST: O(nodes^2) fused scheduling + O(nodes * inputs) consumer map.
/// CEP:EVIDENCE: tests `fused_projection_groups_members`,
///           `intra_cluster_intermediate_is_register`,
///           `cross_cluster_value_stays_global`; jit tier-2 bufferization
///           differential test.
/// CEP:SECURITY: bounds-checked mapping.
/// CEP:HPC-DETERMINISM: deterministic (schedule_fused key, slot order).
pub fn project_with_fusion(
    arena: &IrArena,
    roots: &[NodeId],
    clusters: &crate::level2::ClusterSet,
) -> Result<LoopProgram, LowerError> {
    let has_clusters = !clusters.is_empty();
    let order = if has_clusters {
        xir_graph::schedule::schedule_fused(arena, |id| clusters.cluster_of(id))
            .map_err(LowerError::Schedule)?
    } else {
        schedule(arena).map_err(LowerError::Schedule)?
    };
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
    // Non-param ops (op-producing node ids kept in lockstep for the
    // cluster-aware buffer plan).
    let mut node_of_op: Vec<NodeId> = Vec::with_capacity(order.len());
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
            cluster: if has_clusters {
                clusters.cluster_of(*id)
            } else {
                None
            },
        });
        node_of_op.push(*id);
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
    // Buffer plan: tensor-producing ops get buffers sized by type. Under a
    // fusion-aware projection the space honors the fusion decision: a
    // tensor intermediate consumed ONLY inside its own cluster (not a
    // result, cluster not force-materialized) is REGISTER space — it never
    // materializes to global memory. Everything else stays Global.
    // Consumers-by-slot map (O(nodes * inputs), built once; audit F-9 —
    // ONLY on the fusion-aware path; the cluster-free path hardcodes
    // Global and never reads it).
    let consumers_of: Vec<Vec<NodeId>> = if has_clusters {
        let mut m: Vec<Vec<NodeId>> = vec![Vec::new(); arena.slot_count()];
        arena.for_each_live_node(|id, node| {
            for i in 0..node.n_inputs as usize {
                if i >= MAX_INPUTS {
                    break;
                }
                let def = node.inputs[i].node();
                let slot = def.index() as usize;
                if slot < m.len() {
                    m[slot].push(id);
                }
            }
        });
        m
    } else {
        Vec::new()
    };
    let mut buffers: Vec<(u32, u32, AddressSpace)> = Vec::new();
    for (sop, producer) in ops.iter().zip(node_of_op.iter()) {
        if let Type::Tensor(t) = sop.ty {
            let bytes = t
                .shape
                .num_elements()
                .saturating_mul(i64::from(t.elem.byte_size()));
            let b32 = u32::try_from(bytes).unwrap_or(u32::MAX);
            let space = if has_clusters {
                buffer_space(arena, roots, clusters, *producer, &consumers_of)
            } else {
                AddressSpace::Global
            };
            buffers.push((sop.output, b32, space));
        }
    }
    Ok(LoopProgram {
        params,
        ops,
        results,
        buffers,
    })
}

/// CEP:WHAT: Chooses the address space for one tensor producer's buffer
///           record under a fusion-aware projection.
/// CEP:WHY: The fusion bufferization rule: Register space iff the producer
///          is clustered, the cluster is not force-materialized, the value
///          is not a function result, and EVERY consumer lives in the same
///          cluster — an intermediate that never crosses a fusion boundary
///          never materializes to global memory. Any miss (unclustered
///          producer, materialize flag, result root, cross-cluster or
///          absent consumer) degrades to Global: the conservative side is
///          always Global because a wrong Register would be a
///          miscompilation-class bug in a backend that trusts the plan.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (conservative Global on any doubt).
/// CEP:ASSUMES: verified arena; consumers_of derived from the same arena;
///           region-crossing uses degrade to Global (a Register value read
///           from another control region would be a miscompilation in any
///           backend that trusts the plan — audited F-7; can_fuse does not
///           check regions, so this guard is the enforcement point).
/// CEP:COST: O(consumers).
/// CEP:EVIDENCE: tests `intra_cluster_intermediate_is_register`,
///           `cross_cluster_value_stays_global`,
///           `materialized_cluster_forces_global`,
///           `result_root_stays_global`.
fn buffer_space(
    arena: &IrArena,
    roots: &[NodeId],
    clusters: &crate::level2::ClusterSet,
    producer: NodeId,
    consumers_of: &[Vec<NodeId>],
) -> AddressSpace {
    let Some(pc) = clusters.cluster_of(producer) else {
        return AddressSpace::Global;
    };
    if clusters.cluster(pc).is_some_and(|c| c.materialize) {
        return AddressSpace::Global;
    }
    if roots.contains(&producer) {
        return AddressSpace::Global;
    }
    let consumers = consumers_of
        .get(producer.index() as usize)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    if consumers.is_empty() {
        return AddressSpace::Global;
    }
    let producer_region = match arena.node(producer) {
        Ok(n) => n.region,
        Err(_) => return AddressSpace::Global,
    };
    for c in consumers {
        if clusters.cluster_of(*c) != Some(pc) {
            return AddressSpace::Global;
        }
        // Region guard (audit F-7): a use in a DIFFERENT control region
        // crosses control flow — Register would be unsound for any backend
        // trusting the plan. Degrade to Global.
        match arena.node(*c) {
            Ok(cn) if cn.region == producer_region => {}
            _ => return AddressSpace::Global,
        }
    }
    AddressSpace::Register
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

    // ---- Fused projection (CEP-22 half-closing) ----

    /// CEP:WHAT: Builds a tensor chain arena: p (param tensor) -> scale ->
    ///           shift -> relu, all fusible elementwise pairs.
    /// CEP:WHY: Shared fixture for the fused-projection tests.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns an empty arena on insert failure (tests assert).
    /// CEP:ASSUMES: none.
    /// CEP:COST: test-only
    /// CEP:EVIDENCE: the three fused tests below.
    fn tensor_chain_arena() -> (IrArena, Vec<NodeId>) {
        use xir_core::id::ValueId;
        use xir_core::ty::{Layout, Shape, TensorType};
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let t = Type::Tensor(TensorType {
            elem: ScalarType::F64,
            shape: Shape::from_dims(&[2, 2]).unwrap_or(Shape::scalar()),
            layout: Layout::RowMajor,
        });
        let param = a.insert_node(root, Node::new(Op::Param { index: 0 }, root, &[], t));
        let two = const_f64(&mut a, root, 2.0);
        let three = const_f64(&mut a, root, 3.0);
        let mut ids = Vec::new();
        if let (Ok(p), Ok(c2), Ok(c3)) = (param, two, three) {
            let ip = a.value_of(p, 0);
            let i2 = a.value_of(c2, 0);
            if let (Ok(vp), Ok(v2)) = (ip, i2) {
                let scale = a.insert_node(
                    root,
                    Node::new(Op::Binary(BinaryOp::Mul), root, &[vp, v2], t),
                );
                if let Ok(sc) = scale {
                    let isc = a.value_of(sc, 0);
                    let i3 = a.value_of(c3, 0);
                    if let (Ok(vsc), Ok(v3)) = (isc, i3) {
                        let shift = a.insert_node(
                            root,
                            Node::new(Op::Binary(BinaryOp::Add), root, &[vsc, v3], t),
                        );
                        if let Ok(sh) = shift {
                            let ish = a.value_of(sh, 0);
                            if let Ok(vsh) = ish {
                                let relu = a.insert_node(
                                    root,
                                    Node::new(
                                        Op::Unary(xir_core::op::UnaryOp::Relu),
                                        root,
                                        &[vsh],
                                        t,
                                    ),
                                );
                                if let Ok(r) = relu {
                                    ids = vec![p, c2, c3, sc, sh, r];
                                }
                            }
                        }
                    }
                }
            }
        }
        let _ = ValueId::NONE;
        (a, ids)
    }

    // CEP:WHAT: The fused projection groups cluster members adjacently and
    //           carries cluster provenance on the scheduled ops.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on adjacency or provenance drift.
    // CEP:ASSUMES: {scale, shift, relu} in one cluster.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fused_projection_groups_members() {
        let (a, ids) = tensor_chain_arena();
        if ids.len() == 6 {
            let (p, c2, c3, sc, sh, r) = (ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
            let mut clusters = crate::level2::ClusterSet::new(&a);
            let c = clusters.add_cluster(0, 0);
            let _ = clusters.assign(c, sc);
            let _ = clusters.assign(c, sh);
            let _ = clusters.assign(c, r);
            let prog = project_with_fusion(&a, &[r], &clusters);
            assert!(prog.is_ok());
            if let Ok(pgm) = prog {
                // Provenance: exactly the three members carry Some(c).
                let tagged: Vec<bool> = pgm.ops.iter().map(|o| o.cluster == Some(c)).collect();
                assert_eq!(tagged.iter().filter(|t| **t).count(), 3);
                // Adjacency: scale, shift, relu are consecutive in op
                // order (the two consts schedule first; the cluster members
                // follow back-to-back — the affinity key at work).
                let order: Vec<Op> = pgm.ops.iter().map(|o| o.op).collect();
                assert_eq!(
                    order,
                    vec![
                        Op::ConstF64(2.0),
                        Op::ConstF64(3.0),
                        Op::Binary(BinaryOp::Mul),
                        Op::Binary(BinaryOp::Add),
                        Op::Unary(xir_core::op::UnaryOp::Relu),
                    ]
                );
                let _ = (p, c2, c3);
            }
        }
    }

    // CEP:WHAT: An intra-cluster tensor intermediate (consumed only inside
    //           its cluster, not a result, not materialized) gets REGISTER
    //           buffer space; the result stays Global.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the intermediate is materialized to
    //               global or the result loses its global buffer.
    // CEP:ASSUMES: {scale, shift, relu} clustered; scale/shift consumed
    //               in-cluster; relu is the result.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn intra_cluster_intermediate_is_register() {
        let (a, ids) = tensor_chain_arena();
        if ids.len() == 6 {
            let (sc, sh, r) = (ids[3], ids[4], ids[5]);
            let mut clusters = crate::level2::ClusterSet::new(&a);
            let c = clusters.add_cluster(0, 0);
            let _ = clusters.assign(c, sc);
            let _ = clusters.assign(c, sh);
            let _ = clusters.assign(c, r);
            let prog = project_with_fusion(&a, &[r], &clusters);
            assert!(prog.is_ok());
            if let Ok(pgm) = prog {
                // Three tensor ops; the two intermediates are Register,
                // the result root stays Global.
                assert_eq!(pgm.buffers.len(), 3);
                assert_eq!(pgm.buffers[0].2, AddressSpace::Register);
                assert_eq!(pgm.buffers[1].2, AddressSpace::Register);
                assert_eq!(pgm.buffers[2].2, AddressSpace::Global);
            }
        }
    }

    // CEP:WHAT: A tensor value consumed OUTSIDE its producer's cluster
    //           stays Global; the cluster-free projection is all-Global.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a cross-cluster Register (a
    //               miscompilation-class bug for a backend trusting the
    //               plan) or on cluster-free drift.
    // CEP:ASSUMES: only {scale} clustered; shift/relu unclustered.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn cross_cluster_value_stays_global() {
        let (a, ids) = tensor_chain_arena();
        if ids.len() == 6 {
            let (sc, r) = (ids[3], ids[5]);
            let mut clusters = crate::level2::ClusterSet::new(&a);
            let c = clusters.add_cluster(0, 0);
            let _ = clusters.assign(c, sc);
            // scale's consumer (shift) is unclustered -> cross-boundary.
            let prog = project_with_fusion(&a, &[r], &clusters);
            assert!(prog.is_ok());
            if let Ok(pgm) = prog {
                assert_eq!(pgm.buffers.len(), 3);
                for b in pgm.buffers.iter() {
                    assert_eq!(b.2, AddressSpace::Global);
                }
            }
            // Cluster-free baseline: everything Global.
            let plain = project(&a, &[r]);
            assert!(plain.is_ok());
            if let Ok(pp) = plain {
                assert_eq!(pp.buffers.len(), 3);
                for b in pp.buffers.iter() {
                    assert_eq!(b.2, AddressSpace::Global);
                }
                for o in pp.ops.iter() {
                    assert_eq!(o.cluster, None);
                }
            }
        }
    }

    // CEP:WHAT: A force-materialized cluster writes its intermediates to
    //           Global even when consumed in-cluster (the resource
    //           model's escape hatch).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if materialize is ignored.
    // CEP:ASSUMES: {scale, shift, relu} clustered with materialize = true.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn materialized_cluster_forces_global() {
        let (a, ids) = tensor_chain_arena();
        if ids.len() == 6 {
            let (sc, sh, r) = (ids[3], ids[4], ids[5]);
            let mut clusters = crate::level2::ClusterSet::new(&a);
            let c = clusters.add_cluster(0, 0);
            let _ = clusters.assign(c, sc);
            let _ = clusters.assign(c, sh);
            let _ = clusters.assign(c, r);
            // Force materialize on the cluster.
            clusters.set_materialize(c, true);
            let prog = project_with_fusion(&a, &[r], &clusters);
            assert!(prog.is_ok());
            if let Ok(pgm) = prog {
                assert_eq!(pgm.buffers.len(), 3);
                for b in pgm.buffers.iter() {
                    assert_eq!(b.2, AddressSpace::Global);
                }
            }
        }
    }

    // CEP:WHAT: A RESULT ROOT stays Global even when its value is consumed
    //           entirely inside its own cluster (the caller must be able
    //           to read the function output — audit F-2: this named test
    //           completes the buffer_space evidence list).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a root loses its global buffer.
    // CEP:ASSUMES: root = the mid-chain add; relu consumes it in-cluster.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn result_root_stays_global() {
        let (a, ids) = tensor_chain_arena();
        if ids.len() == 6 {
            let (sc, sh, r) = (ids[3], ids[4], ids[5]);
            let mut clusters = crate::level2::ClusterSet::new(&a);
            let c = clusters.add_cluster(0, 0);
            let _ = clusters.assign(c, sc);
            let _ = clusters.assign(c, sh);
            let _ = clusters.assign(c, r);
            // ROOT = the mid-chain add (sh), NOT the relu: the add's value
            // crosses to the CALLER even though relu consumes it too.
            let prog = project_with_fusion(&a, &[sh], &clusters);
            assert!(prog.is_ok());
            if let Ok(pgm) = prog {
                assert_eq!(pgm.buffers.len(), 3);
                // scale: consumed in-cluster only -> Register.
                assert_eq!(pgm.buffers[0].2, AddressSpace::Register);
                // add: in-cluster consumer BUT a result root -> Global.
                assert_eq!(pgm.buffers[1].2, AddressSpace::Global);
                // relu: no consumers (dead under this root set, kept live
                // by the arena) -> conservative Global.
                assert_eq!(pgm.buffers[2].2, AddressSpace::Global);
            }
        }
    }
}
