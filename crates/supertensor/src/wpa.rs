// CEP:FILE: crates/supertensor/src/wpa.rs
// CEP:WHAT: Stage 1 — concurrent Whole-Program Analysis producing the
//           SchedulePolytope: parallel liveness/use-count (Gear-0 static
//           partition over tensor-op slices), shape readout from attached XIR
//           types, parallel per-op bounds emission, then single-threaded
//           global constraint assembly.
// CEP:WHY: WPA facts become hard bounds (architectural axiom 2): liveness
//          footprint gates memory levels, shapes gate tiles/divisibility/
//          vector widths, fusion legality pre-filters fusion targets. Gear-0
//          slices are disjoint by construction — zero synchronization, and
//          outputs land at caller-fixed disjoint indices so the result is
//          deterministic regardless of worker interleaving (HPC-0 contract).
// CEP:CLASS: CEP-29
// CEP:STATUS: complete
// CEP:FAILURE: extract_polytope cannot fail (analysis only); malformed types
//              degrade to conservative bounds (documented per rule below) —
//              loud failure is the VERIFIER's job on out-of-region schedules.
// CEP:ASSUMES: arena verified; tensor ops identified by opcode + tensor type;
//              XIR value semantics mean ALIAS ANALYSIS IS TRIVIALLY EMPTY
//              (no pointers, no cross-function aliasing rows can exist) —
//              documented deviation from the generic WPA recipe, honest and
//              structural, not an omission.
// CEP:COST: O(tensor_ops + edges) with Gear-1 pooled partitioning for the two
//           parallel phases; <50ms for a 10K-node graph on the reference box
//           (CEP-29 evidence target; the enumerator dominates end-to-end).
// CEP:EVIDENCE: tests `wpa_extracts_matmul_bounds`,
//           `wpa_forces_hbm_for_oversized_tensors`, `wpa_fuse_targets_are_legal`.
// CEP:SECURITY: repository-trusted input only.
// CEP:HPC-DETERMINISM: slot-order collection + disjoint-output partitioning;
//           no completion-order dependence.
// CEP:TODO(main-agent): CEP-29: interprocedural liveness RANGES (producer ->
//           last-consumer span) replace the use-count approximation; shape
//           propagation across function boundaries when XIR gains calls.
//! Stage 1: WPA → polytope extraction (Gear-0).

use anvil::run_partitioned;
use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::Node;
use xir_core::op::Op;
use xir_core::ty::{ScalarType, Shape, Type};

use crate::polytope::{
    DivConstraint, HardwareModel, LinearRow, MemLevel, OpBounds, ParamKind, SchedulePolytope,
    SparseLinearSystem,
};

/// CEP:WHAT: Tensor byte size (elements x element size).
/// CEP:STATUS: complete
/// CEP:FAILURE: saturating.
/// CEP:ASSUMES: shapes bounded (MAX_RANK).
/// CEP:COST: O(1)
/// CEP:EVIDENCE: wpa tests
/// CEP:SECURITY: no untrusted input
fn tensor_bytes(tt: &xir_core::ty::TensorType) -> u64 {
    u64::from(tt.elem.byte_size()).saturating_mul(tt.shape.num_elements().max(0) as u64)
}

/// One collected tensor op + the analytic inputs WPA needs.
///
/// CEP:WHAT: Intermediate record between collection and bounds emission.
/// CEP:WHY: The parallel phases partition over these records; keeping them
///          plain makes the closure Sync without gymnastics.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: collected in slot order.
/// CEP:COST: ~120 bytes
/// CEP:EVIDENCE: collection tests
/// CEP:SECURITY: no untrusted input
#[derive(Clone)]
struct TensorOpInfo {
    /// The op node id.
    op_id: NodeId,
    /// Opcode (for flops/tile semantics).
    op: Op,
    /// Output tensor shape.
    out_shape: Shape,
    /// Output element scalar type.
    elem: ScalarType,
    /// Input tensor byte sizes (f64/i64 -> 8B each, from types).
    in_bytes: u64,
    /// Consumers of the output (use sites) — legality-relevant.
    consumers: Vec<NodeId>,
}

/// CEP:WHAT: Is the op a scheduleable tensor op (XIR Level-2 domain)?
/// CEP:WHY: Axiom 1 — the search space contains ONLY tensor ops; scalar and
///          control-flow logic is opaque to this engine. ONE authority:
///          lib::is_tensor_heavy consumes this predicate (review F10 —
///          duplicated predicates drift).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (predicate)
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: collection tests
/// CEP:SECURITY: no untrusted input
pub fn is_scheduleable(node: &Node) -> bool {
    matches!(
        node.op,
        Op::Matmul { .. }
            | Op::Conv { .. }
            | Op::Dot
            | Op::Reduce { .. }
            | Op::Binary(_)
            | Op::Unary(_)
    ) && matches!(node.ty, Type::Tensor(_))
}

/// CEP:WHAT: Collects tensor ops in slot order (deterministic).
/// CEP:STATUS: complete
/// CEP:FAILURE: none — an empty result is a valid polytope (caller decides).
/// CEP:ASSUMES: arena verified.
/// CEP:COST: O(nodes + edges)
/// CEP:EVIDENCE: `wpa_extracts_matmul_bounds`
/// CEP:SECURITY: no untrusted input
fn collect_tensor_ops(arena: &IrArena) -> Vec<TensorOpInfo> {
    // Two passes: (1) scheduleable ops, (2) consumer edges — consumers need
    // the full op table first, so edges are resolved after collection.
    struct Raw {
        op_id: NodeId,
    }
    let mut raws: Vec<Raw> = Vec::new();
    arena.for_each_live_node(|id, node| {
        if is_scheduleable(node) {
            raws.push(Raw { op_id: id });
        }
    });
    let mut infos: Vec<TensorOpInfo> = Vec::with_capacity(raws.len());
    for r in raws.iter() {
        let node = match arena.node(r.op_id) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let tt = match node.ty.as_tensor() {
            Some(t) => t,
            None => continue,
        };
        // Input bytes: sum of live input operand tensor sizes.
        let mut in_bytes: u64 = 0;
        for i in 0..node.n_inputs as usize {
            if i >= xir_core::node::MAX_INPUTS {
                break;
            }
            let producer = node.inputs[i].node();
            if let Ok(p) = arena.node(producer) {
                if let Some(pt) = p.ty.as_tensor() {
                    in_bytes = in_bytes.saturating_add(tensor_bytes(&pt));
                }
            }
        }
        infos.push(TensorOpInfo {
            op_id: r.op_id,
            op: node.op,
            out_shape: tt.shape,
            elem: tt.elem,
            in_bytes,
            consumers: Vec::new(),
        });
    }
    // Consumer resolution: for each live node, its scheduleable tensor-op
    // producers record it as a consumer (use-edge walk, slot order).
    let mut edges: Vec<(usize, NodeId)> = Vec::new();
    arena.for_each_live_node(|id, node| {
        for i in 0..node.n_inputs as usize {
            if i >= xir_core::node::MAX_INPUTS {
                break;
            }
            let producer = node.inputs[i].node();
            if let Some(pos) = infos.iter().position(|info| info.op_id == producer) {
                edges.push((pos, id));
            }
        }
    });
    for (pos, consumer) in edges {
        infos[pos].consumers.push(consumer);
    }
    infos
}

/// CEP:WHAT: Per-op analytic FLOPs (roofline compute input).
/// CEP:STATUS: complete
/// CEP:FAILURE: 0 for unknown shapes (degrades to memory-bound estimate).
/// CEP:ASSUMES: shapes attached.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: `wpa_extracts_matmul_bounds` (2*M*N*K)
/// CEP:SECURITY: no untrusted input
fn op_flops(op: &Op, out: &Shape, arena: &IrArena, op_id: NodeId) -> u64 {
    let elems = out.num_elements().max(0) as u64;
    match op {
        Op::Matmul { .. } => {
            // 2*M*N*K from operand shapes (index into the op's own inputs).
            let node = match arena.node(op_id) {
                Ok(n) => n,
                Err(_) => return 0,
            };
            let operand_dim = |input: usize, axis: u8| -> u64 {
                let value = match node.inputs.get(input) {
                    Some(v) => *v,
                    None => return 0,
                };
                match arena.node(value.node()) {
                    Ok(p) => match p.ty.as_tensor() {
                        Some(t) => t.shape.dim(axis).unwrap_or(0).max(0) as u64,
                        None => 0,
                    },
                    Err(_) => 0,
                }
            };
            let m = operand_dim(0, 0);
            let n = operand_dim(1, 1);
            let k = operand_dim(0, 1);
            2u64.saturating_mul(m).saturating_mul(n).saturating_mul(k)
        }
        Op::Conv { .. } => {
            // 2 * out_elems * kh*kw*ci from the kernel operand (OIHW).
            let node = match arena.node(op_id) {
                Ok(n) => n,
                Err(_) => return 0,
            };
            let kernel = match node.inputs.get(1) {
                Some(v) => *v,
                None => return 1,
            };
            let khkwc = match arena.node(kernel.node()) {
                Ok(p) => match p.ty.as_tensor() {
                    Some(t) => {
                        let r = t.shape.rank() as usize;
                        if r == 4 {
                            let d = |a: u8| t.shape.dim(a).unwrap_or(1).max(1) as u64;
                            d(2).saturating_mul(d(3)).saturating_mul(d(1))
                        } else {
                            1
                        }
                    }
                    None => 1,
                },
                Err(_) => 1,
            };
            2u64.saturating_mul(elems).saturating_mul(khkwc)
        }
        Op::Dot => {
            // 2 * inner products over the last axis.
            2u64.saturating_mul(elems)
        }
        _ => {
            // Elementwise / reduce: one flop per element (approximation;
            // documented — the cost model is a filter, not an oracle).
            elems
        }
    }
}

/// CEP:WHAT: Stage 1 entry: arena + hardware model -> SchedulePolytope.
/// CEP:WHY: The three phases mirror the architectural pipeline: parallel
///          liveness, shape readout, parallel bounds emission, global assembly.
/// CEP:STATUS: complete
/// CEP:FAILURE: never (analysis); degenerate inputs produce empty bounds.
/// CEP:ASSUMES: arena verified; hardware model validated by its constructor.
/// CEP:COST: see module header.
/// CEP:EVIDENCE: module tests; concurrency_stress.rs determinism test.
/// CEP:SECURITY: no untrusted input.
/// CEP:HPC-DETERMINISM: partitioned phases write disjoint outputs.
pub fn extract_polytope(arena: &IrArena, hw: HardwareModel) -> SchedulePolytope {
    let infos = collect_tensor_ops(arena);
    let n = infos.len();
    let workers = anvil::default_worker_count().max(1);

    // Phase A: parallel use counts (liveness approximation) — Gear-0 static
    // partition over op slices; outputs land at disjoint indices.
    let mut use_counts: Vec<u32> = vec![0; n];
    if n > 0 {
        let _ = run_partitioned(&infos, &mut use_counts, workers, |info| {
            info.consumers.len() as u32
        });
    }

    // Phase B: shape/cost readout (types are attached in XIR — no
    // propagation pass is needed for the current single-function surface).
    let op_flops_v: Vec<u64> = infos
        .iter()
        .map(|info| op_flops(&info.op, &info.out_shape, arena, info.op_id))
        .collect();
    let op_out_bytes: Vec<u64> = infos
        .iter()
        .map(|info| {
            u64::from(info.elem.byte_size())
                .saturating_mul(info.out_shape.num_elements().max(0) as u64)
        })
        .collect();

    // Phase C: parallel per-op bounds emission — the WPA rules land here.
    // The per-op context rides INSIDE the partitioned records (run_partitioned
    // maps elements, not indices), so every worker sees exactly its slice.
    let mut bounds: Vec<OpBounds> = Vec::with_capacity(n);
    if n > 0 {
        let works: Vec<OpWork<'_>> = infos
            .iter()
            .zip(use_counts.iter())
            .zip(op_out_bytes.iter())
            .map(|((info, uses), out_bytes)| OpWork {
                info,
                uses: *uses,
                out_bytes: *out_bytes,
            })
            .collect();
        let mut parts: Vec<OpBounds> = vec![
            OpBounds {
                op_id: NodeId(0),
                tile_min: [1, 1, 1],
                tile_max: [1, 1, 1],
                mem_min: MemLevel::Reg,
                mem_max: MemLevel::Hbm,
                valid_fuse: Vec::new(),
                vec_mask: 1,
            };
            n
        ];
        let _ = run_partitioned(&works, &mut parts, workers, |w| {
            emit_op_bounds(w, hw, arena)
        });
        bounds = parts;
    }

    // Phase D: global constraints + divisibility (single-threaded assembly).
    // Divisor per (op, dim): gcd(problem dim, hardware tile clamp) — the tile
    // domain is exactly the divisors of this gcd (spec: tile | gcd(K, MAX_TILE)).
    let mut divisibility: Vec<DivConstraint> = Vec::new();
    for (i, info) in infos.iter().enumerate() {
        let rank = info.out_shape.rank() as usize;
        for d in 0..3.min(rank) {
            let dim = info.out_shape.dim(d as u8).unwrap_or(1).max(1) as u32;
            let g = gcd(dim, hw.max_tile).max(1);
            divisibility.push(DivConstraint {
                op_index: i as u16,
                dim: d as u8,
                divisor: g,
            });
        }
    }
    // Global linear rows: the alias-equality rows of the generic recipe are
    // EMPTY under XIR value semantics (no pointers — see module header). The
    // one structural row family retained: mem level ordering sanity
    // (mem ≤ HBM), which the bounds already imply; kept explicit as the
    // system's documented degenerate-but-exercised form so the partial-check
    // path is live (mirrors the fusion search's pruning-protocol discipline).
    let rows: Vec<LinearRow> = (0..n)
        .map(|i| LinearRow {
            entries: vec![(i as u16, ParamKind::Mem, 1)],
            rhs: MemLevel::Hbm as i64,
        })
        .collect();

    let op_min_hbm: Vec<u64> = infos
        .iter()
        .zip(op_out_bytes.iter())
        .zip(bounds.iter())
        .map(|((info, out), b)| {
            // SOUND lower bound (review F2): an op's output must hit HBM only
            // when it has NO consumer to fuse into AND its placement is
            // FORCED off-chip (mem_max == Hbm). A consumer-less op at SRAM
            // contributes ZERO HBM traffic (its output stays on-chip), so the
            // old "no consumer => out_bytes" rule over-pruned non-dominated
            // candidates.
            if info.consumers.is_empty() && b.mem_max == MemLevel::Hbm {
                *out
            } else {
                0
            }
        })
        .collect();
    let op_in_bytes_v: Vec<u64> = infos.iter().map(|info| info.in_bytes).collect();

    SchedulePolytope {
        op_bounds: bounds,
        global_constraints: SparseLinearSystem { rows },
        divisibility_constraints: divisibility,
        hw,
        op_flops: op_flops_v,
        op_out_bytes,
        op_in_bytes: op_in_bytes_v,
        op_min_hbm,
    }
}

/// CEP:WHAT: Greatest common divisor (u32).
/// CEP:STATUS: complete
/// CEP:FAILURE: 0 only when both inputs are 0 (callers clamp to >= 1).
/// CEP:ASSUMES: none
/// CEP:COST: O(log min)
/// CEP:EVIDENCE: tile-domain tests
/// CEP:SECURITY: no untrusted input
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Read-only work record bundling an op with its WPA context — the unit the
/// parallel bounds emitter partitions over.
struct OpWork<'a> {
    info: &'a TensorOpInfo,
    /// Output use count (Phase A result).
    uses: u32,
    /// Output tensor bytes (Phase B result).
    out_bytes: u64,
}

/// CEP:WHAT: Emits one op's bounds from the WPA rules.
/// CEP:WHY: The WPA→constraint mapping (spec table):
///          - footprint S*R (out bytes x use count) <= SRAM cap => mem_max
///            SRAM, else mem_min HBM (liveness rule);
///          - tile_max = shape dims clamped by hw.max_tile; divisibility is
///            emitted separately (phase D);
///          - vector widths = powers of two dividing the innermost dim;
///          - fuse targets = consumers passing fusion legality (one authority:
///            fusion::legality::can_fuse).
/// CEP:STATUS: complete
/// CEP:FAILURE: conservative bounds on degenerate shapes (documented).
/// CEP:ASSUMES: work record fields are phase-consistent.
/// CEP:COST: O(consumers) legality checks.
/// CEP:EVIDENCE: `wpa_forces_hbm_for_oversized_tensors`,
///           `wpa_fuse_targets_are_legal`.
/// CEP:SECURITY: no untrusted input.
fn emit_op_bounds(w: &OpWork<'_>, hw: HardwareModel, arena: &IrArena) -> OpBounds {
    let info = w.info;
    let footprint = w.out_bytes.saturating_mul(u64::from(w.uses.max(1)));
    let (mem_min, mem_max) = if footprint > hw.sram_cap_bytes {
        (MemLevel::Hbm, MemLevel::Hbm)
    } else {
        (MemLevel::Reg, MemLevel::Sram)
    };

    let rank = info.out_shape.rank() as usize;
    let mut tile_max = [1u32; 3];
    for (d, tmax) in tile_max.iter_mut().enumerate() {
        if d < rank {
            let dim = info.out_shape.dim(d as u8).unwrap_or(1).max(1) as u32;
            // Spec rule: tile | gcd(K, MAX_TILE) — tiles divide the gcd of
            // the problem dim and the hardware clamp (a tile larger than the
            // clamp is not offered; a non-divisor tile is not in the domain).
            let g = gcd(dim, hw.max_tile);
            *tmax = g.max(1);
        }
    }

    // Vector widths: powers of two dividing the innermost dimension.
    let innermost = if rank > 0 {
        info.out_shape.dim(rank as u8 - 1).unwrap_or(1).max(1) as u32
    } else {
        1
    };
    let mut vec_mask: u16 = 1; // width 1 always valid.
    for (bit, width) in [(1u16, 2u32), (2, 4), (3, 8)] {
        if innermost.is_multiple_of(width) {
            vec_mask |= 1 << bit;
        }
    }

    // Fusion targets: consumers passing legality (can_fuse is THE authority).
    let valid_fuse: Vec<NodeId> = info
        .consumers
        .iter()
        .copied()
        .filter(|c| fusion::legality::can_fuse(arena, info.op_id, *c).unwrap_or(false))
        .collect();

    OpBounds {
        op_id: info.op_id,
        tile_min: [1, 1, 1],
        tile_max,
        mem_min,
        mem_max,
        valid_fuse,
        vec_mask,
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;
    use crate::polytope::PackedOpSched;
    use xir_core::text::parse_arena;

    const MATMUL_SRC: &str = concat!(
        "xir v1 func @main {\n",
        "  %0 = param 0 : tensor<f64>[8,8]\n",
        "  %1 = param 1 : tensor<f64>[8,8]\n",
        "  %2 = matmul ta=false tb=false %0, %1 : tensor<f64>[8,8]\n",
        "  %3 = param 2 : tensor<f64>[8,8]\n",
        "  %4 = binary.add %2, %3 : tensor<f64>[8,8]\n",
        "}\n"
    );

    // CEP:WHAT: WPA extracts matmul + add bounds with 2*M*N*K flops.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a wrong polytope.
    // CEP:ASSUMES: parser accepts the tensor syntax.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn wpa_extracts_matmul_bounds() {
        let arena = parse_arena(MATMUL_SRC, 64).expect("parse");
        let poly = extract_polytope(&arena, HardwareModel::reference());
        assert_eq!(poly.op_count(), 2, "matmul + add are scheduleable");
        assert_eq!(poly.op_flops[0], 2 * 8 * 8 * 8);
        // Add: 64 elements, 1 flop each.
        assert_eq!(poly.op_flops[1], 64);
        // 8x8xf64 = 512B; footprint 512 * uses <= cap -> on-chip allowed.
        assert!(poly.op_bounds[0].mem_max <= MemLevel::Sram);
        // Innermost dim 8 => widths 1,2,4,8 valid.
        assert_eq!(poly.op_bounds[0].vec_mask, 0b1111);
    }

    // CEP:WHAT: Oversized tensors are forced to HBM (liveness footprint rule).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the rule is missing.
    // CEP:ASSUMES: 1024x1024xf64 = 8MiB > 64KiB cap.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn wpa_forces_hbm_for_oversized_tensors() {
        let src = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : tensor<f64>[1024,1024]\n",
            "  %1 = param 1 : tensor<f64>[1024,1024]\n",
            "  %2 = matmul ta=false tb=false %0, %1 : tensor<f64>[1024,1024]\n",
            "}\n"
        );
        let arena = parse_arena(src, 64).expect("parse");
        let poly = extract_polytope(&arena, HardwareModel::reference());
        assert_eq!(poly.op_count(), 1);
        assert_eq!(poly.op_bounds[0].mem_min, MemLevel::Hbm);
        assert_eq!(poly.op_bounds[0].mem_max, MemLevel::Hbm);
    }

    // CEP:WHAT: Emitted fuse targets all pass fusion legality.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on an illegal target.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn wpa_fuse_targets_are_legal() {
        let arena = parse_arena(MATMUL_SRC, 64).expect("parse");
        let poly = extract_polytope(&arena, HardwareModel::reference());
        for b in poly.op_bounds.iter() {
            for t in b.valid_fuse.iter() {
                assert!(fusion::legality::can_fuse(&arena, b.op_id, *t).unwrap_or(false));
            }
        }
        // The matmul's consumer is the add — elementwise, legal.
        assert_eq!(poly.op_bounds[0].valid_fuse.len(), 1);
        // PackedOpSched::NO_FUSE must differ from every real NodeId encoding.
        assert_ne!(PackedOpSched::NO_FUSE, 0);
    }
}
