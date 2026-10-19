// CEP:FILE: crates/supertensor/src/cost.rs
// CEP:WHAT: The analytic roofline cost evaluator — compute cycles from
//           FLOPs/peak(level, vec_width), memory cycles from per-level
//           traffic/bandwidth, roofline max, plus the HBM byte and SRAM
//           footprint outputs the Pareto frontier trades off.
// CEP:WHY: Axiom 3: expensive verification never runs in the hot loop — the
//          analytic model IS the hot filter. Integer arithmetic (ceil-div)
//          keeps evaluation deterministic and allocation-free.
// CEP:CLASS: CEP-29
// CEP:STATUS: complete
// CEP:FAILURE: saturating arithmetic everywhere; unknown placements degrade
//              to the HBM (slowest) assumption — conservative, documented.
// CEP:ASSUMES: schedule indices align with the polytope's op order.
// CEP:COST: O(ops) per candidate; no allocation.
// CEP:EVIDENCE: tests `roofline_is_max_of_compute_and_memory`,
//           `fusion_removes_hbm_traffic`, `vec_width_scales_peak`.
// CEP:SECURITY: no untrusted input.
//! Analytic roofline cost model.

use crate::polytope::{MemLevel, PackedOpSched, SchedulePolytope};

/// The evaluated cost vector (Pareto objectives).
///
/// CEP:WHAT: cycles (roofline), hbm_bytes (off-chip traffic), sram_usage
///           (on-chip footprint).
/// CEP:WHY: Multi-objective frontier — runtime vs traffic vs footprint do not
///          collapse into one scalar early (the architecture's Pareto rule).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: none
/// CEP:COST: 24 bytes, Copy
/// CEP:EVIDENCE: pareto tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostPoint {
    /// Roofline cycles (max of compute and memory time).
    pub cycles: u64,
    /// Off-chip HBM traffic in bytes.
    pub hbm_bytes: u64,
    /// On-chip SRAM footprint in bytes.
    pub sram_usage: u64,
}

/// CEP:WHAT: a strictly dominates b (<= on all axes, < on at least one).
/// CEP:WHY: The frontier's admission predicate; strictness prevents
///          duplicate-point accumulation.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (predicate)
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: pareto tests
/// CEP:SECURITY: no untrusted input
pub fn dominates(a: &CostPoint, b: &CostPoint) -> bool {
    (a.cycles <= b.cycles && a.hbm_bytes <= b.hbm_bytes && a.sram_usage <= b.sram_usage)
        && (a.cycles < b.cycles || a.hbm_bytes < b.hbm_bytes || a.sram_usage < b.sram_usage)
}

/// CEP:WHAT: Ceil division for non-negative integers.
/// CEP:STATUS: complete
/// CEP:FAILURE: 0 when d == 0 (callers pass nonzero constants).
/// CEP:ASSUMES: d > 0.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: roofline tests
/// CEP:SECURITY: no untrusted input
fn ceil_div(n: u64, d: u64) -> u64 {
    if d == 0 {
        return 0;
    }
    n.div_ceil(d)
}

/// CEP:WHAT: Effective compute peak (FLOP/cycle) at a placement + width.
/// CEP:WHY: Vectorization scales the peak linearly in lanes (width/8 of the
///          full-width peak, min 1 lane); REG >= SRAM >= HBM peaks model the
///          bandwidth-fed regimes.
/// CEP:STATUS: complete
/// CEP:FAILURE: saturating.
/// CEP:ASSUMES: width in 1..=8.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: `vec_width_scales_peak`
/// CEP:SECURITY: no untrusted input
fn peak_flops(hw: &crate::polytope::HardwareModel, mem: MemLevel, width: u8) -> u64 {
    // The listed peaks are FULL-WIDTH (8-lane) achievable rates; the rate
    // scales linearly down with the vector width, floored at 1 FLOP/cycle.
    let base = match mem {
        MemLevel::Reg => hw.reg_peak_flops_per_cycle,
        MemLevel::Sram => hw.sram_peak_flops_per_cycle,
        MemLevel::Hbm => hw.hbm_peak_flops_per_cycle,
    };
    let lanes = u64::from(width.clamp(1, 8));
    (base.saturating_mul(lanes) / 8).max(1)
}

/// CEP:WHAT: Evaluates one complete schedule against the polytope's analytic
///           tables — the roofline: cycles = max(Σ compute, Σ memory-time).
/// CEP:WHY: Fusion accounting: an op fused into its consumer keeps its output
///          out of HBM (register epilogue); an op whose PRODUCER fused into it
///          reads that input from registers instead of HBM.
/// CEP:STATUS: complete
/// CEP:FAILURE: saturating sums.
/// CEP:ASSUMES: sched aligned with polytope ops.
/// CEP:COST: O(ops), zero allocation.
/// CEP:EVIDENCE: module tests.
/// CEP:SECURITY: no untrusted input.
pub fn evaluate(sched: &[PackedOpSched], poly: &SchedulePolytope) -> CostPoint {
    let hw = &poly.hw;
    // Pass 1: which ops are fused-into-consumer (their output skips HBM).
    let mut compute_cycles: u64 = 0;
    let mut hbm_bytes: u64 = 0;
    let mut sram_traffic: u64 = 0;
    let mut sram_usage: u64 = 0;
    for (i, s) in sched.iter().enumerate() {
        let mem = s.mem().unwrap_or(MemLevel::Hbm);
        // Compute term: flops at the effective peak.
        let flops = poly.op_flops.get(i).copied().unwrap_or(0);
        let peak = peak_flops(hw, mem, s.vec_width).max(1);
        compute_cycles = compute_cycles.saturating_add(ceil_div(flops, peak));
        // Output traffic: fused outputs stay on-chip.
        let out_fused = s.fuse_target != PackedOpSched::NO_FUSE;
        let out_bytes = poly.op_out_bytes.get(i).copied().unwrap_or(0);
        match mem {
            MemLevel::Hbm => {
                hbm_bytes = hbm_bytes.saturating_add(out_bytes);
            }
            MemLevel::Sram => {
                sram_traffic = sram_traffic.saturating_add(out_bytes);
                sram_usage = sram_usage.saturating_add(out_bytes);
            }
            MemLevel::Reg => {
                // Register-resident: no memory traffic accounted (the compute
                // peak already models register feeding); a REG output that is
                // NOT fused still must spill to HBM (conservative).
                if !out_fused {
                    hbm_bytes = hbm_bytes.saturating_add(out_bytes);
                }
            }
        }
    }
    // Pass 2: input traffic — an input read from a producer that fused into
    // THIS op comes from registers (skip); otherwise HBM.
    for (i, s) in sched.iter().enumerate() {
        let in_bytes = poly.op_in_bytes.get(i).copied().unwrap_or(0);
        if in_bytes == 0 {
            continue;
        }
        // Find whether any earlier op fused INTO this op.
        let mut fed_by_fusion = false;
        for p in sched.iter().take(i) {
            if p.fuse_target == s.op_id {
                fed_by_fusion = true;
                break;
            }
        }
        if !fed_by_fusion {
            hbm_bytes = hbm_bytes.saturating_add(in_bytes);
        }
    }
    let mem_cycles = ceil_div(hbm_bytes, hw.hbm_bw_bytes_per_cycle)
        .saturating_add(ceil_div(sram_traffic, hw.sram_bw_bytes_per_cycle));
    CostPoint {
        cycles: compute_cycles.max(mem_cycles),
        hbm_bytes,
        sram_usage,
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
    use crate::polytope::{
        DivConstraint, HardwareModel, MemLevel, OpBounds, PackedOpSched, SchedulePolytope,
        SparseLinearSystem,
    };
    use xir_core::id::NodeId;

    fn poly(flops: u64, out_bytes: u64, in_bytes: u64, cap: u64) -> SchedulePolytope {
        SchedulePolytope {
            op_bounds: vec![OpBounds {
                op_id: NodeId(1),
                tile_min: [1, 1, 1],
                tile_max: [64, 64, 64],
                mem_min: MemLevel::Reg,
                mem_max: MemLevel::Hbm,
                valid_fuse: vec![NodeId(2)],
                vec_mask: 0b1111,
            }],
            global_constraints: SparseLinearSystem { rows: vec![] },
            divisibility_constraints: vec![DivConstraint {
                op_index: 0,
                dim: 0,
                divisor: 64,
            }],
            hw: HardwareModel {
                sram_cap_bytes: cap,
                reg_peak_flops_per_cycle: 32,
                sram_peak_flops_per_cycle: 16,
                hbm_peak_flops_per_cycle: 4,
                sram_bw_bytes_per_cycle: 256,
                hbm_bw_bytes_per_cycle: 64,
                max_tile: 256,
            },
            op_flops: vec![flops],
            op_out_bytes: vec![out_bytes],
            op_in_bytes: vec![in_bytes],
            op_min_hbm: vec![out_bytes],
        }
    }

    fn sched(mem: MemLevel, fuse: u64, vw: u8) -> Vec<PackedOpSched> {
        vec![PackedOpSched {
            op_id: 1,
            tile: [64, 64, 64],
            fuse_target: fuse,
            mem_level: mem as u8,
            vec_width: vw,
        }]
    }

    // CEP:WHAT: Roofline = max(compute, memory) on a compute-bound schedule.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a wrong roofline.
    // CEP:ASSUMES: out 512B + in 1024B at HBM bw 64B/cyc = 24 cyc; compute
    //              1024 flops at HBM full-width peak 4, width 1 -> rate 1
    //              -> 1024 cyc; max = 1024.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn roofline_is_max_of_compute_and_memory() {
        let p = poly(1024, 512, 1024, 64 * 1024);
        let c = evaluate(&sched(MemLevel::Hbm, PackedOpSched::NO_FUSE, 1), &p);
        assert_eq!(c.cycles, 1024);
        assert_eq!(c.hbm_bytes, 512 + 1024);
    }

    // CEP:WHAT: Fusion removes the output's HBM traffic.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if fusion accounting is missing.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fusion_removes_hbm_traffic() {
        let p = poly(1024, 512, 1024, 64 * 1024);
        let unfused = evaluate(&sched(MemLevel::Hbm, PackedOpSched::NO_FUSE, 1), &p);
        // Same schedule but SRAM + fused: output stays on-chip.
        let fused = evaluate(&sched(MemLevel::Sram, 2, 1), &p);
        assert!(fused.hbm_bytes < unfused.hbm_bytes);
        assert_eq!(fused.hbm_bytes, 1024); // inputs only
    }

    // CEP:WHAT: Wider vectors raise the effective peak (fewer cycles).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if scaling is inverted.
    // CEP:ASSUMES: compute-bound regime.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn vec_width_scales_peak() {
        let p = poly(8192, 1, 1, 64 * 1024);
        let w1 = evaluate(&sched(MemLevel::Reg, PackedOpSched::NO_FUSE, 1), &p);
        let w8 = evaluate(&sched(MemLevel::Reg, PackedOpSched::NO_FUSE, 8), &p);
        assert!(w8.cycles < w1.cycles);
    }
}
