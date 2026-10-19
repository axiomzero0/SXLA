// CEP:FILE: crates/supertensor/src/polytope.rs
// CEP:WHAT: The Schedule Polytope — the central WPA artifact: per-op parameter
//           bounds (tiles, memory level, fusion targets, vector widths), global
//           linear constraints in sparse row form, divisibility constraints,
//           and the hardware model baked in. Plus the packed per-op schedule
//           record used on the typed SPSC hot path.
// CEP:WHY: WPA facts become HARD bounds on schedule parameters: no candidate
//           outside the region is ever enumerated (architectural axiom 2).
//           O(1) per-parameter feasibility checks during enumeration; the
//           sparse row evaluation touches only the rows whose parameters are
//           already bound (partial depth check).
// CEP:CLASS: CEP-29
// CEP:STATUS: complete
// CEP:FAILURE: `check_full` returns `ConstraintViolation` naming the exact
//              failed constraint class — never a panic, never a guess (38.22).
// CEP:ASSUMES: op indices are positions in `op_bounds` (deterministic slot
//              order); NodeIds belong to the arena the polytope was extracted
//              from; shapes/types attached in XIR are authoritative.
// CEP:COST: check_full = O(ops + rows + divisibility); check_partial =
//           O(rows touching bound prefix). All data plain; no allocation in
//           the check paths.
// CEP:EVIDENCE: tests `gemm_schedule_satisfies_polytope`,
//           `divisibility_violation_rejected`, `linear_violation_rejected`,
//           `sram_overflow_rejected`, `partial_check_skips_unbound_rows`.
// CEP:SECURITY: no untrusted input (repository-trusted XIR).
// CEP:HPC-DETERMINISM: fully deterministic data; no time/hash dependence.
//! Schedule polytope: the feasible region produced by WPA.

use xir_core::id::NodeId;

/// Memory placement level for an op's output tensor.
///
/// CEP:WHAT: REG < SRAM < HBM total order (Ord-derived).
/// CEP:WHY: The WPA liveness/footprint rules emit bounds in this order; the
///          enumerator walks it descending (HBM-first is the safe default).
/// CEP:STATUS: complete
/// CEP:FAILURE: from_u8 returns None outside 0..=2 (loud, no panic).
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: tests in this module
/// CEP:SECURITY: no untrusted input
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemLevel {
    /// Output lives in registers (fused epilogue placement).
    Reg = 0,
    /// Output lives in on-chip shared memory.
    Sram = 1,
    /// Output lives in off-chip global memory.
    Hbm = 2,
}

impl MemLevel {
    /// CEP:WHAT: Decodes a level from its discriminant.
    /// CEP:WHY: The packed schedule record stores a u8.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None outside 0..=2.
    /// CEP:ASSUMES: none
    /// CEP:COST: branch
    /// CEP:EVIDENCE: roundtrip in `gemm_schedule_satisfies_polytope`
    /// CEP:SECURITY: bounded input
    pub const fn from_u8(v: u8) -> Option<MemLevel> {
        match v {
            0 => Some(MemLevel::Reg),
            1 => Some(MemLevel::Sram),
            2 => Some(MemLevel::Hbm),
            _ => None,
        }
    }
}

/// The analytic hardware model baked into the polytope.
///
/// CEP:WHAT: Integer constants: SRAM capacity, per-level compute peaks
///           (FLOP/cycle, at 8-wide vectorization), per-level bandwidths
///           (bytes/cycle), and the tile clamp.
/// CEP:WHY: The cost model is an analytic ROOFLINE (architectural axiom 3:
///          SMT never in the hot loop; the analytic model is the hot filter).
///           Integer arithmetic keeps it deterministic and allocation-free.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (constants).
/// CEP:ASSUMES: reference CPU target documented in docs/targets.md; the model
///              is a filter, not an oracle (CEP-29 cost-model refinement is
///              the feedback loop's job).
/// CEP:COST: 40 bytes, copied by value.
/// CEP:EVIDENCE: cost tests bound the roofline outputs.
/// CEP:SECURITY: no untrusted input.
#[derive(Debug, Clone, Copy)]
pub struct HardwareModel {
    /// On-chip capacity gating the WPA footprint rules (bytes).
    pub sram_cap_bytes: u64,
    /// Peak FLOP/cycle at REG placement (full vector width).
    pub reg_peak_flops_per_cycle: u64,
    /// Peak FLOP/cycle at SRAM placement.
    pub sram_peak_flops_per_cycle: u64,
    /// Peak FLOP/cycle at HBM placement (bandwidth-bound regime).
    pub hbm_peak_flops_per_cycle: u64,
    /// SRAM bandwidth (bytes/cycle).
    pub sram_bw_bytes_per_cycle: u64,
    /// HBM bandwidth (bytes/cycle).
    pub hbm_bw_bytes_per_cycle: u64,
    /// Maximum tile dimension clamp (WPA shape rule).
    pub max_tile: u32,
}

impl HardwareModel {
    /// CEP:WHAT: The reference CPU model (documented constants).
    /// CEP:WHY: One authority for the numbers (Law 7); `from_env` style
    ///          configuration is a CEP-29 follow-up — env-driven models would
    ///          make compilation results machine-config dependent, which the
    ///          determinism contract forbids until keyed into the fingerprint.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: reference target (docs/targets.md).
    /// CEP:COST: constant
    /// CEP:EVIDENCE: cost model tests
    /// CEP:SECURITY: no untrusted input
    pub const fn reference() -> HardwareModel {
        HardwareModel {
            sram_cap_bytes: 64 * 1024,
            reg_peak_flops_per_cycle: 32,
            sram_peak_flops_per_cycle: 16,
            hbm_peak_flops_per_cycle: 4,
            sram_bw_bytes_per_cycle: 256,
            hbm_bw_bytes_per_cycle: 64,
            max_tile: 256,
        }
    }
}

/// Which scalar schedule parameter a linear-constraint entry references.
///
/// CEP:WHAT: Parameter selector for sparse rows.
/// CEP:WHY: Rows are expressed over (op_index, ParamKind) pairs; this keeps
///          the CSR-style evaluation dispatch trivial and exhaustive.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: linear row tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    /// The op's memory level (as i64).
    Mem,
    /// Tile dimension 0.
    Tile0,
    /// Tile dimension 1.
    Tile1,
    /// Tile dimension 2.
    Tile2,
    /// Vector width (as i64).
    Vec,
}

/// One sparse linear inequality: Σ coeff_i × param_i ≤ rhs.
///
/// CEP:WHAT: A row over at most a few (op, param) pairs.
/// CEP:WHY: The WPA global rules (e.g. equality-coupled memory placements)
///          land here; evaluation is a short dot product.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: entries reference ops inside the polytope.
/// CEP:COST: O(entries) per evaluation.
/// CEP:EVIDENCE: `linear_violation_rejected`
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone)]
pub struct LinearRow {
    /// (op index, parameter, signed coefficient).
    pub entries: Vec<(u16, ParamKind, i64)>,
    /// Right-hand side bound.
    pub rhs: i64,
}

/// The sparse linear system: all global rows.
///
/// CEP:WHAT: Row collection with full and partial (prefix) evaluation.
/// CEP:WHY: Partial checks let the enumerator reject a prefix as soon as all
///          of a row's parameters are bound — the spec's CSR fast path.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (predicate API)
/// CEP:ASSUMES: rows are emitted once (WPA phase D).
/// CEP:COST: full = O(Σ entries); partial = O(Σ entries over bound prefix).
/// CEP:EVIDENCE: `partial_check_skips_unbound_rows`
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Default)]
pub struct SparseLinearSystem {
    /// Rows in emission order (deterministic).
    pub rows: Vec<LinearRow>,
}

impl SparseLinearSystem {
    /// CEP:WHAT: Evaluates every row against a COMPLETE schedule.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: false on any violated row.
    /// CEP:ASSUMES: sched entries align with op indices.
    /// CEP:COST: O(Σ entries)
    /// CEP:EVIDENCE: `linear_violation_rejected`
    /// CEP:SECURITY: no untrusted input
    pub fn check_full(&self, sched: &[PackedOpSched]) -> bool {
        self.rows.iter().all(|row| {
            let mut acc: i64 = 0;
            for (op, param, coeff) in row.entries.iter() {
                let v = sched
                    .get(*op as usize)
                    .map(|s| s.param_value(*param))
                    .unwrap_or(0);
                acc = acc.saturating_add(coeff.saturating_mul(v));
            }
            acc <= row.rhs
        })
    }

    /// CEP:WHAT: Evaluates only the rows whose parameters are ALL bound at
    ///           prefix depth `depth` (ops 0..depth scheduled).
    /// CEP:WHY: The enumerator's early rejection path.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: false on any violated bound-prefix row.
    /// CEP:ASSUMES: sched.len() >= depth.
    /// CEP:COST: O(Σ entries) with early skip of future rows.
    /// CEP:EVIDENCE: `partial_check_skips_unbound_rows`
    /// CEP:SECURITY: no untrusted input
    pub fn check_partial(&self, sched: &[PackedOpSched], depth: usize) -> bool {
        self.rows.iter().all(|row| {
            let bound = row.entries.iter().all(|(op, _, _)| (*op as usize) < depth);
            if !bound {
                return true;
            }
            let mut acc: i64 = 0;
            for (op, param, coeff) in row.entries.iter() {
                let v = sched[*op as usize].param_value(*param);
                acc = acc.saturating_add(coeff.saturating_mul(v));
            }
            acc <= row.rhs
        })
    }
}

/// Divisibility constraint: tile dim `dim` of op `op_index` must DIVIDE
/// `divisor` (the gcd of the problem dim and the hardware tile clamp —
/// spec: "tile | gcd(K, MAX_TILE)").
///
/// CEP:WHAT: Non-linear but trivially checkable domain restriction.
/// CEP:WHY: Kept separate from the linear system by the spec's design.
///          NOTE the direction: the tile is a DIVISOR of the problem dim
///          (a tile of 4 covering a dim of 8 is legal; a tile of 3 is not).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: op_index < op count; dim < 3.
/// CEP:COST: 8 bytes
/// CEP:EVIDENCE: `divisibility_violation_rejected`
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy)]
pub struct DivConstraint {
    /// Op index into op_bounds.
    pub op_index: u16,
    /// Tile dimension (0..3).
    pub dim: u8,
    /// Required multiple base: valid tiles are the divisors of this value.
    pub divisor: u32,
}

/// Per-op feasible region.
///
/// CEP:WHAT: Bounds + discrete domains for one tensor op's schedule parameters.
/// CEP:WHY: O(1) feasibility checks during enumeration; `valid_fuse` is
///          pre-filtered by fusion legality (one authority: fusion::legality).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: fuse targets are legality-checked at extraction time.
/// CEP:COST: ~64 bytes + fuse vec.
/// CEP:EVIDENCE: WPA tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone)]
pub struct OpBounds {
    /// The scheduled op.
    pub op_id: NodeId,
    /// Minimum tile dims (usually [1,1,1]).
    pub tile_min: [u32; 3],
    /// Maximum tile dims (clamped by shape and hw.max_tile).
    pub tile_max: [u32; 3],
    /// Minimum memory level.
    pub mem_min: MemLevel,
    /// Maximum memory level.
    pub mem_max: MemLevel,
    /// Legality-pre-filtered fusion partners (consumers).
    pub valid_fuse: Vec<NodeId>,
    /// Vector-width bitmask: bit i <=> width 2^i valid (i in 0..4).
    pub vec_mask: u16,
}

/// The packed per-op schedule — the POD record that flows through the typed
/// SPSC channel and composes a complete candidate.
///
/// CEP:WHAT: NodeId-identified op + tile dims + memory level + fusion partner
///           + vector width, all fixed-size plain data.
/// CEP:WHY: The channel hot path demands Pod (anvil::typed::Pod); NodeId is
///          Copy but not field-trivially-verifiable, so the raw u64 is stored
///          and re-packed on the consumer side.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: fuse_target u64::MAX means "no fusion".
/// CEP:COST: 32 bytes; Copy.
/// CEP:EVIDENCE: enumerator tests round-trip the record through the channel.
/// CEP:SECURITY: no untrusted input
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedOpSched {
    /// Raw NodeId of the scheduled op.
    pub op_id: u64,
    /// Tile dims (m, n, k analogues for the op's domain).
    pub tile: [u32; 3],
    /// Raw NodeId of the fusion target; u64::MAX = none.
    pub fuse_target: u64,
    /// Memory level discriminant.
    pub mem_level: u8,
    /// Vector width (1, 2, 4, 8).
    pub vec_width: u8,
}

impl anvil::typed::Pod for PackedOpSched {}

impl PackedOpSched {
    /// CEP:WHAT: Sentinel for "no fusion partner".
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: NodeId::NONE occupies u64::MAX (id.rs) — distinct value.
    /// CEP:COST: constant
    /// CEP:EVIDENCE: enumerator tests
    /// CEP:SECURITY: no untrusted input
    pub const NO_FUSE: u64 = u64::MAX - 1;

    /// CEP:WHAT: Decodes the memory level (loud on corruption).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None outside 0..=2 — callers treat as violation.
    /// CEP:ASSUMES: constructed by the enumerator or verifier.
    /// CEP:COST: branch
    /// CEP:EVIDENCE: verifier tests
    /// CEP:SECURITY: bounded input
    pub fn mem(&self) -> Option<MemLevel> {
        MemLevel::from_u8(self.mem_level)
    }

    /// CEP:WHAT: Parameter accessor for the linear system.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: 0 for unknown parameters (never happens — closed enum).
    /// CEP:ASSUMES: none
    /// CEP:COST: branch
    /// CEP:EVIDENCE: linear row tests
    /// CEP:SECURITY: no untrusted input
    pub fn param_value(&self, param: ParamKind) -> i64 {
        match param {
            ParamKind::Mem => i64::from(self.mem_level),
            ParamKind::Tile0 => i64::from(self.tile[0]),
            ParamKind::Tile1 => i64::from(self.tile[1]),
            ParamKind::Tile2 => i64::from(self.tile[2]),
            ParamKind::Vec => i64::from(self.vec_width),
        }
    }
}

/// Constraint violation report (the verifier's failure payload).
///
/// CEP:WHAT: Exact failed check class + position (38.22 diagnosability).
/// CEP:WHY: Rejections must be explainable, never guessed.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: small enum
/// CEP:EVIDENCE: check_full tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintViolation {
    /// A parameter outside its OpBounds.
    BoundViolation(u16),
    /// A tile dim not a multiple of the problem dim.
    DivViolation(u16, u8),
    /// A global linear row evaluated false.
    LinearViolation(u16),
    /// SRAM pressure exceeded the capacity.
    SramOverflow(u64),
    /// The schedule references an op outside the polytope.
    UnknownOp(u16),
    /// A fused pair failed fusion legality.
    FuseIllegal(u16),
}

/// The feasible region produced by WPA.
///
/// CEP:WHAT: Op bounds + global linear system + divisibility + hardware model
///           + the per-op analytic cost inputs (flops, bytes) the roofline
///           consumes.
/// CEP:WHY: Single artifact consumed by enumerator, cost model, and verifier —
///          one authority (Law 3).
/// CEP:STATUS: complete
/// CEP:FAILURE: `check_full` reports ConstraintViolation.
/// CEP:ASSUMES: extracted from a verified arena (wpa.rs).
/// CEP:COST: O(ops) memory.
/// CEP:EVIDENCE: constraint-soundness test file.
/// CEP:SECURITY: no untrusted input.
/// CEP:HPC-DETERMINISM: deterministic extraction (slot order, Gear-0).
#[derive(Debug, Clone)]
pub struct SchedulePolytope {
    /// Per-op parameter bounds in deterministic slot order.
    pub op_bounds: Vec<OpBounds>,
    /// Global linear constraints.
    pub global_constraints: SparseLinearSystem,
    /// Discrete domain restrictions.
    pub divisibility_constraints: Vec<DivConstraint>,
    /// Hardware constants.
    pub hw: HardwareModel,
    /// Analytic FLOPs per op (roofline input).
    pub op_flops: Vec<u64>,
    /// Output tensor bytes per op.
    pub op_out_bytes: Vec<u64>,
    /// Input tensor bytes per op.
    pub op_in_bytes: Vec<u64>,
    /// Sound per-op HBM lower bound (0 when any fusion target exists).
    pub op_min_hbm: Vec<u64>,
}

impl SchedulePolytope {
    /// CEP:WHAT: Exact feasibility check of a complete schedule.
    /// CEP:WHY: The verifier's constructive proof obligation; also the
    ///          enumerator's completion-time guard.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Err with the exact violation class.
    /// CEP:ASSUMES: sched covers exactly ops 0..op_count in order.
    /// CEP:COST: O(ops + rows + divisibility).
    /// CEP:EVIDENCE: `gemm_schedule_satisfies_polytope` and rejection tests.
    /// CEP:SECURITY: no untrusted input.
    pub fn check_full(&self, sched: &[PackedOpSched]) -> Result<(), ConstraintViolation> {
        if sched.len() != self.op_bounds.len() {
            return Err(ConstraintViolation::UnknownOp(sched.len() as u16));
        }
        for (i, s) in sched.iter().enumerate() {
            let b = &self.op_bounds[i];
            if s.op_id != b.op_id.0 {
                return Err(ConstraintViolation::UnknownOp(i as u16));
            }
            for d in 0..3 {
                if s.tile[d] < b.tile_min[d] || s.tile[d] > b.tile_max[d] {
                    return Err(ConstraintViolation::BoundViolation(i as u16));
                }
            }
            let mem = s
                .mem()
                .ok_or(ConstraintViolation::BoundViolation(i as u16))?;
            if mem < b.mem_min || mem > b.mem_max {
                return Err(ConstraintViolation::BoundViolation(i as u16));
            }
            let vw = s.vec_width;
            if vw == 0 || !vw.is_power_of_two() || vw > 8 {
                return Err(ConstraintViolation::BoundViolation(i as u16));
            }
            let bit = (vw.trailing_zeros()) as u16;
            if b.vec_mask & (1 << bit) == 0 {
                return Err(ConstraintViolation::BoundViolation(i as u16));
            }
            if s.fuse_target != PackedOpSched::NO_FUSE
                && !b.valid_fuse.iter().any(|t| t.0 == s.fuse_target)
            {
                return Err(ConstraintViolation::FuseIllegal(i as u16));
            }
        }
        for d in self.divisibility_constraints.iter() {
            let s = &sched[d.op_index as usize];
            let t = s.tile[d.dim as usize];
            // Spec: the tile DIVIDES the problem gcd (divisor.is_multiple_of(t) —
            // the inverted multiple-check this replaces collapsed the tile
            // domain to a singleton; review F3).
            if t == 0 || !d.divisor.is_multiple_of(t) {
                return Err(ConstraintViolation::DivViolation(d.op_index, d.dim));
            }
        }
        if !self.global_constraints.check_full(sched) {
            // First failing row index for diagnosability.
            for (ri, row) in self.global_constraints.rows.iter().enumerate() {
                let mut acc: i64 = 0;
                for (op, param, coeff) in row.entries.iter() {
                    let v = sched[*op as usize].param_value(*param);
                    acc = acc.saturating_add(coeff.saturating_mul(v));
                }
                if acc > row.rhs {
                    return Err(ConstraintViolation::LinearViolation(ri as u16));
                }
            }
            return Err(ConstraintViolation::LinearViolation(0));
        }
        // SRAM pressure: outputs PLACED at SRAM (mem == Sram) must fit the
        // capacity (REG outputs live in registers — excluded; HBM excluded).
        let mut pressure: u64 = 0;
        for (i, s) in sched.iter().enumerate() {
            if s.mem() == Some(MemLevel::Sram) {
                pressure = pressure.saturating_add(self.op_out_bytes[i]);
            }
        }
        if pressure > self.hw.sram_cap_bytes {
            return Err(ConstraintViolation::SramOverflow(pressure));
        }
        Ok(())
    }

    /// CEP:WHAT: Sound total HBM lower bound (pruning input).
    /// CEP:WHY: Σ per-op minima — each op must write its output somewhere
    ///          unless it fuses into a consumer (min 0 then).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (lower bound).
    /// CEP:ASSUMES: none
    /// CEP:COST: O(ops)
    /// CEP:EVIDENCE: enumerator prune-soundness test
    /// CEP:SECURITY: no untrusted input
    pub fn total_min_hbm(&self) -> u64 {
        self.op_min_hbm
            .iter()
            .fold(0u64, |a, b| a.saturating_add(*b))
    }

    /// CEP:WHAT: Op count (schedule arity).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn op_count(&self) -> usize {
        self.op_bounds.len()
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

    fn gemm_polytope() -> SchedulePolytope {
        let op = NodeId(0x11);
        SchedulePolytope {
            op_bounds: vec![OpBounds {
                op_id: op,
                tile_min: [1, 1, 1],
                tile_max: [8, 8, 8],
                mem_min: MemLevel::Reg,
                mem_max: MemLevel::Hbm,
                valid_fuse: vec![NodeId(0x22)],
                vec_mask: 0b0101, // widths 1 and 4
            }],
            global_constraints: SparseLinearSystem {
                rows: vec![LinearRow {
                    entries: vec![(0, ParamKind::Mem, 1)],
                    rhs: 2,
                }],
            },
            divisibility_constraints: vec![DivConstraint {
                op_index: 0,
                dim: 0,
                divisor: 8,
            }],
            hw: HardwareModel::reference(),
            op_flops: vec![2 * 8 * 8 * 8],
            op_out_bytes: vec![512],
            op_in_bytes: vec![1024],
            op_min_hbm: vec![0],
        }
    }

    // CEP:WHAT: A hand-proven GEMM schedule satisfies the polytope.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a false rejection.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn gemm_schedule_satisfies_polytope() {
        let poly = gemm_polytope();
        let sched = vec![PackedOpSched {
            op_id: 0x11,
            tile: [8, 8, 8],
            fuse_target: PackedOpSched::NO_FUSE,
            mem_level: MemLevel::Hbm as u8,
            vec_width: 4,
        }];
        assert!(poly.check_full(&sched).is_ok());
    }

    // CEP:WHAT: Divisibility violations are rejected with the exact class.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a false acceptance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn divisibility_violation_rejected() {
        let poly = gemm_polytope();
        let sched = vec![PackedOpSched {
            op_id: 0x11,
            tile: [3, 8, 8], // 3 does not divide 8
            fuse_target: PackedOpSched::NO_FUSE,
            mem_level: MemLevel::Hbm as u8,
            vec_width: 4,
        }];
        assert_eq!(
            poly.check_full(&sched),
            Err(ConstraintViolation::DivViolation(0, 0))
        );
        // A proper divisor tile (4 | 8) is accepted.
        let good = vec![PackedOpSched {
            op_id: 0x11,
            tile: [4, 8, 8],
            fuse_target: PackedOpSched::NO_FUSE,
            mem_level: MemLevel::Hbm as u8,
            vec_width: 4,
        }];
        assert!(poly.check_full(&good).is_ok());
    }

    // CEP:WHAT: Linear violations are rejected.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a false acceptance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn linear_violation_rejected() {
        let mut poly = gemm_polytope();
        poly.global_constraints.rows[0].rhs = 1; // mem <= 1 (SRAM max)
        let sched = vec![PackedOpSched {
            op_id: 0x11,
            tile: [8, 8, 8],
            fuse_target: PackedOpSched::NO_FUSE,
            mem_level: MemLevel::Hbm as u8,
            vec_width: 4,
        }];
        assert!(matches!(
            poly.check_full(&sched),
            Err(ConstraintViolation::LinearViolation(_))
        ));
    }

    // CEP:WHAT: SRAM overflow is rejected.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a false acceptance.
    // CEP:ASSUMES: out_bytes 512 x placement — capacity 64KiB, so scale up.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn sram_overflow_rejected() {
        let mut poly = gemm_polytope();
        poly.op_out_bytes[0] = 128 * 1024; // > 64 KiB cap
        let sched = vec![PackedOpSched {
            op_id: 0x11,
            tile: [8, 8, 8],
            fuse_target: PackedOpSched::NO_FUSE,
            mem_level: MemLevel::Sram as u8,
            vec_width: 4,
        }];
        assert!(matches!(
            poly.check_full(&sched),
            Err(ConstraintViolation::SramOverflow(_))
        ));
    }

    // CEP:WHAT: Partial checks skip rows referencing unbound ops.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a future row is evaluated early.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn partial_check_skips_unbound_rows() {
        let poly = gemm_polytope();
        let sched = vec![PackedOpSched {
            op_id: 0x11,
            tile: [8, 8, 8],
            fuse_target: PackedOpSched::NO_FUSE,
            mem_level: MemLevel::Hbm as u8,
            vec_width: 4,
        }];
        // Depth 0: no parameters bound — every row skipped, vacuously true.
        assert!(poly.global_constraints.check_partial(&sched, 0));
        // Depth 1: the row is bound and satisfied.
        assert!(poly.global_constraints.check_partial(&sched, 1));
    }

    // CEP:WHAT: Vector-width mask is enforced (width 2 absent -> rejected).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a false acceptance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn vec_mask_enforced() {
        let poly = gemm_polytope();
        let sched = vec![PackedOpSched {
            op_id: 0x11,
            tile: [8, 8, 8],
            fuse_target: PackedOpSched::NO_FUSE,
            mem_level: MemLevel::Hbm as u8,
            vec_width: 2, // bit 1 clear in mask 0b0101
        }];
        assert_eq!(
            poly.check_full(&sched),
            Err(ConstraintViolation::BoundViolation(0))
        );
    }
}
