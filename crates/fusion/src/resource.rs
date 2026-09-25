// CEP:FILE: crates/fusion/src/resource.rs
// CEP:WHAT: The fusion resource model — shared-memory, register-pressure
//           and occupancy estimates for proposed clusters.
// CEP:WHY: Master architecture section 5: "Analytically estimates register
//          pressure, shared memory usage, and occupancy for any proposed
//          cluster." Estimates gate cluster legality (over-budget clusters
//          go to repair) — bounded resources are a correctness contract
//          (CEP&CC 39), so the budget is a named config constant and
//          over-budget is a loud result, never a silent overflow.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: none (estimates are values); the SEARCH rejects
//             over-budget clusters via ResourceEstimate::over_budget.
// CEP:ASSUMES: byte_size() of node types is trusted (verified types);
//           the CPU target budget is the reference (GPU budgets are
//           documented placeholders).
// CEP:COST: O(cluster size) per estimate.
// CEP:EVIDENCE: tests `budget_gates_clusters`, `tile_bytes_scale`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-DETERMINISM: deterministic.
//! Fusion resource model.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;

/// Shared-memory budget per cluster (bytes).
///
/// CEP:WHAT: The shared-memory ceiling used to gate clusters.
/// CEP:WHY: 64 KiB matches the reference CPU LLC-slice tile and the common
///          GPU shared-memory size (name the number — Law 7); over-budget
///          clusters are repaired (split) instead of silently thrashing.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: reference target (docs/targets.md documents the calibration).
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: test `budget_gates_clusters`
pub const SHARED_MEM_BUDGET_BYTES: u32 = 64 * 1024;

/// Register budget per cluster (abstract units).
///
/// CEP:WHAT: Register-file ceiling in abstract units.
/// CEP:WHY: 256 units approximates 64 FP64 registers x4 pressure classes;
///          exceeding forces spills (priced by the cost model).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: calibration in docs/cost_model.md.
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: resource tests
pub const REGISTER_BUDGET_UNITS: u32 = 256;

/// A cluster's resource estimate.
///
/// CEP:WHAT: Shared bytes, register units and the occupancy hint.
/// CEP:WHY: The search's resource gate: over_budget clusters are illegal
///          as-is and go to repair (arch section 5 "Resource Model").
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: built via resource_estimate().
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceEstimate {
    /// Total shared-memory bytes the cluster's tiles need.
    pub shared_bytes: u32,
    /// Register-pressure units.
    pub register_units: u32,
    /// Occupancy hint: clusters per SM (higher is better).
    pub occupancy_hint: u32,
}

impl ResourceEstimate {
    /// CEP:WHAT: Reports whether the estimate exceeds any budget.
    /// CEP:WHY: The loud gate consumed by the search and repair passes.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 compares
    /// CEP:EVIDENCE: test `budget_gates_clusters`
    pub fn over_budget(&self) -> bool {
        self.shared_bytes > SHARED_MEM_BUDGET_BYTES || self.register_units > REGISTER_BUDGET_UNITS
    }
}

/// CEP:WHAT: Estimates resources for a cluster of nodes.
/// CEP:WHY: Shared-memory bytes = sum of tile footprints (elementwise
///          intermediates live one tile: bytes = elements x elem_size;
///          reductions add an accumulator per axis). Register units = 8 per
///          live intermediate + 16 per reduction. Occupancy = budget /
///          usage (saturating).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (saturating arithmetic — no overflow panics).
/// CEP:ASSUMES: tensor types verified.
/// CEP:COST: O(cluster size).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn resource_estimate(arena: &IrArena, members: &[NodeId]) -> ResourceEstimate {
    let mut shared_bytes: u32 = 0;
    let mut register_units: u32 = 0;
    for m in members {
        let node = match arena.node(*m) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let bytes = node.ty.byte_size();
        let b32 = u32::try_from(bytes.max(0)).unwrap_or(u32::MAX);
        // Elementwise intermediates: one tile lives in shared memory.
        shared_bytes = shared_bytes.saturating_add(b32);
        register_units = register_units.saturating_add(8);
        if matches!(node.op, xir_core::op::Op::Reduce { .. }) {
            register_units = register_units.saturating_add(16);
        }
    }
    let occupancy_hint = if shared_bytes == 0 {
        REGISTER_BUDGET_UNITS / register_units.max(1)
    } else {
        (SHARED_MEM_BUDGET_BYTES / shared_bytes.max(1)).max(1)
    };
    ResourceEstimate {
        shared_bytes,
        register_units,
        occupancy_hint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::IrArena;
    use xir_core::node::Node;
    use xir_core::op::Op;
    use xir_core::ty::{Layout, ScalarType, Shape, TensorType, Type};

    fn tensor(shape: &[i64], elem: ScalarType) -> Type {
        Type::Tensor(TensorType {
            elem,
            shape: Shape::from_dims(shape).ok().unwrap_or(Shape::scalar()),
            layout: Layout::RowMajor,
        })
    }

    // CEP:WHAT: Small clusters fit; a giant tile is over budget.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on gate inversion.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn budget_gates_clusters() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let small = Node::new(
            Op::Binary(xir_core::op::BinaryOp::Add),
            root,
            &[xir_core::id::ValueId::NONE; 2],
            tensor(&[4, 4], ScalarType::F64),
        );
        let sid = a.insert_node(root, small);
        assert!(sid.is_ok());
        if let Ok(s) = sid {
            let est = resource_estimate(&a, &[s]);
            assert!(!est.over_budget());
        }
        // 64 x 1024 f64 tile = 512KiB: over the 64KiB budget.
        let big = Node::new(
            Op::Binary(xir_core::op::BinaryOp::Add),
            root,
            &[xir_core::id::ValueId::NONE; 2],
            tensor(&[64, 1024], ScalarType::F64),
        );
        let bid = a.insert_node(root, big);
        assert!(bid.is_ok());
        if let Ok(b) = bid {
            let est = resource_estimate(&a, &[b]);
            assert!(est.over_budget());
        }
    }

    // CEP:WHAT: Estimates scale with members and element width.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on miscount.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tile_bytes_scale() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let n = Node::new(
            Op::Binary(xir_core::op::BinaryOp::Mul),
            root,
            &[xir_core::id::ValueId::NONE; 2],
            tensor(&[8, 8], ScalarType::F32),
        );
        let id = a.insert_node(root, n);
        assert!(id.is_ok());
        if let Ok(i) = id {
            let est = resource_estimate(&a, &[i]);
            assert_eq!(est.shared_bytes, 8 * 8 * 4);
            assert_eq!(est.register_units, 8);
            assert!(est.occupancy_hint >= 1);
        }
    }
}
