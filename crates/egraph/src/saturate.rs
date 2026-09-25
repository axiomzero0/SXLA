// CEP:FILE: crates/egraph/src/saturate.rs
// CEP:WHAT: Saturation driver — lift + progressive local-rule rounds
//           (analysis entry point).
// CEP:WHY: Master architecture section 4: "E-classes are sharded across
//          Anvil workers by hash. Workers apply rewrite rules locally
//          without synchronization." Local rules (constant folding, identity
//          elimination) are exactly the synchronization-free rule class.
//          The machinery lives in lift.rs and is shared with apply.rs
//          (extraction application, CEP-17); this module is the analysis
//          entry: it returns the saturated graph without touching the
//          arena. The full cross-worker merge protocol (SPSC to a dedicated
//          resolver thread) remains a documented TODO (CEP-17).
// CEP:CLASS: CEP-1 (driver) / CEP-0 (rule bodies)
// CEP:STATUS: partial
// CEP:FAILURE: EgraphError propagation; saturation stops at the node budget
//              and the round bound.
// CEP:ASSUMES: only liftable pure nodes enter the graph (lift discipline:
//              impure inputs poison the node).
// CEP:COST: lift O(nodes); rounds O(rounds * records), bounded by
//           MAX_ROUNDS.
// CEP:EVIDENCE: tests `saturation_folds_constants`,
//           `saturation_is_deterministic`.
// CEP:SECURITY: IR treated as untrusted; all lookups checked.
// CEP:HPC-DETERMINISM: deterministic — record order and rule order are
//           fixed; merges union-by-min-id.
// CEP:TODO(main-agent): CEP-17: cross-worker class merges via SPSC resolver.
//! Saturation driver (analysis entry point).

use xir_core::arena::IrArena;

use crate::egraph::{EGraph, EgraphError};
use crate::lift::{lift, run_rounds};

/// Saturation rounds bound.
///
/// CEP:WHAT: Iteration budget.
/// CEP:WHY: CEP&CC 38.11 (bounded compile time): equality saturation is
///          budgeted explicitly. Progressive rounds (class_const
///          propagation) converge in depth-of-constant-tree rounds; 4
///          covers folded trees of depth 4 and leaves documented headroom.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: apply tests fold (3+4)*5 in two rounds.
pub const MAX_ROUNDS: u32 = 4;

/// CEP:WHAT: Result of a saturation run.
/// CEP:WHY: The driver reports applied rewrites and the saturated graph for
///          extraction by the caller.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: none
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
pub struct SaturationResult {
    /// The saturated e-graph.
    pub graph: EGraph,
    /// Local rewrites applied (constant folds + identities).
    pub rewrites_applied: u32,
}

/// CEP:WHAT: Saturates the arena's pure subgraph (analysis only).
/// CEP:WHY: Lift + progressive rounds; the returned graph is saturated and
///          ready for extract(). The arena is NOT modified — callers that
///          want the rewrites applied use apply::apply (CEP-17).
/// CEP:STATUS: complete
/// CEP:FAILURE: EgraphError propagation.
/// CEP:ASSUMES: verified arena.
/// CEP:COST: see lift.rs.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn saturate(arena: &IrArena, node_budget: usize) -> Result<SaturationResult, EgraphError> {
    let mut lg = lift(arena, node_budget)?;
    let rewrites_applied = run_rounds(&mut lg)?;
    Ok(SaturationResult {
        graph: lg.g,
        rewrites_applied,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::{const_i64, IrArena};
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Type};

    fn arena_with_const_add() -> IrArena {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let add = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::I64),
                );
                let _ = a.insert_node(root, add);
            }
        }
        a
    }

    // CEP:WHAT: Saturation folds constant adds into a const node.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if folding does not fire.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn saturation_folds_constants() {
        let a = arena_with_const_add();
        let r = saturate(&a, 64);
        assert!(r.is_ok());
        if let Ok(res) = r {
            assert!(res.rewrites_applied >= 1);
        }
    }

    // CEP:WHAT: Saturation is deterministic across runs.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn saturation_is_deterministic() {
        let a = arena_with_const_add();
        let r1 = saturate(&a, 64);
        let r2 = saturate(&a, 64);
        assert!(r1.is_ok() && r2.is_ok());
        if let (Ok(x1), Ok(x2)) = (r1, r2) {
            assert_eq!(x1.rewrites_applied, x2.rewrites_applied);
            assert_eq!(x1.graph.node_count(), x2.graph.node_count());
        }
    }
}
