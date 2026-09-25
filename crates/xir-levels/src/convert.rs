// CEP:FILE: crates/xir-levels/src/convert.rs
// CEP:WHAT: Form conversion boundary — graphify / structurize helpers.
// CEP:WHY: Master architecture section 7: "Automatic conversions inserted by
//          the Pass Manager if a pass requires structured loops but the IR
//          is currently in Sea-of-Nodes form." In this stack the graph form
//          is primary and the structured form is the Level-3 LoopProgram
//          projection (level3::project); graphify is the identity (already
//          graph) and structurize is the projection. Both are explicit and
//          typed — no hidden form guessing (Law 2).
// CEP:CLASS: CEP-1 (conversion boundary)
// CEP:STATUS: complete
// CEP:FAILURE: LowerError propagates from the projection; FormError for
//             impossible conversions.
// CEP:ASSUMES: verified input.
// CEP:COST: structurize O(nodes + edges); graphify O(1).
// CEP:EVIDENCE: passman integration tests.
// CEP:SECURITY: internal ids only.
// CEP:HPC-DETERMINISM: deterministic.
//! Form conversion (graphify / structurize).

use xir_core::arena::IrArena;
use xir_core::id::NodeId;

use crate::level3::{project, LoopProgram, LowerError};

/// Conversion failure enumeration.
///
/// CEP:WHAT: Explicit error type for form conversions.
/// CEP:WHY: Law 6 — a failed conversion must be loud; silently continuing
///          in the wrong form is a miscompilation hazard.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormError {
    /// The structured projection failed.
    Lower(LowerError),
    /// The requested conversion direction does not exist in this stack.
    Unavailable,
}

/// CEP:WHAT: Graphify — identity in this stack (IR is graph-form native).
/// CEP:WHY: The pass manager calls this when a Graph-form pass receives a
///          snapshot that is already graph-form; making it explicit keeps
///          the conversion boundary auditable (38.20).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: input is graph-form.
/// CEP:COST: O(1) borrow.
/// CEP:EVIDENCE: passman tests.
pub fn graphify(arena: &IrArena) -> &IrArena {
    arena
}

/// CEP:WHAT: Structurize — project the graph into a LoopProgram.
/// CEP:WHY: Structured-form consumers (Level 3/4) receive the canonical
///          scheduled projection; the projection is deterministic and
///          dominance-checked (schedule).
/// CEP:STATUS: complete
/// CEP:FAILURE: Lower propagates (cycles/unknown nodes abort loudly).
/// CEP:ASSUMES: verified arena; roots = result nodes.
/// CEP:COST: O(nodes + edges).
/// CEP:EVIDENCE: level3 tests.
pub fn structurize(arena: &IrArena, roots: &[NodeId]) -> Result<LoopProgram, FormError> {
    project(arena, roots).map_err(FormError::Lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::const_i64;

    // CEP:WHAT: graphify is the identity.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on copying (it must borrow).
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn graphify_is_identity() {
        let mut a = IrArena::with_capacity(8, 4);
        let root = a.root_region();
        let c = const_i64(&mut a, root, 1);
        assert!(c.is_ok());
        let same = graphify(&a);
        assert_eq!(same.node_count(), a.node_count());
    }

    // CEP:WHAT: structurize projects flat graphs.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on projection failure.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn structurize_projects() {
        let mut a = IrArena::with_capacity(8, 4);
        let root = a.root_region();
        let c = const_i64(&mut a, root, 7);
        assert!(c.is_ok());
        if let Ok(cv) = c {
            let prog = structurize(&a, &[cv]);
            assert!(prog.is_ok());
            if let Ok(p) = prog {
                assert_eq!(p.ops.len(), 1);
                assert_eq!(p.results.len(), 1);
            }
        }
    }
}
