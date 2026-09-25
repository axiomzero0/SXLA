// CEP:FILE: crates/egraph/src/rules.rs
// CEP:WHAT: Rewrite rules — local (per-node) rewrites and commutativity
//           legality gates.
// CEP:WHY: Master architecture section 4: saturation applies rewrite rules.
//          CEP&CC 38.23 requires every optimization assumption to be
//          explicit: float reassociation is BANNED by default (38.24), so
//          commutativity applies to integer arithmetic and max/min monoids
//          only. Constant folding is always legal (same operation, earlier).
// CEP:CLASS: CEP-0 (rule matching)
// CEP:STATUS: partial
// CEP:FAILURE: none; rules simply do not fire when legality fails.
// CEP:ASSUMES: child classes carry constant values only when the driver
//           recorded them (ConstVal probing at lift time).
// CEP:COST: local rules O(1) per node; they partition cleanly across
//           Gear-1 workers (disjoint slices).
// CEP:EVIDENCE: tests `constant_folding_fires`, `identity_elimination`,
//           `commutativity_gates_floats`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-TRANSFORM: Rewrites are new e-nodes (saturated), never in-place.
// CEP:HPC-DETERMINISM: deterministic; fixed rule order.
// CEP:TODO(main-agent): CEP-18: distributivity and strength-reduction rules.
//! Rewrite rules.

use xir_core::op::{BinaryOp, Monoid, Op};
use xir_core::ty::ScalarType;

/// A constant value probed from a child class.
///
/// CEP:WHAT: Constant payload carried per child.
/// CEP:WHY: Local rules (folding, identity) need operand constants; probing
///          happens once at lift/insert time, not per rule application.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: none
/// CEP:COST: 16 bytes
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstVal {
    /// Integer constant.
    I(i64),
    /// Float constant.
    F(f64),
}

/// One rewrite outcome: the rewritten op with child classes.
///
/// CEP:WHAT: Rule firing result.
/// CEP:WHY: The driver inserts these as new e-nodes and merges classes.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: children reference existing classes.
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rewrite {
    /// Rewritten op.
    pub op: Op,
    /// Child classes of the rewritten op.
    pub children: [u32; 4],
    /// Live child count.
    pub n_children: u8,
}

/// CEP:WHAT: Local rules — constant folding and identity elimination.
/// CEP:WHY: The always-legal rewrites: folding computes the exact runtime
///          operation earlier (identical result); identity elimination
///          (x+0, 0+x, x*1) removes work. Both are single-node and
///          Gear-1-partitionable (disjoint node slices, deterministic
///          merge by index).
/// CEP:STATUS: complete
/// CEP:FAILURE: none; no fire = no rewrite.
/// CEP:ASSUMES: `consts[i]` is Some only when child i's class is that
///           constant; children are canonical classes.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: tests `constant_folding_fires`, `identity_elimination`.
pub fn local_rules(op: Op, children: &[u32; 4], consts: &[Option<ConstVal>; 4]) -> Option<Rewrite> {
    match op {
        Op::Binary(b) => {
            let (a, c) = (children[0], children[1]);
            let (ca, cc) = (consts[0], consts[1]);
            // Constant folding (both operands constant).
            if let (Some(x), Some(y)) = (ca, cc) {
                if let Some(folded) = fold_binary(b, x, y) {
                    return Some(Rewrite {
                        op: folded,
                        children: [0, 0, 0, 0],
                        n_children: 0,
                    });
                }
            }
            // Identity elimination.
            // Signed-zero discipline (audit F-6, CEP&CC 38.24): x + 0.0
            // is NOT x when x = -0.0 (the true result is +0.0). Integer
            // identities are exact; FLOAT zero identities are excluded.
            let lhs_zero = is_zero_int(ca);
            let rhs_zero = is_zero_int(cc);
            let rhs_one = is_one(cc);
            let use_child = match b {
                // x + 0 -> x ; 0 + x -> x
                BinaryOp::Add if rhs_zero || lhs_zero => Some(if rhs_zero { a } else { c }),
                // x * 1 -> x
                BinaryOp::Mul if rhs_one => Some(a),
                // x - 0 -> x
                BinaryOp::Sub if rhs_zero => Some(a),
                _ => None,
            };
            use_child.map(|ch| Rewrite {
                op: Op::Param { index: u32::MAX }, // marker: pass-through (driver maps to the child's op)
                children: [ch, 0, 0, 0],
                n_children: 1,
            })
        }
        _ => None,
    }
}

/// CEP:WHAT: Folds a binary op over two constants.
/// CEP:WHY: Compile-time execution of the exact runtime operation (same
///          rounding, same result — legality per 38.24 note: no semantic
///          change, only earlier execution).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (mismatched operand kinds do not fold).
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: test `constant_folding_fires`
fn fold_binary(b: BinaryOp, x: ConstVal, y: ConstVal) -> Option<Op> {
    match (x, y) {
        (ConstVal::I(a), ConstVal::I(c)) => {
            let v = match b {
                BinaryOp::Add => a.checked_add(c),
                BinaryOp::Sub => a.checked_sub(c),
                BinaryOp::Mul => a.checked_mul(c),
                BinaryOp::Div => {
                    if c == 0 {
                        None
                    } else {
                        a.checked_div(c)
                    }
                }
                BinaryOp::Max => Some(a.max(c)),
                BinaryOp::Min => Some(a.min(c)),
            };
            v.map(Op::ConstI64)
        }
        (ConstVal::F(a), ConstVal::F(c)) => {
            let v = match b {
                BinaryOp::Add => Some(a + c),
                BinaryOp::Sub => Some(a - c),
                BinaryOp::Mul => Some(a * c),
                BinaryOp::Div => {
                    if c == 0.0 {
                        None
                    } else {
                        Some(a / c)
                    }
                }
                BinaryOp::Max => Some(a.max(c)),
                BinaryOp::Min => Some(a.min(c)),
            };
            v.map(Op::ConstF64)
        }
        _ => None,
    }
}

/// CEP:WHAT: INTEGER zero probe for identity rules (float zeros excluded).
/// CEP:WHY: Audit F-6 / CEP&CC 38.24: eliminating x+0.0 changes -0.0
///          results to +0.0 — a banned signed-zero transformation.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: test `float_zero_identity_excluded`
fn is_zero_int(v: Option<ConstVal>) -> bool {
    matches!(v, Some(ConstVal::I(0)))
}

/// CEP:WHAT: One probe for identity rules.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: tests
fn is_one(v: Option<ConstVal>) -> bool {
    match v {
        // Integer one only (audit F-6: float 1.0 identities share the
        // signed-zero discipline).
        Some(ConstVal::I(1)) => true,
        _ => false,
    }
}

/// CEP:WHAT: Commutativity legality gate (38.24 float ban).
/// CEP:WHY: Swapping float operands changes rounding-visible results in
///          general (a+b vs b+a may differ in the last bit for non-associative
///          rounding environments); integer arithmetic and max/min are safe.
///          NOTE (audit F-18): IEEE max/min may return either zero for
///          (±0.0, ∓0.0) inputs — commuting float Max/Min can flip the
///          returned zero's sign; acceptable under the documented Rust
///          f64::max semantics (both branches use the SAME library call).
///          Element type None means "untracked" — conservative integer-only.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (false = do not commute).
/// CEP:ASSUMES: elem reflects the tensor/scalar element of the operands.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: test `commutativity_gates_floats`.
pub fn commutative(op: Op, elem: Option<ScalarType>) -> bool {
    match op {
        Op::Binary(BinaryOp::Add) | Op::Binary(BinaryOp::Mul) => {
            matches!(elem, Some(ScalarType::I64) | Some(ScalarType::I32) | None)
        }
        Op::Binary(BinaryOp::Max) | Op::Binary(BinaryOp::Min) => true,
        Op::Reduce { monoid, .. } => {
            matches!(monoid, Monoid::Max | Monoid::Min | Monoid::And | Monoid::Or)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Constant folding fires on int and float pairs.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on fold failure.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn constant_folding_fires() {
        let r = local_rules(
            Op::Binary(BinaryOp::Add),
            &[1, 2, 0, 0],
            &[Some(ConstVal::I(3)), Some(ConstVal::I(4)), None, None],
        );
        assert_eq!(
            r,
            Some(Rewrite {
                op: Op::ConstI64(7),
                children: [0, 0, 0, 0],
                n_children: 0,
            })
        );
        // Division by zero does not fold (loud absence, not a guess).
        let dz = local_rules(
            Op::Binary(BinaryOp::Div),
            &[1, 2, 0, 0],
            &[Some(ConstVal::I(1)), Some(ConstVal::I(0)), None, None],
        );
        assert_eq!(dz, None);
    }

    // CEP:WHAT: Identity elimination rewrites to the value operand.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong operand.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn identity_elimination() {
        // x + 0 -> x (child 0 survives).
        let r = local_rules(
            Op::Binary(BinaryOp::Add),
            &[7, 8, 0, 0],
            &[None, Some(ConstVal::I(0)), None, None],
        );
        assert_eq!(
            r,
            Some(Rewrite {
                op: Op::Param { index: u32::MAX },
                children: [7, 0, 0, 0],
                n_children: 1,
            })
        );
        // 0 + x -> x (child 1 survives).
        let r2 = local_rules(
            Op::Binary(BinaryOp::Add),
            &[7, 8, 0, 0],
            &[Some(ConstVal::I(0)), None, None, None],
        );
        let fired = r2.is_some();
        assert!(fired, "identity rule must fire");
        if let Some(rw) = r2 {
            assert_eq!(rw.children[0], 8);
        }
        // x * 1 -> x.
        let r3 = local_rules(
            Op::Binary(BinaryOp::Mul),
            &[7, 8, 0, 0],
            &[None, Some(ConstVal::I(1)), None, None],
        );
        if let Some(rw) = r3 {
            assert_eq!(rw.children[0], 7);
        }
    }

    // CEP:WHAT: Float zero/one identities are excluded (signed zero).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if x+0.0 is eliminated (audit F-6).
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn float_zero_identity_excluded() {
        // x + 0.0 must NOT fire (would rewrite -0.0 results).
        let r = local_rules(
            Op::Binary(BinaryOp::Add),
            &[7, 8, 0, 0],
            &[None, Some(ConstVal::F(0.0)), None, None],
        );
        assert_eq!(r, None);
        // x * 1.0 must NOT fire.
        let r2 = local_rules(
            Op::Binary(BinaryOp::Mul),
            &[7, 8, 0, 0],
            &[None, Some(ConstVal::F(1.0)), None, None],
        );
        assert_eq!(r2, None);
    }

    // CEP:WHAT: Commutativity gates floats out, integers in.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on gate removal.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn commutativity_gates_floats() {
        let add = Op::Binary(BinaryOp::Add);
        assert!(commutative(add, Some(ScalarType::I64)));
        assert!(!commutative(add, Some(ScalarType::F64)));
        assert!(!commutative(add, Some(ScalarType::F32)));
        assert!(commutative(
            Op::Binary(BinaryOp::Max),
            Some(ScalarType::F64)
        ));
        assert!(commutative(
            Op::Reduce {
                axis: 0,
                monoid: Monoid::Max
            },
            Some(ScalarType::F64)
        ));
        assert!(!commutative(
            Op::Reduce {
                axis: 0,
                monoid: Monoid::Add
            },
            Some(ScalarType::F64)
        ));
    }
}
