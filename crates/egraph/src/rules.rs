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
//           `commutativity_gates_floats`, `sub_self_is_zero_int_only`,
//           `mul_by_zero_int_only`.
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

/// CEP:WHAT: Local rules — constant folding, identity elimination, and the
///           integer annihilation rewrites (CEP-18).
/// CEP:WHY: The always-legal rewrites: folding computes the exact runtime
///          operation earlier (identical result); identity elimination
///          (x+0, 0+x, x*1, x-0) removes work; annihilation (x-x -> 0,
///          x*0 -> 0) removes provably dead arithmetic. x-x is sound
///          because SAME e-class means provably-equal values (i64: exact;
///           banned for floats: x-x is NaN at x=inf, +0.0 vs -0.0
///           sign-preservation is not the issue, the inf/NaN cases are).
///          x*0 is exact for integers only (float x*0.0 = NaN at inf/NaN
///           and ±0.0 signed zeros). All rules are single-node and
///          Gear-1-partitionable (disjoint node slices, deterministic
///          merge by index).
/// CEP:STATUS: complete
/// CEP:FAILURE: none; no fire = no rewrite.
/// CEP:ASSUMES: `consts[i]` is Some only when child i's class is that
///           constant; children are canonical classes; `elem` reflects
///           the node's ANNOTATED element type. The verifier guarantees
///           value-producing Binary/Unary nodes carry a concrete
///           (non-None) result type, so elem None is a defensive fallback
///           that treats rules as integer-eligible — sound only under
///           that verifier guarantee (mis-annotated programs are a
///           documented verifier gap: no operand/result type-agreement
///           check exists yet; see conformance.md).
/// CEP:COST: O(1).
/// CEP:EVIDENCE: tests `constant_folding_fires`, `identity_elimination`,
///           `sub_self_is_zero_int_only`, `mul_by_zero_int_only`.
pub fn local_rules(
    op: Op,
    children: &[u32; 4],
    consts: &[Option<ConstVal>; 4],
    elem: Option<ScalarType>,
) -> Option<Rewrite> {
    // Integer gate: I64 only (audit round 4, F-3). I32 is EXCLUDED
    // deliberately: the Op vocabulary has no I32 constants, so a fold or
    // annihilation rewrite produces ConstI64 — which application's type
    // gate (fold_in_place) refuses on I32 nodes. Advertising the rewrite
    // for I32 would be analysis-only saturation; the gate matches exactly
    // what can land.
    let is_int = matches!(elem, None | Some(ScalarType::I64));
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
            let r = use_child.map(|ch| Rewrite {
                op: Op::Param { index: u32::MAX }, // marker: pass-through (driver maps to the child's op)
                children: [ch, 0, 0, 0],
                n_children: 1,
            });
            if r.is_some() {
                return r;
            }
            // Integer annihilation (CEP-18). Element-typed gates only:
            // float x-x is NaN at x=inf; float x*0.0 is ±0.0/NaN — both
            // banned (38.24 discipline: exact rewrites only).
            if is_int {
                match b {
                    // x - x -> 0 : same e-class == provably equal values
                    // (the verifier's arity contract guarantees Sub carries
                    // both operands live).
                    BinaryOp::Sub if a == c => {
                        return Some(Rewrite {
                            op: Op::ConstI64(0),
                            children: [0, 0, 0, 0],
                            n_children: 0,
                        });
                    }
                    // x * 0 -> 0 ; 0 * x -> 0 (integer exactness).
                    BinaryOp::Mul if is_zero_int(cc) || is_zero_int(ca) => {
                        return Some(Rewrite {
                            op: Op::ConstI64(0),
                            children: [0, 0, 0, 0],
                            n_children: 0,
                        });
                    }
                    _ => {}
                }
            }
            None
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
///          rounding environments); integer arithmetic is exact. Float
///          Max/Min are ALSO banned (audit round 4, F-1): IEEE maxnum may
///          return either zero for (+0.0, -0.0) inputs and common hardware
///          lowerings return the SECOND operand on ties — commuting float
///          Max/Min can flip the returned zero's sign, a concrete
///          miscompile chain through fold-then-merge. Element type None
///          means "untracked" — conservative integer-only for every op.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (false = do not commute).
/// CEP:ASSUMES: elem reflects the tensor/scalar element of the operands;
///           verified value-producing Binary nodes always carry a concrete
///           type (the verifier rejects Type::None results), so None is a
///          defensive fallback, not a live path.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: test `commutativity_gates_floats`,
///           `float_max_commute_banned`.
pub fn commutative(op: Op, elem: Option<ScalarType>) -> bool {
    let int_only = matches!(elem, None | Some(ScalarType::I64) | Some(ScalarType::I32));
    match op {
        Op::Binary(BinaryOp::Add) | Op::Binary(BinaryOp::Mul) => int_only,
        // Max/Min: integer exact; float banned (signed-zero asymmetry —
        // audit round 4 F-1 supersedes the old F-18 note).
        Op::Binary(BinaryOp::Max) | Op::Binary(BinaryOp::Min) => int_only,
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
            Some(ScalarType::I64),
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
            Some(ScalarType::I64),
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
            Some(ScalarType::I64),
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
            Some(ScalarType::I64),
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
            Some(ScalarType::I64),
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
            Some(ScalarType::F64),
        );
        assert_eq!(r, None);
        // x * 1.0 must NOT fire.
        let r2 = local_rules(
            Op::Binary(BinaryOp::Mul),
            &[7, 8, 0, 0],
            &[None, Some(ConstVal::F(1.0)), None, None],
            Some(ScalarType::F64),
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
        // Float Max/Min are banned too (audit round 4 F-1: signed-zero
        // asymmetry of IEEE maxnum on ties).
        assert!(!commutative(
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

    // CEP:WHAT: x - x -> 0 fires for integers (same e-class = provably
    //           equal values) and is banned for floats (x-x is NaN at
    //           x=inf — not zero).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on gate drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn sub_self_is_zero_int_only() {
        let r = local_rules(
            Op::Binary(BinaryOp::Sub),
            &[5, 5, 0, 0],
            &[None, None, None, None],
            Some(ScalarType::I64),
        );
        assert_eq!(
            r,
            Some(Rewrite {
                op: Op::ConstI64(0),
                children: [0, 0, 0, 0],
                n_children: 0,
            })
        );
        // Float element type: the rule must NOT fire.
        let rf = local_rules(
            Op::Binary(BinaryOp::Sub),
            &[5, 5, 0, 0],
            &[None, None, None, None],
            Some(ScalarType::F64),
        );
        assert_eq!(rf, None);
        // Different operands: no fire.
        let rd = local_rules(
            Op::Binary(BinaryOp::Sub),
            &[5, 6, 0, 0],
            &[None, None, None, None],
            Some(ScalarType::I64),
        );
        assert_eq!(rd, None);
    }

    // CEP:WHAT: x * 0 -> 0 and 0 * x -> 0 fire for integers; float
    //           multiplication by zero is banned (inf*0=NaN, -x*0=-0.0).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on gate drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn mul_by_zero_int_only() {
        let r = local_rules(
            Op::Binary(BinaryOp::Mul),
            &[7, 8, 0, 0],
            &[None, Some(ConstVal::I(0)), None, None],
            Some(ScalarType::I64),
        );
        assert_eq!(r.map(|rw| rw.op), Some(Op::ConstI64(0)));
        let r2 = local_rules(
            Op::Binary(BinaryOp::Mul),
            &[7, 8, 0, 0],
            &[Some(ConstVal::I(0)), None, None, None],
            Some(ScalarType::I64),
        );
        assert_eq!(r2.map(|rw| rw.op), Some(Op::ConstI64(0)));
        // Float: banned.
        let rf = local_rules(
            Op::Binary(BinaryOp::Mul),
            &[7, 8, 0, 0],
            &[None, Some(ConstVal::F(0.0)), None, None],
            Some(ScalarType::F64),
        );
        assert_eq!(rf, None);
    }

    // CEP:WHAT: Float Max/Min commutativity is BANNED — the signed-zero
    //           miscompile chain (audit round 4, F-1): max(+0.0, -0.0) and
    //           max(-0.0, +0.0) fold to different bit patterns on
    //           tie-second lowerings; commuting would merge their classes
    //           and application could rewrite one to the other's value.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the float gate leaks.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn float_max_commute_banned() {
        assert!(!commutative(
            Op::Binary(BinaryOp::Max),
            Some(ScalarType::F64)
        ));
        assert!(!commutative(
            Op::Binary(BinaryOp::Min),
            Some(ScalarType::F64)
        ));
        // Integers still commute.
        assert!(commutative(
            Op::Binary(BinaryOp::Max),
            Some(ScalarType::I64)
        ));
        assert!(commutative(Op::Binary(BinaryOp::Min), None));
    }
}
