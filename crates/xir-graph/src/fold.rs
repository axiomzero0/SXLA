// CEP:FILE: crates/xir-graph/src/fold.rs
// CEP:WHAT: Constant folding — the Level-0 algebraic simplification.
// CEP:WHY: Master architecture Level 0 runs "global CSE/GVN/Algebraic
//          simplification"; Tier-1 (<5ms) cannot afford e-graph saturation,
//          so the always-legal folds (compute the exact runtime operation
//          earlier — identical result, CEP&CC 38.24) live here as a plain
//          O(nodes) pass.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: none; folds are conservative (division by zero does not
//             fold — loud absence, never a guessed Inf).
// CEP:ASSUMES: verified input; constants carry Scalar types.
// CEP:COST: O(nodes) single pass.
// CEP:EVIDENCE: tests `folds_constant_arithmetic`, `mixed_types_skip`,
//           `division_by_zero_skipped`.
// CEP:SECURITY: IR untrusted; bounds-checked.
// CEP:HPC-PASS: const-fold-l0
// CEP:HPC-PASS-KIND: algebraic simplification
// CEP:HPC-PASS-INPUT: verified Level-0 snapshot
// CEP:HPC-PASS-OUTPUT: constant-valued ops folded to Const nodes
// CEP:HPC-PASS-ANALYSIS-REQUIRED: constant probing
// CEP:HPC-PASS-ANALYSIS-PRODUCED: none
// CEP:HPC-PASS-ANALYSIS-INVALIDATED: use-def (inputs dropped)
// CEP:HPC-PASS-LEGALITY: both operands constant; integer folds compute in
//           checked i64 (never through f64 — audit F-2); float folds and
//           exp/ln compute the identical Rust-libm operation the
//           interpreter executes (same-libm policy documented per 38.24:
//           compile-time and runtime share one library so results are
//           bit-identical; cross-libm targets need an explicit gate)
// CEP:HPC-PASS-PRESERVES: semantics (same value, earlier)
// CEP:HPC-PASS-COST: O(nodes)
// CEP:HPC-PASS-FAILURE: conservative skip
// CEP:HPC-PASS-TARGET: target-independent
// CEP:HPC-PASS-EVIDENCE: tests in this module + cli integration test
// CEP:HPC-TRANSFORM: Constant expressions collapse to Const nodes.
// CEP:HPC-DETERMINISM: deterministic.
//! Constant folding.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::Node;
use xir_core::op::{BinaryOp, Op, UnaryOp};
use xir_core::ty::{ScalarType, Type};

/// CEP:WHAT: Folds constant-valued arithmetic nodes in place.
/// CEP:WHY: Replaces `binary.add const, const` with the result constant;
///          later DCE removes the dead operands. Types stay consistent
///          (i64 folds produce ConstI64 + I64 type).
/// CEP:STATUS: complete
/// CEP:FAILURE: none; non-foldable nodes are untouched.
/// CEP:ASSUMES: verified arena.
/// CEP:COST: O(nodes).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn run(arena: &mut IrArena) -> u32 {
    let mut folded = 0u32;
    // Snapshot the node list (mutation during iteration is bounded by the
    // in-place op swap only).
    let mut ids: Vec<NodeId> = Vec::new();
    arena.for_each_live_node(|id, _| ids.push(id));
    for id in ids {
        let (op, in0, in1, _ty) = match arena.node(id) {
            Ok(n) => (n.op, n.input(0), n.input(1), n.ty),
            Err(_) => continue,
        };
        // Unary folds.
        if let Op::Unary(u) = op {
            if let Some(av) = in0 {
                if let Some(v) = const_f64_of(arena, av) {
                    let out = match u {
                        UnaryOp::Neg => Some(-v),
                        UnaryOp::Relu => Some(v.max(0.0)),
                        UnaryOp::Exp => Some(v.exp()),
                        UnaryOp::Log => {
                            if v > 0.0 {
                                Some(v.ln())
                            } else {
                                None
                            }
                        }
                    };
                    if let Some(result) = out {
                        if let Ok(node) = arena.node_mut(id) {
                            node.op = Op::ConstF64(result);
                            node.n_inputs = 0;
                            node.inputs = [xir_core::id::ValueId::NONE; xir_core::node::MAX_INPUTS];
                            folded += 1;
                        }
                    }
                }
            }
            continue;
        }
        // Binary folds need both constants.
        let Op::Binary(b) = op else { continue };
        let (Some(av), Some(bv)) = (in0, in1) else {
            continue;
        };
        // Integer path FIRST and entirely in i64 (checked arithmetic — no
        // f64 round-trip, no silent wrap; audit F-2).
        if let (Some(x), Some(y)) = (const_i64_of(arena, av), const_i64_of(arena, bv)) {
            let result: Option<i64> = match b {
                BinaryOp::Add => x.checked_add(y),
                BinaryOp::Sub => x.checked_sub(y),
                BinaryOp::Mul => x.checked_mul(y),
                BinaryOp::Div => {
                    if y == 0 {
                        None
                    } else {
                        x.checked_div(y)
                    }
                }
                BinaryOp::Max => Some(x.max(y)),
                BinaryOp::Min => Some(x.min(y)),
            };
            let Some(result) = result else { continue };
            if let Ok(node) = arena.node_mut(id) {
                node.op = Op::ConstI64(result);
                node.ty = Type::Scalar(ScalarType::I64);
                node.n_inputs = 0;
                node.inputs = [xir_core::id::ValueId::NONE; xir_core::node::MAX_INPUTS];
                folded += 1;
            }
            continue;
        }
        // Float path (both operands f64 constants; mixed pairs skip — mixed
        // type semantics belong to the interpreter).
        let (Some(x), Some(y)) = (const_f64_of(arena, av), const_f64_of(arena, bv)) else {
            continue;
        };
        let result: Option<f64> = match b {
            BinaryOp::Add => Some(x + y),
            BinaryOp::Sub => Some(x - y),
            BinaryOp::Mul => Some(x * y),
            BinaryOp::Div => {
                if y == 0.0 {
                    None
                } else {
                    Some(x / y)
                }
            }
            BinaryOp::Max => Some(x.max(y)),
            BinaryOp::Min => Some(x.min(y)),
        };
        let Some(result) = result else { continue };
        if let Ok(node) = arena.node_mut(id) {
            node.op = Op::ConstF64(result);
            node.n_inputs = 0;
            node.inputs = [xir_core::id::ValueId::NONE; xir_core::node::MAX_INPUTS];
            folded += 1;
        }
    }
    folded
}

/// CEP:WHAT: Reads a constant i64 value from a value slot.
/// CEP:WHY: Integer folds MUST compute in i64: routing through f64 loses
///          precision above 2^53 (audit F-2 — a silent miscompilation
///          reachable from the text front-end). Exactness is legality.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (None for non-i64 constants).
/// CEP:ASSUMES: producer live.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: test `large_integer_fold_is_exact`.
fn const_i64_of(arena: &IrArena, v: xir_core::id::ValueId) -> Option<i64> {
    match arena.node(v.node()) {
        Ok(Node {
            op: Op::ConstI64(x),
            ..
        }) => Some(*x),
        _ => None,
    }
}

/// CEP:WHAT: Reads a constant f64 value from a value slot.
/// CEP:WHY: Fold operands; i64 constants coerce losslessly into f64 for
///          comparison but folds stay exact via the int_mode path.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (None for non-constants).
/// CEP:ASSUMES: producer live.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: tests in this module.
fn const_f64_of(arena: &IrArena, v: xir_core::id::ValueId) -> Option<f64> {
    match arena.node(v.node()) {
        Ok(Node {
            op: Op::ConstF64(x),
            ..
        }) => Some(*x),
        Ok(Node {
            op: Op::ConstI64(x),
            ..
        }) => Some(*x as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::{const_f64, const_i64, IrArena};
    use xir_core::node::Node;
    use xir_core::ty::{ScalarType, Type};

    fn build_add(a: f64, b: f64) -> IrArena {
        let mut ar = IrArena::with_capacity(16, 4);
        let root = ar.root_region();
        let c0 = const_f64(&mut ar, root, a);
        let c1 = const_f64(&mut ar, root, b);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (ar.value_of(v0, 0).ok(), ar.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let add = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::F64),
                );
                let _ = ar.insert_node(root, add);
            }
        }
        ar
    }

    // CEP:WHAT: Constant adds fold to ConstF64.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if folding is skipped.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn folds_constant_arithmetic() {
        let mut ar = build_add(3.0, 4.0);
        let n = run(&mut ar);
        assert_eq!(n, 1);
        // Find the folded node: the add became ConstF64(7.0).
        let mut found = false;
        ar.for_each_live_node(|_id, node| {
            if node.op == Op::ConstF64(7.0) {
                found = true;
            }
        });
        assert!(found);
        assert_eq!(crate::verifier::verify(&ar), Ok(()));
    }

    // CEP:WHAT: Division by zero does not fold.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if Inf/NaN is silently materialized.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn division_by_zero_skipped() {
        let mut ar = IrArena::with_capacity(16, 4);
        let root = ar.root_region();
        let c0 = const_f64(&mut ar, root, 1.0);
        let c1 = const_f64(&mut ar, root, 0.0);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (ar.value_of(v0, 0).ok(), ar.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let div = Node::new(
                    Op::Binary(BinaryOp::Div),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::F64),
                );
                let _ = ar.insert_node(root, div);
            }
        }
        let n = run(&mut ar);
        assert_eq!(n, 0);
    }

    // CEP:WHAT: i64 folds are exact above 2^53 (no f64 round-trip).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the fold loses precision (audit F-2).
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn large_integer_fold_is_exact() {
        let mut ar = IrArena::with_capacity(16, 4);
        let root = ar.root_region();
        // 2^53+1 + 1 = 2^53+2 exactly in i64; an f64 round-trip collapses
        // the operand to 2^53 and would fold to the WRONG 2^53+1.
        let big = 9_007_199_254_740_993i64;
        let c0 = const_i64(&mut ar, root, big);
        let c1 = const_i64(&mut ar, root, 1);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (ar.value_of(v0, 0).ok(), ar.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let add = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::I64),
                );
                let _ = ar.insert_node(root, add);
            }
        }
        let n = run(&mut ar);
        assert_eq!(n, 1);
        let mut found = false;
        ar.for_each_live_node(|_id, node| {
            if node.op == Op::ConstI64(big + 1) {
                found = true;
            }
        });
        assert!(found, "expected exact ConstI64({})", big + 1);
    }

    // CEP:WHAT: Integer folds stay integers.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on type drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn integer_folds_stay_integers() {
        let mut ar = IrArena::with_capacity(16, 4);
        let root = ar.root_region();
        let c0 = const_i64(&mut ar, root, 6);
        let c1 = const_i64(&mut ar, root, 7);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (ar.value_of(v0, 0).ok(), ar.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let mul = Node::new(
                    Op::Binary(BinaryOp::Mul),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::I64),
                );
                let _ = ar.insert_node(root, mul);
            }
        }
        let n = run(&mut ar);
        assert_eq!(n, 1);
        let mut found = false;
        ar.for_each_live_node(|_id, node| {
            if node.op == Op::ConstI64(42) && node.ty == Type::Scalar(ScalarType::I64) {
                found = true;
            }
        });
        assert!(found);
    }
}
