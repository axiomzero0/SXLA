// CEP:FILE: crates/mss/src/interp.rs
// CEP:WHAT: The differential interpreter — O(n) slot-order evaluation with
//           an explicit values array (no recursion), wrapping i64 semantics
//           matching the mutation operators' Tier-1 preconditions exactly.
// CEP:WHY: Correctness anchor of the whole engine: EVERY search result is
//          differentially evaluated against its input program (design §16:
//          randomized differential testing before any candidate is trusted).
//          Slot-order evaluation is sound because the builder enforces
//          topological order and mutations only REPLACE node definitions in
//           place (operand slots still reference earlier slots).
// CEP:CLASS: CEP-31
// CEP:STATUS: complete
// CEP:FAILURE: Err(InterpError) on structural problems (dead root, missing
//              param, dangling operand) — loud, never a silent wrong value.
// CEP:ASSUMES: topological slot order; Alias chains resolve forward.
// CEP:COST: O(n) time, O(n) values array.
// CEP:EVIDENCE: tests in son.rs + tests/differential.rs.
// CEP:SECURITY: no untrusted input.
//! Differential interpreter.

use crate::son::{BinOp, Kind, Program, UnOp, NO_OPERAND};

/// Interpreter failures.
///
/// CEP:WHAT: Loud failure report.
/// CEP:WHY: The differential test must distinguish structural corruption
///          from value disagreement (38.22 diagnosability).
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size
/// CEP:EVIDENCE: tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpError {
    /// The root slot is not a live node.
    DeadRoot,
    /// An operand slot is dangling.
    DanglingOperand(u32),
    /// A parameter index exceeds the provided binding.
    MissingParam(u32),
    /// An unknown opcode discriminant (corrupt arena).
    UnknownOpcode(u8),
}

/// CEP:WHAT: Evaluates the program's root against the parameter binding.
/// CEP:STATUS: complete
/// CEP:FAILURE: Err(InterpError) — see the enum.
/// CEP:ASSUMES: params covers every Param node's index.
/// CEP:COST: O(n).
/// CEP:EVIDENCE: son.rs roundtrip test; differential suite.
/// CEP:SECURITY: no untrusted input.
pub fn evaluate(program: &Program, params: &[i64]) -> Result<i64, InterpError> {
    let arena = &program.arena;
    if arena.kinds.get(program.root as usize) != Some(&Kind::Const)
        && arena.kinds.get(program.root as usize) != Some(&Kind::Param)
        && arena.kinds.get(program.root as usize) != Some(&Kind::Bin)
        && arena.kinds.get(program.root as usize) != Some(&Kind::Un)
        && arena.kinds.get(program.root as usize) != Some(&Kind::Alias)
    {
        return Err(InterpError::DeadRoot);
    }
    // Values array: NO_VAL marks uncomputed/dead slots.
    const NO_VAL: i64 = 0;
    let mut computed: Vec<bool> = vec![false; arena.kinds.len()];
    let mut values: Vec<i64> = vec![NO_VAL; arena.kinds.len()];
    let resolve = |slot: u32| -> Option<u32> {
        // Follow Alias chains forward (bounded by array length).
        let mut hops = 0;
        let mut cur = slot;
        loop {
            let idx = cur as usize;
            let kind = *arena.kinds.get(idx)?;
            if kind == Kind::Alias {
                let next = *arena.lhs.get(idx)?;
                if next == NO_OPERAND {
                    return None;
                }
                cur = next;
                hops += 1;
                if hops > arena.kinds.len() {
                    return None; // cycle guard (corrupt arena)
                }
            } else {
                return Some(cur);
            }
        }
    };
    for slot in 0..arena.kinds.len() as u32 {
        let idx = slot as usize;
        let kind = arena.kinds[idx];
        if kind == Kind::Dead {
            continue;
        }
        // Resolve through aliases to the defining node.
        let target = match resolve(slot) {
            Some(t) => t,
            None => {
                if slot == program.root {
                    return Err(InterpError::DanglingOperand(slot));
                }
                continue; // dead alias subtree not needed
            }
        };
        if target != slot {
            // This slot is an alias whose target is computed below/above;
            // nothing to compute here (value read via resolve at use time).
            continue;
        }
        let value = match kind {
            Kind::Const => arena.imms[idx],
            Kind::Param => {
                let pidx = arena.imms[idx];
                if pidx < 0 || pidx as usize >= params.len() {
                    return Err(InterpError::MissingParam(pidx as u32));
                }
                params[pidx as usize]
            }
            Kind::Bin => {
                let l = resolve(arena.lhs[idx]).ok_or(InterpError::DanglingOperand(slot))?;
                let r = resolve(arena.rhs[idx]).ok_or(InterpError::DanglingOperand(slot))?;
                let lv = *values
                    .get(l as usize)
                    .filter(|_| computed[l as usize])
                    .ok_or(InterpError::DanglingOperand(l))?;
                let rv = *values
                    .get(r as usize)
                    .filter(|_| computed[r as usize])
                    .ok_or(InterpError::DanglingOperand(r))?;
                match arena.opcodes[idx] {
                    op if op == BinOp::Add as u8 => lv.wrapping_add(rv),
                    op if op == BinOp::Sub as u8 => lv.wrapping_sub(rv),
                    op if op == BinOp::Mul as u8 => lv.wrapping_mul(rv),
                    op if op == BinOp::Min as u8 => lv.min(rv),
                    op if op == BinOp::Max as u8 => lv.max(rv),
                    other => return Err(InterpError::UnknownOpcode(other)),
                }
            }
            Kind::Un => {
                let o = resolve(arena.lhs[idx]).ok_or(InterpError::DanglingOperand(slot))?;
                let ov = *values
                    .get(o as usize)
                    .filter(|_| computed[o as usize])
                    .ok_or(InterpError::DanglingOperand(o))?;
                match arena.opcodes[idx] {
                    op if op == UnOp::Neg as u8 => ov.wrapping_neg(),
                    other => return Err(InterpError::UnknownOpcode(other)),
                }
            }
            Kind::Alias | Kind::Dead => continue,
        };
        computed[idx] = true;
        values[idx] = value;
    }
    let root_target = resolve(program.root).ok_or(InterpError::DeadRoot)?;
    if !computed[root_target as usize] {
        return Err(InterpError::DeadRoot);
    }
    Ok(values[root_target as usize])
}
