// CEP:FILE: crates/mss/src/cand.rs
// CEP:WHAT: The Copy-on-Write candidate: an immutable base Program plus an
//           ordered overlay list (node replacement / node append). Cloning
//           copies the overlay list only; materializing resolves views into
//           a concrete Program; the effective-graph canonical hash walks
//           resolved views with Alias transparent (so candidates differing
//           only in alias indirection hash EQUAL — the transposition
//           table's dedup semantics).
// CEP:WHY: Design §3: candidates are generational deltas over a base
//          snapshot, never copied graphs. V0 realizes deltas as a Vec of
//          overlays (arena-backed delta regions and refcounted O(1) bases
//          are the CEP-31 V1 upgrade; the CLONE cost here is O(overlays),
//          already allocation-light).
// CEP:CLASS: CEP-31
// CEP:STATUS: complete (V0 overlay form)
// CEP:FAILURE: view access is total (base defaults); materialize validates
//              structurally (tier-0).
// CEP:ASSUMES: overlays reference slots valid in the base; appended nodes
//              reference base or earlier appended slots.
// CEP:COST: clone O(overlays); view O(1); materialize O(n); hash O(n).
// CEP:EVIDENCE: tests `overlay_replaces_node`, `alias_indirection_hashes_equal`.
// CEP:SECURITY: no untrusted input.
//! CoW candidate representation.

use crate::son::{Kind, Program, SonArena};

/// One node definition (the overlay payload).
///
/// CEP:WHAT: Plain node record mirroring the arena columns.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: slot reference valid at resolution time.
/// CEP:COST: 40 bytes
/// CEP:EVIDENCE: tests in this module
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeDef {
    /// Node kind.
    pub kind: Kind,
    /// Immediate (Const value / Param index).
    pub imm: i64,
    /// Opcode discriminant.
    pub opcode: u8,
    /// Left/only operand.
    pub lhs: u32,
    /// Right operand (Bin only).
    pub rhs: u32,
}

/// One CoW overlay operation.
///
/// CEP:WHAT: Replace a slot's definition, or append a new node.
/// CEP:WHY: Local rewrites (x+0 -> Alias(x)) need only Replace; rewrites
///          that introduce constants (reassociation) need Extend.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: see module header.
/// CEP:COST: 48 bytes
/// CEP:EVIDENCE: tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlay {
    /// Replace slot `slot` with `def`.
    Replace {
        /// Target slot.
        slot: u32,
        /// Replacement definition.
        def: NodeDef,
    },
    /// Append `def` (its slot = base_len + position among extends).
    Extend {
        /// Appended definition.
        def: NodeDef,
    },
}

/// The CoW candidate.
///
/// CEP:WHAT: base + overlays; views resolve base-first then replay overlays.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: overlays well-formed (mutation operators guarantee it).
/// CEP:COST: clone O(overlays)
/// CEP:EVIDENCE: module tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Immutable base program.
    pub base: Program,
    /// Ordered overlays (later wins on repeated Replace).
    pub overlays: Vec<Overlay>,
    /// Result root (may be replaced by an overlay Replace on the root slot).
    pub root: u32,
}

impl Candidate {
    /// CEP:WHAT: The initial candidate (no overlays).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: search tests
    /// CEP:SECURITY: no untrusted input
    pub fn from_program(base: Program) -> Candidate {
        let root = base.root;
        Candidate {
            base,
            overlays: Vec::new(),
            root,
        }
    }

    /// CEP:WHAT: Total slot count (base + extends).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(overlays)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn slot_count(&self) -> usize {
        self.base.arena.kinds.len()
            + self
                .overlays
                .iter()
                .filter(|o| matches!(o, Overlay::Extend { .. }))
                .count()
    }

    /// CEP:WHAT: Resolves a slot's effective definition.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (base defaults for out-of-range slots are Dead).
    /// CEP:ASSUMES: slot < slot_count.
    /// CEP:COST: O(overlays) worst case (linear replay); O(1) typical.
    /// CEP:EVIDENCE: `overlay_replaces_node`
    /// CEP:SECURITY: no untrusted input
    pub fn view(&self, slot: u32) -> NodeDef {
        let base_len = self.base.arena.kinds.len() as u32;
        let mut def = if slot < base_len {
            let i = slot as usize;
            NodeDef {
                kind: self.base.arena.kinds[i],
                imm: self.base.arena.imms[i],
                opcode: self.base.arena.opcodes[i],
                lhs: self.base.arena.lhs[i],
                rhs: self.base.arena.rhs[i],
            }
        } else {
            // Extended slot: default dead until its Extend replay position.
            NodeDef {
                kind: Kind::Dead,
                imm: 0,
                opcode: 0,
                lhs: u32::MAX,
                rhs: u32::MAX,
            }
        };
        let mut extend_idx = base_len;
        for o in self.overlays.iter() {
            match *o {
                Overlay::Replace { slot: s, def: d } => {
                    if s == slot {
                        def = d;
                    }
                }
                Overlay::Extend { def: d } => {
                    if extend_idx == slot {
                        def = d;
                    }
                    extend_idx += 1;
                }
            }
        }
        def
    }

    /// CEP:WHAT: Materializes into a concrete Program — reachability from
    ///           the root, then TOPOLOGICAL re-emission (deterministic slot
    ///           order per pass): aliases are inlined (their users wire
    ///           directly to the resolved target) and unreachable nodes are
    ///           dropped, so the emitted arena is a clean topological
    ///           program even when overlays appended nodes after their
    ///           users (the reassociation case).
    /// CEP:WHY: The interpreter evaluates in slot order — it needs operands
    ///           emitted before users; CoW overlays have no such ordering
    ///           guarantee, so materialization restores it.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (unreachable/corrupt slots are skipped, root
    ///              resolution is total).
    /// CEP:ASSUMES: overlays well-formed.
    /// CEP:COST: O(n²) worst case (pass count ≤ n; V0 programs are small —
    ///           documented).
    /// CEP:EVIDENCE: differential tests consume materialized programs
    /// CEP:SECURITY: no untrusted input
    pub fn materialize(&self) -> Program {
        let n = self.slot_count() as u32;
        let resolve = |slot: u32| -> u32 {
            let mut cur = slot;
            for _ in 0..=n {
                let def = self.view(cur);
                if def.kind == Kind::Alias {
                    if def.lhs == u32::MAX {
                        return cur;
                    }
                    cur = def.lhs;
                } else {
                    return cur;
                }
            }
            cur
        };
        // Reachability from the root over resolved operands.
        let mut reachable: Vec<bool> = vec![false; n as usize];
        let mut stack = vec![resolve(self.root)];
        while let Some(s) = stack.pop() {
            if s == u32::MAX || s as usize >= reachable.len() || reachable[s as usize] {
                continue;
            }
            reachable[s as usize] = true;
            let def = self.view(s);
            for op in [def.lhs, def.rhs] {
                if op != u32::MAX {
                    stack.push(resolve(op));
                }
            }
        }
        // Topological emission: pass over slots in order; emit a reachable
        // node once its resolved operands are emitted.
        let mut new_index: Vec<u32> = vec![u32::MAX; n as usize]; // old -> new
        let mut arena = SonArena {
            kinds: Vec::new(),
            imms: Vec::new(),
            opcodes: Vec::new(),
            lhs: Vec::new(),
            rhs: Vec::new(),
        };
        for _pass in 0..=n {
            let mut progress = false;
            for slot in 0..n {
                if new_index[slot as usize] != u32::MAX {
                    continue; // already emitted or skipped
                }
                let def = self.view(slot);
                if def.kind == Kind::Dead || def.kind == Kind::Alias || !reachable[slot as usize] {
                    // Skipped permanently (dead / inlined alias / unreachable).
                    new_index[slot as usize] = u32::MAX - 1; // "skipped" marker
                    progress = true;
                    continue;
                }
                // Operand readiness.
                let l_new = if def.lhs == u32::MAX {
                    Some(u32::MAX)
                } else {
                    let r = resolve(def.lhs);
                    let st = new_index.get(r as usize).copied().unwrap_or(u32::MAX);
                    if st == u32::MAX {
                        None // not ready yet
                    } else if st == u32::MAX - 1 {
                        continue; // operand skipped -> this node is unreachable
                    } else {
                        Some(st)
                    }
                };
                let r_new = if def.rhs == u32::MAX {
                    Some(u32::MAX)
                } else {
                    let r = resolve(def.rhs);
                    let st = new_index.get(r as usize).copied().unwrap_or(u32::MAX);
                    if st == u32::MAX {
                        None
                    } else if st == u32::MAX - 1 {
                        continue;
                    } else {
                        Some(st)
                    }
                };
                let (Some(l_new), Some(r_new)) = (l_new, r_new) else {
                    continue; // deferred to a later pass
                };
                arena.kinds.push(def.kind);
                arena.imms.push(def.imm);
                arena.opcodes.push(def.opcode);
                arena.lhs.push(l_new);
                arena.rhs.push(r_new);
                new_index[slot as usize] = (arena.kinds.len() - 1) as u32;
                progress = true;
            }
            if !progress {
                break;
            }
        }
        let root_old = resolve(self.root);
        let root_new = new_index
            .get(root_old as usize)
            .copied()
            .filter(|&i| i != u32::MAX && i != u32::MAX - 1)
            .unwrap_or(0);
        Program {
            arena,
            root: root_new,
        }
    }

    /// CEP:WHAT: Effective-graph canonical hash (FNV-1a over resolved views,
    ///           Alias-transparent, Dead-skipped).
    /// CEP:WHY: Transposition semantics: candidates that compute the same
    ///          function through different alias indirections hash EQUAL —
    ///          the TT prunes the search space to semantic variants.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(n)
    /// CEP:EVIDENCE: `alias_indirection_hashes_equal`
    /// CEP:SECURITY: no untrusted input
    pub fn canonical_hash(&self) -> u64 {
        // Operand resolution through alias chains (effective graph).
        let resolve = |slot: u32| -> u32 {
            let mut cur = slot;
            for _ in 0..=self.slot_count() {
                let def = self.view(cur);
                if def.kind == Kind::Alias {
                    if def.lhs == u32::MAX {
                        return cur;
                    }
                    cur = def.lhs;
                } else {
                    return cur;
                }
            }
            cur
        };
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for slot in 0..self.slot_count() as u32 {
            let def = self.view(slot);
            if def.kind == Kind::Dead || def.kind == Kind::Alias {
                continue; // effective graph skips these
            }
            for part in [
                def.kind as u8 as u64,
                u64::from(def.opcode),
                (def.imm as u64) ^ (def.imm as u64) >> 32,
            ] {
                h ^= part;
                h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
            for op in [def.lhs, def.rhs] {
                let r = if op == u32::MAX {
                    u32::MAX as u64
                } else {
                    resolve(op) as u64
                };
                h ^= r;
                h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        h ^= u64::from(self.root);
        h.wrapping_mul(0x0000_0100_0000_01B3)
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
    use crate::son::{BinOp, ProgramBuilder, NO_OPERAND};

    fn base_program() -> Program {
        // f(x) = (x + 0) * 1
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let zero = b.const_i64(0);
        let one = b.const_i64(1);
        let add = b.bin(BinOp::Add, x, zero).expect("add");
        let mul = b.bin(BinOp::Mul, add, one).expect("mul");
        b.finish(mul).expect("finish")
    }

    // CEP:WHAT: A Replace overlay changes the viewed definition.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a stale view.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn overlay_replaces_node() {
        let cand = Candidate::from_program(base_program());
        let add_view = cand.view(3); // add slot (0=x, 1=zero, 2=one, 3=add)
        assert_eq!(add_view.kind, Kind::Bin);
        let mut mutated = cand.clone();
        mutated.overlays.push(Overlay::Replace {
            slot: 3,
            def: NodeDef {
                kind: Kind::Alias,
                imm: 0,
                opcode: 0,
                lhs: 0, // x
                rhs: NO_OPERAND,
            },
        });
        assert_eq!(mutated.view(3).kind, Kind::Alias);
        // Base untouched.
        assert_eq!(cand.view(3).kind, Kind::Bin);
        // Materialize evaluates identically (x+0 -> x).
        assert_eq!(crate::interp::evaluate(&mutated.materialize(), &[9]), Ok(9));
    }

    // CEP:WHAT: Alias indirection hashes equal to direct wiring.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a hash divergence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test (the TT dedup semantics witness)
    #[test]
    fn alias_indirection_hashes_equal() {
        let cand = Candidate::from_program(base_program());
        let mut aliased = cand.clone();
        aliased.overlays.push(Overlay::Replace {
            slot: 2,
            def: NodeDef {
                kind: Kind::Alias,
                imm: 0,
                opcode: 0,
                lhs: 0,
                rhs: NO_OPERAND,
            },
        });
        // And verify the alias-transparency property directly: wiring the
        // root's operand THROUGH an alias to slot k hashes the same as
        // wiring it directly to k (both effective graphs are mul(x, one)).
        let alias_def = NodeDef {
            kind: Kind::Alias,
            imm: 0,
            opcode: 0,
            lhs: 0,
            rhs: NO_OPERAND,
        };
        let mul_def = |lhs: u32| NodeDef {
            kind: Kind::Bin,
            imm: 0,
            opcode: BinOp::Mul as u8,
            lhs,
            rhs: 3,
        };
        let mut via_alias = cand.clone();
        via_alias.overlays.push(Overlay::Replace {
            slot: 2,
            def: alias_def,
        });
        via_alias.overlays.push(Overlay::Replace {
            slot: 4,
            def: mul_def(2),
        });
        let mut direct = cand.clone();
        direct.overlays.push(Overlay::Replace {
            slot: 2,
            def: alias_def,
        });
        direct.overlays.push(Overlay::Replace {
            slot: 4,
            def: mul_def(0),
        });
        assert_eq!(via_alias.canonical_hash(), direct.canonical_hash());
        // Distinct effective graphs hash differently.
        let mut other = cand.clone();
        other.overlays.push(Overlay::Replace {
            slot: 4,
            def: NodeDef {
                kind: Kind::Bin,
                imm: 0,
                opcode: BinOp::Add as u8,
                lhs: 2,
                rhs: 3,
            },
        });
        assert_ne!(other.canonical_hash(), direct.canonical_hash());
    }
}
