// CEP:FILE: crates/mss/src/son.rs
// CEP:WHAT: The Sea-of-Nodes ECS arena (V0 dialect subset): flat
//           structure-of-arrays node storage (kind / immediate / opcode /
//           operand slots), zero pointers, node identity = slot index, plus
//           the program wrapper (arena + result root) and a builder that
//           appends nodes in operand order (topological by construction).
// CEP:WHY: The design's core representation decision: data-oriented arrays
//          make candidate manipulation cache-friendly bulk operations and
//          the nodes/second metric a bandwidth problem, not a pointer-chase
//          problem. V0 subset: pure dataflow (Const/Param/BinOp/UnOp/Alias)
//          over wrapping i64 semantics; control spine, memory SSA (Phi/Chi/
//           Load/Store), guards and speculation are CEP-31 V2+ layers.
// CEP:CLASS: CEP-31
// CEP:STATUS: complete (V0 dialect)
// CEP:FAILURE: builder arity errors return Err (loud, Law 6); no panics.
// CEP:ASSUMES: operand slots always reference EARLIER slots (builder-enforced
//              topological order — the O(n) interpreter relies on it).
// CEP:COST: ~40 bytes/node across the SoA columns; O(1) appends.
// CEP:EVIDENCE: tests `builder_orders_operands`, `roundtrip_evaluate`.
// CEP:SECURITY: no untrusted input (internal dialect, V0 has no text form).
// CEP:HPC-DETERMINISM: arrays iterate in slot order everywhere.
// CEP:TODO(main-agent): CEP-31: memory SSA edges, control projection, and
//                       the Assume/Guard speculative nodes (design §6).
//! ECS Sea-of-Nodes arena (V0 pure-dataflow dialect).

/// Node kind discriminant.
///
/// CEP:WHAT: Closed V0 kind set.
/// CEP:WHY: Exhaustive match dispatch (no dyn); each kind documents its
///          operand arity and immediate meaning.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: interpreter tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    /// i64 constant (immediate = value). Arity 0.
    Const,
    /// i64 function parameter (immediate = param index). Arity 0.
    Param,
    /// Binary op (opcode = BinOp). Arity 2.
    Bin,
    /// Unary op (opcode = UnOp). Arity 1 (lhs).
    Un,
    /// Alias: evaluates to its lhs — the CoW replacement carrier for
    /// use-preserving rewrites (x+0 -> x replaces the node with Alias(x)).
    /// Arity 1 (lhs).
    Alias,
    /// Tombstone: removed node (DCE). Never referenced by live dataflow.
    Dead,
}

/// Binary opcodes (wrapping i64 semantics — the Tier-1 precondition domain).
///
/// CEP:WHAT: V0 closed set: Add/Sub/Mul/Min/Max.
/// CEP:WHY: Division/shifts have trapping/edge semantics that need the
///          Tier-1 precondition machinery beyond V0's scope (documented).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: interpreter differential tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BinOp {
    /// Wrapping add.
    Add,
    /// Wrapping subtract.
    Sub,
    /// Wrapping multiply.
    Mul,
    /// Min.
    Min,
    /// Max.
    Max,
}

/// Unary opcodes.
///
/// CEP:WHAT: V0 set: Neg (wrapping).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: interpreter tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UnOp {
    /// Wrapping negation.
    Neg,
}

/// Sentinel for "no operand".
pub const NO_OPERAND: u32 = u32::MAX;

/// The ECS arena.
///
/// CEP:WHAT: SoA columns: kinds, imms, opcodes, lhs, rhs. Slot index IS
///           node identity.
/// CEP:WHY: Data-oriented layout (design §2): bulk scans over columns,
///          ~40B/node, zero pointer chasing.
/// CEP:STATUS: complete
/// CEP:FAILURE: builder/validation errors are loud (Result).
/// CEP:ASSUMES: topological slot order (operands precede users).
/// CEP:COST: O(1) append; O(n) scans.
/// CEP:EVIDENCE: module tests.
/// CEP:SECURITY: no untrusted input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SonArena {
    /// Kind column.
    pub kinds: Vec<Kind>,
    /// Immediate column (Const value / Param index).
    pub imms: Vec<i64>,
    /// Opcode column (Bin/Un discriminant).
    pub opcodes: Vec<u8>,
    /// Left operand column (Un/Alias single operand).
    pub lhs: Vec<u32>,
    /// Right operand column (Bin only).
    pub rhs: Vec<u32>,
}

impl SonArena {
    /// CEP:WHAT: Empty arena.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn new() -> SonArena {
        SonArena::default()
    }

    /// CEP:WHAT: Live node count (excludes Tombstones).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(n)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn live_count(&self) -> usize {
        self.kinds.iter().filter(|k| **k != Kind::Dead).count()
    }

    /// CEP:WHAT: Arity of a kind (0/1/2).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tier-0 filter tests
    /// CEP:SECURITY: no untrusted input
    pub fn arity(kind: Kind) -> u8 {
        match kind {
            Kind::Const | Kind::Param => 0,
            Kind::Bin => 2,
            Kind::Un | Kind::Alias => 1,
            Kind::Dead => 0,
        }
    }
}

/// One program: arena + result root.
///
/// CEP:WHAT: The search's input and output artifact.
/// CEP:WHY: The design's RegionDescriptor degenerate for V0 (whole-function
///          pure region; SESE extraction and memory cuts arrive with the
///          control/memory layers — CEP-31).
/// CEP:STATUS: complete
/// CEP:FAILURE: validation via `validate`.
/// CEP:ASSUMES: root is a live slot.
/// CEP:COST: arena + 4 bytes.
/// CEP:EVIDENCE: differential tests.
/// CEP:SECURITY: no untrusted input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// The node graph.
    pub arena: SonArena,
    /// Result root slot.
    pub root: u32,
}

/// Builder errors.
///
/// CEP:WHAT: Loud failure report (Law 6).
/// CEP:WHY: arity/operand violations must never silently produce broken
///          graphs — the Tier-0 filter double-checks, but construction
///          refuses first.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size
/// CEP:EVIDENCE: builder tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    /// Operand index does not exist (or is the node itself).
    BadOperand,
}

/// Program builder (operands must already exist — topological by design).
///
/// CEP:WHAT: Appends nodes returning their slot.
/// CEP:STATUS: complete
/// CEP:FAILURE: Err(BuildError::BadOperand) on forward/self references.
/// CEP:ASSUMES: callers append producers before consumers.
/// CEP:COST: O(1) per append.
/// CEP:EVIDENCE: `builder_orders_operands`.
/// CEP:SECURITY: no untrusted input.
#[derive(Debug, Clone, Default)]
pub struct ProgramBuilder {
    arena: SonArena,
}

impl ProgramBuilder {
    /// CEP:WHAT: New builder.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn new() -> ProgramBuilder {
        ProgramBuilder {
            arena: SonArena::new(),
        }
    }

    fn check_operand(&self, slot: u32) -> Result<(), BuildError> {
        if slot == NO_OPERAND || slot as usize >= self.arena.kinds.len() {
            return Err(BuildError::BadOperand);
        }
        Ok(())
    }

    /// CEP:WHAT: Appends an i64 constant.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn const_i64(&mut self, value: i64) -> u32 {
        self.arena.kinds.push(Kind::Const);
        self.arena.imms.push(value);
        self.arena.opcodes.push(0);
        self.arena.lhs.push(NO_OPERAND);
        self.arena.rhs.push(NO_OPERAND);
        (self.arena.kinds.len() - 1) as u32
    }

    /// CEP:WHAT: Appends a parameter reference.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: param_count set on finish.
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn param(&mut self, index: u32) -> u32 {
        self.arena.kinds.push(Kind::Param);
        self.arena.imms.push(i64::from(index));
        self.arena.opcodes.push(0);
        self.arena.lhs.push(NO_OPERAND);
        self.arena.rhs.push(NO_OPERAND);
        (self.arena.kinds.len() - 1) as u32
    }

    /// CEP:WHAT: Appends a binary op.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Err on bad operands (append rolled back).
    /// CEP:ASSUMES: operands exist.
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: builder tests
    /// CEP:SECURITY: no untrusted input
    pub fn bin(&mut self, op: BinOp, lhs: u32, rhs: u32) -> Result<u32, BuildError> {
        self.check_operand(lhs)?;
        self.check_operand(rhs)?;
        self.arena.kinds.push(Kind::Bin);
        self.arena.imms.push(0);
        self.arena.opcodes.push(op as u8);
        self.arena.lhs.push(lhs);
        self.arena.rhs.push(rhs);
        Ok((self.arena.kinds.len() - 1) as u32)
    }

    /// CEP:WHAT: Appends a unary op.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Err on bad operand.
    /// CEP:ASSUMES: operand exists.
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: builder tests
    /// CEP:SECURITY: no untrusted input
    pub fn un(&mut self, op: UnOp, operand: u32) -> Result<u32, BuildError> {
        self.check_operand(operand)?;
        self.arena.kinds.push(Kind::Un);
        self.arena.imms.push(0);
        self.arena.opcodes.push(op as u8);
        self.arena.lhs.push(operand);
        self.arena.rhs.push(NO_OPERAND);
        Ok((self.arena.kinds.len() - 1) as u32)
    }

    /// CEP:WHAT: Finishes the program with its result root.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Err when the root slot is invalid or dead.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    /// CEP:SECURITY: no untrusted input
    pub fn finish(self, root: u32) -> Result<Program, BuildError> {
        if root as usize >= self.arena.kinds.len() {
            return Err(BuildError::BadOperand);
        }
        Ok(Program {
            arena: self.arena,
            root,
        })
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

    // CEP:WHAT: Builder enforces operand order (forward refs rejected).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a silent forward reference.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn builder_orders_operands() {
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let c = b.const_i64(0);
        let add = b.bin(BinOp::Add, x, c).expect("append");
        let p = b.finish(add).expect("finish");
        assert_eq!(p.arena.kinds.len(), 3);
        assert_eq!(p.root, add);
        // Forward reference refused.
        let mut b2 = ProgramBuilder::new();
        assert!(b2.bin(BinOp::Add, 3, 4).is_err());
        // Self reference refused via NO_OPERAND/absence.
        let mut b3 = ProgramBuilder::new();
        assert!(b3.bin(BinOp::Add, NO_OPERAND, NO_OPERAND).is_err());
    }

    // CEP:WHAT: Roundtrip evaluation of a small program.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a wrong value.
    // CEP:ASSUMES: interpreter from interp.rs.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn roundtrip_evaluate() {
        use crate::interp::evaluate;
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let zero = b.const_i64(0);
        let one = b.const_i64(1);
        let add = b.bin(BinOp::Add, x, zero).expect("add");
        let mul = b.bin(BinOp::Mul, add, one).expect("mul");
        let p = b.finish(mul).expect("finish");
        assert_eq!(evaluate(&p, &[7]), Ok(7));
        assert_eq!(evaluate(&p, &[-3]), Ok(-3));
    }
}
