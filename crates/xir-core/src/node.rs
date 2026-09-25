// CEP:FILE: crates/xir-core/src/node.rs
// CEP:WHAT: The IR node record — op, region, value inputs, effect chain.
// CEP:WHY: HPC-IR contract (CEP&CC 38.17): IR must not rely on pointer
//          identity or hidden state. Nodes are plain values with stable
//          handles (NodeId), fixed-capacity input arrays (CEP-0: no hidden
//          allocation), strictly-typed edge roles (Value = inputs[], Effect =
//          effect_in/out chain, Control = region membership, Memory =
//          address-space-typed MemRefs, Schedule = level-3 dep lists).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: node insertion beyond MAX_INPUTS returns ArenaError::InputOverflow
//              from the arena; no panic paths.
// CEP:ASSUMES: inputs beyond n_inputs are NONE sentinels (maintained only by
//              arena constructors and edits — invariant checked by verifier).
// CEP:COST: 128 bytes per node; O(1) field access.
// CEP:EVIDENCE: verifier tests in xir-graph; arena tests here.
// CEP:SECURITY: no pointers; value semantics only.
// CEP:HPC-DETERMINISM: deterministic; field order is fixed.
//! IR node record.

use crate::id::{NodeId, RegionId, ValueId};
use crate::op::Op;
use crate::ty::Type;

/// Maximum value inputs per node.
///
/// CEP:WHAT: Input arity bound.
/// CEP:WHY: Fixed arrays keep nodes allocation-free and cache-dense; the
///          architecture's ops need at most 3 value inputs (matmul/conv);
///          graph.custom may chain up to 6. Exceeding the bound is a loud
///          error, never a silent Vec (Law 1/6).
/// CEP:STATUS: complete
/// CEP:FAILURE: ArenaError::InputOverflow beyond 6.
/// CEP:ASSUMES: none
/// CEP:COST: 6 * 8 bytes per node.
/// CEP:EVIDENCE: arena tests.
pub const MAX_INPUTS: usize = 6;

/// Maximum outputs per node.
///
/// CEP:WHAT: Output arity bound (drives ValueId slot packing).
/// CEP:WHY: graph.if yields two region results; no current op exceeds 2;
///          bound keeps ValueId's slot field at 4 bits (id.rs static assert).
/// CEP:STATUS: complete
/// CEP:FAILURE: arena construction beyond 2 outputs returns OutputOverflow.
/// CEP:ASSUMES: none
/// CEP:COST: compile-time bound.
/// CEP:EVIDENCE: id.rs static assert + arena tests.
pub const MAX_OUTPUTS: usize = 2;

/// One IR node.
///
/// CEP:WHAT: Plain-value node record.
/// CEP:WHY: Sea-of-nodes storage: value edges reference ValueIds (SSA),
///          control edges come from region membership, effect edges chain
///          through effect_in, and the producing op carries immediates.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (arena-level ops report errors).
/// CEP:ASSUMES: see module header.
/// CEP:COST: O(1) access; 128 bytes.
/// CEP:EVIDENCE: arena + verifier tests.
/// CEP:SECURITY: value semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    /// Opcode + immediates.
    pub op: Op,
    /// Result type (slot 0; slots 1.. for multi-output ops).
    pub ty: Type,
    /// Secondary result type (NONE for single-output ops).
    pub ty1: Type,
    /// Owning control region.
    pub region: RegionId,
    /// Value inputs (SSA use list); NONE-padded.
    pub inputs: [ValueId; MAX_INPUTS],
    /// Live input count.
    pub n_inputs: u8,
    /// Effect-token input (side-effect ordering); NONE for pure ops.
    pub effect_in: ValueId,
    /// Effect-token output (this node's position in the effect chain);
    /// NONE when the op does not produce a new token.
    pub effect_out: ValueId,
    /// Intrusive next node in the same region (scheduling order);
    /// NONE at list end.
    pub next_in_region: NodeId,
}

impl Node {
    /// CEP:WHAT: Builds a node with NONE-padded inputs.
    /// CEP:WHY: Single constructor maintains the padding invariant.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (arity checked at arena insertion).
    /// CEP:ASSUMES: inputs.len() <= MAX_INPUTS (debug assert mirrors).
    /// CEP:COST: O(MAX_INPUTS) copy.
    /// CEP:EVIDENCE: arena tests.
    pub fn new(op: Op, region: RegionId, inputs: &[ValueId], ty: Type) -> Node {
        debug_assert!(inputs.len() <= MAX_INPUTS);
        let mut padded = [ValueId::NONE; MAX_INPUTS];
        for (i, v) in inputs.iter().enumerate() {
            if i < MAX_INPUTS {
                padded[i] = *v;
            }
        }
        Node {
            op,
            ty,
            ty1: Type::None,
            region,
            inputs: padded,
            n_inputs: inputs.len().min(MAX_INPUTS) as u8,
            effect_in: ValueId::NONE,
            effect_out: ValueId::NONE,
            next_in_region: NodeId::NONE,
        }
    }

    /// CEP:WHAT: Input accessor with bounds check.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None when i >= n_inputs.
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 compares + load.
    /// CEP:EVIDENCE: verifier tests.
    pub const fn input(&self, i: usize) -> Option<ValueId> {
        if i < self.n_inputs as usize && i < MAX_INPUTS {
            Some(self.inputs[i])
        } else {
            None
        }
    }

    /// CEP:WHAT: Hashes the node canonically (type + op + inputs + region).
    /// CEP:WHY: GVN value numbering and snapshot fingerprints share one
    ///          canonical form (CEP&CC 38.19: stable IR hashing).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: caller mixes the node id separately when identity
    ///              matters (GVN hashes value structure only).
    /// CEP:COST: O(1) fixed field hashing.
    /// CEP:EVIDENCE: GVN tests; snapshot hash tests.
    pub fn hash_into(&self, h: &mut crate::hash::Fnv64) {
        self.op.hash_into(h);
        for i in 0..self.n_inputs as usize {
            if i < MAX_INPUTS {
                h.write_u64(self.inputs[i].0);
            }
        }
        h.write_u64(self.region.0);
        h.write_u64(self.effect_in.0);
        // Type fingerprint via the debug-free canonical encoding: tag bytes
        // and shape dims.
        hash_type(&self.ty, h);
    }
}

/// CEP:WHAT: Canonical type hashing helper (no Debug formatting — CEP-0).
/// CEP:WHY: Debug strings would allocate; the canonical byte encoding keeps
///          hashing allocation-free and endianness-stable.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(rank)
/// CEP:EVIDENCE: snapshot hash tests
fn hash_type(ty: &Type, h: &mut crate::hash::Fnv64) {
    match ty {
        Type::Scalar(s) => {
            h.write_tag(1);
            h.write_tag(*s as u8);
        }
        Type::Tensor(t) => {
            h.write_tag(2);
            h.write_tag(t.elem as u8);
            h.write_tag(t.layout as u8);
            for d in t.shape.as_slice() {
                h.write_i64(*d);
            }
        }
        Type::Token => h.write_tag(3),
        Type::MemRef(t, space) => {
            h.write_tag(4);
            h.write_tag(t.elem as u8);
            h.write_tag(*space as u8);
            for d in t.shape.as_slice() {
                h.write_i64(*d);
            }
        }
        Type::None => h.write_tag(0),
    }
}
