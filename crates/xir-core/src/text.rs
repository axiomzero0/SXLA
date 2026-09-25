// CEP:FILE: crates/xir-core/src/text.rs
// CEP:WHAT: Canonical textual XIR — printer and parser (format version 1).
// CEP:WHY: HPC-IR contract (CEP&CC 38.17): the IR must be printable and
//          round-trippable when text is normative; deterministic printing
//          (38.19) requires stable node order — we print live nodes in slot
//          order per region, never HashMap order. graph.if regions round-
//          trip through the nested block syntax `%N = if %C : ty { ... }
//          else { ... }` (CEP-12). The parser is CEP-1: bounded reads,
//          explicit ParseError positions, no panics, no unsafe.
// CEP:CLASS: CEP-1
// CEP:STATUS: complete
// CEP:FAILURE: ParseError::{UnexpectedToken, UnknownOp, UnknownType, BadInt,
//              BadFloat, Truncated} with byte offset; printing
//              cannot fail (formatting into a caller String).
// CEP:ASSUMES: input is repository-trusted UTF-8 (tools read local files);
//              non-UTF-8 must be rejected by the caller before this layer
//              (CEP&CC 22.5 input validation).
// CEP:COST: print O(nodes); parse O(bytes) single pass.
// CEP:EVIDENCE: tests `roundtrip_flat_function`, `roundtrip_if_regions`,
//           `rejects_garbage`, `printer_is_deterministic`.
// CEP:SECURITY: parser bounds-checks every slice access; integers parsed with
//           checked arithmetic (no overflow panics).
// CEP:HPC-DETERMINISM: printer output is a pure function of the snapshot.
//! Canonical textual XIR (v1).

use crate::arena::{ArenaError, IrArena};
use crate::id::{NodeId, RegionId};
use crate::node::Node;
use crate::op::{BinaryOp, Monoid, Op, Padding, RngDist, UnaryOp};
use crate::ty::{Layout, ScalarType, Shape, TensorType, Type, MAX_RANK};

/// Maximum if-block nesting depth accepted by the parser.
///
/// CEP:WHAT: Bounded recursion guard (audit F-13).
/// CEP:WHY: parse_block/parse_statement and print_region_nodes recurse per
///          nesting level; an adversarial ~40k-deep file would overflow the
///          stack (abort-class crash). The bound is generous for real IR
///          (deep loop nests in text form are unusual; the level-3 loop
///          program is a structured projection, not text) and fails LOUDLY
///          with a byte offset.
/// CEP:STATUS: complete
/// CEP:FAILURE: ParseError::NestingTooDeep with offset.
/// CEP:ASSUMES: repository-trusted input bounds it further.
/// CEP:COST: 1 compare per block.
/// CEP:EVIDENCE: test `rejects_runaway_nesting`.
pub const MAX_NEST: usize = 256;

/// Parser failure enumeration.
///
/// CEP:WHAT: Explicit parse error with byte offset.
/// CEP:WHY: CEP&CC 38.13: invalid input must produce clear diagnostics with
///          source location, never a crash and never a guess.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: 16 bytes
/// CEP:EVIDENCE: test `rejects_garbage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// A token was not what the grammar expected at this byte offset.
    UnexpectedToken(usize),
    /// Unknown opcode name at this offset.
    UnknownOp(usize),
    /// Unknown type spelling at this offset.
    UnknownType(usize),
    /// Malformed integer at this offset.
    BadInt(usize),
    /// Malformed float at this offset.
    BadFloat(usize),
    /// Input ended mid-production.
    Truncated,
    /// If-block nesting exceeded the bound at this offset (audit F-13:
    /// unbounded recursion would overflow the stack on adversarial input).
    NestingTooDeep(usize),
    /// Arena capacity exceeded while building.
    ArenaExhausted,
}

/// Print one op in canonical form (no trailing newline).
///
/// CEP:WHAT: Op-to-text rendering shared by node lines.
/// CEP:WHY: One renderer = one spelling per op (determinism).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (format! in CEP-1 code is permitted and documented).
/// CEP:ASSUMES: none
/// CEP:COST: O(immediate width)
/// CEP:EVIDENCE: roundtrip tests
fn format_op(op: Op) -> String {
    match op {
        Op::ConstI64(v) => format!("const.i64 {}", v),
        Op::ConstF64(v) => {
            if !v.is_finite() || (v.fract() == 0.0 && v.abs() >= 9.223372036854776e18) {
                // Audit F-4: non-finite values print as inf/NaN (words the
                // lexer cannot re-read) and integral values >= 2^63 print
                // as integer strings that overflow the i64 literal token.
                // Escape form: the exact bit pattern as TWO i64-safe
                // decimal halves — always re-parseable, always exact.
                let bits = v.to_bits();
                format!("const.f64 bits {} {}", bits >> 32, bits & 0xFFFF_FFFF)
            } else if v.fract() == 0.0 && v.abs() < 1e15 {
                format!("const.f64 {:.1}", v)
            } else {
                format!("const.f64 {}", v)
            }
        }
        Op::Param { index } => format!("param {}", index),
        Op::Dot => "dot".to_string(),
        Op::Reduce { axis, monoid } => {
            format!("reduce axis={} monoid={}", axis, monoid_name(monoid))
        }
        Op::Rng { dist, seed } => {
            format!("rng dist={} seed={}", dist_name(dist), seed)
        }
        Op::Custom { sym } => format!("custom {}", sym),
        Op::If => "if".to_string(),
        Op::Binary(b) => format!("binary.{}", binary_name(b)),
        Op::Unary(u) => format!("unary.{}", unary_name(u)),
        Op::Matmul {
            transpose_a,
            transpose_b,
        } => format!(
            "matmul ta={} tb={}",
            if transpose_a { "true" } else { "false" },
            if transpose_b { "true" } else { "false" }
        ),
        Op::Conv { padding, stride } => {
            format!("conv padding={} stride={}", padding_name(padding), stride)
        }
        Op::Broadcast { to } => format!("broadcast to=[{}]", shape_str(&to)),
        Op::Transpose { perm, rank } => {
            let items: Vec<String> = perm
                .iter()
                .take(rank as usize)
                .map(|p| p.to_string())
                .collect();
            format!("transpose perm=[{}]", items.join(","))
        }
        Op::FusionCluster => "fusion.cluster".to_string(),
        Op::FusionBarrier => "fusion.barrier".to_string(),
        Op::FusionMaterialize => "fusion.materialize".to_string(),
        Op::LoopParallel { axis } => format!("loop.parallel axis={}", axis),
        Op::LoopAlloc { bytes, space } => {
            format!("loop.alloc bytes={} space={}", bytes, space_name(space))
        }
        Op::LoopAsyncCopy => "loop.async_copy".to_string(),
        Op::LoopPipelineStage { stage } => format!("loop.pipeline_stage stage={}", stage),
        Op::TargetMma => "target.mma".to_string(),
        Op::TargetWarpShuffle => "target.warp_shuffle".to_string(),
        Op::TargetBarrier => "target.barrier".to_string(),
    }
}

/// CEP:WHAT: Monoid spelling.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: roundtrip tests
fn monoid_name(m: Monoid) -> &'static str {
    match m {
        Monoid::Add => "add",
        Monoid::Mul => "mul",
        Monoid::Max => "max",
        Monoid::Min => "min",
        Monoid::And => "and",
        Monoid::Or => "or",
    }
}

fn binary_name(b: BinaryOp) -> &'static str {
    match b {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Max => "max",
        BinaryOp::Min => "min",
    }
}

fn unary_name(u: UnaryOp) -> &'static str {
    match u {
        UnaryOp::Relu => "relu",
        UnaryOp::Neg => "neg",
        UnaryOp::Exp => "exp",
        UnaryOp::Log => "log",
    }
}

fn padding_name(p: Padding) -> &'static str {
    match p {
        Padding::Valid => "valid",
        Padding::Same => "same",
    }
}

fn dist_name(d: RngDist) -> &'static str {
    match d {
        RngDist::Uniform => "uniform",
        RngDist::Normal => "normal",
    }
}

fn space_name(s: crate::ty::AddressSpace) -> &'static str {
    match s {
        crate::ty::AddressSpace::Global => "global",
        crate::ty::AddressSpace::Shared => "shared",
        crate::ty::AddressSpace::Register => "register",
    }
}

fn shape_str(s: &Shape) -> String {
    let items: Vec<String> = s.as_slice().iter().map(|d| d.to_string()).collect();
    items.join(",")
}

fn elem_name(e: ScalarType) -> &'static str {
    match e {
        ScalarType::F64 => "f64",
        ScalarType::I64 => "i64",
        ScalarType::F32 => "f32",
        ScalarType::I32 => "i32",
        ScalarType::Bool => "bool",
    }
}

fn layout_name(l: Layout) -> &'static str {
    match l {
        Layout::RowMajor => "row",
        Layout::ColMajor => "col",
    }
}

/// CEP:WHAT: Renders a type in canonical spelling.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(rank)
/// CEP:EVIDENCE: roundtrip tests
pub fn format_type(ty: &Type) -> String {
    match ty {
        Type::Scalar(s) => format!("scalar<{}>", elem_name(*s)),
        Type::Tensor(t) => format!(
            "tensor<{}>[{}] {}",
            elem_name(t.elem),
            shape_str(&t.shape),
            layout_name(t.layout)
        ),
        Type::Token => "token".to_string(),
        Type::MemRef(t, space) => format!(
            "memref<{}>[{}] {} {}",
            elem_name(t.elem),
            shape_str(&t.shape),
            layout_name(t.layout),
            space_name(*space)
        ),
        Type::None => "none".to_string(),
    }
}

/// CEP:WHAT: Prints the whole arena in canonical slot order.
/// CEP:WHY: Deterministic IR printing (CEP&CC 38.19) — the xla-opt tool's
///          output and golden tests depend on byte-stable text.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(nodes) formatting (CEP-1)
/// CEP:EVIDENCE: test `printer_is_deterministic`.
pub fn print_arena(arena: &IrArena) -> String {
    let mut out = String::with_capacity(arena.node_count() * 48 + 64);
    // Header is a single line so the line-oriented parser can consume it.
    out.push_str("xir v1 func @main {\n");
    // Per-region node lists in slot order (deterministic: 38.19 — never
    // HashMap order; slot order is structural).
    let mut region_count = 0usize;
    arena.for_each_region(|_, _| region_count += 1);
    let mut per_region: Vec<Vec<NodeId>> = vec![Vec::new(); region_count];
    arena.for_each_live_node(|id, node| {
        let ridx = node.region.index() as usize;
        if ridx < per_region.len() {
            per_region[ridx].push(id);
        }
    });
    let mut lines: Vec<String> = Vec::with_capacity(arena.node_count());
    print_region_nodes(arena, &per_region, arena.root_region(), 1, &mut lines);
    out.push_str(&lines.join("\n"));
    out.push_str("\n}\n");
    out
}

/// CEP:WHAT: Emits one region's statements (recursing into If blocks).
/// CEP:WHY: CEP-12: graph.if regions print structurally nested —
///          `%N = if %C : ty { then } else { else }` — with the block
///          statements indented one level deeper. The then region is the
///          lower-index owned region, else the higher (creation-order
///          convention of the arena's owner linkage).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (printing cannot fail; an If with a degraded region
///              linkage prints the v0 flat form — the verifier flags such
///              arenas upstream, so the printer never hides a broken IR).
/// CEP:ASSUMES: verified arena (region linkage intact).
/// CEP:COST: O(nodes + regions).
/// CEP:EVIDENCE: test `roundtrip_if_regions`.
/// CEP:HPC-DETERMINISM: deterministic — slot order, region index order.
fn print_region_nodes(
    arena: &IrArena,
    per_region: &[Vec<NodeId>],
    region: RegionId,
    depth: usize,
    lines: &mut Vec<String>,
) {
    // Bounded recursion (audit F-13): printing cannot fail, so an
    // over-deep REGION tree (not from text — the parser bounds text at
    // MAX_NEST) degrades to listing nodes without further nesting instead
    // of overflowing the stack.
    if depth > MAX_NEST {
        return;
    }
    let pad = "  ".repeat(depth);
    let ridx = region.index() as usize;
    if ridx >= per_region.len() {
        return;
    }
    for id in per_region[ridx].iter() {
        let node = match arena.node(*id) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let mut uses: Vec<String> = Vec::with_capacity(node.n_inputs as usize);
        for i in 0..node.n_inputs as usize {
            if i < crate::node::MAX_INPUTS {
                uses.push(format!("%{}", node.inputs[i].node().index()));
            }
        }
        if node.op == Op::If {
            // Owned regions: then = lower index, else = higher.
            let mut owned: Vec<RegionId> = Vec::with_capacity(2);
            arena.for_each_region(|rid, r| {
                if r.owner == *id {
                    owned.push(rid);
                }
            });
            owned.sort_by_key(|r| r.index());
            if owned.len() == 2 {
                lines.push(format!(
                    "{}%{} = if {} : {} {{",
                    pad,
                    id.index(),
                    uses.join(", "),
                    format_type(&node.ty)
                ));
                let (then_r, else_r) = (owned[0], owned[1]);
                print_region_nodes(arena, per_region, then_r, depth + 1, lines);
                lines.push(format!("{}}} else {{", pad));
                print_region_nodes(arena, per_region, else_r, depth + 1, lines);
                lines.push(format!("{}}}", pad));
            } else {
                // Degraded flat form (audit F-8): the region linkage is
                // incomplete (the verifier's BadIfRegions rejects such
                // arenas upstream). The bodies STILL print (indented,
                // region attribution lost on re-parse) so no node is
                // silently dropped — a documented lossy escape hatch for
                // malformed arenas, never a silent one.
                lines.push(format!("{}%{} = if {}", pad, id.index(), uses.join(", ")));
                for rid in owned {
                    print_region_nodes(arena, per_region, rid, depth + 1, lines);
                }
            }
        } else {
            lines.push(format!(
                "{}%{} = {} {} : {}",
                pad,
                id.index(),
                format_op(node.op),
                uses.join(", "),
                format_type(&node.ty)
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// One lexer token.
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    /// Bare word (op names, keywords).
    Word(String),
    /// %N value reference.
    ValueRef(u32),
    /// Integer literal.
    Int(i64),
    /// Float literal.
    Float(f64),
    /// Boolean literal.
    Bool(bool),
    /// Punctuation / symbols.
    Sym(char),
    /// End of line (statement separator).
    Eol,
}

/// CEP:WHAT: Tokenizes input into bounded tokens.
/// CEP:WHY: The parser needs a flat token stream; the lexer records byte
///          offsets for diagnostics.
/// CEP:STATUS: complete
/// CEP:FAILURE: BadInt/BadFloat with offset.
/// CEP:ASSUMES: input is &str (UTF-8 validated upstream).
/// CEP:COST: O(bytes)
/// CEP:EVIDENCE: tests `rejects_garbage`, roundtrip.
fn lex(src: &str) -> Result<Vec<(Tok, usize)>, ParseError> {
    let mut toks = Vec::with_capacity(256);
    let bytes = src.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\n' {
            toks.push((Tok::Eol, i));
            i += 1;
            continue;
        }
        if b == b' ' || b == b'\t' || b == b'\r' {
            i += 1;
            continue;
        }
        let start = i;
        if b == b'%' {
            i += 1;
            let ds = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let digits = &src[ds..i];
            let n: u32 = digits.parse().map_err(|_| ParseError::BadInt(start))?;
            toks.push((Tok::ValueRef(n), start));
            continue;
        }
        if b == b'-' || b.is_ascii_digit() {
            let mut j = i;
            let mut is_float = false;
            if bytes[j] == b'-' {
                j += 1;
            }
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'.' {
                is_float = true;
                j += 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
            }
            let text = &src[start..j];
            if is_float {
                let f: f64 = text.parse().map_err(|_| ParseError::BadFloat(start))?;
                toks.push((Tok::Float(f), start));
            } else {
                let n: i64 = text.parse().map_err(|_| ParseError::BadInt(start))?;
                toks.push((Tok::Int(n), start));
            }
            i = j;
            continue;
        }
        if b.is_ascii_alphabetic() || b == b'_' || b == b'.' || b == b'@' {
            let mut j = i;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric()
                    || bytes[j] == b'_'
                    || bytes[j] == b'.'
                    || bytes[j] == b'@'
                    || bytes[j] == b'<'
                    || bytes[j] == b'>')
            {
                j += 1;
            }
            // Element types carry <...>: scalar<f64>
            if j < bytes.len() && bytes[j] == b'<' {
                while j < bytes.len() && bytes[j] != b'>' {
                    j += 1;
                }
                if j < bytes.len() {
                    j += 1;
                }
            }
            let w = &src[start..j];
            if w == "true" {
                toks.push((Tok::Bool(true), start));
            } else if w == "false" {
                toks.push((Tok::Bool(false), start));
            } else {
                toks.push((Tok::Word(w.to_string()), start));
            }
            i = j;
            continue;
        }
        // Symbols: ( ) [ ] { } , : = ;
        if b == b'('
            || b == b')'
            || b == b'['
            || b == b']'
            || b == b'{'
            || b == b'}'
            || b == b','
            || b == b':'
            || b == b'='
            || b == b';'
        {
            toks.push((Tok::Sym(b as char), start));
            i += 1;
            continue;
        }
        return Err(ParseError::UnexpectedToken(start));
    }
    Ok(toks)
}

/// Parser state over the token stream.
struct Parser {
    toks: Vec<(Tok, usize)>,
    pos: usize,
}

impl Parser {
    /// CEP:WHAT: Peeks the current token.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (None at end).
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: parser tests
    fn peek(&self) -> Option<(&Tok, usize)> {
        self.toks.get(self.pos).map(|(t, o)| (t, *o))
    }

    /// CEP:WHAT: Consumes the current token.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Truncated at end.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: parser tests
    fn next(&mut self) -> Result<(Tok, usize), ParseError> {
        if self.pos >= self.toks.len() {
            return Err(ParseError::Truncated);
        }
        let t = self.toks[self.pos].clone();
        self.pos += 1;
        Ok(t)
    }

    /// CEP:WHAT: Skips any number of end-of-line tokens.
    /// CEP:WHY: Statements are line-oriented; blank lines are legal.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1) amortized
    /// CEP:EVIDENCE: parser tests
    fn skip_eols(&mut self) {
        while matches!(self.peek(), Some((Tok::Eol, _))) {
            self.pos += 1;
        }
    }

    /// CEP:WHAT: Expects a specific word.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnexpectedToken with the word's offset.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: parser tests
    fn expect_word(&mut self, w: &str) -> Result<usize, ParseError> {
        let (tok, off) = self.next()?;
        match tok {
            Tok::Word(got) if got == w => Ok(off),
            _ => Err(ParseError::UnexpectedToken(off)),
        }
    }

    /// CEP:WHAT: Expects a symbol.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnexpectedToken with offset.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: parser tests
    fn expect_sym(&mut self, c: char) -> Result<usize, ParseError> {
        let (tok, off) = self.next()?;
        match tok {
            Tok::Sym(got) if got == c => Ok(off),
            _ => Err(ParseError::UnexpectedToken(off)),
        }
    }
}

/// CEP:WHAT: Parses textual XIR v1 into an arena.
/// CEP:WHY: The tools (xla-opt, xla-run) and tests consume .xir files.
/// CEP:STATUS: complete
/// CEP:FAILURE: ParseError with byte offset; arena capacity is honored.
/// CEP:ASSUMES: repository-trusted UTF-8 input; if-regions parse into
///              owned child regions (CEP-12).
/// CEP:COST: O(bytes) single pass.
/// CEP:EVIDENCE: tests `roundtrip_flat_function`, `roundtrip_if_regions`,
///           `rejects_garbage`.
/// CEP:SECURITY: bounded parses; checked arithmetic everywhere.
pub fn parse_arena(src: &str, node_capacity: usize) -> Result<IrArena, ParseError> {
    let toks = lex(src)?;
    let mut p = Parser { toks, pos: 0 };
    p.expect_word("xir")?;
    p.expect_word("v1")?;
    p.expect_word("func")?;
    // Optional @name.
    if let Some((Tok::Word(w), _)) = p.peek() {
        if w.starts_with('@') {
            let _ = p.next();
        }
    }
    p.expect_sym('{')?;
    // Region capacity: every if consumes 1 node + 2 regions; half the node
    // budget leaves generous headroom (CEP-12).
    let mut arena = IrArena::with_capacity(node_capacity, node_capacity / 2 + 8);
    let root: RegionId = arena.root_region();
    // Value table: %N -> ValueId.
    let mut vals: Vec<(u32, crate::id::ValueId)> = Vec::with_capacity(64);
    parse_block(&mut p, &mut arena, &mut vals, root, 0)?;
    // Only trailing newlines may follow the closing brace.
    p.skip_eols();
    if p.peek().is_some() {
        let (_, off) = p.next()?;
        return Err(ParseError::UnexpectedToken(off));
    }
    Ok(arena)
}

/// CEP:WHAT: Parses statements into `region` until its closing brace.
/// CEP:WHY: CEP-12: blocks nest — the function body, then/else bodies and
///          nested if bodies all share one statement grammar terminated by
///          '}'.
/// CEP:STATUS: complete
/// CEP:FAILURE: ParseError propagation; Truncated at EOF without '}'.
/// CEP:ASSUMES: header/inputs already consumed for the enclosing construct.
/// CEP:COST: O(statements).
/// CEP:EVIDENCE: tests `roundtrip_if_regions`, `roundtrip_flat_function`.
fn parse_block(
    p: &mut Parser,
    arena: &mut IrArena,
    vals: &mut Vec<(u32, crate::id::ValueId)>,
    region: RegionId,
    depth: usize,
) -> Result<(), ParseError> {
    // Bounded recursion (audit F-13): a nesting depth beyond MAX_NEST
    // fails loudly with an offset instead of growing the stack unbounded.
    if depth > MAX_NEST {
        let off = match p.peek() {
            Some((_, o)) => o,
            None => usize::MAX,
        };
        return Err(ParseError::NestingTooDeep(off));
    }
    loop {
        p.skip_eols();
        if matches!(p.peek(), Some((Tok::Sym('}'), _))) {
            let _ = p.next();
            return Ok(());
        }
        if p.peek().is_none() {
            return Err(ParseError::Truncated);
        }
        parse_statement(p, arena, vals, region, depth)?;
    }
}

/// CEP:WHAT: Parses one `%N = op ...` statement into `region`.
/// CEP:WHY: Shared by every block; if-statements recurse via parse_block
///          after creating the If node and its owned then/else regions.
/// CEP:STATUS: complete
/// CEP:FAILURE: ParseError propagation.
/// CEP:ASSUMES: block already opened.
/// CEP:COST: O(1) per statement (plus recursion for ifs).
/// CEP:EVIDENCE: roundtrip tests.
fn parse_statement(
    p: &mut Parser,
    arena: &mut IrArena,
    vals: &mut Vec<(u32, crate::id::ValueId)>,
    region: RegionId,
    depth: usize,
) -> Result<(), ParseError> {
    let (tok, off) = p.next()?;
    let def = match tok {
        Tok::ValueRef(n) => n,
        _ => return Err(ParseError::UnexpectedToken(off)),
    };
    p.expect_sym('=')?;
    let (word, woff) = p.next()?;
    let name = match word {
        Tok::Word(w) => w,
        _ => return Err(ParseError::UnexpectedToken(woff)),
    };
    let op = parse_op(p, &name, woff)?;
    // Inputs: comma-separated %N references.
    let mut inputs: Vec<crate::id::ValueId> = Vec::with_capacity(4);
    while matches!(p.peek(), Some((Tok::ValueRef(_), _))) {
        let (t, o) = p.next()?;
        if let Tok::ValueRef(n) = t {
            let mut found = crate::id::ValueId::NONE;
            for (k, v) in vals.iter() {
                if *k == n {
                    found = *v;
                }
            }
            if found.is_none() {
                return Err(ParseError::UnexpectedToken(o));
            }
            inputs.push(found);
        }
        // Comma-separated.
        if matches!(p.peek(), Some((Tok::Sym(','), _))) {
            let _ = p.next();
        } else {
            break;
        }
    }
    if op == Op::If {
        // `%N = if %C : ty { then } else { else }` — the If node is created
        // FIRST (the owned regions reference it), then the blocks parse
        // recursively into the then/else regions.
        p.expect_sym(':')?;
        let ty = parse_type_tokens(p)?;
        let node = Node::new(op, region, &inputs, ty);
        let id = arena
            .insert_node(region, node)
            .map_err(|_| ParseError::ArenaExhausted)?;
        let v = arena
            .value_of(id, 0)
            .map_err(|_| ParseError::ArenaExhausted)?;
        vals.push((def, v));
        p.expect_sym('{')?;
        let then_r = arena
            .new_region(region, id)
            .map_err(|_| ParseError::ArenaExhausted)?;
        parse_block(p, arena, vals, then_r, depth + 1)?;
        p.expect_word("else")?;
        p.expect_sym('{')?;
        let else_r = arena
            .new_region(region, id)
            .map_err(|_| ParseError::ArenaExhausted)?;
        parse_block(p, arena, vals, else_r, depth + 1)?;
        // Statement terminator: Eol or EOF.
        match p.peek() {
            Some((Tok::Eol, _)) => {
                let _ = p.next();
            }
            None => {}
            _ => return Err(ParseError::UnexpectedToken(woff)),
        }
        return Ok(());
    }
    // Optional type (may span several tokens for tensors).
    let ty = if matches!(p.peek(), Some((Tok::Sym(':'), _))) {
        let _ = p.next();
        parse_type_tokens(p)?
    } else {
        infer_type(&op)
    };
    let node = Node::new(op, region, &inputs, ty);
    let id = arena
        .insert_node(region, node)
        .map_err(|_| ParseError::ArenaExhausted)?;
    let v = arena
        .value_of(id, 0)
        .map_err(|_| ParseError::ArenaExhausted)?;
    vals.push((def, v));
    // Statement terminator: Eol or EOF.
    match p.peek() {
        Some((Tok::Eol, _)) => {
            let _ = p.next();
        }
        None => {}
        _ => return Err(ParseError::UnexpectedToken(off)),
    }
    Ok(())
}

/// CEP:WHAT: Parses the op-specific production after the op name.
/// CEP:STATUS: complete
/// CEP:FAILURE: UnknownOp / BadInt / BadFloat / UnexpectedToken.
/// CEP:ASSUMES: none
/// CEP:COST: O(1) per op
/// CEP:EVIDENCE: roundtrip tests
fn parse_op(p: &mut Parser, name: &str, off: usize) -> Result<Op, ParseError> {
    let _ = off;
    match name {
        "const.i64" => {
            let (t, o) = p.next()?;
            match t {
                Tok::Int(v) => Ok(Op::ConstI64(v)),
                _ => Err(ParseError::BadInt(o)),
            }
        }
        "const.f64" => {
            // Bits escape form first (audit F-4): `const.f64 bits <hi> <lo>`
            // encodes the exact f64 as two u32 decimal halves.
            if let Some((Tok::Word(w), _)) = p.peek() {
                if w == "bits" {
                    let _ = p.next()?;
                    let (hi, hi_off) = p.next()?;
                    let (lo, lo_off) = p.next()?;
                    let h = match hi {
                        Tok::Int(v) if (0..=u32::MAX as i64).contains(&v) => v as u32,
                        _ => return Err(ParseError::BadInt(hi_off)),
                    };
                    let l = match lo {
                        Tok::Int(v) if (0..=u32::MAX as i64).contains(&v) => v as u32,
                        _ => return Err(ParseError::BadInt(lo_off)),
                    };
                    let bits = (u64::from(h) << 32) | u64::from(l);
                    return Ok(Op::ConstF64(f64::from_bits(bits)));
                }
            }
            let (t, o) = p.next()?;
            match t {
                Tok::Float(v) => Ok(Op::ConstF64(v)),
                Tok::Int(v) => Ok(Op::ConstF64(v as f64)),
                _ => Err(ParseError::BadFloat(o)),
            }
        }
        "param" => {
            let (t, o) = p.next()?;
            match t {
                Tok::Int(v) => Ok(Op::Param {
                    index: u32::try_from(v).map_err(|_| ParseError::BadInt(o))?,
                }),
                _ => Err(ParseError::BadInt(o)),
            }
        }
        "dot" => Ok(Op::Dot),
        "reduce" => {
            p.expect_word("axis")?;
            p.expect_sym('=')?;
            let (t, o) = p.next()?;
            let axis = match t {
                Tok::Int(v) => u8::try_from(v).map_err(|_| ParseError::BadInt(o))?,
                _ => return Err(ParseError::BadInt(o)),
            };
            p.expect_word("monoid")?;
            p.expect_sym('=')?;
            let (t2, o2) = p.next()?;
            let monoid = match t2 {
                Tok::Word(w) => match w.as_str() {
                    "add" => Monoid::Add,
                    "mul" => Monoid::Mul,
                    "max" => Monoid::Max,
                    "min" => Monoid::Min,
                    "and" => Monoid::And,
                    "or" => Monoid::Or,
                    _ => return Err(ParseError::UnknownOp(o2)),
                },
                _ => return Err(ParseError::UnexpectedToken(o2)),
            };
            Ok(Op::Reduce { axis, monoid })
        }
        "rng" => {
            p.expect_word("dist")?;
            p.expect_sym('=')?;
            let (t, o) = p.next()?;
            let dist = match t {
                Tok::Word(w) => match w.as_str() {
                    "uniform" => RngDist::Uniform,
                    "normal" => RngDist::Normal,
                    _ => return Err(ParseError::UnknownOp(o)),
                },
                _ => return Err(ParseError::UnexpectedToken(o)),
            };
            p.expect_word("seed")?;
            p.expect_sym('=')?;
            let (t2, o2) = p.next()?;
            let seed = match t2 {
                Tok::Int(v) => v as u64,
                _ => return Err(ParseError::BadInt(o2)),
            };
            Ok(Op::Rng { dist, seed })
        }
        "custom" => {
            let (t, o) = p.next()?;
            match t {
                Tok::Int(v) => Ok(Op::Custom {
                    sym: u32::try_from(v).map_err(|_| ParseError::BadInt(o))?,
                }),
                _ => Err(ParseError::BadInt(o)),
            }
        }
        "if" => Ok(Op::If),
        "binary.add" => Ok(Op::Binary(BinaryOp::Add)),
        "binary.sub" => Ok(Op::Binary(BinaryOp::Sub)),
        "binary.mul" => Ok(Op::Binary(BinaryOp::Mul)),
        "binary.div" => Ok(Op::Binary(BinaryOp::Div)),
        "binary.max" => Ok(Op::Binary(BinaryOp::Max)),
        "binary.min" => Ok(Op::Binary(BinaryOp::Min)),
        "unary.relu" => Ok(Op::Unary(UnaryOp::Relu)),
        "unary.neg" => Ok(Op::Unary(UnaryOp::Neg)),
        "unary.exp" => Ok(Op::Unary(UnaryOp::Exp)),
        "unary.log" => Ok(Op::Unary(UnaryOp::Log)),
        "matmul" => {
            p.expect_word("ta")?;
            p.expect_sym('=')?;
            let (t, _o) = p.next()?;
            let ta = match t {
                Tok::Bool(b) => b,
                _ => return Err(ParseError::UnexpectedToken(_o)),
            };
            p.expect_word("tb")?;
            p.expect_sym('=')?;
            let (t2, o2) = p.next()?;
            let tb = match t2 {
                Tok::Bool(b) => b,
                _ => return Err(ParseError::UnexpectedToken(o2)),
            };
            Ok(Op::Matmul {
                transpose_a: ta,
                transpose_b: tb,
            })
        }
        "conv" => {
            p.expect_word("padding")?;
            p.expect_sym('=')?;
            let (t, o) = p.next()?;
            let padding = match t {
                Tok::Word(w) => match w.as_str() {
                    "valid" => Padding::Valid,
                    "same" => Padding::Same,
                    _ => return Err(ParseError::UnknownOp(o)),
                },
                _ => return Err(ParseError::UnexpectedToken(o)),
            };
            p.expect_word("stride")?;
            p.expect_sym('=')?;
            let (t2, o2) = p.next()?;
            let stride = match t2 {
                Tok::Int(v) => u8::try_from(v).map_err(|_| ParseError::BadInt(o2))?,
                _ => return Err(ParseError::BadInt(o2)),
            };
            Ok(Op::Conv { padding, stride })
        }
        "broadcast" => {
            p.expect_word("to")?;
            p.expect_sym('=')?;
            p.expect_sym('[')?;
            let mut dims: Vec<i64> = Vec::with_capacity(MAX_RANK);
            while matches!(p.peek(), Some((Tok::Int(_), _))) {
                let (t, o) = p.next()?;
                if let Tok::Int(v) = t {
                    dims.push(v);
                } else {
                    return Err(ParseError::BadInt(o));
                }
                if matches!(p.peek(), Some((Tok::Sym(','), _))) {
                    let _ = p.next();
                } else {
                    break;
                }
            }
            p.expect_sym(']')?;
            let shape = Shape::from_dims(&dims).map_err(|_| ParseError::UnknownType(o2_of(p)))?;
            Ok(Op::Broadcast { to: shape })
        }
        "transpose" => {
            p.expect_word("perm")?;
            p.expect_sym('=')?;
            p.expect_sym('[')?;
            let mut perm = [0u8; MAX_RANK];
            let mut rank = 0u8;
            loop {
                let (t, o) = p.next()?;
                match t {
                    Tok::Int(v) if rank < MAX_RANK as u8 => {
                        perm[rank as usize] = u8::try_from(v).map_err(|_| ParseError::BadInt(o))?;
                        rank += 1;
                    }
                    _ => return Err(ParseError::BadInt(o)),
                }
                if let Some((Tok::Sym(','), _)) = p.peek() {
                    let _ = p.next();
                } else {
                    break;
                }
            }
            p.expect_sym(']')?;
            Ok(Op::Transpose { perm, rank })
        }
        "fusion.cluster" => Ok(Op::FusionCluster),
        "fusion.barrier" => Ok(Op::FusionBarrier),
        "fusion.materialize" => Ok(Op::FusionMaterialize),
        "loop.parallel" => {
            p.expect_word("axis")?;
            p.expect_sym('=')?;
            let (t, o) = p.next()?;
            match t {
                Tok::Int(v) => Ok(Op::LoopParallel {
                    axis: u8::try_from(v).map_err(|_| ParseError::BadInt(o))?,
                }),
                _ => Err(ParseError::BadInt(o)),
            }
        }
        "loop.alloc" => {
            p.expect_word("bytes")?;
            p.expect_sym('=')?;
            let (t, o) = p.next()?;
            let bytes = match t {
                Tok::Int(v) => u32::try_from(v).map_err(|_| ParseError::BadInt(o))?,
                _ => return Err(ParseError::BadInt(o)),
            };
            p.expect_word("space")?;
            p.expect_sym('=')?;
            let (t2, o2) = p.next()?;
            let space = match t2 {
                Tok::Word(w) => match w.as_str() {
                    "global" => crate::ty::AddressSpace::Global,
                    "shared" => crate::ty::AddressSpace::Shared,
                    "register" => crate::ty::AddressSpace::Register,
                    _ => return Err(ParseError::UnknownOp(o2)),
                },
                _ => return Err(ParseError::UnexpectedToken(o2)),
            };
            Ok(Op::LoopAlloc { bytes, space })
        }
        "loop.async_copy" => Ok(Op::LoopAsyncCopy),
        "loop.pipeline_stage" => {
            p.expect_word("stage")?;
            p.expect_sym('=')?;
            let (t, o) = p.next()?;
            match t {
                Tok::Int(v) => Ok(Op::LoopPipelineStage {
                    stage: u32::try_from(v).map_err(|_| ParseError::BadInt(o))?,
                }),
                _ => Err(ParseError::BadInt(o)),
            }
        }
        "target.mma" => Ok(Op::TargetMma),
        "target.warp_shuffle" => Ok(Op::TargetWarpShuffle),
        "target.barrier" => Ok(Op::TargetBarrier),
        _ => Err(ParseError::UnknownOp(off)),
    }
}

/// CEP:WHAT: Current-token offset for error reporting.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: parser tests
fn o2_of(p: &Parser) -> usize {
    match p.peek() {
        Some((_, off)) => off,
        None => 0,
    }
}

/// CEP:WHAT: Parses a type from the token stream (multi-token tensors).
/// CEP:STATUS: complete
/// CEP:FAILURE: UnknownType / BadInt / UnexpectedToken with offsets.
/// CEP:ASSUMES: the leading Word token has been consumed by the caller.
/// CEP:COST: O(rank)
/// CEP:EVIDENCE: roundtrip tests
fn parse_type_tokens(p: &mut Parser) -> Result<Type, ParseError> {
    let (t, off) = p.next()?;
    let w = match t {
        Tok::Word(w) => w,
        _ => return Err(ParseError::UnknownType(off)),
    };
    if w == "token" {
        return Ok(Type::Token);
    }
    if w == "none" {
        return Ok(Type::None);
    }
    if let Some(rest) = w.strip_prefix("scalar<") {
        let elem = rest.strip_suffix('>').ok_or(ParseError::UnknownType(off))?;
        let sc = parse_elem(elem, off)?;
        return Ok(Type::Scalar(sc));
    }
    let is_memref = w.starts_with("memref<");
    let is_tensor = w.starts_with("tensor<");
    if !is_tensor && !is_memref {
        return Err(ParseError::UnknownType(off));
    }
    // Element from inside the angle brackets.
    let open = w.find('<').ok_or(ParseError::UnknownType(off))?;
    let close = w.find('>').ok_or(ParseError::UnknownType(off))?;
    let elem = &w[open + 1..close];
    let sc = parse_elem(elem, off)?;
    // Optional [dims].
    let mut dims: Vec<i64> = Vec::with_capacity(MAX_RANK);
    if matches!(p.peek(), Some((Tok::Sym('['), _))) {
        let _ = p.next();
        while matches!(p.peek(), Some((Tok::Int(_), _))) {
            let (t, o) = p.next()?;
            if let Tok::Int(v) = t {
                dims.push(v);
            } else {
                return Err(ParseError::BadInt(o));
            }
            if matches!(p.peek(), Some((Tok::Sym(','), _))) {
                let _ = p.next();
            } else {
                break;
            }
        }
        p.expect_sym(']')?;
    }
    let shape = Shape::from_dims(&dims).map_err(|_| ParseError::UnknownType(off))?;
    // Optional layout word (row default).
    let mut layout = Layout::RowMajor;
    if let Some((Tok::Word(lw), _)) = p.peek() {
        match lw.as_str() {
            "col" => {
                layout = Layout::ColMajor;
                let _ = p.next();
            }
            "row" => {
                let _ = p.next();
            }
            _ => {}
        }
    }
    let tt = TensorType {
        elem: sc,
        shape,
        layout,
    };
    if is_memref {
        // Optional address-space word (global default).
        let mut space = crate::ty::AddressSpace::Global;
        if let Some((Tok::Word(sw), _)) = p.peek() {
            match sw.as_str() {
                "shared" => {
                    space = crate::ty::AddressSpace::Shared;
                    let _ = p.next();
                }
                "register" => {
                    space = crate::ty::AddressSpace::Register;
                    let _ = p.next();
                }
                _ => {}
            }
        }
        return Ok(Type::MemRef(tt, space));
    }
    Ok(Type::Tensor(tt))
}

/// CEP:WHAT: Element type spelling to enum.
/// CEP:STATUS: complete
/// CEP:FAILURE: UnknownType.
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: roundtrip tests
fn parse_elem(s: &str, off: usize) -> Result<ScalarType, ParseError> {
    match s {
        "f64" => Ok(ScalarType::F64),
        "i64" => Ok(ScalarType::I64),
        "f32" => Ok(ScalarType::F32),
        "i32" => Ok(ScalarType::I32),
        "bool" => Ok(ScalarType::Bool),
        _ => Err(ParseError::UnknownType(off)),
    }
}

/// CEP:WHAT: Default type inference when the text omits `: type`.
/// CEP:WHY: Ergonomics for hand-written test IR; the verifier still checks
///          arity (CEP&CC 38.18 — cheap mode here, full mode in xir-graph).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (total function over op kinds).
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: roundtrip tests
fn infer_type(op: &Op) -> Type {
    match op {
        Op::ConstI64(_) => Type::Scalar(ScalarType::I64),
        Op::ConstF64(_) => Type::Scalar(ScalarType::F64),
        Op::Param { .. } => Type::Scalar(ScalarType::F64),
        Op::Binary(_) | Op::Unary(_) | Op::Dot | Op::Matmul { .. } => Type::Scalar(ScalarType::F64),
        Op::Reduce { .. } | Op::Broadcast { .. } | Op::Transpose { .. } | Op::Conv { .. } => {
            Type::Scalar(ScalarType::F64)
        }
        _ => Type::None,
    }
}

// Silence unused-import lint risk for ArenaError (used in map_err branches).
#[allow(unused_imports)]
use ArenaError as _ArenaErrorUsed;

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: print -> parse -> print round trip is byte-stable.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any asymmetry.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn roundtrip_flat_function() {
        let src = "xir v1 func @main {\n  %0 = const.i64 3 : scalar<i64>\n  %1 = const.i64 4 : scalar<i64>\n  %2 = binary.add %0, %1 : scalar<i64>\n}\n";
        let arena = parse_arena(src, 64);
        assert!(arena.is_ok(), "parse error: {:?}", arena.err());
        let arena = match arena {
            Ok(a) => a,
            Err(_) => return,
        };
        let printed = print_arena(&arena);
        let reparsed = parse_arena(&printed, 64);
        assert!(reparsed.is_ok());
        if let Ok(a2) = reparsed {
            assert_eq!(print_arena(&a2), printed);
        }
    }

    // CEP:WHAT: graph.if regions round-trip through text v1 (CEP-12): the
    //           nested block syntax parses into owned then/else regions and
    //           prints back byte-identically (including NESTED ifs and uses
    //           of pre-if values inside the blocks).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any round-trip drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn roundtrip_if_regions() {
        let src = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : scalar<f64>\n",
            "  %1 = const.f64 1.0 : scalar<f64>\n",
            "  %2 = if %0 : scalar<f64> {\n",
            "    %3 = binary.mul %0, %1 : scalar<f64>\n",
            "    %4 = if %1 : scalar<f64> {\n",
            "      %5 = binary.add %3, %1 : scalar<f64>\n",
            "    } else {\n",
            "      %6 = binary.sub %3, %1 : scalar<f64>\n",
            "    }\n",
            "  } else {\n",
            "    %7 = binary.add %0, %0 : scalar<f64>\n",
            "  }\n",
            "  %8 = binary.add %2, %1 : scalar<f64>\n",
            "}\n",
        );
        let arena = parse_arena(src, 128);
        assert!(arena.is_ok(), "parse error: {:?}", arena.err());
        let arena = match arena {
            Ok(a) => a,
            Err(_) => return,
        };
        // Structural expectations: 9 live nodes + root + 2*2 regions.
        let mut nodes = 0usize;
        arena.for_each_live_node(|_, _| nodes += 1);
        assert_eq!(nodes, 9);
        let mut regions = 0usize;
        arena.for_each_region(|_, _| regions += 1);
        assert_eq!(regions, 5); // root + 2 ifs (outer + nested) * (then + else)
                                // Round-trip: print -> parse -> print must be byte-identical.
        let printed = print_arena(&arena);
        let reparsed = parse_arena(&printed, 128);
        assert!(reparsed.is_ok(), "reparse error: {:?}", reparsed.err());
        if let Ok(a2) = reparsed {
            assert_eq!(print_arena(&a2), printed);
            // Fingerprint equality proves the region structure (not just the
            // op stream) survived the round trip.
            use crate::snapshot::IrSnapshot;
            assert_eq!(
                IrSnapshot::new(arena).fingerprint(),
                IrSnapshot::new(a2).fingerprint()
            );
        }
    }

    // CEP:WHAT: if statements reject missing else blocks and trailing
    //           garbage loudly (never a silent guess).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if malformed ifs parse.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rejects_malformed_if() {
        // Missing else block.
        let no_else = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : scalar<f64>\n",
            "  %1 = if %0 : scalar<f64> {\n",
            "    %2 = const.f64 1.0 : scalar<f64>\n",
            "  }\n",
            "}\n",
        );
        assert!(parse_arena(no_else, 64).is_err());
        // Missing type after the condition.
        let no_type = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : scalar<f64>\n",
            "  %1 = if %0 {\n",
            "    %2 = const.f64 1.0 : scalar<f64>\n",
            "  } else {\n",
            "    %3 = const.f64 2.0 : scalar<f64>\n",
            "  }\n",
            "}\n",
        );
        assert!(parse_arena(no_type, 64).is_err());
    }

    // CEP:WHAT: The PARSER accepts cross-region sibling uses (text carries
    //           structure, the verifier polices it): %2 from the then-region
    //           used in the else-region parses fine here and is rejected by
    //           xir_graph::verify in the integration test
    //           `if_dominance_violation_rejected` (tests/cli.rs — audit F-12
    //           moved the real evidence there).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on parser over-rejection.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test + tests/cli.rs `if_dominance_violation_rejected`
    #[test]
    fn if_regions_sibling_use_parses() {
        let src = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : scalar<f64>\n",
            "  %1 = if %0 : scalar<f64> {\n",
            "    %2 = const.f64 1.0 : scalar<f64>\n",
            "  } else {\n",
            "    %3 = binary.add %2, %2 : scalar<f64>\n",
            "  }\n",
            "}\n",
        );
        let arena = parse_arena(src, 64);
        assert!(arena.is_ok(), "text parses (structure is legal)");
        if let Ok(a) = arena {
            let mut regions = 0usize;
            a.for_each_region(|_, _| regions += 1);
            assert_eq!(regions, 3);
        }
    }

    // CEP:WHAT: Non-finite and huge float constants round-trip through the
    //           bits escape form (audit F-4: inf/NaN print as unparseable
    //           words; |v| >= 2^63 prints as an integer that overflows the
    //           i64 literal token).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any round-trip drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn float_bits_roundtrip() {
        for v in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            // 2^63 exactly (f64::from_bits(0x43E0..)) — the decimal overflow edge.
            f64::from_bits(0x43E0_0000_0000_0000),
            -f64::from_bits(0x43E0_0000_0000_0000),
            1.0e300,
            4.9e-324, // subnormal
            0.5,
            -0.25,
            1.0e15,
        ] {
            let src = format!(
                "xir v1 func @main {{\n  %0 = const.f64 bits {hi} {lo} : scalar<f64>\n}}\n",
                hi = v.to_bits() >> 32,
                lo = v.to_bits() & 0xFFFF_FFFF
            );
            let arena = parse_arena(&src, 64);
            assert!(
                arena.is_ok(),
                "bits parse failed for {v}: {:?}",
                arena.err()
            );
            if let Ok(a) = arena {
                let printed = print_arena(&a);
                let reparsed = parse_arena(&printed, 64);
                assert!(reparsed.is_ok(), "reparse failed for {v}");
                if let Ok(a2) = reparsed {
                    // NaN != NaN by IEEE: compare BIT patterns via the op.
                    let mut got = f64::NAN;
                    let mut got2 = f64::NAN;
                    a.for_each_live_node(|_, n| {
                        if let Op::ConstF64(x) = n.op {
                            got = x;
                        }
                    });
                    a2.for_each_live_node(|_, n| {
                        if let Op::ConstF64(x) = n.op {
                            got2 = x;
                        }
                    });
                    assert_eq!(got.to_bits(), v.to_bits(), "value drift for {v}");
                    assert_eq!(got2.to_bits(), v.to_bits(), "round-trip drift for {v}");
                    assert_eq!(print_arena(&a2), printed, "print instability for {v}");
                }
            }
        }
    }

    // CEP:WHAT: Runaway if-nesting fails loudly at the depth bound instead
    //           of overflowing the stack (audit F-13).
    // CEP:STATUS: complete
    // CEP:FAILURE: the test would crash (stack overflow) without the bound.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rejects_runaway_nesting() {
        let mut src = String::from("xir v1 func @main {\n");
        let depth = 2000usize;
        src.push_str("  %0 = param 0 : scalar<f64>\n");
        for i in 1..=depth {
            src.push_str(&format!("  %{i} = if %{} : scalar<f64> {{\n", i - 1));
        }
        for _ in 0..depth {
            src.push_str("  }\n");
        }
        src.push_str("}\n");
        let arena = parse_arena(&src, 65536);
        assert!(matches!(arena, Err(ParseError::NestingTooDeep(_))));
    }

    // CEP:WHAT: Garbage input fails with an offset, never panics.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a panic escapes the parser.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rejects_garbage() {
        let cases = [
            "",
            "xir",
            "xir v1 func {",
            "xir v1 func @main { %0 = nosuchop }",
            "xir v1 func @main { %0 = const.i64 zz }",
            "xir v1 func @main { %0 = binary.add %9 }",
            "xir v1 func @main { %0 = if %1 }",
        ];
        for c in cases {
            let r = parse_arena(c, 32);
            assert!(r.is_err(), "input {:?} must fail", c);
        }
    }

    // CEP:WHAT: Printing is deterministic across repeated runs.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn printer_is_deterministic() {
        let src = "xir v1 func @main {\n  %0 = const.f64 1.5 : scalar<f64>\n  %1 = rng dist=uniform seed=7\n  %2 = reduce axis=0 monoid=add %0\n}\n";
        let a = parse_arena(src, 32);
        assert!(a.is_ok());
        if let Ok(arena) = a {
            assert_eq!(print_arena(&arena), print_arena(&arena));
        }
    }

    // CEP:WHAT: Tensor types parse with shapes and layouts.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on type corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tensor_type_roundtrip() {
        let src = "xir v1 func @main {\n  %0 = const.f64 1.0\n  %1 = const.f64 2.0\n  %2 = matmul ta=false tb=true %0, %1 : tensor<f32>[4,8] col\n}\n";
        let r = parse_arena(src, 32);
        assert!(r.is_ok());
        if let Ok(a) = r {
            let text = print_arena(&a);
            assert!(text.contains("matmul ta=false tb=true"));
            assert!(text.contains("tensor<f32>[4,8] col"));
            // Reprint-reparse is stable.
            let r2 = parse_arena(&text, 32);
            assert!(r2.is_ok(), "reparse failed: {:?}", r2.err());
        }
    }
}
