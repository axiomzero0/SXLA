# Fusion Legality: Implemented Subset and Proof Obligations

The full engine is Presburger/ISL constraint solving (CEP-19). The
implemented subset (crates/fusion/src/legality.rs) is:

1. **Effect gating**: impure ops never fuse (token-threaded ordering).
2. **Barrier gating**: fusion.barrier / fusion.materialize are hard fences.
3. **Broadcast compatibility** (elementwise): per-dim equal-or-1 (the NumPy
   rule) — the affine-compatible subset of index-map equality.
4. **Reduction legality**: innermost-axis reductions fuse; non-innermost
   axes require the split-reduction strategy (Universe B inserts partials),
   so plain fusion is rejected.
5. **Layout pricing**: mismatched layouts price a conversion term
   (`layout_conversion_cost`) rather than silently accepting.

Rejection is the default: `can_fuse` returns `Err` naming the failed check;
the search never transforms on unproven legality (CEP&CC 38.22).
