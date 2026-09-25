# Versioned Pass Pipelines (CEP&CC 38.20)

Changing pass order is a semantic event requiring review, benchmark evidence,
golden IR update, disassembly review and a regression test.

## sxla-tier1-2026-09

```
verify(entry) -> canonicalize-l0 (const-fold -> GVN -> DCE) -> verify -> structurize -> lower-target
```

Budget: 5 ms on the reference machine (docs/targets.md). The canonicalize
pass runs as one transactional clone-mutate-publish unit; verification gates
publication.

## sxla-tier2-2026-09

```
verify(entry) -> canonicalize-l0 -> egraph-saturate (analysis; ReadOnly) ->
fusion-search (universes A/B/C, atomic pruning) -> verify -> structurize -> lower-target
```

The fusion search is a scheduling transformation; semantic equivalence is
enforced by the differential test `differential_fusion_equivalence`
(CEP&CC 38.45).

## sxla-tier0 (fallback)

```
verify(entry) -> structurize -> lower-target -> interpret
```

Correctness path used while compilation is pending and the semantic oracle
for differential testing.

## Pass certification index

Every pass carries its full `CEP:HPC-PASS-*` block at its implementation:

- `canonicalize-l0` — crates/xir-levels/src/passman.rs
- `const-fold-l0` — crates/xir-graph/src/fold.rs
- `gvn` — crates/xir-graph/src/gvn.rs
- `dce` — crates/xir-graph/src/dce.rs
- `layout-infer` — crates/xir-levels/src/level1.rs
- `egraph-saturate` — crates/egraph/src/saturate.rs (driver fields)
- `fusion-search` — crates/fusion/src/search.rs
