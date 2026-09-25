# Target Configuration

Reference target for all calibrated constants (CEP&CC 7.2, Law 7):

| Item | Value | Source |
|------|-------|--------|
| Architecture | x86_64 | CI runner |
| Cache line floor | 64 B | x86_64/aarch64 minimum; `anvil::config::CACHE_LINE_BYTES` |
| Shared-memory budget | 64 KiB | `fusion::resource::SHARED_MEM_BUDGET_BYTES` |
| Register budget | 256 units | `fusion::resource::REGISTER_BUDGET_UNITS` |
| Tile edge | 64 elements | `codegen::tile::TILE_CACHE_LINE` (8 f64/line * 8 lines) |
| Vector width | 4 elements | `codegen::vectorize` (quarter cache line) |
| Chase-Lev capacity | 8192 tasks/worker | `anvil::config::DEQUE_CAPACITY` |
| Workers | logical cores, cap 64 | `anvil::config::MAX_WORKERS` |
| FP model | IEEE 754 binary64; no fast-math; reassociation banned for floats by default | CEP&CC 38.24 |
| Endianness | little (hashes serialize LE explicitly) | `xir-core::hash` |
| Deterministic builds | no time/locale/hash-seed inputs; overflow checks ON in release | Cargo.toml |

GPU targets (mma/warp_shuffle as real instructions) are the CEP-16
placeholder; the CPU lowering keeps them as schedule hints with loud
`UnsupportedInstr` semantics where execution would be required.
