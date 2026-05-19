# Kernel arc — session 11 landed (enabling work; Phase 1 still ahead)

2026-05-19. Session 11 opened on the SDPA → `simdgroup_matrix` plan
(`kernel_arc_session11_plan.md`) but the work that landed is the
*enabling* refactor — the QK^T → `MMATile` rewrite itself (plan
"Phase 1") has **not** happened yet. Next session continues there.

## What landed (5 commits on moeflux main)

- `401960b` — corrected the plan's Path B claim. Path B (vendor MLX
  `steel/attn`) was recorded as dead-for-our-shape on a bad "~48 KB
  threadgroup" figure. Verified: `BD=256` fits a sub-32 KB config
  (`BQ=16/BK=32` = 28.25 KB; M2 Max cap confirmed 32768 B). Path A
  still wins — but on "untuned tile shape / unverified register
  pressure" grounds, not a hard cap. Path B is a real `BD=256`
  fallback if Path A underperforms.
- `4743778` — renamed crate `moeflux-mlx` → `moeflux-metal` (it now
  hosts moeflux's own kernels too, not just vendored MLX).
- `e0a0bde` — unified `QmmKernels` → `Kernels` + a `KernelCall` trait.
  One `Kernels` struct owns the library + every pipeline; dispatch is
  the uniform generic `kernels.encode(cmdbuf, &SomeCall { .. })`. Each
  kernel = one `Call` struct + one `impl KernelCall`. No `encode_*`
  method zoo.
- `0458c8a` — migrated `attn_sdpa_causal_flash` into `moeflux-metal`:
  `crates/moeflux-metal/shaders/sdpa.metal` joins the assembled TU
  **after the steel headers**, so the Phase-1 rewrite can build on
  `mlx::steel::MMATile`. `SdpaCall` + `impl KernelCall` replace the
  14-arg `encode_sdpa_causal_flash` (retired one
  `#[allow(clippy::too_many_arguments)]`). Kernel body byte-identical;
  5 `sdpa_causal_flash_*` diff tests cosine-1.0 vs `sdpa_cpu`.

Phase 0 (Xcode GPU capture) — **skipped** per Mike: a prior capture
produced unusable GB of data. The per-op `profile_8192` after the
Phase-1 change is the signal instead.

## Next session — Phase 1: QK^T → steel `MMATile`

Now happens IN `crates/moeflux-metal/shaders/sdpa.metal` (Phase 2 of
the kernel, the `simd_sum`-per-dot QK^T loop). Design already worked
out (see `kernel_arc_session11_plan.md` + the session-11 thread):

- Partition: `WM=8` simdgroups over the 64 query rows (8 each → 1
  frag-row/simdgroup), `WN=1`. Per-simdgroup register tiles then match
  today's footprint exactly: Q = `MMATile<float,1,32>` (= `qreg[8][8]`,
  64 fl/lane), O = `MMATile<float,1,32>` (= `acc[8][8]`).
- Q: `load_safe` once before the KV loop, clamped to `br_valid` —
  dissolves the ragged-tile OOB, no producer pad.
- Per KV block: scores `C = MMATile<float,1,2>`; stream K transposed
  from `kv_stage` 8-ktile at a time, `tile_matmad` into C. K=256 = 32
  exact ktiles. Then `MMATile::store` C → tg `scores`, and the
  **existing** scalar scale+mask + Phase 3/4/5 run unchanged.
- The `simd_sum` per dot is what disappears.
- Build wiring already done: steel `mma.h` is in the same TU as
  `sdpa.metal`. No build change needed for Phase 1.

Deferred: Phase 2 (P·V → `MMATile`, with the per-row `corr` rescale —
the cleaner element-wise option, scaling each fragment's `vec<float,2>`
by `corr[row]` via `get_coord`, is preferred over the plan's 8-diagonal
`simdgroup_multiply`). Phase 3 (GQA-fold) — session 12.

Verification: the `flash_diff_tokenwise` battery (`batched_diff_oracle`,
5 `sdpa_causal_flash_*` tests, cosine ≥ 0.9999 vs `sdpa_cpu`), then
`profile_8192` (expect SDPA's % to drop) + post-reboot `bench.py`.
