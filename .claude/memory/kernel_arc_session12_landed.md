# Kernel arc — session 12 landed (SDPA Phase 1: QK^T → steel MMATile)

2026-05-19. Executed "Phase 1" of the SDPA simdgroup-matrix rewrite
(`kernel_arc_session11_plan.md` / `_landed.md`). Phase 1 landed and
verified; Phase 2 (P·V) was validated but deliberately **not** built —
see below.

## What landed (commit `bfa1be5` on moeflux main)

`attn_sdpa_causal_flash` Phase 2 (the QK^T loop) is now a steel
`MMATile` simdgroup-matrix GEMM, in `crates/moeflux-metal/shaders/sdpa.metal`:

- Q is staged once into a register-resident `MMATile<float,1,32>` `A`;
  K is streamed transposed from `kv_stage` one 8-deep ktile at a time
  (`str_x=1, str_y=FA_HD`); fragments multiply-accumulate via
  `BaseMMAFrag::mma`. The per-(row,c) scalar `fma` dot + 32-lane
  `simd_sum` reduce is gone.
- Scale + causal mask are fused into the score frags **in-fragment**
  before `C.store`, so no extra barrier.
- Phases 3/4/5 + finalize untouched.

Verified: all 5 `sdpa_causal_flash_*` diff tests **cosine = 1.0** vs
`sdpa_cpu` (incl. `m1500_deep_chunk`, the ragged-tile case);
`graph_diff_oracle` canary 11/11. Stale `FA_BR=32` comment in
`batched_diff_oracle.rs` fixed → 64.

**Perf: not yet benched.** Phase 1 removes a real cross-lane reduction
so it should be a genuine win, but a clean post-reboot `bench.py` /
`prefill_profile` A/B is the session-13 first task (Mike was on mobile;
no reboot possible this session). Expected: SDPA's 33%-of-wall share
drops.

## Discovery — steel `load_safe` is buggy (report upstream)

The session-11 plan's "Design Decision #1" assumed
`MMATile::load_safe` clamps the ragged Q tile. **It cannot**, on two
counts, found while validating against `mma.h`:

1. `BaseMMAFrag::load_safe` (`steel/gemm/mma.h:92`) computes the
   address as `src[(off_x+i)*str_x + (off_x+j)*str_y]` — `off_x` is
   used in the column term where `off_y` belongs (the bounds check on
   `:90` uses `off_y` correctly). Any tile with >1 column-frag reads
   garbage. **Confirmed present in current upstream MLX** (`~/Projects/mlx`,
   pinned commit `7b7c12407f...`, 2026-05-15). `store_safe` (`:138`)
   and `store_slice` (`:174`) two lines down are *correct* — so it is
   an isolated upstream typo, a clean one-line fix. **Filed upstream:
   `ml-explore/mlx` PR #3565** (`(off_x + j)` → `(off_y + j)`),
   discovery credited to Claude Opus 4.7, 2026-05-19.
2. Even bug-free, `load_safe`'s bounds check is at tile-frag
   granularity and the per-lane row is folded into the *pointer* via
   `get_coord` — so it cannot clamp individual ragged rows anyway.

Phase 1's actual ragged fix: stage Q through `kv_stage` in
`FA_BR/FA_BC` = 4 one-time rounds before the KV loop; simdgroup `g`
loads its rows in round `g/2` with the ragged guard. `shaders.metal`-
only, no producer/test contract change.

## Phase 2 (P·V) — validated, NOT built, deferred

Per the session decision ("Phase 1, then Phase 2 if clean"), the P·V
encoding was validated by a Plan agent (full spec in
`kernel_arc_session13_plan.md`). It is implementable at cosine=1.0.
**But it was not built**, because the validation's honest read — and
the session-11 plan's own caveat — is that it is a *uniformity
refactor, not a perf win*:

- QK^T's win was structural: it deleted a cross-lane `simd_sum`.
- P·V has **no reduction to delete** — scalar Phase 5 is already
  embarrassingly parallel (each lane owns its `acc[ri][k]`, zero
  cross-lane traffic). The only lever is moving MACs onto the tensor
  ALU.
- The contraction is `FA_BC=16` = **2 ktiles** — too short to
  amortize `mma` issue overhead. P·V is the wrong shape for
  simdgroup matrices (wide N=256, tiny K=16).
- The finalize gets materially more complex: a fragment-layout `O`
  can't be scalar-written, forcing a `kv_stage` round-trip.

Building an unverifiable (no reboot/bench this session) marginal
refactor on main was judged not worth it. The spec is saved; session
13 can execute it in ~20 min *if chosen*.

## Next session (session 13)

Plan-of-record in `kernel_arc_session13_plan.md`:
1. Post-reboot bench Phase 1 (the deferred measurement).
2. **GQA-fold** — the agent's recommended bigger lever: one
   threadgroup per `(q_tile, kv_h)`, K/V block staged once and reused
   across the `heads_per_kv` query-heads sharing it → K/V global
   traffic cut by `heads_per_kv`×. A real bandwidth win, vs. P·V's
   marginal ALU reassignment.
3. P·V → `MMATile` — optional uniformity follow-up; spec is ready.
