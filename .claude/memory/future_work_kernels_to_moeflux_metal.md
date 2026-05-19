# Future work — relocate RIIR-backend kernels into the `moeflux-metal` crate

Mike, session 14: the RIIR-backend Metal kernels should live in the
`moeflux-metal` crate, not in `crates/moeflux/shaders/shaders.metal`.

## Current state

Two kernel homes:

- `crates/moeflux/shaders/shaders.metal` — the monolithic RIIR-backend
  kernel library, `include_str!`'d by `backend/gpu/metal.rs`, lazy-
  compiled by name via `MetalContext::pipeline(name)` against
  `ALL_KERNELS`. ~40 kernels: matvec/dequant family, the `*_n_tokens`
  batched family (`residual_add_n_tokens`, `rope_n_tokens`,
  `rms_norm_bf16_fused_n_tokens`, …), MoE, conv1d, etc.
- `crates/moeflux-metal/shaders/` — separately-compiled specialist
  kernels (`qmm.metal`, `sdpa.metal`, the `steel` GEMM) with their
  own pipeline API.

## Intent

Move the `shaders.metal` kernels into `moeflux-metal` so there is one
kernel home. The existing `shaders.metal` set moves as one batch.

## Going forward — Mike's call (session 14)

**New kernels go into `moeflux-metal` as they are written** — that
shrinks the eventual migration instead of growing it. Mike chose
this over my "keep new ones with their family in `shaders.metal`,
move all at once" argument: the two-home transition cost is minor
and shrinking, and fewer-to-move wins.

Practical heads-up for the next author: `moeflux-metal` kernels use
a **different integration path** than `shaders.metal` — separate
compilation + `moeflux-metal`'s own pipeline API, *not* `ALL_KERNELS`
+ `MetalContext::pipeline(name)`. Adding a kernel there is not a
drop-in like `rope_n_tokens` was; budget for wiring it through that
API. From Phase-1 item 2 on (sigmoid-gate, Q-split, embedding
gather), new prefill-arc kernels target `moeflux-metal`.

`rope_n_tokens` (prefill arc Phase 1, commit `82e4a52`) went into
`shaders.metal` and moves with the existing-set bulk migration.
