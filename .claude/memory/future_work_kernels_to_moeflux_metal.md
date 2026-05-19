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
kernel home. **Do it as one coherent batch**, not piecemeal — moving
single kernels split-brains the set across two crates + two
integration paths.

## Until then

New kernels (e.g. the remaining Phase-1 prefill-arc kernels:
sigmoid-gate, Q-split, embedding gather) go into `shaders.metal`
alongside their family, so the eventual bulk move relocates a
consistent set. Don't start the split early.

`rope_n_tokens` (prefill arc Phase 1, commit `82e4a52`) is the most
recent addition to `shaders.metal` and will move with the rest.
