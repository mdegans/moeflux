---
name: future-work-mlx-steel-attention
description: MLX's steel/attn/kernels/steel_attention.h is a real batched-prefill FlashAttention reference using simdgroup MMA, fast::exp2 softmax, configurable WM×WN tiling. Worth a dedicated exploration session as an alternative path to closing the SDPA performance gap if our v2 simdgroup-MMA port's diagnose arc doesn't pan out.
metadata:
  type: project
---

# Context

Our v2 simdgroup-MMA port (commits 92266e6 + follow-ups) is
correct (diff oracle cosine 1.0) but engine-level ~10% slower
than v1 scalar FMA path. See [[prefill-sdpa-v2-diagnose-arc]]
for the diagnose state.

Mike, 2026-05-21, raised `sdpa_vector.h` in `~/Projects/mlx`
as a potential reference. On inspection: `sdpa_vector.h` is
the wrong target — it's the **decode/single-query** kernel
(BN=32, BD=32, one query per TG, scalar `simd_sum` dot
products). Same shape as our v1 just without GQA folding.
Won't help.

The right MLX reference is
**`mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`**
(476 lines), supported by `attn.h` (295), `mma.h` (750),
`nax.h` (887, M3+ NAX path), `loader.h` (264), `params.h`,
`transforms.h`. Together: real batched FlashAttention,
templated, with simdgroup MMA throughout.

# Things to learn from / port

1. **`fast::exp2` instead of `exp`** in the softmax — Apple
   Metal `fast::exp2` is ~3-5× faster than `exp` per Apple
   docs. Our v2 (and v1) use `exp`. This alone might be a
   meaningful win and is a low-risk standalone change worth
   trying BEFORE a full steel_attention port.
   - Note: `exp2(x) = e^(x * ln2)`. Mathematically: `exp(s)
     = exp2(s * (1/ln2)) = exp2(s * 1.4426950408889634)`.
     The softmax scale absorbs the constant: pre-multiply
     scores by `1/ln2` and use `exp2` directly.
   - llama.cpp's flash-attn (`ggml-metal.metal:2160`) does
     this too — see lines around the softmax computation.

2. **MMATile abstraction** (`steel/gemm/mma.h`, already
   vendored — we just don't use it). MMATile wraps
   `simdgroup_multiply_accumulate` with explicit tile
   geometry control. Could replace our raw simdgroup matrix
   array allocations. Per the Plan agent's flagged trouble
   spot, MMATile has `STEEL_PRAGMA_UNROLL` controls that
   our raw kernel doesn't — might solve Suspect 2 (Metal
   compiler aggressive unroll → register spill).

3. **Configurable WM × WN tiling.** MLX parameterizes the
   simdgroup layout (warps in M and N dimensions). We
   hardcode FA_SIMDS=8 in M direction. Trying different
   WM/WN combinations could find a better register-pressure
   sweet spot.

4. **AttnParams struct** — they pass attention parameters
   via a struct buffer rather than individual scalars. Tidier
   and slightly fewer encoder calls.

5. **NAX path** (`nax.h`) — uses Apple's matrix coprocessor
   on M3+/M4 chips. We're on M2 Max so this doesn't apply,
   but worth knowing about for the M5 path.

# Exploration scope for the future session

- Read `steel_attention.h` + `attn.h` + `mma.h` end-to-end
  (1500 lines, comparable to our sdpa.metal+kernel_bench)
- Identify the highest-leverage delta vs our v2:
  - If it's `fast::exp2`: standalone PR, low risk, measure
  - If it's MMATile structure: port v2 to use MMATile,
    re-measure
  - If it's WM×WN tiling: experiment with our existing
    kernel's `FA_SIMDS` constant
- Decide: incremental hardening of our v2 vs full port to
  MLX-style structure

# Don't merge speculatively

Per [[feedback-vendor-recommended-lever-priority]]: this is
a vendor-recommended path (MLX = Apple), but the deviation
cost matters. The best move is **measure** which specific
delta closes our gap. Don't blindly port the whole thing.

# Cross-references

- [[prefill-sdpa-v2-diagnose-arc]] — why we need a different
  approach
- [[llama-cpp-moe-differentiators]] — superseded MoE framing
  (kept for history)
- [[prefill-sdpa-dominant-finding]] — the trace that
  motivated the v2 arc; the SDPA kernel is still the gap
- Files in `~/Projects/mlx/mlx/backend/metal/kernels/steel/`:
  - `attn/kernels/steel_attention.h:1-476` — main batched
    attention kernel
  - `attn/mma.h:1-750` — attention-specific MMA helpers
  - `gemm/mma.h` — MMATile abstraction (already vendored in
    `moeflux-metal/shaders/steel/gemm/mma.h`)
