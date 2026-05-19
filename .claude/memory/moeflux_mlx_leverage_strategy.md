# moeflux-mlx — leverage Apple's MLX kernels, vendor not link

Strategic decision (2026-05-19, Mike-raised). Where a kernel has an
MLX equivalent, prefer vendoring/adapting MLX's over hand-rolling.
Track record: the biggest GPU wins come from Apple primitives
(`simdgroup_matrix`; the CoreML/NPU roadmap). `moeflux-mlx` was
created exactly for this — it replaced a hand-rolled ~5%-of-peak
matvec with MLX's properly-tiled `qmm_t`.

## Rules

- **Vendor, don't link libmlx.** MLX kernels are Metal `.metal` +
  device-side `.h` templates. Copy them into
  `crates/moeflux-mlx/shaders/`, then `#include` or dispatch them.
  Linking the MLX C++ runtime would be a huge dependency for code we
  can splice header-only.
- **`moeflux-mlx` is the home.** Workspace crate, thin Rust dispatch
  wrappers, own diff-test harness (`qmm_diff`, `gather_qmm_diff`,
  `compile`). New vendored subtrees go there.
- **Attribution discipline.** `NOTICE` pins the upstream MLX commit;
  every local edit to a vendored file is marked with a `moeflux-mlx:`
  comment; upstream copyright headers kept verbatim. MLX is MIT.
- MLX source is checked out at `~/Projects/mlx` for reference.

## Two integration modes (you cannot call a kernel from a kernel)

1. **Dispatch a whole MLX kernel as an op** — e.g. `affine_qmm_t`
   via the `QmmKernels` wrapper.
2. **`#include` MLX device-side template headers**, build your own
   `kernel void` on them — e.g. `MMATile` from `steel/gemm/mma.h`.
   This is how you reuse at sub-kernel granularity; Metal has no
   kernel-to-kernel call.

## Already vendored

- `steel/gemm/{mma,loader,transforms}.h` — `MMATile` register-blocked
  GEMM over `simdgroup_matrix`, plus `load_safe`/`store_safe`
  bounds-checked tile loaders. **Build new GEMM kernels on `MMATile`,
  not raw `simdgroup_matrix` intrinsics.**
- `qmm.metal` — `affine_qmm_t` (quantized GEMM) + `affine_gather_qmm
  _rhs` (the gathered MoE quantized GEMM — session-12 MoE work may be
  mostly *wiring it in*).

## Not yet vendored, wanted

- `steel/attn/` — MLX's flash-attention. Candidate for the session-11
  SDPA rewrite (Path B). See `kernel_arc_session11_plan.md`.
