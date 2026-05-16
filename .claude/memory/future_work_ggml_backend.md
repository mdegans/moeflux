# Future work — a GGML-backed Backend (oracle + benchmark)

Mike's idea, 2026-05-16. Deferred deliberately — written down so it
isn't lost.

## The idea

Implement moeflux's `Backend` trait with a third backend backed by
GGML / llama.cpp (`llama-cpp-sys` as a dependency). The graph-mode
`Backend` trait + `Op` enum is exactly the seam that makes this
possible without disturbing the rest of the code.

Two payoffs in one:
- **Standing per-Op benchmark** — run moeflux's `Op`s on GGML-Metal
  and on moeflux's MetalBackend with identical inputs; the per-Op
  ratio is a direct, apples-on-the-same-backend "how far behind"
  number.
- **A 4th independent oracle** — GGML is a battle-tested reference,
  alongside `CpuBackend` (`graph_diff_oracle`) and `mlx_regression`.

## Why it's deferred (not this arc)

1. `test-backend-ops perf` already gives the GGML-Metal `MUL_MAT`
   throughput — the matmul is the dominant prefill cost, so the
   signal we need *now* is already obtainable cheaply.
2. The oracle role is not actually vacant — `CpuBackend` +
   `mlx_regression` already cover it (the dead C oracle's job is
   done). A GGML oracle would be a *4th* check, not a missing one.
3. **Op-equivalence is not 1:1, and it's weakest where it'd matter
   most.** moeflux's `Op` enum is model-driven and fused —
   `SwigluFusedBatched`, `MoeCombine`, `GatedDeltaNetStep`,
   `SdpaCausalTiled`. GGML's op set is primitive (`MUL_MAT`,
   `RMS_NORM`, `SOFT_MAX`, `FLASH_ATTN_EXT`, `SSM_*`...). Each fused
   moeflux Op becomes a hand-built GGML sub-graph; the 4-bit
   dequant-matmul is not bit-comparable (MLX-4bit vs Q4_K). So the
   backend's hardest work buys the least clean comparison; the ops
   it would nail (primitives) are the ones `test-backend-ops`
   already covers.
4. Implementing the full trait — associated types (`Pool`,
   `EncodeCtx`, `Handle`), ~15 `encode_op` arms, pool, graph
   execution — against GGML's C API is a multi-session arc, a
   detour from the kernel-optimization work.

## When to revisit

After the first kernel rewrites land. By then we'll know which `Op`s
actually warrant a standing benchmark/oracle harness, and whether
the per-Op comparison is worth the build. If yes, it slots in as a
new `Backend` impl with no churn to existing code.
