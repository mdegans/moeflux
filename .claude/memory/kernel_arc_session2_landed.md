# Kernel arc — session 2 landed (profiling + focus list)

2026-05-16. Continues `kernel_arc_session1_landed.md`. Executed the
profiling/benchmarking plan (`virtual-petting-pelican.md`, supersedes
`sunny-stirring-grove.md`) Phases 1–4. All measurement; the kernel
rewrite is the *next* session.

## Landed (moeflux main)

- `1911430` Phase 1 — per-phase prefill instrumentation. Graph path
  routed through labeled commits (`Backend::submit_and_wait`/`execute`
  gained `label: &'static str`); `RsCtx::cmdbuf_stats` /
  `reset_cmdbuf_stats` accessors; `tests/prefill_profile.rs`.
- `fab529f` Phase 2 — `tests/kernel_bench.rs` hot-kernel microbench.

## THE RESULT — SDPA is the prefill bottleneck

**`batched_sdpa_causal_tiled` (the full-attn SDPA kernel) is the
focus.** Everything points at it:

Phase 1 — per-phase prefill breakdown (a3b, `tests/prefill_profile.rs`):

| phase                          | M=1536 | M=8192 (real chunk) | scaling   |
|--------------------------------|--------|---------------------|-----------|
| `batched_sdpa_causal_tiled`    |  8.8%  | **43.7%** (62.8 s)  | ~30× — O(M²) |
| `graph_linear_attn`            | 32.2%  | 25.8%               | ~5× linear |
| `graph_moe`                    | 18.2%  | 15.4%               | ~5× linear |
| `batched_shared_ffn_moe_combine`| 6.4%  |  5.3%               | ~5× linear |
| norms (`rms_norm_qkv`,`oproj`) |  ~5%   |  ~5%                | linear    |
| unaccounted (setup/lm_head)    | ~29%   |  5.3% (fixed ~7 s)  | constant  |

SDPA is the **only superlinear phase** — causal attention is O(M²)
within a chunk. The M-sweep is what surfaced it: at M=1536 SDPA is a
minor 8.8%; at the production `BATCHED_CHUNK_SIZE=8192` chunk it is
43.7%. Benching only at M=1536 would have mis-fingered the linear-attn
graph.

Phase 2 — microbench (`tests/kernel_bench.rs`, n=3, <1% spread):
- SDPA M=8192/kv=8192: 6273 ms, **87.6 GFLOP/s** (= the Phase-1
  6285 ms/commit — clean cross-check).
- SDPA M=8192/kv=32768: 36133 ms, 106.5 GFLOP/s.
- 4-bit matvec: 510–800 GFLOP/s, flat across M.

Phase 3 — llama.cpp `test-backend-ops perf` on Metal (M2 Max):
- FLASH_ATTN_EXT prefill case (`nh=16 kv=5776 nb=5776`): **4.61 TFLOPS**.
- MUL_MAT only tests narrow batch (n≤8): q4_K n=8 ≈ 1.8 TFLOPS — not
  a clean wide-prefill matvec baseline; matvec gap left imprecise.

## The gap

- moeflux SDPA prefill: **~0.088 TFLOP/s** (~0.7% of M2 Max's
  ~13.6 TFLOP/s fp32 peak).
- llama.cpp FlashAttention prefill: **~4.6 TFLOP/s** (~34% of peak),
  and it hits that **without tensor units** (`has tensor = false`
  pre-M5 — pure simdgroup ops). 4.6 TFLOP/s is a reachable target.
- **SDPA gap ≈ 52×.** This single kernel is most of moeflux's ~20×
  prefill gap vs llama.cpp.

## Why the kernel is slow (root cause — read of `gpu_attn.rs:451`)

`encode_sdpa_causal_tiled` dispatches `n_tokens * num_heads`
threadgroups of 128 threads — **one query per threadgroup**. KV is
tiled at 4096 but every query-threadgroup re-reads the entire K and V
from global memory → **M-fold redundant K/V traffic**. Scalar dot
products; no Q-tiling; no threadgroup-memory K/V staging; no
simdgroup-matrix use. A proper FlashAttention tiles *both* Q and KV so
a K/V tile staged in threadgroup memory serves a whole Q tile.

## Focus list (feeds the kernel-rewrite plan)

1. **`batched_sdpa_causal_tiled` — rewrite as a real tiled
   FlashAttention.** Dominant (43.7% @ prod chunk), superlinear,
   ~52× behind llama.cpp. This is the next session's work and
   warrants a design conversation + plan-mode pass (it's a new Metal
   kernel + Rust encoder + cosine-1.0 diff test vs the current
   kernel, not a small tweak).
2. **4-bit dequant matvec** — secondary. ~4–5% of peak, linear,
   spread across the graph buckets. Worth a pass *after* SDPA.
3. **bf16 matmul** — not on a labeled hot path. Deferred.

## Notes / leftovers

- `~/Projects/llama-cpp-sys/external/llama.cpp/build` was wiped and
  reconfigured fresh — the old CMakeCache pinned a removed SDK
  (`MacOSX26.4.sdk`); a macOS update broke it. `test-backend-ops` now
  built at `build/bin/test-backend-ops`. The Metal backend is named
  `MTL0`, not `Metal`, for `-b`.
- Pre-existing `unused_mut` warnings in test-target builds
  (`encoder.rs:223` cfg(test) block, batched_diff_oracle) — not
  introduced this arc; lib build is zero-warning.
