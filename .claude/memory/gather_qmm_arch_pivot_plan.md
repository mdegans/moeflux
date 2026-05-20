---
name: gather-qmm-arch-pivot-plan
description: DEAD 2026-05-20 — empirical microbench + real htpe distribution showed this plan's per-expert dense dispatch loses 12% on the MoE matmul stage using our `affine_qmm_t` kernel. Superseded by `gather_qmm_pivot_dead.md` (data) + `llama_cpp_moe_differentiators.md` (what to attack instead). Kept as a pre-empirical reference for the analytical reasoning that didn't survive contact with the data.
metadata:
  type: project
---

# DEAD — see `gather_qmm_pivot_dead.md`

This plan was killed the same day it was drafted by three empirical
measurements:

1. **Small-M `affine_qmm_t` sweep** showed dense crosses gather at
   M≈700 for gate/up and M≈400 for down. The plan's "M=512 → 7740
   GFLOP/s threshold" was a single-point heuristic that didn't
   account for the cross-over shape.

2. **Real htpe distribution from a3b prefill** revealed the plan's
   mean-tokens-per-expert assumption was wrong by 2× (a3b has 256
   experts not 128 — mean htpe is 256, not 512). Distribution is
   also heavy-left: ~30% of cells live at M<64 where dense is
   catastrophic (37% of gather throughput).

3. **Direct microbench**
   (`bench_per_expert_vs_gather_real_distribution`) confirmed:
   dense loses by 12% on the MoE matmul stage at the real
   distribution. ~6% end-to-end prefill regression if shipped.

The 1k-tok/s llama.cpp reference that motivated this plan turned
out to come from **kernel-level differentiators** (single-dispatch
per-expert grid with SIMD-group MMA, plus MTLResidencySet for
buffer pinning), not from the dispatch pattern itself. See
`llama_cpp_moe_differentiators.md`.

Everything below this line is the original 2026-05-20 plan, kept
intact as a record of the analytical reasoning that didn't survive
empirical contact.

---

# Why pivot

2026-05-20 measurements (post-reboot, in-process A/B, n=5 trials,
high-perf power):

| variant | expert_gate_up GFLOP/s | expert_down GFLOP/s |
|---|---|---|
| BM=16 WM=2 | 6965 (-10%) | 6714 (-7%) |
| **BM=32 WM=2 (MLX default)** | **7768** | **7186** |
| BM=64 WM=4 | 5923 (-24%) | 5315 (-26%) |

Dense `affine_qmm_t` at comparable shapes hits **~9150 GFLOP/s**
consistently (88% of M2 Max FP32 peak). The 18-25% gap between the
gather kernel and the dense kernel is the in-kernel gather
indirection — the per-row `indices[r]` lookup + per-expert weight
base address computation eat registers and serialize part of the
load path.

The BM sweep (16/32/64) is exhaustive on the tile-M axis under
the MLX template constraints. MLX's default is the apex. We can't
tile-tune our way past dense's ceiling because the indirection
itself is the cost.

# What to build

Llama.cpp's `mul_mat_id` pattern, adapted to moeflux:

**Stage 1 — routing kernel.** Builds two arrays:
- `htpe[num_experts]` — how many tokens (assignments) hit each expert
- `hids[num_assignments]` — flat list of token-row indices, ordered
  by expert. `hids[htpe_offset[e]..htpe_offset[e+1]]` is expert e's
  token list.

For our pre-permute-fuse layout, `hids` is essentially identity
(rows are already grouped by expert in `bucket_input`). So we may
not even need a separate kernel — the existing `bucket_accumulate`
pass already produces the layout. Verify before writing it.

**Stage 2 — per-expert dense matmul.** For each expert with non-zero
htpe, call `affine_qmm_t` with:
- `input = bucket_input + htpe_offset[e] * hidden_dim`
- `weights = expert_base + e * expert_stride`
- `output = bucket_output + htpe_offset[e] * out_dim`
- `n_tokens = htpe[e]`

This issues `num_experts_hit` (typically all 128 for a full prefill
chunk) plain matmul calls per layer per matmul type. Each call sees
its own M=htpe[e] (mean = 65536/128 = 512), with the gain that
each call hits the 9150 GFLOP/s ceiling.

# Trade-off analysis

**Cost going in:**
- Per-call overhead: encoder + dispatch headers per call. For 128
  experts × 2 calls per layer (gate_up + down) × 60 layers × 2
  chunks = 30720 calls per prefill, vs current 360 gather calls.
  Need to encode all into a single cmdbuf to amortize commit
  overhead (Apple's encoder is per-command-buffer cheap; the cost
  is the kernel launch latency itself).
- Bookkeeping: per-expert offset tables, htpe construction
  (likely already free from bucket_accumulate).

**Expected gain:**
- Dense throughput on the MoE path: 9150 GFLOP/s vs current 7200-
  7740. Net: ~15-20% on the MoE matmul total.
- Since MoE matmul is ~53% of prefill GPU time, this is **~8-11%
  end-to-end prefill speedup** if the per-call overhead doesn't
  eat the gain.

**Risk:**
- Per-call latency might dominate at M=512. Need to measure
  `affine_qmm_t` at small M (it was benched at M=8192). A separate
  bench at M ∈ {64, 128, 256, 512, 1024} will tell us where the
  small-M efficiency drops.

# Implementation order

1. **Bench `affine_qmm_t` at small M.** Add cases to `kernel_bench.rs`
   for M ∈ {64, 256, 512, 1024} at expert_gate_up + expert_down shapes.
   If GFLOP/s stays >7740 at M=512, the pivot is a definite win.
   If it collapses below 7740 at M<=512, we need a different
   design (fewer experts active, or fused multi-expert kernel).

2. **Design the moeflux-side API.** Two paths:
   - **Easy**: keep `GatherQmmCall` as today's signature, but its
     `encode` impl loops over experts and issues plain qmm_t
     dispatches per. Existing callers don't change.
   - **Clean**: new `MultiExpertQmmCall { ... }` type that takes
     `htpe_offsets: &[u32; num_experts + 1]` explicitly. Old
     `GatherQmmCall` either deprecated or kept for non-permuted
     dispatch.

   The Easy path is the right Phase-1 cut. Clean restructure is
   Phase 2 once we know the new approach holds.

3. **Wire it up.** Change `moe_block_forward` (linear_attn_forward.rs
   ~line 2580) to use the new dispatch. Stage 1 routing is mostly
   free if we leverage the existing bucket-permuted layout.

4. **Bench + verify.** Run `kernel_bench` + the full canary
   battery + a prefill samply against `prefill_prompt_long.txt`.
   Expected: prefill tok/s up by 8-11% if the small-M qmm_t bench
   was favorable.

5. **Decide the second-order moves.** If the pivot wins, the
   resulting GPU profile will shift — re-measure to find the new
   pole. Likely candidates next: the SDPA kernel (3.6% currently;
   may grow in proportion as gather shrinks) or the bf16 matmul
   parts.

# What NOT to do

- **Don't keep tile-tuning the gather kernel.** The sweep is done.
- **Don't write a custom Metal kernel from scratch** for the gather
  case until we know dense-per-expert isn't enough. MLX's dense
  kernel is already 88% of peak; we'd need to beat Apple's tuned
  output, which is a high bar.
- **Don't ship without benching the new path against a3b prefill
  end-to-end.** Per-kernel wins don't always translate. The
  `feedback_bench_discipline.md` reboot + high-perf protocol
  applies.

# Cross-references

- [[kernel_bench_bm_sweep_landed]] (forthcoming) — the data this
  pivot is motivated by.
- [[qwen_graph_mode_session6_partA_landed]] — the existing
  `gpu_moe_router` work, which is the routing-kernel building
  block we'll reuse.
- llama.cpp survey result (this session, no memory file) —
  reference design for the two-stage approach. Key file:
  `~/Projects/llama-cpp-sys/external/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
  lines 9618-10120 (`mul_mm_id_map0` + `mul_mm_id`).
