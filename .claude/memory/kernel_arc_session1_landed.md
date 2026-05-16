# Kernel arc — session 1 landed

2026-05-16. Continues `cleanup_arc_session4_landed.md`. The cleanup
arc is closed; this session **opened the kernel/prefill arc** —
mostly planning + a dead-end ruled out + Step 0.

## Where the kernel arc is going

Goal: close moeflux's **~20× prefill gap vs llama.cpp**. Per the
session-12 profile prefill is **GPU-bound** (CPU 5-10%) — it's Metal
kernel throughput, not orchestration or IO. Generation is already
competitive; prefill is the target.

## Dead end ruled out — GPU frame capture

`MTLCaptureManager` / `.gputrace` is **infeasible for moeflux** and
will not be retried. A frame capture serializes the *contents* of
every Metal buffer the captured commands touch; moeflux streams
~18 GB of model weights through GPU buffers, so a capture of any
forward pass serializes tens of GB. Measured: an 8-token prefill
produced a **58 GB** trace (libtest's own `finished in 39.77s` timer
confirms it genuinely ran — not a permission-wait artifact); a
1536-token attempt hit 42 GB and was killed. No knob fixes this —
the trace format needs resource state. The `gpu_capture.rs`
experiment was reverted (Step 0).

## Approved plan

`~/.claude/plans/sunny-stirring-grove.md` — wall-clock profiling, no
GPU timestamps, no giant artifacts:
- **Phase 1** — moeflux in-situ per-phase breakdown (which phase owns
  prefill GPU time).
- **Phase 2** — microbench moeflux's hot kernels at a3b prefill
  shapes (GFLOP/s).
- **Phase 3** — llama.cpp `test-backend-ops perf` for `MUL_MAT`
  (Q4_K, Metal) → the baseline.
- **Phase 4** — synthesize: focus = phase that dominates moeflux ×
  kernel far behind llama.cpp.

GgmlBackend idea (per-Op benchmark + 4th oracle) considered and
**deferred** — see `future_work_ggml_backend.md`. `test-backend-ops`
gives the matmul gap without the multi-session build.

## Landed this session

- Step 0 — the `.gputrace` capture experiment reverted; working tree
  clean at `6d4d4ee` (the cleanup-arc session-4 memo commit). No new
  commit — the experiment was never committed.

## RESUME POINT — Phase 1

Most of the infrastructure already exists in
`backend/gpu/metal.rs`: `commit_and_wait_labeled(cmdbuf, label)`
records per-label CPU wall-clock (≈ GPU time, it waits) into a stats
map; `cmdbuf_stats()` snapshots it; `reset_cmdbuf_stats()` clears.
The map is currently **never consumed** anywhere. Phase 1 work:
1. The linear-attn / MoE *graph* path commits via
   `Backend::submit_and_wait` (`backend/gpu/mod.rs:545`) — unlabeled.
   Route it through a labeled commit so graph cmdbufs are counted.
   (The batched full-attn path already uses labeled phases:
   `batched_rms_norm_qkv_proj`, `batched_sdpa_causal_tiled`,
   `batched_shared_ffn_moe_combine`.)
2. Surface the stats — env-gated dump after a prefill, or a public
   `RsCtx` accessor. Note: `RsCtx.backend` is `None` until lazily
   built mid-forward (`ensure_backend`), so read stats *after*
   `step_internal`, not before.
3. Drive one ~1536-token a3b prefill (the `gpu_capture.rs` test was a
   fine prefill-driver scaffold — re-create a minimal one), dump →
   per-phase ms.

Gate: `cargo build` zero-warnings; a3b suite green with `--skip
resuming_prefill_after_seq_rm_matches_full_prefill` (known breakage,
`b4a4d61`). `cargo fmt` is not a gate (crate predates it — see
`cleanup_arc_session4_landed.md`).
