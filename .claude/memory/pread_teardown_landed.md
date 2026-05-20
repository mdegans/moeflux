---
name: pread-teardown-landed
description: Why ExpertIoMode::Pread was removed from the batched prefill path on 2026-05-20, what stays behind, and what would need to be true to re-introduce it.
metadata:
  type: project
---

# What was removed

`ExpertIoMode` enum and the `MOEFLUX_EXPERT_IO=pread` mode it gated:
the synchronous `for each bucket: read_expert; pool.upload_at` loop
that staged every layer's hit experts into a single GPU buffer on
the calling thread. Lived in `moe_block_forward` (
`linear_attn_forward.rs` around the old `let expert_base_id = if mode
== ExpertIoMode::Mmap { ... } else { ... }` block). The
`MoeGraphScratch::expert_base: Option<BufId>` field, the
`pread_mode: bool` arg to `MoeGraphScratch::new`, and the
`expert_files.mode()` accessor are all gone. The dead `if
prefetch_enabled` dispatch in `step_internal_batched_gqa` is also
gone — the batched orchestrator never fires prefetch (the N=1
re-route was reverted in session 5, see
[[qwen_batched_prefill_session5_landed]]).

# Why

Measured 2026-05-20 against a 15692-token prefill on Qwen3.6-A3B
(M2 Max, prefill_prompt_long.txt, METAL_CAPTURE_ENABLED=1 with
mid-prefill GPU capture of layers 30-31 in chunk 0):

| mode | wall-clock | tokio-rt-worker samples | `pread` self-time | `memmove` self-time |
|---|---|---|---|---|
| pread (default) | 74.53s | 15522 | 31.2% | 27.8% |
| mmap | 73.84s | 9913 | gone | gone |

Same wall-clock within noise (was already GPU-bound at the 20.83%
register-pressure occupancy ceiling on MLX's
`affine_gather_qmm_rhs_float_gs_64_b_4_t_true`). Main-thread CPU
load dropped ~36% in mmap mode. The pread loop wasn't gating
prefill tok/s — the OS page cache + readahead was already doing
what pread mode was meant to manage, and Apple Silicon's unified
memory lets the gather GEMM read straight from the mmap'd buffer
with zero staging copy.

**The pread path is dead weight on supported hardware** (96+ GB
RAM, expert working set comfortably in page cache). It would only
help on low-RAM machines where the working set thrashes — but
that's never been our target deployment, and `madvise(MADV_RANDOM
| MADV_DONTNEED)` on the mmap is a more proportionate response
than serial pread on the GPU's critical path.

# What stays (for now)

- `ExpertFiles::read_expert(...)` (the `file.read_at` wrapper) —
  the per-token oracle decode path's prefetch state machine still
  calls it via `io_pool.spawn` (parallel reads in the background
  during decode). [[qwen_prefetch_set_based_landed]] documented
  decode hit rate 5.2% → 36% from set-based matching. Whether that
  hit rate actually translates to decode tok/s is **untested as of
  2026-05-20** — see task-7 in the active session task list.
- `PrefetchState`, `MoeBuffers::data_prefetch_slots`, `io_pool` —
  the decode prefetch tower. Will be torn out alongside the
  prefetch dispatch in the per-token oracle path *only if* the A/B
  shows it isn't load-bearing.
- The mmap'd layer buffers (`mmap_layers`, `mmap_buffers`,
  `mmap_buf_ids`) — now unconditional. Used by the gather GEMM in
  both batched prefill and per-token oracle.

# Don't add it back

If you find yourself wanting a "pread" fallback for some new
hardware target: don't add the staging-blob path. Specifically:

1. **It's not a low-RAM solution.** The staging buffer is sized
   `num_experts * expert_size_4bit()` — `128 * ~30MB = ~3.8GB` for
   Qwen3-A3B. That's bigger than what we'd save in the cache-cold
   case. Low-RAM machines need *less* memory, not more.
2. **mmap can be tuned.** `madvise(MADV_RANDOM)` disables OS
   readahead (we already do this via `disable_readahead`).
   `MADV_DONTNEED` after a layer drops its working set. Both run
   on the kernel side without blocking the GPU's encoder thread.
3. **If unified-memory ever stops being a thing** (e.g., we port
   to a discrete-GPU platform): the right move is a parallel I/O
   pump that reads-ahead the next layer's experts onto the GPU
   *off* the encoder thread — not a serial pump on it. Look at
   what the decode prefetch state machine does for the K=8 case
   and generalize.

The 2026-05-20 measurement is the load-bearing data point. If you
re-introduce a synchronous expert-load path on the prefill
critical-path thread, you are walking backwards from this measured
baseline.
