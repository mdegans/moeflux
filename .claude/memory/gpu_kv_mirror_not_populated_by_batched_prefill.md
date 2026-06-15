# BUG: oracle decode attends a zero KV mirror after batched prefill

> **FIX LANDED 2026-06-15** (moeflux `6e4c103`, Fable 5) — Option A
> (the "preferred" direction below). The oracle GPU SDPA encoders now
> read the canonical pool KV (`GpuAttnEncodeArgs.k_cache/v_cache`);
> the `gpu_kv_k/v` mirrors, the per-token mirror memcpy, the
> `state_load` mirror sync, and `reset_gpu_attn_kv_mirrors` are all
> deleted. Verified: byte-identical layout (position-major, stride =
> kv_dim; SDPA K/V indexing is seq_stride-independent), canonical KV
> already carries every N+M row after prefill+decode (oracle path
> writes canonical at full_attn_forward.rs:408-416; sub-32 CPU
> fallback already read canonical). Builds clean.
> **STILL PENDING: GPU acceptance-test run** on the 18 GB artifacts —
> the four red diagnostics below must flip green. Until then this memo
> stays; delete it once verified and issue #2 is closed.

**Found 2026-06-12** during drama_llama v0.8.0 pre-publish validation,
by Claude (Fable 5), chasing a failing `partial_hit_output_matches_
fresh_session`. Affects **moeflux 0.1.0-pre.3 as published** and
current main (404e57c). Severity: production-correctness.

## The bug

The oracle per-token decode path's GPU SDPA fast path
(`full_attn_forward.rs` gate: `kv_len >= 32 && kv_len < GPU_KV_SEQ`)
reads the **GPU KV mirrors** `LayerForwardBuffers::gpu_kv_k/v`
(slice 5d-7b). Those mirrors are populated by exactly three writers:

1. the oracle path's own per-token append (`full_attn_forward.rs:451`),
2. `state_load` (snapshot restore),
3. `memory_clear` (zeroing).

The **batched prefill path does not write them** — its
`Op::KvCacheAppendNTokens` targets only the canonical pool KV
(`kv_state.k_id/v_id`), which is what the batched SDPA reads.

Consequence: after any batched `eval_prompt` of ≥ 32 tokens, every
subsequent oracle `eval_token` (the **default** decode mode, and what
drama_llama uses) runs full-attention SDPA against **zeros for the
entire prompt region** of all full-attn layers. Generated rows do get
appended, so the model self-attends its own generation correctly —
the prompt is what's invisible.

## Why it looked coherent in production

`full_attn_interval = 4` (a3b/a17b): only 10 of 40 layers are full
attention; the 30 linear-attention layers carry prompt context
normally. Output stays fluent and on-topic with degraded long-range /
factual recall. This is a strong candidate for residual fact-swap
ghosts (Apollo 11→13, 1969↔1989 class) previously attributed wholly
to repetition-penalty tuning.

Why the correctness mesh missed it: the gate only fires at
`kv_len ≥ 32`. Short-prompt tests (coherence workers, the original
`checkpoint_restore` suite at 6-token prompts, cross-backend probes)
never reach it; long-prompt runs were perf-oriented where "fluent but
imperfect" wasn't flagged.

## How it surfaced (the red-herring chain)

drama_llama's `partial_hit_output_matches_fresh_session` diverged
deterministically. The restore path was suspected, but
checkpoint/restore is **exonerated**: snapshot save is stable,
save→load→save is byte-identity, restore-after-long-decode round-trips
(see the new tests). The arms differed because `state_load` *fills*
the mirror from canonical — the restored session's mirror held the
true prefix while the fresh session's mirror held zeros. Two different
wrongs, deterministic divergence.

## Diagnostic tests (committed, `#[ignore]`d, in tests/checkpoint_restore.rs)

- `oracle_decode_after_batched_prefill_matches_oracle_prefill` —
  THE direct repro. Batched-prefill 40 toks + oracle decode vs
  oracle-fed 40 toks + decode: `[334, 334, 10223×6]` vs
  `[360, 71, 693, 369, 524, 264, 3377, 13]`.
- `restore_after_generation_matches_fresh_prefill` (three-arm A/B/C),
  `restore_divergence_grid`, `restore_divergence_prefix_sweep_zero_drift`
  — the localization chain (threshold at p≥16 → kv_len 32 boundary
  during decode).
- `restore_round_trips_after_long_decode`,
  `restore_round_trips_after_capacity_growth`,
  `restore_then_batched_tail_small`, `state_save_is_stable_after_prefill`,
  `state_load_save_identity_at_chunk_boundary`,
  `state_save_alone_does_not_perturb_live_state`,
  `prefetch_invalidate_between_prefills_is_sound`,
  `save_plus_invalidate_then_batched_tail`,
  `batched_prefill_syncs_canonical_kv` (canonical KV *is* synced —
  max f32 delta 2.6e-3 vs oracle), `restore_logit_delta_magnitude`
  (step-0 tail logits bit-identical; first oracle step Δlogit 5.7,
  growing to 20) — these all PASS and pin down what the bug is *not*.

These currently-failing ones will flip green when the fix lands:
three-arm (arm C), grid, sweep, and the direct mirror repro.

## Fix direction

Post-Phase-0b the canonical KV (`k_id/v_id`) is **already
GPU-resident** in the pool. The mirrors are vestigial duplication
from when canonical KV was host-only. Preferred fix: point the oracle
GPU SDPA encoders (`post_attention_pre_moe`, Enc A1/A3) at the
canonical buffers with the right `seq_stride` (canonical is
MAX_SEQ_LEN-row contiguous vs the mirror's GPU_KV_SEQ), delete
`gpu_kv_k/v`, the per-token mirror memcpy, the `state_load` mirror
refresh, and `reset_gpu_attn_kv_mirrors`. Alternative (smaller, less
clean): make `Op::KvCacheAppendNTokens` also write the mirror.

After the fix: re-run the full checkpoint_restore suite (all should
pass), drama_llama's moeflux_session_pollution + hash-cache tests,
and re-baseline the qwen3 long-form degradation observations — the
rep-penalty conclusions may partially attribute to this.
