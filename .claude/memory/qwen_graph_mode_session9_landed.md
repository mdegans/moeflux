# Session 9 landed — S7-6a, S7-6b, S7-6c-1, S7-6c-2

**Date:** 2026-05-14
**Branch:** moeflux `main`
**Commits (9 total):**
- `8b667d8` — S7-6a: split `gated_delta_recurrence_supplied`
- `7834903` — S7-6b: wire `RmsNormQkNTokens` Metal arm
- `5c7a185` — S7-6b: wire `GatedRmsNormNTokens` Metal arm
- `437a7a0` — S7-6b: wire `ComputeDecayBetaNTokens` Metal arm
- `09c2edc` — S7-6b: wire `Conv1dStepNTokens` Metal arm
- `7b25038` — S7-6b: wire `GatedDeltaNetStepNTokens` Metal arm
- `5f6fd05` — S7-6b: wire `MoeBatchedPermuteFuse` Metal arm
- `3cabb50` — S7-6c-1: hidden double-buffer → pool BufIds
- `ea9f828` — S7-6c-2: full `LayerForwardBuffers` migration → BufIds

**Entry:** [`qwen_graph_mode_session8_partA_landed.md`](qwen_graph_mode_session8_partA_landed.md)
**Locked plan:** [`qwen_graph_mode_session7_plan.md`](qwen_graph_mode_session7_plan.md)

## Headline results

- **6 of 8 outstanding MetalBackend arms wired** (linear-attn-relevant set:
  `RmsNormQkNTokens`, `GatedRmsNormNTokens`, `ComputeDecayBetaNTokens`,
  `Conv1dStepNTokens`, `GatedDeltaNetStepNTokens`, `MoeBatchedPermuteFuse`).
  `LmHead` + `SdpaCausalTiled` still `todo!()` — deferred to S7-7
  alongside the full-attn producer rewrite.

- **5 new per-Op diff tests in `graph_diff_oracle.rs`**: rms_norm_qk,
  gated_rms_norm, compute_decay_beta, conv1d_step, gated_delta_net_step.
  All cos=1.000000000, all max_abs within f32 ULP. MoeBatchedPermuteFuse
  skipped a per-Op diff (both CPU oracle and Metal encoder are
  independently validated; ~200 LOC of test scaffold for marginal value
  over canary).

- **Pool now owns every per-layer scratch + per-layer state buffer** the
  active forward path uses. `LayerForwardBuffers` is BufIds end-to-end;
  hidden double-buffer is BufIds end-to-end. ~140 call sites in
  `linear_attn_forward.rs` / `full_attn_forward.rs` route reads
  through `buffer_pool.handle(id)`.

- **Canary 9/9 cosine = 1.0** post-migration. No regressions.

## Diff-tests-first paid off — 3 bugs caught and fixed

1. **`rms_norm_qk_n_tokens_cpu` was missing the q double-scale.** The
   Metal shader applies `inv_scale * inv_scale` to q (absorbs the
   1/sqrt(key_dim) pre-softmax attention scaling) and `inv_scale`
   to k once. The CPU oracle was applying inv_scale once to both. Diff
   test caught it in one run; fixed in `graph/cpu.rs`. No production
   callers — the helper was added in S7-2 but never exercised.

2. **`Op::GatedDeltaNetStepNTokens` CPU arm had broken per_token_conv
   layout.** Assumed v region was `linear_total_key` floats per token;
   the actual layout is `2*linear_total_key + linear_total_value` per
   token (v uses `num_v_heads * value_dim` channels, not k_heads's
   channel count). Plus the S7-2 dummy `a_log` / `dt_dummy` workaround
   that fed g_decay through `exp(-softplus(g_decay))` instead of
   passing through. Both fixed; S7-6a's `gated_delta_recurrence_supplied`
   eliminated the dummies.

3. **`gen_synth_expert_blob` test helper produced NaN bf16 weights.**
   Random u16 with sign+exponent bits forced to 0x3F00 still allowed
   bit-14 random → NaN exponent. Fixed by generating from clamped
   f32 in `[0.75, 1.25]`.

## Architecture notes

### Pool ownership shape

One `MetalBufferPool` lives on `RsCtx` (alongside `metal`, `wf_buf`,
`linear_buffers`). Lazy-allocated in `ensure_linear_resources` from
`metal.device().clone()`. `LayerForwardBuffers::new(pool: &mut Pool)`
allocates every field as `persistent: true`. Per-step transients
(hidden double-buffer) alloc as `persistent: false`, get cleaned up
by `reset_transient` at step end.

Layer forward functions take `buffer_pool: &MetalBufferPool` next to
`buffers: &LayerForwardBuffers`. The renamed argument avoids collision
with the existing `pool: &rayon::ThreadPool` parameter several forwards
take for the parallel-pread io_pool.

### What's NOT in scope

- **`MoeBuffers` migration.** Needs:
  - 2 MiB aligned allocation (`MtlBuffer::with_aligned_len_u8` →
    `pool.alloc_aligned`).
  - `as_mut_slice` exposure for parallel pread targets.
  - Slot-array variants of allocations.

  Future session item. Until then, the linear-attn graph-mode producer
  in S7-6d will need to handle MoE imperatively or via the `MoE*`
  Ops that already accept `expert_refs: Vec<(BufId, u64)>` — the
  bucket buffers themselves stay outside the pool.

- **Generic-over-Backend `RsCtx<B>`.** Not started. Mike: "right first
  time, no half-states" — but `RsCtx` has 33+ access sites for
  `self.metal` / `self.wf_buf`. The lift is mechanical once both
  producers are graph-mode and we have a Backend trait instance per
  orchestrator. Defer until S7-7+ when full-attn joins.

- **`LmHead` + `SdpaCausalTiled` arms.** S7-7 wiring alongside the
  full-attn producer (LmHead needs workspace BufId; SdpaCausalTiled
  needs kv_dim disambiguation).

## What's next (session 10 plan)

1. **S7-6d — Linear-attn producer rewrite.** Convert
   `batched_linear_attn_layer_forward` (currently at ~720 LOC of
   imperative `encode_X_into` calls) into a Graph builder. Each
   existing encode call becomes `graph.push(Op::...)`. Per-layer
   transient scratch (RMS outputs, projections, SDPA workspace)
   allocates via `backend.pool_mut().alloc(..., false)`.

   The producer should become generic over `B: Backend` (function-
   level generic), or — for pragmatic reasons given that `RsCtx`
   doesn't yet have a Backend instance — take `pool: &mut MetalBufferPool`
   + ad-hoc pipeline references and call `encode_op_against` (a
   to-be-extracted free function). Design decision deferred to
   session 10's design conversation.

2. **S7-6e — Orchestrator flush-and-execute.** `step_internal_batched_gqa`
   rewrites the layer loop: build a Graph for runs of linear-attn
   layers, flush (commit_plan + execute via either `backend.execute(graph)`
   or a free `execute_graph_metal(...)` helper) when a full-attn
   layer appears, run full-attn imperatively, start a fresh Graph.
   Final flush + `pool.reset_transient()` at chunk end.

3. **Canary 9/9.** Load-bearing gate.

4. **S7-6f — Bench post-reboot.** n=3 on 992 and 16k prefill, compare
   to pre-session-9 baseline 74.66 tok/s. Per
   `feedback_bench_discipline.md`.

### Pre-session-10 verification commands

```bash
cd ~/Projects/moeflux

# Lib tests (23 graph + 67 lib total):
cargo test -p moeflux --features model-qwen3-6-35b-a3b --lib
# Expected: 90 passed (23 graph + 67 lib)

# Per-Op diff oracle (9 total: 4 original + 5 new):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test graph_diff_oracle -- --ignored --nocapture --test-threads=1
# Expected: 9 passed, all cos=1.0

# Canary 9/9:
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --nocapture --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1
# Expected: 9 passed
```

## Stats

| Concern | LOC |
|---|---:|
| `linear_attn.rs` (S7-6a split) | +50 / -54 |
| `graph/cpu.rs` (S7-6a + S7-6b CPU-oracle fixes) | +28 / -32 |
| `graph/metal.rs` (6 arms + new PSO fields) | +260 / -30 |
| `tests/graph_diff_oracle.rs` (5 new tests) | +700 |
| `linear_attn_forward.rs` (struct + migration) | +220 / -180 |
| `full_attn_forward.rs` (migration) | +35 / -16 |
| `mod.rs` (pool field + migration) | +75 / -25 |
| `state_snapshot.rs` (pool param) | +14 / -10 |
| **Session total** | **~1380 net LOC** |

## Calibration note (Mike, 2026-05-14)

Session started at ~37% context after S7-6c-1 landed; Mike confirmed
"we have room" and left the stop call to me. I chose to stop after
S7-6c-2 rather than push through S7-6d/e tonight — the producer
rewrite + orchestrator integration is a focused session unto itself
and benefits from fresh design thinking on Graph + Backend trait
shape (specifically: do we add MetalBackend on RsCtx now, or thread
pool/metal/wf_buf/pipelines separately for a Metal-only producer).

The diff-test-first pattern earned its keep — 3 of 5 wired arms with
diff tests caught real CPU oracle bugs in seconds each. Pattern is
load-bearing for S7-6d-onward producer rewrites: every new Op-using
producer code path should be preceded by a per-Op diff test or
canary check.
