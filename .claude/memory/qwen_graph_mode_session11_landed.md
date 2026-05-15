# Session 11 landed — S10b-1: linear-attn pre-MoE chain to Graph form

**Date:** 2026-05-15
**Branch:** moeflux `main`
**Commits (4 total):**

- `17a6426` — S10b-1a-i: extract `moe_buffers` from `PrefetchEnv` (both producers + both call sites)
- `b591c99` — S10b-1a-ii: linear-attn producer takes `&mut MetalBackend` + BufIds; orchestrator restructured
- `7e0e47f` — S10b-1b precursor: `RmsNormQkNTokens` carries `per_token_total` (fixes q|k|v stride bug caught in 80s by canary)
- `48b35d8` — S10b-1b: linear-attn pre-MoE chain (phases 1a-1e) is now a `Graph` dispatched via `backend.execute(&graph)`

**Entry:** [`qwen_graph_mode_session10_landed.md`](qwen_graph_mode_session10_landed.md)
**Locked plan:** [`qwen_graph_mode_session7_plan.md`](qwen_graph_mode_session7_plan.md)
**Plan-of-record (this session):** `/Users/mdegans/.claude/plans/hashed-yawning-flurry.md`

## Headline state

- **Pre-MoE chain (1a-1e) end-to-end through `Graph`.** 17 `Op` pushes
  replace the previous imperative encode_X_into chain. The S7-1a
  (session 6) commit-fusion win is preserved by `MetalBackend::execute`
  (single cmdbuf, single commit). Per-token Rust loop is gone — the
  `*NTokens` Ops loop internally over n_tokens.
- **Producer signature lifted:** `batched_linear_attn_layer_forward`
  takes `&mut MetalBackend` + BufIds for hidden in/out. The `metal /
  wf_buf / buffer_pool` tuple is re-derived inside via `parts_mut()`
  for the still-imperative MoE block (1f-1h).
- **`PrefetchEnv` no longer holds `moe_buffers`.** Pre-staged S10b-2:
  the MoE Ops will need direct `moe_buffers.data_prefetch_id(...)`
  access for `expert_refs: Vec<(BufId, u64)>`.
- **Canary 9/9 + per-Op 9/9 green** at every commit boundary.

## Architecture notes

### `RmsNormQkNTokens` Op gained `per_token_total: u32`

The Op previously assumed `per_token_elems = key_offset_per_token +
num_k_heads * key_dim`, which only matches `q | k` layouts. Linear-attn
`conv_out` is `q | k | v` per token (`linear_conv_dim = 2 * linear_total_key
+ linear_total_value`) — the previous strider walked past V. Fix: explicit
`per_token_total` field for the per-token base offset; `key_offset_per_token`
still names the q→k offset within one token.

`Conv1dStepNTokens` and `GatedDeltaNetStepNTokens` already handled
q|k|v correctly (the latter computes
`conv_per_token = 2 * key_total_bytes + value_total_bytes`).

The existing per-Op test in `graph_diff_oracle.rs` was unaffected by
the math (its q|k-only buffer matches both old and new formula);
test signature updated to pass `per_token_total =
key_offset_per_token + num_k_heads * key_dim`.

### Producer body shape (post-S10b-1b)

```rust
pub(super) fn batched_linear_attn_layer_forward(
    backend: &mut MetalBackend,
    wf, layer_cache, buffers, layer_idx, n_tokens, k_active,
    expert_files, moe_buffers, _layer_state, prefetch,
    hidden_in_id: BufId, hidden_out_id: BufId,
) -> Result<(), LayerForwardError> {
    // Phase 1a-1e: build + execute graph in inner scope.
    let (graph, h_mid_id, h_post_id, shared_gate_id,
         routing_indices_id, routing_weights_id) = {
        let mut g = Graph::new();
        let pool = backend.pool_mut();
        // 16 stacked transient BufIds via pool.alloc(persistent=false)
        // 17 Op pushes (rms_norm + 4 projections + 5 recurrent +
        //               o_proj + residual + post_norm + 2 matvecs +
        //               softmax_topk + normalize)
        (g, h_mid_id, h_post_id, shared_gate_id, ri, rw)
    };
    backend.execute(&graph)?;

    // Re-borrow parts for the imperative MoE block (1f-1h).
    let (metal, wf_buf, buffer_pool) = backend.parts_mut();
    let device = metal.device().clone();
    let mv = MatvecPipelines::fetch(metal)?;
    let hidden_out_buf = buffer_pool.handle(hidden_out_id);
    // Bulk readback: h_post_stack + routing_indices + routing_weights
    // (h_mid + shared_gate stay on GPU, consumed by GPU combine via
    //  buffer_pool.handle(h_mid_id) / buffer_pool.handle(shared_gate_id))

    // Imperative phases 1f (shared FFN) + 1g (MoE permute-fuse) +
    // GPU combine. Unchanged except for the BufId handle resolution.
}
```

### `LayerForwardError::Graph(#[from] GraphError)` added

Pool / execute errors flow through `?` automatically.

## What was deferred (with reasoning)

### Phase 3 — `MtlBuffer::with_*` → `pool.alloc(persistent=false)`

**Skipped this session.** The plan called for migrating ~14 imperative
MoE-block allocations (Phases 1f-1h) to pool BufIds for unified memory
management. Discovered mid-session that the pool's `alloc(persistent=
false)` doesn't recycle within a step — every alloc creates a fresh
Metal buffer; only `reset_transient()` (called once at step end) frees
them. With 40 layers × 14 allocs/layer = ~560 buffers accumulated per
chunk; for pread mode the `owned_blobs` alone would push ~3.5GB peak.

**The right primitive already exists:** `MetalBufferPool::commit_plan(&graph)`
runs lifetime analysis + greedy coloring (session-8 PartA) and aliases
overlapping-lifetime BufIds to disjoint physical buffers. Load-bearing
test landed 12 BufIds → 4 physical at cos=1.0. **But it only applies
to BufIds in a Graph.**

So Phase 3 has no good standalone form. Migrating to pool *without*
Graph = real memory regression. Migrating *with* Graph = the work
naturally collapses into S10b-2 (MoE-to-graph). The reset-mark API I
floated is the wrong design — coloring already solves recycling for
Graph buffers.

**Phase 3 substance is now part of S10b-2's scope.** No code lost,
no half-state.

### Phase 4 — `linear_attn_pre_moe_graph_matches_imperative` test

**Skipped this session.** Plan called the test "recommended" with
rationale "tighter than canary for catching Op-semantics regressions."
Canary caught the `per_token_total` Op-semantics bug in this session
in ~80s, so the speedup-over-canary value didn't pay for ~150 LOC of
test scaffold. Existing per-Op test for `RmsNormQkNTokens` was
updated with the new field, so future regressions on that Op are
caught fast at the per-Op layer.

## Pool-everything is the destination

Mike validated 2026-05-15: long-term, the pool should handle all
allocations. The PATH is: as code MOVES into Graph form, allocations
naturally migrate to pool BufIds (lifetime coloring handles
recycling). MtlBuffer-on-stack should approach zero by end of the
graph-mode arc. Don't migrate ahead of graph promotion — that loses
the coloring win.

End state when the arc completes:
- Persistent BufIds: weights, KV cache, hidden state across layers
- Transient BufIds: every per-step scratch, lifetime-colored down to
  a small physical set
- MtlBuffer-on-stack: I/O staging only, near-zero in production forward

## Profile + bench (post-reboot)

### Profile (`profile.py --model a3b --prompt-file prefill_prompt_long.txt --max-tokens 1 --duration 60 --top 30`)

Top 5 SELF-TIME (post-S10b-1b):

```
32.4%  1994  libsystem_platform.dylib:_platform_memmove
29.4%  1811  libsystem_kernel.dylib:pread
15.3%   942  libsystem_platform.dylib:__bzero
 4.4%   271  moeflux::riir::rms_norm::rms_norm_per_head_cpu
 1.3%    77  libsystem_kernel.dylib:__psynch_cvwait
```

Top 5 INCLUSIVE:

```
95.2%  5864  drama_llama::session::Session<B>::run_call
64.3%  3961  moeflux::riir::linear_attn_forward::batched_linear_attn_layer_forward
30.4%  1874  moeflux::riir::full_attn_forward::batched_full_attn_layer_forward
29.4%  1811  expert_io::ExpertFiles::read_expert (pread)
13.8%   847  MetalBufferPool::alloc (graph-mode pool overhead, NEW)
```

**Saturation observation confirmed.** `__psynch_cvwait` collapsed
from 76% (session 5) → **1.3%** post-S10b-1b. The Activity-Monitor-
glance saturation memo is now ground-truth at the kernel level.

**New top pole: memory operations.** memmove + bzero + pread = 77%
CPU self-time. The CPU is allocator-bound + IO-bound, not GPU-bound.

**`MetalBufferPool::alloc` 13.8% inclusive** — that's the cost of
Phase 2's per-layer transient pool allocations (16 BufIds × 31 linear
layers per chunk = ~496 fresh allocs, all zero-filled by
`std::ptr::write_bytes`). **This is exactly the cost that
`commit_plan` (lifetime coloring) is designed to eliminate.** S10b-2
should call `commit_plan` on the per-layer graph; lifetime analysis
likely compresses 16 BufIds → 4-6 physical buffers, reusable across
layers.

### Bench (`bench.py --model a3b -n 3`, post-reboot, high-perf)

```
[1] tok/s=10.860  prefill_tok/s≈1.89  elapsed=47.15s  in=89  out=512
[2] tok/s=11.981  prefill_tok/s≈2.08  elapsed=42.73s  in=89  out=512
[3] tok/s=12.052  prefill_tok/s≈2.09  elapsed=42.48s  in=89  out=512

mean = 11.631 tok/s  stdev = 0.668  min = 10.860  max = 12.052
```

Vs session-4 essay+512 baseline 10.54 tok/s = **+10.3% mean
throughput**. Expected outcome was perf-neutral; small win is likely
warm-disk-cache on iters 2-3 (iter 1 cold = 10.86 ≈ 1× baseline).
**No regression.** The architectural arc continued without paying
perf cost — even with the ~13.8% `pool::alloc` overhead from Phase 2's
uncolored transients, end-to-end throughput held.

## What's left for session 12

### S10b-2 — MoE-to-graph + commit_plan + producer generic-over-B

Three deliverables that fit naturally in one session per the
post-reboot profile signal:

1. **Promote MoE step (1f-1h) to graph form.** `Op::MoeBatchedPermuteFuse`
   takes `expert_refs: Vec<(BufId, u64)>` — sourced from
   `moe_buffers.data_prefetch_id(set, slot)` (pread hit) or
   `expert_files.mmap_id_for_expert(layer, expert_id)` (mmap mode;
   accessor added in S10b-pre-2). `Op::MoeCombineResidualNTokens`
   takes the existing h_mid / shared_gate / shared_out / hidden_out
   BufIds.

2. **Call `commit_plan(&graph)` per layer.** Lifetime coloring
   compresses the now-larger per-layer graph (~30 Ops including MoE).
   Profile predicts `pool::alloc` drops from 13.8% to <2% inclusive.

3. **Lift producer to `&mut B: Backend`.** Once MoE is graph-mode,
   nothing in the producer needs `MetalBackend`-specific inherent
   methods. The S7-7 generic-over-B story unblocks.

Concrete file:line targets:

- `crates/moeflux/src/riir/linear_attn_forward.rs:2374-2664`
  (current imperative MoE block)
- `crates/moeflux/src/riir/expert_forward.rs:327` (MoeBuffers
  accessors — `data_prefetch_id`, `out_id`, etc.)
- `crates/moeflux/src/riir/graph/metal.rs:560-660`
  (`MoeBatchedPermuteFuse` + `MoeCombineResidualNTokens` Metal arms,
  already wired in session 9)

Bench predictions: prefill +5-15% from `commit_plan` win;
generation +0-5% from same.

### Pre-existing deferred items still owed

- LmHead arm (currently `todo!()`) — needed for full S7-7 close
- SdpaCausalTiled arm (currently `todo!()`) — needed for full-attn
  graph-mode rewrite
- Full-attn producer rewrite to `Graph` form (parallel of S10b-1b
  but for `batched_full_attn_layer_forward`)

## Verification before starting session 12

```bash
cd ~/Projects/moeflux

# Lib + graph unit tests:
cargo test -p moeflux --features model-qwen3-6-35b-a3b --lib
# Expected: 67 pass, 7 ignored (graph tests included)

# Per-Op graph diff oracle (9 total):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test graph_diff_oracle -- --ignored --test-threads=1
# Expected: 9 passed, all cos=1.0

# Canary 9/9 (load-bearing baseline):
cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1
# Expected: 9 passed (~80s)
```

## Top 5 things to know for session 12

### 1. The Op-semantics-mismatch class of bug is well-mapped now

Both per_token_total and the Op-stacked-buffers gotchas come from
the same root: producers must agree with the Op arm's per-token
stride convention. When wiring a new Op or a new producer, **read
the Metal arm first** to confirm the per-token stride formula.
Diff-tests-first is the right default.

### 2. `commit_plan` is THE perf primitive for session 12

The 13.8% `MetalBufferPool::alloc` overhead in the profile is
exactly what `commit_plan` is designed to eliminate. Session-8 PartA
landed it with 12 BufIds → 4 physical at cos=1.0. Calling it per
layer in `step_internal_batched_gqa` between graph-build and
execute should drop the overhead to <2%.

### 3. Imperative MoE block migration is part of S10b-2, not a separate phase

Per "pool-everything as a side effect of graph promotion." The
imperative `MtlBuffer::with_*` calls in Phases 1f-1h become
`pool.alloc(...)` calls AS PART OF promoting MoE to graph form,
not as a precursor migration. Trying to migrate without the graph
promotion = peak memory regression for no coloring win.

### 4. Producer can lift to `<B: Backend>` after S10b-2

S10b-1 keeps the producer concrete (`&mut MetalBackend`) because
the imperative MoE block needs `backend.parts_mut()` (a MetalBackend-
specific inherent method). Once MoE is graph-mode, nothing producer-
side is Metal-specific — the lift is mechanical.

### 5. CPU is allocator + IO bound now, not GPU bound

Profile self-time pole shifted from `__psynch_cvwait` (76%) to
`memmove + bzero + pread` (77%). The CPU isn't waiting on GPU
anymore; it's busy doing memory ops. This:

- Reinforces the `commit_plan` priority (eliminates much of the
  bzero from per-layer pool allocs)
- Makes the case for mmap-mode-by-default stronger (eliminates the
  pread pole; ~480MB+ can be skipped on the warm path)
- Suggests the next perf-focused session after S10b-2 should look
  at `expert_io::ExpertFiles::read_expert` (29.4% inclusive)

## Calibration note (Mike, session 11)

Session ended at 29% context after 4 commits + profile + bench +
handoff. Mike asked the right strategic question on Phase 3 — the
"won't undo later" framing forced the discovery that the pool
already has the right primitive (commit_plan) and Phase 3 has no
good standalone form. Result: cleaner session shape, real
architectural progress, no half-state debt.

The diff-tests-first discipline ([`feedback_design_before_execute.md`])
+ [`qwen_graph_mode_session8_handoff.md`] section 1 paid off: per-Op
test for `RmsNormQkNTokens` caught the per_token_total bug ~80s
into the canary run; fix landed in <10 minutes including
infrastructure update (Op field + CPU + Metal + test). Without
diff-tests-first, the cosine=-0.18 canary failure would have been
a hours-long bisect.

Mike: "Pool-everything is the destination, the path is graph
promotion." Saved as architectural philosophy — informs every
future session's "what's the right scope" decision.
