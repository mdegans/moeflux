# Session 12 landed — S10b-2: MoE-to-graph + once-per-run colored buffers

**Date:** 2026-05-16
**Branch:** moeflux `main`
**Commits (8 total, `6039406..2d93b0c`):**

- `6a28f54` — precursor: fix 3 `#[cfg(test)]` call sites missing
  `per_token_total` (session 11 added the field but never ran
  `cargo test --lib`, so the unit-test build was broken)
- `e621692` — Phase 1: `MoeCombineResidualNTokens` graph-diff test
- `4c7de74` — Phase 2a: prefix semantics for `upload`/`download`
- `c547b94` — Phase 2: hoist linear-attn graph transients to run-lifetime
- `2b87825` — Phase 3a: `Op::ZeroBuffer`
- `c64756d` — Phase 3: promote the MoE block (1f-1h) to graph form
- `471c7fe` — Phase 4: `commit_plan` once + pin (+ re-entrancy fix)
- `2d93b0c` — Phase 5: lift producer to `<B: Backend>`

**Entry:** [`qwen_graph_mode_session11_landed.md`](qwen_graph_mode_session11_landed.md)
**Plan-of-record (this session):** `/Users/mdegans/.claude/plans/ticklish-dancing-whisper.md`

## Headline state

**S10b-2 is fully landed — and exceeded the original handoff scope.**
The session-11 handoff scoped three deliverables (MoE→graph,
`commit_plan`, producer-generic). Mike chose "Path B" up front: also
hoist the transient allocation to **run-lifetime** so `commit_plan`
lands its full win. All of it shipped:

- **The entire linear-attn forward is graph-mode.** graph1 (pre-MoE
  1a-1e, landed session 11) + graph2 (MoE 1f-1h, this session:
  3× shared-FFN matvec + swiglu + `ZeroBuffer` + `MoeBatchedPermuteFuse`
  + `MoeCombineResidualNTokens`). The imperative `moe_cmdbuf` /
  `parts_mut` / per-layer `MtlBuffer` block is gone.
- **`BatchedGraphScratch`** — run-lifetime scratch, allocated once in
  `ensure_linear_resources()` at `BATCHED_CHUNK_SIZE` (8192) token
  width. Every step and layer reuses these BufIds; a smaller chunk
  (or decode at N=1) processes a prefix. Replaces ~496 fresh
  zero-filled `pool.alloc`s per chunk (the 13.8% `MetalBufferPool::alloc`
  profile pole) + the per-layer expert-blob `MtlBuffer::with_data`
  churn (~3.5 GB peak across 40 layers → a constant `num_experts`
  blob set, pread mode only).
- **`commit_plan` runs once per run**, lifetime-colors the 21 intra-graph
  transients (12 graph1 + 9 graph2), and **auto-pins** them
  (`persistent=true`) so `reset_transient` keeps them. Gated by a
  `Cell<bool>` latch on the first producer call. Boundary buffers
  (`h_mid`/`h_post`/`shared_gate`/`routing_*`/`hidden_a`/`hidden_b`
  + uploaded inputs) stay `persistent=true` — never aliased.
- **Producer is `<B: Backend>`** — no `MetalBackend`-specific calls
  remain; everything routes through the `Backend` + `BufferPool`
  traits.

**Canary 9/9 + graph_diff_oracle 10/10 + lib 67/7 green at every
commit boundary.**

## The commit_plan re-entrancy bug (Phase 4)

Phase 4's first cut broke the canary — `eval_prompt` cosine **0.58**.
Diagnosed by bisect (graph1-only `commit_plan` passed → graph2's
second call was the culprit), in the spirit of `feedback_pivot_on_discovery`.

**Root cause:** `commit_plan`'s Phase-1 "preserve non-aliasable BufIds
via swap" assumed every non-aliasable BufId owns a *unique* physical
buffer. False after a prior `commit_plan`: aliased BufIds share a
physical. The second call's swap moved the real buffer out for the
first BufId of a shared physical, then swapped out an already-placed
**placeholder** (1-byte) buffer for the rest → graph1's aliased
transients became 1-byte buffers → corruption.

**Fix:** both pools dedup the physical move via an `old_physical →
new_physical` map — each physical moved once, all BufIds sharing it
remapped to the same new slot. `commit_plan` is now correctly
re-callable. (First-call behaviour unchanged: pre-`commit_plan` every
BufId has a unique physical, so the dedup is a no-op.)

**Lesson:** the `graph_metal_matches_cpu_colored` diff test only
exercised a `ResidualAdd` chain with a *single* `commit_plan`. It did
not cover the *two-graph, two-commit_plan* path. The canary caught it
instead — a ~5-minute bisect, not hours. A colored test with two
sequential `commit_plan`s would localize this class faster; worth
adding if the multi-graph pattern recurs.

## Architecture notes

### `Op::ZeroBuffer { buf, n_bytes }`

New typed Op. The MoE permute-fuse `out_sum` accumulator is
scatter-*added* into by `moe_bucket_accumulate`, so a run-lifetime
`out_sum` must be cleared each layer. Metal arm: one blit
`fill_buffer` (hardware memset, no kernel). CPU arm: byte-slice fill.
`reads()` empty, `writes()=[buf]`.

### `commit_plan` now auto-pins + is re-callable

Two semantic changes to the trait method (doc updated in `graph/mod.rs`):
1. After coloring, every colored BufId is pinned (`persistent=true`)
   — its layout is frozen for the run; it and the shared color buffer
   survive `reset_transient`. A run-lifetime scratch set is therefore
   allocated `persistent=false`, `commit_plan`'d once, thereafter
   persistent.
2. Re-callable (the dedup fix above).

### `upload`/`download` prefix semantics

`BufferPool::upload`/`download` now accept a `host` slice *shorter*
than the buffer (copy the leading `host.len()` bytes); too-large is
still rejected. Lets once-per-run buffers be max-sized while a small
chunk touches only a prefix.

### Two graphs, two `commit_plan`s — coloring is per-graph

graph1 and graph2 are colored independently (separate `commit_plan`
calls). Their transient color buffers never overlap — slightly less
optimal than joint coloring, but correct and simple. The
graph1→graph2 boundary buffers are `persistent` (uncolored), so they
bridge the CPU-readback split safely. **R1 (cross-boundary coloring)
from the plan was eliminated by construction, not just mitigated.**

## Phase 6 — bench + profile (done, post-reboot)

**Generation bench** (`bench.py --model a3b -n 3`, essay 89-in/512-out):
mean **11.193 tok/s** (stdev 0.479) vs session-11's 11.631 — within
noise, **perf-neutral**. Expected: decode at N=1 never stressed the
allocator, so killing the alloc/bzero poles can't speed it up.

**Prefill bench** (`--prompt-file prefill_prompt_long.txt
--max-tokens 1`, 15 692-token prompt): **mean 43.62 prefill tok/s**
(iters 43.23 / 43.83 / 43.80, stdev ~0.34). llama.cpp a3b prefill
≈ 900-1000 → **~20× gap at this context length** (session-6
measured ~13× at 992 tokens; the gap widens with context).

**Profile + Activity Monitor** confirm the architectural win: during
the prefill profile, **CPU 5-10% / GPU 90-95%** — session 11 was
CPU-bound (alloc+bzero+pread = 77% CPU self-time). The machine is now
**GPU-bound**. The run-lifetime + `commit_plan` arc did what it was
for: the allocator is no longer the bottleneck.

**But** the prefill *bench* showed GPU only ~75% (vs 90-95% on the
profile glance) — i.e. ~25% of prefill is still **CPU-gap**: the
per-layer host readback + `build_expert_buckets` + `bucket_input`
permute + pread between graph1 and graph2. That is the orchestration
gap; see "What's left".

## The 20× prefill gap — decomposition (session-13+ lead)

- **CPU orchestration gap: ~1.3×.** The 75%-GPU. Closing it (GPU MoE
  bucketing + indirect dispatch + a GPU gather Op, eliminating the
  readback) recovers `1/0.75`.
- **GPU kernel throughput gap: ~15×.** The mountain. Our Metal
  kernels do the same FLOPs as llama.cpp ~15× slower. **This is a
  kernel problem, not orchestration.** Leading *hypothesis* (not yet
  confirmed — needs a GPU capture): the prefill GEMM kernels may not
  use `simdgroup_matrix` (Apple's matrix units); that alone is 8-16×
  on GEMM-bound prefill. samply gives CPU profiles only — the kernel
  arc must open with a **Metal GPU capture / per-kernel timing** to
  turn this from hypothesis into a targeted plan.

## What's left after S10b-2

- **Full-attn producer rewrite** — `batched_full_attn_layer_forward`
  still has an imperative MoE block + per-step `MtlBuffer` allocs.
  Same treatment as S10b-1b/S10b-2 but for the full-attn path. Profile
  ranks it 30.4% inclusive (vs linear-attn 64.3%) — secondary.
- **`LmHead` + `SdpaCausalTiled` Op arms** — still `todo!()` in
  `graph/metal.rs`. Needed to fully close S7-7 and for a full-attn
  graph-mode rewrite.
- **Joint graph1+graph2 coloring** — minor: would let graph1 and
  graph2 transients share physical storage. Needs one combined graph
  for `commit_plan`; `Op` would need `Clone`. Low priority.
- Producer doc comment (`linear_attn_forward.rs` ~line 2086) still
  describes the old imperative `encode_matvec_n_tokens` MoE phases —
  stale, harmless, worth a refresh.

## Verification before starting session 13

```bash
cd ~/Projects/moeflux
cargo test -p moeflux --features model-qwen3-6-35b-a3b --lib
# → 67 pass, 7 ignored

cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test graph_diff_oracle -- --ignored --test-threads=1
# → 10 passed (added moe_combine this session)

cargo test --release -p moeflux --features model-qwen3-6-35b-a3b \
  --test diff_oracle -- --ignored --test-threads=1 \
  state_round_trip_rust eval_token_matches_c_single_step \
  eval_prompt_matches_per_token_oracle slot_reuse_race_regression_rust \
  state_load_rust_from_c_save eval_prompt_matches_c_multi_token \
  eval_prompt_chunked_matches_eval_prompt_whole_prompt \
  prompt_cache_start_pos_nonzero_matches diag_b2_eval_prompt_chunk_1
# → 9 passed (~80s)
```

## Plan of record after session 12 (agreed with Mike 2026-05-16)

The graph-mode architecture arc is **done**. Two arcs follow, in
order:

1. **Cleanup / organization arc — NEXT (1-2 sessions).** Codebase
   debt: `#[allow(clippy::…)]` hiding unreadable code, `unsafe` where
   safe abstractions belong, oversized modules. **Crucial discipline:
   exclude the hot kernel files** (`gpu_matvec`, `gpu_attn`,
   `gpu_moe_router`, `.metal` sources) — the kernel arc rewrites
   them, so cleaning them now is throwaway. Clean the *stable*
   surfaces: orchestrator, graph abstraction, error types, producer
   signatures, module layout. Open it with an audit (Explore agent:
   catalog every `#[allow]`, every `unsafe` block, oversized files)
   then plan-mode. Start from the existing
   `future_work_rust_audit.md` memo (drama_llama).
   Concrete targets seen this session:
   - The repeated `unsafe { slice::from_raw_parts(ptr as *const u8,
     …) }` byte-casting (~6× in `linear_attn_forward.rs` for
     upload/download) → a safe typed-buffer helper. The test harness
     already has `bytes_of_f32`/`f32_of_bytes` — promote that pattern
     into the lib.
   - `#[allow(clippy::too_many_arguments)]` on the producers
     (`batched_linear_attn_layer_forward` has ~15 args) → a params
     struct (judgment per-site).
   - `linear_attn_forward.rs` ~2800 lines → split like `graph.rs` →
     `graph/` (session 8).
   - 8 stale `unused import` warnings in the moeflux lib build.

2. **Kernel / prefill arc — closes the 20× gap.** Phase 0 = Metal
   GPU capture / per-kernel timing (the prerequisite — turns the
   `simdgroup_matrix` hypothesis into a real breakdown). Then
   kernel-efficiency phases ordered by that breakdown. **The
   orchestration gap folds in here** — GPU MoE bucketing + indirect
   dispatch + GPU gather is kernel work, and this arc touches the MoE
   permute-fuse anyway. llama.cpp's Metal kernels are open source —
   read how they hit ~1000 tok/s.

## Top things to know for session 13

1. **Session 13 opens the cleanup arc** — audit first, then
   plan-mode. Not perf work; needs no profiling data; oracles make
   it low-risk (canary 9/9 after every commit). Don't touch the
   kernel files.
2. **`commit_plan` is now auto-pinning + re-callable.** The dedup
   fix handles repeated calls. But a colored-diff test covering two
   sequential `commit_plan`s is still missing — add it before
   trusting a 3-graph pattern.
3. **`BatchedGraphScratch` is the template** for the full-attn
   producer's eventual run-lifetime hoist: colorable transients
   `persistent=false`, boundary/uploaded-input buffers
   `persistent=true`, `commit_plan` once via a latch.
4. **Memory cost is real and accepted** — the scratch holds ~1 GB+ of
   8192-token-wide buffers resident for the run (coloring compresses
   the colorables to ~5+6 physicals). The "pool-everything is the
   destination" tradeoff; not a regression vs the imperative
   per-step prefill peak.
5. The pre-MoE→MoE split still does a **host readback** (h_post +
   routing) between graph1 and graph2 — the orchestration gap.
   `build_expert_buckets` on GPU (a counting-sort kernel) +
   indirect dispatch removes it; that is the MoE phase of the kernel
   arc, not standalone work.
