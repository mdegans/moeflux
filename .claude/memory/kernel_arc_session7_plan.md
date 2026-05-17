# Kernel arc — session 7 plan

Carries over from `kernel_arc_session6_landed.md` (P7 + per-op
breakdown). Goal of the arc: **close the a3b prefill gap with
llama.cpp** — we're at 231 tok/s on 992-token prefill, llama.cpp is
~900, so ~4× to go. a3b prefill is GPU-compute-bound.

## Task 1 — warm-up: binary↔weights mismatch guard
Full spec in `future_work_model_binary_mismatch_guard.md`. 3-part
(moeflux descriptive `RsError::ModelMismatch` at `RsCtx::open` +
drama_llama predictor un-panic + blallama `map_session_err`). ~30 min.
Do this first — small, self-contained, clears the `--no-build`
footgun that cost a debugging detour.

## Task 2 — headline: `gated_delta_net_step` (28% of prefill)
The single biggest prefill pole (per-op breakdown in the session-6
memo). `Op::GatedDeltaNetStepNTokens`, runs ×30 — every linear-attn
layer. The GatedDeltaNet linear-attention state recurrence.

### The idea (carry this over)
Session 5's "batched linear-attn" batched the *projections* (qkv/z/
etc., 3.42×) but the **recurrence core is still a sequential
per-token scan** — and that's now the wall. The known fast path is
the **chunkwise-parallel DeltaNet** formulation:

- Split the sequence into chunks of size C.
- *Within* a chunk: the delta-rule recurrence reformulates into
  matmuls (the within-chunk dependency becomes a triangular solve /
  cumulative form) — fully parallel, GPU-friendly.
- *Across* chunks: carry the recurrent state matrix sequentially —
  only n/C sequential steps instead of n.
- This is the standard "parallelizing linear transformers with the
  delta rule" / Gated DeltaNet chunkwise algorithm. llama.cpp's
  linear-attn throughput comes from exactly this.

### How to run the session
1. **Design conversation first** (per `feedback_design_before_
   execute` — this is real architectural work): the chunkwise math,
   chunk size C, how the gated state carries across chunks, what new
   Ops / Metal kernels are needed vs reusing existing matmul Ops.
2. **Diff oracle**: the *existing* per-token `gated_delta_net_step`
   is the oracle. The chunkwise version must match it cosine 1.0 —
   same pattern the batched-prefill sessions used. Write the diff
   test before trusting the kernel.
3. Plan-mode → execute.

### Open questions to resolve in design
- What exactly did session 5 batch? Confirm the recurrence core is
  still token-stepped (read the `GatedDeltaNetStepNTokens` producer +
  kernel). Don't assume.
- Chunk size — interacts with `BATCHED_CHUNK_SIZE`.
- New kernels vs composing existing batched matmul Ops.

## Task 3 — optional / later: `moe_permute_fuse` (21%)
#2 pole. P7's gather GEMM already landed here. The remaining lever is
the gather kernel's tile efficiency for ragged M (992 isn't ÷32 → the
unaligned-M PSO). Smaller win — only after the recurrence.

## Don't forget
- Per-op profiler now works + emits the breakdown (drama_llama
  `f650d64`, `MOEFLUX_PROFILE_PER_OP=1`). Use it to re-measure after
  the recurrence lands — confirm the pole moved.
- Bench discipline: reboot before claim-grade A/B; directional
  in-session A/B (env toggle, one binary) is fine for sign+magnitude.
- Build the *right* feature (`--no-build` + wrong `--model` = the
  footgun Task 1 fixes).
