# Kernel arc — session 13 plan: GQA-fold (SDPA is memory-bound)

Session 12 measured SDPA at ~600–670 GFLOP/s = ~7% of simdgroup-matrix
peak → **memory/overhead-bound**. The simdgroup-matrix QK^T rewrite
regressed 9–13% and was reverted; P·V-on-tensors is dead by the same
argument. The lever for a memory-bound SDPA is **cutting memory
traffic**, not moving math onto the tensor ALU.

## Phase 0 — GPU capture FIRST (do not skip this time)

Session 12's headline was lost precisely because the session-11
plan's Phase 0 bound-analysis was skipped. Before touching the
kernel: an Xcode GPU capture / counter read on `attn_sdpa_causal_flash`
at the 8192 shape — confirm the bottleneck (threadgroup-memory
bandwidth on K/V re-staging? barrier stalls? the serial Phase-3
online-softmax?). The earlier "capture produces unusable GB of data"
worry: scope the capture to one threadgroup / one dispatch, read the
counters, don't trace everything. This capture *orders* the work
below — GQA-fold assumes K/V re-staging is the cost; verify it.

## Phase 1 — GQA-fold (the headline, assuming Phase 0 confirms)

Today: one threadgroup per `(q_tile, head)`. The `heads_per_kv = 8`
query-heads sharing a KV head each independently re-stage the *same*
K/V blocks from global → `kv_stage`. GQA-fold: **one threadgroup per
`(q_tile, kv_head)`**, stage each K/V block once, inner-loop over the
8 shared query-heads → K/V global reads and `kv_stage` writes cut
~8×. That is a real bandwidth reduction on the memory-bound path.

Scope (touches Rust — `feedback_design_before_execute`: design
conversation → plan mode → execute):
- Encoder grid (`gpu_attn.rs`): grid becomes
  `num_q_tiles × num_kv_heads` instead of `× num_heads`.
- Kernel restructure (`sdpa.metal`): outer KV-block loop stages K/V
  once; inner loop over the 8 query-heads. Each query-head needs its
  own `O` accumulator + online-softmax state (`row_m/l/corr`).
  Process heads sequentially so only one head's `O`/state is live
  (register budget), OR widen state ×8 if it fits — Phase 0's
  occupancy read decides.
- Causal mask / start-offset logic carries per-head unchanged.
- The kernel stays **scalar** (the session-12 result: scalar QK^T /
  P·V is fine; the math was never the bottleneck).
Gate: 5 `sdpa_causal_flash_*` diff tests cosine ≥ 0.9999 + canary +
`kernel_bench` A/B (the same harness that caught the regression).

## Dead / removed from the plan

- **QK^T → MMATile**: reverted, negative result. Do not retry.
- **P·V → MMATile**: dead — SDPA is memory-bound and Phase 5 was
  already reduction-free; the session-12 P·V spec (in git history at
  `kernel_arc_session13_plan.md` before this rewrite, commit `50a8217`)
  is not worth executing.
- The steel `MMATile` machinery stays vendored (`moeflux-metal`) — it
  is still the right tool for the *quantized* matmuls (`qmm_t` hits
  9000+ GFLOP/s). It just doesn't belong in SDPA.

## Tooling note

`kernel_bench` (`crates/moeflux/tests/kernel_bench.rs`) is the
confirmed SDPA bench: pure GPU, no model weights, warm-up + n=3 tight
trials, reports ms/dispatch + GFLOP/s at the 1536 / 8192 / 8192-32k
shapes. GFLOP/s vs the `qmm_t` rows = the bound check. Use it as the
A/B gate for any SDPA change.
