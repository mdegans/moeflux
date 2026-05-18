# Kernel arc — session 8 plan: chunkwise DeltaNet Phase 4

Continues `kernel_arc_session7_landed.md`. The chunkwise Gated
DeltaNet arc — Phases 1 (CPU ref `69fe0d9`), 2 (`Op` + CpuBackend
arm `b72fc06`), 3 (Metal kernel `9310a1a`) all landed and green at
cosine = 1.0. **Phase 4 is the last step: make the win real.**

`gated_delta_net_step` was ~28% of a3b prefill (per-op breakdown).
The chunkwise kernel `gated_delta_net_chunkwise` exists, is wired as
`Op::GatedDeltaNetChunkwise`, and is bit-equivalent to the per-token
oracle — but **no live path emits it yet**. Phase 4 swaps it in and
measures.

## Task 1 — producer swap

`crates/moeflux/src/riir/attn/linear_attn_forward.rs:2270` — the
`g.push(Op::GatedDeltaNetStepNTokens { … })` call site. Swap to
`Op::GatedDeltaNetChunkwise { …, chunk_size: 16 }`.

- `chunk_size` **must be 16** — the kernel's `CW_C` is a compile-time
  constant; the MetalBackend arm `debug_assert_eq!`s it.
- Decision: route **all** token counts through chunkwise, including
  decode (`n_tokens = 1`). The kernel handles `n=1` (diff test
  cos = 1.0 at n=1 — one chunk, `c=1`). Simpler than keeping the
  per-token kernel for a decode special-case. If a later decode
  bench shows a regression, revisit (the per-token `Op` +
  kernel stay in the tree, unused — cheap to re-route).
- Check whether 2270 is the only emit site and whether it feeds both
  prefill and decode (it is the batched linear-attn producer; trace
  `n_tokens`).

## Task 2 — real-model canary

Phase 3 touched no live path, so nothing has been model-validated
yet. After the swap, run the canary battery (needs the real a3b
model dir — not checked out during session 7). Expect 9/9 cosine
green. This is the gate before trusting the swap — the diff tests
cover n ≤ 64; a full 60-layer × multi-k-token prefill is what the
canary exercises.

## Task 3 — prefill A/B bench

Per `feedback_bench_discipline` + `feedback_reboot_before_benches`:
reboot, high-perf power, n ≥ 3. A = `GatedDeltaNetStepNTokens`
(revert the swap or a flag), B = chunkwise. Measure 992-token and
larger prefill. `MOEFLUX_PROFILE_PER_OP=1` — confirm the
`gated_delta_net_step` 28% pole shrank / moved. `bench.py`.

Reboot + real model = a **Mike-assisted step**; the swap + canary
can be done first, the claim-grade bench needs his machine state.

## Tuning levers (only if the bench underwhelms)

The Phase-3 kernel is correctness-first, `C = 16`, fully
self-contained. If prefill doesn't move enough:
- Larger `C` (32/64) — needs a device-scratch `U` BufId (tg-mem
  budget); more parallel work per chunk, fewer chunk-steps.
- Stage `q` in tg-mem (currently device-read in the A/kqg build).
- MLX `steel/gemm` vendored for the matmul-heavy phases (per the
  session-7 MLX investigation — `moeflux-mlx` is a vendor-and-wrap
  crate).

## Carry-overs

- blallama mismatch-guard Part 3 still pending (do when blallama is
  checked out) — `future_work_model_binary_mismatch_guard.md`.
