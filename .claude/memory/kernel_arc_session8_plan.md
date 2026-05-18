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
self-contained. Ordered roughly cheapest → biggest rewrite. If
prefill doesn't move enough:

1. **Larger `C` (32/64)** — fewer sequential chunk-steps, more
   parallel work per chunk. `C=32` already busts the 32 KB tg-mem
   budget (`U` at `C·128` is 16 KB alone), so it needs a
   device-scratch `U` BufId (one new Op field + producer wire-up +
   lifetime coloring). Biggest single dial.
2. **Stage `q` in tg-mem** — currently the A/kqg build reads `q`
   from device memory; `k` is already staged in `kc`. Cheap, but
   competes with `U` for the tg-mem budget (see #1).
3. **Blocked triangular inverse (UT transform)** — the Phase-3
   forward substitution is per-thread sequential in `l` (a
   length-`C` dependency chain). At `C=16` this is *not* a
   bottleneck (~128 MACs/thread, dwarfed by the A-build and
   matmuls) — but if #1 pushes `C` to 32/64 the chain grows. `A` is
   strictly-lower-triangular ⇒ nilpotent, so `(I+A)⁻¹` has an exact
   blocked/recursive-doubling form (standard in the DeltaNet /
   flash-linear-attention literature). **Only pair this with #1** —
   pointless at `C=16`.
4. **Decomposed multi-kernel path** — biggest rewrite. Replace the
   single fat kernel with separate cumsum / matmul / triangular-
   solve / output kernels, intermediates in device-memory scratch
   BufIds. More occupancy and per-phase parallelism; enables
   vendoring MLX `steel/gemm` for the matmul phases and MLX `scan`
   for the cumsum (`moeflux-mlx` is a vendor-and-wrap crate — see
   the session-7 MLX investigation). Costs ~6 scratch BufIds
   plumbed through the Op + producer + lifetime coloring; that
   plumbing cost is exactly why Phase 3 chose the single kernel.

Already done inside the Phase-3 kernel (not pending levers, noted so
they aren't re-suggested): `kc` staged in tg-mem; the phase-6
per-`i` decay ratio × `U_i` hoisted out of the `ki` loop; `k_i·q_l`
precomputed once into `kqg` rather than recomputed per output
thread.

## Carry-overs

- blallama mismatch-guard Part 3 still pending (do when blallama is
  checked out) — `future_work_model_binary_mismatch_guard.md`.
