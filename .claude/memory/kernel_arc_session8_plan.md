# Kernel arc — session 8 plan

Continues `kernel_arc_session7_landed.md`. The chunkwise Gated
DeltaNet arc: Phase 1 (CPU reference `gated_delta_chunkwise`, math
gate at cosine = 1.0) landed `69fe0d9`. Goal of the arc: kill the
28%-of-prefill `gated_delta_net_step` sequential scan.

## Task 1 — Phase 2: `Op` + CpuBackend graph arm + graph diff test

Promote the Phase-1 function into the graph. **Self-contained, no
GPU kernel work** — this is the CpuBackend-first step (session-7
trait discipline: a backend gets the Op working before Metal).

1. **New `Op::GatedDeltaNetChunkwise`** in `backend/mod.rs` (near
   `GatedDeltaNetStepNTokens`, ~line 530). Same fields as
   `GatedDeltaNetStepNTokens` (state, conv_out, g_decay, beta_gate,
   output, num_v_heads, value_dim, k_heads_per_v, n_tokens) **plus
   `chunk_size: u32`**.
2. **CpuBackend arm** in `backend/cpu/mod.rs` (~line 1070, beside
   the existing `GatedDeltaNetStepNTokens` arm). Slice `conv_out`
   into q|k|v views (per-token layout: `[q key_total | k key_total |
   v value_total]`, see the existing arm) and call
   `gated_delta_chunkwise`. Note `gated_delta_chunkwise` takes
   *separate* q/k/v arrays — either gather them, or add a
   conv_out-slicing variant; gather is simplest for Phase 2.
3. **Graph diff test** in `tests/graph_diff_oracle.rs` (clone the
   `check_gated_delta_net_step` harness at line 1220). Run BOTH
   `GatedDeltaNetStepNTokens` and `GatedDeltaNetChunkwise` through
   **CpuBackend** with identical inputs; assert output + state
   cosine ≥ 0.9999. (This is CpuBackend-vs-CpuBackend — the Metal
   arm comes in Phase 3. The Phase-1 unit test already covers the
   math; this covers the Op plumbing.)
4. MetalBackend arm: `todo!()` with a named deferral to Phase 3.

Decision deferred to Phase 3, not now: single coarse Op vs
producer-expands-to-sub-Ops. Keep the coarse Op for Phase 2.

## Task 2 — Phase 3: Metal kernels (the headline, likely its own session)

Diff target = Phase 2's CpuBackend arm. Decompose `gated_delta_
chunkwise` into GPU work; `C` (inner chunk) a tunable const, default
64. a3b dims: v_heads 64, k_heads 16, key_dim 128, value_dim 128,
k_heads_per_v 4.

Building blocks (per v-head, per inner chunk of C):
- **log-cumsum** of `ln(g)` → `L_l`. Tiny (C≤64 per head) — hand
  kernel, or MLX `scan`/`CumSum`.
- **dense matmuls** — `k_i·k_l` (C×C), `S_0·k_l` / `S_0·q_l`
  (C×value_dim), `k_i·q_l` (C×C), `Σ U_i·k_iᵀ` (value_dim×key_dim).
  Candidates for vendored MLX `steel/gemm` (`moeflux-mlx`).
- **C×C triangular solve** `(I+A)·U=B` — the genuinely novel
  kernel, hand-written. Forward substitution, per head; C=64.
- elementwise scaling by β, γ, Γ — fold into the above or a small
  kernel.

Likely a new set of `Op`s (or sub-Ops the producer emits) + their
MetalBackend arms, each diff-tested. Per-token kernel stays for
decode (`n_tokens=1`).

## Task 3 — Phase 4: orchestrator swap + bench

Swap the producer at `linear_attn_forward.rs:2270` from
`GatedDeltaNetStepNTokens` to the chunkwise Op. Bench with
`MOEFLUX_PROFILE_PER_OP=1` — confirm the 28% pole moved. Reboot-grade
A/B per `feedback_bench_discipline` (n≥3, high-perf power).

## Carry-overs / notes

- Design conversation for Phase 3 happened in session 7 (the math
  is locked in `gated_delta_chunkwise`'s doc comment + Phase-1
  test). Phase 3 still warrants plan-mode for the kernel
  decomposition + Op shape.
- `feedback_design_before_execute`: Phase 2 is small enough to go
  straight to plan-mode; Phase 3 wants a real design pass on the
  kernel split and MLX-vendor-vs-handwrite call.
- blallama mismatch-guard Part 3 still pending (do when blallama is
  checked out) — see `future_work_model_binary_mismatch_guard.md`.
