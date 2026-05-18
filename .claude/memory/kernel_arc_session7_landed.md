# Kernel arc — session 7: chunkwise Gated DeltaNet Phase 1

2026-05-18. Continues `kernel_arc_session6_landed.md`. Plan file:
`~/.claude/plans/silly-coalescing-giraffe.md`. Followed the
session-7 plan (`kernel_arc_session7_plan.md`): warm-up guard, then
the headline `gated_delta_net_step` (28% of a3b prefill).

## Warm-up — binary↔weights mismatch guard (landed)

3-part fix for the footgun where a feature-gated moeflux binary
pointed at the wrong model loaded cleanly then panicked deep in
prefill. Detail in `future_work_model_binary_mismatch_guard.md`.

- moeflux `65ae254`: `RsCtx::open` probes the manifest →
  `RsError::ModelMismatch { expected, detail }`. `probe_variant_match`
  + pure `check_variant_dims` (5 unit tests).
- drama_llama `f0f8edb`: `MoefluxError::ModelMismatch(String)` +
  `From<MfError>` arm; also fixed `moeflux` feature missing a
  `tracing` dep (unrelated pre-existing build break).
- **Part 2 (un-panic predictor.rs) turned out moot** — detection
  moved to open time, so the predictor is never built with a
  mismatched model. blallama Part 3 deferred (repo not checked out).

## Headline — chunkwise Gated DeltaNet, Phase 1 (landed)

moeflux `69fe0d9`. **Confirmed without assuming** (the plan's open
question): session 5 batched the *projections*; the delta-rule
recurrence is still a fully sequential `for t` scan inside the
Metal kernel `gated_delta_net_step` (`shaders.metal:1706-1734`),
parallelism frozen at `num_v_heads × value_dim` threads.

Phase 1 = `gated_delta_chunkwise` in `linear_attn.rs` — the
pure-Rust CPU reference for the chunkwise-parallel delta rule. The
math (locked, see the function's doc comment): per v-head, per
chunk of C tokens, with cumulative log-decay `L_l` and decay ratio
`Γ_{l,i}=exp(L_l−L_i)`:

```
A_{l,i} = β_l·Γ_{l,i}·(k_i·k_l)        (strictly lower)
B_l     = β_l·v_l − β_l·γ_l·(S_0·k_l)
(I+A)·U = B                            (forward substitution)
out_l   = γ_l·(S_0·q_l) + Σ_{i≤l} Γ_{l,i}·(k_i·q_l)·U_i
S_C     = γ_{C-1}·S_0 + Σ_i Γ_{C-1,i}·U_i·k_iᵀ
```

Log-space keeps every decay ratio bounded in (0,1] — no γ
underflow. **Math gate passed**: `gated_delta_chunkwise_matches_
per_token` diff-tests vs the sequential oracle across n ∈
{1,7,64,65,200} × C ∈ {16,64} — all 10 combos output+state
**cosine = 1.0**, max_abs ~1e-7 (f32 reassociation only). First try.

No new Op, no GPU, no orchestrator change yet — the algebra is now
proven in pure Rust before any Metal work.

## MLX reuse — investigated (Mike's question)

No turnkey gated-delta kernel exists. MLX core has only general
primitives (`steel/gemm`, `scan`/`CumSum`, `sdpa`, …); Gated
DeltaNet lives in mlx-lm as composed `mx` ops (algorithm reference,
not vendorable). `moeflux-mlx` is a vendor-and-wrap crate — Phase 3
can vendor MLX's `steel/gemm` + `scan` for the matmul/cumsum
building blocks; we hand-write only the C×C triangular solve.

## Next

`kernel_arc_session8_plan.md` — Phase 2 (`Op` + CpuBackend graph
arm + graph diff test), then Phase 3 (Metal kernels).
