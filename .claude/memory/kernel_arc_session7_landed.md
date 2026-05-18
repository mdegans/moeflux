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
not vendorable). `moeflux-mlx` is a vendor-and-wrap crate — a future
decomposed path could vendor MLX's `steel/gemm` + `scan`. (The
Phase-3 kernel below ended up self-contained, so no vendoring was
needed yet — kept as a Phase-4 tuning lever.)

## Headline — chunkwise Gated DeltaNet, Phase 2 (landed)

moeflux `b72fc06`. `Op::GatedDeltaNetChunkwise` (same fields/buffer
roles as `GatedDeltaNetStepNTokens` + `chunk_size: u32`) + CpuBackend
arm (gathers q|k|v from `conv_out`, calls `gated_delta_chunkwise`) +
`label`/`variant_name`/`reads`/`writes` arms. MetalBackend arm
`todo!()` (filled by Phase 3). Diff test
`cpu_chunkwise_matches_cpu_per_token_gated_delta` — chunkwise Op vs
per-token Op through CpuBackend, n ∈ {1,4,16,64} × C ∈ {8,16}, all 8
combos cosine = 1.0.

## Headline — chunkwise Gated DeltaNet, Phase 3 (landed)

moeflux `9310a1a`. The Metal kernel `gated_delta_net_chunkwise` —
single self-contained kernel (no decomposed pipeline: `encode_op`
arms can't allocate scratch on the fly). Threadgroup-per-head
(mirrors `gated_delta_net_step`'s dispatch), inner chunk loop, all
transients in threadgroup memory. `CW_C = 16` keeps every transient
within the 32 KB tg-mem budget — no new Op fields, no producer
change. **Key property**: thread `vi` owns state/output/U row `vi`
exclusively; only kc/A/kqg/log_decay/beta_s are cross-thread
(written once per chunk, read-only after) — so the forward
substitution needs no barriers, only 3 per chunk total.

Diff test `graph_metal_matches_cpu_gated_delta_chunkwise` (`#[ignore]`,
GPU): Metal vs the CpuBackend chunkwise arm, n ∈ {1,4,16,64}
(singleton, ragged, multi-chunk) — output and state **cosine = 1.0**,
max_abs ~1e-6. First try. lib 75/75 green.

Phase 3 touches no live path (producer still emits
`GatedDeltaNetStepNTokens`) — so the real-model canary cannot
regress from it; canary belongs to Phase 4.

## Next

`kernel_arc_session8_plan.md` — Phase 4: swap the producer
(`linear_attn_forward.rs:2270`) to `GatedDeltaNetChunkwise`, run the
real-model canary, then a reboot-grade prefill A/B bench.
