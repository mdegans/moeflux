# Kernel arc — session 5 landed (batched linear-attn per-token ops)

2026-05-17. Continues `kernel_arc_session4_landed.md` (MLX quantized
GEMM). That session's re-profile left the linear-attn **per-token-loop
ops** as a known target: five ops in the linear-attn prefill path each
dispatched one Metal compute encoder *per token* (up to 8192/op/layer),
each launching a tiny 16–64-threadgroup grid — the GPU finished each
micro-dispatch before the next was encoded. Plan:
`~/.claude/plans/wobbly-mapping-firefly.md`.

## What landed

The five ops are named `*NTokens` but the encoder arms still looped
per-token. Phase 0 `MOEFLUX_PROFILE_PER_OP` baseline (M=8192, wall
66.76 s) put them at **22.0 % of prefill** (the plan estimated ~16 %).

Each op's per-token loop collapsed to a single batched dispatch. The
`Op` enum shapes did **not** change (they already carried `n_tokens` +
all stride fields) — only the Metal kernels and their encoder arms.
Two patterns:

- **Pattern A — fully parallel** (token-independent ops): the grid
  gains a token dimension; the kernel computes per-token buffer
  offsets internally. `rms_norm_qk`, `gated_rms_norm`,
  `compute_decay_beta`, and `conv1d`.
- **Pattern B — in-kernel time loop** (recurrent op):
  `gated_delta_net_step`. The recurrence is sequential over time but
  fully parallel over (head, value-row) — each thread owns one state
  row, every step independent across rows. The `for t` loop moved
  *inside* the kernel: one dispatch, head-parallel.

conv1d is two dispatches (`conv1d_step` compute + new
`conv1d_state_update`): a single kernel would race on the
`conv_state` read/write across token threadgroups. It's a width-4
causal depthwise conv — the virtual sequence `[conv_state[0..3],
input[0..n]]` makes the tap rule uniform, so decode (`n_tokens<3`)
needs no special case.

**Two consumers, both updated.** Besides the `Op`-encoder arms
(batched producer, `eval_prompt`), the `encode_*` helpers in
`gpu_linear_attn.rs` feed `linear_attn_layer_forward` →
`step_internal_per_token_oracle`, the **default `eval_token` (decode)
path**. The helpers became thin `n_tokens=1` dispatchers of the same
batched kernels — no kernel duplication. The canary
`eval_prompt_matches_per_token_oracle` cross-checks the two GPU paths.

## Results

Re-profile (`MOEFLUX_PROFILE_PER_OP`, M=8192), Phase 0 → Phase 4:

| op | before | after | speedup |
|---|---|---|---|
| `compute_decay_beta` | 606 ms | 7 ms | 90.6× |
| `rms_norm_qk` | 4875 ms | 86 ms | 56.7× |
| `gated_rms_norm` | 2769 ms | 88 ms | 31.5× |
| `conv1d_step` | 683 ms | 76 ms | 9.0× |
| `gated_delta_net_step` | 5770 ms | 5109 ms | **1.13×** |
| **five-op total** | **14703 ms** | **5366 ms** | **2.74×** |

Five-op share **22.0 % → 8.8 %** of prefill. Production-mode
(commit-fused) prefill wall **57.0 s** @ M=8192.

**The `gated_delta_net_step` finding.** Pattern B only moved 1.13×:
its per-token loop's *dispatch* overhead was only ~11 % of its cost —
the rest is genuine recurrence arithmetic. The in-kernel form is
parallelism-capped at `num_v_heads × value_dim` = 8192 threads with a
serial time axis (the recurrence). Going meaningfully faster there
needs the **chunked / matrix DeltaNet form** (partial time-axis
parallelism) — a real kernel-design project, deliberately out of
scope this session. The four Pattern-A ops were genuinely
dispatch-bound and collapsed 35× as a group, exactly as the plan
premised.

## Verification

Each batched kernel diff-tested vs `CpuBackend` *before* trusting it
(`graph_diff_oracle`, cosine ≥ 0.9999): all five cosine=1.0.
conv1d + gated_delta_net_step tested at `n_tokens ∈ {1,2,4,16}` /
`{1,4,16}` (decode + prefill shapes) — conv1d asserts `conv_out` AND
`conv_state`, delta_net asserts the persistent `state` (state
bit-exact). Caught nothing this time, but the SwiGLU-style
diff-tests-first discipline held. One Metal-compile bug caught fast:
`uint2`/`uint` mixed position attributes — Metal requires all
`[[*_position_*]]` params the same type (→ both `uint3`).

Canary `eval_prompt_matches_per_token_oracle` cosine=1.0 after every
phase; a3b smoke PASS; all 10 `graph_diff_oracle` green;
`metal_backend_compiles_all_kernels` green (new `conv1d_state_update`
registered). `cargo build` zero-warning.

## Open / next

**Production wall A/B is unfinished.** No clean pre-batching
production baseline was captured this session, and a rigorous A/B
needs a reboot (`feedback_bench_discipline` / reboot-before-benches):
stash to `f64650b` vs HEAD, n≥3, high-perf, post-reboot. Mike action.

**P7 — the next kernel-arc session.** Re-profile top phases (M=8192):
`graph_moe` / `moe_permute_fuse` **34.9 %** (the gathered MoE expert
matmul — MLX `affine_gather_qmm_t` adapts the same way, shares
`qmm_t_impl`, so the `ScaleT` work from session 4 carries over);
`batched_sdpa_causal_flash` 13.4 %; `batched_shared_ffn_moe_combine`
12.6 % (dense matvecs → route through `qmm_t`). Together ~60 % of
prefill — P7 is the big remaining step. Fresh adaptation effort, its
own session start.

**Chunked DeltaNet** — if `gated_delta_net_step`'s residual 8.4 %
becomes worth attacking after P7, the matrix/chunked form is the
lever; until then it's the floor for that op.

## Commits (moeflux main)

`4542388` batch the 3 token-independent ops · `03ba57f` batch conv1d
(two-pass) · `34f82e3` batch gated_delta_net_step · (+ doc-comment
cleanup).
