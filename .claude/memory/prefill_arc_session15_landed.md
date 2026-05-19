# Prefill arc — session 15 landed (full-attn GPU migration, Phase 0b+1+2)

2026-05-19. 7 commits on moeflux `main`, all independently green.
Plan-of-record: `prefill_arc_fullattn_migration_plan.md` (Progress
section updated). Continues session 14 (Phase 0 + RoPE).

## What landed

| Commit | Phase | What |
|--------|-------|------|
| `2cd4b46` | 0b | `KvCache` → pool `BufId`s |
| `01b8434` | 1a | `Op::SdpaCausalTiled` Metal arm + field trim |
| `34e7581` | 1b | `Op::SigmoidGateNTokens` |
| `3e833ae` | 1c | `Op::SplitQGate` |
| `f9d3309` | 1d | `Op::RmsNormPerHeadNTokens` |
| `3e62813` | 1e | `Op::KvCacheAppendNTokens` |
| `d3b342b` | 2  | `FullAttnGraphScratch` + graph1 rewrite |

End state: `batched_full_attn_layer_forward`'s pre-MoE half is one
19-Op `Graph`, `commit_plan`'d once, one cmdbuf — structurally
identical to `batched_linear_attn_layer_forward`'s graph1. The two CPU
islands (per-head norm + RoPE + KV append; sigmoid gate) and the
per-token loop are gone. Verified: batched_diff_oracle 23/23,
diff_oracle 12/12 (incl. `eval_prompt_matches_per_token_oracle`),
checkpoint_restore 7/7, graph_diff_oracle 17/17.

## Decisions banked

- **KV cache uses `register_borrowed`, not `pool.alloc`.** `pool.alloc`
  zero-fills the whole buffer — for a multi-GB KV reservation that is
  the cold-init memset the MLA path fought to drop. `register_borrowed`
  keeps Mach-VM lazy-commit. Registered eagerly in
  `ensure_linear_resources` (the one site with `&mut pool`).
- **`truncate` zero-window dropped** — relies on overwrite-before-read.
  `checkpoint_restore` 7/7 confirms it.
- **`Op::SdpaCausalTiled`** wired to the production `SdpaCall` flash
  kernel; `fold` derived internally from `heads_per_kv`. The 3
  accumulator fields (`running_max`/`denom`/`v_partial`) were vestigial
  → deleted.
- **`RmsNormQkNTokens` is weight-free** and was NOT reusable for
  full-attn — q_norm/k_norm are learned bf16 tensors. New
  `Op::RmsNormPerHeadNTokens`, pushed twice (q and k separate buffers).
- **`KvCacheAppendNTokens` is a compute kernel, not a blit** — keeps
  graph1 all-compute (no encoder-type switch in `execute`). The
  KvCacheAppend→SDPA ordering within one cmdbuf works (canary green) —
  the hazard-tracking concern flagged in planning was a non-issue.
- **`batched_full_attn_layer_forward` stays concrete `&mut
  MetalBackend`** — the imperative MoE tail still needs `MetalContext`.
  Goes generic `<B: Backend>` in Phase 3.

## Next session — resume at Phase 3

- **Phase 3 — graph2 (MoE block).** The MoE block (shared FFN +
  permute-fuse + combine) is still imperative in
  `batched_full_attn_layer_forward` after the host readback. Mirror
  `batched_linear_attn_layer_forward`'s graph2 (`linear_attn_forward.rs`
  ~`let graph2 = { … }`) — it is the working template. Then the
  producer goes generic `<B: Backend>`.
- `FullAttnGraphScratch` must grow the graph2 fields — shared FFN
  intermediates, bucket-flat buffers, `out_sum`, `expert_base`,
  `expert_indices` — exactly what `BatchedGraphScratch` carries.
  `FullAttnGraphScratch::new` then takes `k_active` + `pread_mode`
  (like `BatchedGraphScratch::new`).
- **Phase 4 — orchestrator cleanup + GPU embedding gather + GPU final
  norm.** The `Op::LmHead` Metal arm is still `todo!()` (the only
  remaining one) — Phase 4 wires it. Embedding-gather Op is built here
  (deferred from Phase 1 — no consumer until now).
- **Phase 5 — measure.** Reprofile (`profile.py … prefill_prompt_long.txt`)
  + bench post-reboot per `feedback_bench_discipline`.

## Notes for next-session-me

- Diff-test-first earned its keep: the 1d kernel's Metal error (`uint2`
  threadgroup-pos attr mixed with scalar `uint` — Metal needs uniform
  vector width) was caught in one build. Keep the discipline.
- The per-token `full_attn_pre_moe_layer_forward` path still has the
  CPU islands — that is correct and intentional (it is the per-token
  diff oracle). Phase 2 only migrated the *batched* path.
- Build needs a model feature: `cargo build -p moeflux --features
  model-qwen3-6-35b-a3b`. Canary tests are `--release --ignored`,
  artifacts on `/Volumes/Temp Backup/models/moeflux/`.
