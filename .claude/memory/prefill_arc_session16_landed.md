# Prefill arc — session 16 landed (Phase 3: shared MoE block)

2026-05-19. 6 commits on moeflux `main`, every one independently green.
Plan-of-record: `prefill_arc_fullattn_migration_plan.md` (Progress
updated). Continues session 15 (Phase 0b+1+2).

## What landed

| Commit | Phase | What |
|--------|-------|------|
| `9bdc488` | 3a | scratch split — 4 structs |
| `c61d2bb` | — | memory: bufid arena correction + segv note |
| `4a647d6` | 3b/1 | extract `moe_block_forward<B: Backend>` |
| `5aef553` | 3b/2 | full-attn MoE block → shared `moe_block_forward` |
| `3c6f25b` | 3c | `batched_full_attn_layer_forward` generic |

## End state

Both batched attention producers are now `<B: Backend>` and
**structurally identical**: build one `graph1`, then call the shared
`moe_block_forward`. Full-attn's ~390-line imperative MoE tail
(`parts_mut` + 3 imperative encoder passes + `commit_and_wait`) is
gone — its MoE block is graph-mode (one `graph2`, one cmdbuf), pooled.

Scratch ownership (was the double-duty smell):
- `LinearAttnGraphScratch` / `FullAttnGraphScratch` — only per-layer
  `graph1` attention transients + their own `commit_plan` latch.
- `MoeGraphScratch` — 5 graph1→MoE boundary buffers + 14-buffer
  shared-FFN/permute-fuse working set + its own latch. **One**
  orchestrator-owned instance, reused by every layer's MoE block
  (linear + full alike — layers run sequentially, no aliasing hazard).
- `HiddenDoubleBuffer` — the run-level `hidden_a`/`hidden_b`, lifted
  to orchestrator scope.
The orchestrator (`RsCtx` / `ensure_linear_resources` /
`step_internal_batched_gqa`) owns the run-level structs and passes
`&MoeGraphScratch` to both producers.

## Decisions banked

- **Design upgraded mid-plan.** Plan-of-record said "copy linear-attn's
  graph2 into full-attn". The pre-execution design conversation with
  Mike changed it to "extract once, shared" — `moe_block_forward` is
  written against `&MoeGraphScratch`, both producers call it. Same
  work, no duplication.
- **`MoeGraphScratch` is one shared instance**, not per-attention-kind.
  The boundary + MoE buffers were converging; composition (a shared
  struct) beat flattening (one struct with both attention transient
  sets) because the attention transients genuinely differ.
- **Commit latches split** — each `graph1` gated by its attention
  scratch's latch; the shared `graph2` by `MoeGraphScratch.commit_
  planned`. `graph2`'s transient topology is identical regardless of
  which attention kind produced its inputs (only the persistent
  `hidden_out_id` alternates, which `commit_plan` ignores), so one
  pass holds for the run.
- **`reset_transient` index-order invariant.** Pool truncates BufIds
  past the last `persistent=true` one. `LinearAttnGraphScratch` is
  all-transient — `ensure_linear_resources` constructs it *before* the
  persistent-ending structs so the highest BufId stays persistent.
  Documented in-code. (`commit_plan` pins transients after the first
  call, so the window is only pre-first-commit — but kept robust.)
- **The function split was done in place** — inject a function
  boundary into the existing tail rather than transcribe ~286/~390
  lines. For 3b/2's deletion, an anchor-checked Python splice.

## Two findings (in memory)

- `future_work_diff_oracle_parallel_segv` — `diff_oracle` SIGSEGVs
  under the *default multi-threaded* test harness. Confirmed
  pre-existing vs clean HEAD `dd91850` (git-stash test). **Canary
  protocol: `-- --ignored --test-threads=1`.** Serialized: all green.
- `future_work_bufid_leak_audit` — corrected: the pool is an arena
  (`reset_transient` + RAII `metal::Buffer` drop), no memory leak.
  Real hazard = stale BufId across a `reset_transient` epoch. Mike
  wants a non-Copy RAII `BufId` eventually — its own design
  conversation.

## Next session — Phase 4

Orchestrator cleanup in `mod.rs` (`step_internal_batched_gqa`) + GPU
embedding gather + GPU final rms_norm via the `Op::LmHead` Metal arm
(the **last `todo!()`**). Phase 4 spans several orchestrator seams —
**give it its own design + plan-mode pass** per
`feedback_design_before_execute`; the plan-of-record's "O1–O7" line is
only a skeleton. Then Phase 5 (measure: reprofile vs the session-13
58.5 s / 15,692-tok baseline + bench post-reboot per
`feedback_bench_discipline`).

Verification: `cargo build -p moeflux --features
model-qwen3-6-35b-a3b`; canary `cargo test -p moeflux --release
--features model-qwen3-6-35b-a3b --test <suite> -- --ignored
--test-threads=1` (graph_diff_oracle 17, batched_diff_oracle 23,
diff_oracle 12, checkpoint_restore 7). Artifacts on
`/Volumes/Temp Backup/models/moeflux/`.
