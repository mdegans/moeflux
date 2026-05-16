# Cleanup arc — session 1 landed

2026-05-16. The "cleanup/organization arc" from the session-12
plan-of-record. Audit-driven refactor of the moeflux crate
(repetition, unsafe, flat module tree). Full plan:
`~/.claude/plans/stateful-booping-acorn.md` (9 phases).

## Landed — 6 commits on moeflux `main`, all gated green

- `dd99e0c` Phase 0 — remove dead code, clear pre-existing warnings.
  Deleted `post_attention_residual_norm_route` (stale ~107-line "Phase
  B4" variant), `zero_shared_buffer`, 7 unused `MoeBuffers` accessors.
  Lib now builds **warning-free** under all three model features.
- `693b0de` Phase 1 — unified 6 `bytes_as_*_panic` helpers in
  `graph/cpu.rs` into generic `bytes_as<T>` / `bytes_as_mut<T>`.
- `a3800fd` Phase 2 — `MtlBuffer::as_slice` + free `buffer_as_slice` /
  `buffer_as_mut_slice` (centralise `from_raw_parts(buf.contents())`,
  ~12 sites); `bytemuck::cast_slice` for the 4 KV-cache snapshot casts.
  New dep: `bytemuck`.
- `da8115e` Phase 3 cluster 1 — `GpuLayerCtx` (new module `gpu_ctx`).
  Copy struct bundling 5 shared borrows (wf, wf_buf, layer_cache,
  buffers, buffer_pool); 6 layer-forward fns lose 5 params each.
- `6b2ef4c` Phase 3 cluster 2 — `ExpertPayload` (host-slice variant).
  Bundles h_post/h_mid/shared_out/expert_weights/shared_gate_score
  across `gpu_batched_experts_forward` / `_encode` /
  `gpu_batched_experts_begin`.
- `3f17627` (+ this file) — memory.

Verified each commit: 3-feature `cargo build` warning-free + fast
tests; full `--include-ignored` gate (canary 9/9 + diff oracles
bit-exact) after Phases 0/1/2 and Phase 3.

Tag `pre-cleanup-baseline` marks the pre-Phase-0 tree.

## The big finding: the audit's later items are imprecise — re-verify before doing

The three Explore-agent audits were good at *finding* repetition but
**over-generalised**. Verified this session by checking actual code:

- **Phase 3 clusters 3-7 are mostly not real clusters.** Cluster 1 was
  genuine (6 fns × 5 identical params). Cluster 2 was half-real (the
  host-slice variant — 3 fns; the pre-staged variant is a separate
  2-fn `&BufferRef` shape, skipped as marginal). Cluster 3
  ("AttnGeometry across 5 fns") is **not a cluster** — the 5 fns use
  different dim sets (`encode_attn_*_batched_into` →
  kv_dim/seq_stride/heads_per_kv; `gpu_attn_*_batched` → num_kv_heads;
  `encode_sdpa_causal_tiled` → 7 different dims). Clusters 4-7 not
  individually verified but expect the same.
- **The "huge functions" finding is also suspect.** `graph/{cpu,metal}.rs`
  `encode_op` (432/467 lines) are long only because the `Op` enum has
  many variants; each `match` arm is a tidy 3-6 line block. Splitting
  them (Phase 6) relocates code without reducing complexity — abandoned
  as marginal.

**Lesson for next session:** treat the audit as a list of *candidates*.
Before editing any audit item, read the actual code and confirm the
repetition is real and the shape is uniform. Phase 0/1/2 and Phase 3
cluster 1 were the genuinely-uniform wins.

## Gate corrections (folded into the plan doc)

- Bare `cargo build` (no model feature) does NOT compile — a model
  feature is mandatory. Gate = the three feature builds.
- `clippy -D warnings` is not a viable gate: ~88 pre-existing
  moeflux-lib lints. Gate = 3-feature build warning-free + tests.
- Known baseline failure: `resuming_prefill_after_seq_rm_matches_full_prefill`
  fails on `--include-ignored` — pre-existing accepted `seq_rm`
  breakage (`qwen3_a3b_llama_cpp_rewind_diagnosis`). "Green" = all
  pass except that one.

## Recommended remaining work (re-scoped)

The plan's Phases 4-9 should be re-judged the same way — verify before
doing. Honest read:

- **Phase 8 (module reorg) — the clear remaining win.** `src/riir/` is
  flat (38 files); the hierarchy is wanted independent of audit
  precision. Mechanical, low correctness risk (compiler catches broken
  paths), delegable to a subagent. Target shape: `backend/{mod,cpu/,gpu/}`
  (the `Backend` trait + its two impls + substrate), concern modules
  `attn/ moe/ io/ snapshot/` at top level; `graph/` folds into
  `backend/`. Concern-first, NOT target-first (CPU path is the GPU
  path's diff oracle — keep them siblings). Do as one subdir per commit.
- **Phase 4 (encoder builder)** — the `new_compute_command_encoder →
  set_buffer×N → set_bytes×N → dispatch → end_encoding` boilerplate is
  genuinely repeated ~20× (e.g. `gpu_attn.rs`), but each kernel binds a
  different count — a zero-cost builder helps. HIGH risk (hot kernel
  sites, silent GPU corruption) — one file per commit + full gate each.
  Verify the sites are uniform enough first.
- **Phase 7 (split the two 600-line batched fns)** — real, but verify
  the phase-banner seams are clean before committing.
- **Phases 5, 6** — marginal; 6 already abandoned.

If picking one thing next session: **Phase 8 reorg.**
