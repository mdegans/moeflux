# Cleanup arc — session 1 landed

2026-05-16. The "cleanup/organization arc" from the session-12
plan-of-record. Audit-driven refactor of the moeflux crate
(repetition, unsafe, flat module tree, huge functions). Full plan:
`~/.claude/plans/stateful-booping-acorn.md` (9 phases, ~32-35 commits).

## Landed this session — 4 commits on moeflux main, all gated green

- `dd99e0c` Phase 0 — remove dead code, clear pre-existing warnings.
  Deleted `post_attention_residual_norm_route` (stale ~107-line "Phase
  B4" variant), `zero_shared_buffer`, 7 unused `MoeBuffers` accessors;
  dropped dead imports/vars. The lib now builds **warning-free** under
  all three model features.
- `693b0de` Phase 1 — unified 6 `bytes_as_*_panic` helpers in
  `graph/cpu.rs` into generic `bytes_as<T>` / `bytes_as_mut<T>`.
- `a3800fd` Phase 2 — `MtlBuffer::as_slice` + free `buffer_as_slice` /
  `buffer_as_mut_slice` (centralise `from_raw_parts(buf.contents())`,
  ~12 sites); `bytemuck::cast_slice` for the 4 KV-cache snapshot casts
  (removes that unsafe). New dep: `bytemuck`.
- `da8115e` Phase 3 cluster 1 — `GpuLayerCtx` (new module `gpu_ctx`).
  Copy struct bundling the 5 shared borrows (wf, wf_buf, layer_cache,
  buffers, buffer_pool); `&mut MetalContext` stays separate. 6
  layer-forward fns lose 5 params each; bodies destructure at entry.

Each commit verified: 3-feature `cargo build` warning-free + fast
tests; full `--include-ignored` gate (canary 9/9 + diff oracles
bit-exact) after Phase 0/1/2 and cluster 1.

## Tag

`pre-cleanup-baseline` marks the pre-Phase-0 tree (for the Phase-9
perf bench bisect).

## Discoveries that re-scope the rest of the plan

1. **Gate corrections (already folded into the plan doc).** Bare
   `cargo build` (no model feature) does NOT compile — 54 errors; a
   model feature is mandatory. And `clippy -D warnings` is not a
   viable gate: ~88 pre-existing moeflux-lib lints (`div_ceil` ×26,
   `is_multiple_of` ×15, 9 un-allowed `too_many_arguments`, …). The
   real gate is **3-feature `cargo build` warning-free + tests**.
2. **Known baseline test failure.** `resuming_prefill_after_seq_rm_matches_full_prefill`
   fails on `--include-ignored` — pre-existing, the accepted `seq_rm`
   partial-truncate breakage (see `qwen3_a3b_llama_cpp_rewind_diagnosis`).
   "Green" = all pass except this one.
3. **Phase 3's audit clusters are imprecise — re-scope before doing.**
   Cluster 2 ("ExpertPayload, 6 fns") is really TWO clusters: a
   host-slice payload (`h_post/h_mid/shared_out: &[f32]` +
   `expert_weights` + `shared_gate_score`) across `gpu_batched_experts_forward`
   / `_encode` / `gpu_batched_experts_begin`; and a pre-staged payload
   (`input/h_mid/shared_out: &BufferRef` + …) across the two
   `*_pre_staged` fns. Also `gpu_batched_experts_forward` is a
   **diff-oracle-tested API** — called from `tests/diff_oracle.rs` and
   `tests/common/c_backend.rs`; converting it touches the test files.
   Expect clusters 3-7 to have similar wrinkles — verify each
   cluster's real shape from the actual signatures before editing.
4. **`moeflux-sys` C-side warning** (`la_debug_count` unused in
   `infer.m`) is pre-existing, out of this Rust audit's scope.

## Next session — resume at Phase 3 cluster 2

Plan doc has the full phase list. Recommended order unchanged
(reorg Phase 8 last). Per-cluster recipe proven by cluster 1:
define `#[derive(Clone,Copy)]` struct → swap the param block for one
`ctx`/`payload` param → destructure at fn entry (bodies unchanged) →
convert intra-cluster calls + external callers → build 3 features,
fix unused-var warnings with `_`-prefix, fast test, commit.

Cluster 1's edit mechanics that worked: anchor signature edits with
the `metal:`/`cmdbuf:` line for uniqueness; the destructure-at-entry
trick keeps bodies untouched; expect a couple of `wf: _, layer_cache: _`
partial destructures where a fn only threads the ctx onward.

Highest-risk phases still ahead: Phase 4 (encoder builder, ~20
hot-kernel sites — silent-GPU-corruption risk, one file per commit +
full gate each) and Phase 7 (split the two 600-line batched fns
along their existing phase-banner comments). Do these fresh.
