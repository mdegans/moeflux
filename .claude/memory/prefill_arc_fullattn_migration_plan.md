# Prefill arc — full-attn GPU migration (full-residency refactor)

Plan-of-record. Session 14+. Supersedes the Axis-1 sketch in
`prefill_arc_session14_plan.md` with a catalogued, phased plan.

## Context

Session-13 profile (`profile.py`, 15,692-token prefill, 58.5 s): prefill
is **CPU/memory-bound, not GPU-bound** — 41% `_platform_memmove`, 26%
`pread`, GPU idle 97.5%. The kernels tuned sessions 9–13 are fast and
*starved*.

A full catalog of host↔device transfers / copies / allocations / syncs in
the prefill path pinpointed the cause: **`batched_full_attn_layer_forward`
is fully imperative** — ~52 of ~82 eliminable items, **30 fresh Metal
allocations per layer** (~360 `newBufferWithBytes`/chunk), **4 GPU
syncs/layer**. Three syncs are *artifacts* (readback → CPU island →
re-upload):

- Seam 1: Q/K/V proj → **CPU per-head Q/K rms_norm + RoPE + KV append**
- Seam 2: SDPA → **CPU sigmoid gate**
- Seam 3: the legitimate router seam (graph1→graph2 boundary)

`batched_linear_attn_layer_forward` was migrated to `Graph`/`Op`/
`BufferPool`/`commit_plan` in sessions 9–12 — **0 per-layer allocs, 2
syncs/layer**. Full-attn never was. **This refactor makes full-attn
structurally identical to linear-attn.**

End state — only the per-layer router seam + final logits readback cross
the bus; all activations + KV stay GPU-resident.

Out of scope (later arc): Axis 2 expert streaming / warm cache; the
zero-sync fully-fused layer (GPU CSR bucket-build + mmap-everything,
gated on verifying the mmap-GPU-page contract). `pread`/`mmap` untouched.

## Phases (canary at every boundary; diff-test-first for new kernels)

**Phase 0 — resident GPU KV cache.** Persistent pooled KV buffer,
`256k × kv_dim` f32/side (~1 GB k+v), alloc-once, never re-uploaded.
Phase 1c KV append writes into it directly; SDPA reads slices; delete the
`with_data(&kv_state.k_cache[..])` re-uploads (F20–F22). Host `kv_state`
becomes a lazy download-shadow for snapshot + diff oracle. **Phase 0b**:
unify the decode path onto the same buffer (retires `gpu_kv_k/v` 8192-cap
+ CPU-SDPA cliff). *In-arc.*

**Phase 1 — missing kernels + Ops** (each lands with its diff test
first): RoPE n-tokens (port `mlx/backend/metal/kernels/rope.metal`,
`rope_freqs` variant) → `Op::RopeNTokens`; sigmoid-gate →
`Op::SigmoidGateNTokens`; Q-split/deinterleave; embedding-gather Op.
Reuse existing `RmsNormQkNTokens`, `SdpaCausalTiled`, `LmHead`,
`ResidualAddNTokens`, `MatvecNTokens`, `ZeroBuffer`, the MoE Ops.

**Phase 2 — `FullAttnGraphScratch` + graph1.** Mirror
`BatchedGraphScratch`; rewrite full-attn phases 1+1c+2+3a+3b as one
`Graph`, `commit_plan` once. Kills Seams 1 & 2 + ~20 allocs.

**Phase 3 — graph2 (MoE) + pooled expert staging.** Mirror linear-attn's
graph2; pooled `expert_base`/`bucket_*`, `Op::ZeroBuffer`. Router seam
stays. Kills ~12 allocs.

**Phase 4 — orchestrator cleanup (`mod.rs` O1–O7).** GPU embedding
gather; GPU final rms_norm → existing `LmHead` Op. *In-arc.*

**Phase 5 — measure.** Canary 12/12 + reprofile; bench post-reboot
(`feedback_bench_discipline`).

## Critical files

- `crates/moeflux/src/riir/attn/full_attn_forward.rs` — problem surface.
- `crates/moeflux/src/riir/attn/linear_attn_forward.rs` —
  `batched_linear_attn_layer_forward` + `BatchedGraphScratch` (template).
- `crates/moeflux/src/riir/backend/gpu/mod.rs` — `MetalBufferPool`,
  `Graph`/`commit_plan`/`execute`; new `Op` arms.
- `crates/moeflux/src/riir/backend/mod.rs` — `Op` enum.
- `crates/moeflux/src/riir/backend/cpu/mod.rs` — CpuBackend Op arms.
- `crates/moeflux/src/riir/mod.rs` — `step_internal_batched_gqa`.
- `crates/moeflux/src/riir/snapshot/state.rs` — `KvCache`.
- `crates/moeflux-metal/shaders/` — new RoPE + sigmoid-gate kernels.

## Verification

`cargo build` + canary battery (`batched_diff_oracle`,
`eval_prompt_matches_per_token_oracle`) at every phase boundary. Each new
Phase-1 kernel lands its diff test before being wired. Phase 5:
`./profile.py --model a3b --prompt-file prefill_prompt_long.txt
--max-tokens 1 --duration 180 --top 35` after reboot, vs session-13.

## Progress

- **Phase 0 — DONE** (`caa44fa`). KV cache GPU-resident; batched SDPA
  reads it directly, per-layer re-upload gone.
- **Phase 0b — DONE** (session 15, `2cd4b46`). `KvCache` reworked from
  raw `Option<Buffer>` to pool `BufId`s — `register_borrowed` (not
  `pool.alloc`, which would zero-fill multi-GB). Registered eagerly in
  `ensure_linear_resources`. `truncate` zero-window dropped (overwrite-
  before-read). Canary 4/4 suites green.
- **Phase 1 — DONE** (session 15). All Ops for graph1 landed
  diff-test-first; graph_diff_oracle 17/17:
  - item 1 RoPE — `Op::RopeNTokens` (`82e4a52`, session 14).
  - 1a SDPA — `Op::SdpaCausalTiled` Metal arm wired to the `SdpaCall`
    flash kernel; the 3 vestigial accumulator fields trimmed
    (`01b8434`).
  - 1b `Op::SigmoidGateNTokens` (`34e7581`) — reuses the existing
    `sigmoid_gate` kernel flat.
  - 1c `Op::SplitQGate` (`3e833ae`) — new `split_q_gate` kernel.
  - 1d `Op::RmsNormPerHeadNTokens` (`f9d3309`) — new weighted per-head
    rms-norm kernel. `RmsNormQkNTokens` could NOT be reused (it is
    weight-free; full-attn q_norm/k_norm are learned bf16 tensors).
  - 1e `Op::KvCacheAppendNTokens` (`3e62813`) — new compute kernel.
  - **Embedding-gather (orig. item 4) moved to Phase 4** — its only
    consumer is the orchestrator; building it now would be unused.
- **Phase 2 — DONE** (session 15, `d3b342b`). `FullAttnGraphScratch` +
  `batched_full_attn_layer_forward` pre-MoE half rewritten as one
  19-Op `Graph`. Signature now mirrors the linear-attn branch
  (`&mut MetalBackend` + BufIds + scratch). MoE block still imperative,
  rewired onto the graph1 boundary BufIds. Canary: batched_diff_oracle
  23/23, diff_oracle 12/12, checkpoint_restore 7/7.

- **Phase 3 — DONE** (session 16, `9bdc488` `4a647d6` `5aef553`
  `3c6f25b`). Done as 3 sub-phases — the design conversation upgraded
  the plan from "copy linear-attn's graph2 into full-attn" to "extract
  the MoE block once, shared":
  - **3a** (`9bdc488`) — split `BatchedGraphScratch` into four structs:
    `LinearAttnGraphScratch` (12 graph1 transients), `MoeGraphScratch`
    (5 boundary + 14 MoE buffers, one shared instance), `HiddenDouble
    Buffer` (run-level), trimmed `FullAttnGraphScratch`. Killed the
    double-duty smell — neither attention scratch owns run-level state;
    the orchestrator does. Commit latches split (each `graph1` + the
    shared `graph2`).
  - **3b/1** (`4a647d6`) — extracted `moe_block_forward<B: Backend>`:
    host readback → CPU bucket build → expert staging → `graph2`.
    Linear-attn routes through it. `graph2` Op labels `linear_attn.*`
    → `moe.*`.
  - **3b/2** (`5aef553`) — full-attn's ~390-line imperative MoE tail
    deleted; calls the shared `moe_block_forward`. Full-attn's MoE
    block is now graph-mode (one `graph2`, one cmdbuf), the pooled
    `MoeGraphScratch` buffers replacing ~12 per-layer fresh allocs.
  - **3c** (`3c6f25b`) — `batched_full_attn_layer_forward` generic
    `<B: Backend>`. Both batched producers are now structurally
    identical: one `graph1`, then shared `moe_block_forward`.
  Canary green at every boundary (serialized — see
  `future_work_diff_oracle_parallel_segv`). All 3 model variants
  compile.

- **Phase 4 — DONE** (sessions 18-19, 2026-05-19/05-20). Landed
  opportunistically across five commits — the plan's "O1–O7"
  sketch translated cleanly to discrete Op work without needing a
  separate design pass:
  - `17c198a` (2026-05-19) — `Op::EmbedGatherNTokens` kernel + Op
    (CPU oracle + Metal `embed_gather_4bit` shader). Diff-test-
    first per the arc's discipline; `graph_metal_matches_cpu_embed_gather`
    landed green.
  - `4c37f3a` (2026-05-19) — `Op::LmHead` deleted as redundant.
    Final-norm + lm_head projection is exactly `RmsNormBf16NTokens`
    + `MatvecNTokens` (both already wired on both backends). The
    Metal arm `todo!()` is gone; `encode_op` has zero `todo!()`
    arms.
  - `4cbf14f` (2026-05-19) — orchestrator HEAD swap. CPU
    `embed_lookup` (67 MB host stack + pool.upload) → upload token
    ids only (N×4 bytes) → `g_head` graph with single
    `Op::EmbedGatherNTokens` dequantizing rows straight into
    `hidden_a` GPU buffer. Persistent `HeadTailScratch { token_ids,
    logits }` BufIds, allocated once at construction. Canary:
    graph_diff_oracle 18/18, batched_diff_oracle 23/23,
    diff_oracle 12/12, checkpoint_restore 7/7.
  - `ab40bb0` (2026-05-19) — orchestrator TAIL swap. The 67 MB
    `n × hidden_dim` download + CPU rms_norm + reupload path is
    eliminated. `g_tail` graph: `RmsNormBf16NTokens` (hidden_a →
    hidden_b) + `MatvecNTokens` projecting last token's row →
    logits. Only logits (~vocab×4 bytes) cross the bus. Canary
    battery green.
  - `5d93ba6` (2026-05-20) — `pread` mode teardown in
    `moe_block_forward`. The `ExpertIoMode::Pread` synchronous
    staging loop + `MOEFLUX_EXPERT_IO=pread` enum value deleted.
    Mmap'd expert buffers are now unconditional; the OS page
    cache serves what pread tried to manage. Citation:
    [[pread-teardown-landed]] (74.5s wall-clock vs 73.8s, same GPU
    time at 20.8% occupancy — 36% main-thread CPU savings, no GPU
    cost). Session-13's 26% `pread` line item is **gone** from
    the prefill critical path.

- Phase 5 — pending (measure). Reprofile vs session-13 +
  warm-state bench post-reboot. Note for the reprofile: the
  expected GPU-idle drop from 97.5% should be the headline. If
  GPU is still substantially idle, there's more host-side work
  to find that we haven't catalogued. Also: the bench should
  cover **both** `MOEFLUX_MOE_GATHER_ID` settings (default OFF
  on the old `affine_gather_qmm_rhs` path; ON on the new
  `moeflux_mm_id` port from session 19). The 333 tok/s number in
  [[prefill-residency-set-landed]] was on the OFF path; the A/B
  has not yet been recorded. See that memo's
  "Clarification (added 2026-05-21)" section.
