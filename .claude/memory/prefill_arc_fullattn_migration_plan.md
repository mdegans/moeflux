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
