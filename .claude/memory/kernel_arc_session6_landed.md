# Kernel arc — session 6 (P7): MoE gather-QMM — Part 2 landed, Part 1 handed off

2026-05-17. Continues `kernel_arc_session5_landed.md`. P7 targets the
gathered MoE expert matmul (`graph_moe`, ~35% of prefill) and the dense
`shared_ffn` / projection matmuls. Split into **Part 2** (dense matmuls
→ MLX `qmm_t`) and **Part 1** (the gather MoE itself). Part 2 landed and
verified; Part 1 was implemented backend-side then reverted at the
caller boundary — see Discovery.

Plan file: `~/.claude/plans/splendid-dancing-canyon.md`.

## Landed & verified — Part 2 (uncommitted working tree)

**Dense matmuls onto MLX `qmm_t`.** All 9 dense `encode_matvec_n_tokens`
callsites in `full_attn_forward.rs` — q/k/v/o projections + MoE router
gate + shared-expert gate + shared-FFN gate/up/down — now route through
`encode_dense_matmul_n_tokens` (new, `gpu_matvec.rs`): 4-bit → MLX
`qmm_t`, 8-bit → the old matvec fallback. Canary
`eval_prompt_matches_per_token_oracle` cosine = 1.0000000 on a3b.
Covers `batched_rms_norm_qkv_proj` (6.2%) + `batched_oproj_post_attn_
route` (3.0%) + `shared_ffn_moe_combine` (12.6%) ≈ 22% of prefill.
linear-attn's dense projections already route via `Op::MatvecNTokens`
→ `qmm_t` (session 4) — no change needed there.

**moeflux-mlx — the gather kernel is bound and diff-tested** (ready for
Part 1 to consume):
- `affine_gather_qmm_rhs` in `quantized.h`: added a `ScaleT` template
  param (bf16 scales, mirroring `affine_qmm_t`); replaced the derived
  `stride_w`/`stride_s` with explicit `buffer(9)`/`buffer(10)` kernel
  args — moeflux packs gate|up|down in one expert block, so the
  inter-expert stride is the whole block, not one tensor's N·K.
- `qmm.metal` instantiates `affine_gather_qmm_rhs_float_gs_64_b_4_t_true`.
- `lib.rs`: `GatherQmmCall` struct, `encode_gather_qmm_rhs`, 8
  function-constant PSOs (`align_M/N/K` at `function_constant`
  200/201/202, indexed `(M)|(N<<1)|(K<<2)`).
- `tests/gather_qmm_diff.rs`: 7/7 cosine = 1.0 (aligned / ragged-M /
  ragged-N / both / single-expert / shuffled / a3b shape).
- `QmmKernels` moved onto `MetalContext` (`metal.qmm()`), compiled once;
  `MetalBackend` no longer holds its own copy.

## NOT landed — Part 1 (the gather MoE)

The Phase-5 backend refactor (Op enum + both executors + the
`encode_moe_batched_permute_fuse` rewrite) was implemented, then
**reverted at the caller boundary** — `git checkout` of `backend/mod.rs`,
`cpu/mod.rs`, `expert_forward.rs` (pure Phase-5) + hand-revert of 3
spots in `gpu/mod.rs` (mixed Phase-4/5). Checkpoint compiles clean,
Part 2 intact. Reverted because the `linear_attn` caller integration is
deeper than the plan assumed (Discovery) and rushing BufId-space code
tired was the wrong move.

### The design — re-apply next session

One `Op` shape; `MOEFLUX_MOE_GATHER` env picks the encode path:
- `Op::MoeBatchedPermuteFuse`: replace `expert_refs: Vec<(BufId,u64)>`
  with `expert_base: BufId` + `expert_stride: u64` + `expert_indices:
  BufId` (per-row u32 slot — the gather kernel's `indices`) +
  `expert_slots: Vec<u32>` (per-bucket slot — the fallback selector).
  `reads()` → `[expert_base, expert_indices, bucket_input,
  bucket_token_idx, bucket_weights]`; `writes()` unchanged.
- `encode_moe_batched_permute_fuse` → a dispatcher over `gather: bool`:
  `encode_moe_gather` (3× `encode_gather_qmm_rhs` for gate/up/down +
  one flat swiglu over `total*moe_inter` + per-bucket scatter) and
  `encode_moe_per_bucket` (the old loop; expert at `expert_base +
  expert_slots[bi]*expert_stride`); shared `encode_bucket_scatter`.
- `MOEFLUX_MOE_GATHER` (default on, `=0` → per-bucket fallback), read
  once in `MetalBackend::new` like `profile_per_op`.
- Caller produces a uniform layout regardless of IO mode:
  - mmap, no prefetch hits → `expert_base` = the *borrowed* per-layer
    mmap buffer; `expert_slots[bi]` = raw expert id.
  - pread / prefetch / mixed → compact into one `B*expert_size` buffer,
    host-side `Vec<u8>`: `read_expert` fills non-prefetch buckets (it
    preads the file regardless of mode), memcpy from the prefetch
    buffer's `.contents()` for hits; `expert_slots[bi]` = bi.
  - `expert_indices` = `expert_slots` expanded by `buckets.offsets`.
- gather-GEMM offsets: `packed/scales/biases_offset` = `gate/up/down_
  {w,s,b}_off_4bit()` (expert 0's sub-offsets); `stride_w` =
  `expert_size_4bit()` bytes; `stride_s` = `expert_size_4bit()/2`
  (bf16 elements). gate/up: N=moe_inter, K=hidden; down: N=hidden,
  K=moe_inter. CpuBackend executor slices `expert_base` by
  `expert_slots[bi]*expert_stride` — stays the per-bucket diff oracle.

### Discovery — why it stopped at `linear_attn`

`full_attn_forward.rs` MoE caller (~1045-1139) is plain `&Buffer`-space
— a shared `resolve_moe_experts` helper + a direct call is
straightforward.

`linear_attn_forward.rs` MoE caller (~2440-2610) is **not**: experts
go through `BatchedGraphScratch` (`linear_attn_forward.rs:200`) — a
run-lifetime set of pool `BufId`s lifetime-colored by `commit_plan`.
`scratch.expert_blobs: Vec<BufId>` is per-expert run-lifetime slots;
`expert_files.mmap_id_for_expert` and `moe_buffers.data_prefetch_id`
return `BufId`s. Producing one uniform `expert_base` BufId there needs
a deliberate decision, made *before* writing code:
- a fixed run-lifetime `num_experts*expert_size` scratch slot (~339 MB
  for a3b) filled per layer, OR
- a per-layer `pool.register_borrowed(buf, bytes, label, persistent)`
  (`gpu/mod.rs:~129`) of a freshly-built `MtlBuffer` — and how that
  interacts with `commit_plan` coloring (register_borrowed BufIds must
  be non-colorable / pinned, and per-layer registration grows the
  BufId space — check whether that is the existing pattern or a leak).

### Next-session order
1. Decide the `linear_attn` `expert_base` BufId strategy (above).
2. Write the shared `resolve_moe_experts` helper; finish `full_attn`.
3. Finish `linear_attn` per the strategy from (1).
4. Re-apply Op enum + both executors + the `encode_*` rewrite.
5. `graph_diff_oracle`: update the `MoeBatchedPermuteFuse` test push
   (new fields); confirm cosine 1.0 MetalBackend vs CpuBackend.
6. Verify: `gather_qmm_diff` (still green), `graph_diff_oracle`,
   canary, a3b smoke in **both** mmap and pread modes + once with
   `MOEFLUX_MOE_GATHER=0`.
7. Bench A/B (Mike action; reboot, n≥3, per `feedback_bench_discipline`).

## Gate status
`cargo build -p moeflux --features model-qwen3-6-35b-a3b` clean,
zero-warning. moeflux-mlx: 12/12 tests (5 `qmm_diff` + 7
`gather_qmm_diff`) + compile + doctest green. Phase-4 canary cosine
1.0. Nothing committed — Mike commits Part 2.

## Commits
None yet — entire P7 arc is uncommitted working tree. Part 2 is the
commit-ready unit.
