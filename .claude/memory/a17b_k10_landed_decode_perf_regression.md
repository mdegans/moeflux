# a17b top-k=10 landed — CORRECT, but an open decode perf regression

Status as of this session (on top of `0.1.0-pre.3`, **NOT yet published**).
Plan file: `/Users/mdegans/.claude/plans/melodic-enchanting-lecun.md`.

## What landed (all verified correct)

1. **`experts_per_tok` removed from `RsCtx::open`** (`riir/mod.rs`). Top-k is
   now model shape only — `VARIANT.num_experts_per_tok` (a3b=8, a17b=10). This
   is the root-cause fix for the k-mismatch class: the k=4 test-crash and the
   a17b k=8 under-activation both came from a runtime knob disagreeing with the
   variant. All moeflux test call sites + the drama_llama chain
   (`decoder.rs`/`engine.rs`/`session/mod.rs` + tests/bin) updated.
2. **Prefill gather-id kernel → k=10.** Added `moeflux_mm_id_map0_k10`
   instantiation (the map0 template was already `<K_TOPK>`) + PSO +
   dispatch-by-`k` + threadgroup mem from `n_experts*k*2`
   (`moeflux-metal/{shaders/gather_mm_id.metal, src/lib.rs}`).
   `MOE_MM_ID_TOPK` → `MOE_MM_ID_SUPPORTED_TOPK = &[8,10]` +
   `moe_mm_id_supports_topk()`. **Validated:** kernel diff oracle
   `gather_mm_id_diff` cosine 1.0 on new `k10-gate`/`k10-down` cases, AND a17b
   prefill ran clean end-to-end at k=10.
3. **Decode matvec → k>8 via chunking.** `dequant_matvec_4bit_v3_experts` is
   hardwired to **8 expert blobs** (`blob0..blob7`). Rather than balloon to 16
   params, added an `expert_base` offset (buffer 19) and made
   `encode_matvec_experts` (`moe/expert_forward.rs`) **dispatch in chunks of
   ≤8** (2 chunks for k=10; disjoint `out` regions, hazard-free in one
   encoder). `use_batched` relaxed `k<=8` → `k<=MAX_K(16)`. swiglu already
   covers `k*moe_intermediate`; `moe_combine_residual_flat` already runtime-K.
4. a3b gate **green** (182 pass; only the documented `resuming_prefill…`
   known-failure). Plus: `load_golden_or_skip` hardened (skip only on
   file-absent), stale comment fixes (`MAX_K` is 16, not 8).

## The OPEN regression — a17b decode ~8× slower at k=10

- **~0.19 tok/s** vs the warm k=8 baseline **1.589 tok/s** (~8×). Two
  post-reboot 48-token runs: 0.195 then 0.189 (decode_hit 0.315 both) — stable,
  so NOT cold-cache. Profile: **~idle GPU (3.6%), ~single-core CPU** — vs
  normal a17b decode **>100% CPU (parallel pread) + 20-30% GPU**. Prefill
  normal (prefill_tok/s ~0.35).
- **Output is COHERENT** (fresh, correct essay text) — so this is a *perf*
  regression, not correctness.
- **Reproducible post-reboot** (runs 1 and 2 both wall at the decode
  transition) → NOT accumulated system state / not cold-cache.
- `decode_hit = 0.315` (baseline 0.293) → prefetch is fine; **NOT** a
  prefetch-miss problem.

## Ruled OUT (don't re-chase)
- Prefetch coverage: `prefetch.dispatch` loops `0..k` over MAX_K(16) arrays.
- Miss-read serialization: `linear_attn_forward.rs:~2000` reads misses in
  **parallel** (`pool.install(|| dsts[..k].par_iter_mut()…read_expert)`,
  8-thread io_pool) — handles k=10, no serial path introduced.
- System state (post-reboot), correctness (coherent + oracle cosine 1.0).

## Suspects / bisect plan (fresh eyes next session)
1. **Decode matvec chunking overhead** (suspect #1): k=10 → 2 chunk-dispatches
   per matvec ×3 (gate/up/down) ×60 layers = 2× the tiny per-token GPU
   dispatches. But 2× ≠ 8× — look for something pathological (barrier/sync
   between chunks? per-dispatch command-buffer overhead dominating the tiny
   1-token decode work? second chunk of only 2 experts still paying full
   `x_shared[4096]` TG cost?).
2. **Check a3b decode perf** — a3b is k=8 → 1 chunk, `expert_base=0`, should be
   byte-identical to pre-session. If a3b decode also regressed → not
   chunking/k-specific. If a3b is fine → it's k=10/chunking-specific. Cheapest
   discriminator.
3. **Fair cold-vs-cold**: the 1.589 baseline was WARM/mid-session; the k=10
   runs were post-reboot/cold. Measure k=8 (revert the drama_llama patch →
   published pre.3) COLD vs k=10 COLD for honest attribution.
4. **Profile per-token decode**: GPU command-buffer time vs IO-wait — is the
   idle GPU IO-bound (disk) or dispatch-overhead-bound?

## Publish status: ON HOLD
- Target `0.1.0-pre.4` (bump workspace `Cargo.toml:9` + `moeflux`'s
  `moeflux-metal` dep; publish `-p moeflux-metal` then `-p moeflux`). **Not
  done** — holding on the decode regression (don't ship a17b 8× slower). The
  a3b portion is green & publishable; decide next session whether to publish
  with a known-issue note or fix first.
- **Loose ends to clean up:** drama_llama has a **TEMP `[patch.crates-io]
  moeflux = { path = local }`** in its `Cargo.toml` (added for local k=10
  validation) — REMOVE at/before publish. drama_llama's call-site edits
  (experts_per_tok removal) are uncommitted and depend on the unpublished
  moeflux API.
