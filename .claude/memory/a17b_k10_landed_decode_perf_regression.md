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

## ROOT CAUSE FOUND (2026-06-18) — it is NOT k=10/chunking; it is `fcf0ae4`

Bisected this session (Opus 4.8). The k=10/chunking hypothesis below is
**DISPROVEN**. Evidence chain:
- a3b (current HEAD, k=8, **Mmap** path): healthy (16.7 tok/s warm) — the
  shared code + chunking-at-k=8 are fine.
- a17b forced to **k=8** on current code (`MOEFLUX_FORCE_K=8` build knob):
  **0.204 tok/s** — still broken. k=10 is a no-op at k=8 (1 chunk,
  `expert_base=0`), so **k is exonerated**. Same-k bracket: pre.3 1.589 →
  current 0.204, only engine delta = `fcf0ae4`.
- Only non-test engine commit since pre.3 is **`fcf0ae4`** ("oracle SDPA
  reads canonical KV, deleting the vestigial GPU mirror"). `f63352f` is
  test-only.

**Mechanism:** `fcf0ae4` pointed the per-token GPU SDPA fast path at the
**canonical KV pool** (`kv_state.k_id/v_id`, sized `[MAX_SEQ_LEN(1M) ×
num_kv_heads × head_dim]` = ~4.3 GB/full-attn-layer, ~64 GB across a17b's
~15 full-attn layers) instead of the old `GPU_KV_SEQ(8192)`-row mirror
(~320 MB total). On the RAM-tight a17b Pread config (202 GiB experts, 96
GiB RAM) the GPU now making the 64 GB canonical buffers resident evicts
expert pages → disk thrash → idle-GPU/single-core-CPU decode. a3b is
immune (total footprint fits RAM).

**Confirmed by experiment** (`MOEFLUX_FORCE_CPU_SDPA=1`, full_attn_forward
gate → CPU SDPA, GPU never touches canonical KV): decode **0.204 → 0.644
tok/s at k=10** (3.2× despite MORE experts; prefill 0.38→1.19). Does not
fully reach 1.589 because CPU SDPA over 15 full-attn layers is itself
slower than the baseline's small-mirror GPU SDPA — it is a diagnostic,
not the fix.

## FIX LANDED (2026-06-18) — mode-gated CPU-SDPA in Pread mode

Tried two fixes:
- **(B) small GPU-resident mirror** synced from canonical: works (a17b
  512-tok **1.338**) but is ~2% over CPU-SDPA (1.313) — full-attn SDPA is
  a tiny slice of an expert-streaming-bound decode, so the mirror's GPU
  speedup is noise. Reverted: not worth the buffers + sync state.
- **(A, SHIPPED) mode-gate the GPU SDPA fast path off in Pread mode.**
  `full_attn_pre_moe_layer_forward`'s `gpu_attn_ready` gate gains
  `&& !gpu.expert_io_mode.is_pread()`. Pread (a17b/cogito) → CPU SDPA
  (reads only the host KV prefix, never binds the 64 GB canonical →
  no residency thrash; CPU-SDPA was always correct — the zero-mirror bug
  was the *mirror*, not canonical). Mmap (a3b, fits RAM) → keeps GPU SDPA
  (a3b CPU-SDPA measured −17%, so gating matters). Threaded
  `expert_io_mode` into `GpuLayerCtx` (gpu_ctx.rs + 4 construct sites in
  mod.rs; 5 destructures get `..`).

**Validated:** a17b 512-tok **0.204 → 1.292** (6.3×, correct k=10; residual
vs 1.589 is the honest +25%-experts I/O cost). a3b warm **17.0** (no
regression). checkpoint_restore **23/23 pass single-threaded** incl. the
zero-mirror diagnostic. NOTE: the suite SIGSEGVs under *parallel* test
threads (23 model contexts contending for one GPU) — pre-existing harness
artifact, reproduces without this change; run model-test binaries with
`--test-threads=1`.

KEPT (per user): `forced_k` + `MOEFLUX_FORCE_K` build override
(variants.rs) — debug A/B knob, `#[allow(dead_code)]`, no-op when unset.
The `MOEFLUX_FORCE_CPU_SDPA` bisect env knob was removed (replaced by the
mode gate).

## (HISTORICAL, pre-bisect) The regression — a17b decode ~8× slower

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
