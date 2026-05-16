# Cleanup arc — session 4 landed

2026-05-16. Continues `cleanup_arc_session3_landed.md`. This session
ran Phase 4a (the `ComputeEncoder` builder) and **deliberately
trimmed it** — see below. The cleanup arc is now effectively done;
the kernel/prefill arc is next.

## Landed — 3 commits on moeflux `main`

- `658d146` — add Claude Opus 4.6 to workspace `authors` (closed the
  session-3 loose end; `authors` now matches the LICENSE line).
- `b4a4d61` — document `resuming_prefill_after_seq_rm_matches_full_prefill`
  as known breakage: `#[ignore = "..."]` reason + fn-level comment.
  It was bare-`#[ignore]` (loads ~18 GB), which hid that it is *also*
  a known-failing test under `--include-ignored`. See drama_llama
  `.claude/memory/future_work_resuming_prefill_failure.md` (updated
  this session: the C-comparison investigation path is closed — C is
  gone — and the accepted fix is snapshot/restore via the existing
  `state_save`/`state_load`; Mike confirmed, breakpoints are limited
  so cache-miss cost is bounded).
- `cbbfd48` — **Phase 4a: the `ComputeEncoder` builder.** New builder
  in `backend/gpu/encoder.rs` over a Metal compute encoder:
  `begin`/`wrap` → `.pipeline().buffer().bytes().dispatch()` → `.end()`
  or `Drop`. `.dispatch()` re-callable (multi-dispatch encoders
  exist — `emit_batched_experts`); `.end()`/`Drop` guarded by an
  `ended` flag so `end_encoding` runs exactly once. `.bytes::<T>()`
  computes byte length from the type, retiring the
  `set_bytes(i,4,(&v as *const T).cast())` pointer dance. Unit tests
  cover the no-double-end logic. `gpu_rope.rs`'s `encode_yarn_rope_apply`
  migrated as the proof-of-use (1 of 55 sites).

## Phase 4a was trimmed — decision + rationale

The approved plan (`~/.claude/plans/sunny-stirring-grove.md`) was 13
commits: builder + migrate all 55 `new_compute_command_encoder` sites.
**Only the builder + one proof site landed.** The remaining 54 sites
are left to migrate **opportunistically** when those files are next
touched (the kernel arc reopens most of them).

Why: a standalone 54-site mass-migration is mechanical churn on
working code — silent-corruption risk per site (a transposed buffer
index is wrong math, no panic), slow per-commit model-load gates, no
fmt safety net — for zero behavior change. The builder *is* the
deliverable; it pays off most as new kernel-arc code uses it. Mike
agreed: kernels are the priority and the higher-leverage work.

## Discovery — moeflux is not `cargo fmt`-clean

No `rustfmt.toml`; `cargo fmt --check` reports **606 hunks across 52
files** (essentially the whole crate — hand-formatted narrower than
rustfmt's 100-col default, never machine-formatted). Implications:
- `cargo fmt --check` is **not a usable CI gate** — always red.
- Match surrounding style by eye; don't bulk-`rustfmt` files (it
  reformats untouched code into churn — bit me once this session).
- The real per-commit gate is `cargo build` zero-warnings + the
  behavioral test suite.
- Bare `rustfmt` defaults to edition 2015; the crate is 2024. If
  ever formatting, use `cargo fmt` or `rustfmt --edition 2024`.

**Future cleanup session (Mike, 2026-05-16):** a dedicated session
to add a `rustfmt.toml`, reformat the whole crate in one pass, then
add a fmt hook (a git pre-commit `cargo fmt --check`) to keep it
clean. Its own session on purpose — a whole-crate reformat is a huge
diff that would collide with any in-flight kernel work, so it waits
for a quiet point, not interleaved with the kernel arc.

## Future work — generic `ComputeEncoder`

Mike's note: `ComputeEncoder` could become backend-generic (an
associated-type seam, like the `Backend`/`Op` trait already is) when
CUDA/CoreML backends land. Not built speculatively — the
Metal-concrete version is the honest scope today. Revisit at the
CUDA/CoreML backend.

## RESUME POINT — the kernel/prefill arc

Plan-of-record (from `qwen_graph_mode_session12_landed.md`): prefill
is **~20× behind llama.cpp**, **GPU-bound** (CPU 5-10%). The arc is
Metal kernel-throughput work. Design conversation started this
session; methodology agreed:

1. **GPU capture before optimizing.** Env-gated `MTLCaptureManager`
   scope around one prefill → `.gputrace` → per-kernel GPU time +
   ALU/bandwidth/occupancy classification.
2. **Hypothesis to confirm (not assume):** the batched 4-bit-dequant
   matmul kernels likely do not use Apple `simdgroup_matrix` 8×8
   instructions — the standard prefill-GEMM win, and what ggml-metal
   (llama.cpp) leans on. Confirm the dominant kernel + its bound from
   the capture first.
3. **One kernel per change**, diff-oracle gated (`batched_diff_oracle`
   / `graph_diff_oracle`, cosine ≥ 0.9999), then bench (n≥3, reboot,
   high-perf — `feedback_bench_discipline`). Re-capture after each win.

Open questions put to Mike (await answers before plan-mode):
- Is the 20× clean apples-to-apples (same a3b/prompt/machine; is
  llama.cpp streaming experts or resident)?
- Capture tooling: programmatic `MTLCaptureManager` (preferred) vs
  Instruments Metal System Trace.
- Fixed benchmark workload: 992-tok / 15.7k / 40-60k Agora range.
