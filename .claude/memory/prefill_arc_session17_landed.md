# Prefill arc — session 17 landed (Phase 4: orchestrator HEAD + TAIL → graph mode)

2026-05-19. 5 commits on moeflux `main`, every one green. Plan-of-
record: `prefill_arc_fullattn_migration_plan.md` Phase 4 done.
Continues session 16 (Phase 3: shared MoE block).

## What landed

| Commit | What |
|--------|------|
| `17c198a` | `Op::EmbedGatherNTokens` — GPU 4-bit embedding-gather kernel + Op + diff test |
| `4c37f3a` | deleted dead `Op::LmHead` — `encode_op` now has **zero `todo!()`s** |
| `4cbf14f` | orchestrator HEAD swap — embedding gathers GPU-side into `hidden_a`, no host stack |
| `ab40bb0` | orchestrator TAIL swap — GPU final norm + lm_head, no 67 MB readback round-trip |
| `a534a69` | cleanup-tracker memory note (`future_work_cleanup_arc.md`) |

## End state

The graph compiler is **complete**: every `Op` variant wired on
both backends, `encode_op` has no `todo!()` arms left, and the
orchestrator is uniformly graph-mode end-to-end: GPU embed →
layer loop → GPU final norm + lm_head. The `hidden_stack_host`
and `hidden_final_host` 67 MB host buffers are gone from the
batched path; only the N token ids go in (N×4 bytes) and the
vocab-sized logits come out (~600 KB). Two new `unsafe slice::
from_raw_parts` reinterprets that would have followed went
inline-replaced with `bytemuck::cast_slice` from the start.

New run-lifetime `HeadTailScratch { token_ids, logits }` —
persistent BufIds, allocated next to `HiddenDoubleBuffer` in
`ensure_linear_resources`.

## Decisions banked

- **`Op::LmHead` deleted, not extended.** Discovery in design: the
  variant fused norm + matvec, but the Phase 4 split into separate
  norm + matvec made `Op::LmHead` exactly `RmsNorm` + `MatvecNTokens`
  — both already wired. Decision C: delete the redundant variant
  (the enum doc's own "don't add unused variants" cuts that way).
  `GpuLmHead` struct and `io::lm_head::lm_head_cpu` stay (per-token
  MLA path still uses them).
- **`hidden_b` post-loop local, not the field.** Under odd
  `num_layers` `hidden_double_buffer.hidden_b` aliases the final
  hidden state still being read. The norm target is the local
  `hidden_b_id` after the loop's last `mem::swap`. Documented in-
  code; the Plan agent caught the would-be parity bug.
- **`embed_lookup_at` factored out** as the variant-agnostic core
  of `embed_lookup` — `hidden_dim` from `out.len()`, no `VARIANT`
  shape lock. Lets the CpuBackend arm and small synthetic test
  fixtures reuse the bit-exact arithmetic; `embed_lookup` stays
  the by-name VARIANT-checking wrapper.
- **Pre-existing stale `one_of_each` op-count asserts** (15→19)
  fixed as part of commit 2 — `--lib` unit tests aren't in the
  canary battery, so the drift went unnoticed.
- **`bench.py` / `profile.py` `--no-build` flag removed** (drama_
  llama tooling, gitignored). Wrong-model footgun: a `--no-build`
  against a binary compiled for a different feature silently
  benchmarks the wrong thing. cargo no-op-rebuilds in ~1 s.
- **Always-build is in `benchmarks.md`** protocol now, with a
  machine-settle note (page-cache + CPU contention).

## Findings (Phase 5 measurement, partial)

Reboot was warm (not cold — Mike, retroactively). SDPA heisenbug
reproducer (issue #1) was **clean 100/100**, so machine state was
plausibly OK on the SDPA path. Bench (settled machine, load avg
~1.5, high-power mode):

| | Phase 4 warm | `a236a0e` warm | delta |
|---|---|---|---|
| short 992 | 231.4, 231.8 | 254.0, 254.8 | **−9 %** |
| long 15.7 k | 318.5, 318.7 | 253.6, 256.6 | **+25 %** |

Both deltas reproduce across two clean bench runs. **Different
signs on the two prompts** — the divergence is real, not noise.

### Short-prompt regression — leading theory

`Op::MatvecNTokens` 4-bit dispatches MLX's `QmmCall` (`affine_
qmm_t`) per `gpu/mod.rs:716-758`. The OLD `GpuLmHead::forward`
dispatched **`dequant_matvec_4bit_v3`** — the dedicated per-row-
tile matvec, designed for M=1. MLX qmm comment says "~12× the
old matvec **at prefill shapes**" (i.e. high M). At M=1 (lm_head
with one row, vocab-sized output), the dedicated matvec is
likely faster than the GEMM tuned for batched M.

**Fix (next session):** route `MatvecNTokens` 4-bit `n_tokens
== 1` through `encode_matvec_n_tokens` / `dequant_matvec_4bit_
v3` instead of `QmmCall`. ~3-line change in the 4-bit branch.
Should recover the short prompt without touching long.

### Long-prompt crash — backtrace captured (NOT Phase 4 code)

2/5 single-iter `bench.py` attempts crashed on the long prompt
with identical panic:

```
thread 'tokio-rt-worker' panicked at moeflux/.../backend/gpu/metal.rs:267:13:
Metal command buffer 'graph_full_attn' completed with error status;
rerun with MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 for the fault detail
```

Stack: `commit_and_wait_labeled` → `MetalBackend::execute` →
**`batched_full_attn_layer_forward`** → `step_internal`. The
crash is in the **full-attn layer's `graph1` cmdbuf** — Phase
2/3 code, not Phase 4 path. The `[a236a0e]` baseline ran 3/3
long iters clean, so we may have made the rate worse, but the
failing code is pre-existing.

**Pattern:** only the long prompt (n=8192 in chunk 1 full-attn),
only early attempts in each `bench.py` invocation, subsequent
iters always succeed at ~318 tok/s. Leading theory: **macOS GPU
watchdog timeout on a cold-Metal-pipeline submission** at
n=8192. Warm cache → fast → no watchdog. Short prompt (n=992)
fits inside the watchdog window even cold.

**Diagnostic gap:** `metal.rs:267` panics without surfacing
`cmdbuf.error()`'s `.reason` — could be timeout, page fault,
OOM, or something else. The fix-first task: surface the reason.

### Other notes

- **Crash is not SDPA heisenbug.** Different symptom (cmdbuf
  error status vs wrong output), different cmdbuf
  (`graph_full_attn` vs `sdpa_causal_flash` parallel test).
- **`feedback-reboot-on-gpu-weirdness` updated** with cold-vs-
  warm reboot caveat (per Mike retroactively: session 13's fix
  was a cold boot; warm may not clear Apple/Metal state).
- **+25 % on long is real when it works.** Couldn't fully
  explain the *magnitude* — Phase 4's head/tail savings shouldn't
  buy 12 s on a 62 s run by themselves. Possible the crash chase
  surfaces a related state effect.

## Next session — Session 18 (Phase 5 follow-up)

Sharp entry. Two small fixes + a clean bench.

**A — surface `cmdbuf.error()` at `metal.rs:267`.** Currently
panics with just "completed with error status". Change to include
the `MTLCommandBuffer` error's `.localizedDescription` / reason
code in the panic message. Tiny patch, no behaviour change for
success paths. Then re-run the long bench 5× and capture which
reason it is (timeout / page fault / OOM / etc). Decides the fix
shape:
- *timeout* → pre-warm Metal pipelines on first call, or split
  the n=8192 full-attn cmdbuf into smaller commits.
- *page fault* → memory aliasing or BufId lifetime bug to chase.
- *OOM* → expected at high context; document.

**B — route 4-bit `MatvecNTokens` at `n_tokens == 1` to `_v3`.**
`gpu/mod.rs:716-758`, the 4-bit branch. Currently always uses
`QmmCall`; at `n_tokens == 1` switch to `encode_matvec_n_tokens`
(which dispatches `dequant_matvec_4bit_v3`). Re-bench short and
verify the −9 % closes. Long should stay at +25 %.

**C — clean `benchmarks.md` row.** After A and B, reboot **cold**
(per the updated `feedback-reboot-on-gpu-weirdness`), settle,
bench n=3 on both prompts, record the Phase 4 + fixes row.

**Optional D — cleanup-arc items** from `future_work_cleanup_arc
.md` if time/energy: generic `<T: Pod>` pool IO, `gpu_norm.rs`
rename, the `from_raw_parts` reinterpret sweep.

## Verification

`cargo build -p moeflux --features model-qwen3-6-35b-a3b`.
Canary at every boundary: `cargo test -p moeflux --release
--features model-qwen3-6-35b-a3b --test <suite> -- --ignored
--test-threads=1` for graph_diff_oracle (18), batched_diff_
oracle (23), diff_oracle (12), checkpoint_restore (7). All
green at session close.

Reproducer for the still-open crash:

```
for i in 1 2 3 4 5; do
  RUST_BACKTRACE=full ./bench.py --model a3b -n 1 \
    --prompt-file prefill_prompt_long.txt --max-tokens 1
  cp /tmp/bench_blallama.log /tmp/bench_iter_${i}.log
done
```

~40 % per-attempt crash rate today. Look for `panicked at .../
metal.rs:267` in the preserved iter logs.
