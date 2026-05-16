# Cleanup arc — session 2 landed

2026-05-16. Continues `cleanup_arc_session1_landed.md`. Master plan:
`~/.claude/plans/stateful-booping-acorn.md`; this session's plan:
`~/.claude/plans/melodic-herding-river.md`.

User chose full remaining scope (multi-session). Re-verified the
remaining phases against real code first — **Phase 7, 6, 5b all
dropped** (Phase 7's 600-line fns are already graph-decomposed or
have commit boundaries crossing their banners; 5b already specialised;
6 abandoned in session 1).

## Landed — 7 code/structural commits on moeflux `main` + memory

- `af963a9` **Phase 8** — module reorg. Flat 38-file `src/riir/` →
  `backend/{,cpu/,gpu/}` · `attn/` · `moe/` · `io/` · `snapshot/`.
  One atomic commit, pure `git mv` + import rewrite. Tag `cleanup-phase8`.
- `199db66` **Phase 5a** — `full_kv_mut` helper (byte-identical block
  ×3 in `mod.rs` deduped).
- `44f9c85` **Phase 4b** — `pipeline_bundle!` macro, 9 of 10 bundles
  converted (−91 lines). `DenseMlpPipelines` left hand-rolled
  (heterogeneous). Tag `cleanup-phase4b`.
- `7e38c5d` **C oracle retired from the `moeflux` crate** — see below.
- memory commits: `634c35c`, `0e98c49`, `02ac764` (C-oracle finding),
  this file.

**Phase 5c skipped** — needs an 8-arg helper; relocates complexity,
doesn't reduce it. Prefetch is perf-only, so site drift isn't a
correctness risk.

## The C oracle — diagnosed, and retirement started

Investigation (full detail: `future_work_c_oracle_broken_post_riir.md`):

- The C diff-oracle is **broken and non-deterministic** — end-to-end C
  paths emit zeros/NaN, the failing-test set drifts run-to-run (11→13).
  Traced to the RIIR cutover (`2039619`, 2026-04-28): the `#[ignore]`s
  on the C-comparison tests were applied *in that commit*. The C rot is
  not recent drift — it coincides with the RIIR hollowing out the C
  host dispatch.
- **Our Rust is not broken.** Proven: three independent Rust-internal
  oracles all green — `graph_diff_oracle` (CpuBackend vs MetalBackend),
  `batched_diff_oracle` (batched vs per-token), and the Rust per-token
  oracle. `state_round_trip_rust` (Rust save→load→eval) passes — Rust's
  snapshot code is correct. The failing tests show C emitting zero
  vectors while Rust emits plausible token IDs.
- "Port the C tests to a CpuBackend oracle" was considered and
  **rejected as circular** — `CpuBackend`'s Op arms *call the same*
  `*_cpu` primitive functions the diff_oracle primitive tests exercise
  (`conv1d_step`, `gated_delta_recurrence_supplied`, etc.). The genuine
  cross-check (MetalBackend vs CpuBackend, two independent impls)
  already exists in `graph_diff_oracle`.

`7e38c5d` removed the C from the **`moeflux` crate**: 34 C-vs-Rust test
fns + `CBackend` + `tests/common/c_backend.rs` deleted (12 pure-Rust
tests remain in `diff_oracle.rs`); `moeflux-sys` dev-dep + feature
forwards dropped; stale C doc-comments in `src/` fixed.

## RESUME POINT — two threads, in order

### Thread 1 (first): C-removal close-out — ~3 small commits

The `moeflux` crate is C-free, but `crates/moeflux-sys/` still exists
as an orphan workspace member. Finishing it:

- **moeflux-sys is entangled — it is NOT a clean delete.** Besides the
  C (`infer.m`, headers, `Makefile`, binaries, Rust bindings — delete
  those), it holds **needed Python tooling**:
  - `metal_infer/{extract_weights,export_vocab,export_tokenizer}.py` —
    the artifact-export pipeline (produces `model_weights.bin` /
    `vocab.bin` that the **Rust** engine consumes). Relocate to
    top-level `scripts/`. Light docstring touch-up ("the C inference
    engine" → "moeflux").
  - `metal_infer/tests/mlx_reference/` — an **MLX-based** reference
    harness (`generate_goldens.py` et al.). `crates/moeflux/tests/
    mlx_regression.rs` depends on it to regenerate its golden fixtures.
    MLX is a genuinely *independent* oracle — arguably the successor to
    the dead C. Relocate somewhere live (e.g. `tools/mlx_reference/`),
    do not lose it.
- Then delete the C remnants of `crates/moeflux-sys/`, remove it from
  the workspace `Cargo.toml` (`members` line 4, `[workspace.
  dependencies]` line 20).
- **Docs:** update `README.md`, `CLAUDE.md`, `NOTES.md`,
  `docs/model_variants.md` — drop C references, frame moeflux as a
  Rust port. Update doc paths to the relocated scripts.
- **LICENSE / README / `danveloper` (Mike's steer, 2026-05-16):**
  remove the `danveloper` acknowledgement *from the LICENSE* — the
  LICENSE stays purely the license. Instead: add a **port note at the
  top of `README.md`** (moeflux is a Rust port of danveloper's
  original work) and put the **`danveloper` acknowledgement at the
  bottom of `README.md`**. Read current LICENSE/README wording and
  show Mike before committing.

### Thread 2: Phase 4a — encoder builder (the last cleanup phase)

New `ComputeEncoder` builder in `backend/gpu/encoder.rs` (the file 4b
created — extend it). **Design refinement from this session:** 42
`new_compute_command_encoder()` sites, 2–3 of which encode MULTIPLE
dispatches per encoder (`moe/expert_forward.rs`, `attn/gpu_attn.rs`).
The builder must NOT auto-`end_encoding()` after `.dispatch()` — make
`.dispatch()` re-callable, end on explicit `.end()` / `Drop`. Headline
risk of the arc (a wrong `set_buffer` index is silent GPU corruption):
one file per commit, full gate after each. ~6–7 commits, a session of
its own.

Then Phase 9: `n≥3` post-reboot bench vs `pre-cleanup-baseline`.

## Verification gate (corrected — the plan doc had it wrong)

Compile gate: per-feature (`cargo build --features model-{qwen3-5-a17b,
qwen3-6-35b-a3b,cogito-v2-671b}`), zero `moeflux`-lib warnings.
Behavioral gate: **`cargo test --no-fail-fast --features
model-qwen3-6-35b-a3b -- --include-ignored`** (a3b, NOT a17b — the
oracle suites load the a3b model; `--no-fail-fast` so one
model-mismatched binary doesn't mask the rest). `checkpoint_restore.rs`
is a3b-only and fails under a17b — expected. "Green" = all Rust-path
tests pass. Post-C-removal there should be no C-comparison failures
left to explain away.
