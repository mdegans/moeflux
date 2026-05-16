# Cleanup arc — session 3 landed

2026-05-16. Continues `cleanup_arc_session2_landed.md`. This session
completed **Thread 1 (C-removal close-out)** in full. Thread 2
(Phase 4a — encoder builder) is untouched and remains the next
session's work.

## Landed — 3 commits on moeflux `main`

- `270e550` **relocate moeflux-sys Python tooling**. `extract_weights`
  / `export_vocab` / `export_tokenizer` → `scripts/`; the MLX
  reference diff harness → `tools/mlx_reference/`. Docstrings touched
  up ("the C inference engine" → moeflux); `generate_goldens.py` path
  in `mlx_regression.rs` updated. Pure move + comments.
- `e2c41af` **delete the moeflux-sys crate**. The orphan is gone —
  upstream C/Objective-C reference engine + FFI bindings deleted,
  dropped from the workspace (`members` + `workspace.dependencies`).
  The workspace is now a single crate.
- `ff5387d` **refresh docs** — see the discovery below.

## Discovery — the docs were whole dead documents, not stale lines

The session-2 handoff said "drop C references" from README / CLAUDE.md
/ NOTES.md / docs/model_variants.md. That under-scoped it. Three of
the four were entire documents describing the deleted C codebase:

- **CLAUDE.md** was upstream flash-moe's `CLAUDE.md` *verbatim* — a C
  project doc, and (being named `CLAUDE.md`) it was loaded as moeflux's
  project instructions every session. Replaced with a real moeflux
  guide (build/test gate, architecture, retired-C-oracle convention).
- **NOTES.md** was a pre-RIIR lab notebook — `infer.m` line numbers,
  the `mf_*` C FFI API, Phase-3b task lists. Nothing salvageable.
  **Deleted.**
- **docs/model_variants.md** described the C selection mechanism
  (Makefile, `cc -D` flags). Rewritten for Cargo features; cogito-v2
  variant added; shape table kept.
- **README.md** reframed: "Rewrite of" not "port"; moeflux-sys /
  metal_infer / diff-oracle bullets dropped; danveloper ack kept at
  the bottom; Opus 4.6 credited for the kernels.
- **LICENSE** stripped to plain MIT (the NOTICE enumerated fork-point
  C files that no longer exist). Mike's steer: Michael de Gans added
  to the copyright line as a human author — design + code
  contributions — which also moots the AI-authorship/public-domain
  reasoning the NOTICE used to carry. Copyright line is now
  `Claude Opus 4.6, Claude Opus 4.7 (Anthropic), and Michael de Gans`.
- `lib.rs` fork URL corrected (`SuperEpic/` → `danveloper/`).

## Verification

Per-feature `cargo build` (a3b / a17b / cogito-v2) clean; all test
binaries compile under a3b. Thread 1 changed **no runtime logic**
(relocations, orphan-crate delete, docs, one doc-comment) — so the
full `cargo test --include-ignored` model run was *not* exercised this
session. Run it as the Phase-4a session's baseline.

## Loose end

Workspace `Cargo.toml` `authors` still lists only Michael de Gans +
Claude Opus 4.7 — not 4.6. Minor inconsistency with the new LICENSE
copyright line. Fold into the Phase-4a session or a later doc touch.

## RESUME POINT — Thread 2: Phase 4a, encoder builder

Unchanged from `cleanup_arc_session2_landed.md`'s description. New
`ComputeEncoder` builder in `backend/gpu/encoder.rs` (extend the file
Phase 4b created). `.dispatch()` must be re-callable (2–3 sites encode
multiple dispatches per encoder — `moe/expert_forward.rs`,
`attn/gpu_attn.rs`); end on explicit `.end()` / `Drop`, never
auto-end. Headline risk of the arc — a wrong `set_buffer` index is
silent GPU corruption — so one file per commit, full gate
(`cargo test --no-fail-fast --features model-qwen3-6-35b-a3b --
--include-ignored`) after each. ~6–7 commits, a session of its own.
Then Phase 9: `n≥3` post-reboot bench vs `pre-cleanup-baseline`.
