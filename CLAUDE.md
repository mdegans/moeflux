# CLAUDE.md

Guidance for Claude Code working in the moeflux repository.

## What moeflux is

moeflux is a pure-Rust streaming-experts Mixture-of-Experts inference
engine for Apple Silicon. It began as a fork of danveloper's flash-moe
and has been rewritten in Rust on `metal-rs`; the Metal kernels were
authored by Claude Opus 4.6 for flash-moe and carry over. See
`README.md` for provenance.

moeflux is consumed by
[`drama_llama`](https://github.com/mdegans/drama_llama), which is
where this work is usually launched from — drama_llama is the
downstream consumer and is path-pointed at this crate. Treat
drama_llama's needs as the API's north star.

## Build & test

macOS only; on other targets the crate exposes no symbols. Exactly
one model-variant feature must be selected:

```bash
# Build, per variant
cargo build --features model-qwen3-6-35b-a3b
cargo build --features model-qwen3-5-a17b
cargo build --features model-cogito-v2-671b

# Behavioral gate — the oracle suites load the a3b model
cargo test --no-fail-fast --features model-qwen3-6-35b-a3b -- --include-ignored
```

`--no-fail-fast` so one model-mismatched test binary doesn't mask the
rest. `checkpoint_restore.rs` is a3b-only and fails under a17b — that
is expected. `resuming_prefill_after_seq_rm_matches_full_prefill` is
a known failure (linear-attn recurrence state is not
position-truncatable — see the fn-level comment); `--skip` it.
"Green" means every other Rust-path test passes.

`cargo fmt` is *not* a gate — the crate predates machine formatting
and is ~600 hunks off default rustfmt with no `rustfmt.toml`. Match
surrounding style by eye; don't bulk-reformat.

## Architecture

- `crates/moeflux/` — the only crate. `RsCtx::open` opens a model;
  `eval_prompt` / `eval_token` / `state_save` / `state_load` are the
  public surface (`RsCtx` is re-exported as `Ctx`).
- `src/riir/` — the engine. `backend/{cpu,gpu}/` is the `Backend`
  trait with CPU and Metal implementations; `attn/`, `moe/`, `io/`
  (weight streaming), `snapshot/` (state save/load) hold the rest.
- `src/riir/variants.rs` — compile-time per-variant shape constants,
  gated by the model features. See `docs/model_variants.md`.
- `crates/moeflux/shaders/shaders.metal` — the Metal kernels, embedded
  via `include_str!` and compiled at runtime.

## Conventions

- **Land work directly on `main`** while drama_llama is the sole
  consumer. Revisit when external users appear.
- **Model shape is compile-time.** One `model-*` feature at a time;
  runtime variant dispatch is future work.
- **The C oracle is retired.** moeflux is C-free; correctness is
  cross-checked Rust-internally — `graph_diff_oracle` (CpuBackend vs
  MetalBackend), `batched_diff_oracle` (batched vs per-token), and
  `mlx_regression` (vs an MLX reference). See
  `.claude/memory/future_work_c_oracle_broken_post_riir.md` for why.

## Durable context

Session-spanning notes live in `.claude/memory/` — graph-mode arc
landings, the cleanup arc, future-work items. Read the most recent
`*_landed.md` when picking up.
