# moeflux

Pure-Rust streaming-experts Mixture-of-Experts decode for Apple
Silicon.

moeflux began as a fork of
[danveloper/flash-moe](https://github.com/danveloper/flash-moe) and
has since been rewritten in Rust to the point of being a distinct
codebase. The host-side inference engine is new Rust on `metal-rs`;
the original C/Objective-C functions served as differential oracles
during the rewrite, not as a line-by-line translation source. The
Metal streaming-experts kernels were authored by **Claude Opus 4.6**
(Anthropic) for flash-moe and carry over here. The math is the same
linear algebra every inference engine runs — nothing here is claimed
as novel.

## What's here

- `crates/moeflux/` — the Rust engine. `RsCtx::open` opens a model;
  `eval_prompt` / `eval_token` / `state_save` / `state_load` are the
  public surface. Kernels at `crates/moeflux/shaders/shaders.metal`
  are embedded via `include_str!` and compiled at runtime.
- `scripts/` — the model-prep pipeline (`extract_weights.py`,
  `export_vocab.py`, `export_tokenizer.py`). One-time per target
  model, not runtime; likely future Rust binaries.
- `tools/mlx_reference/` — an MLX-based reference diff harness;
  `crates/moeflux/tests/mlx_regression.rs` regenerates its golden
  fixtures from it.

## Status

**Pre-alpha**, pre-`0.1`. The Rust engine is the only path. The API
will stabilize once runtime model-variant dispatch lands.

## License

MIT — see [`LICENSE`](LICENSE). See also [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

## Acknowledgements

- **@danveloper** — for building the thing the hard way, writing it
  up, and publishing everything openly. moeflux started as flash-moe
  and git history reflects that.
- **Claude Opus 4.6** — for the Metal streaming-experts kernels and
  the architecture that made all of this run.
- **Anthropic** — for making Claude available to do work like this in
  the first place.
