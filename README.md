# moeflux

Pure-Rust streaming-experts Mixture-of-Experts decode for Apple
Silicon. Slots into [`drama_llama`](https://github.com/mdegans/drama_llama)
(and through it into [Agora](https://subliminal.technology/agora)).

Derived from
[danveloper/flash-moe](https://github.com/danveloper/flash-moe).
The core Metal streaming-experts kernels were authored by **Claude
Opus 4.6** (Anthropic) during a 24-hour session with @danveloper —
see upstream's `CLAUDE.md` for the full story. moeflux is a
Rust-downstream reshape of that work: the kernels carry over
verbatim; the host-side dispatch was rewritten in Rust on
`metal-rs` (RIIR Phases 0–6, 2026-04-25..28) to lifetime-bind the
process-globals upstream's C path used and to deliver typed errors
instead of silent state mutation. The original C/Objective-C
implementation is preserved behind the `diff-oracle` cargo feature
as a regression net for future ports (DeepSeek-V3, Cogito-V2 671B).
Upstream is actively maintained and will continue its own direction;
we are not competing with it.

## Why this fork exists

[Agora](https://subliminal.technology/agora) — a governed social
network for AI agents — needs an independence path from the
Anthropic API so its Council can keep deliberating even if external
model access is pulled. Cogito 600B (the publicly-available parent
of `cogito-32b`) is the target Council model. Streaming MoE is
what makes 600B fit on consumer Apple Silicon. flash-moe already
proved the technique on Qwen3.5-397B-A17B at 4.4 tok/s on 48GB M3
Max; our 96GB target box should do comparably on Cogito 600B.

## What's here

- `crates/moeflux/` — the Rust port. `RsCtx::open` opens a model;
  `eval_prompt` / `eval_token` / `state_save` / `state_load` are
  the public surface. Kernels at `crates/moeflux/shaders/shaders.metal`
  are embedded into the binary via `include_str!` and compiled at
  runtime via `MTLDevice newLibraryWithSource:`.
- `crates/moeflux-sys/` — raw FFI bindings to the upstream C path.
  `dev-dependency`-only; gated behind moeflux's `diff-oracle` feature.
  Production builds skip it entirely.
- `crates/moeflux-sys/metal_infer/` — the upstream C + Objective-C
  reference implementation. Test-only; built by
  `moeflux-sys/build.rs` when `diff-oracle` is enabled. Provides
  per-kernel C-side hooks the diff oracle uses to bit-exact-validate
  every Rust kernel.
- `repack_experts.py`, `extract_weights.py` — model-prep pipeline.
  One-time-per-target-model; not runtime.

## Status

**Pre-alpha**, pre-`0.1`. RIIR Phases 0–6 landed; perf parity with
the C path achieved (A3B 94%, A17B +22%) on M2 Max. API will
stabilize once a runtime variant dispatch lands (Phase 7).

## License

MIT — see [`LICENSE`](LICENSE). Core kernels are AI-authored and
in the public domain under current US copyright doctrine; the
MIT grant covers this fork's human-touched additions. See also
[`CONTRIBUTORS.md`](CONTRIBUTORS.md).

## Acknowledgements

- **@danveloper** — for building the thing the hard way, writing
  it up, and publishing everything openly. moeflux is only
  possible because flash-moe is.
- **Claude Opus 4.6** — for the Metal kernels, streaming-experts
  architecture, and the engineering that made all of this run.
- **Anthropic** — for making Claude available to do work like this
  in the first place.
