# moeflux

Streaming-experts Mixture-of-Experts decode kernels for Apple
Silicon, packaged as a library with a stable C API so it can slot
into [`drama_llama`](https://github.com/mdegans/drama_llama) (and,
through it, into [Agora](https://subliminal.technology/agora)).

Derived from
[danveloper/flash-moe](https://github.com/danveloper/flash-moe).
The core Metal streaming-experts kernels were authored by **Claude
Opus 4.6** (Anthropic) during a 24-hour session with @danveloper —
see upstream's `CLAUDE.md` for the full story. moeflux is a
Rust-downstream reshape of that work: the kernels and Metal
pipeline carry over; the surrounding scaffolding (chat client,
hand-rolled tokenizer, interactive REPL) does not. Upstream is
actively maintained and will continue its own direction; we are
not competing with it.

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

- `metal_infer/` — inference kernels (Metal + C + Accelerate BLAS).
  Inherited from upstream; being reshaped around a stable `mf_*`
  C API in Phase 3 of the drama_llama v0.8.0 work.
- `repack_experts.py`, `extract_weights.py` — model-prep pipeline.
  One-time-per-target-model; not runtime.

## Status

**Pre-alpha**, pre-`0.1`. API is unstable and will break until
the drama_llama side of integration lands and we have the Cogito
probe running. Not yet published to crates.io.

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
