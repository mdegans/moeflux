# Contributors

moeflux is a downstream reshape of
[danveloper/flash-moe](https://github.com/danveloper/flash-moe),
oriented toward use as a decode backend for
[`drama_llama`](https://github.com/mdegans/drama_llama) (and, through
it, for the [Agora](https://subliminal.technology/agora) agent
network).

Per the AI-authorship disclosure in upstream's `CLAUDE.md` — "an AI
and a human built this in 24 hours" — the core Metal streaming-experts
kernels are model-authored and, under current US copyright doctrine,
public domain. The credits below reflect the hands that built each
part rather than any copyright claim.

## Core implementation

- **Claude Opus 4.6** (Anthropic) — Metal compute shaders, 4-bit and
  2-bit dequantized matmul kernels, GatedDeltaNet linear-attention
  pipeline, full-attention layers, SSD expert-streaming orchestration,
  the "Trust the OS" caching architecture, and most of `infer.m`.
  Originally authored inside the upstream flash-moe project during a
  24-hour collaborative session with @danveloper.

## Upstream

- **@danveloper** (Dan Woods) — original
  [`flash-moe`](https://github.com/danveloper/flash-moe) project,
  build system, paper, experimental harness, and the human half of
  the original 24-hour session. moeflux exists because flash-moe
  exists; the Rust-downstream angle is our adaptation, not a
  judgment of the original's direction.

## moeflux fork

- **Claude Opus 4.7** (Anthropic) — fork reshape, KV-seq_rm patch,
  stable C API (`mf_*` surface in `moeflux.h`), Rust-facing wrapper
  design, drama_llama integration.
- **@mdegans** (Michael de Gans) — project lead, fork maintainer,
  Agora / drama_llama deployment target.

## Future contributors

PRs welcome once the API stabilizes. Before then, expect churn —
this is pre-`0.1`.
