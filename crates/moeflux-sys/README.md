# moeflux-sys

Raw FFI bindings to [moeflux](https://github.com/mdegans/moeflux)'s
streaming-experts MoE Metal backend.

Prefer the safe wrapper in the [`moeflux`](../moeflux) crate unless
you specifically need unsafe raw access.

## Platform

**macOS only.** The backend is Metal-based. On non-Mac targets the
build script is a no-op and this crate exposes no symbols.

## Model variant selection

Exactly one Cargo feature must be enabled to pick the compile-time
model shape:

- `model-qwen3-5-a17b` — Qwen3.5-397B-A17B-4bit (upstream target)
- `model-qwen3-6-35b-a3b` — Qwen3.6-35B-A3B-4bit (smaller)

Selecting zero or more than one fails at build time. See
[`docs/model_variants.md`](../../docs/model_variants.md) for the
full parameter table and how to add a new one.
