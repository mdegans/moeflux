# Binary ↔ weights mismatch guard — LANDED (blallama tail remains)

A moeflux binary is feature-gated to one model (`moeflux-model-*` →
compile-time `VARIANT`). Pointing it at the wrong weights used to load
cleanly, log `session_ready`, then panic deep in prefill. Cost a real
debugging detour on 2026-05-17.

## Landed 2026-05-18 (session 7 warm-up)

- **moeflux** `65ae254`: `RsCtx::open` probes the manifest right after
  `WeightFile::open` — top `model.layers.{N}` index + 1 vs
  `VARIANT.num_layers`, layer-0 `input_layernorm.weight` length vs
  `VARIANT.hidden_dim`. On mismatch returns the new descriptive
  `RsError::ModelMismatch { expected, detail }`. `probe_variant_match`
  does the I/O; `check_variant_dims` is the pure decision half, 5
  unit tests, variant-agnostic.
- **drama_llama** `f0f8edb`: new `MoefluxError::ModelMismatch(String)`
  + `From<MfError>` arm. Also fixed: the `moeflux` feature now
  depends on `tracing` (was an unrelated pre-existing build break).

## Part 2 (un-panic predictor.rs:341) — turned out MOOT for this bug

The spec assumed the mismatch could only be caught at prefill, so
predictor.rs:341's `.expect()` had to be un-panicked. But Part 1
moved detection to `RsCtx::open` — the error now surfaces at
`MoefluxDecoder::open` / `MoefluxEngine::from_path*`, through the
existing `Result` path, *before any predictor is constructed*. So the
predictor.rs panic is never reached for the mismatch case.

The predictor.rs `.expect()`s (lines 341, 382, 432) are still a
genuine footgun for *other* prefill/step failures (OOM, eval failure)
— but un-panicking them makes `CandidatePredictor::new` /
`TokenPredictor::new` / … fallible, an API-breaking cascade through
the whole predictor stack + Engine entry points. That is a separate,
deliberate API change — NOT part of this guard. Left as future work.

## Remaining: blallama Part 3 (not done — repo not checked out)

blallama's `map_session_err` should give `MoefluxError::ModelMismatch`
a 4xx with the descriptive message. Not done this session — blallama
isn't checked out under `~/Projects`. Until then the error still
propagates non-panic with a readable message (via blallama's generic
error path, likely 500). Low urgency; do it next time blallama is
open.

## Verify (manual, needs both model dirs)

Build the a3b binary, point it at the a17b model dir (and vice
versa) — expect a clean `ModelMismatch` error, no panic.
