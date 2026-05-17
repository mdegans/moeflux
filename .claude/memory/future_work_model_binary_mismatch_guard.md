# Future work — fail fast on binary ↔ weights mismatch

A moeflux binary is feature-gated to one model (`moeflux-model-*` →
compile-time `VARIANT`). Point an a17b-built `blallama` at a3b weights
(trivially easy via `bench.py --no-build` after a different `--model`
build) and it loads fine, logs `session_ready`, then **panics deep in
prefill**: `prefill failed in CandidatePredictor::new: InitFailed`
(drama_llama `predictor.rs:341` `.expect()`s the swallowed moeflux
error). Cost a real debugging detour on 2026-05-17.

Goal: detect the mismatch and reject with a descriptive message
instead of an opaque panic.

It is a 3-part fix, not a one-liner:

1. **moeflux — detect + descriptive error.** In `RsCtx::open`
   (`riir/mod.rs:336`), right after `WeightFile::open` succeeds, probe
   the manifest against `VARIANT`. Robust signals: max `model.layers.
   {N}` index in `wf.iter()` vs `VARIANT.num_layers`, and a hidden-dim
   tensor's shape vs `VARIANT.hidden_dim` (confirm an always-present
   tensor name from `LayerWeightCache::build_all`). On mismatch return
   a new `RsError::ModelMismatch { expected: &'static str /* VARIANT.
   name */, detail: String }` — message should name the binary's
   variant, what was found, and the remedy ("rebuild with --features
   moeflux-model-… or point at the matching model dir").

2. **drama_llama — stop panicking.** `predictor.rs:341` `.expect()`s
   prefill failure → panic. It must *propagate* the error so the
   request fails cleanly. Trace `CandidatePredictor::new` /
   `TokenPredictor::new` prefill error handling.

3. **blallama — map it.** `map_session_err` turns the error into an
   HTTP response; the new variant gets a 4xx with the descriptive
   message so the caller sees why.

Verify: build the a3b binary, point it at the a17b model dir (and
vice versa) — expect a clean descriptive error, no panic.

Good next-session warm-up before the `gated_delta_net_step` design.
