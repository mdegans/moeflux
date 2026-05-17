# Known-broken: MOEFLUX_PROFILE_PER_OP crashes prefill

2026-05-17. Running blallama with `MOEFLUX_PROFILE_PER_OP=1` panics
during prefill init: `prefill failed in CandidatePredictor::new:
InitFailed` (drama_llama `src/predictor.rs:341`; `InitFailed` is a
moeflux error). Without the env, prefill is fine (benched 231 tok/s
a3b).

The per-op path (`MetalBackend`, `gpu/mod.rs` — `if self.profile_per_op
{ commit each Op as its own cmdbuf }`) almost certainly rotted during
the graph-mode arc (sessions 6–12): `commit_plan` lifetime-coloring
and the two-phase graph orchestration assume a graph is committed as a
unit, not Op-by-Op. Not yet bisected to a commit.

Not a P7 regression in the functional sense — P7's path is verified
green independently (canary 1.0 across {mmap,pread}×{gather on,off}).
This is dev-only instrumentation.

Implication for the prefill arc: don't rely on this tool. A real GPU
capture (Instruments Metal System Trace, or `MTLCaptureManager` →
Xcode) is the better instrument anyway — true per-kernel GPU timeline,
no commit-fusion forfeit. Fix per-op only if a cheap per-Op proportion
number is wanted later; otherwise consider deleting it.
