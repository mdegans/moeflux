# Pre-existing: diff_oracle SIGSEGVs under parallel test execution

Discovered session 16 (2026-05-19), prefill-arc phase 3a verification.

## Symptom

`cargo test -p moeflux --release --features model-qwen3-6-35b-a3b
--test diff_oracle -- --ignored` (default multi-threaded harness)
crashes with `signal: 11, SIGSEGV` partway through. The 12 tests each
spin up a full model engine and run GPU forward passes; the harness
runs ~8 concurrently.

## Confirmed pre-existing, not a phase-3a regression

`git stash`'d all phase-3a changes, rebuilt clean HEAD (`dd91850`,
session 15), re-ran the same command — **identical SIGSEGV**. So the
crash predates phase 3a. Session 15's "diff_oracle 12/12" must have
been verified serialized or per-test.

`--test-threads=1` → all 12 pass (75s). `graph_diff_oracle`,
`batched_diff_oracle`, `checkpoint_restore` pass multi-threaded —
they are lighter / fewer concurrent engines.

## Canary protocol

Run `diff_oracle` (and, to be safe, the whole canary battery) with
`-- --ignored --test-threads=1`. Putting multiple `--test` targets in
one `cargo test` invocation also means a diff_oracle crash aborts the
run before later `--test` targets execute — run serialized, or run
each `--test` target in its own invocation.

## Likely cause (unconfirmed)

~8 concurrent model engines on one GPU — Metal resource exhaustion or
Apple driver state under load. Matches the `feedback_reboot_on_gpu_
weirdness` class. Not yet root-caused. Candidate real fix: a global
test mutex serializing engine-creating tests, or cap the harness at
`--test-threads` low for the heavy suites. Low priority — serialized
runs are green and that is a fine canary protocol.
