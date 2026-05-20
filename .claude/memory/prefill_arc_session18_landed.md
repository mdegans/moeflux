# Prefill arc — session 18 landed (Phase 5 follow-up: diagnostic A + matvec M1 env-gate)

2026-05-20. 2 commits on moeflux `main`. Plan-of-record: the
session-17 sharp-entry — surface `cmdbuf.error()`, fix the
short-prompt -9%, clean bench. Outcome differs from plan; see
findings.

## What landed

| Commit | What |
|--------|------|
| `bca910c` | metal: surface NSError detail (code + localizedDescription) on cmdbuf Error status — names the failure class (Timeout / PageFault / OutOfMemory / …) in the panic |
| `e9d9b9c` | matvec: route 4-bit `MatvecNTokens` at `n_tokens == 1` to `dequant_matvec_4bit_v3` (OLD `GpuLmHead` kernel), env-gated by `MOEFLUX_MATVEC_M1_V3` |

## End state

A's diagnostic is in place: any future `cmdbuf.commit_and_wait`
panic now reports e.g. `Metal command buffer 'graph_full_attn'
completed with error status: Timeout(2): … `. B preserves the
OLD `GpuLmHead::forward` dispatch (`_v3` per-row-tile matvec)
for any future single-row 4-bit matvec callsite, with a
runtime A/B knob.

New direct workspace dep: `objc = "0.2"` — already a transitive
dep of `metal-rs`, promoted to direct so we can use `msg_send!`
for the NSError accessor that metal-rs 0.32 doesn't expose
(`MTLCommandBufferError` is defined but has no getter). No
Cargo.lock churn.

## Decisions banked

- **Plan A first, B after classification.** Session 17 said
  classify-the-crash before fixing — A surfaces the class, B
  is independent. Ran A's repro 25 times across three
  conditions before deciding what B's fix shape should be.
- **B ships as future-proofing, not a perf fix.** Session 17
  measured a -9 % short-prompt regression and hypothesised
  QmmCall vs `_v3` at M=1 as the cause. Today's A/B
  (Phase 4 alone vs Phase 4 + B at the same machine state)
  came back 238.5 vs 236.3 — bench-noise, ~2 tok/s stdev.
  Hypothesis unfalsified-in-either-direction. Commit anyway
  because the OLD `GpuLmHead` semantics are the architectural
  default and `_v3` is what the kernel was designed for at M=1.
- **Runtime env gate `MOEFLUX_MATVEC_M1_V3`.** Default ON
  (use `_v3` at M=1), set `=0` to force QmmCall — A/B without
  rebuild. Pattern mirrors `MOEFLUX_MOE_GATHER`.
- **Did not chase a cold-boot bench.** Mike on mobile, can't
  reboot. The cold-boot row for benchmarks.md is deferred to
  the next-session opening.
- **Did not commit the per-Op profile breakdown** to a memo
  yet — the data is in this memo's Findings, future-session
  refs can grep here.

## Findings (Phase 5 measurement, this session)

### A — cmdbuf crash repro: 25/25 CLEAN across three regimes

Goal was classifying the session-17 cmdbuf-error panic (40 %
empirical rate). Today the crash did not reproduce at all.

| Condition | Iters | Crashes | Prefill tok/s (long 15.7 k) |
|---|---|---|---|
| Cold-start (first 5 of session) | 5 | 0 | 288.45 cold; 316-318 warm |
| Continued-warm | 15 | 0 | 313-319 |
| Under 11/12-core CPU stress | 5 | 0 | 311-315 |

P(0 in 5 \| 40 % rate) ≈ 7.8 %; P(0 in 20 \| 40 % rate) ≈
0.004 %. The session-17 rate cannot reflect today's underlying
probability. Either we're in a different machine state, or the
true rate is ≲ 5 % and session 17 got hit twice by chance
(P ≈ 2 %). CPU contention barely moved prefill tok/s (~1 %
lower under stress) — confirms prefill is GPU-bound and the
bench thread is getting scheduled even with 11/12 cores
spinning.

A's diagnostic is armed for the next occurrence; no further
action this session.

### B — short-prompt regression: didn't reproduce; B is perf-neutral

Session 17 reported (warm bench):

| | Phase 4 warm | `a236a0e` warm | delta |
|---|---|---|---|
| short 992 | 231.4, 231.8 | 254.0, 254.8 | **-9 %** |
| long 15.7 k | 318.5, 318.7 | 253.6, 256.6 | **+25 %** |

Session 18 head-to-head A/B at the same machine state, short
prompt 992 tokens:

| | mean | stdev | iters |
|---|---|---|---|
| Phase 4 alone (B reverted) | 238.5 | ~2 | 3 |
| **Phase 4 + B (env default)** | **236.3** | ~2 | 3 |

B is bench noise. **The -9 % regression itself does not
reproduce**: Phase 4 alone today is 238, not session 17's 231.
Long prompt today reproduces the +25 % cleanly (~317 tok/s).

### Per-Op profile rules out Phase 4 Ops as the regression source

`MOEFLUX_PROFILE_PER_OP=1` single-iter, short prompt — Phase 4
Ops are negligible:

```
embed_gather   10.69 ms   0.41 %
lm_head         1.68 ms   0.06 %
final_norm      0.20 ms   0.01 %
                          ─────
                          0.48 %  total
```

Top of the cost table (kv_append 34.9 %, moe.permute_fuse 20.2 %,
gated_delta_net_step 11.1 %, sdpa 7.1 %) is all pre-existing
work, untouched by Phase 4. Whatever produced session 17's -9 %
lives in machine-state, not in any of Phase 4's three new Ops.

### Other notes

- **Test gap surfaced.** `graph_diff_oracle` doesn't currently
  cover `Op::MatvecNTokens` (4-bit or 8-bit) through both
  backends, despite the module doc-comment claiming it does.
  `batched_diff_oracle` tests `encode_matvec_n_tokens` and the
  MLX qmm gate directly (bypassing the Op dispatch), so the
  matvec kernels are well-covered, but a graph-level
  `MatvecNTokens` arm is not. Worth a test addition next time
  matvec needs surgery.
- **Per-Op profile numbers are dilated.** Profile mode commits
  each Op as its own cmdbuf, losing the S7-1a commit fusion
  and adding ~580 cmdbuf round-trips on the short prompt.
  Profile-mode total wall was 4.31 s vs default 4.20 s
  (~3 % overhead, low) — useful for proportion analysis,
  never for absolute bench.

## Next session — Session 19

Three small things, in order.

### A — cold-boot bench + benchmarks.md row

The deferred item from session-17 protocol. Cold boot per
`feedback_reboot_on_gpu_weirdness`, settle (no other apps,
load avg < 1), high-power mode, then:

```
./bench.py --model a3b -n 3 --prompt-file prefill_prompt.txt --max-tokens 1
./bench.py --model a3b -n 3 --prompt-file prefill_prompt_long.txt --max-tokens 1
```

Record both rows in `drama_llama/benchmarks.md`. With B's env
gate landed, a 4th cell (`MOEFLUX_MATVEC_M1_V3=0` on the short
prompt) settles whether B is perf-positive, perf-neutral, or
perf-negative at a cold-boot starting state — without needing
another commit/revert dance.

### B — investigate the session-17 vs session-18 short-prompt gap

Today's Phase 4 alone is 238 tok/s; session-17 measured 231.
Both warm-bench, ~same protocol. 3 % gap, 2σ-ish — easily
machine state, but worth one more controlled look on a cold
boot. If the gap holds cold, look at what changed *between*
session-17's reboot and today's idle-overnight state: page
cache, Metal shader cache (`~/Library/Caches/com.apple.metal*`),
GPU driver state. The cargo-clean experiment (force-invalidate
per-binary shader cache) is the cheapest single probe — was
discussed this session, deferred for time.

### C — close test gap: `graph_diff_oracle` `MatvecNTokens` coverage

Add `graph_metal_matches_cpu_matvec_n_tokens` (4-bit + 8-bit,
n_tokens ∈ {1, 4, 16}) so the Op dispatch (not just the
underlying kernels) is exercised against the CPU oracle. Would
have caught a real dispatch-level regression today by failing
loud; we got lucky that the QmmCall vs `_v3` swap is cosine-1.0
equivalent.

## Open known-unknowns

- **What machine state triggers the cmdbuf-error panic.** A's
  diagnostic will tell us the class when it next fires. Likely
  one of: cold Metal shader cache compile (Timeout/Internal),
  pagefault flood from cold mmap (PageFault), or a real latent
  race condition in the graph_full_attn cmdbuf (Internal).
- **The session-17 -9 %**. Most likely machine-state, possibly
  bench noise (n=2 there). One more cold-boot bench settles it.
- **Whether B is perf-positive on any hardware/prompt.** Today
  it's perf-neutral on M2 Max + 992-token prompt. The env gate
  makes future bench cells cheap.

## Verification

`cargo build --release -p moeflux --features model-qwen3-6-35b-a3b`
green for both commits.

Canary battery (60/60, post-B, env default):
- `graph_diff_oracle` 18/18
- `batched_diff_oracle` 23/23
- `diff_oracle` 12/12
- `checkpoint_restore` 7/7

Env-gate sanity (3× single-iter benches, default / `=0` /
default): all 237-241 tok/s, no crashes, distinguishable only
by the dispatched kernel.
