---
name: prefill-next-session-plan
description: 2026-05-20 — plan-of-record for the next prefill perf session, drafted at the close of the diagnostic arc that produced `gather_qmm_pivot_dead.md` + `llama_cpp_moe_differentiators.md`. Headline phase: MTLResidencySet integration. Sized to land in one focused session.
metadata:
  type: project
---

# Headline

**Integrate `MTLResidencySet` for expert mmap buffers (and probably
KV cache + other persistent GPU resources).** Cheapest of the three
identified differentiators from `llama_cpp_moe_differentiators.md`;
expected to (a) close the variance gap entirely, (b) deliver some
throughput on top, (c) give us a clean baseline to size the
remaining kernel-rewrite work.

# Pre-session inputs

Before starting:

1. **Run n=5 bench** of moeflux a3b on `prefill_prompt_long.txt` if
   not already in the log. Pattern: `bench.py --model a3b
   --prompt-file prefill_prompt_long.txt -n 5 --max-tokens 1`.
   Tells us the warm-state plateau (vs iter-1 287, iter-2 333).
   This number is the "before" for the residency-set A/B.

2. **Read `llama_cpp_moe_differentiators.md`** (this session's
   sister memo). Has the integration sketch + version-gating
   requirements.

3. **Confirm macOS version** is ≥15.0 on dev machine. Required for
   `MTLResidencySet`. (Mike's M2 Max is on Sequoia or later per
   recent session output.)

# Phases

## Phase 1 — objc bindings for MTLResidencySet (~1-2 hours)

In `crates/moeflux-metal`, add a new module `residency_set.rs` with:

- `pub struct ResidencySet { obj: *mut objc::runtime::Object }`
- `Drop` impl that calls `[obj endResidency]` + `release`
- `ResidencySet::new(device: &Device) -> Option<Self>` — version-
  gated; returns `None` on macOS < 15. Use a runtime probe (e.g.
  `NSProcessInfo` `operatingSystemVersion`) since the build target
  may be older than the runtime.
- `add_allocation(&self, buffer: &Buffer)` — calls `[rset
  addAllocation:metal_buffer]`
- `commit(&self)` — calls `[rset commit]`
- `request_residency(&self)` — calls `[rset requestResidency]`

`objc::msg_send!` is the right tool. `metal-rs 0.32.0` exposes
`Buffer::as_ptr() -> *mut MTLBuffer` (or equivalent) so we can pass
the underlying object to the residency-set API. Reference llama.cpp's
`ggml-metal-device.m:1351-1386` for the exact call pattern.

Test: a small synthetic — create a ResidencySet, add a dummy buffer,
commit + requestResidency, and verify the calls don't crash on
macOS 15+.

## Phase 2 — pin expert buffers (~30 min)

In `crates/moeflux/src/riir/io/expert_io.rs` —
`ExpertFiles::attach_to_device`. After the `newBufferWithBytesNoCopy`
call for each mmap'd layer, add the buffer to a shared residency set
and request residency.

Decision: one shared residency set for all expert buffers, or one
per layer? llama.cpp creates one per buffer (line 1480, 1577). For
us, one shared set is simpler and covers all 60 layers' experts in
one `requestResidency` call. Default to shared unless that triggers
a known Metal limit.

## Phase 3 — pin KV cache + persistent GPU buffers (~30 min)

`KvCache::ensure_buffers` in `crates/moeflux/src/riir/...` allocates
the GPU-resident KV cache once at startup (per
`completed_work.md`'s Phase 0b note). Add to the residency set.

Probably also: any other long-lived GPU buffer (BatchedGraphScratch,
moe_buffers). Audit pass — find every `pool.alloc` that survives
across cmdbuf submissions.

## Phase 4 — bench A/B (~15 min reboot + 15 min bench)

Reboot (per `feedback_bench_discipline.md`), then:

```bash
# Baseline (already in log from pre-session inputs)
./bench.py --model a3b --prompt-file prefill_prompt_long.txt \
    -n 5 --max-tokens 1

# With residency sets
./bench.py --model a3b --prompt-file prefill_prompt_long.txt \
    -n 5 --max-tokens 1
```

Expected outcomes (in order of likelihood):

1. **Variance collapse to <1% AND throughput gain**. Best case.
   Residency-set is doing real work both on first-run and steady-
   state. Estimate gap shrink from observed magnitude.

2. **Variance collapse to <1% AND throughput-neutral**. Means our
   page-cache + Metal driver already keeps things resident in
   steady-state; residency-set just makes it deterministic.
   Quality-of-life win; throughput gap is in Differentiator 1
   (kernel).

3. **No measurable change**. Means Metal driver was already doing
   the right thing implicitly on this hardware/OS. The
   `requestResidency` is a hint, not a guarantee — possible the
   driver auto-pins frequently-accessed buffers. Move on to
   Differentiator 1.

## Phase 5 — write the landed memo (~15 min)

`prefill_residency_set_landed.md` with the A/B numbers, what's
pinned, what's not, version-gating coverage, and the residual
gap-to-llama.cpp after the change.

# Out-of-scope for this session

- **Kernel rewrite** (Differentiator 1 from
  `llama_cpp_moe_differentiators.md`). Big. Multi-session.
- **Quant-format change** (Differentiator 3). Bundle with kernel
  rewrite.
- **Task 7** (decode prefetch A/B → ~600 LOC cleanup). Independent;
  can slot before or after residency-set work. Lean toward after,
  to keep the perf signal clean across this session's bench.

# Risk: residency set might do nothing

The `requestResidency` API is documented as a *request*, not a
guarantee. The driver MAY auto-pin frequently-accessed buffers
already. If Phase 4 shows zero throughput change AND zero variance
change, that's the signal — abandon residency-set as a lever and
go directly to the kernel investigation.

This is a real possibility because:
- Apple Silicon's unified memory means there's no "page out" cost
  to GPU memory the way there is on discrete GPUs.
- The driver may already keep recently-used buffers in some
  GPU-friendly mapping.

But the variance signal in our data is too strong to ignore — 16%
iter-1-to-iter-2 jump isn't nothing. So worth the experiment.

# What success looks like (target framing)

- Variance: <1% across 5 iterations (matches llama.cpp).
- Throughput: warm-state ≥ 333 tok/s with the variance-corrected
  baseline. Stretch goal: 400+ tok/s if residency-set is doing
  real throughput work too.
- Cleaner baseline for sizing the kernel-rewrite call. We need to
  know: "with the easy wins banked, how big is the kernel gap?"
  This session answers that.

# Cross-references

- [[gather_qmm_pivot_dead]] — the dead pivot that motivated this
  diagnostic arc.
- [[llama_cpp_moe_differentiators]] — the three differentiators
  identified; this plan attacks #2.
- [[pread_teardown_landed]] — earlier expert-IO teardown; the
  mmap path stays. Residency-set is a pinning layer on top of
  mmap, not a replacement.
- [[feedback_bench_discipline]] — n≥3, high-perf power, reboot
  between revisions.
