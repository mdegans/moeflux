---
name: prefill-residency-set-landed
description: 2026-05-20 — MTLResidencySet integration landed for expert mmap buffers. Variance fixed (16% iter-1-to-iter-2 jump → 2.3% peak-to-peak across n=5). Throughput unchanged (333 → 334 mean). Confirms Differentiator 2 is a variance-fix lever, not a throughput lever; the remaining 2.5× gap is in Differentiator 1 (kernel algorithm).
metadata:
  type: project
---

# What landed

Phase 1 + Phase 2 of `prefill_next_session_plan.md`. Phase 3 (KV
cache + persistent transients) scoped down to deferred — see
"Scope changes" below.

## Phase 1 — `MTLResidencySet` bindings

`crates/moeflux-metal/src/residency_set.rs` — raw `objc::msg_send!`
bindings. Public surface:

```rust
pub struct ResidencySet { /* +1-retained *mut Object */ }
impl ResidencySet {
    pub fn new(device, label, initial_capacity) -> Option<Self>;
    pub fn add_allocation(&self, buffer: &BufferRef);
    pub fn commit(&self);
    pub fn request_residency(&self);
}
impl Drop for ResidencySet { /* end → remove → release */ }
pub fn is_available() -> bool;
```

`is_available()` runtime-probes
`Class::get("MTLResidencySetDescriptor")`; returns `false` on
macOS < 15. `new()` short-circuits to `None` if so.

Smoke test in the same file verifies create + add + commit +
request_residency + Drop on a 4096-byte dummy buffer. Passes on
macOS 26.4.1 (M2 Max dev box).

## Phase 2 — pin expert mmap buffers

`crates/moeflux/src/riir/io/expert_io.rs` — `ExpertFiles` gained
an `Option<ResidencySet>` field declared *first* in the struct so
it drops *before* `mmap_buffers` (endResidency runs while the
registered buffers are still alive).

`attach_to_device(pool)` lazy-inits the rset on first call, calls
`add_allocation` for each per-layer mmap buffer as it's wrapped,
and commits + requestResidency once at end of loop. The second
(idempotent) attach call exits without re-committing because
nothing was added.

`ResidencySet` is marked `Send + Sync` (unsafe impls). Justified:
build on one thread during `attach_to_device`, then read-only for
the lifetime of inference. Required because `ExpertFiles` is
shared across rayon threads in `cogito_moe_gpu.rs:321`.

# Bench

5-iter post-reboot bench, `prefill_prompt_long.txt` (15692 tokens,
single chunk, max-tokens=1):

| iter | prefill tok/s |
|---|---|
| 1 | 335.94 |
| 2 | 337.12 |
| 3 | 335.26 |
| 4 | 333.22 |
| 5 | 329.49 |

Mean: 334.2. Peak-to-peak: 7.63 (2.3%). vs the pre-pinning bench
(287 → 333 = 16% iter-1-to-iter-2 jump): cold-iter penalty fully
eliminated.

Throughput vs the warm-state baseline (333 tok/s): essentially
identical.

# Reading

Per the plan's Outcome 2 framing: **variance collapse AND
throughput-neutral**. Means our page-cache + Metal driver was
already keeping pages effectively resident in steady-state on this
hardware (M2 Max, 96 GB UMA, isolated bench). `MTLResidencySet`
just makes the residency deterministic across cold runs.

Strategic implication: the **2.5-3× gap to llama.cpp is genuinely
in Differentiator 1 (kernel algorithm)**, not in residency or
page-cache state. The pinning will matter more under memory
contention (other processes competing for working-set), but does
not contribute to peak throughput on an isolated bench.

# Clarification (added 2026-05-21): kernel path used for this bench

The 333 tok/s number above was measured with `MOEFLUX_MOE_GATHER_ID`
**unset** (default at the time), routing the MoE matmul through
the OLD `affine_gather_qmm_rhs` path via `Op::MoeBatchedPermuteFuse`.

The NEW `moeflux_mm_id` kernel (the llama.cpp `kernel_mul_mm_id`
port, in `crates/moeflux-metal/shaders/gather_mm_id.metal`) landed
in the **same session-19 commit** (`5a45af3`) as the residency-set
integration, but defaulted OFF pending A/B.

# A/B result (2026-05-21): kernel port wins +11%; default flipped

Engine-level A/B run on a3b + `prefill_prompt_long.txt` (15,692
tokens, max-tokens=1), post-reboot + post-system-settle (load
average 1m=3.07, well below 5m=40.13 — load falling):

| path | tok/s (n=5) | peak-to-peak | spread |
|---|---|---|---|
| OFF (`affine_gather_qmm_rhs`) | ~333 mean | (recalled steady) | — |
| **ON (`moeflux_mm_id` port)** | **369.8 mean** | 1.81 | 0.49% |

ON breakdown: 369.05 / 370.86 / 369.05 / 369.85 / 370.24.

**+11.0% throughput.** Stable, repeatable.

Coherence harness ([[moeflux-hardening-session-c-landed]] artifact 2)
run 2026-05-21:

| prompt | jaccard mean (top-20) | cosine mean |
|---|---|---|
| `tobe` | 0.9469 | 0.999973 |
| `constitution` | 1.0000 | 1.000000 |
| `hobbit` | 0.9873 | 0.999984 |

Test green (`moeflux_coherence_a_vs_b ... ok`, 59.23s). Both paths
mathematically equivalent; safe to flip.

**Default flipped 2026-05-21**:
`crates/moeflux/src/riir/attn/linear_attn_forward.rs:40-65` —
inverted to ON by default; `MOEFLUX_MOE_GATHER_ID=0` / `false` /
`off` forces the old path (preserved for diff-oracle + A/B work).

# Strategic implication (revised 2026-05-21)

The "kernel algorithm closes most of the remaining gap" framing in
[[llama-cpp-moe-differentiators]] was **wrong** — empirically, the
kernel port closed **11%** of the gap, not "most." The remaining
gap to llama.cpp (~370 vs ~855-900 tok/s = ~2.3-2.4×) is in
**something not catalogued in that memo**. The three named
differentiators are now either landed (residency, kernel) or
deprioritized per [[feedback-vendor-recommended-lever-priority]]
(quant format — Apple-recommended, measure-to-rule-out only).

Next: profile the new-default config to find where the residual
~2.3× lives. Candidate unnamed levers — none of these are
hypotheses yet, just buckets the profile could land in:
- Non-kernel GPU overhead (cmdbuf encoding, dispatch scheduling)
- MoE routing / bucket-build cost
- Sub-kernel detail differences vs llama.cpp not captured by the
  algorithm-level port
- Host-side work that survived Phase 4 of the full-attn migration

# Sustained-throughput measurement (added 2026-05-21)

After fixing the `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`
crash via the `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` env var (mirrors
llama.cpp `ggml-metal.cpp:921-923`; fix landed in
`crates/moeflux/src/riir/backend/gpu/metal.rs:201-225` with
provenance comment), we captured sustained throughput from a
long-running blallama serving multiple sequential prefills:

| iter | prefill_tok/s | elapsed |
|---|---|---|
| 1 | 363.55 | 43.16s |
| 2 | 376.83 | 41.64s |
| 3 | 375.71 | 41.77s |

**Sustained: ~376 tok/s. First-prefill: 363 tok/s. Delta: ~3.5%.**

The first-prefill overhead is the `commit_plan` re-zero one-time
cost — `MetalBufferPool::commit_plan` (lines 420-425 in
`gpu/mod.rs`) re-zeroes color buffers that were already zero-filled
at alloc time. Subsequent prefills hit the pinned pool and pay no
re-zero. The agent's `__bzero` audit predicted ~8-12% improvement;
measured is ~3.5%. Smaller than estimated but real.

**Honest gap to llama.cpp**: ~376 vs ~855-900 = **~2.3-2.4× in
steady state**. The startup-amortization story was directionally
right but small — it didn't shrink the gap meaningfully. The
residual is in genuine steady-state GPU work efficiency, not
in startup-amplified-by-bench-protocol effects.

**Future work — `commit_plan` redundant re-zero (3.5% lever):**
Track "newly allocated this commit_plan" transients; skip re-zero
for them. Filed for a future session. Small lever but cheap.

# Cross-references (updated 2026-05-21)

- [[llama-cpp-moe-differentiators]] — superseded. The
  "Differentiator 1 closes most of the gap" framing didn't survive
  the A/B (kernel port closed 11%); the remaining 2.3× is in
  un-catalogued territory.
- [[feedback-vendor-recommended-lever-priority]] — applied:
  Q4_K_S quant format stays deprioritized; measure-to-rule-out
  before chasing.

# Scope changes from the plan

## Phase 3 deferred

Plan called for pinning KV cache + persistent GPU buffers. Scoped
down to deferred this session. Rationale:

- KV cache (~10 MB total for a3b's 20 full-attn layers) is written
  every token — the driver doesn't evict actively-touched pages,
  so pinning is unlikely to add measurable value.
- Colored transient pools (post-`commit_plan`) are heavily-
  accessed during inference, same argument.
- Implementing a pool-side residency set requires surviving
  `commit_plan`'s buffer-swap dance (physical buffers move between
  Vec slots; persistent BufIds preserve their NSObject identity
  but index-based dedup would break). Real complexity for unclear
  win.

Per `feedback_pivot_on_discovery.md`. Revisit only if a future
bench shows residual variance with expert pinning alone.

## Why we keep this in even though throughput didn't move

1. Variance fix is real (16% → 2.3% peak-to-peak). Cold runs no
   longer carry a penalty.
2. Under memory contention (parallel processes, Agora reactor
   with multiple Engines, etc.) the driver's implicit residency
   is more fragile. Explicit pinning protects us from regressions.
3. The bindings layer in `moeflux-metal` is small (~180 LOC
   including tests) and reusable for any future MTL-protocol API
   we want to bind (NSObject + objc::msg_send! pattern,
   deterministic teardown).

# Cross-references

- [[llama_cpp_moe_differentiators]] — Differentiator 2 of three;
  #1 (kernel algorithm) is the next target.
- [[prefill_next_session_plan]] — the plan this memo closes out.
- [[gather_qmm_pivot_dead]] — the diagnostic arc that produced
  the three-differentiator framing.
- [[feedback_pivot_on_discovery]] — Phase 3 scope change.
