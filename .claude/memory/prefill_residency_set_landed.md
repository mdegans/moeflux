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
