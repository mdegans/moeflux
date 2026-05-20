---
name: llama-cpp-moe-differentiators
description: 2026-05-20 — having ruled out dispatch pattern and M3+ HW as explanations for llama.cpp's 2.5-3× prefill lead on a3b/M2 Max, three actual differentiators were identified from reading the llama.cpp Metal source. Ranked by win-per-effort. Replaces the simplistic "copy their dispatch pattern" plan from `gather_qmm_arch_pivot_plan.md`.
metadata:
  type: project
---

# Context

After `gather_qmm_pivot_dead.md` killed the dispatch-pattern pivot,
we read llama.cpp's Metal source to understand the actual
differentiators. Path:
`~/Projects/llama-cpp-sys/external/llama.cpp/ggml/src/ggml-metal/`.

Two files are most relevant:
- `ggml-metal.metal` — kernel sources (lines 9618-10120 for the
  MoE matmul family)
- `ggml-metal-device.m` — buffer / memory layer (lines 1311-1576
  for buffer + residency-set lifecycle)

# Differentiator 1: kernel algorithm (biggest, hardest)

`kernel_mul_mm_id` at `ggml-metal.metal:9684` is structurally
different from both our gather kernel AND the per-expert dispatch
loop the pivot plan would have built.

**Algorithm**: ONE dispatch covers all experts. Each threadgroup is
indexed by `(tgpig.z = expert_id, tgpig.y = output_row_block,
tgpig.x = output_col_block)`. The threadgroup reads
`neh1 = tpe_u32[im]` (htpe for *this* expert) and
**early-returns if its row block exceeds neh1**.

So:
- Per-expert M-aware work distribution, in one dispatch
- Contiguous weight load per expert (NOT per-row indirection like
  our gather)
- Tile size 64×32, uses **SIMD-group MMA primitives** (`S0_4x4`,
  `S0_8x8` matrix-tile types — hardware-accelerated matmul on M2+)
- A `mul_mm_id_map0` kernel pre-builds `htpe` (counts per expert)
  + `hids` (token id per assignment) in one O(n_tokens × n_experts)
  pass. Cheap. We could likely synthesize this from our
  `ExpertBuckets` for free.

**Why this is better than our gather**: our `affine_gather_qmm_rhs`
reads per-row `indices[r]` and computes the expert weight base
address per row. That serializes the load path and eats registers
(see `gather_qmm_arch_pivot_plan.md` §2 for why this kernel is
18-25% behind dense throughput at large M). Their kernel KNOWS the
expert from `tgpig.z` and loads contiguously.

**Why this is better than per-expert dispatch loop**: no per-call
encoding overhead. One cmdbuf entry covers all experts.

**Cost to port**: significant. Multi-session. Requires writing a
new Metal kernel that uses `simdgroup_load` / `simdgroup_multiply`
/ `simdgroup_store` matrix primitives. Likely needs to be paired
with a different on-disk quant format (see Differentiator 3).

**Expected win**: hard to size without trying. Plausibly closes
most of the remaining gap after Differentiator 2 lands.

# Differentiator 2: MTLResidencySet (cheapest, attackable now)

`ggml-metal-device.m:1351-1386` shows `ggml_metal_buffer_rset_init`.
It creates an `MTLResidencySet` per buffer, adds the buffer's
allocations, then calls:

```objc
[buf->rset addAllocation:buf->buffers[i].metal];
[buf->rset commit];
[buf->rset requestResidency];
```

`requestResidency` tells the Metal driver: keep these buffers
GPU-resident across cmdbuf boundaries. Don't page them out.

The API was introduced in **macOS 15.0 (Sequoia, late 2024)**. It's
explicitly version-gated:
```objc
if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
```

We currently do NOT use residency sets (grep on moeflux returned
zero). Our expert-weight mmaps are wrapped via
`newBufferWithBytesNoCopy + MTLResourceStorageModeShared` (same as
llama.cpp), but the Metal driver is free to evict pages.

**Why this is load-bearing**: our run-to-run variance was 287→333
tok/s on the same workload (16% jump from iter-1 to iter-2). That's
classic mmap page-cache + Metal-residency warming. llama.cpp had
<1% variance across 3 runs — they never get evicted.

**Cost to port**: ~50-100 lines of raw `objc` `msg_send!` bindings
in `crates/moeflux-metal`. `metal-rs 0.32.0` does NOT have these
bindings, so we go direct. Manage retain/release manually. Version-
gate the `requestResidency` call on macOS 15+.

**Expected win**: closes the variance gap entirely (target: <1%
run-to-run like llama.cpp). Throughput win depends on how much of
our current best-case (333 tok/s) is already after-warming. If
warm-state plateaus at 333, residency-set may not add much
throughput beyond eliminating the cold penalty. If warm-state keeps
climbing past 333, residency-set is bigger.

**Critical sub-question**: does our n=5 bench (run pending as of
this writing) show warm-state plateau, or continued climb? Fold
the answer back into this memo when known.

**Integration sketch**:
1. In `crates/moeflux-metal`, add a thin wrapper for `MTLResidencySet`
   via `objc::msg_send!` macros. Types: `*mut objc::runtime::Object`.
2. Provide a `pin_buffer(&self, buffer: &Buffer)` API on `MetalContext`.
3. In `ExpertFiles::attach_to_device` (the mmap → Metal-buffer
   wrap), pin each expert buffer after wrapping.
4. Also pin the KV cache buffers and any other persistent GPU
   resource that lives across cmdbuf boundaries.
5. Bench: should see variance collapse to <1% AND probably some
   throughput gain.

# Differentiator 3: quantization layout (intertwined with #1)

Their model is `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`. Quant format
specifics:
- **Q4_K_S**: 4-bit weights in 256-element super-blocks, with
  6-bit min/max per 32-element sub-block, and fp16 super-block
  scales.
- **UD (unsloth dynamic)**: per-tensor mixed-precision sub-quants;
  some tensors may be in higher precision.

Our format: 4-bit gs_64 (groups of 64 elements, bf16 scales+biases
per group).

**Why this matters**: the dequant cost per matmul row is different.
Q4_K_S's super-block layout amortizes the scale-load cost across
256 elements vs our 64. Smaller per-element scale-lookup overhead
at small M (where dequant overhead is the largest fraction of
total work).

**Cost to port**: multi-session. Requires re-packing all expert
weights to a new on-disk format, updating the dequant code in the
gather kernel AND any per-expert dense kernel, and validating
canary parity.

**Expected win**: hard to size. Could be 5-15% on the kernel-time
slice, larger at small M. Intertwined with Differentiator 1 — a
new kernel design probably wants a new quant format anyway.

**Tactical implication**: do NOT try to port Q4_K_S in isolation.
Bundle it with the kernel-port work if/when we go there.

# Ranking by win-per-effort

| Differentiator | Effort | Expected win | Variance fix? |
|---|---|---|---|
| MTLResidencySet | ~1 session (small) | small-to-medium throughput + variance gone | YES |
| Kernel algorithm | 3-6 sessions | most of remaining gap | partial |
| Quant layout | bundle with kernel | 5-15% | no |

**Recommended order**: MTLResidencySet first. It's the cheapest, it
fixes the variance issue regardless of throughput, and the post-
residency-set bench numbers give us a much cleaner baseline to size
the remaining gap before committing to the kernel rewrite.

# What we ruled out today

- M3+ MMA hardware as a requirement: llama.cpp gets 855 tok/s on
  M2 Max. So whatever they're doing is achievable on Apple Silicon
  without specialized hardware.
- Per-expert dispatch as the differentiator: see
  `gather_qmm_pivot_dead.md`. Their per-expert dispatch is
  embedded INSIDE a single kernel, not done as a Rust dispatch
  loop. The dispatch-pattern question is moot.
- "Just copy llama.cpp's pattern" as the plan: the pattern is
  algorithmic, not architectural at the dispatch-loop level. You
  can't capture it without porting the kernel.

# Cross-references

- [[gather_qmm_pivot_dead]] — what motivates this memo.
- [[gather_qmm_arch_pivot_plan]] — the now-dead pre-empirical plan.
- [[prefill_next_session_plan]] — the next-session execution plan.
- [[pread_teardown_landed]] — earlier mmap-cost teardown.
