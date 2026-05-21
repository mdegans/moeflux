---
name: prefill-sdpa-dominant-finding
description: 2026-05-21 Metal capture — SDPA flash-attn is the dominant cost (60.97% of GPU time), not MoE (11.30%). The residual 2.3× gap to llama.cpp lives in `attn_sdpa_causal_flash_gqa2`, which is register-pressure-bound at 18.7% theoretical occupancy. Supersedes the MoE-focused framing in `llama_cpp_moe_differentiators.md`.
metadata:
  type: project
---

# Capture context

- Prompt: `prefill_prompt_long.txt` (15,692 tokens), `--max-tokens 1`
- Window: chunk 0, layers 30-31 (linear+full pair, a3b
  `full_attn_interval=4` makes layer 31 the full-attn)
- Env: production defaults (`MOEFLUX_MOE_GATHER_ID` ON, residency-
  set pinning on; `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` is set
  programmatically in `MetalContext::new` so no shell env needed)
- Bundle: `/tmp/moeflux.gputrace` (53 GB — dominated by
  residency-pinned expert mmaps; Xcode opens fine)
- Effective GPU Time in window: 1329.38 ms / 61 encoders / 24
  pipeline states

# Top-shaders breakdown

| Rank | Kernel | % | Detail |
|---:|---|---:|---|
| 1 | **attn_sdpa_causal_flash_gqa2** | **60.97** | 1 dispatch, full-attn layer 31 |
| 2 | affine_qmm_t_float_gs_64_b_4 | 12.92 | 19 dispatches (projections + shared FFN) |
| 3 | **moeflux_mm_id** | **11.30** | 8 dispatches (MoE matmul, new port) |
| 4 | gated_delta_net_chunkwise | 8.04 | linear-attn delta-state, layer 30 |
| 5 | dequant_matvec_4bit_v3 | 3.39 | projection matvecs |
| 6 | moe_softmax_topk | 1.06 | router topK |
| 7–23 | tail (norms, combine, residual, swiglu, rope, kv-append, conv1d, …) | ~3 | each <1% |

Top-6 = **96.6% of GPU time**.

# Per-shader detail (from Shaders tab)

**attn_sdpa_causal_flash_gqa2:**
- Max theoretical occupancy: **18.7%**
- 512 temp registers, **1.05 KB spilled bytes**
- 1631 ALU instructions, 56 FP16, 680 FP32, 174 Int16, 558 Int32
- 91 threadgroup-load, 64 device-load, 64 device-store
- Runtime: 40.9% ALU, 22.4% Integer, 15.9% Float, 11.0%
  Mem-load, 12.7% Mem-store, 23.0% Wait+WaitMemory, 1.1% Sync

**moeflux_mm_id:**
- Max theoretical occupancy: **27.1%** (also register-bound)
- 60 temp registers, 116 high registers
- 389 ALU, 59 FP16, 116 Int16, 110 Int32
- Runtime: 81.9% ALU, 31.9% Integer, 15.4% Float Matrix, 9% Boolean

**affine_qmm_t (healthy reference):**
- Max theoretical occupancy: **68%**
- 48 temp, 132 high
- 67.2% ALU, 19.9% Integer, 10.9% Float
- Genuinely compute-bound, well-tuned.

# Strategic shift

The kernel-port arc (`moeflux_mm_id` replacing
`affine_gather_qmm_rhs`) measured **+11% throughput** at the
engine level. This trace shows `moeflux_mm_id` at **11.3% of GPU
time**. The numbers match: closing the algorithmic gap on an
11%-share kernel produced an 11% engine-level lift. **You
cannot dent a 2.3× gap by further optimizing this kernel.**

The 60% share in `attn_sdpa_causal_flash_gqa2` is the only
single-kernel lever big enough to matter. With 18.7% occupancy
and 1 KB of register spill, the kernel is leaving a lot on the
table — llama.cpp's `kernel_flash_attn_ext_f16<128>` (Metal,
6540-line `ggml-metal.metal`) is the obvious comparison target.

# Caveats on this trace

- **Counters tab limiter data didn't populate** (all reading
  100% across ALU/Memory/Texture/Buffer limiters). That's
  almost certainly an Xcode load issue, not real data — the
  per-shader detail panel (Shaders tab) numbers ARE trustworthy
  because they come from compiler stats + lightweight sampling.
- **Timeline view partially broken** (red render artifact at
  bottom; "Running in background" still active). Means we
  can't directly inspect dispatch gaps to confirm/deny the
  host-bucket-build hypothesis. The kernel-share data argues
  against it being dominant.

# Cross-references

- [[llama-cpp-moe-differentiators]] — **superseded** as the
  primary framing. The "three MoE differentiators" were real
  but two are landed (kernel port, residency set) and the
  third (quant format) is deprioritized per
  [[feedback-vendor-recommended-lever-priority]]. The residual
  gap is in attention, not MoE.
- [[prefill-residency-set-landed]] — earlier in the same
  session, the +11% engine result that this trace explains.
- `crates/moeflux-metal/shaders/sdpa.metal` — our kernel,
  501 lines.
- `~/Projects/llama_cpp_my/ggml-metal.metal:2160` —
  llama.cpp's `kernel_flash_attn_ext_f16`, templated by
  head_dim; the `h128` instantiation is the comparable.

# Audit findings (2026-05-21, same session)

Side-by-side read of `sdpa.metal` vs
`ggml-metal.metal:2160 kernel_flash_attn_ext_f16<128>` (Explore
agent) identified two MAJOR diffs and one secondary:

## 1. Scalar FMA vs simdgroup MMA (MAJOR)

**Ours** (sdpa.metal:157-177 for QK^T, 229-243 for P·V): scalar
FMA in tight loops — `simd_sum(fma(...))` over scalar partials.
**Zero simdgroup MMA.**

**llama.cpp** (ggml-metal.metal:2300-2312, 2376-2390):
`simdgroup_multiply_accumulate(mqk, mq[i], mk, mqk)` for Q·K^T,
same for P·V. **Extensive 8×8 simdgroup MMA**, using Apple's
dedicated matrix pipes — not the ALU.

Apple matrix pipes are ~5-8× more power-efficient per-FLOP than
scalar FMA AND don't compete with ALU/memory pipes for issue
slots. This is the #1 occupancy killer for ours: scalar FMA
saturates ALU, blocking the memory overlap that would otherwise
hide K/V load latency.

## 2. Per-thread register footprint (MAJOR)

**Ours** holds 128-256 floats/thread in registers (`qreg`,
`acc` arrays sized `FA_RPS × FA_VPL × FA_GROUP`). For GQA2
that's 256 floats/thread = 1 KB. Apple M1/M2 budget is ~256
registers/thread total — we're at 2× llama.cpp's footprint.
Directly explains the 512 temp registers + 1.05 KB spill in
the capture and the 18.7% occupancy ceiling.

**llama.cpp** holds 64-80 floats/thread of state. Q lives in
threadgroup SRAM (loaded once at line 2211-2224); K/V matrices
live as `simdgroup_half8x8` distributed across SRAM, not
per-thread registers.

## 3. Threadgroup-mem pre-staging (secondary)

Ours pre-stages full K/V blocks to threadgroup mem (16 KB +
4 KB scores = ~21.5 KB/TG). llama.cpp streams K/V on-the-fly
via simdgroup loads (no pre-staging, ~9-10 KB/TG). Trading
shared mem for register footprint hurts us here because we're
register-bound, not bandwidth-bound (scalar FMA is throughput-
limited, not latency-limited).

## What this means for the gap

The 18.7% occupancy isn't a tuning constant — it's the direct
output of (1) + (2): per-thread register footprint is too big
because the per-thread accumulators are too big because the
matmul is scalar instead of distributed across simdgroup
matrices. Fix (1) and (2) and occupancy should land in the
40-60% range, matching llama.cpp.

## Port-or-rewrite scope

Three bullets, ordered by win-per-effort:

1. **Replace scalar QK^T / P·V with simdgroup MMA** (sdpa.metal
   lines 157-177 and 229-243). Reshape inner loops to produce
   `simdgroup_float8x8` tiles; use
   `simdgroup_multiply_accumulate`. Recovers ~50% of occupancy
   gap by itself.
2. **Move Q to threadgroup SRAM** (mirror llama.cpp 2211-2224).
   Cuts per-thread register footprint by 200+ bytes. Occupancy
   jumps from 18.7% toward 40%+.
3. **Audit GQA register footprint** at `FA_GROUP > 1`.
   `qreg[FA_GROUP][FA_RPS][FA_VPL]` may need shrinking or
   per-head dispatch if (1)+(2) leave it too large.

GQA broadcasting (one K/V block reused across heads_per_kv Q
heads) is the one place ours wins on bandwidth — keep that,
just shrink the register arrays.

Minor diffs (not the bottleneck): pipelining granularity (per-
8×8 vs per-block barrier), F16/F32 mixed precision (theirs is
half-Q/K/V + float accum; ours is pure float), causal-mask
short-circuit. Worth porting alongside (1)+(2) for cleanliness
but not for occupancy.
