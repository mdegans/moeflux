# SDPA direct-device arc — session learnings (2026-05-22)

## What worked

**Direct-device reads (llama.cpp-style):** 7× kernel speedup, 1.85×
end-to-end prefill improvement (370 → 685 tok/s). Eliminated the
96.5% staging bottleneck by reading K/V direct from device memory
and using simdgroup MMA for QK^T and PV.

## What didn't work

**GQA fold on direct-device:** 3% *slower* than unfolded. The
direct-device kernel is compute-bound at the fold boundary, not
bandwidth-bound. Serializing 2 heads inside each threadgroup costs
more in GPU parallelism than it saves in L2 cache reuse. The fold
amortization pattern works for the staging kernel (where K/V staging
is the bottleneck) but not for direct-device (where occupancy and
compute throughput are the levers).

**Key insight:** don't assume a bandwidth optimization (fold) helps
when the kernel's bottleneck has shifted to compute. The 7× SDPA win
moved the bottleneck — optimizations that targeted the old bottleneck
(staging bandwidth) are now counterproductive.

## Numerical precision

The direct-device kernel (simdgroup MMA) produces slightly different
FP32 results than the staging kernel (scalar FMA + simd_sum). Both
are correct vs CPU oracle (cosine >= 0.9999), but the difference
cascades through autoregressive decode. On prompts with near-
equiprobable tokens (imperfect memorization), a single argmax flip
causes all subsequent tokens to diverge. This is expected FP-order
sensitivity, not a correctness issue.

## Where the remaining gap lives

Post-SDPA improvement, Amdahl's law says SDPA is now ~9% of total
GPU time (was 61%). The ~1.19× gap to llama.cpp (685 vs 815 tok/s)
is in MoE (~24%), matmul projections (~58%), and overhead (~9%).
Further SDPA micro-optimization has deeply diminishing returns.

Next: GPU capture with the improved kernel to identify the new
bottleneck in non-SDPA ops.
