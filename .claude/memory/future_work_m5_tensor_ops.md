---
name: future-work-m5-tensor-ops
description: 2026-05-20 — when the M5 Studio lands (planned: 512 GB RAM for DeepSeek), revisit llama.cpp's `GGML_METAL_HAS_TENSOR` path. It uses Apple's `mpp::tensor_ops::matmul2d` cooperative-tensor API for the MoE matmul — M5+ hardware only. We skipped it in the initial W4_gs64 port of `kernel_mul_mm_id` because the dev box is M2 Max; add a second instantiation when the M5 arrives.
metadata:
  type: project
---

# What this is

llama.cpp's `kernel_mul_mm_id` has two MMA paths gated on
`GGML_METAL_HAS_TENSOR`:

- **Pre-tensor path** (the one we ported): manual
  `simdgroup_load` / `simdgroup_multiply_accumulate` /
  `simdgroup_store` over `simdgroup_half8x8` fragments. Works on
  all Apple Silicon back to M1.
- **Tensor path** (skipped): `mpp::tensor_ops::matmul2d` cooperative
  tensors with `execution_simdgroups<4>`. Uses Apple's M5+
  TensorOps hardware (announced 2025). The kernel literally has
  `mm.run(sB, sA, cT)` where the whole `NR1×NR0` matmul collapses
  into one API call that the driver maps to dedicated MMA units.

Reference: `~/Projects/llama-cpp-sys/external/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9699-9772` —
the `#ifdef GGML_METAL_HAS_TENSOR` blocks.

# Why we skipped it for now

- Dev box is M2 Max → no TensorOps hardware.
- Validating the pre-tensor port first gives a single source of
  truth for correctness. Adding the tensor path later is purely
  additive (different kernel instantiation, same args struct,
  same map0).
- Saves ~50 LOC of conditional kernel code in the initial port.

# When to add it back

When the M5 Studio is in hand AND a profile shows the matmul is
bottlenecked on simdgroup throughput rather than (a) memory
bandwidth, (b) the host-side commit/encode cost, or (c) the
W4_gs64 dequant inner loop. Likely worth doing pre-bench as a
single-session add since it's small.

# Adding it back — concrete steps

1. Add a `MOEFLUX_METAL_HAS_TENSOR` cfg flag (build.rs probes for
   M5+ via `device.supportsFamily(MTL::GPUFamily::Apple9)` or
   similar; cargo cfg).
2. In `shaders/gather_mm_id.metal`, wrap the K-loop's MMA section
   with `#ifdef MOEFLUX_HAS_TENSOR` / `#else` / `#endif`. The
   `#else` branch is what we have today. The `#ifdef` branch
   mirrors llama.cpp's `mpp::tensor_ops::matmul2d` block (their
   lines 9764-9772 + 9941-9945 for the store).
3. Build a second host-name PSO (e.g. `moeflux_mm_id_tensor`)
   alongside the existing `moeflux_mm_id`.
4. Pick the PSO at `Kernels::new` time via the cfg flag — same
   `Kernels` struct field, value depends on which kernel compiled.
5. Re-run the Phase 2 diff oracle on M5 hardware to validate
   numerical parity with both the existing pre-tensor PSO and the
   `affine_gather_qmm_rhs` diff oracle.

# Cross-references

- [[prefill_kernel_port_landed]] — (when written) the initial
  pre-tensor port memo.
- [[llama_cpp_moe_differentiators]] — Differentiator 1 was the
  algorithmic kernel; this is the *hardware-acceleration overlay*
  on top of it.
- llama.cpp source: `ggml-metal.metal:9699-9772, 9940-9953` for
  the exact tensor-path blocks.

# Hardware context

M5 Studio acquisition tied to the DeepSeek 671B+ run (planned
target: 512 GB unified memory, supports the full unquantized
weights with comfortable headroom). MoE workloads on M5 should
see the largest TensorOps win because the per-expert matmuls are
small-M and dispatch-heavy — the area where MMA-unit throughput
helps most relative to the existing simdgroup MMA path.
