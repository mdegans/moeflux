# Delta-net sequential kernel — session findings (2026-05-22)

## GPU capture findings (pre-optimization)

Single-layer Metal capture with improved SDPA kernel active:

| Kernel | % of GPU time |
|--------|--------------|
| `gated_delta_net_chunkwise` | 33% |
| `affine_qmm_t_float` | 29% |
| `moeflux_mm_id` | 22% |
| everything else | ~16% |

Xcode hung on "Profiling GPU Trace..." with a 2-layer capture.
Single-layer capture worked. Known Xcode issue with large dispatch
count traces.

## Chunkwise vs sequential design

The chunkwise kernel (6-phase, simdgroup GEMM, CW_C=16) pays heavy
overhead for within-chunk parallelism that doesn't help at this state
size (128×128):

- **937 chunks × 6 phases × barriers** = 5600+ barrier syncs
- **~27 KB TG memory** per threadgroup
- **State read/written from device** at every chunk boundary
- **64 threadgroups** (one per v_head)

llama.cpp uses a simple sequential-recurrent kernel: state in
registers, loop over all tokens, `simd_sum` for dot products. Zero
barriers, zero TG memory, 2048 threadgroups.

## Key insight

For 128×128 state: the full state row (128 floats = 512 bytes) fits
in 4 registers per thread (32 threads × 4 elements = 128). The
chunkwise kernel's matrix operations (build Amat, triangular solve,
GEMM contractions) are solving a problem that doesn't exist when the
state fits in registers. The sequential kernel's simplicity wins
because the overhead of chunkwise parallelism exceeds its benefit.

This may NOT generalize to larger state sizes (e.g. 256×256 or
512×512) where register pressure would force the sequential kernel
to spill. But for Qwen3.6-A3B's 128×128, sequential is correct.

## Result

Clean A/B on a3b ~15k prefill (post-reboot, high-perf power):
- **Chunkwise (vA):** 675 tok/s
- **Sequential (vB):** 770-780 tok/s — **+15%**
- **llama.cpp:** 810-820 tok/s
- **Gap:** ~1.05× (was ~1.19× before this + SDPA session)

Diff oracle: cosine 1.000000000 at N=1,4,16,64.

## Files

- Kernel: `crates/moeflux/shaders/shaders.metal` — `gated_delta_net_sequential`
- Gate: `linear_attn_forward.rs::delta_net_vb_enabled()` — `MOEFLUX_DELTA_NET_VB`
- Dispatch: `backend/gpu/mod.rs` — `Op::GatedDeltaNetChunkwise` handler
  branches on `delta_net_vb_enabled()`
- Commit: `fd1341d`
