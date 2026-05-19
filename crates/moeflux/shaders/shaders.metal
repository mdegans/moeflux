/*
 * shaders.metal — Optimized Metal compute shaders for 4-bit quantized MoE inference
 *
 * Core operations:
 *   1. dequant_matvec_4bit: Naive 4-bit affine dequant matvec (reference)
 *   2. dequant_matvec_4bit_fast: SIMD-optimized with simd_sum reduction
 *   3. dequant_matvec_4bit_v3: Fully optimized — tiled threadgroup, vector loads,
 *      coalesced access, shared input cache. Target: <0.1ms per matmul.
 *   4. swiglu_fused / swiglu_fused_vec4: SwiGLU activation
 *   5. weighted_sum: combine expert outputs with routing weights
 *   6. rms_norm: RMS normalization
 *
 * Quantization format (MLX affine 4-bit, group_size=64):
 *   - Weights stored as uint32, each holding 8 x 4-bit values
 *   - Per-group scale and bias in bfloat16
 *   - Dequantized value = uint4_val * scale + bias
 *   - Groups of 64 elements share one (scale, bias) pair
 *
 * Matrix layout for expert projections:
 *   gate_proj/up_proj: [1024, 512] uint32 = [1024, 4096] logical (out=1024, in=4096)
 *   down_proj: [4096, 128] uint32 = [4096, 1024] logical (out=4096, in=1024)
 *
 *   Scales/biases: [out_dim, in_dim/group_size]
 *   gate/up scales: [1024, 64]   (4096/64 = 64 groups)
 *   down scales:    [4096, 16]   (1024/64 = 16 groups)
 */

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

inline uint16_t f32_to_bf16(float f) {
    return uint16_t(as_type<uint>(f) >> 16);
}


// ============================================================================
// Kernel 1: 4-bit dequantized matrix-vector multiply (NAIVE — reference)
// ============================================================================

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    float acc = 0.0f;

    device const uint32_t* w_row = W_packed + tid * packed_cols;
    device const uint16_t* s_row = scales + tid * num_groups;
    device const uint16_t* b_row = biases + tid * num_groups;

    for (uint g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            for (uint n = 0; n < 8; n++) {
                uint nibble = (packed >> (n * 4)) & 0xF;
                float w_val = float(nibble) * scale + bias;
                acc += w_val * x[x_base + n];
            }
        }
    }

    out[tid] = acc;
}


// ============================================================================
// Kernel 1b: 4-bit dequant matvec — SIMD-optimized (legacy, kept for compat)
// ============================================================================

kernel void dequant_matvec_4bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            acc += (float((packed >>  0) & 0xF) * scale + bias) * x[x_base + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x[x_base + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x[x_base + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x[x_base + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x[x_base + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x[x_base + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x[x_base + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x[x_base + 7];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}

// ============================================================================
// Fused gate+up+SwiGLU: reads x ONCE, computes silu(gate(x)) * up(x)
// Saves one input read + one kernel dispatch per expert
// ============================================================================
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint16_t* gate_s    [[buffer(1)]],
    device const uint16_t* gate_b    [[buffer(2)]],
    device const uint32_t* up_W      [[buffer(3)]],
    device const uint16_t* up_s      [[buffer(4)]],
    device const uint16_t* up_b      [[buffer(5)]],
    device const float*    x         [[buffer(6)]],
    device float*          out       [[buffer(7)]],
    constant uint&         out_dim   [[buffer(8)]],
    constant uint&         in_dim    [[buffer(9)]],
    constant uint&         group_size [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint16_t* gs = gate_s + tgid * num_groups;
    device const uint16_t* gb = gate_b + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint16_t* us = up_s   + tgid * num_groups;
    device const uint16_t* ub = up_b   + tgid * num_groups;
    float ga = 0.0f, ua = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float gsc = bf16_to_f32(gs[g]), gbi = bf16_to_f32(gb[g]);
        float usc = bf16_to_f32(us[g]), ubi = bf16_to_f32(ub[g]);
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp+p], up = ur[bp+p];
            for (uint i = 0; i < 8; i++) {
                float xv = x[bx + p*8 + i];
                ga += (float((gp>>(i*4))&0xF)*gsc+gbi)*xv;
                ua += (float((up>>(i*4))&0xF)*usc+ubi)*xv;
            }
        }
    }
    threadgroup float sg[32], su[32];
    float rg = simd_sum(ga), ru = simd_sum(ua);
    uint sl = lid%32, si = lid/32, ns = (tg_size+31)/32;
    if (sl==0) { sg[si]=rg; su[si]=ru; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (si==0 && sl<ns) {
        float vg=simd_sum(sg[sl]), vu=simd_sum(su[sl]);
        if (sl==0) out[tgid] = (vg/(1.0f+exp(-vg))) * vu;
    }
}

// ============================================================================
// Kernel 1c: FULLY OPTIMIZED 4-bit dequant matvec
// ============================================================================
//
// Design for M3 Max (40-core GPU, SIMD width 32):
//
// Strategy: Each threadgroup handles ROWS_PER_TG output rows.
//   - Threadgroup size = 256 (8 SIMD groups of 32)
//   - Each SIMD group handles one output row
//   - Within a SIMD group, 32 threads split the input dimension
//   - Each thread processes in_dim/32 input elements using vector loads
//   - Reduction via simd_sum (single instruction)
//
// Memory optimizations:
//   - Input vector x cached in threadgroup shared memory (loaded once)
//   - uint4 vector loads for weights (128 bits = 32 nibbles per load)
//   - float4 vector loads for x (128 bits = 4 floats per load)
//   - Coalesced weight reads: adjacent threads read adjacent uint4 vectors
//
// For gate/up_proj [1024, 4096]: 1024/8 = 128 threadgroups, 256 threads each
//   - 128 * 256 = 32768 threads across 40 cores = good occupancy
//   - Each thread processes 4096/32 = 128 input elements = 16 uint32 packed words
//     = 4 uint4 loads per thread per row
//
// For down_proj [4096, 1024]: 4096/8 = 512 threadgroups
//   - Each thread processes 1024/32 = 32 input elements = 4 uint32 packed words
//     = 1 uint4 load per thread per row

// Number of output rows per threadgroup = number of SIMD groups (256/32 = 8)
#define ROWS_PER_TG 8

kernel void dequant_matvec_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],     // which tile of rows
    uint lid    [[thread_position_in_threadgroup]],    // 0..255
    uint simd_lane  [[thread_index_in_simdgroup]],    // 0..31
    uint simd_group [[simdgroup_index_in_threadgroup]] // 0..7
) {
    // Which output row this SIMD group handles
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;      // uint32 columns per row
    uint num_groups  = in_dim / group_size;

    // ---- Cache input vector in threadgroup shared memory ----
    // Max in_dim = 4096, so we need 4096 floats = 16KB shared memory
    // This is well within the 32KB threadgroup memory limit on M3
    threadgroup float x_shared[4096];

    // Cooperative load: 256 threads load 4096 floats (16 per thread)
    // ALL threads must participate in this load + barrier, even if their
    // row is out of bounds. Early return before the barrier causes only
    // partial loading of x_shared, corrupting results for valid rows.
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now safe to bail out for out-of-bounds rows
    if (row >= out_dim) return;

    // ---- Pointer setup for this row ----
    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    // ---- Each lane processes a strided slice of the packed columns ----
    // Lane k processes columns: k, k+32, k+64, ...
    // This gives coalesced reads: adjacent lanes read adjacent uint32 words.

    float acc = 0.0f;

    // Process packed columns in strides of 32 (one per SIMD lane)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // Determine which group this column belongs to
        // packed_per_group = group_size / 8 = 64 / 8 = 8
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        // Dequantize 8 nibbles and multiply with cached x
        // Rearranged: (nibble * scale + bias) * x = nibble * (scale*x) + bias*x
        // Pre-compute scale*x and bias*x, then use FMA for dequant+multiply in one op.
        // This reduces per-nibble from (convert + mul + add + mul + add) to (convert + FMA + add).
        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    // ---- SIMD reduction: sum across 32 lanes ----
    float sum = simd_sum(acc);

    // Lane 0 writes the result
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1d-v3-n: 4-bit dequant matvec, N tokens (in_dim ≤ 4096 path).
// ============================================================================
//
// Batched-prefill variant of `dequant_matvec_4bit_v3`. Same weights
// applied to `n_tokens` stacked input vectors. Per-(row, token)
// arithmetic matches v3 exactly: ROWS_PER_TG=8, 256 threads/TG,
// per-row simd_sum reduction. So N=1 is bit-exact vs encode_matvec
// on the v3 path.
//
// Each threadgroup caches one token's input in 16KB threadgroup memory
// (same as v3 — in_dim ≤ 4096 floats fits in 16KB). Output layout
// `[n_tokens, out_dim]`.
//
// Grid linearization: tg_idx_flat = row_tile + token * num_row_tiles.
// Threadgroups total = num_row_tiles * n_tokens.

kernel void dequant_matvec_4bit_v3_n_tokens(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x_inputs   [[buffer(3)]],  // [n_tokens, in_dim]
    device float*          out        [[buffer(4)]],  // [n_tokens, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         n_tokens   [[buffer(8)]],
    constant uint&         num_row_tiles [[buffer(9)]],  // (out_dim + 7) / 8
    uint tgid_flat  [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint token    = tgid_flat / num_row_tiles;
    uint row_tile = tgid_flat % num_row_tiles;
    uint row      = row_tile * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];

    device const float* x_token =
        x_inputs + (size_t)token * (size_t)in_dim;

    // Cooperative load: all threads participate before bailing on row OOB.
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x_token[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (token >= n_tokens) return;
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales   + row * num_groups;
    device const uint16_t* b_row = biases   + row * num_groups;

    float acc = 0.0f;
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[(size_t)token * (size_t)out_dim + row] = sum;
    }
}


// ============================================================================
// Kernel 1d-fast-n: 4-bit dequant matvec, N tokens (in_dim > 4096 path).
// ============================================================================
//
// Batched-prefill variant of `dequant_matvec_4bit_fast`. Used when the
// input vector doesn't fit in 16KB threadgroup memory. Reads x directly
// from device memory.
//
// One threadgroup per (output row, token). Each TG has tg_size (64)
// threads that stride over the row's groups, accumulating into
// `acc`, then reduce via simd_sum + cross-simd shared-memory pass —
// same tail as `dequant_matvec_4bit_fast`. Per-(row, token)
// arithmetic matches the single-row fast kernel exactly, so N=1 is
// bit-exact vs encode_matvec on the fast path.
//
// Grid linearization: tg_idx_flat = row + token * out_dim.

kernel void dequant_matvec_4bit_fast_n_tokens(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x_inputs   [[buffer(3)]],   // [n_tokens, in_dim]
    device float*          out        [[buffer(4)]],   // [n_tokens, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         n_tokens   [[buffer(8)]],
    uint tgid_flat [[threadgroup_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]]
) {
    uint token = tgid_flat / out_dim;
    uint row   = tgid_flat % out_dim;
    if (token >= n_tokens) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* w_row = W_packed + (size_t)row * (size_t)packed_cols;
    device const uint16_t* s_row = scales   + (size_t)row * (size_t)num_groups;
    device const uint16_t* b_row = biases   + (size_t)row * (size_t)num_groups;
    device const float*    x_tok = x_inputs + (size_t)token * (size_t)in_dim;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            acc += (float((packed >>  0) & 0xF) * scale + bias) * x_tok[x_base + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x_tok[x_base + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x_tok[x_base + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x_tok[x_base + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x_tok[x_base + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x_tok[x_base + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x_tok[x_base + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x_tok[x_base + 7];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[(size_t)token * (size_t)out_dim + row] = val;
        }
    }
}


// ============================================================================
// Kernel 1c-8bit: fully optimized 8-bit dequant matvec
// ============================================================================
// Mirrors dequant_matvec_4bit_v3 but unpacks 4 bytes per uint32 instead of
// 8 nibbles. Needed for models (e.g. Qwen3.6-35B-A3B) that quantize a small
// subset of tensors (mlp.gate, shared_expert_gate) at 8-bit while leaving
// everything else at 4-bit.
//
// Same group-affine layout as 4-bit (scale + bias per GROUP_SIZE=64 values,
// stored bfloat16). Just a different values-per-uint32 (4 vs 8).

kernel void dequant_matvec_8bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/4]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 4;      // uint32 columns per row (4 bytes each)
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];

    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // packed_per_group = group_size / 4 = 64 / 4 = 16
        uint g = col / (group_size / 4);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 4;

        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];

        acc += fma(float((packed >>  0) & 0xFFu), sx0, bx0);
        acc += fma(float((packed >>  8) & 0xFFu), sx1, bx1);
        acc += fma(float((packed >> 16) & 0xFFu), sx2, bx2);
        acc += fma(float((packed >> 24) & 0xFFu), sx3, bx3);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel: 8-bit dequant matvec, N tokens
// ============================================================================
// Batched-prefill variant of `dequant_matvec_8bit_v3`. Same per-(row, token)
// arithmetic, with an outer token-axis sweep encoded as
// `tgid_flat = token * num_row_tiles + row_tile`. Per-(row, token) result
// is bit-exact against `dequant_matvec_8bit_v3` for N=1.
//
// `x_shared` size: 4096 floats = 16 KB tg-mem (limit on Apple Silicon
// is 32 KB). in_dim must be ≤ 4096.

kernel void dequant_matvec_8bit_v3_n_tokens(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/4]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x_inputs   [[buffer(3)]],  // [n_tokens, in_dim]
    device float*          out        [[buffer(4)]],  // [n_tokens, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         n_tokens   [[buffer(8)]],
    constant uint&         num_row_tiles [[buffer(9)]],  // (out_dim + ROWS_PER_TG-1) / ROWS_PER_TG
    uint tgid_flat  [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint token    = tgid_flat / num_row_tiles;
    uint row_tile = tgid_flat % num_row_tiles;
    uint row      = row_tile * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 4;
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];

    device const float* x_token =
        x_inputs + (size_t)token * (size_t)in_dim;

    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x_token[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (token >= n_tokens) return;
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales   + row * num_groups;
    device const uint16_t* b_row = biases   + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 4);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 4;

        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];

        acc += fma(float((packed >>  0) & 0xFFu), sx0, bx0);
        acc += fma(float((packed >>  8) & 0xFFu), sx1, bx1);
        acc += fma(float((packed >> 16) & 0xFFu), sx2, bx2);
        acc += fma(float((packed >> 24) & 0xFFu), sx3, bx3);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[(size_t)token * (size_t)out_dim + row] = sum;
    }
}


// ============================================================================
// Kernel 1f: 4-bit dequant matvec with LUT (eliminates uint→float conversions)
// ============================================================================
// Instead of converting each nibble to float (expensive conversion instruction),
// pre-compute a 16-entry LUT per group: lut[v] = float(v) * scale + bias.
// Then inner loop is just: acc += lut[nibble] * x_shared[i] — pure math, no conversions.
// The LUT is recomputed every group_size/8 iterations (amortized).

#define ROWS_PER_TG_V5 8

kernel void dequant_matvec_4bit_v5(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_V5 + simd_group;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;
    uint prev_g = 0xFFFFFFFF;
    float lut[16];

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_group;

        // Rebuild LUT when group changes
        if (g != prev_g) {
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);
            for (uint v = 0; v < 16; v++) {
                lut[v] = float(v) * scale + bias;
            }
            prev_g = g;
        }

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += lut[(packed >>  0) & 0xF] * x_shared[x_base + 0];
        acc += lut[(packed >>  4) & 0xF] * x_shared[x_base + 1];
        acc += lut[(packed >>  8) & 0xF] * x_shared[x_base + 2];
        acc += lut[(packed >> 12) & 0xF] * x_shared[x_base + 3];
        acc += lut[(packed >> 16) & 0xF] * x_shared[x_base + 4];
        acc += lut[(packed >> 20) & 0xF] * x_shared[x_base + 5];
        acc += lut[(packed >> 24) & 0xF] * x_shared[x_base + 6];
        acc += lut[(packed >> 28) & 0xF] * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 1e: 2-bit affine dequant matvec (same structure as v3)
// ============================================================================
// Packs 16 x 2-bit values per uint32. Each value is 0-3, dequantized as:
//   val = uint2 * scale + bias (same affine quantization, just 2-bit range)
// Same group structure: group_size elements share one (scale, bias) pair.
// packed_cols = in_dim / 16 (16 values per uint32, vs 8 for 4-bit)

kernel void dequant_matvec_2bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/16]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 16;  // 16 values per uint32 for 2-bit
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    // Each lane processes strided columns (16 values per uint32)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // group_size/16 packed words per group
        uint g = col / (group_size / 16);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 16;

        // Unroll 16 x 2-bit extractions
        acc += (float((packed >>  0) & 0x3) * scale + bias) * x_shared[x_base +  0];
        acc += (float((packed >>  2) & 0x3) * scale + bias) * x_shared[x_base +  1];
        acc += (float((packed >>  4) & 0x3) * scale + bias) * x_shared[x_base +  2];
        acc += (float((packed >>  6) & 0x3) * scale + bias) * x_shared[x_base +  3];
        acc += (float((packed >>  8) & 0x3) * scale + bias) * x_shared[x_base +  4];
        acc += (float((packed >> 10) & 0x3) * scale + bias) * x_shared[x_base +  5];
        acc += (float((packed >> 12) & 0x3) * scale + bias) * x_shared[x_base +  6];
        acc += (float((packed >> 14) & 0x3) * scale + bias) * x_shared[x_base +  7];
        acc += (float((packed >> 16) & 0x3) * scale + bias) * x_shared[x_base +  8];
        acc += (float((packed >> 18) & 0x3) * scale + bias) * x_shared[x_base +  9];
        acc += (float((packed >> 20) & 0x3) * scale + bias) * x_shared[x_base + 10];
        acc += (float((packed >> 22) & 0x3) * scale + bias) * x_shared[x_base + 11];
        acc += (float((packed >> 24) & 0x3) * scale + bias) * x_shared[x_base + 12];
        acc += (float((packed >> 26) & 0x3) * scale + bias) * x_shared[x_base + 13];
        acc += (float((packed >> 28) & 0x3) * scale + bias) * x_shared[x_base + 14];
        acc += (float((packed >> 30) & 0x3) * scale + bias) * x_shared[x_base + 15];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1d: FULLY OPTIMIZED with uint4 vector loads
// ============================================================================
//
// Same structure as v3 but uses uint4 loads (128-bit / 16 bytes) to maximize
// memory bandwidth per thread. Each uint4 = 4 uint32 = 32 nibbles.
//
// For gate/up (packed_cols=512): each thread processes 512/32 = 16 uint32
//   = 4 uint4 loads per thread
// For down (packed_cols=128): each thread processes 128/32 = 4 uint32
//   = 1 uint4 load per thread

kernel void dequant_matvec_4bit_v4(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache input vector — ALL threads must participate before the barrier
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    // Pointers — cast to uint4 for vector loads
    device const uint4* w_row_v = (device const uint4*)(W_packed + row * packed_cols);
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    uint vec4_cols = packed_cols / 4;  // number of uint4 vectors per row

    float acc = 0.0f;

    // Each lane processes vec4_cols / 32 vectors (coalesced: adjacent lanes read adjacent uint4)
    for (uint vi = simd_lane; vi < vec4_cols; vi += 32) {
        uint4 packed4 = w_row_v[vi];

        // Each uint4 covers 4 * 8 = 32 input elements
        // Starting packed column index = vi * 4
        uint base_col = vi * 4;
        uint x_base = base_col * 8;  // starting input element

        // Process each of the 4 uint32 words in the uint4
        // Unroll all 4 words x 8 nibbles = 32 multiply-adds
        #pragma unroll
        for (uint w = 0; w < 4; w++) {
            uint32_t packed = packed4[w];
            uint col = base_col + w;
            uint g = col / (group_size / 8);
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);

            uint xb = x_base + w * 8;
            acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[xb + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[xb + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[xb + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[xb + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[xb + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[xb + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[xb + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[xb + 7];
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1e: Multi-expert batched matvec
// ============================================================================
//
// Dispatch multiple experts simultaneously. The grid's Y dimension indexes
// the expert, so K experts' matmuls run as parallel threadgroups.
//
// Buffer layout: W_packed, scales, biases are arrays of K experts concatenated.
// x_inputs:  K input vectors concatenated [K * in_dim]
// out:       K output vectors concatenated [K * out_dim]
// expert_offsets: byte offset into W_packed buffer for each expert's weights
//                 (allows non-contiguous expert data in a shared buffer)

kernel void dequant_matvec_4bit_batched(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x_inputs   [[buffer(3)]],  // [K, in_dim]
    device float*          out        [[buffer(4)]],  // [K, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    // Per-expert offsets into the weight/scale/bias buffers (in elements)
    device const uint*     w_offsets  [[buffer(8)]],  // [K] offset in uint32 elements
    device const uint*     s_offsets  [[buffer(9)]],  // [K] offset in uint16 elements
    device const uint*     b_offsets  [[buffer(10)]], // [K] offset in uint16 elements
    constant uint&         num_row_tiles [[buffer(11)]], // ceil(out_dim / ROWS_PER_TG)
    uint tgid_flat [[threadgroup_position_in_grid]],  // linearized (row_tile + expert * num_row_tiles)
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // De-linearize: tgid_flat = row_tile + expert_k * num_row_tiles
    uint expert_k = tgid_flat / num_row_tiles;
    uint row_tile = tgid_flat % num_row_tiles;
    uint row = row_tile * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache this expert's input vector
    threadgroup float x_shared[4096];
    device const float* x_k = x_inputs + expert_k * in_dim;
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x_k[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Point to this expert's weights
    device const uint32_t* w_row = W_packed + w_offsets[expert_k] + row * packed_cols;
    device const uint16_t* s_row = scales   + s_offsets[expert_k] + row * num_groups;
    device const uint16_t* b_row = biases   + b_offsets[expert_k] + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[x_base + 0];
        acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[x_base + 1];
        acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[x_base + 2];
        acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[x_base + 3];
        acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[x_base + 4];
        acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[x_base + 5];
        acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[x_base + 6];
        acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert_k * out_dim + row] = sum;
    }
}


// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================

kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// Vectorized SwiGLU: process 4 elements per thread
kernel void swiglu_fused_vec4(
    device const float4* gate [[buffer(0)]],
    device const float4* up   [[buffer(1)]],
    device float4*       out  [[buffer(2)]],
    constant uint&       dim  [[buffer(3)]],  // original dim (must be multiple of 4)
    uint tid [[thread_position_in_grid]]
) {
    uint vec_dim = dim / 4;
    if (tid >= vec_dim) return;

    float4 g = gate[tid];
    float4 silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 2b: Batched SwiGLU for K experts
// ============================================================================

kernel void swiglu_fused_batched(
    device const float* gate [[buffer(0)]],  // [K * dim]
    device const float* up   [[buffer(1)]],  // [K * dim]
    device float*       out  [[buffer(2)]],  // [K * dim]
    constant uint&      dim  [[buffer(3)]],
    constant uint&      K    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = K * dim;
    if (tid >= total) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 3: Weighted sum of expert outputs
// ============================================================================

kernel void weighted_sum(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      K           [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * dim + tid];
    }
    out[tid] = acc;
}


// ============================================================================
// Kernel 4: RMS Normalization
// ============================================================================

kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];

    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            sum_sq[0] = val;
        }
    }
}

kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}


// ============================================================================
// Kernel 4b: RMS Normalization with bf16 weights
// ============================================================================
// Same as rms_norm_apply but reads weights as bfloat16 (uint16_t) and
// converts to float32 inline. Used in the fused o_proj+norm+routing path
// where norm weights come directly from the mmap'd weight file (bf16).

kernel void rms_norm_apply_bf16(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],  // bf16 weights
    device const float*    sum_sq  [[buffer(2)]],
    device float*          out     [[buffer(3)]],
    constant uint&         dim     [[buffer(4)]],
    constant float&        eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    float w = bf16_to_f32(weight[tid]);
    out[tid] = x[tid] * rms * w;
}


// ============================================================================
// Kernel 5: Residual add
// ============================================================================
// out[i] = a[i] + b[i]
// Used to fuse the residual connection into a GPU command buffer,
// eliminating a CPU round-trip between o_proj and routing.

kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}


// ============================================================================
// Kernel: BF16 matvec (un-dequantized weights)
// ============================================================================
// output[r] = Σ_i input[i] * bf16_to_f32(w[r, i])
//
// Used by the Cogito-V2 / DeepSeek-V3 MoE router-gate matvec where the
// gate weights are stored as bf16 (not 4-bit) — `model.layers.{i}.mlp
// .gate.weight` at shape [num_experts=256, hidden_dim=7168].
//
// Threadgroup-per-output-row layout: tg_idx selects the output row
// (= expert index); lanes within the threadgroup parallelize over the
// in_dim and reduce via threadgroup memory. 256 threads/group is the
// sweet spot for partials reduction on Apple Silicon.

kernel void bf16_matvec(
    device const uint16_t* w        [[buffer(0)]],   // bf16 weights, row-major [out_dim, in_dim]
    device const float*    input    [[buffer(1)]],   // [in_dim] f32
    device float*          output   [[buffer(2)]],   // [out_dim] f32
    constant uint&         in_dim   [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    uint tg_idx  [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tg_idx >= out_dim) return;

    threadgroup float partials[256];

    const device uint16_t* row = w + (size_t)tg_idx * (size_t)in_dim;

    float sum = 0.0;
    for (uint i = lid; i < in_dim; i += tg_size) {
        sum += input[i] * bf16_to_f32(row[i]);
    }
    partials[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduce over partials[0..tg_size). Assumes tg_size is a
    // power of two (we dispatch with 256, satisfies it).
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            partials[lid] += partials[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[tg_idx] = partials[0];
    }
}


// ============================================================================
// Kernel 5b: BF16-weight matmul, N tokens (same weights, different inputs).
// ============================================================================
//
// Batched-prefill primitive. Applies a single [out_dim, in_dim] BF16
// weight matrix to a run of N token activations [n_tokens, in_dim] and
// writes [n_tokens, out_dim] f32.
//
// Per-(row, token) reduction matches `bf16_matvec` exactly: same
// stride-by-tg_size traversal, same tree reduce. Running N calls of
// `bf16_matvec` and one call of this with n_tokens=N must produce
// bit-exact-mod-fp-reorder identical output — they are the same
// arithmetic.
//
// Grid: linearized (row + token * out_dim). 256 threads/group.
// out_dim × n_tokens threadgroups total. Fits comfortably in MTLSize
// width for any (hidden_dim, prefill_batch) we ship.

kernel void bf16_matmul_n_tokens(
    device const uint16_t* w        [[buffer(0)]],   // [out_dim, in_dim] bf16 row-major
    device const float*    input    [[buffer(1)]],   // [n_tokens, in_dim] f32
    device float*          output   [[buffer(2)]],   // [n_tokens, out_dim] f32
    constant uint&         in_dim   [[buffer(3)]],
    constant uint&         out_dim  [[buffer(4)]],
    constant uint&         n_tokens [[buffer(5)]],
    uint tg_idx_flat [[threadgroup_position_in_grid]],
    uint lid         [[thread_position_in_threadgroup]],
    uint tg_size     [[threads_per_threadgroup]]
) {
    uint token = tg_idx_flat / out_dim;
    uint row   = tg_idx_flat % out_dim;
    if (token >= n_tokens) return;

    threadgroup float partials[256];

    const device uint16_t* w_row = w + (size_t)row * (size_t)in_dim;
    const device float*    x_row = input + (size_t)token * (size_t)in_dim;

    float sum = 0.0;
    for (uint i = lid; i < in_dim; i += tg_size) {
        sum += x_row[i] * bf16_to_f32(w_row[i]);
    }
    partials[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            partials[lid] += partials[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        output[(size_t)token * (size_t)out_dim + row] = partials[0];
    }
}


// ============================================================================
// Kernel 6: Batched GPU attention scores (Q @ K^T, scaled) — all heads at once
// ============================================================================
//
// Computes scores[h, p] = sum_d(Q[h, d] * K[p, kv_h*head_dim + d]) * scale
// for all heads h in [0, num_heads) and positions p in [0, seq_len).
//
// Grid: linearized (pos + h * num_seq_tgs) — one threadgroup per (position, head).
// Each threadgroup of 256 threads reduces over head_dim=256.
//
// GQA mapping: kv_head = h / heads_per_kv (e.g. 16 query heads share 1 KV head)
//
// Output layout: scores[h * seq_stride + p] where seq_stride = MAX_SEQ_LEN

kernel void attn_scores_batched(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       scores     [[buffer(2)]],  // [num_heads, seq_stride]
    constant uint&      head_dim   [[buffer(3)]],  // 256
    constant uint&      kv_dim     [[buffer(4)]],  // 512
    constant uint&      seq_len    [[buffer(5)]],  // current seq length
    constant uint&      seq_stride [[buffer(6)]],  // MAX_SEQ_LEN
    constant float&     scale      [[buffer(7)]],  // 1/sqrt(head_dim)
    constant uint&      heads_per_kv [[buffer(8)]], // 16 (GQA ratio)
    constant uint&      num_seq_tgs  [[buffer(9)]],  // = seq_len
    uint tgid  [[threadgroup_position_in_grid]],    // linearized: pos + h * num_seq_tgs
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    device const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

    float acc = 0.0f;
    for (uint d = lid; d < head_dim; d += tg_size) {
        acc += qh[d] * kp[d];
    }

    // SIMD reduction
    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}


// ============================================================================
// Kernel 7: Batched softmax — one threadgroup per head
// ============================================================================

kernel void attn_softmax_batched(
    device float*    scores     [[buffer(0)]],  // [num_heads, seq_stride]
    constant uint&   seq_len    [[buffer(1)]],
    constant uint&   seq_stride [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],     // head index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    device float* s = scores + tgid * seq_stride;

    // Pass 1: find max
    threadgroup float shared_max[32];
    float local_max = -1e30f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        local_max = max(local_max, s[i]);
    }
    float sm = simd_max(local_max);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared_max[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_max = simd_max(shared_max[simd_lane]);
    }
    threadgroup float broadcast_max;
    if (lid == 0) broadcast_max = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = broadcast_max;

    // Pass 2: exp and sum
    threadgroup float shared_sum[32];
    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        float val = exp(s[i] - global_max);
        s[i] = val;
        local_sum += val;
    }
    float simd_s = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_group] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_sum = simd_sum(shared_sum[simd_lane]);
    }
    threadgroup float broadcast_sum;
    if (lid == 0) broadcast_sum = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = broadcast_sum;

    // Pass 3: normalize
    float inv_sum = 1.0f / global_sum;
    for (uint i = lid; i < seq_len; i += tg_size) {
        s[i] *= inv_sum;
    }
}


// ============================================================================
// Kernel 8: Batched attention value aggregation (scores @ V) — all heads
// ============================================================================
//
// For each head h: output[h*head_dim + d] = sum_p(scores[h*seq_stride+p] * V[p*kv_dim + kv_h*head_dim + d])
//
// Grid: linearized over (head_dim * num_heads) — one thread per (dimension, head).

kernel void attn_values_batched(
    device const float* scores   [[buffer(0)]],  // [num_heads, seq_stride]
    device const float* V_cache  [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       out      [[buffer(2)]],  // [num_heads, head_dim]
    constant uint&      head_dim  [[buffer(3)]],  // 256
    constant uint&      kv_dim    [[buffer(4)]],  // 512
    constant uint&      seq_len   [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant uint&      heads_per_kv [[buffer(7)]],
    uint tid [[thread_position_in_grid]]          // linearized: d + h * head_dim
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint p = 0; p < seq_len; p++) {
        acc += s[p] * V_cache[p * kv_dim + kv_h * head_dim + d];
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Kernel 9: Sigmoid element-wise gate
// ============================================================================
// out[i] = x[i] * sigmoid(gate[i])

kernel void sigmoid_gate(
    device float*       x_out  [[buffer(0)]],  // [dim] in/out
    device const float* gate   [[buffer(1)]],  // [dim] gate values
    constant uint&      dim    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = 1.0f / (1.0f + exp(-gate[tid]));
    x_out[tid] = x_out[tid] * g;
}


// ============================================================================
// Kernel 10: GatedDeltaNet linear attention step (batched over the token axis)
// ============================================================================
//
// Implements the GatedDeltaNet recurrence, per token:
//   1. State decay:  S[vi][ki] *= g_decay
//   2. Memory read:  kv_mem[vi] = sum_ki(S[vi][ki] * k[ki])
//   3. Delta:        delta[vi] = (v[vi] - kv_mem[vi]) * beta_gate
//   4. State update: S[vi][ki] += k[ki] * delta[vi]
//   5. Output:       out[vi] = sum_ki(S[vi][ki] * q[ki])
//
// The recurrence is sequential over time but fully parallel over
// (head, vi): thread (head_id, vi) owns state row S[head_id][vi][:] and
// every step is independent across vi (no cross-thread reduction). So
// the time loop runs INSIDE the kernel — one dispatch for the whole
// chunk — and the recurrence stays correct because it is each thread's
// own `for t` loop over its private state row.
//
// Dispatch: num_v_heads threadgroups, 128 threads each (one per vi).
// State layout: [num_v_heads * 128 * 128] float, persisted across calls.
// `conv_out` is [n_tokens * (2*key_total + num_v_heads*128)]: per token,
// q | k (each `key_total` floats) | v (num_v_heads*128 floats).
// k-head sharing: `k_heads_per_v` v-heads share 1 k-head.

kernel void gated_delta_net_step(
    device float *state,             // [num_v_heads * 128 * 128] persistent
    device const float *conv_out,    // [n_tokens * conv_per_token]
    device const float *g_decay,     // [n_tokens * num_v_heads]
    device const float *beta_gate,   // [n_tokens * num_v_heads]
    device float *output,            // [n_tokens * num_v_heads * 128]
    constant uint &k_heads_per_v,    // = 4
    constant uint &n_tokens,
    constant uint &key_total,        // q / k region size per token (= 2048)
    constant uint &num_v_heads,      // = 64
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    uint value_total = num_v_heads * 128;
    uint conv_per_token = 2 * key_total + value_total;

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_off_in_token = kh * 128;  // q and k both indexed by kh*128

    for (uint t = 0; t < n_tokens; t++) {
        uint tok_base = t * conv_per_token;
        uint q_base = tok_base + k_off_in_token;
        uint k_base = tok_base + key_total + k_off_in_token;
        uint v_base = tok_base + 2 * key_total + head_id * 128;
        float g = g_decay[t * num_v_heads + head_id];
        float beta = beta_gate[t * num_v_heads + head_id];

        // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
        float kv_mem = 0.0f;
        for (uint ki = 0; ki < 128; ki++) {
            float s = state[state_base + ki] * g;
            state[state_base + ki] = s;
            kv_mem += s * conv_out[k_base + ki];
        }

        // Step 3+4: Delta update — S[vi][ki] += k[ki] * delta
        float delta = (conv_out[v_base + vi] - kv_mem) * beta;
        for (uint ki = 0; ki < 128; ki++) {
            state[state_base + ki] += conv_out[k_base + ki] * delta;
        }

        // Step 5: Output — out[vi] = dot(S[vi][:], q[:])
        float out_val = 0.0f;
        for (uint ki = 0; ki < 128; ki++) {
            out_val += state[state_base + ki] * conv_out[q_base + ki];
        }
        output[t * value_total + head_id * 128 + vi] = out_val;
    }
}


// ============================================================================
// Kernel 10b: Gated DeltaNet — chunkwise-parallel recurrence
// ============================================================================
//
// Chunkwise-parallel reformulation of `gated_delta_net_step`. Same
// delta-rule recurrence, but the within-chunk computation is expressed
// as matmuls + a triangular solve, so only the chunk-to-chunk state
// carry stays sequential (n / CW_C steps instead of n).
//
// CPU reference + math derivation: `gated_delta_chunkwise` in
// linear_attn.rs. Diff oracle: the CpuBackend `GatedDeltaNetChunkwise`
// arm (which matches the per-token recurrence). With cumulative
// log-decay L_l and decay ratio Gamma_{l,i} = exp(L_l - L_i):
//
//   A_{l,i} = beta_l * Gamma_{l,i} * (k_i . k_l)   (strictly lower)
//   B_l     = beta_l * v_l - beta_l * gamma_l * (S_0 . k_l)
//   (I + A) U = B                                  (forward subst.)
//   out_l   = gamma_l * (S_0 . q_l) + sum_{i<=l} Gamma_{l,i}(k_i.q_l) U_i
//   S_C     = gamma_{C-1} * S_0 + sum_i Gamma_{C-1,i} * U_i * k_i^T
//
// Dispatch: num_v_heads threadgroups, 128 threads each (one per vi).
// Each threadgroup loops over inner chunks of CW_C tokens internally.
// Thread `vi` owns state/output/U row `vi` exclusively — the only
// shared (cross-thread) threadgroup state is kc / A / kqg / log_decay
// / beta_s, all written once per chunk then read-only. Buffer layout
// is identical to `gated_delta_net_step`.

#define CW_C 16          // inner chunk length C
#define CW_STRIP 16      // Phase-6 GEMM column-strip width

kernel void gated_delta_net_chunkwise(
    device float *state,             // [num_v_heads * 128 * 128] persistent
    device const float *conv_out,    // [n_tokens * conv_per_token]
    device const float *g_decay,     // [n_tokens * num_v_heads]
    device const float *beta_gate,   // [n_tokens * num_v_heads]
    device float *output,            // [n_tokens * num_v_heads * 128]
    constant uint &k_heads_per_v,    // = 4
    constant uint &n_tokens,
    constant uint &key_total,        // q / k region size per token (= 2048)
    constant uint &num_v_heads,      // = 64
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    uint value_total = num_v_heads * 128;
    uint conv_per_token = 2 * key_total + value_total;
    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_off = kh * 128;  // q and k both indexed by kh*128

    threadgroup float kc[CW_C * 128];      // staged k vectors    (8 KB)
    threadgroup float Umat[CW_C * 128];    // RHS, solved in place (8 KB)
    threadgroup float Amat[CW_C * CW_C];   // beta*Gamma*(k.k)    (1 KB)
    threadgroup float kqg[CW_C * CW_C];    // Gamma*(k.q)         (1 KB)
    threadgroup float log_decay[CW_C];
    threadgroup float beta_s[CW_C];
    // Phase-6 delta-strip staging, laid out [D = 128, CW_STRIP].
    // Reused each strip; total threadgroup memory now ~26.8 KB of 32 KB.
    threadgroup float sdc[CW_C * 128];

    for (uint chunk_start = 0; chunk_start < n_tokens;
         chunk_start += CW_C) {
        uint c = min((uint)CW_C, n_tokens - chunk_start);

        // --- Phase 1: stage kc, cumulative log-decay, beta ---
        for (uint l = 0; l < c; l++) {
            uint t = chunk_start + l;
            kc[l * 128 + vi] =
                conv_out[t * conv_per_token + key_total + k_off + vi];
        }
        if (vi == 0) {
            float acc = 0.0f;
            for (uint l = 0; l < c; l++) {
                uint t = chunk_start + l;
                // Clamp away from 0 — a strong-forget gate can be
                // exactly 0.0, and ln(0) = -inf poisons the chunk's
                // exp(L_l - L_i) with (-inf)-(-inf) = NaN. Floor matches
                // the CPU reference's `G_DECAY_LN_FLOOR` (linear_attn.rs).
                float g = max(g_decay[t * num_v_heads + head_id], 1e-30f);
                acc += precise::log(g);
                log_decay[l] = acc;
                beta_s[l] = beta_gate[t * num_v_heads + head_id];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2: build Amat (strictly lower) + kqg (incl. diag) ---
        // CW_C*CW_C entries spread across the 128 threads.
        for (uint slot = vi; slot < CW_C * CW_C; slot += 128) {
            uint l = slot / CW_C;
            uint i = slot % CW_C;
            float a = 0.0f;
            float g = 0.0f;
            if (l < c && i <= l) {
                float gamma_li =
                    precise::exp(log_decay[l] - log_decay[i]);
                if (i < l) {
                    float kk = 0.0f;
                    for (uint d = 0; d < 128; d++) {
                        kk += kc[i * 128 + d] * kc[l * 128 + d];
                    }
                    a = beta_s[l] * gamma_li * kk;
                }
                uint q_base =
                    (chunk_start + l) * conv_per_token + k_off;
                float kq = 0.0f;
                for (uint d = 0; d < 128; d++) {
                    kq += kc[i * 128 + d] * conv_out[q_base + d];
                }
                g = gamma_li * kq;
            }
            Amat[slot] = a;
            kqg[slot] = g;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 3: RHS  U_l = beta_l v_l - beta_l gamma_l (S_0.k_l) ---
        // The S_0.k contraction is a GEMM  s0k[C,D] = kc[C,D] @ S_0ᵀ[D,D],
        // contraction over d (D=128 — 16 full tiles, no raggedness). The
        // chunk index l is an *output row*; a ragged l only corrupts an
        // unused output row, so — unlike Phase 6, where raggedness sits on
        // the contraction axis — no zero-fill is needed here.
        //   a = kc   [C,D] tg-staged, stride 128, no transpose.
        //   b = S_0ᵀ loaded transposed from the [vi,d] row-major device
        //       `state` buffer (transpose gives [d,vi]).
        // 4 simdgroups (sg = vi/32) tile the [C=16,D=128] output into sdc.
        {
            uint sg = vi / 32;
            uint col_tiles = 128 / 8;               // D / 8 = 16
            uint n_tiles = (CW_C / 8) * col_tiles;  // 2 * 16 = 32
            for (uint ti = sg; ti < n_tiles; ti += 4) {
                uint rt = ti / col_tiles;   // C row tile (0..1)
                uint ct = ti % col_tiles;   // D col tile (0..15)
                simdgroup_matrix<float, 8, 8> sacc;
                for (uint kt = 0; kt < 128 / 8; kt++) {
                    simdgroup_matrix<float, 8, 8> a, b;
                    // a = kc tile [l = rt, d = kt].
                    simdgroup_load(a, &kc[rt * 8 * 128 + kt * 8], 128);
                    // b = S_0ᵀ tile [d = kt, vi = ct]: `state` is [vi,d]
                    //     row-major (stride 128) — transpose to [d,vi].
                    simdgroup_load(
                        b,
                        &state[head_id * 128 * 128 + ct * 8 * 128 + kt * 8],
                        128, ulong2(0, 0), /*transpose=*/true);
                    if (kt == 0) {
                        simdgroup_multiply(sacc, a, b);
                    } else {
                        simdgroup_multiply_accumulate(sacc, a, b, sacc);
                    }
                }
                simdgroup_store(sacc, &sdc[rt * 8 * 128 + ct * 8], 128);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Epilogue: thread vi owns column vi of U for the rest of the
        // chunk. s0k[l,vi] now lives in sdc[l*128 + vi].
        for (uint l = 0; l < c; l++) {
            uint t = chunk_start + l;
            float v_l = conv_out[t * conv_per_token + 2 * key_total
                                 + head_id * 128 + vi];
            float gamma_l = precise::exp(log_decay[l]);
            Umat[l * 128 + vi] =
                beta_s[l] * v_l - beta_s[l] * gamma_l * sdc[l * 128 + vi];
        }

        // --- Phase 4: forward substitution (column vi is thread-private,
        //     so no barriers) — U_l -= sum_{i<l} A_{l,i} U_i ---
        for (uint l = 0; l < c; l++) {
            float acc = Umat[l * 128 + vi];
            for (uint i = 0; i < l; i++) {
                acc -= Amat[l * CW_C + i] * Umat[i * 128 + vi];
            }
            Umat[l * 128 + vi] = acc;
        }

        // --- Phase 5: output  out_l = gamma_l (S_0.q_l)
        //     + sum_{i<=l} Gamma_{l,i}(k_i.q_l) U_i ---
        // The kqg.U term is a GEMM  kqg[C,C] @ U[C,D] -> [C,D],
        // contraction over i (C=16). kqg already carries the i<=l mask
        // from Phase 2, so the dense matmul is exact. But the contraction
        // axis IS the ragged axis here: zero-fill U rows c..CW_C first —
        // a stale (possibly uninitialized → NaN) U row would poison every
        // output via 0*NaN. The S_0.q term (s0q) stays per-vi scalar
        // (Tier 2 — q is not tg-staged).
        for (uint i = c; i < CW_C; i++) {
            Umat[i * 128 + vi] = 0.0f;
        }
        // Barrier: the GEMM reads U across all columns; until now phases
        // 4->5 needed none (the scalar code read only its own vi column).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint sg = vi / 32;
            uint col_tiles = 128 / 8;               // D / 8 = 16
            uint n_tiles = (CW_C / 8) * col_tiles;  // 2 * 16 = 32
            for (uint ti = sg; ti < n_tiles; ti += 4) {
                uint rt = ti / col_tiles;   // C row tile (0..1)
                uint ct = ti % col_tiles;   // D col tile (0..15)
                simdgroup_matrix<float, 8, 8> oacc;
                // Contraction over i = C (CW_C/8 tiles of 8).
                for (uint kt = 0; kt < CW_C / 8; kt++) {
                    simdgroup_matrix<float, 8, 8> a, b;
                    // a = kqg tile [l = rt, i = kt], stride CW_C.
                    simdgroup_load(a, &kqg[rt * 8 * CW_C + kt * 8], CW_C);
                    // b = U tile [i = kt, vi = ct], stride 128.
                    simdgroup_load(b, &Umat[kt * 8 * 128 + ct * 8], 128);
                    if (kt == 0) {
                        simdgroup_multiply(oacc, a, b);
                    } else {
                        simdgroup_multiply_accumulate(oacc, a, b, oacc);
                    }
                }
                simdgroup_store(oacc, &sdc[rt * 8 * 128 + ct * 8], 128);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Epilogue: thread vi owns output column vi. The kqg.U term is
        // now in sdc[l*128 + vi]; s0q stays per-vi scalar.
        for (uint l = 0; l < c; l++) {
            uint t = chunk_start + l;
            uint q_base = t * conv_per_token + k_off;
            float s0q = 0.0f;
            for (uint d = 0; d < 128; d++) {
                s0q += state[state_base + d] * conv_out[q_base + d];
            }
            float out_val =
                precise::exp(log_decay[l]) * s0q + sdc[l * 128 + vi];
            output[t * value_total + head_id * 128 + vi] = out_val;
        }

        // --- Phase 6: new state  S_C = gamma_{c-1} S_0
        //     + sum_i Gamma_{c-1,i} U_i k_i^T ---
        // As a GEMM: delta[D,D] = RUᵀ[D,C] @ kc[C,D], with
        //   RU[i,vi] = exp(L_{c-1} - L_i) * U[i,vi].
        // The 128 threads act as 4 simdgroups (sg = vi/32) that
        // cooperatively tile the matmul; the gamma·S0 + delta epilogue
        // stays per-vi scalar (precision-sensitive). delta is too large
        // for threadgroup memory ([D,D] = 64 KB), so it is produced one
        // [D,CW_STRIP] column strip at a time into `sdc`.
        if (c > 0) {
            uint last = c - 1;
            float g_last = precise::exp(log_decay[last]);
            uint sg = vi / 32;   // simdgroup index (1-D threadgroup)

            // (a) Overwrite the (now-dead) U matrix in place with RU,
            //     and zero-fill ragged rows c..CW_C of both contraction
            //     operands — simdgroup_load always reads a full 8x8
            //     tile, and 0 * NaN = NaN, so a stale row must be
            //     cleared in *both* RU and kc, not just one.
            for (uint i = 0; i < c; i++) {
                Umat[i * 128 + vi] *=
                    precise::exp(log_decay[last] - log_decay[i]);
            }
            for (uint i = c; i < CW_C; i++) {
                Umat[i * 128 + vi] = 0.0f;
                kc[i * 128 + vi] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // (b)+(c) per column strip: cooperative matmul -> sdc,
            //     then per-vi scalar  state = g_last*S0 + delta.
            for (uint cs = 0; cs < 128; cs += CW_STRIP) {
                // Output tiles of this [D,CW_STRIP] strip: (D/8) row
                // tiles x (CW_STRIP/8) col tiles, split across the 4
                // simdgroups.
                uint col_tiles = CW_STRIP / 8;
                uint n_tiles = (128 / 8) * col_tiles;
                for (uint ti = sg; ti < n_tiles; ti += 4) {
                    uint rt = ti / col_tiles;
                    uint ct = ti % col_tiles;
                    simdgroup_matrix<float, 8, 8> dacc;
                    // Contraction over C = CW_C (CW_C/8 tiles of 8).
                    for (uint kt = 0; kt < CW_C / 8; kt++) {
                        simdgroup_matrix<float, 8, 8> a, b;
                        // a = RUᵀ tile: RU[C,D] block loaded transposed.
                        simdgroup_load(a, &Umat[kt * 8 * 128 + rt * 8],
                                       128, ulong2(0, 0),
                                       /*transpose=*/true);
                        // b = kc tile: kc[C,D] block, no transpose.
                        simdgroup_load(b,
                                       &kc[kt * 8 * 128 + cs + ct * 8],
                                       128);
                        if (kt == 0) {
                            simdgroup_multiply(dacc, a, b);
                        } else {
                            simdgroup_multiply_accumulate(dacc, a, b,
                                                          dacc);
                        }
                    }
                    simdgroup_store(dacc,
                                    &sdc[rt * 8 * CW_STRIP + ct * 8],
                                    CW_STRIP);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // (c) epilogue: thread vi owns state row vi.
                for (uint j = 0; j < CW_STRIP; j++) {
                    uint ki = cs + j;
                    state[state_base + ki] =
                        g_last * state[state_base + ki]
                        + sdc[vi * CW_STRIP + j];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


// ============================================================================
// Kernel 11: Conv1d depthwise step — batched compute (causal depthwise conv)
// ============================================================================
//
// Width-4 causal depthwise 1D convolution over a token chunk. The virtual
// sequence is `[conv_state[0..3], input[0..n_tokens]]`; output token `t`
// convolves virtual positions `[t, t+1, t+2, t+3]`:
//   acc = sum_{j=0..3} virtual[t+j][c] * weight[c][j]
//   output[t][c] = SiLU(acc) = acc / (1 + exp(-acc))
// where virtual[vp] = conv_state[vp*conv_dim+c] if vp<3 else input[(vp-3)*..].
//
// This kernel is compute-only: it reads `conv_state` but does NOT write it.
// The history shift is a separate dispatch (`conv1d_state_update`) so the
// state read/write hazard across token threadgroups can't race.
//
// Weight layout: [channels * 4] bf16, weight[c*4 + j].
// Conv state layout: [3 * conv_dim] row-major, state[r*conv_dim + c].
//
// Dispatch: ((conv_dim+255)/256, n_tokens) threadgroups × 256 threads.
//   gid.x = channel (guarded), gid.y = token.

kernel void conv1d_step(
    device const float *conv_state,   // [3 * conv_dim] history (read-only)
    device const float *input,        // [n_tokens * conv_dim]
    device const uint16_t *weights,   // [conv_dim * 4] bf16 as uint16
    device float *output,             // [n_tokens * conv_dim]
    constant uint &conv_dim,          // = 12288
    uint3 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint t = gid.y;
    if (c >= conv_dim) return;

    uint w_base = c * 4;
    float acc = 0.0f;
    for (uint j = 0; j < 4; j++) {
        uint vp = t + j;
        float val = (vp < 3)
            ? conv_state[vp * conv_dim + c]
            : input[(vp - 3) * conv_dim + c];
        acc += val * bf16_to_f32(weights[w_base + j]);
    }
    // SiLU activation
    output[t * conv_dim + c] = acc / (1.0f + exp(-acc));
}

// ============================================================================
// Kernel 11b: Conv1d history-state update (companion to conv1d_step)
// ============================================================================
//
// After the chunk's outputs are computed, the new `conv_state` is the last
// 3 entries of the virtual sequence `[conv_state[0..3], input[0..n_tokens]]`:
//   new_state[r] = virtual[n_tokens + r]   for r in {0,1,2}
// For n_tokens>=3 this is just input rows [n-3, n-2, n-1]; for n_tokens<3 it
// straddles old state and input — the `vp<3` branch handles both uniformly.
//
// One thread per channel: the thread reads all 3 source values into
// registers BEFORE writing any, so the n_tokens<3 case (which reads
// `conv_state` while writing it) has no cross-thread race — each thread
// touches only its own channel's 3 rows.
//
// Dispatch: ((conv_dim+255)/256) threadgroups × 256 threads.

kernel void conv1d_state_update(
    device float *conv_state,         // [3 * conv_dim] in/out
    device const float *input,        // [n_tokens * conv_dim]
    constant uint &conv_dim,
    constant uint &n_tokens,
    uint c [[thread_position_in_grid]]
) {
    if (c >= conv_dim) return;
    float nv[3];
    for (uint r = 0; r < 3; r++) {
        uint vp = n_tokens + r;
        nv[r] = (vp < 3)
            ? conv_state[vp * conv_dim + c]
            : input[(vp - 3) * conv_dim + c];
    }
    for (uint r = 0; r < 3; r++) {
        conv_state[r * conv_dim + c] = nv[r];
    }
}


// ============================================================================
// Kernel 12: Per-head RMS normalize for q and k vectors (batched)
// ============================================================================
// `x` is the in/out token-major buffer [n_tokens * per_token_total]; each
// token's q region starts at `t*per_token_total`, its k region at
// `t*per_token_total + key_offset_per_token`. Normalize each head
// independently; scale q by 1/sqrt(key_dim)^2, k by 1/sqrt(key_dim).
// Dispatch: (num_k_heads, n_tokens) threadgroups × key_dim threads.

kernel void rms_norm_qk(
    device float *x,                     // [n_tokens * per_token_total] in/out
    constant uint &key_dim,              // = 128
    constant float &inv_scale,           // = 1/sqrt(key_dim)
    constant uint &per_token_total,      // stride between tokens (floats)
    constant uint &key_offset_per_token, // q->k region offset (floats)
    uint3 tg3 [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]
) {
    uint head = tg3.x;
    uint t = tg3.y;
    uint tid = tid3.x;
    uint q_base = t * per_token_total + head * key_dim;
    uint k_base = q_base + key_offset_per_token;

    // RMS norm for q
    threadgroup float q_sum_sq;
    if (tid == 0) q_sum_sq = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qval = (tid < key_dim) ? x[q_base + tid] : 0;
    // Use threadgroup atomic add for sum of squares
    float q_sq_local = qval * qval;
    // Simple reduction: thread 0 accumulates (key_dim=128, fits in one pass)
    threadgroup float q_partial[128];
    q_partial[tid] = q_sq_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += q_partial[i];
        q_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_inv_rms = rsqrt(q_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        x[q_base + tid] = qval * q_inv_rms * inv_scale * inv_scale;  // q gets extra scale
    }

    // RMS norm for k
    threadgroup float k_sum_sq;
    float kval = (tid < key_dim) ? x[k_base + tid] : 0;
    threadgroup float k_partial[128];
    k_partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += k_partial[i];
        k_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_inv_rms = rsqrt(k_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        x[k_base + tid] = kval * k_inv_rms * inv_scale;
    }
}


// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet (batched)
// ============================================================================
// Per (v-head, token): g_decay = exp(-A * softplus(alpha + dt_bias)),
//   beta_gate = sigmoid(beta). alpha/beta/g_decay/beta_gate are token-major
//   [n_tokens * num_v_heads]; A_log/dt_bias are per-head, shared across tokens.
// Dispatch: (n_tokens) threadgroups × (num_v_heads) threads — `idx` flattens
//   to `t * num_v_heads + head`, so `head = idx % num_v_heads`.

kernel void compute_decay_beta(
    device const float *alpha_out,   // [n_tokens * num_v_heads] from projection
    device const float *beta_out,    // [n_tokens * num_v_heads] from projection
    device const float *A_log,       // [num_v_heads] log of decay base (shared)
    device const uint16_t *dt_bias,  // [num_v_heads] bf16 (shared)
    device float *g_decay,           // [n_tokens * num_v_heads] output
    device float *beta_gate,         // [n_tokens * num_v_heads] output
    constant uint &num_v_heads,
    uint idx [[thread_position_in_grid]]
) {
    uint head = idx % num_v_heads;
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[head]);
    float A_val = exp(A_log[head]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}


// ============================================================================
// Kernel 14: Gated RMS norm (z-gated output normalization, batched)
// ============================================================================
// output[i] = rms_norm(values[i]) * SiLU(z[i]) * weight[i]
// Per (v-head, token): normalize values, gate with z, scale with weight.
// values/z/output are token-major [n_tokens * num_v_heads * value_dim];
// weight is [value_dim], shared across heads and tokens.
// Dispatch: (num_v_heads, n_tokens) threadgroups × value_dim threads.

kernel void gated_rms_norm(
    device const float *values,       // [n_tokens * num_v_heads * value_dim]
    device const float *z,            // [n_tokens * num_v_heads * value_dim]
    device const uint16_t *weight,    // [value_dim] bf16 norm weights (shared)
    device float *output,             // [n_tokens * num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant float &eps,              // = 1e-6
    constant uint &num_v_heads,
    uint3 tg3 [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]
) {
    uint head = tg3.x;
    uint t = tg3.y;
    uint tid = tid3.x;
    uint base = (t * num_v_heads + head) * value_dim;

    float val = (tid < value_dim) ? values[base + tid] : 0;

    // RMS norm reduction
    threadgroup float partial[128];
    partial[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < value_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(value_dim) + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));  // SiLU
        float w = bf16_to_f32(weight[tid]);
        output[base + tid] = normed * gate * w;
    }
}


// ============================================================================
// Kernel 12: MoE combine + residual + shared expert gate (fused)
// ============================================================================
// Fused operation for CMD3 GPU-side combine:
//   hidden[i] = h_mid[i] + sum_k(expert_weight[k] * expert_out[k][i])
//               + sigmoid(shared_gate_score) * shared_out[i]
//
// All MAX_K=16 expert output buffers are always bound (unused ones have weight=0).
// This avoids variable buffer bindings and keeps the dispatch simple.
//
// Dispatch: (dim + 255) / 256 threadgroups, 256 threads each.

kernel void moe_combine_residual(
    device const float* h_mid       [[buffer(0)]],   // [dim]
    device const float* shared_out  [[buffer(1)]],   // [dim]
    device float*       hidden_out  [[buffer(2)]],   // [dim] output
    device const float* expert_out0 [[buffer(3)]],   // [dim] expert 0
    device const float* expert_out1 [[buffer(4)]],   // [dim] expert 1
    device const float* expert_out2 [[buffer(5)]],   // [dim] expert 2
    device const float* expert_out3 [[buffer(6)]],   // [dim] expert 3
    device const float* expert_out4 [[buffer(7)]],   // [dim] expert 4
    device const float* expert_out5 [[buffer(8)]],   // [dim] expert 5
    device const float* expert_out6 [[buffer(9)]],   // [dim] expert 6
    device const float* expert_out7  [[buffer(10)]], // [dim] expert 7
    device const float* expert_out8  [[buffer(11)]], // [dim] expert 8
    device const float* expert_out9  [[buffer(12)]], // [dim] expert 9
    device const float* expert_out10 [[buffer(13)]], // [dim] expert 10
    device const float* expert_out11 [[buffer(14)]], // [dim] expert 11
    device const float* expert_out12 [[buffer(15)]], // [dim] expert 12
    device const float* expert_out13 [[buffer(16)]], // [dim] expert 13
    device const float* expert_out14 [[buffer(17)]], // [dim] expert 14
    device const float* expert_out15 [[buffer(18)]], // [dim] expert 15
    device const float* params       [[buffer(19)]], // [18]: weights[0..15], shared_gate_score, (unused)
    constant uint&      dim          [[buffer(20)]],
    constant uint&      K            [[buffer(21)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    // Read expert weights and shared gate from params buffer.
    // Layout: params[0..15] = expert weights, params[16] = shared_gate_score, params[17] = pad.
    float shared_gate = 1.0f / (1.0f + exp(-params[16]));  // sigmoid(shared_gate_score)

    // Weighted sum of expert outputs
    float moe = 0.0f;
    // Unrolled for MAX_K=16 with branch on K to avoid reading invalid buffers
    if (K >  0) moe += params[ 0] * expert_out0[tid];
    if (K >  1) moe += params[ 1] * expert_out1[tid];
    if (K >  2) moe += params[ 2] * expert_out2[tid];
    if (K >  3) moe += params[ 3] * expert_out3[tid];
    if (K >  4) moe += params[ 4] * expert_out4[tid];
    if (K >  5) moe += params[ 5] * expert_out5[tid];
    if (K >  6) moe += params[ 6] * expert_out6[tid];
    if (K >  7) moe += params[ 7] * expert_out7[tid];
    if (K >  8) moe += params[ 8] * expert_out8[tid];
    if (K >  9) moe += params[ 9] * expert_out9[tid];
    if (K > 10) moe += params[10] * expert_out10[tid];
    if (K > 11) moe += params[11] * expert_out11[tid];
    if (K > 12) moe += params[12] * expert_out12[tid];
    if (K > 13) moe += params[13] * expert_out13[tid];
    if (K > 14) moe += params[14] * expert_out14[tid];
    if (K > 15) moe += params[15] * expert_out15[tid];

    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}

// ============================================================================
// Kernel 12b: MoE combine + residual (UNSCALED shared-expert variant)
// ============================================================================
// Sibling of `moe_combine_residual` for DeepSeek-V3 / Cogito-V2: the
// shared-expert output is added unconditionally, no sigmoid, no gate
// scalar. Mirrors `deepseek_moe_cpu` (moe_cpu.rs:168-173): raw
// `out[i] += shared[i]`. Variant-flagged by `VARIANT.shared_expert_gate
// == Unscaled` (variants.rs:115). Same buffer layout as the parent
// kernel so the dispatcher just swaps PSOs.
//
// Dispatch:
//   threadgroups = (ceil(dim/256), 1, 1)
//   threads      = (256, 1, 1)
//
// Note: `params[16]` (shared_gate_score) is bound to keep the params
// layout identical, but the kernel does not read it.

kernel void moe_combine_residual_unscaled(
    device const float* h_mid       [[buffer(0)]],   // [dim]
    device const float* shared_out  [[buffer(1)]],   // [dim]
    device float*       hidden_out  [[buffer(2)]],   // [dim] output
    device const float* expert_out0 [[buffer(3)]],
    device const float* expert_out1 [[buffer(4)]],
    device const float* expert_out2 [[buffer(5)]],
    device const float* expert_out3 [[buffer(6)]],
    device const float* expert_out4 [[buffer(7)]],
    device const float* expert_out5 [[buffer(8)]],
    device const float* expert_out6 [[buffer(9)]],
    device const float* expert_out7  [[buffer(10)]],
    device const float* expert_out8  [[buffer(11)]],
    device const float* expert_out9  [[buffer(12)]],
    device const float* expert_out10 [[buffer(13)]],
    device const float* expert_out11 [[buffer(14)]],
    device const float* expert_out12 [[buffer(15)]],
    device const float* expert_out13 [[buffer(16)]],
    device const float* expert_out14 [[buffer(17)]],
    device const float* expert_out15 [[buffer(18)]],
    device const float* params       [[buffer(19)]], // [18]: weights[0..15], (16/17 unused here)
    constant uint&      dim          [[buffer(20)]],
    constant uint&      K            [[buffer(21)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float moe = 0.0f;
    if (K >  0) moe += params[ 0] * expert_out0[tid];
    if (K >  1) moe += params[ 1] * expert_out1[tid];
    if (K >  2) moe += params[ 2] * expert_out2[tid];
    if (K >  3) moe += params[ 3] * expert_out3[tid];
    if (K >  4) moe += params[ 4] * expert_out4[tid];
    if (K >  5) moe += params[ 5] * expert_out5[tid];
    if (K >  6) moe += params[ 6] * expert_out6[tid];
    if (K >  7) moe += params[ 7] * expert_out7[tid];
    if (K >  8) moe += params[ 8] * expert_out8[tid];
    if (K >  9) moe += params[ 9] * expert_out9[tid];
    if (K > 10) moe += params[10] * expert_out10[tid];
    if (K > 11) moe += params[11] * expert_out11[tid];
    if (K > 12) moe += params[12] * expert_out12[tid];
    if (K > 13) moe += params[13] * expert_out13[tid];
    if (K > 14) moe += params[14] * expert_out14[tid];
    if (K > 15) moe += params[15] * expert_out15[tid];

    hidden_out[tid] = h_mid[tid] + moe + shared_out[tid];
}

// ============================================================================
// Kernel 12c: MoE bucket-accumulate (batched permute-and-fuse)
// ============================================================================
// Scatter-weighted add for the batched MoE permute-and-fuse path:
//   out_sum[token_idx[b] * hidden_dim + h]
//       += weights[b] * bucket_out[b * hidden_dim + h]
//
// One dispatch per non-empty expert bucket. `bucket_size` is the number
// of (token, slot) tuples that picked this expert; `bucket_out` is the
// stacked expert outputs `[bucket_size, hidden_dim]` produced by the
// gate/up/swiglu/down sequence over the bucket's stacked inputs.
//
// Atomic-free: top-K returns distinct experts per token, so the same
// `token_idx[b]` value never appears twice within one bucket — no two
// threads write to the same `out_sum` slot within a single dispatch.
// Cross-bucket accumulation is handled by Metal's encoder-internal
// serialization (we dispatch buckets sequentially in one cmdbuf).
//
// Dispatch:
//   threadgroups = (bucket_size, ceil(hidden_dim / 256), 1)
//   threads      = (1, 256, 1)

kernel void moe_bucket_accumulate(
    device const float* bucket_out  [[buffer(0)]],   // [bucket_size, hidden_dim]
    device const int*   token_idx   [[buffer(1)]],   // [bucket_size]
    device const float* weights     [[buffer(2)]],   // [bucket_size]
    device float*       out_sum     [[buffer(3)]],   // [n_tokens, hidden_dim]
    constant uint&      hidden_dim  [[buffer(4)]],
    constant uint&      bucket_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.x;
    uint h = gid.y;
    if (b >= bucket_size || h >= hidden_dim) return;

    float val = bucket_out[b * hidden_dim + h] * weights[b];
    uint  t   = (uint)token_idx[b];
    out_sum[t * hidden_dim + h] += val;
}

// ============================================================================
// Kernel 13: YaRN RoPE (DeepSeek-V3 / Cogito-V2 MLA)
// ============================================================================
// In-place rotation of a `[num_heads, rotary_dim]` buffer using a
// pre-computed inv_freq table and a position scalar with mscale baked
// into both cos and sin terms. Mirrors `apply_rotary_emb_yarn`
// (rope.rs:306) bit-for-bit modulo libm-vs-Metal-fast-math drift.
//
// Pairing convention: x[h, i] paired with x[h, i + half] (MLX style),
// where half = rotary_dim / 2.
//
// Dispatch:
//   threadgroups = (num_heads, half, 1)
//   threads      = (1, 1, 1)              (one rotation per thread)
//
// One thread handles one (head, i) pair; no inter-thread coordination
// needed. The trivial threadgroup geometry is fine because num_heads ×
// half ≈ 128 * 32 = 4096 threads for Cogito-V2 — plenty of work for
// the GPU without any reduction.

kernel void yarn_rope_apply(
    device   float*       x          [[buffer(0)]],   // [num_heads, rotary_dim]
    constant float*       inv_freq   [[buffer(1)]],   // [half]
    constant uint&        num_heads  [[buffer(2)]],
    constant uint&        rotary_dim [[buffer(3)]],
    constant float&       pos_f      [[buffer(4)]],
    constant float&       mscale     [[buffer(5)]],
    uint2 tg_pos [[threadgroup_position_in_grid]]
) {
    uint h        = tg_pos.x;
    uint i        = tg_pos.y;
    uint half_dim = rotary_dim / 2;
    if (h >= num_heads || i >= half_dim) return;

    // `precise::cos` / `precise::sin` keep accuracy at large angles
    // (RoPE positions can drive `pos * inv_freq[i]` into the
    // thousands). Default Metal `cos`/`sin` are fast-math and lose
    // accuracy badly past a few rotations — measured ~3e-4 absolute
    // drift vs libm `cosf` at pos=4096 in the CPU diff oracle.
    float angle = pos_f * inv_freq[i];
    float cos_a = metal::precise::cos(angle) * mscale;
    float sin_a = metal::precise::sin(angle) * mscale;

    uint  base = h * rotary_dim;
    float x0   = x[base + i];
    float x1   = x[base + i + half_dim];
    x[base + i]            = x0 * cos_a - x1 * sin_a;
    x[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
}

// ============================================================================
// Kernel 14: MLA folded — q' = q_nope @ kv_b_proj_K_per_head (4-bit)
// ============================================================================
// Computes per-head:
//   q'[h, c] = Σ_{i=0..nope} q_nope[h, i] * dequant(W[h * kv_b_per_head + i, c])
// where W is `kv_b_proj` (`[num_heads * kv_b_per_head, kv_lora_rank]`,
// 4-bit affine MLX layout). The K-portion uses rows
// `[h * kv_b_per_head, h * kv_b_per_head + nope)`; the V-portion sits
// in the next `v_head_dim` rows and is consumed by `mla_out_per_head_4bit`.
//
// Dispatch:
//   threadgroups = ((num_heads * kv_lora_rank + 255) / 256, 1, 1)
//   threads      = (256, 1, 1)
// Each thread owns one output element (h, c) and runs the full 128-step
// dot product. With 65,536 outputs for Cogito-V2 the geometry is fine
// without tiling.
//
// Memory access is "row-wise" relative to standard matvec (varying
// row index inside the dot product), so we don't reuse
// `dequant_matvec_4bit_v3`'s per-row-per-SIMD pattern. Each thread reads
// 128 group-scale/bias pairs and 128 nibbles per output — uncoalesced
// at the byte level but the working set per thread is small.
//
// Layout invariants (assumed):
// - `group_size` divides `kv_lora_rank` (64 | 512 ✓)
// - 8 nibbles per packed uint32 (4 bits × 8 = 32 bits ✓)
// - scales/biases stored row-major `[num_heads * kv_b_per_head, num_groups]`

kernel void mla_q_prime_4bit(
    device const uint32_t* W_packed     [[buffer(0)]],   // [num_heads * kv_b_per_head, kv_lora_rank/8]
    device const uint16_t* scales       [[buffer(1)]],   // [num_heads * kv_b_per_head, num_groups]
    device const uint16_t* biases       [[buffer(2)]],   // [num_heads * kv_b_per_head, num_groups]
    device const float*    q_nope       [[buffer(3)]],   // [num_heads, nope]
    device float*          q_prime      [[buffer(4)]],   // [num_heads, kv_lora_rank]
    constant uint&         num_heads    [[buffer(5)]],
    constant uint&         nope         [[buffer(6)]],
    constant uint&         kv_lora_rank [[buffer(7)]],
    constant uint&         kv_b_per_head[[buffer(8)]],
    constant uint&         group_size   [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = num_heads * kv_lora_rank;
    if (tid >= total) return;
    uint h = tid / kv_lora_rank;
    uint c = tid - h * kv_lora_rank;

    uint num_groups   = kv_lora_rank / group_size;
    uint packed_cols  = kv_lora_rank / 8;
    uint g            = c / group_size;          // which group on the row
    uint c_in_packed  = c >> 3;                  // = c / 8
    uint c_nibble     = c & 7;                   // = c % 8
    uint nibble_shift = c_nibble * 4;

    float acc = 0.0f;
    uint base_row = h * kv_b_per_head;
    for (uint i = 0; i < nope; ++i) {
        uint row = base_row + i;
        float scale = bf16_to_f32(scales[row * num_groups + g]);
        float bias  = bf16_to_f32(biases[row * num_groups + g]);
        uint32_t packed = W_packed[row * packed_cols + c_in_packed];
        float nib = float((packed >> nibble_shift) & 0xF);
        float w = nib * scale + bias;
        acc += q_nope[h * nope + i] * w;
    }
    q_prime[h * kv_lora_rank + c] = acc;
}

// ============================================================================
// Kernel 15: MLA folded — out_per_head = V_combine @ kv_b_proj_V_per_head (4-bit)
// ============================================================================
// Computes per-head:
//   out[h, f] = Σ_{c=0..kv_lora_rank}
//                  V_combine[h, c] * dequant(W[h * kv_b_per_head + nope + f, c])
// V-portion rows of `kv_b_proj` are contiguous per head — same packed
// matrix as `mla_q_prime_4bit`, just different row offsets.
//
// Dispatch:
//   threadgroups = ((num_heads * v_head_dim + 31) / 32, 1, 1)
//   threads      = (32, 1, 1)        (one SIMD group per output element)
// Threads in a SIMD group cooperate on the 512-wide dot product via
// `simd_sum`. Lane k handles columns k, k+32, k+64, … (stride 32).

kernel void mla_out_per_head_4bit(
    device const uint32_t* W_packed      [[buffer(0)]],
    device const uint16_t* scales        [[buffer(1)]],
    device const uint16_t* biases        [[buffer(2)]],
    device const float*    v_combine     [[buffer(3)]],   // [num_heads, kv_lora_rank]
    device float*          out_per_head  [[buffer(4)]],   // [num_heads, v_head_dim]
    constant uint&         num_heads     [[buffer(5)]],
    constant uint&         nope          [[buffer(6)]],
    constant uint&         kv_lora_rank  [[buffer(7)]],
    constant uint&         v_head_dim    [[buffer(8)]],
    constant uint&         kv_b_per_head [[buffer(9)]],
    constant uint&         group_size    [[buffer(10)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    uint total_outputs = num_heads * v_head_dim;
    if (tgid >= total_outputs) return;
    uint h = tgid / v_head_dim;
    uint f = tgid - h * v_head_dim;

    uint row = h * kv_b_per_head + nope + f;
    uint num_groups  = kv_lora_rank / group_size;
    uint packed_cols = kv_lora_rank / 8;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales   + row * num_groups;
    device const uint16_t* b_row = biases   + row * num_groups;
    device const float*    v_h   = v_combine + h * kv_lora_rank;

    // Lane k processes packed columns k, k+32, k+64, … (each carries
    // 8 nibbles → 8 input dims).
    float acc = 0.0f;
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);
        uint32_t packed = w_row[col];
        uint x_base = col * 8;
        // Standard 8-nibble fused dequant·multiply (mirrors v3 kernel).
        for (uint k = 0; k < 8; ++k) {
            float nib = float((packed >> (k * 4)) & 0xF);
            float w   = nib * scale + bias;
            acc       = fma(v_h[x_base + k], w, acc);
        }
    }
    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out_per_head[h * v_head_dim + f] = sum;
    }
}

// ============================================================================
// Kernel 16: MLA folded — SDPA over latent + rope-K cache
// ============================================================================
// One threadgroup per attention head. Inside the head:
//
//   scores[t] = q'[h] · latent_cache[t] + q_pe[h] · rope_k_cache[t]
//   scores  *= scale         (= 1/sqrt(qk_head_dim) * mscale²)
//   softmax(scores) over t in 0..cache_len
//   v_combine[h, c] = Σ_t  scores[t] * latent_cache[t, c]
//
// Threads in the group cooperate on the dot products + softmax
// reductions. Geometry is `(num_heads, 1, 1)` threadgroups with
// `(THREADS_PER_HEAD = 128, 1, 1)` threads each — same as Cogito-V2's
// `qk_nope_head_dim`, but the value is unrelated to that dim; it's
// just chosen so each thread carries one cached-position slot for
// short contexts (`cache_len ≤ 128`) and tiles for longer ones.
//
// Dispatch:
//   threadgroups = (num_heads, 1, 1)
//   threads      = (THREADS_PER_HEAD, 1, 1)
//
// Cache-length cap: `scores[]` lives in 32 KB threadgroup memory;
// 4096 floats × 4 bytes = 16 KB leaves headroom for the
// `lane0_acc[]` simd-broadcast scratch (one float per simdgroup,
// 4 lanes/group max in this dispatch). Long-context tiling (100k+)
// is a follow-up — at that point we'll dispatch the scores in
// chunks instead of bumping the per-tg cap.
constant uint MLA_THREADS_PER_HEAD = 128;
constant uint MLA_MAX_CACHE_TG     = 4096;

kernel void mla_sdpa_folded(
    device const float* q_prime          [[buffer(0)]],   // [num_heads, kv_lora_rank]
    device const float* q_pe             [[buffer(1)]],   // [num_heads, qk_rope_head_dim]
    device const float* latent_cache     [[buffer(2)]],   // [cache_len, kv_lora_rank]
    device const float* rope_k_cache     [[buffer(3)]],   // [cache_len, qk_rope_head_dim]
    device float*       v_combine        [[buffer(4)]],   // [num_heads, kv_lora_rank]
    constant uint&      num_heads        [[buffer(5)]],
    constant uint&      kv_lora_rank     [[buffer(6)]],
    constant uint&      qk_rope_head_dim [[buffer(7)]],
    constant uint&      cache_len        [[buffer(8)]],
    constant float&     softmax_scale    [[buffer(9)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (tg >= num_heads) return;

    uint h = tg;
    threadgroup float scores[MLA_MAX_CACHE_TG];
    // One float per simdgroup (max 4 simdgroups @ 32 lanes ⇒ 128
    // threads). Used as the cross-simd scratch for max + sum
    // reductions, so we don't need single-element tg-shared
    // broadcasts that would push us over the 32 KB cap.
    threadgroup float simd_scratch[8];

    // ---- 1. scores[t] = q'[h] · latent[t] + q_pe[h] · rope_k[t] ----
    device const float* q_h    = q_prime + h * kv_lora_rank;
    device const float* q_pe_h = q_pe    + h * qk_rope_head_dim;
    for (uint t = lid; t < cache_len; t += MLA_THREADS_PER_HEAD) {
        device const float* lat_t = latent_cache  + t * kv_lora_rank;
        device const float* rkt   = rope_k_cache  + t * qk_rope_head_dim;
        float s = 0.0f;
        for (uint c = 0; c < kv_lora_rank; ++c) {
            s = fma(q_h[c], lat_t[c], s);
        }
        for (uint r = 0; r < qk_rope_head_dim; ++r) {
            s = fma(q_pe_h[r], rkt[r], s);
        }
        scores[t] = s * softmax_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 2. softmax(scores) — two-stage reduction ----
    // Stage A: each lane folds its strided slice; simd_max within
    // simdgroup; lane 0 of each simdgroup writes to simd_scratch;
    // lane 0 of simdgroup 0 reduces across simdgroups.
    uint simd_lane  = lid & 31;
    uint simd_group = lid >> 5;
    uint num_simdgroups = MLA_THREADS_PER_HEAD / 32;
    float local_max = -INFINITY;
    for (uint t = lid; t < cache_len; t += MLA_THREADS_PER_HEAD) {
        local_max = max(local_max, scores[t]);
    }
    float simd_max_v = simd_max(local_max);
    if (simd_lane == 0) { simd_scratch[simd_group] = simd_max_v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxv;
    if (simd_group == 0) {
        float v = (simd_lane < num_simdgroups)
            ? simd_scratch[simd_lane]
            : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) { simd_scratch[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxv = simd_scratch[0];

    // Stage B: exp + accumulate (same two-stage pattern for sum).
    float local_sum = 0.0f;
    for (uint t = lid; t < cache_len; t += MLA_THREADS_PER_HEAD) {
        float e = exp(scores[t] - maxv);
        scores[t] = e;
        local_sum += e;
    }
    float simd_sum_v = simd_sum(local_sum);
    if (simd_lane == 0) { simd_scratch[simd_group] = simd_sum_v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sum;
    if (simd_group == 0) {
        float v = (simd_lane < num_simdgroups)
            ? simd_scratch[simd_lane]
            : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) { simd_scratch[1] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_sum = simd_scratch[1];
    float inv_sum = 1.0f / total_sum;
    for (uint t = lid; t < cache_len; t += MLA_THREADS_PER_HEAD) {
        scores[t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 3. v_combine[h, c] = Σ_t scores[t] * latent[t, c] ----
    device float* vc_h = v_combine + h * kv_lora_rank;
    for (uint c = lid; c < kv_lora_rank; c += MLA_THREADS_PER_HEAD) {
        float acc = 0.0f;
        for (uint t = 0; t < cache_len; ++t) {
            acc = fma(scores[t], latent_cache[t * kv_lora_rank + c], acc);
        }
        vc_h[c] = acc;
    }
}


// ============================================================================
// Phase 6 — tiled folded SDPA for cache_len > MLA_MAX_CACHE_TG
// ============================================================================
//
// Flash-Attention-style online softmax across `cache_len` positions
// processed in chunks of `MLA_TILE_SIZE`. Two kernels:
//
//   1. `mla_sdpa_tile_accumulate` — process one tile, update the
//      running (max, denom, v_combine_partial) per head.
//   2. `mla_sdpa_tile_finalize` — divide v_combine_partial by denom
//      to produce the final v_combine.
//
// The running-state buffers must be sized [num_heads] (max, denom)
// and [num_heads, kv_lora_rank] (v_combine_partial). The dispatcher
// loops over tiles and dispatches `accumulate` per tile, then
// `finalize` once.
//
// Bit-exact-against-single-shot only for `cache_len == MLA_TILE_SIZE`
// (one tile, no merging). Multi-tile output is mathematically
// equivalent up to floating-point reordering — cosine ≥ 0.9999 vs
// the single-shot reference is the validation target.

constant uint MLA_TILE_SIZE = 4096;

kernel void mla_sdpa_tile_accumulate(
    device const float* q_prime              [[buffer(0)]],
    device const float* q_pe                 [[buffer(1)]],
    device const float* latent_cache         [[buffer(2)]],
    device const float* rope_k_cache         [[buffer(3)]],
    device float*       running_max          [[buffer(4)]],
    device float*       running_denom        [[buffer(5)]],
    device float*       v_combine_partial    [[buffer(6)]],
    constant uint&      num_heads            [[buffer(7)]],
    constant uint&      kv_lora_rank         [[buffer(8)]],
    constant uint&      qk_rope_head_dim     [[buffer(9)]],
    constant uint&      tile_start           [[buffer(10)]],
    constant uint&      tile_end             [[buffer(11)]],
    constant float&     softmax_scale        [[buffer(12)]],
    constant uint&      is_first_tile        [[buffer(13)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (tg >= num_heads) return;
    uint h = tg;
    uint tile_size = tile_end - tile_start;
    if (tile_size == 0) return;

    threadgroup float scores[MLA_TILE_SIZE];
    threadgroup float simd_scratch[8];

    device const float* q_h    = q_prime + h * kv_lora_rank;
    device const float* q_pe_h = q_pe    + h * qk_rope_head_dim;

    // 1. scores[i] = (q'_h · latent[tile_start+i]) + (q_pe_h · rope_k[tile_start+i])
    for (uint i = lid; i < tile_size; i += MLA_THREADS_PER_HEAD) {
        uint t = tile_start + i;
        device const float* lat_t = latent_cache + t * kv_lora_rank;
        device const float* rkt   = rope_k_cache + t * qk_rope_head_dim;
        float s = 0.0f;
        for (uint c = 0; c < kv_lora_rank; ++c) {
            s = fma(q_h[c], lat_t[c], s);
        }
        for (uint r = 0; r < qk_rope_head_dim; ++r) {
            s = fma(q_pe_h[r], rkt[r], s);
        }
        scores[i] = s * softmax_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. tile_max = max over scores[0..tile_size)
    uint simd_lane  = lid & 31;
    uint simd_group = lid >> 5;
    uint num_simdgroups = MLA_THREADS_PER_HEAD / 32;
    float local_max = -INFINITY;
    for (uint i = lid; i < tile_size; i += MLA_THREADS_PER_HEAD) {
        local_max = max(local_max, scores[i]);
    }
    float simd_max_v = simd_max(local_max);
    if (simd_lane == 0) { simd_scratch[simd_group] = simd_max_v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tile_max;
    if (simd_group == 0) {
        float v = (simd_lane < num_simdgroups)
            ? simd_scratch[simd_lane]
            : -INFINITY;
        v = simd_max(v);
        if (simd_lane == 0) { simd_scratch[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_max = simd_scratch[0];

    // 3. exp(scores - tile_max) into scores[], tile_sum.
    float local_sum = 0.0f;
    for (uint i = lid; i < tile_size; i += MLA_THREADS_PER_HEAD) {
        float e = exp(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    float simd_sum_v = simd_sum(local_sum);
    if (simd_lane == 0) { simd_scratch[simd_group] = simd_sum_v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tile_sum;
    if (simd_group == 0) {
        float v = (simd_lane < num_simdgroups)
            ? simd_scratch[simd_lane]
            : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) { simd_scratch[1] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_sum = simd_scratch[1];

    // 4. Merge with running state.
    //    new_max   = max(prev_max, tile_max)
    //    scale_old = exp(prev_max - new_max)
    //    scale_new = exp(tile_max  - new_max)
    //    new_denom = prev_denom * scale_old + tile_sum * scale_new
    //    V[c]      = prev_V[c]  * scale_old + tile_partial[c] * scale_new
    float prev_max   = (is_first_tile != 0u) ? -INFINITY : running_max[h];
    float prev_denom = (is_first_tile != 0u) ? 0.0f      : running_denom[h];
    float new_max    = max(prev_max, tile_max);
    // exp(-INFINITY - new_max) = 0 by IEEE — covers the first-tile
    // branch without needing a special case. tile_max <= new_max so
    // scale_new is in [0, 1].
    float scale_old  = exp(prev_max - new_max);
    float scale_new  = exp(tile_max  - new_max);
    float new_denom  = prev_denom * scale_old + tile_sum * scale_new;

    device float* v_partial_h = v_combine_partial + h * kv_lora_rank;
    for (uint c = lid; c < kv_lora_rank; c += MLA_THREADS_PER_HEAD) {
        // Tile partial for this c: Σ_i exp(scores[i]-tile_max) * latent[tile_start+i, c].
        // scores[i] currently holds exp(scores[i] - tile_max).
        float tile_partial = 0.0f;
        for (uint i = 0; i < tile_size; ++i) {
            uint t = tile_start + i;
            tile_partial = fma(
                scores[i],
                latent_cache[t * kv_lora_rank + c],
                tile_partial);
        }
        float prev_v = (is_first_tile != 0u) ? 0.0f : v_partial_h[c];
        v_partial_h[c] = prev_v * scale_old + tile_partial * scale_new;
    }

    if (lid == 0) {
        running_max[h]   = new_max;
        running_denom[h] = new_denom;
    }
}

kernel void mla_sdpa_tile_finalize(
    device const float* v_combine_partial [[buffer(0)]],
    device const float* running_denom     [[buffer(1)]],
    device float*       v_combine         [[buffer(2)]],
    constant uint&      num_heads         [[buffer(3)]],
    constant uint&      kv_lora_rank      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = num_heads * kv_lora_rank;
    if (tid >= total) return;
    uint h = tid / kv_lora_rank;
    float d = running_denom[h];
    // Defensive: guard against denom=0 (would only happen on
    // cache_len=0 which the dispatcher filters).
    float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
    v_combine[tid] = v_combine_partial[tid] * inv;
}

// ============================================================================
// Kernel 17: MLA fan-out split (Phase 4a — kill MLA sync points)
// ============================================================================
// One dispatch fans `q_full` (post `q_b_proj`) and `kv_pre` (post
// `kv_a_proj_with_mqa`) out into the four downstream MLA buffers
// (`q_nope`, `q_pe`, `kv_lat`, `k_pe`). Replaces the host-side scatter
// at `mla_attn_forward.rs:380-407` so the whole MLA forward can stay
// in one Metal command buffer (Plan-of-record's Phase 4a refactor).
//
// Layout per head in `q_full`: `[q_nope_part | q_pe_part]` of widths
// `qk_nope_head_dim` and `qk_rope_head_dim`. Layout in `kv_pre`:
// `[kv_lat (kv_lora_rank) | k_pe (qk_rope_head_dim)]`.
//
// Dispatch:
//   threadgroups = ceil(max_out / 256)  where max_out = max(num_heads*qk_nope,
//                                                            num_heads*qk_rope,
//                                                            kv_lora_rank,
//                                                            qk_rope_head_dim)
//   threads      = (256, 1, 1)
//
// Each thread checks each output's bound and writes if in range —
// pure scatter, no math. Cogito-V2 max_out = 128 * 128 = 16 384.

kernel void mla_split_q_kv(
    device const float* q_full          [[buffer(0)]],
    device const float* kv_pre          [[buffer(1)]],
    device float*       q_nope          [[buffer(2)]],
    device float*       q_pe            [[buffer(3)]],
    device float*       kv_lat          [[buffer(4)]],
    device float*       k_pe            [[buffer(5)]],
    constant uint&      num_heads       [[buffer(6)]],
    constant uint&      qk_nope         [[buffer(7)]],
    constant uint&      qk_rope         [[buffer(8)]],
    constant uint&      kv_lora_rank    [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    uint qk_head_dim = qk_nope + qk_rope;

    // q_nope[h, i] = q_full[h, i]
    uint q_nope_total = num_heads * qk_nope;
    if (tid < q_nope_total) {
        uint h = tid / qk_nope;
        uint i = tid - h * qk_nope;
        q_nope[h * qk_nope + i] = q_full[h * qk_head_dim + i];
    }

    // q_pe[h, i] = q_full[h, qk_nope + i]
    uint q_pe_total = num_heads * qk_rope;
    if (tid < q_pe_total) {
        uint h = tid / qk_rope;
        uint i = tid - h * qk_rope;
        q_pe[h * qk_rope + i] = q_full[h * qk_head_dim + qk_nope + i];
    }

    // kv_lat[c] = kv_pre[c]
    if (tid < kv_lora_rank) {
        kv_lat[tid] = kv_pre[tid];
    }

    // k_pe[r] = kv_pre[kv_lora_rank + r]
    if (tid < qk_rope) {
        k_pe[tid] = kv_pre[kv_lora_rank + tid];
    }
}

// ============================================================================
// Kernel 18: MLA cache append (Phase 4a)
// ============================================================================
// Append `(kv_lat, k_pe)` into the per-layer MLA KV cache rows at
// position `pos`. Mirrors the host-side memcpy at
// `mla_attn_forward.rs:467-481`; runs as a Metal kernel so the whole
// forward stays in one cmdbuf.
//
// Caller is responsible for incrementing `kv_cache.len = pos + 1`
// after the cmdbuf commits — only the GPU-visible cache row update
// happens here; the Rust-side `len` field is invisible to the kernel.
//
// Dispatch:
//   threadgroups = ceil(max(kv_lora_rank, qk_rope_head_dim) / 256)
//   threads      = (256, 1, 1)

kernel void mla_kv_cache_append(
    device const float* kv_lat        [[buffer(0)]],
    device const float* k_pe          [[buffer(1)]],
    device float*       latent_cache  [[buffer(2)]],
    device float*       rope_k_cache  [[buffer(3)]],
    constant uint&      kv_lora_rank  [[buffer(4)]],
    constant uint&      qk_rope       [[buffer(5)]],
    constant int&       pos           [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint p = uint(pos);
    if (tid < kv_lora_rank) {
        latent_cache[p * kv_lora_rank + tid] = kv_lat[tid];
    }
    if (tid < qk_rope) {
        rope_k_cache[p * qk_rope + tid] = k_pe[tid];
    }
}


// ============================================================================
// Kernel: MoE combine + residual + shared expert gate, batched over N tokens
// ============================================================================
// Batched-prefill variant of the existing `moe_combine_residual` for the
// post-MoE-permute-fuse path. Inputs are already aggregated per token
// (the bucket-accumulate kernel summed expert outputs into `moe_sum`),
// so no per-expert weighted reduction needed here. Just:
//
//   hidden_out[t, i] = h_mid[t, i] + moe_sum[t, i]
//                    + sigmoid(shared_gate[t]) * shared_out[t, i]
//
// One thread per (token, dim) pair. Eliminates the CPU combine loop
// (`for t in 0..n_tokens { sigmoid + multiply + add }`) and lets the
// orchestrator keep hidden_out on GPU as the next layer's hidden_in.
//
// Dispatch:
//   threadgroups = (ceil(n_tokens * dim / 256), 1, 1)
//   threads      = (256, 1, 1)

kernel void moe_combine_residual_n_tokens(
    device const float* h_mid        [[buffer(0)]],  // [n_tokens, dim]
    device const float* moe_sum      [[buffer(1)]],  // [n_tokens, dim]
    device const float* shared_out   [[buffer(2)]],  // [n_tokens, dim]
    device const float* shared_gate  [[buffer(3)]],  // [n_tokens] f32 (pre-sigmoid)
    device float*       hidden_out   [[buffer(4)]],  // [n_tokens, dim]
    constant uint&      n_tokens     [[buffer(5)]],
    constant uint&      dim          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = n_tokens * dim;
    if (tid >= total) return;
    uint t = tid / dim;
    float sg = 1.0f / (1.0f + exp(-shared_gate[t]));
    hidden_out[tid] = h_mid[tid] + moe_sum[tid] + sg * shared_out[tid];
}


// ============================================================================
// Kernel: Fused RMS norm with bf16 weights, batched over N tokens
// ============================================================================
// One threadgroup per token. Each threadgroup computes its own sum-of-squares
// via simd_sum + cross-simd reduction in tg-mem, then applies
// `out[t, i] = x[t, i] * rsqrt(sum_sq/dim + eps) * bf16_to_f32(w[i])`.
//
// Fused vs the single-token chain (`rms_norm_sum_sq` + `rms_norm_apply_bf16`):
// `sum_sq` stays in tg-mem instead of going to global, so the second pass
// reads it via threadgroup broadcast — one cmdbuf, one dispatch, no
// intermediate global write.
//
// Dispatch:
//   threadgroups = (n_tokens, 1, 1)
//   threads      = (tg_size, 1, 1)    // 256 is the sweet spot
//
// Weights `w` are shared across tokens (per-channel norm weight).

kernel void rms_norm_bf16_fused_n_tokens(
    device const float*    x         [[buffer(0)]],   // [n_tokens, dim]
    device const uint16_t* weight    [[buffer(1)]],   // [dim] bf16
    device float*          out       [[buffer(2)]],   // [n_tokens, dim]
    constant uint&         dim       [[buffer(3)]],
    constant float&        eps       [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],   // token index
    uint lid      [[thread_position_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]]
) {
    device const float* xt = x   + tgid * dim;
    device       float* ot = out + tgid * dim;

    uint simd_lane       = lid % 32;
    uint simd_group      = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    // Phase 1: parallel sum-of-squares reduction.
    threadgroup float shared[32];
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float v = xt[i];
        acc += v * v;
    }
    float sm = simd_sum(acc);
    if (simd_lane == 0) shared[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        total = simd_sum(shared[simd_lane]);
    }
    threadgroup float bc_inv_rms;
    if (lid == 0) {
        bc_inv_rms = rsqrt(total / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = bc_inv_rms;

    // Phase 2: apply.
    for (uint i = lid; i < dim; i += tg_size) {
        float w = bf16_to_f32(weight[i]);
        ot[i] = xt[i] * inv_rms * w;
    }
}


// ============================================================================
// Kernel: Residual add batched over N tokens
// ============================================================================
// `out[t*dim + i] = a[t*dim + i] + b[t*dim + i]`. Trivially data-parallel —
// just enough infrastructure to share a single dispatch across the n_tokens
// stack instead of one cmdbuf per token.
//
// Dispatch:
//   threadgroups = (ceil(n_tokens * dim / 256), 1, 1)
//   threads      = (256, 1, 1)

kernel void residual_add_n_tokens(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      total   [[buffer(3)]],   // n_tokens * dim
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total) return;
    out[tid] = a[tid] + b[tid];
}


// ============================================================================
// Kernel: RoPE (rotary position embedding) batched over N tokens
// ============================================================================
// Vanilla RoPE in the GPT-NeoX (half-split) layout, batched across an
// `[n_tokens, num_heads, head_dim]` stack. Rotates the first `rotary_dim`
// channels of each head; channels [rotary_dim, head_dim) are left
// untouched (partial rotary). Token `t`'s absolute position is
// `start_pos + t`.
//
// `inv_freq` is a precomputed `rotary_dim/2`-length table — vanilla
// `1/theta^(2i/rotary_dim)` or a YaRN-rescaled table; the kernel is
// agnostic to which (the table is the runtime-tunable seam — see the
// `factor` discussion on Op::RopeNTokens). Mirrors the CPU reference
// `apply_rotary_emb` (attn/rope.rs). In-place.
//
// `precise::cos`/`sin`: RoPE angles reach the thousands, where
// fast-math cos/sin drift ~3e-4 (see yarn_rope_apply's note above).
//
// Dispatch:
//   one thread per (token, head, i), i in [0, rotary_dim/2)
//   threadgroups = (ceil(n_tokens * num_heads * (rotary_dim/2) / 256), 1, 1)
//   threads      = (256, 1, 1)

kernel void rope_n_tokens(
    device float*        x          [[buffer(0)]],   // [n_tokens, num_heads, head_dim], in-place
    device const float*  inv_freq   [[buffer(1)]],   // [rotary_dim/2]
    constant uint&       n_tokens   [[buffer(2)]],
    constant uint&       num_heads  [[buffer(3)]],
    constant uint&       head_dim   [[buffer(4)]],
    constant uint&       rotary_dim [[buffer(5)]],
    constant int&        start_pos  [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_rd = rotary_dim / 2;
    uint total   = n_tokens * num_heads * half_rd;
    if (tid >= total) return;

    uint i     = tid % half_rd;
    uint head  = (tid / half_rd) % num_heads;
    uint token = tid / (half_rd * num_heads);

    float angle = float(start_pos + int(token)) * inv_freq[i];
    float cos_a = metal::precise::cos(angle);
    float sin_a = metal::precise::sin(angle);

    uint  base = token * num_heads * head_dim + head * head_dim + i;
    float x0   = x[base];
    float x1   = x[base + half_rd];
    x[base]           = x0 * cos_a - x1 * sin_a;
    x[base + half_rd] = x0 * sin_a + x1 * cos_a;
}


// ============================================================================
// Kernel: MoE softmax + selection-sort top-K
// ============================================================================
// Per-token GPU port of `moe_router_cpu`'s softmax → top-K. One threadgroup
// per token: lanes cooperate on the softmax reductions; lane 0 then runs the
// running-min selection-sort to pick K experts.
//
// Slot order matches the CPU oracle: each new winner overwrites the running-
// minimum slot, so the output is NOT sorted by score. This makes the diff
// test bit-exact per slot (no set-sort needed).
//
// Caps:
//   - n_experts ≤ MAX_EXPERTS (512 — covers Qwen3-A3B's 128 and Cogito-V2 /
//     DeepSeek-V3's 256 with headroom).
//   - k ≤ MAX_K (16 — current models use 8).
//
// Dispatch:
//   threadgroups = (n_tokens, 1, 1)
//   threads      = (tg_size, 1, 1)   // typically 64-128, must be ≥ 32
//
// Caller has full control over tg_size: at minimum 32 for one simd group;
// 64-128 is a sensible sweet spot for the parallel softmax pass.

kernel void moe_softmax_topk(
    device const float* logits       [[buffer(0)]],   // [n_tokens, n_experts]
    device int*         out_indices  [[buffer(1)]],   // [n_tokens, k]
    device float*       out_weights  [[buffer(2)]],   // [n_tokens, k]
    constant uint&      n_experts    [[buffer(3)]],
    constant uint&      k            [[buffer(4)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint lid      [[thread_position_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]]
) {
    constexpr uint MAX_EXPERTS = 512;
    constexpr uint MAX_K       = 16;
    threadgroup float probs[MAX_EXPERTS];

    device const float* l = logits + tgid * n_experts;

    uint simd_lane       = lid % 32;
    uint simd_group      = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    // === Pass 1: max ===
    threadgroup float shared_max[32];
    float local_max = -1e30f;
    for (uint i = lid; i < n_experts; i += tg_size) {
        local_max = max(local_max, l[i]);
    }
    float sm = simd_max(local_max);
    if (simd_lane == 0) shared_max[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_max = simd_max(shared_max[simd_lane]);
    }
    threadgroup float bc_max;
    if (lid == 0) bc_max = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = bc_max;

    // === Pass 2: exp + partial sum (write probs into tg-mem) ===
    threadgroup float shared_sum[32];
    float local_sum = 0.0f;
    for (uint i = lid; i < n_experts; i += tg_size) {
        float v = exp(l[i] - global_max);
        probs[i] = v;
        local_sum += v;
    }
    float ss = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_group] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_sum = simd_sum(shared_sum[simd_lane]);
    }
    threadgroup float bc_sum;
    if (lid == 0) bc_sum = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = bc_sum;

    // === Pass 3: in-place softmax normalize in tg-mem ===
    float inv = 1.0f / global_sum;
    for (uint i = lid; i < n_experts; i += tg_size) {
        probs[i] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Pass 4: serial selection-sort top-K (lane 0 only) ===
    // Mirrors `cpu_topk`: each candidate either replaces the current
    // running-minimum slot or is discarded. Output slot order is the
    // running-min replacement order — not sorted by score, but bit-exact
    // against the CPU oracle.
    if (lid == 0) {
        int   sel_idx[MAX_K];
        float sel_val[MAX_K];
        for (uint s = 0; s < k; ++s) {
            sel_idx[s] = 0;
            sel_val[s] = -1e30f;
        }
        for (uint i = 0; i < n_experts; ++i) {
            float v = probs[i];
            uint min_k = 0;
            for (uint s = 1; s < k; ++s) {
                if (sel_val[s] < sel_val[min_k]) min_k = s;
            }
            if (v > sel_val[min_k]) {
                sel_val[min_k] = v;
                sel_idx[min_k] = int(i);
            }
        }
        device int*   ix = out_indices + tgid * k;
        device float* wt = out_weights + tgid * k;
        for (uint s = 0; s < k; ++s) {
            ix[s] = sel_idx[s];
            wt[s] = sel_val[s];
        }
    }
}


// ============================================================================
// Kernel: MoE per-token weight normalization
// ============================================================================
// Per-token: weights[t, 0..k] /= sum(weights[t, 0..k]) if sum > 0; else
// untouched. Matches `cpu_normalize_weights`'s guarded divide exactly.
//
// Dispatch:
//   threadgroups = (n_tokens, 1, 1)
//   threads      = (k, 1, 1)         // k is small (≤ 16); one thread/slot

kernel void moe_normalize_weights(
    device float*  weights  [[buffer(0)]],   // [n_tokens, k] in-place
    constant uint& k        [[buffer(1)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]]
) {
    device float* w = weights + tgid * k;

    threadgroup float bc_inv;
    if (lid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < k; ++i) s += w[i];
        // Same guard as `cpu_normalize_weights`: skip the divide if sum ≤ 0
        // (defensive; softmax outputs always satisfy sum > 0 in practice).
        bc_inv = (s > 0.0f) ? (1.0f / s) : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = bc_inv;

    if (lid < k) {
        w[lid] = w[lid] * inv;
    }
}


// ============================================================================
// Kernel: SplitQGate — deinterleave q_proj into q + gate stacks
// ============================================================================
// q_proj is laid out per head as `[q_h (head_dim) | gate_h (head_dim)]` —
// stride `2*head_dim` per head, `num_heads` heads per token. Deinterleave it
// into two contiguous `[n_tokens, num_heads, head_dim]` stacks. One thread per
// (token, head, channel). Mirrors the CPU loop at `full_attn_forward.rs`
// (the per-token q/gate split).
//
// Dispatch: flat over `n_tokens * num_heads * head_dim`.

kernel void split_q_gate(
    device const float* q_proj    [[buffer(0)]],  // [n_tokens, num_heads, 2*head_dim]
    device float*       q_out     [[buffer(1)]],  // [n_tokens, num_heads, head_dim]
    device float*       gate_out  [[buffer(2)]],  // [n_tokens, num_heads, head_dim]
    constant uint&      num_heads [[buffer(3)]],
    constant uint&      head_dim  [[buffer(4)]],
    constant uint&      n_tokens  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = n_tokens * num_heads * head_dim;
    if (tid >= total) return;

    uint c     = tid % head_dim;
    uint head  = (tid / head_dim) % num_heads;
    uint token = tid / (head_dim * num_heads);

    uint src_base = token * num_heads * 2 * head_dim + head * 2 * head_dim;
    uint dst      = token * num_heads * head_dim + head * head_dim + c;
    q_out[dst]    = q_proj[src_base + c];
    gate_out[dst] = q_proj[src_base + head_dim + c];
}


// ============================================================================
// Kernel: Weighted per-head RMS norm, batched over N tokens
// ============================================================================
// In-place per-head RMS normalize with a learned bf16 weight:
//   xh = xh * rsqrt(sum_sq/head_dim + eps) * bf16_to_f32(w[i])
// `x` is `[n_tokens, num_heads, head_dim]`; each (token, head) slice is
// normalized independently. The `head_dim`-long weight is shared across all
// heads and tokens. Diff oracle: `attn::rms_norm::rms_norm_per_head_cpu`.
//
// Unlike `rms_norm_qk` (weight-free, q/k regions in one buffer, the linear-
// attn shape), full-attn's q_norm/k_norm are learned tensors and q/k live in
// separate buffers — hence this distinct kernel, invoked once per buffer.
//
// Dispatch:
//   threadgroups = (num_heads, n_tokens, 1)
//   threads      = (tg_size, 1, 1)        // 256 is the sweet spot

kernel void rms_norm_per_head_n_tokens(
    device float*          x         [[buffer(0)]],   // [n_tokens, num_heads, head_dim] in-place
    device const uint16_t* weight    [[buffer(1)]],   // [head_dim] bf16
    constant uint&         num_heads [[buffer(2)]],
    constant uint&         head_dim  [[buffer(3)]],
    constant float&        eps       [[buffer(4)]],
    uint3 tgid3   [[threadgroup_position_in_grid]],   // (head, token)
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint3 tgsz3   [[threads_per_threadgroup]]
) {
    // Metal requires every position-attributed parameter to share one
    // vector width; the dispatch is 2D over threadgroups, 1D within.
    uint head    = tgid3.x;
    uint token   = tgid3.y;
    uint lid     = lid3.x;
    uint tg_size = tgsz3.x;
    device float* xh = x + (token * num_heads + head) * head_dim;

    uint simd_lane       = lid % 32;
    uint simd_group      = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    // Phase 1: parallel sum-of-squares reduction.
    threadgroup float shared[32];
    float acc = 0.0f;
    for (uint i = lid; i < head_dim; i += tg_size) {
        float v = xh[i];
        acc += v * v;
    }
    float sm = simd_sum(acc);
    if (simd_lane == 0) shared[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        total = simd_sum(shared[simd_lane]);
    }
    threadgroup float bc_inv_rms;
    if (lid == 0) {
        bc_inv_rms = rsqrt(total / float(head_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = bc_inv_rms;

    // Phase 2: apply the learned weight.
    for (uint i = lid; i < head_dim; i += tg_size) {
        float w = bf16_to_f32(weight[i]);
        xh[i] = xh[i] * inv_rms * w;
    }
}


// ============================================================================
// Kernel: KvCacheAppendNTokens — append k/v scratch into the resident cache
// ============================================================================
// Strided copy of the per-token k/v scratch stacks `[n_tokens, kv_dim]` into
// the GPU-resident KV cache `[MAX_SEQ_LEN, kv_dim]` at row `kv_start`. Both
// the source and the destination window are contiguous; one thread per
// (token, channel) copies k and v together. Diff oracle: two windowed
// `copy_from_slice`s.

kernel void kv_cache_append_n_tokens(
    device const float* k_src    [[buffer(0)]],  // [n_tokens, kv_dim]
    device const float* v_src    [[buffer(1)]],  // [n_tokens, kv_dim]
    device float*       k_cache  [[buffer(2)]],  // [MAX_SEQ_LEN, kv_dim]
    device float*       v_cache  [[buffer(3)]],  // [MAX_SEQ_LEN, kv_dim]
    constant uint&      kv_dim   [[buffer(4)]],
    constant uint&      n_tokens [[buffer(5)]],
    constant uint&      kv_start [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = n_tokens * kv_dim;
    if (tid >= total) return;

    uint t   = tid / kv_dim;
    uint i   = tid % kv_dim;
    uint dst = (kv_start + t) * kv_dim + i;
    k_cache[dst] = k_src[tid];
    v_cache[dst] = v_src[tid];
}
