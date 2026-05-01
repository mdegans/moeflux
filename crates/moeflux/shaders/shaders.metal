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
// Kernel 10: GatedDeltaNet linear attention step (single token, all heads)
// ============================================================================
//
// Implements the GatedDeltaNet recurrence for autoregressive generation:
//   1. State decay:  S[vi][ki] *= g_decay
//   2. Memory read:  kv_mem[vi] = sum_ki(S[vi][ki] * k[ki])
//   3. Delta:        delta[vi] = (v[vi] - kv_mem[vi]) * beta_gate
//   4. State update: S[vi][ki] += k[ki] * delta[vi]
//   5. Output:       out[vi] = sum_ki(S[vi][ki] * q[ki])
//
// Dispatch: 64 threadgroups (one per v-head), 128 threads each (one per vi).
// Each thread owns one row S[head_id][vi][:] of the 128x128 state matrix.
//
// State layout: [64 * 128 * 128] float = 4MB total, persisted across tokens.
// k-head sharing: 4 v-heads share 1 k-head (64 v-heads / 16 k-heads).

kernel void gated_delta_net_step(
    device float *state,             // [64 * 128 * 128] persistent state
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &k_heads_per_v,    // = 4
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Step 3+4: Delta update — S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < 128; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output — out[vi] = dot(S[vi][:], q[:])
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        out_val += state[state_base + ki] * q[k_base + ki];
    }
    output[v_base + vi] = out_val;
}


// ============================================================================
// Kernel 11: Conv1d depthwise step (single token, incremental inference)
// ============================================================================
//
// Depthwise 1D convolution for one new input token:
//   output[c] = sum_k(history[k][c] * weight[c][k]) + input[c] * weight[c][3]
//   then SiLU activation: output[c] = output[c] / (1 + exp(-output[c]))
//
// After computing, shifts the history buffer left and appends the new input.
//
// Weight layout: [channels * kernel_size] bf16, weight[c * kernel_size + k]
// Conv state layout: [(kernel_size-1) * channels] row-major, state[k * channels + c]
// kernel_size = 4 (hardcoded), so 3 history slots + 1 new input.
//
// Dispatch: conv_dim threads (12288), one per channel.

kernel void conv1d_step(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    device const float *input,        // [conv_dim] current input
    device const uint16_t *weights,   // [conv_dim * 4] bf16 as uint16
    device float *output,             // [conv_dim] convolution output
    constant uint &conv_dim,          // = 12288
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    // Convolution: dot product of history + new input with weights
    // weight layout: weight[c * 4 + k] for channel c, position k
    uint w_base = idx * 4;
    float acc = 0.0f;

    // 3 history slots (k=0,1,2)
    acc += conv_state[0 * conv_dim + idx] * bf16_to_f32(weights[w_base + 0]);
    acc += conv_state[1 * conv_dim + idx] * bf16_to_f32(weights[w_base + 1]);
    acc += conv_state[2 * conv_dim + idx] * bf16_to_f32(weights[w_base + 2]);

    // New input (k=3)
    float inp = input[idx];
    acc += inp * bf16_to_f32(weights[w_base + 3]);

    // SiLU activation
    output[idx] = acc / (1.0f + exp(-acc));

    // Shift history: move slots 1,2 -> 0,1, append input at slot 2
    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = inp;
}


// ============================================================================
// Kernel 12: Per-head RMS normalize for q and k vectors
// ============================================================================
// q: [num_k_heads * key_dim], k: [num_k_heads * key_dim]
// Normalize each head independently, then scale by 1/sqrt(key_dim)^2 for q, 1/sqrt(key_dim) for k
// Dispatch: num_k_heads threadgroups, key_dim threads each

kernel void rms_norm_qk(
    device float *q,              // [num_k_heads * key_dim] in/out
    device float *k,              // [num_k_heads * key_dim] in/out
    constant uint &key_dim,       // = 128
    constant float &inv_scale,    // = 1/sqrt(key_dim)
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * key_dim;

    // RMS norm for q
    threadgroup float q_sum_sq;
    if (tid == 0) q_sum_sq = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qval = (tid < key_dim) ? q[base + tid] : 0;
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
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;  // q gets extra scale
    }

    // RMS norm for k
    threadgroup float k_sum_sq;
    float kval = (tid < key_dim) ? k[base + tid] : 0;
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
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}


// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
// Per v-head: g_decay = exp(-A * softplus(alpha + dt_bias)), beta_gate = sigmoid(beta)
// Dispatch: num_v_heads threads (64)

kernel void compute_decay_beta(
    device const float *alpha_out,   // [num_v_heads] from projection
    device const float *beta_out,    // [num_v_heads] from projection
    device const float *A_log,       // [num_v_heads] log of decay base (persistent)
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay,           // [num_v_heads] output
    device float *beta_gate,         // [num_v_heads] output
    uint idx [[thread_position_in_grid]]
) {
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = exp(A_log[idx]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}


// ============================================================================
// Kernel 14: Gated RMS norm (z-gated output normalization)
// ============================================================================
// output[i] = rms_norm(values[i]) * SiLU(z[i]) * weight[i]
// Per v-head: normalize values, gate with z, scale with weight
// Dispatch: num_v_heads threadgroups, value_dim threads each

kernel void gated_rms_norm(
    device const float *values,       // [num_v_heads * value_dim] delta-net output
    device const float *z,            // [num_v_heads * value_dim] gate values
    device const uint16_t *weight,    // [value_dim] bf16 norm weights (shared across heads)
    device float *output,             // [num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant float &eps,              // = 1e-6
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * value_dim;

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
