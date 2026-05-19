// moeflux's own kernel — not vendored from MLX. Concatenated into the
// crate's single translation unit after the steel headers (see
// `SHADER_PARTS`), so a later revision can build QK^T / P·V on
// `mlx::steel::MMATile`.

// ============================================================================
// FlashAttention-2 causal SDPA — batched prefill.
// ============================================================================
//
// One threadgroup owns a tile of FA_BR query tokens for one head, and runs
// the whole online-softmax loop over all KV blocks internally — no init /
// finalize dispatches, no global running-state scratch.
//
// Residency (the point of the rewrite — both Q and K/V read from global
// exactly once):
//   - The Q tile is register-resident for the threadgroup's lifetime.
//     Simdgroup g owns rows [g*FA_RPS, (g+1)*FA_RPS); each lane holds 8 of
//     a row's 256 head_dim values (lane, lane+32, ..., lane+224).
//   - Each KV block is staged once into threadgroup `kv_stage` and reused
//     across the whole query tile (K, then time-multiplexed for V).
//   - The output accumulator is register-resident, same lane partition.
//
// head_dim is fixed to FA_HD=256 (a3b); the threadgroup arrays are sized
// at compile time. The encoder asserts head_dim == 256.
//
// Diff target: per-token cosine >= 0.9999 vs sdpa_cpu. Follow-up levers
// (own arcs): simdgroup_matrix QK^T/AV; GQA-fold (one TG per (q_tile,kv_h),
// K/V block reused across the 8 shared query-heads).

constant uint FA_BR      = 64;   // query tile
constant uint FA_BC      = 16;   // KV block
constant uint FA_HD      = 256;  // head_dim (a3b)
constant uint FA_THREADS = 256;  // 8 simdgroups
constant uint FA_SIMDS   = FA_THREADS / 32;   // 8
constant uint FA_RPS     = FA_BR / FA_SIMDS;  // rows per simdgroup
constant uint FA_VPL     = FA_HD / 32;        // values per lane = 8
constant uint FA_TPR     = FA_THREADS / FA_BR; // Phase-3 threads per row

// --- Ablation function constants (session 13 bound analysis) ----------------
// All default false via `is_function_constant_defined`, so a PSO built with
// no `FunctionConstantValues` (production `Kernels::new`) is bit-identical to
// the un-ablated kernel — every gated phase compiles away or stays whole at
// PSO-compile time, no runtime branch. Set only by `kernel_bench`'s ablation
// PSOs to isolate each phase's cost. Barriers are never gated: kernel
// structure / scheduling is held constant so barrier cost falls out as a
// residual.
constant bool ABLATE_SKIP_QK_      [[function_constant(100)]];
constant bool ABLATE_SKIP_SOFTMAX_ [[function_constant(101)]];
constant bool ABLATE_SKIP_PV_      [[function_constant(102)]];
constant bool ABLATE_SKIP_STAGE_   [[function_constant(103)]];
constant bool ABLATE_SKIP_QK =
    is_function_constant_defined(ABLATE_SKIP_QK_)      ? ABLATE_SKIP_QK_      : false;
constant bool ABLATE_SKIP_SOFTMAX =
    is_function_constant_defined(ABLATE_SKIP_SOFTMAX_) ? ABLATE_SKIP_SOFTMAX_ : false;
constant bool ABLATE_SKIP_PV =
    is_function_constant_defined(ABLATE_SKIP_PV_)      ? ABLATE_SKIP_PV_      : false;
constant bool ABLATE_SKIP_STAGE =
    is_function_constant_defined(ABLATE_SKIP_STAGE_)   ? ABLATE_SKIP_STAGE_   : false;

// Stage one KV block (FA_BC rows x FA_HD) from `cache` into the threadgroup
// staging buffer `kv_stage4` with float4 loads — 4 contiguous head_dim
// values per load instead of 16 strided scalar loads. Phase-0 measured
// this +0.6–10% over scalar (shape-dependent); it is now the only path.
// Rows >= bc_valid are zeroed. Caller barriers afterwards.
static inline void stage_kv_block(
    threadgroup float4* kv_stage4,
    device const float* cache,
    uint c0, uint kv_dim, uint kv_h, uint bc_valid, uint lid
) {
    // 4096 elems / (256 threads * 4) = 4 float4s per thread. The 4
    // contiguous elems share `c` and never cross a head_dim boundary
    // (lid*4 is a multiple of 4, head_dim 256 is too), so one bc guard
    // and one aligned float4 load/store cover the group.
    for (uint g = 0; g < FA_BC * FA_HD; g += FA_THREADS * 4) {
        uint e = g + lid * 4;
        uint c = e / FA_HD, d = e % FA_HD;
        float4 val = float4(0.0f);
        if (c < bc_valid) {
            val = *(device const float4*)
                (cache + (size_t)(c0 + c) * kv_dim + kv_h * FA_HD + d);
        }
        kv_stage4[e / 4] = val;
    }
}

kernel void attn_sdpa_causal_flash(
    device const float* Q             [[buffer(0)]],  // [M, num_heads, head_dim]
    device const float* K_cache       [[buffer(1)]],  // [kv_len, kv_dim]
    device const float* V_cache       [[buffer(2)]],  // [kv_len, kv_dim]
    device float*       out           [[buffer(3)]],  // [M, num_heads, head_dim]
    constant uint&      n_tokens      [[buffer(4)]],
    constant uint&      num_heads     [[buffer(5)]],
    constant uint&      heads_per_kv  [[buffer(6)]],
    constant uint&      kv_dim        [[buffer(7)]],
    constant uint&      start_pos     [[buffer(8)]],
    constant uint&      kv_len        [[buffer(9)]],
    constant float&     softmax_scale [[buffer(10)]],
    uint tg_idx [[threadgroup_position_in_grid]],   // q_tile * num_heads + h
    uint lid    [[thread_position_in_threadgroup]]
) {
    uint q_tile = tg_idx / num_heads;
    uint h      = tg_idx % num_heads;
    uint q0     = q_tile * FA_BR;
    if (q0 >= n_tokens) return;
    uint br_valid = min(FA_BR, n_tokens - q0);
    uint kv_h     = h / heads_per_kv;

    uint simd_id = lid / 32;   // 0..7
    uint lane    = lid % 32;   // 0..31

    // float4-backed so the VEC4_STAGE ablation path has a 16-byte-aligned
    // staging buffer; `kv_stage` aliases it for the scalar QK / P·V phases.
    threadgroup float4 kv_stage4[FA_BC * FA_HD / 4];  // 16 KB — K, then V
    threadgroup float* kv_stage = (threadgroup float*)kv_stage4;
    threadgroup float scores  [FA_BR * FA_BC];  // 2 KB
    threadgroup float row_m   [FA_BR];          // running max  per row
    threadgroup float row_l   [FA_BR];          // running denom per row
    threadgroup float row_corr[FA_BR];          // exp(m_old - m_new) this block

    // Q tile + O accumulator, register-resident. qreg[ri][k] / acc[ri][k]
    // is row (q0 + simd_id*FA_RPS + ri), head_dim index (lane + k*32).
    float qreg[FA_RPS][FA_VPL];
    float acc [FA_RPS][FA_VPL];
    for (uint ri = 0; ri < FA_RPS; ++ri) {
        uint trow = simd_id * FA_RPS + ri;          // tile-local row
        uint grow = q0 + trow;                       // global query token
        device const float* qsrc =
            Q + ((size_t)grow * num_heads + h) * FA_HD;
        for (uint k = 0; k < FA_VPL; ++k) {
            qreg[ri][k] = (trow < br_valid) ? qsrc[lane + k * 32] : 0.0f;
            acc[ri][k]  = 0.0f;
        }
    }

    for (uint r = lid; r < FA_BR; r += FA_THREADS) {
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal extent: query (q0 + r) attends to KV [0, start_pos+q0+r+1).
    uint kv_max_last  = start_pos + q0 + br_valid;  // exclusive, last row
    uint kv_max_first = start_pos + q0 + 1;         // exclusive, row 0

    for (uint c0 = 0; c0 < kv_max_last; c0 += FA_BC) {
        uint bc_valid   = min(FA_BC, kv_len - c0);
        bool needs_mask = (c0 + FA_BC) > kv_max_first;

        // -- Phase 1: stage K block --
        if (!ABLATE_SKIP_STAGE) {
            stage_kv_block(kv_stage4, K_cache, c0, kv_dim, kv_h,
                           bc_valid, lid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 2: QK^T — simdgroup-cooperative dot, simd_sum reduce --
        if (!ABLATE_SKIP_QK) {
            for (uint ri = 0; ri < FA_RPS; ++ri) {
                uint trow = simd_id * FA_RPS + ri;
                for (uint c = 0; c < FA_BC; ++c) {
                    float partial = 0.0f;
                    for (uint k = 0; k < FA_VPL; ++k) {
                        partial = fma(qreg[ri][k],
                                      kv_stage[c * FA_HD + lane + k * 32],
                                      partial);
                    }
                    float dot = simd_sum(partial);
                    if (lane == 0) {
                        uint kv_pos = c0 + c;
                        bool masked = (c >= bc_valid) ||
                            (needs_mask &&
                             kv_pos >= start_pos + q0 + trow + 1);
                        scores[trow * FA_BC + c] =
                            masked ? -INFINITY : dot * softmax_scale;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 3: per-row online-softmax update. FA_TPR threads
        //    per row; FA_TPR divides 32, so a row's lanes are
        //    contiguous within one simdgroup. --
        if (!ABLATE_SKIP_SOFTMAX) {
            uint row = lid / FA_TPR;     // 0..FA_BR-1
            uint sub = lid % FA_TPR;
            float lm = -INFINITY;
            for (uint c = sub; c < FA_BC; c += FA_TPR) {
                lm = max(lm, scores[row * FA_BC + c]);
            }
            for (uint s = 1; s < FA_TPR; s <<= 1) {
                lm = max(lm, simd_shuffle_xor(lm, s));
            }
            float blk_max = lm;

            float m_old = row_m[row];
            float m_new = max(m_old, blk_max);
            bool  active = (m_new != -INFINITY);
            float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);

            float ls = 0.0f;
            for (uint c = sub; c < FA_BC; c += FA_TPR) {
                float p = active
                    ? exp(scores[row * FA_BC + c] - m_new)
                    : 0.0f;
                scores[row * FA_BC + c] = p;
                ls += p;
            }
            for (uint s = 1; s < FA_TPR; s <<= 1) {
                ls += simd_shuffle_xor(ls, s);
            }

            if (sub == 0) {
                row_m[row]    = m_new;
                row_corr[row] = corr;
                row_l[row]    = row_l[row] * corr + ls;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 4: stage V block (K now dead, reuse kv_stage) --
        if (!ABLATE_SKIP_STAGE) {
            stage_kv_block(kv_stage4, V_cache, c0, kv_dim, kv_h,
                           bc_valid, lid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 5: rescale O, accumulate P·V --
        if (!ABLATE_SKIP_PV) {
            for (uint ri = 0; ri < FA_RPS; ++ri) {
                uint trow = simd_id * FA_RPS + ri;
                float rc = row_corr[trow];
                for (uint k = 0; k < FA_VPL; ++k) {
                    uint d = lane + k * 32;
                    float a = acc[ri][k] * rc;
                    for (uint c = 0; c < FA_BC; ++c) {
                        a = fma(scores[trow * FA_BC + c],
                                kv_stage[c * FA_HD + d], a);
                    }
                    acc[ri][k] = a;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Finalize: out = O / row_l, guarded by the partial-tile tail. --
    for (uint ri = 0; ri < FA_RPS; ++ri) {
        uint trow = simd_id * FA_RPS + ri;
        if (trow >= br_valid) continue;
        float denom = row_l[trow];
        float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        device float* osrc =
            out + ((size_t)(q0 + trow) * num_heads + h) * FA_HD;
        for (uint k = 0; k < FA_VPL; ++k) {
            osrc[lane + k * 32] = acc[ri][k] * inv;
        }
    }
}

// ============================================================================
// FlashAttention-2 causal SDPA — GQA-folded (session 13).
// ============================================================================
//
// Phase 0's ablation found SDPA is ~97% K/V staging and latency-bound. Under
// GQA, the `heads_per_kv` query-heads sharing a KV head each re-stage the
// *same* K/V blocks — `heads_per_kv`× redundant staging. This kernel folds
// `FA_GROUP` of those query-heads into one threadgroup: each KV block is
// staged once and reused across all `FA_GROUP` heads, cutting the dominant
// staging-transaction count `FA_GROUP`-fold.
//
// One threadgroup owns a tile of FA_BR query tokens for `FA_GROUP` query-
// heads of one KV head; grid is `num_q_tiles × (num_heads / FA_GROUP)`.
// `qreg`/`acc` are register-resident *per head*, so register footprint is
// `128·FA_GROUP` floats/thread — this is the fold's hard ceiling (`FA_GROUP`
// is a compile-time template arg so the arrays size correctly). K and V are
// co-staged into one 16 KB buffer at FA_GQA_BC=8, so a block is one barrier.
//
// Diff target: per-token cosine >= 0.9999 vs sdpa_cpu, same as the unfolded
// kernel. `FA_GROUP` must divide `heads_per_kv`.

constant uint FA_GQA_BC = 8;   // KV block — K and V co-staged (16 KB)

// Stage one FA_GQA_BC-row KV block from `cache` into `dst4` with float4
// loads (the +5% Phase-0 measured). Rows >= bc_valid are zeroed.
static inline void stage_gqa_block(
    threadgroup float4* dst4,
    device const float* cache,
    uint c0, uint kv_dim, uint kv_h, uint bc_valid, uint lid
) {
    for (uint g = 0; g < FA_GQA_BC * FA_HD; g += FA_THREADS * 4) {
        uint e = g + lid * 4;
        uint c = e / FA_HD, d = e % FA_HD;
        float4 val = float4(0.0f);
        if (c < bc_valid) {
            val = *(device const float4*)
                (cache + (size_t)(c0 + c) * kv_dim + kv_h * FA_HD + d);
        }
        dst4[e / 4] = val;
    }
}

template<uint FA_GROUP>
static void sdpa_gqa_impl(
    device const float* Q,
    device const float* K_cache,
    device const float* V_cache,
    device float*       out,
    uint n_tokens, uint num_heads, uint heads_per_kv,
    uint kv_dim, uint start_pos, uint kv_len, float softmax_scale,
    // Threadgroup storage — declared in the kernel entry point (MSL
    // forbids threadgroup-space variables in a non-kernel function) and
    // passed in. `row_*` are [FA_GROUP * FA_BR], row-major over heads.
    threadgroup float4* kv_stage4,
    threadgroup float*  scores,
    threadgroup float*  row_m,
    threadgroup float*  row_l,
    threadgroup float*  row_corr,
    uint tg_idx, uint lid
) {
    uint num_groups = num_heads / FA_GROUP;   // head-groups per q-tile
    uint q_tile = tg_idx / num_groups;
    uint hg     = tg_idx % num_groups;
    uint h_base = hg * FA_GROUP;              // first query-head of the group
    uint q0     = q_tile * FA_BR;
    if (q0 >= n_tokens) return;
    uint br_valid = min(FA_BR, n_tokens - q0);
    uint kv_h     = h_base / heads_per_kv;    // shared by all FA_GROUP heads

    uint simd_id = lid / 32;
    uint lane    = lid % 32;

    // K | V co-staged: kv_K4 = [0, FA_GQA_BC*FA_HD), kv_V4 = the rest.
    threadgroup float*  kv_K  = (threadgroup float*)kv_stage4;
    threadgroup float*  kv_V  = kv_K + FA_GQA_BC * FA_HD;
    threadgroup float4* kv_K4 = kv_stage4;
    threadgroup float4* kv_V4 = kv_stage4 + FA_GQA_BC * FA_HD / 4;

    // Per-head Q tile + O accumulator, register-resident (128·FA_GROUP
    // floats/thread). Same lane partition as the unfolded kernel.
    float qreg[FA_GROUP][FA_RPS][FA_VPL];
    float acc [FA_GROUP][FA_RPS][FA_VPL];
    for (uint g = 0; g < FA_GROUP; ++g) {
        uint h = h_base + g;
        for (uint ri = 0; ri < FA_RPS; ++ri) {
            uint trow = simd_id * FA_RPS + ri;
            uint grow = q0 + trow;
            device const float* qsrc =
                Q + ((size_t)grow * num_heads + h) * FA_HD;
            for (uint k = 0; k < FA_VPL; ++k) {
                qreg[g][ri][k] =
                    (trow < br_valid) ? qsrc[lane + k * 32] : 0.0f;
                acc[g][ri][k] = 0.0f;
            }
        }
    }
    for (uint r = lid; r < FA_GROUP * FA_BR; r += FA_THREADS) {
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_max_last  = start_pos + q0 + br_valid;  // exclusive, last row
    uint kv_max_first = start_pos + q0 + 1;         // exclusive, row 0

    for (uint c0 = 0; c0 < kv_max_last; c0 += FA_GQA_BC) {
        uint bc_valid   = min(FA_GQA_BC, kv_len - c0);
        bool needs_mask = (c0 + FA_GQA_BC) > kv_max_first;

        // -- stage K and V once (shared by all FA_GROUP heads) --
        stage_gqa_block(kv_K4, K_cache, c0, kv_dim, kv_h, bc_valid, lid);
        stage_gqa_block(kv_V4, V_cache, c0, kv_dim, kv_h, bc_valid, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint g = 0; g < FA_GROUP; ++g) {
            // -- QK^T for head g --
            for (uint ri = 0; ri < FA_RPS; ++ri) {
                uint trow = simd_id * FA_RPS + ri;
                for (uint c = 0; c < FA_GQA_BC; ++c) {
                    float partial = 0.0f;
                    for (uint k = 0; k < FA_VPL; ++k) {
                        partial = fma(qreg[g][ri][k],
                                      kv_K[c * FA_HD + lane + k * 32],
                                      partial);
                    }
                    float dot = simd_sum(partial);
                    if (lane == 0) {
                        uint kv_pos = c0 + c;
                        bool masked = (c >= bc_valid) ||
                            (needs_mask &&
                             kv_pos >= start_pos + q0 + trow + 1);
                        scores[trow * FA_GQA_BC + c] =
                            masked ? -INFINITY : dot * softmax_scale;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // -- online-softmax update for head g --
            {
                uint row = lid / FA_TPR;
                uint sub = lid % FA_TPR;
                float lm = -INFINITY;
                for (uint c = sub; c < FA_GQA_BC; c += FA_TPR) {
                    lm = max(lm, scores[row * FA_GQA_BC + c]);
                }
                for (uint s = 1; s < FA_TPR; s <<= 1) {
                    lm = max(lm, simd_shuffle_xor(lm, s));
                }
                float blk_max = lm;

                float m_old = row_m[g * FA_BR + row];
                float m_new = max(m_old, blk_max);
                bool  active = (m_new != -INFINITY);
                float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);

                float ls = 0.0f;
                for (uint c = sub; c < FA_GQA_BC; c += FA_TPR) {
                    float p = active
                        ? exp(scores[row * FA_GQA_BC + c] - m_new)
                        : 0.0f;
                    scores[row * FA_GQA_BC + c] = p;
                    ls += p;
                }
                for (uint s = 1; s < FA_TPR; s <<= 1) {
                    ls += simd_shuffle_xor(ls, s);
                }

                if (sub == 0) {
                    uint idx = g * FA_BR + row;
                    row_m[idx]    = m_new;
                    row_corr[idx] = corr;
                    row_l[idx]    = row_l[idx] * corr + ls;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // -- rescale O, accumulate P·V for head g --
            for (uint ri = 0; ri < FA_RPS; ++ri) {
                uint trow = simd_id * FA_RPS + ri;
                float rc = row_corr[g * FA_BR + trow];
                for (uint k = 0; k < FA_VPL; ++k) {
                    uint d = lane + k * 32;
                    float a = acc[g][ri][k] * rc;
                    for (uint c = 0; c < FA_GQA_BC; ++c) {
                        a = fma(scores[trow * FA_GQA_BC + c],
                                kv_V[c * FA_HD + d], a);
                    }
                    acc[g][ri][k] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // -- Finalize: out = O / row_l, per head, guarded by the partial tail. --
    for (uint g = 0; g < FA_GROUP; ++g) {
        uint h = h_base + g;
        for (uint ri = 0; ri < FA_RPS; ++ri) {
            uint trow = simd_id * FA_RPS + ri;
            if (trow >= br_valid) continue;
            float denom = row_l[g * FA_BR + trow];
            float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
            device float* osrc =
                out + ((size_t)(q0 + trow) * num_heads + h) * FA_HD;
            for (uint k = 0; k < FA_VPL; ++k) {
                osrc[lane + k * 32] = acc[g][ri][k] * inv;
            }
        }
    }
}

#define SDPA_GQA_ENTRY(NAME, G)                                            \
kernel void NAME(                                                          \
    device const float* Q             [[buffer(0)]],                       \
    device const float* K_cache       [[buffer(1)]],                       \
    device const float* V_cache       [[buffer(2)]],                       \
    device float*       out           [[buffer(3)]],                       \
    constant uint&      n_tokens      [[buffer(4)]],                       \
    constant uint&      num_heads     [[buffer(5)]],                       \
    constant uint&      heads_per_kv  [[buffer(6)]],                       \
    constant uint&      kv_dim        [[buffer(7)]],                       \
    constant uint&      start_pos     [[buffer(8)]],                       \
    constant uint&      kv_len        [[buffer(9)]],                       \
    constant float&     softmax_scale [[buffer(10)]],                      \
    uint tg_idx [[threadgroup_position_in_grid]],                          \
    uint lid    [[thread_position_in_threadgroup]]                         \
) {                                                                        \
    threadgroup float4 kv_stage4[FA_GQA_BC * FA_HD * 2 / 4];   /* 16 KB */  \
    threadgroup float  scores[FA_BR * FA_GQA_BC];                           \
    threadgroup float  row_m[(G) * FA_BR];                                  \
    threadgroup float  row_l[(G) * FA_BR];                                  \
    threadgroup float  row_corr[(G) * FA_BR];                               \
    sdpa_gqa_impl<G>(Q, K_cache, V_cache, out, n_tokens, num_heads,         \
        heads_per_kv, kv_dim, start_pos, kv_len, softmax_scale,             \
        kv_stage4, scores, row_m, row_l, row_corr, tg_idx, lid);            \
}

SDPA_GQA_ENTRY(attn_sdpa_causal_flash_gqa2, 2)
SDPA_GQA_ENTRY(attn_sdpa_causal_flash_gqa4, 4)
SDPA_GQA_ENTRY(attn_sdpa_causal_flash_gqa8, 8)
