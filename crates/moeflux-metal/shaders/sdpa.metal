// moeflux's own kernel — not vendored from MLX. Concatenated into the
// crate's single translation unit after the steel headers (see
// `SHADER_PARTS`), so a later revision can build QK^T / P·V on
// `mlx::steel::MMATile`.

// ============================================================================
// FlashAttention-2 causal SDPA — slot vB (staging-based, kept for A/B).
// ============================================================================
//
// Original staging-based design; replaced as the production kernel by the
// direct-device vA (below) after a 7× speedup on rebooted A/B. Kept in
// the vB experimental slot for diff-oracle and future comparison.
//
// One threadgroup owns a tile of FA_BR query tokens for one head, and runs
// the whole online-softmax loop over all KV blocks internally.
//
// Residency: Q tile + O accumulator are register-resident (128 floats/thread).
// Each KV block is staged once into threadgroup memory and reused.
//
// head_dim is fixed to FA_HD=256 (a3b); the threadgroup arrays are sized
// at compile time. The encoder asserts head_dim == 256.

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

kernel void attn_sdpa_causal_flash_vb(
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

    // Pre-fold log2(e) into the softmax scale so the per-block scoring
    // and online softmax can use the cheaper `fast::exp2` instead of
    // `exp`. exp(x) == exp2(x * log2(e)); folding it into the scale
    // means scores, m, and corr all live in log2 domain consistently.
    const float log2_scale = softmax_scale * M_LOG2E_F;

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
                            masked ? -INFINITY : dot * log2_scale;
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
            float corr = (m_old == -INFINITY) ? 0.0f
                                              : fast::exp2(m_old - m_new);

            float ls = 0.0f;
            for (uint c = sub; c < FA_BC; c += FA_TPR) {
                float p = active
                    ? fast::exp2(scores[row * FA_BC + c] - m_new)
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

    // Pre-fold log2(e) into the softmax scale — see the unfolded
    // kernel for the rationale. Same trick, same per-thread cost.
    const float log2_scale = softmax_scale * M_LOG2E_F;

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
                            masked ? -INFINITY : dot * log2_scale;
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
                float corr = (m_old == -INFINITY) ? 0.0f
                                                  : fast::exp2(m_old - m_new);

                float ls = 0.0f;
                for (uint c = sub; c < FA_GQA_BC; c += FA_TPR) {
                    float p = active
                        ? fast::exp2(scores[row * FA_GQA_BC + c] - m_new)
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

SDPA_GQA_ENTRY(attn_sdpa_causal_flash_gqa2_va, 2)
SDPA_GQA_ENTRY(attn_sdpa_causal_flash_gqa4_va, 4)
SDPA_GQA_ENTRY(attn_sdpa_causal_flash_gqa8_va, 8)

// ============================================================================
// FlashAttention-2 causal SDPA — slot vA (direct-device, production).
// ============================================================================
//
// llama.cpp-style direct-device design: reads K/V direct from device memory
// inside QK^T and PV phases, eliminating the threadgroup staging that was
// 96.5% of the old kernel's wall time. Apple's MPP guide recommends this
// pattern on Apple silicon when occupancy is high.
//
// Geometry (matches llama.cpp at Q=8, C=64, NSG=8):
//   VB_Q  = 8   queries per threadgroup
//   VB_C  = 64  KV cols per block
//   VB_NSG = 8  simdgroups per TG (= FA_SIMDS)
//   VB_NQ  = VB_Q / VB_NSG = 1  query row owned per simdgroup
//   VB_NO  = (FA_HD / 8) / VB_NSG = 4  O column-tiles per simdgroup
//
// Per-TG footprint: ~18 KB (Q 8 KB + O 8 KB + scores 2 KB).
// Per-lane registers: ~50 B (vs ~512 B for the staging kernel).
//
// Diff target: per-token cosine >= 0.9999 vs sdpa_cpu.
// GQA-fold (G>=2): NOT in this kernel yet. fold > 1 dispatches to
// the staging-based `_gqa2_va` / `_gqa4_va` / `_gqa8_va` entries.

constant uint VB_Q   = 8;                       // queries per TG
constant uint VB_C   = 64;                      // KV cols per block
constant uint VB_NSG = FA_SIMDS;                // simdgroups per TG (= 8)
constant uint VB_D8  = FA_HD / 8;               // head_dim tiles of 8 (= 32)
constant uint VB_C8  = VB_C / 8;                // KV chunks of 8 in a block (= 8)
constant uint VB_NO  = VB_D8 / VB_NSG;          // O column-tiles per simdgroup (= 4)

kernel void attn_sdpa_causal_flash_va(
    device const float* Q             [[buffer(0)]],
    device const float* K_cache       [[buffer(1)]],
    device const float* V_cache       [[buffer(2)]],
    device float*       out           [[buffer(3)]],
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
    uint q0     = q_tile * VB_Q;
    if (q0 >= n_tokens) return;
    uint br_valid = min(VB_Q, n_tokens - q0);
    uint kv_h     = h / heads_per_kv;

    uint simd_id = lid / 32;   // 0..VB_NSG-1 (= 0..7)
    uint lane    = lid % 32;   // 0..31

    // Pre-fold log2(e) so the per-block scoring and online softmax can
    // use `fast::exp2` consistently — same trick as vA.
    const float log2_scale = softmax_scale * M_LOG2E_F;

    // Threadgroup memory (~18 KB total).
    threadgroup float sq[VB_Q * FA_HD];     //  8 KB  Q tile
    threadgroup float so[VB_Q * FA_HD];     //  8 KB  O accumulator (unnormalized)
    threadgroup float ss[VB_Q * VB_C];      //  2 KB  scores → P matrix

    // Load Q tile into sq. With VB_Q == VB_NSG, each simdgroup owns one
    // query row of FA_HD floats. Lane i covers head_dim positions
    // [i, i+32, ..., i+224] — 8 stores/lane.
    {
        uint trow = simd_id;
        if (trow < br_valid) {
            device const float* qsrc =
                Q + ((size_t)(q0 + trow) * num_heads + h) * FA_HD;
            for (uint k = 0; k < FA_HD / 32; ++k) {
                sq[trow * FA_HD + lane + k * 32] = qsrc[lane + k * 32];
            }
        } else {
            for (uint k = 0; k < FA_HD / 32; ++k) {
                sq[trow * FA_HD + lane + k * 32] = 0.0f;
            }
        }
    }

    // Zero so. 256 threads each clear VB_Q*FA_HD / FA_THREADS = 8 elements.
    for (uint i = lid; i < VB_Q * FA_HD; i += FA_THREADS) {
        so[i] = 0.0f;
    }

    // Per-thread running max/denom. Each simdgroup owns one query row
    // (trow == simd_id); within a simdgroup the 32 lanes hold the same
    // M and S value (kept consistent by simd_max / simd_sum reductions).
    float M = -INFINITY;
    float S = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal extent — same math as vA.
    uint kv_max_last  = start_pos + q0 + br_valid;  // exclusive, last row
    uint kv_max_first = start_pos + q0 + 1;         // exclusive, row 0

    for (uint c0 = 0; c0 < kv_max_last; c0 += VB_C) {
        uint bc_valid   = min(VB_C, kv_len - c0);
        bool needs_mask = (c0 + VB_C) > kv_max_first;

        // -- QK^T: each simdgroup computes one 8x8 mqk tile = 8 query
        //    rows × 8 KV cols. Simdgroup s owns KV cols
        //    [c0 + s*8, c0 + (s+1)*8). K read DIRECT from device with
        //    `transpose=true` on the simdgroup_load. --
        {
            uint trow = simd_id;
            (void)trow;
            simdgroup_float8x8 mqk = simdgroup_float8x8(0.0f);
            // Walk head_dim in pairs of 8-step k-tiles (16 floats/iter),
            // matching llama.cpp's ILP-packing pattern.
            for (uint k_pair = 0; k_pair < VB_D8 / 2; ++k_pair) {
                simdgroup_float8x8 mq0, mq1, mk0, mk1;
                uint k_off = k_pair * 16;
                simdgroup_load(mq0, sq + k_off + 0, FA_HD);
                simdgroup_load(mq1, sq + k_off + 8, FA_HD);
                device const float* pk =
                    K_cache + (size_t)(c0 + simd_id * 8) * kv_dim
                            + kv_h * FA_HD + k_off;
                simdgroup_load(mk0, pk + 0, kv_dim, ulong2(0, 0), true);
                simdgroup_load(mk1, pk + 8, kv_dim, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }
            // Store mqk to ss at simdgroup s's slot (cols [s*8, s*8+8)).
            simdgroup_store(mqk, ss + simd_id * 8, VB_C,
                            ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Softmax + mask + O rescale.
        //    Each simdgroup owns ONE row (trow = simd_id, 0..7). M and
        //    S live in per-thread scalars, replicated across the 32
        //    lanes — simd_max / simd_sum reductions keep them in sync.
        {
            uint trow = simd_id;
            uint kv_row_max = start_pos + q0 + trow + 1;

            // Each lane reads 2 cols of the row's scores (32 lanes × 2 = 64).
            float s0 = ss[trow * VB_C + lane]      * log2_scale;
            float s1 = ss[trow * VB_C + lane + 32] * log2_scale;
            if (bc_valid < VB_C || needs_mask) {
                uint kv_pos0 = c0 + lane;
                uint kv_pos1 = c0 + lane + 32;
                if (lane      >= bc_valid
                    || (needs_mask && kv_pos0 >= kv_row_max)) {
                    s0 = -INFINITY;
                }
                if (lane + 32 >= bc_valid
                    || (needs_mask && kv_pos1 >= kv_row_max)) {
                    s1 = -INFINITY;
                }
            }

            float blk_max = simd_max(max(s0, s1));
            float m_old = M;
            float m_new = max(m_old, blk_max);
            bool  active = (m_new != -INFINITY);
            float corr = (m_old == -INFINITY) ? 0.0f
                                              : fast::exp2(m_old - m_new);

            float p0 = active ? fast::exp2(s0 - m_new) : 0.0f;
            float p1 = active ? fast::exp2(s1 - m_new) : 0.0f;
            ss[trow * VB_C + lane]      = p0;
            ss[trow * VB_C + lane + 32] = p1;

            float l_block = simd_sum(p0 + p1);
            S = S * corr + l_block;
            M = m_new;

            // Rescale this row of so by corr. Skip when corr == 1 (M
            // unchanged this block) — covers the common steady-state
            // path. When corr == 0 (first valid block, m_old was
            // -INFINITY), so is still all zeros from init, so the
            // multiply is also a no-op; we'd only enter the loop for
            // 0 < corr < 1 (m grew).
            if (corr != 1.0f && corr != 0.0f) {
                for (uint k = 0; k < FA_HD / 32; ++k) {
                    so[trow * FA_HD + lane + k * 32] *= corr;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- PV: O += P · V.
        //    Each simdgroup handles VB_NO (= 4) output column-tiles,
        //    striped across the head_dim. Simdgroup s owns d-cols
        //      [s*8 + ii*8*VB_NSG  ..  +8)  for ii in [0, VB_NO).
        //    lo[ii] is 8x8 (8 query rows × 8 d-cols); P (vs) is 8x8 at
        //    the current cc-chunk of ss; V (mv) is read direct from
        //    device. --
        {
            simdgroup_float8x8 lo[VB_NO];
            for (uint ii = 0; ii < VB_NO; ++ii) {
                uint d_off = simd_id * 8 + ii * 8 * VB_NSG;
                simdgroup_load(lo[ii], so + d_off, FA_HD,
                               ulong2(0, 0), false);
            }
            for (uint cc = 0; cc < VB_C8; ++cc) {
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + cc * 8, VB_C,
                               ulong2(0, 0), false);
                for (uint ii = 0; ii < VB_NO; ++ii) {
                    simdgroup_float8x8 mv;
                    uint d_off = simd_id * 8 + ii * 8 * VB_NSG;
                    device const float* pv =
                        V_cache + (size_t)(c0 + cc * 8) * kv_dim
                                + kv_h * FA_HD + d_off;
                    simdgroup_load(mv, pv, kv_dim,
                                   ulong2(0, 0), false);
                    simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                }
            }
            for (uint ii = 0; ii < VB_NO; ++ii) {
                uint d_off = simd_id * 8 + ii * 8 * VB_NSG;
                simdgroup_store(lo[ii], so + d_off, FA_HD,
                                ulong2(0, 0), false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Finalize: write so / S to device out.
    //    Each simdgroup writes one query row of FA_HD floats. --
    {
        uint trow = simd_id;
        if (trow >= br_valid) return;
        float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
        device float* osrc =
            out + ((size_t)(q0 + trow) * num_heads + h) * FA_HD;
        for (uint k = 0; k < FA_HD / 32; ++k) {
            osrc[lane + k * 32] = so[trow * FA_HD + lane + k * 32] * inv;
        }
    }
}

// ============================================================================
// FlashAttention-2 causal SDPA — GQA-folded, direct-device.
// ============================================================================
//
// Same direct-device design as vA but processes G query-heads per
// threadgroup. K/V hit L2 cache on heads 1..G-1 (same KV head for the
// whole group). Q is reloaded from device per head per block (~6%
// overhead vs ~50% K/V savings at G=2).
//
// TG memory: sq (8 KB) + so (8·G KB) + ss (2 KB).
// 32 KB Apple Silicon TG limit ⇒ G ≤ 2 at VB_Q=8.

template<uint G>
static void sdpa_dd_gqa_impl(
    device const float* Q,
    device const float* K_cache,
    device const float* V_cache,
    device float*       out,
    uint n_tokens, uint num_heads, uint heads_per_kv,
    uint kv_dim, uint start_pos, uint kv_len, float softmax_scale,
    threadgroup float* sq,
    threadgroup float* so,
    threadgroup float* ss,
    uint tg_idx, uint lid
) {
    uint num_groups = num_heads / G;
    uint q_tile = tg_idx / num_groups;
    uint hg     = tg_idx % num_groups;
    uint h_base = hg * G;
    uint q0     = q_tile * VB_Q;
    if (q0 >= n_tokens) return;
    uint br_valid = min(VB_Q, n_tokens - q0);
    uint kv_h     = h_base / heads_per_kv;

    uint simd_id = lid / 32;
    uint lane    = lid % 32;

    const float log2_scale = softmax_scale * M_LOG2E_F;

    for (uint i = lid; i < G * VB_Q * FA_HD; i += FA_THREADS) {
        so[i] = 0.0f;
    }

    float M_h[G], S_h[G];
    for (uint g = 0; g < G; ++g) {
        M_h[g] = -INFINITY;
        S_h[g] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_max_last  = start_pos + q0 + br_valid;
    uint kv_max_first = start_pos + q0 + 1;

    for (uint c0 = 0; c0 < kv_max_last; c0 += VB_C) {
        uint bc_valid   = min(VB_C, kv_len - c0);
        bool needs_mask = (c0 + VB_C) > kv_max_first;

        for (uint g = 0; g < G; ++g) {
            uint h = h_base + g;

            // -- Reload Q[h] from device into sq --
            {
                uint trow = simd_id;
                if (trow < br_valid) {
                    device const float* qsrc =
                        Q + ((size_t)(q0 + trow) * num_heads + h) * FA_HD;
                    for (uint k = 0; k < FA_HD / 32; ++k) {
                        sq[trow * FA_HD + lane + k * 32] =
                            qsrc[lane + k * 32];
                    }
                } else {
                    for (uint k = 0; k < FA_HD / 32; ++k) {
                        sq[trow * FA_HD + lane + k * 32] = 0.0f;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // -- QK^T --
            {
                simdgroup_float8x8 mqk = simdgroup_float8x8(0.0f);
                for (uint k_pair = 0; k_pair < VB_D8 / 2; ++k_pair) {
                    simdgroup_float8x8 mq0, mq1, mk0, mk1;
                    uint k_off = k_pair * 16;
                    simdgroup_load(mq0, sq + k_off + 0, FA_HD);
                    simdgroup_load(mq1, sq + k_off + 8, FA_HD);
                    device const float* pk =
                        K_cache
                        + (size_t)(c0 + simd_id * 8) * kv_dim
                        + kv_h * FA_HD + k_off;
                    simdgroup_load(mk0, pk + 0, kv_dim,
                                   ulong2(0, 0), true);
                    simdgroup_load(mk1, pk + 8, kv_dim,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                    simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
                }
                simdgroup_store(mqk, ss + simd_id * 8, VB_C,
                                ulong2(0, 0), false);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // -- Softmax + rescale so[g] --
            {
                uint trow = simd_id;
                uint kv_row_max = start_pos + q0 + trow + 1;

                float s0 = ss[trow * VB_C + lane]      * log2_scale;
                float s1 = ss[trow * VB_C + lane + 32] * log2_scale;
                if (bc_valid < VB_C || needs_mask) {
                    uint kv_pos0 = c0 + lane;
                    uint kv_pos1 = c0 + lane + 32;
                    if (lane      >= bc_valid
                        || (needs_mask && kv_pos0 >= kv_row_max)) {
                        s0 = -INFINITY;
                    }
                    if (lane + 32 >= bc_valid
                        || (needs_mask && kv_pos1 >= kv_row_max)) {
                        s1 = -INFINITY;
                    }
                }

                float blk_max = simd_max(max(s0, s1));
                float m_old = M_h[g];
                float m_new = max(m_old, blk_max);
                bool  active = (m_new != -INFINITY);
                float corr = (m_old == -INFINITY) ? 0.0f
                    : fast::exp2(m_old - m_new);

                float p0 = active ? fast::exp2(s0 - m_new) : 0.0f;
                float p1 = active ? fast::exp2(s1 - m_new) : 0.0f;
                ss[trow * VB_C + lane]      = p0;
                ss[trow * VB_C + lane + 32] = p1;

                float l_block = simd_sum(p0 + p1);
                S_h[g] = S_h[g] * corr + l_block;
                M_h[g] = m_new;

                if (corr != 1.0f && corr != 0.0f) {
                    uint so_base = g * VB_Q * FA_HD;
                    for (uint k = 0; k < FA_HD / 32; ++k) {
                        so[so_base + trow * FA_HD + lane + k * 32]
                            *= corr;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // -- PV: O += P · V --
            {
                uint so_base = g * VB_Q * FA_HD;
                simdgroup_float8x8 lo[VB_NO];
                for (uint ii = 0; ii < VB_NO; ++ii) {
                    uint d_off = simd_id * 8 + ii * 8 * VB_NSG;
                    simdgroup_load(lo[ii], so + so_base + d_off,
                                   FA_HD, ulong2(0, 0), false);
                }
                for (uint cc = 0; cc < VB_C8; ++cc) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + cc * 8, VB_C,
                                   ulong2(0, 0), false);
                    for (uint ii = 0; ii < VB_NO; ++ii) {
                        simdgroup_float8x8 mv;
                        uint d_off = simd_id * 8 + ii * 8 * VB_NSG;
                        device const float* pv =
                            V_cache
                            + (size_t)(c0 + cc * 8) * kv_dim
                            + kv_h * FA_HD + d_off;
                        simdgroup_load(mv, pv, kv_dim,
                                       ulong2(0, 0), false);
                        simdgroup_multiply_accumulate(
                            lo[ii], vs, mv, lo[ii]);
                    }
                }
                for (uint ii = 0; ii < VB_NO; ++ii) {
                    uint d_off = simd_id * 8 + ii * 8 * VB_NSG;
                    simdgroup_store(lo[ii], so + so_base + d_off,
                                    FA_HD, ulong2(0, 0), false);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // -- Finalize: write so / S to device out, per head. --
    for (uint g = 0; g < G; ++g) {
        uint h = h_base + g;
        uint trow = simd_id;
        if (trow < br_valid) {
            float inv = (S_h[g] > 0.0f) ? (1.0f / S_h[g]) : 0.0f;
            uint so_base = g * VB_Q * FA_HD;
            device float* osrc =
                out + ((size_t)(q0 + trow) * num_heads + h) * FA_HD;
            for (uint k = 0; k < FA_HD / 32; ++k) {
                osrc[lane + k * 32] =
                    so[so_base + trow * FA_HD + lane + k * 32] * inv;
            }
        }
    }
}

#define SDPA_DD_GQA_ENTRY(NAME, G)                                         \
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
    threadgroup float sq[VB_Q * FA_HD];               /* 8 KB */           \
    threadgroup float so[(G) * VB_Q * FA_HD];         /* 8·G KB */         \
    threadgroup float ss[VB_Q * VB_C];                /* 2 KB */           \
    sdpa_dd_gqa_impl<G>(Q, K_cache, V_cache, out, n_tokens, num_heads,    \
        heads_per_kv, kv_dim, start_pos, kv_len, softmax_scale,           \
        sq, so, ss, tg_idx, lid);                                         \
}

SDPA_DD_GQA_ENTRY(attn_sdpa_causal_flash_gqa2_dd, 2)
