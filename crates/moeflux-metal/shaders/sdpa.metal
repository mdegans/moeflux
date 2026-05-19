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

    threadgroup float kv_stage[FA_BC * FA_HD];  // 16 KB — K, then V
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
        for (uint idx = lid; idx < FA_BC * FA_HD; idx += FA_THREADS) {
            uint c = idx / FA_HD, d = idx % FA_HD;
            kv_stage[idx] = (c < bc_valid)
                ? K_cache[(size_t)(c0 + c) * kv_dim + kv_h * FA_HD + d]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 2: QK^T — simdgroup-cooperative dot, simd_sum reduce --
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
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 3: per-row online-softmax update. FA_TPR threads
        //    per row; FA_TPR divides 32, so a row's lanes are
        //    contiguous within one simdgroup. --
        {
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
        for (uint idx = lid; idx < FA_BC * FA_HD; idx += FA_THREADS) {
            uint c = idx / FA_HD, d = idx % FA_HD;
            kv_stage[idx] = (c < bc_valid)
                ? V_cache[(size_t)(c0 + c) * kv_dim + kv_h * FA_HD + d]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Phase 5: rescale O, accumulate P·V --
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
