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
//   - The Q tile is register-resident for the threadgroup's lifetime, as a
//     steel `MMATile` (simdgroup-matrix fragment layout). Simdgroup g owns
//     rows [g*FA_RPS, (g+1)*FA_RPS); it is staged once through `kv_stage`
//     before the KV loop (device `MMATile::load_safe` is unusable — a
//     steel-header bug corrupts multi-column-frag tiles).
//   - Each KV block is staged once into threadgroup `kv_stage` and reused
//     across the whole query tile (K, then time-multiplexed for V).
//   - The output accumulator is register-resident, same lane partition.
//
// head_dim is fixed to FA_HD=256 (a3b); the threadgroup arrays are sized
// at compile time. The encoder asserts head_dim == 256.
//
// Diff target: per-token cosine >= 0.9999 vs sdpa_cpu. Phase 2 (QK^T) is a
// steel `MMATile` simdgroup-matrix GEMM. Follow-up levers (own arcs):
// simdgroup_matrix P·V; GQA-fold (one TG per (q_tile,kv_h), K/V block
// reused across the 8 shared query-heads).

constant uint FA_BR      = 64;   // query tile
constant uint FA_BC      = 16;   // KV block
constant uint FA_HD      = 256;  // head_dim (a3b)
constant uint FA_THREADS = 256;  // 8 simdgroups
constant uint FA_SIMDS   = FA_THREADS / 32;   // 8
constant uint FA_RPS     = FA_BR / FA_SIMDS;  // rows per simdgroup
constant uint FA_VPL     = FA_HD / 32;        // values per lane = 8
constant uint FA_TPR     = FA_THREADS / FA_BR; // Phase-3 threads per row
constant uint FA_KTILES  = FA_HD / 8;          // 8-deep head_dim ktiles = 32

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

    // O accumulator, register-resident. acc[ri][k] is row
    // (q0 + simd_id*FA_RPS + ri), head_dim index (lane + k*32).
    float acc[FA_RPS][FA_VPL];
    for (uint ri = 0; ri < FA_RPS; ++ri) {
        for (uint k = 0; k < FA_VPL; ++k) {
            acc[ri][k] = 0.0f;
        }
    }

    for (uint r = lid; r < FA_BR; r += FA_THREADS) {
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -- Q tile -> register-resident MMATile `A` (the QK^T GEMM operand). --
    //
    // Steel `MMATile` loads/stores expect the caller to pre-offset the
    // pointer by the lane's simdgroup-matrix fragment coordinate `co`
    // (the `BlockMMA`-constructor idiom): co.y = frag row, co.x = frag col.
    //
    // `A` is [FA_RPS x FA_HD] for this simdgroup = 1 row-frag, FA_KTILES
    // col-frags. The device `MMATile::load_safe` cannot load it (steel
    // `BaseMMAFrag::load_safe` uses off_x where off_y belongs, corrupting
    // any >1-column-frag tile), and the full Q tile is 64 KB — too large
    // for threadgroup memory. So stage Q through `kv_stage` in FA_BR/FA_BC
    // rounds of FA_BC rows; simdgroup g loads its FA_RPS rows in round
    // g/2. One-time, before the KV loop; `kv_stage` is free afterwards.
    using Frag  = mlx::steel::BaseMMAFrag<float, 8, 8>;
    using QTile = mlx::steel::MMATile<float, 1, FA_KTILES>;
    short2 co = Frag::get_coord(lane);
    QTile A;
    for (uint r = 0; r < FA_BR / FA_BC; ++r) {
        for (uint idx = lid; idx < FA_BC * FA_HD; idx += FA_THREADS) {
            uint c = idx / FA_HD, d = idx % FA_HD;   // local row, head_dim
            uint trow = r * FA_BC + c;               // tile-local query row
            kv_stage[idx] = (trow < br_valid)
                ? Q[((size_t)(q0 + trow) * num_heads + h) * FA_HD + d]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id / 2 == r) {
            uint local_row0 = (simd_id - 2 * r) * FA_RPS;   // 0 or FA_RPS
            A.load<float, 1, 1, FA_HD, 1>(
                kv_stage + local_row0 * FA_HD + co.y * FA_HD + co.x);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

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

        // -- Phase 2: QK^T = A·Kᵀ as a steel MMATile simdgroup-matrix GEMM.
        //    K is streamed transposed from `kv_stage` one 8-deep ktile at
        //    a time (B-row = head_dim, B-col = kv pos => str_x=1,
        //    str_y=FA_HD). `C` is [FA_RPS x FA_BC] for this simdgroup,
        //    re-zeroed by the MMATile ctor each KV block. --
        mlx::steel::MMATile<float, 1, 2> C;
        for (uint k = 0; k < FA_KTILES; ++k) {
            mlx::steel::MMATile<float, 1, 2> Bk;
            Bk.load<float, 1, 1, 1, FA_HD>(
                kv_stage + k * 8 + co.y + co.x * FA_HD);
            for (uint n = 0; n < 2; ++n) {
                Frag::mma(C.frag_at(0, n), A.frag_at(0, k),
                          Bk.frag_at(0, n), C.frag_at(0, n));
            }
        }
        // Scale + causal mask, fused into the score frags before the store
        // (no extra barrier). Lane `co`, frag `n` owns the score cells at
        // trow = simd_id*FA_RPS + co.y, c = n*8 + co.x + e for e in {0,1}.
        for (uint n = 0; n < 2; ++n) {
            uint trow = simd_id * FA_RPS + co.y;
            for (uint e = 0; e < 2; ++e) {
                uint c      = n * 8 + co.x + e;
                uint kv_pos = c0 + c;
                bool masked = (c >= bc_valid) ||
                    (needs_mask && kv_pos >= start_pos + q0 + trow + 1);
                C.frag_at(0, n)[e] = masked
                    ? -INFINITY
                    : C.frag_at(0, n)[e] * softmax_scale;
            }
        }
        C.store<float, 1, 1, FA_BC, 1>(
            scores + simd_id * FA_RPS * FA_BC + co.y * FA_BC + co.x);
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
