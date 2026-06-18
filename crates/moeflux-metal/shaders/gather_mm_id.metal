// moeflux-metal: MoE gathered matmul, expert-as-grid-Z variant.
//
// Provenance
// ----------
// The two `moe_mul_mm_id_*` kernels in this file are a port of the
// `kernel_mul_mm_id_map0` + `kernel_mul_mm_id` pair from llama.cpp's
// Metal backend (`ggml-metal.metal:9618` and `:9684` respectively).
// Upstream is © 2023-2024 Georgi Gerganov & contributors, MIT.
// See ../NOTICE for the attribution block.
//
// What's ported 1:1
//   * Algorithm shape: one dispatch covers all experts (`tgpig.z =
//     expert`); per-expert early-return via `htpe[expert]`; gather
//     of activations via `hids[expert][n] = token*k + slot` rather
//     than a host-side bucket-permute.
//   * Tile sizes (NR0=64, NR1=32, NK=32), threadgroup-memory layout
//     (`sa` at 0, `sb` at 4096), simdgroup MMA fragment indexing
//     (`mc[8]` indexed by `(sgitg, ik)`), output store ordering.
//   * `mul_mm_id_map0`: identical structure, hardcoded `K = 8`
//     (a3b top-k), uint16_t-per-id staging in threadgroup memory.
//
// What's moeflux-adapted
//   * Quant format: **W4_gs64** (4-bit affine, group_size=64, bf16
//     scales+biases per group). The templated `dequantize_func`
//     hook in llama.cpp is replaced with an inline dequant of the
//     16 nibbles the kernel needs per K-tile (8 bytes packed →
//     16 f32 weights with `nibble*scale + bias`).
//   * Input/output element type fixed to f32 (llama.cpp templates
//     over half/bfloat/f32). Saves the half<->f32 cast cost the
//     templated path adds.
//   * `combine_topk` reduction kernel — a small, moeflux-specific
//     post-pass that multiplies the slot-indexed gather output by
//     the per-token router weights to produce `[n_tokens,
//     hidden_dim]`. Not present in llama.cpp (their MoE pipeline
//     does the combine in a separate `ggml_mul_mat` node).
//
// What we DON'T port
//   * The `GGML_METAL_HAS_TENSOR` path (Apple TensorOps; M5-only,
//     hardware we don't target).
//   * The `FC_mul_mm_bc_inp` byte-copy fast-path (only meaningful
//     for float inputs, not quantized; we always dequant).
//   * Multi-quant template instantiations (we have ONE format).

// ===============================================================
// kargs
// ===============================================================

struct moeflux_mm_id_map0_args {
    // Number of experts (= threadgroup-Z dim of the main kernel; =
    // threads-per-threadgroup of this map0 kernel).
    int32_t  n_experts;
    // Number of tokens in this chunk (= ne21 in llama.cpp).
    int32_t  n_tokens;
    // Top-k. Compile-time-baked as `K_TOPK = 8` for now — passed
    // for sanity / future-proofing only.
    int32_t  k;
};

struct moeflux_mm_id_args {
    // Contraction dimension. Gate/up: K = hidden_dim. Down: K =
    // moe_intermediate.
    int32_t  K;
    // Output dim — rows of one expert's weight matrix. Gate/up:
    // N = moe_intermediate. Down: N = hidden_dim.
    int32_t  N;
    // Per-expert byte stride into `src0` (the layer's expert
    // blob base). For a3b this is `expert_size_4bit = 1,769,472`
    // bytes (same for all three projections).
    uint64_t nb02;
    // Packed-weight row stride for THIS matmul, in bytes.
    // Gate/up: K=hidden=2048 → 1024 B. Down: K=moe_inter=512 → 256 B.
    uint64_t nb01_w;
    // Scales row stride for THIS matmul, in bytes (bf16). Gate/up:
    // 64 B/row. Down: 16 B/row. Biases share the same stride.
    uint64_t nb01_s;
    // Within-expert byte offset to the packed weights for THIS
    // projection. 0 for gate; `expert_block_bytes_4bit` for up;
    // `2*expert_block_bytes_4bit` for down. Scales/biases offsets
    // are derived (packed_off + expert_weight_bytes_4bit and
    // packed_off + expert_weight_bytes_4bit + expert_scale_bytes).
    uint64_t packed_off;
    uint64_t scales_off;
    uint64_t biases_off;
    // Activation tensor layout, llama.cpp-style nb10/nb11/nb12.
    // nb10 = sizeof(f32) = 4. nb11 = bytes between slots within
    // one token (only meaningful for the down matmul, where input
    // is `[n_tokens, k, moe_inter]`; pass 0 for gate/up). nb12 =
    // bytes between tokens.
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    // Token count.
    int32_t  n_tokens;
    // top-k.
    int32_t  k;
    // ne11 for the activation tensor: 1 for gate/up (no slot dim),
    // k for down. Matches llama.cpp's broadcast handling.
    int32_t  ne11;
};

// ===============================================================
// map0 — build htpe[] + hids[][] from per-token expert indices.
//
// One dispatch, 1 threadgroup, `n_experts` threads. Each thread
// scans all tokens and emits its expert's assignments.
//
// `K_TOPK` is the (compile-time) top-k. a3b: 8.
// ===============================================================

template <short K_TOPK>
kernel void moeflux_mm_id_map0_impl(
    constant moeflux_mm_id_map0_args & args,
    device const int32_t * indices,   // [n_tokens, k] i32
    device uint32_t      * htpe,      // [n_experts]
    device int32_t       * hids,      // [n_experts, n_tokens]
    threadgroup char     * shmem      [[threadgroup(0)]],
    ushort tpitg [[thread_position_in_threadgroup]],
    ushort ntg   [[threads_per_threadgroup]]
) {
    const short ide = (short)tpitg;  // this thread's expert id
    const int   n_tokens = args.n_tokens;

    uint32_t n_all = 0;
    device int32_t * hids_row = hids + (int)ide * n_tokens;

    // Stage `ntg` consecutive tokens' indices into threadgroup
    // memory per outer-loop iteration, then every thread scans
    // those `ntg * K_TOPK` entries against its own expert id.
    // This amortizes the index-load cost across threads (one
    // global read per token, vs n_experts reads per token in
    // the naive version).
    for (int t_base = 0; t_base < n_tokens; t_base += ntg) {
        if (t_base + tpitg < n_tokens) {
            device const int32_t * src_i32 =
                indices + (t_base + tpitg) * K_TOPK;
            threadgroup uint16_t * sids =
                (threadgroup uint16_t *)shmem + tpitg * K_TOPK;
            #pragma unroll
            for (short s = 0; s < K_TOPK; s++) {
                sids[s] = (uint16_t)src_i32[s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short tt = 0; tt < ntg; tt++) {
            if (t_base + tt >= n_tokens) {
                break;
            }
            threadgroup const uint16_t * sids =
                (threadgroup const uint16_t *)shmem + tt * K_TOPK;

            short sel = 0;
            #pragma unroll
            for (short s = 0; s < K_TOPK; s++) {
                // (sids[s] == ide) is 0/1; multiply by (s+1) so
                // sel = (matched slot)+1, or 0 if no match. The
                // pattern matches llama.cpp's `sel` accumulator
                // and lets us mask via `n_all += sel > 0`.
                sel += ((sids[s] == ide) ? (s + 1) : 0);
            }

            // Encode: id = token * K_TOPK + slot. We emit only
            // when sel > 0 (a match was found).
            hids_row[n_all] = (t_base + tt) * K_TOPK + (sel - 1);
            n_all += (sel > 0) ? 1u : 0u;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    htpe[ide] = n_all;
}

// Per-top-k instantiations. a3b routes top-8, a17b top-10. The
// template body is shared — only the compile-time `K_TOPK` (loop
// bounds, indices stride, hids encoding) differs, so the inner
// loops still unroll per variant. The dispatch in `lib.rs`
// (`MoeIdMap0Call::encode`) picks the PSO by the runtime `k`.
template [[host_name("moeflux_mm_id_map0_k8")]]
kernel decltype(moeflux_mm_id_map0_impl<8>)
moeflux_mm_id_map0_impl<8>;

template [[host_name("moeflux_mm_id_map0_k10")]]
kernel decltype(moeflux_mm_id_map0_impl<10>)
moeflux_mm_id_map0_impl<10>;

// ===============================================================
// Main matmul — one dispatch over all experts.
//
// Grid: `((n_tokens + NR1 - 1) / NR1, (N + NR0 - 1) / NR0,
//         n_experts)`. Each threadgroup is 128 threads (4
// simdgroups). The `tgpig.z = expert` dimension lets us index
// into the per-expert weight block and the per-expert htpe[]
// early-return.
//
// MMA fragment layout — preserved 1:1 from llama.cpp:
//   sa: NR0 × NK = 64 × 32 fp32 weights, threadgroup memory at 0
//   sb: NR1 × NK = 32 × 32 fp32 activations, threadgroup memory at 4096
//   mc[8]: 8 accumulator fragments per simdgroup, holding the
//          64 × 32 output tile. Indexed by (sgitg, ik).
//   At store: `temp_str` indexing recasts the 8 fragments back
//   into the (64 × 32) output tile. Order MUST match the load
//   order or output is silently scrambled.
// ===============================================================

constant constexpr int NR0 = 64;  // output rows / N tile
constant constexpr int NR1 = 32;  // M rows / token tile
constant constexpr int NK  = 32;  // K stride per outer step

kernel void moeflux_mm_id(
    constant moeflux_mm_id_args & args,
    device const char    * src0,      // expert blob base
    device const char    * src1,      // activations
    device const uint32_t * htpe,     // [n_experts]
    device const int32_t  * hids,     // [n_experts, n_tokens]
    device       float    * dst,      // [n_tokens, k, N]
    threadgroup  char     * shmem     [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Half storage for sa/sb matches llama.cpp's quant-path
    // instantiation and keeps the (sa, sb) pair fitting in the
    // first 8 KB of shmem — the same 8 KB block is reused as the
    // f32 output-staging area later. Accumulator type stays f32
    // via `simdgroup_float8x8 mc[8]`.
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NL0 = NK / 16;  // = 2; threads-per-N-row in loader
    constexpr int NL1 = NK / 8;   // = 4; threads-per-M-row in loader

    const int im = (int)tgpig.z;          // expert
    const int r0 = (int)tgpig.y * NR0;    // output-row block start
    const int r1 = (int)tgpig.x * NR1;    // M-row block start

    const int32_t neh1 = (int32_t)htpe[im];

    // Skip threadgroups whose M-block is past this expert's count.
    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.N    - r0 < NR0) ? (short)(args.N    - r0) : (short)NR0;
    const short nr1 = (neh1      - r1 < NR1) ? (short)(neh1      - r1) : (short)NR1;

    const short lr0 = (((short)tiitg / NL0) < nr0) ? ((short)tiitg / NL0) : (short)(nr0 - 1);
    const short lr1 = (((short)tiitg / NL1) < nr1) ? ((short)tiitg / NL1) : (short)(nr1 - 1);

    const short il0 = (short)(tiitg % NL0);

    // Map this thread's M-row to its (token, slot) via hids.
    const int id  = hids[im * args.n_tokens + r1 + lr1];
    const int i12 = id / args.k;                       // token
    const int i11 = (args.ne11 > 1) ? (id % args.k) : 0;  // slot or 0

    // Per-expert weight base pointer (gate/up/down all share nb02
    // = expert_size_4bit; the projection-within-expert byte offset
    // comes from packed_off / scales_off / biases_off).
    const uint64_t expert_off = (uint64_t)im * args.nb02;
    device const uint8_t  * x_packed_base =
        (device const uint8_t  *)(src0 + expert_off + args.packed_off);
    device const bfloat16_t * x_scales_base =
        (device const bfloat16_t *)(src0 + expert_off + args.scales_off);
    device const bfloat16_t * x_biases_base =
        (device const bfloat16_t *)(src0 + expert_off + args.biases_off);

    // Activation base pointer for this thread's (token, slot).
    // `iy` = column-within-K this thread's loader handles.
    const short iy = (short)(8 * (tiitg % NL1));
    device const float * y = (device const float *)(
        src1
      + args.nb12 * i12
      + args.nb11 * i11
      + args.nb10 * iy);

    // Initialize 8 simdgroup output fragments to zero. Each
    // `simdgroup_float8x8` is one 8x8 tile; mc[8] covers the
    // 64x32 output tile (2 simdgroups along N × 4 along M, or
    // similar — exact layout depends on sgitg routing below).
    simdgroup_float8x8 mc[8];
    #pragma unroll
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    // ---------------------------------------------------------
    // K-loop: for each NK-wide slab of the contraction dim:
    //   1) Dequant 16 nibbles of weight (W4_gs64) → sa tile.
    //   2) Load 8 floats of activation → sb tile.
    //   3) simdgroup_load both tiles; simdgroup_multiply_accumulate.
    // ---------------------------------------------------------

    // Inner W4_gs64 dequant: each thread loads `8 bytes packed =
    // 16 nibbles = 16 f32 weights` per K-tile. With group_size=64
    // and NK=32, two consecutive K-tiles share the same (scale,
    // bias) — tracked via the `loop_k / 64` group index below.
    // `il0` (= `tiitg % NL0`, value 0 or 1) is the within-K-tile
    // thread-offset and is fixed per thread.

    // Each thread writes 16 floats to `sa`. Row in the (64 × 32)
    // tile: `(tiitg / NL0)`, columns: `16 * il0 + i` for i in 0..16.
    // The threadgroup-memory permutation (8x8 sub-blocks for MMA
    // staging) follows llama.cpp's pattern.

    for (int loop_k = 0; loop_k < args.K; loop_k += NK) {
        // --- weight load + dequant -------------------------------
        // Pointer to this thread's 8 bytes of packed weight for
        // this K-tile. `(r0 + lr0)` is the output-row index;
        // `nb01_w * (r0 + lr0)` is the row offset; `8 * il0` is
        // the within-K-tile byte offset (il0 splits the 16-byte
        // NK=32 slab into two halves); `loop_k / 2` is the K-tile
        // byte offset within the row (2 nibbles = 1 byte).
        device const uint8_t * w_thread = x_packed_base
            + args.nb01_w * (r0 + lr0)
            + 8 * il0
            + (loop_k / 2);

        // Scale/bias for this thread's group. Group index =
        // `(loop_k) / GROUP_SIZE = loop_k / 64`. nb01_s is the
        // byte stride between rows in the scales/biases planes
        // (each row has K/GROUP_SIZE bf16 entries).
        device const bfloat16_t * s_thread = x_scales_base
            + (args.nb01_s / 2) * (r0 + lr0)
            + (loop_k / 64);
        device const bfloat16_t * b_thread = x_biases_base
            + (args.nb01_s / 2) * (r0 + lr0)
            + (loop_k / 64);
        float scale = (float)(*s_thread);
        float bias  = (float)(*b_thread);

        // Dequant: read 8 bytes, unpack 16 nibbles, write 16 floats
        // into `sa` at the right (sub-tile-permuted) positions.
        // The permutation matches llama.cpp's "pre-tensor" sa store
        // pattern at lines 9794-9815 of ggml-metal.metal.
        float dq[16];
        #pragma unroll
        for (short i = 0; i < 8; i++) {
            uint8_t byte = w_thread[i];
            dq[2 * i + 0] = scale * (float)(byte & 0x0f) + bias;
            // Note: the high nibble represents the *next* element
            // in the K direction. MLX's dequant uses `scale/16` for
            // the high-nibble scale because it bit-shifts the high
            // nibble in place; we extract via shift so plain `scale`
            // applies. Match the llama.cpp dequant convention here.
            dq[2 * i + 1] = scale * (float)((byte >> 4) & 0x0f) + bias;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;            // 0..3
            const short sy = ((short)tiitg / NL0) / 8;   // 0..7
            const short lx = ((short)tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = (half)dq[i];
        }

        // --- activation load (no dequant; f32 → half) ------------
        {
            const short sx = (short)(tiitg % NL1);
            const short sy = ((short)tiitg / NL1) / 8;
            const short ly = ((short)tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;

            // Load 8 contiguous f32s from `y`, cast to half, write
            // to sb. With NL1=4 and tiitg cycling, each pair of
            // threads covers one 8-wide K-slice.
            #pragma unroll
            for (short i = 0; i < 8; i++) {
                sb[64 * ib + 8 * ly + i] = (half)y[i];
            }
        }

        // Advance activation pointer for next K-tile. (No `il`
        // advance — `il0` is a fixed per-thread offset; the K-tile
        // byte offset is encoded via `loop_k / 2` in the weight
        // pointer construction above.)
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- simdgroup matmul ------------------------------------
        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        #pragma unroll
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            simdgroup_half8x8 ma[4];
            #pragma unroll
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            simdgroup_half8x8 mb[2];
            #pragma unroll
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(
                    mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // ---------------------------------------------------------
    // Output stage — store fragments to threadgroup, then scatter
    // to per-(token, slot, row) destination addresses via hids.
    // ---------------------------------------------------------
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;

    #pragma unroll
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i],
                        temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                        NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup-team writes 8 rows of the M-tile back to dst.
    // Iterate j = sgitg, sgitg+4, ... so each simdgroup gets 8 rows
    // of the 32-row tile.
    for (short j = sgitg; j < nr1; j += 4) {
        const int id  = hids[im * args.n_tokens + r1 + j];
        const int slot = id % args.k;
        const int tok  = id / args.k;

        device float * D = dst
            + (int64_t)tok * args.k * args.N
            + (int64_t)slot * args.N
            + r0;

        threadgroup const float * C = (threadgroup const float *)shmem
                                    + j * NR0;

        // SIMD-cooperative copy: each lane writes a stride-32 strip.
        int i = (int)tiisg;
        for (; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
}

// ===============================================================
// combine_topk — reduce slot-indexed gather output to per-token
// hidden vectors by weighted sum over the k slots.
//
//   out[t, r] = sum_{s=0..k-1} weights[t, s] * mid[t, s, r]
//
// One threadgroup per token; threads cover the hidden-dim axis in
// stride-tg_size strides. Cheap O(n_tokens × k × hidden_dim) pass;
// trivial vs the matmul.
// ===============================================================

struct moeflux_combine_topk_args {
    int32_t n_tokens;
    int32_t hidden_dim;
    int32_t k;
};

kernel void moeflux_combine_topk(
    constant moeflux_combine_topk_args & args,
    device const float * mid,        // [n_tokens, k, hidden_dim]
    device const float * weights,    // [n_tokens, k]
    device       float * out,        // [n_tokens, hidden_dim]
    uint  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_position_in_threadgroup]],
    ushort ntg   [[threads_per_threadgroup]]
) {
    const int t = (int)tgpig;
    if (t >= args.n_tokens) {
        return;
    }
    const int k          = args.k;
    const int hidden_dim = args.hidden_dim;

    device const float * mid_t = mid + (int64_t)t * k * hidden_dim;
    device const float * w_t   = weights + t * k;
    device       float * out_t = out + (int64_t)t * hidden_dim;

    for (int r = (int)tiitg; r < hidden_dim; r += (int)ntg) {
        float acc = 0.0f;
        for (int s = 0; s < k; s++) {
            acc += w_t[s] * mid_t[(int64_t)s * hidden_dim + r];
        }
        out_t[r] = acc;
    }
}
