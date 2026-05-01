//! GPU MLA forward — Phase 3 of the GPU MLA port.
//!
//! Three Metal kernels implement the folded MLA form for DeepSeek-V3
//! / Cogito-V2:
//!
//! 1. [`encode_mla_q_prime_4bit`] — `q'[h, c] = Σ_i q_nope[h, i] *
//!    dequant(W[h * kv_b_per_head + i, c])` for `i ∈ [0, nope)`.
//!    Output `[num_heads, kv_lora_rank]`. Per-head matvec with a
//!    "row-wise" access pattern that doesn't fit the standard
//!    `dequant_matvec_4bit_v3` per-row dispatch — custom kernel.
//! 2. [`encode_mla_sdpa_folded`] — per head:
//!    `scores[t] = q'[h] · latent[t] + q_pe[h] · rope_k[t]`,
//!    softmax with the YaRN scale baked in, then
//!    `v_combine[h, c] = Σ_t scores[t] * latent[t, c]`.
//! 3. [`encode_mla_out_per_head_4bit`] — `out[h, f] = Σ_c
//!    v_combine[h, c] * dequant(W[h * kv_b_per_head + nope + f, c])`.
//!    SIMD-cooperative dot product per output element.
//!
//! The naive form (decompress `kv_b_proj @ latent[j]` per cached
//! position) lives in [`super::mla_attn_cpu`] and is the diff oracle
//! for these kernels — algebraically equivalent up to floating-point
//! reduction order.
//!
//! ## Validation
//!
//! Per-kernel unit tests construct synthetic inputs, compute the
//! expected output via folded arithmetic on the host, and assert
//! the GPU result matches within a tolerance covering Metal's
//! reduction nondeterminism (`≤ 1e-3` absolute on integer-quantized
//! inputs, looser than YaRN's `4 ULP` because the dot products
//! span hundreds of terms).

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, MTLSize, NSUInteger,
};

use super::metal::{MetalBackend, MetalError};

/// Errors from the GPU MLA dispatchers.
#[derive(Debug, thiserror::Error)]
pub enum GpuMlaError {
    #[error("cache_len {got} exceeds MAX_CACHE_TG {cap}")]
    CacheLenTooLarge { got: u32, cap: u32 },
    #[error("Metal backend: {0}")]
    Metal(#[from] MetalError),
}

/// Threadgroup-shared `scores[]` capacity in [`encode_mla_sdpa_folded`].
/// Must match the Metal `MLA_MAX_CACHE_TG` constant. Bumping this
/// requires bumping the kernel-side constant first.
pub const MLA_MAX_CACHE_TG: u32 = 4096;

/// Threads per `mla_sdpa_folded` threadgroup. Must match the
/// `MLA_THREADS_PER_HEAD` Metal constant.
pub const MLA_THREADS_PER_HEAD: u32 = 128;

/// Pre-fetched compute pipelines for the MLA kernels. Built once per
/// `RsCtx` and threaded into per-token dispatch helpers. Phase 4a
/// added `split_q_kv` and `cache_append` so the whole MLA forward
/// fits in one Metal command buffer (no host bounces between
/// projections, RoPE, and SDPA).
pub struct MlaPipelines {
    pub q_prime: ComputePipelineState,
    pub sdpa: ComputePipelineState,
    pub out_per_head: ComputePipelineState,
    pub split_q_kv: ComputePipelineState,
    pub cache_append: ComputePipelineState,
    /// Phase 6 — tiled SDPA accumulator + finalize, used when
    /// `cache_len > MLA_MAX_CACHE_TG` to keep `scores[]` within the
    /// 32 KB threadgroup-memory budget.
    pub sdpa_tile_accumulate: ComputePipelineState,
    pub sdpa_tile_finalize: ComputePipelineState,
}

impl MlaPipelines {
    pub fn new(metal: &mut MetalBackend) -> Result<Self, MetalError> {
        Ok(Self {
            q_prime: metal.pipeline("mla_q_prime_4bit")?.clone(),
            sdpa: metal.pipeline("mla_sdpa_folded")?.clone(),
            out_per_head: metal.pipeline("mla_out_per_head_4bit")?.clone(),
            split_q_kv: metal.pipeline("mla_split_q_kv")?.clone(),
            cache_append: metal.pipeline("mla_kv_cache_append")?.clone(),
            sdpa_tile_accumulate: metal
                .pipeline("mla_sdpa_tile_accumulate")?
                .clone(),
            sdpa_tile_finalize: metal
                .pipeline("mla_sdpa_tile_finalize")?
                .clone(),
        })
    }
}

/// Encode `q' = q_nope @ kv_b_proj_K_per_head` into `cmdbuf`.
///
/// Buffer ownership / shapes:
/// - `w_packed`, `scales`, `biases`: full `kv_b_proj` matrix at byte
///   offset 0 (use `set_buffer(.., offset)` if the shared weight
///   buffer holds other tensors before this one).
/// - `q_nope`: shared-storage buffer, `num_heads * nope` floats.
/// - `q_prime`: shared-storage buffer, `num_heads * kv_lora_rank`
///   floats. Overwritten.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_q_prime_4bit(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    w_packed: &Buffer,
    w_packed_off: u64,
    scales: &Buffer,
    scales_off: u64,
    biases: &Buffer,
    biases_off: u64,
    q_nope: &Buffer,
    q_prime: &Buffer,
    num_heads: u32,
    nope: u32,
    kv_lora_rank: u32,
    kv_b_per_head: u32,
    group_size: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(w_packed), w_packed_off);
    enc.set_buffer(1, Some(scales), scales_off);
    enc.set_buffer(2, Some(biases), biases_off);
    enc.set_buffer(3, Some(q_nope), 0);
    enc.set_buffer(4, Some(q_prime), 0);
    enc.set_bytes(5, 4, (&num_heads as *const u32).cast());
    enc.set_bytes(6, 4, (&nope as *const u32).cast());
    enc.set_bytes(7, 4, (&kv_lora_rank as *const u32).cast());
    enc.set_bytes(8, 4, (&kv_b_per_head as *const u32).cast());
    enc.set_bytes(9, 4, (&group_size as *const u32).cast());
    let total_outputs = num_heads * kv_lora_rank;
    let num_tgs = total_outputs.div_ceil(256);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode the folded SDPA inner loop for all heads.
///
/// `softmax_scale` is the post-multiply applied to `scores[t]` before
/// `softmax` (typically `(1/sqrt(qk_head_dim)) * mscale²`).
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_sdpa_folded(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    q_prime: &Buffer,
    q_pe: &Buffer,
    latent_cache: &Buffer,
    rope_k_cache: &Buffer,
    v_combine: &Buffer,
    num_heads: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
    cache_len: u32,
    softmax_scale: f32,
) -> Result<(), GpuMlaError> {
    if cache_len > MLA_MAX_CACHE_TG {
        return Err(GpuMlaError::CacheLenTooLarge {
            got: cache_len,
            cap: MLA_MAX_CACHE_TG,
        });
    }
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(q_prime), 0);
    enc.set_buffer(1, Some(q_pe), 0);
    enc.set_buffer(2, Some(latent_cache), 0);
    enc.set_buffer(3, Some(rope_k_cache), 0);
    enc.set_buffer(4, Some(v_combine), 0);
    enc.set_bytes(5, 4, (&num_heads as *const u32).cast());
    enc.set_bytes(6, 4, (&kv_lora_rank as *const u32).cast());
    enc.set_bytes(7, 4, (&qk_rope_head_dim as *const u32).cast());
    enc.set_bytes(8, 4, (&cache_len as *const u32).cast());
    enc.set_bytes(9, 4, (&softmax_scale as *const f32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_heads as NSUInteger, 1, 1),
        MTLSize::new(MLA_THREADS_PER_HEAD as NSUInteger, 1, 1),
    );
    enc.end_encoding();
    Ok(())
}

/// Phase 6 — encode the tiled folded SDPA into `cmdbuf`. Uses the
/// online-softmax accumulator to support `cache_len > MLA_MAX_CACHE_TG`
/// (the threadgroup-memory cap of the single-shot kernel).
///
/// Running state buffers (`running_max`, `running_denom`,
/// `v_combine_partial`) are caller-owned and must be sized
/// `[num_heads]`, `[num_heads]`, `[num_heads * kv_lora_rank]`. They're
/// internally consistent across tile dispatches via Metal's implicit
/// device-memory ordering between consecutive compute encoders in
/// the same cmdbuf.
///
/// Multi-tile output is mathematically equivalent to single-shot up
/// to floating-point reordering — bit-exact only at `cache_len ==
/// MLA_TILE_SIZE` (one tile, no merging). Cosine ≥ 0.9999 vs the
/// single-shot reference is the validation target for cache_len > tile_size.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_sdpa_folded_tiled(
    cmdbuf: &CommandBufferRef,
    pipe_accumulate: &ComputePipelineState,
    pipe_finalize: &ComputePipelineState,
    q_prime: &Buffer,
    q_pe: &Buffer,
    latent_cache: &Buffer,
    rope_k_cache: &Buffer,
    v_combine: &Buffer,
    running_max: &Buffer,
    running_denom: &Buffer,
    v_combine_partial: &Buffer,
    num_heads: u32,
    kv_lora_rank: u32,
    qk_rope_head_dim: u32,
    cache_len: u32,
    softmax_scale: f32,
) -> Result<(), GpuMlaError> {
    if cache_len == 0 {
        return Ok(());
    }
    let tile_size = MLA_MAX_CACHE_TG;
    let num_tiles = cache_len.div_ceil(tile_size);

    for tile_idx in 0..num_tiles {
        let tile_start = tile_idx * tile_size;
        let tile_end = (tile_start + tile_size).min(cache_len);
        let is_first: u32 = if tile_idx == 0 { 1 } else { 0 };
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe_accumulate);
        enc.set_buffer(0, Some(q_prime), 0);
        enc.set_buffer(1, Some(q_pe), 0);
        enc.set_buffer(2, Some(latent_cache), 0);
        enc.set_buffer(3, Some(rope_k_cache), 0);
        enc.set_buffer(4, Some(running_max), 0);
        enc.set_buffer(5, Some(running_denom), 0);
        enc.set_buffer(6, Some(v_combine_partial), 0);
        enc.set_bytes(7, 4, (&num_heads as *const u32).cast());
        enc.set_bytes(8, 4, (&kv_lora_rank as *const u32).cast());
        enc.set_bytes(9, 4, (&qk_rope_head_dim as *const u32).cast());
        enc.set_bytes(10, 4, (&tile_start as *const u32).cast());
        enc.set_bytes(11, 4, (&tile_end as *const u32).cast());
        enc.set_bytes(12, 4, (&softmax_scale as *const f32).cast());
        enc.set_bytes(13, 4, (&is_first as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new(num_heads as NSUInteger, 1, 1),
            MTLSize::new(MLA_THREADS_PER_HEAD as NSUInteger, 1, 1),
        );
        enc.end_encoding();
    }

    // Finalize: v_combine[h, c] = v_combine_partial[h, c] / running_denom[h].
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe_finalize);
    enc.set_buffer(0, Some(v_combine_partial), 0);
    enc.set_buffer(1, Some(running_denom), 0);
    enc.set_buffer(2, Some(v_combine), 0);
    enc.set_bytes(3, 4, (&num_heads as *const u32).cast());
    enc.set_bytes(4, 4, (&kv_lora_rank as *const u32).cast());
    let total = num_heads * kv_lora_rank;
    let num_tgs = total.div_ceil(256);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();

    Ok(())
}

/// Encode `out_per_head = v_combine @ kv_b_proj_V_per_head` into `cmdbuf`.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_out_per_head_4bit(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    w_packed: &Buffer,
    w_packed_off: u64,
    scales: &Buffer,
    scales_off: u64,
    biases: &Buffer,
    biases_off: u64,
    v_combine: &Buffer,
    out_per_head: &Buffer,
    num_heads: u32,
    nope: u32,
    kv_lora_rank: u32,
    v_head_dim: u32,
    kv_b_per_head: u32,
    group_size: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(w_packed), w_packed_off);
    enc.set_buffer(1, Some(scales), scales_off);
    enc.set_buffer(2, Some(biases), biases_off);
    enc.set_buffer(3, Some(v_combine), 0);
    enc.set_buffer(4, Some(out_per_head), 0);
    enc.set_bytes(5, 4, (&num_heads as *const u32).cast());
    enc.set_bytes(6, 4, (&nope as *const u32).cast());
    enc.set_bytes(7, 4, (&kv_lora_rank as *const u32).cast());
    enc.set_bytes(8, 4, (&v_head_dim as *const u32).cast());
    enc.set_bytes(9, 4, (&kv_b_per_head as *const u32).cast());
    enc.set_bytes(10, 4, (&group_size as *const u32).cast());
    let total_outputs = num_heads * v_head_dim;
    enc.dispatch_thread_groups(
        MTLSize::new(total_outputs as NSUInteger, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLResourceOptions;

    /// Pack `weights` (one f32 per element, row-major
    /// `[out_rows, in_cols]`) into the MLX 4-bit affine format with
    /// `group_size = 64`. Returns `(packed_u32_words, scales_bf16,
    /// biases_bf16)`. The unit tests use this to drive the dequant
    /// kernels with reproducible weight values; production weights
    /// come straight from the manifest.
    fn pack_4bit_mlx(
        weights: &[f32],
        out_rows: usize,
        in_cols: usize,
        group_size: usize,
    ) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
        assert!(in_cols % group_size == 0);
        assert!(group_size % 8 == 0);
        let num_groups = in_cols / group_size;
        let packed_cols = in_cols / 8;
        let mut packed = vec![0u32; out_rows * packed_cols];
        let mut scales = vec![0u16; out_rows * num_groups];
        let mut biases = vec![0u16; out_rows * num_groups];
        for r in 0..out_rows {
            for g in 0..num_groups {
                let row_start = r * in_cols + g * group_size;
                let group =
                    &weights[row_start..row_start + group_size];
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for &x in group {
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
                // Affine quant: q ∈ [0, 15], x ≈ q * scale + bias
                // with bias = lo and scale = (hi - lo) / 15.
                let scale = if hi > lo {
                    (hi - lo) / 15.0
                } else {
                    1.0
                };
                let bias = lo;
                scales[r * num_groups + g] = f32_to_bf16(scale);
                biases[r * num_groups + g] = f32_to_bf16(bias);
                // Quantize each element + pack 8-per-uint32.
                for k in 0..group_size / 8 {
                    let mut word = 0u32;
                    for j in 0..8 {
                        let idx = g * group_size + k * 8 + j;
                        let x = weights[r * in_cols + idx];
                        let q = if scale > 0.0 {
                            ((x - bias) / scale).round().clamp(0.0, 15.0)
                                as u32
                        } else {
                            0
                        };
                        word |= q << (j * 4);
                    }
                    packed[r * packed_cols + g * (group_size / 8) + k] =
                        word;
                }
            }
        }
        (packed, scales, biases)
    }

    fn f32_to_bf16(x: f32) -> u16 {
        let bits = x.to_bits();
        // Round-to-nearest-even (RNE) — matches the standard
        // f32→bf16 conversion that the production weight pipeline
        // emits.
        let rounding_bias = ((bits >> 16) & 1) + 0x7fff;
        ((bits.wrapping_add(rounding_bias)) >> 16) as u16
    }

    fn bf16_to_f32(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    /// Dequant a packed 4-bit matrix back to f32. Used to compute
    /// the host-side reference for the GPU diff.
    fn dequant_4bit(
        packed: &[u32],
        scales: &[u16],
        biases: &[u16],
        out_rows: usize,
        in_cols: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let num_groups = in_cols / group_size;
        let packed_cols = in_cols / 8;
        let mut out = vec![0.0f32; out_rows * in_cols];
        for r in 0..out_rows {
            for c in 0..in_cols {
                let g = c / group_size;
                let scale =
                    bf16_to_f32(scales[r * num_groups + g]);
                let bias = bf16_to_f32(biases[r * num_groups + g]);
                let word =
                    packed[r * packed_cols + (c / 8)];
                let nib = ((word >> ((c % 8) * 4)) & 0xF) as f32;
                out[r * in_cols + c] = nib * scale + bias;
            }
        }
        out
    }

    fn shared_buf_with_data<T: Copy>(
        device: &metal::Device,
        data: &[T],
    ) -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr().cast(),
            (data.len() * std::mem::size_of::<T>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn shared_buf_zeroed(
        device: &metal::Device,
        n_floats: usize,
    ) -> Buffer {
        let bytes = (n_floats * std::mem::size_of::<f32>()) as NSUInteger;
        let b = device.new_buffer(
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::write_bytes(
                b.contents() as *mut u8,
                0,
                bytes as usize,
            );
        }
        b
    }

    fn read_back_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        unsafe {
            let p = buf.contents() as *const f32;
            let s = std::slice::from_raw_parts(p, n);
            out.copy_from_slice(s);
        }
        out
    }

    /// Synthetic 3a check — small dimensions, single head.
    /// Constructs random q_nope and W, packs W as 4-bit, computes
    /// expected `q'` by dequantizing W on the host, then asserts
    /// the GPU `mla_q_prime_4bit` output matches.
    #[test]
    fn mla_q_prime_4bit_matches_host() {
        let mut metal = match MetalBackend::new() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[gpu_mla] skipping: Metal init failed: {e:?}");
                return;
            }
        };

        let num_heads: u32 = 2;
        let nope: u32 = 16;
        let kv_lora_rank: u32 = 64;
        let v_head_dim: u32 = 16;
        let kv_b_per_head = nope + v_head_dim;
        let group_size: u32 = 64;

        let total_w_rows = (num_heads * kv_b_per_head) as usize;
        let in_cols = kv_lora_rank as usize;
        let mut weights = vec![0.0f32; total_w_rows * in_cols];
        for (i, w) in weights.iter_mut().enumerate() {
            // Reproducible deterministic-ish values, range ~[-2, 2].
            *w = ((i % 41) as f32 * 0.1) - 2.0;
        }
        let (packed, scales, biases) = pack_4bit_mlx(
            &weights,
            total_w_rows,
            in_cols,
            group_size as usize,
        );
        let dq = dequant_4bit(
            &packed,
            &scales,
            &biases,
            total_w_rows,
            in_cols,
            group_size as usize,
        );

        let mut q_nope =
            vec![0.0f32; (num_heads * nope) as usize];
        for (i, q) in q_nope.iter_mut().enumerate() {
            *q = ((i as f32) * 0.0137).sin();
        }

        // Host reference: q'[h, c] = Σ_i q_nope[h, i] * dq[h*kv_b_per_head + i, c].
        let mut q_prime_ref =
            vec![0.0f32; (num_heads * kv_lora_rank) as usize];
        for h in 0..num_heads as usize {
            for c in 0..kv_lora_rank as usize {
                let mut acc = 0.0f32;
                for i in 0..nope as usize {
                    let row = h * kv_b_per_head as usize + i;
                    acc += q_nope[h * nope as usize + i]
                        * dq[row * in_cols + c];
                }
                q_prime_ref[h * kv_lora_rank as usize + c] = acc;
            }
        }

        let pipe = MlaPipelines::new(&mut metal).unwrap();
        let device = metal.device().to_owned();
        let buf_w = shared_buf_with_data(&device, &packed);
        let buf_s = shared_buf_with_data(&device, &scales);
        let buf_b = shared_buf_with_data(&device, &biases);
        let buf_q = shared_buf_with_data(&device, &q_nope);
        let buf_qp = shared_buf_zeroed(
            &device,
            (num_heads * kv_lora_rank) as usize,
        );

        let cmdbuf = metal.queue().new_command_buffer();
        encode_mla_q_prime_4bit(
            cmdbuf,
            &pipe.q_prime,
            &buf_w,
            0,
            &buf_s,
            0,
            &buf_b,
            0,
            &buf_q,
            &buf_qp,
            num_heads,
            nope,
            kv_lora_rank,
            kv_b_per_head,
            group_size,
        );
        cmdbuf.commit();
        cmdbuf.wait_until_completed();

        let q_prime_gpu =
            read_back_f32(&buf_qp, q_prime_ref.len());
        let max_drift = q_prime_gpu
            .iter()
            .zip(&q_prime_ref)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_drift < 1e-3,
            "GPU/host drift {max_drift} on q_prime"
        );
    }

    /// Synthetic 3b check — small cache_len, two heads. Validates
    /// the SDPA inner loop by computing expected output via the
    /// reference math on the host.
    #[test]
    fn mla_sdpa_folded_matches_host() {
        let mut metal = match MetalBackend::new() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[gpu_mla] skipping: Metal init failed: {e:?}");
                return;
            }
        };

        let num_heads: u32 = 2;
        let kv_lora_rank: u32 = 64;
        let qk_rope_head_dim: u32 = 16;
        let cache_len: u32 = 8;
        let softmax_scale: f32 = 0.125;

        let mut q_prime =
            vec![0.0f32; (num_heads * kv_lora_rank) as usize];
        for (i, q) in q_prime.iter_mut().enumerate() {
            *q = ((i as f32) * 0.011).cos();
        }
        let mut q_pe =
            vec![0.0f32; (num_heads * qk_rope_head_dim) as usize];
        for (i, q) in q_pe.iter_mut().enumerate() {
            *q = ((i as f32) * 0.017).sin();
        }
        let mut latent =
            vec![0.0f32; (cache_len * kv_lora_rank) as usize];
        for (i, x) in latent.iter_mut().enumerate() {
            *x = ((i as f32) * 0.013).sin();
        }
        let mut rope_k =
            vec![0.0f32; (cache_len * qk_rope_head_dim) as usize];
        for (i, x) in rope_k.iter_mut().enumerate() {
            *x = ((i as f32) * 0.019).cos();
        }

        // Host reference per head:
        //   scores[t] = q'[h] · latent[t] + q_pe[h] · rope_k[t]
        //   scaled, softmaxed, then v_combine[h, c] = Σ_t scores[t] * latent[t, c]
        let mut v_combine_ref =
            vec![0.0f32; (num_heads * kv_lora_rank) as usize];
        for h in 0..num_heads as usize {
            let mut scores = vec![0.0f32; cache_len as usize];
            for t in 0..cache_len as usize {
                let mut s = 0.0f32;
                for c in 0..kv_lora_rank as usize {
                    s += q_prime[h * kv_lora_rank as usize + c]
                        * latent[t * kv_lora_rank as usize + c];
                }
                for r in 0..qk_rope_head_dim as usize {
                    s += q_pe[h * qk_rope_head_dim as usize + r]
                        * rope_k[t * qk_rope_head_dim as usize + r];
                }
                scores[t] = s * softmax_scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for c in 0..kv_lora_rank as usize {
                let mut acc = 0.0f32;
                for t in 0..cache_len as usize {
                    acc += scores[t]
                        * latent[t * kv_lora_rank as usize + c];
                }
                v_combine_ref[h * kv_lora_rank as usize + c] = acc;
            }
        }

        let pipe = MlaPipelines::new(&mut metal).unwrap();
        let device = metal.device().to_owned();
        let buf_qp = shared_buf_with_data(&device, &q_prime);
        let buf_qpe = shared_buf_with_data(&device, &q_pe);
        let buf_lat = shared_buf_with_data(&device, &latent);
        let buf_rk = shared_buf_with_data(&device, &rope_k);
        let buf_vc = shared_buf_zeroed(
            &device,
            (num_heads * kv_lora_rank) as usize,
        );

        let cmdbuf = metal.queue().new_command_buffer();
        encode_mla_sdpa_folded(
            cmdbuf,
            &pipe.sdpa,
            &buf_qp,
            &buf_qpe,
            &buf_lat,
            &buf_rk,
            &buf_vc,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            cache_len,
            softmax_scale,
        )
        .unwrap();
        cmdbuf.commit();
        cmdbuf.wait_until_completed();

        let v_combine_gpu =
            read_back_f32(&buf_vc, v_combine_ref.len());
        let max_drift = v_combine_gpu
            .iter()
            .zip(&v_combine_ref)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_drift < 1e-5,
            "GPU/host drift {max_drift} on v_combine"
        );
    }

    /// Synthetic 3c check — same machinery as 3a, different stride.
    #[test]
    fn mla_out_per_head_4bit_matches_host() {
        let mut metal = match MetalBackend::new() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[gpu_mla] skipping: Metal init failed: {e:?}");
                return;
            }
        };

        let num_heads: u32 = 2;
        let nope: u32 = 16;
        let kv_lora_rank: u32 = 64;
        let v_head_dim: u32 = 16;
        let kv_b_per_head = nope + v_head_dim;
        let group_size: u32 = 64;

        let total_w_rows = (num_heads * kv_b_per_head) as usize;
        let in_cols = kv_lora_rank as usize;
        let mut weights = vec![0.0f32; total_w_rows * in_cols];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = ((i % 41) as f32 * 0.1) - 2.0;
        }
        let (packed, scales, biases) = pack_4bit_mlx(
            &weights,
            total_w_rows,
            in_cols,
            group_size as usize,
        );
        let dq = dequant_4bit(
            &packed,
            &scales,
            &biases,
            total_w_rows,
            in_cols,
            group_size as usize,
        );

        let mut v_combine =
            vec![0.0f32; (num_heads * kv_lora_rank) as usize];
        for (i, x) in v_combine.iter_mut().enumerate() {
            *x = ((i as f32) * 0.0073).sin();
        }

        // Host reference: out[h, f] = Σ_c v_combine[h, c] *
        //                              dq[h*kv_b_per_head + nope + f, c]
        let mut out_ref =
            vec![0.0f32; (num_heads * v_head_dim) as usize];
        for h in 0..num_heads as usize {
            for f in 0..v_head_dim as usize {
                let row = h * kv_b_per_head as usize + nope as usize + f;
                let mut acc = 0.0f32;
                for c in 0..in_cols {
                    acc += v_combine[h * kv_lora_rank as usize + c]
                        * dq[row * in_cols + c];
                }
                out_ref[h * v_head_dim as usize + f] = acc;
            }
        }

        let pipe = MlaPipelines::new(&mut metal).unwrap();
        let device = metal.device().to_owned();
        let buf_w = shared_buf_with_data(&device, &packed);
        let buf_s = shared_buf_with_data(&device, &scales);
        let buf_b = shared_buf_with_data(&device, &biases);
        let buf_v = shared_buf_with_data(&device, &v_combine);
        let buf_o = shared_buf_zeroed(
            &device,
            (num_heads * v_head_dim) as usize,
        );

        let cmdbuf = metal.queue().new_command_buffer();
        encode_mla_out_per_head_4bit(
            cmdbuf,
            &pipe.out_per_head,
            &buf_w,
            0,
            &buf_s,
            0,
            &buf_b,
            0,
            &buf_v,
            &buf_o,
            num_heads,
            nope,
            kv_lora_rank,
            v_head_dim,
            kv_b_per_head,
            group_size,
        );
        cmdbuf.commit();
        cmdbuf.wait_until_completed();

        let out_gpu = read_back_f32(&buf_o, out_ref.len());
        let max_drift = out_gpu
            .iter()
            .zip(&out_ref)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_drift < 1e-3,
            "GPU/host drift {max_drift} on out_per_head"
        );
    }
}

/// Encode the Phase-4a fan-out scatter: split `q_full` into
/// `(q_nope, q_pe)` per head and `kv_pre` into `(kv_lat, k_pe)`. Pure
/// scatter — kills sync point #1 of the synchronous MLA forward by
/// promoting the host-side memcpys at `mla_attn_forward.rs:380-407`
/// into a Metal kernel.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_split_q_kv(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    q_full: &Buffer,
    kv_pre: &Buffer,
    q_nope: &Buffer,
    q_pe: &Buffer,
    kv_lat: &Buffer,
    k_pe: &Buffer,
    num_heads: u32,
    qk_nope: u32,
    qk_rope: u32,
    kv_lora_rank: u32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(q_full), 0);
    enc.set_buffer(1, Some(kv_pre), 0);
    enc.set_buffer(2, Some(q_nope), 0);
    enc.set_buffer(3, Some(q_pe), 0);
    enc.set_buffer(4, Some(kv_lat), 0);
    enc.set_buffer(5, Some(k_pe), 0);
    enc.set_bytes(6, 4, (&num_heads as *const u32).cast());
    enc.set_bytes(7, 4, (&qk_nope as *const u32).cast());
    enc.set_bytes(8, 4, (&qk_rope as *const u32).cast());
    enc.set_bytes(9, 4, (&kv_lora_rank as *const u32).cast());
    let q_nope_total = num_heads * qk_nope;
    let q_pe_total = num_heads * qk_rope;
    let max_out = q_nope_total
        .max(q_pe_total)
        .max(kv_lora_rank)
        .max(qk_rope);
    let num_tgs = max_out.div_ceil(256);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}

/// Encode the Phase-4a MLA cache append. Writes `kv_lat` →
/// `latent_cache[pos, :]` and `k_pe` → `rope_k_cache[pos, :]`. Kills
/// sync point #2 of the synchronous MLA forward (host-side memcpy at
/// `mla_attn_forward.rs:467-481`).
///
/// Caller is still responsible for `kv_cache.len = pos + 1` after the
/// cmdbuf commits — that's a Rust-side bookkeeping field the kernel
/// can't see.
#[allow(clippy::too_many_arguments)]
pub fn encode_mla_kv_cache_append(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    kv_lat: &Buffer,
    k_pe: &Buffer,
    latent_cache: &Buffer,
    rope_k_cache: &Buffer,
    kv_lora_rank: u32,
    qk_rope: u32,
    pos: i32,
) {
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(kv_lat), 0);
    enc.set_buffer(1, Some(k_pe), 0);
    enc.set_buffer(2, Some(latent_cache), 0);
    enc.set_buffer(3, Some(rope_k_cache), 0);
    enc.set_bytes(4, 4, (&kv_lora_rank as *const u32).cast());
    enc.set_bytes(5, 4, (&qk_rope as *const u32).cast());
    enc.set_bytes(6, 4, (&pos as *const i32).cast());
    let max_out = kv_lora_rank.max(qk_rope);
    let num_tgs = max_out.div_ceil(256);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs as NSUInteger, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
}
