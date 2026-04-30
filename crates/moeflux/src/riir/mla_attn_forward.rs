//! GPU MLA per-layer attention forward — Phase 4 of the GPU MLA port.
//!
//! [`mla_attn_layer_forward_gpu`] mirrors
//! [`super::mla_attn_cpu::mla_attn_layer_forward_cpu`] step-for-step
//! but runs the heavy compute on Metal:
//!
//! 1. Q chain: `q_a_proj` → `q_a_layernorm` → `q_b_proj` (3× matvec
//!    + per-head rmsnorm).
//! 2. KV chain: `kv_a_proj_with_mqa` → `kv_a_layernorm` (matvec +
//!    rmsnorm on the latent half).
//! 3. YaRN RoPE on `q_pe` and `k_pe` halves.
//! 4. Append `(kv_lat, k_pe)` to the per-layer
//!    [`super::state::MlaKvCacheGpu`].
//! 5. Folded SDPA via the Phase 3 kernels (`q_prime`, `mla_sdpa_folded`,
//!    `mla_out_per_head_4bit`).
//! 6. `o_proj` final matvec.
//!
//! The output is the post-`o_proj` hidden state in `out_buf` (shared
//! storage; caller reads host-side or chains the buffer into the next
//! kernel). On entry the buffer set already has
//! `pre_norm_buf` populated with the rms-normed input — the caller
//! owns the pre-attn norm so the same machinery can drive both
//! GPU-MLA and the CPU diff path without two norm calls.
//!
//! ## Hybrid mode (this slice)
//!
//! The MoE / dense MLP block stays CPU-side for first run — see
//! [`super::mod::step_internal_mla_gpu`]. That bounces post-MLA hidden
//! states through host every layer, but the bounce is cheap (`hidden_dim
//! = 7168` floats) compared to the projection / SDPA cost we just moved
//! to GPU. Full-GPU MoE integration with the deferred ring is a
//! follow-up perf slice.

use metal::{
    Buffer, Device, MTLResourceOptions, MTLSize, NSUInteger,
};

use super::gpu_matvec::{
    encode_matvec, MatvecPipelines, MatvecSpec,
};
use super::gpu_mla::{
    encode_mla_out_per_head_4bit, encode_mla_q_prime_4bit,
    encode_mla_sdpa_folded, GpuMlaError, MlaPipelines,
};
use super::gpu_norm::{
    encode_rms_norm_bf16_into, RmsNormBf16Pipelines,
};
use super::gpu_rope::encode_yarn_rope_apply;
use super::metal::{MetalBackend, MetalError};
use super::mtl_weight_buf::MtlWeightBuf;
use super::state::MlaKvCacheGpu;
use super::variants::{RMS_NORM_EPS, VARIANT};
use super::weight_file::WeightFile;

/// Per-token GPU scratch for the MLA forward. One set is reused across
/// every layer — attention is sequential per token, so layer N+1 doesn't
/// need its own copy. Total ~3 MB on Cogito-V2 (the per-head q_prime
/// and v_combine dominate).
pub struct MlaForwardBuffers {
    /// Q chain.
    pub q_lat: Buffer, // [q_lora_rank]
    pub q_full: Buffer, // [num_heads, qk_head_dim]
    pub q_nope: Buffer, // [num_heads, qk_nope_head_dim]   (packed view for mla_q_prime)
    pub q_pe: Buffer,   // [num_heads, qk_rope_head_dim]
    /// KV chain.
    pub kv_pre: Buffer, // [kv_lora_rank + qk_rope_head_dim]
    pub kv_lat: Buffer, // [kv_lora_rank]   (post-norm, also written to MlaKvCacheGpu.latent)
    pub k_pe: Buffer, // [qk_rope_head_dim] (post-RoPE, also written to MlaKvCacheGpu.rope_k)
    /// Folded MLA scratch.
    pub q_prime: Buffer, // [num_heads, kv_lora_rank]
    pub v_combine: Buffer, // [num_heads, kv_lora_rank]
    pub out_per_head: Buffer, // [num_heads, v_head_dim]
    /// Pre/post forward I/O.
    pub pre_norm: Buffer, // [hidden_dim]   (caller writes the rms-normed hidden here)
    pub out: Buffer,    // [hidden_dim]   (post-o_proj output)
    /// Sum-sq scratch for the q_a_layernorm / kv_a_layernorm rms norms.
    pub q_a_sum_sq: Buffer, // [1]
    pub kv_lat_sum_sq: Buffer, // [1]
}

impl MlaForwardBuffers {
    pub fn new(device: &Device) -> Self {
        let v = VARIANT;
        let f32_buf = |n: usize| {
            let b = device.new_buffer(
                (n * std::mem::size_of::<f32>()) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            // SAFETY: shared storage, no GPU work in flight on a
            // freshly allocated buffer.
            unsafe {
                std::ptr::write_bytes(
                    b.contents() as *mut u8,
                    0,
                    n * std::mem::size_of::<f32>(),
                );
            }
            b
        };
        let qk_head_dim = v.qk_nope_head_dim + v.qk_rope_head_dim;
        Self {
            q_lat: f32_buf(v.q_lora_rank),
            q_full: f32_buf(v.num_attn_heads * qk_head_dim),
            q_nope: f32_buf(v.num_attn_heads * v.qk_nope_head_dim),
            q_pe: f32_buf(v.num_attn_heads * v.qk_rope_head_dim),
            kv_pre: f32_buf(v.kv_lora_rank + v.qk_rope_head_dim),
            kv_lat: f32_buf(v.kv_lora_rank),
            k_pe: f32_buf(v.qk_rope_head_dim),
            q_prime: f32_buf(v.num_attn_heads * v.kv_lora_rank),
            v_combine: f32_buf(v.num_attn_heads * v.kv_lora_rank),
            out_per_head: f32_buf(v.num_attn_heads * v.v_head_dim),
            pre_norm: f32_buf(v.hidden_dim),
            out: f32_buf(v.hidden_dim),
            q_a_sum_sq: f32_buf(1),
            kv_lat_sum_sq: f32_buf(1),
        }
    }
}

/// Lazily-built per-`RsCtx` YaRN tables for MLA. `inv_freq` lives in
/// shared-storage Metal memory so the GPU YaRN kernel reads it as a
/// constant buffer; `mscale` is captured at build time and passed as
/// a scalar `set_bytes` argument.
pub struct MlaYarnTables {
    pub inv_freq: Buffer,
    pub mscale: f32,
}

impl MlaYarnTables {
    pub fn new(device: &Device) -> Self {
        use super::rope::{compute_yarn_inv_freq, yarn_get_mscale_full};
        use super::variants::ROPE_THETA;
        let v = VARIANT;
        let inv_freq = compute_yarn_inv_freq(
            v.qk_rope_head_dim,
            ROPE_THETA,
            v.yarn_factor,
            v.yarn_original_max_pos as f32,
            v.yarn_beta_fast,
            v.yarn_beta_slow,
        );
        let mscale = yarn_get_mscale_full(
            v.yarn_factor,
            v.yarn_mscale,
            v.yarn_mscale_all_dim,
        );
        let buf = device.new_buffer_with_data(
            inv_freq.as_ptr().cast(),
            (inv_freq.len() * std::mem::size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            inv_freq: buf,
            mscale,
        }
    }
}

/// Pipelines pre-fetched for the MLA forward. Ownership / lifetime
/// matches `MoeBuffers` etc. — built once at engine init.
pub struct MlaForwardPipelines {
    pub mla: MlaPipelines,
    pub matvec: MatvecPipelines,
    pub norms: RmsNormBf16Pipelines,
    pub yarn_rope: metal::ComputePipelineState,
}

impl MlaForwardPipelines {
    pub fn new(metal: &mut MetalBackend) -> Result<Self, MetalError> {
        Ok(Self {
            mla: MlaPipelines::new(metal)?,
            matvec: MatvecPipelines::fetch(metal)?,
            norms: RmsNormBf16Pipelines::fetch(metal)?,
            yarn_rope: metal.pipeline("yarn_rope_apply")?.clone(),
        })
    }
}

/// Errors produced by [`mla_attn_layer_forward_gpu`].
#[derive(Debug, thiserror::Error)]
pub enum MlaForwardGpuError {
    #[error("MLA only valid on MLA variants (this build's attn_kind is {kind:?})")]
    NotMlaVariant { kind: super::variants::AttnKind },
    #[error("kv_cache.len {len} would exceed MAX_SEQ_LEN={max} after append")]
    CacheFull { len: i32, max: usize },
    #[error("pos {pos} != kv_cache.len {cache_len} (single-step decode)")]
    PosMismatch { pos: i32, cache_len: i32 },
    #[error("kv_cache buffers not allocated (call ensure_buffers first)")]
    CacheNotReady,
    #[error("Metal weight tensor: {name}")]
    MissingTensor { name: String },
    #[error("Metal: {0}")]
    Metal(#[from] MetalError),
    #[error("MLA dispatch: {0}")]
    Mla(#[from] GpuMlaError),
}

/// Per-layer GPU MLA forward. Synchronous: encodes all dispatches into
/// `cmdbuf`, the caller commits + waits.
///
/// Reads `pre_norm` (shared storage, `hidden_dim` floats) — the
/// rms-normed input. Writes `out` (shared storage, `hidden_dim`
/// floats) — the post-`o_proj` MLA contribution to the residual
/// stream. Caller does the residual add.
///
/// `kv_cache.len` is bumped by 1 on success.
#[allow(clippy::too_many_arguments)]
pub fn mla_attn_layer_forward_gpu(
    metal: &mut MetalBackend,
    pipes: &MlaForwardPipelines,
    wf: &WeightFile,
    wf_buf: &MtlWeightBuf,
    yarn: &MlaYarnTables,
    bufs: &mut MlaForwardBuffers,
    kv_cache: &mut MlaKvCacheGpu,
    layer_idx: usize,
    pos: i32,
) -> Result<(), MlaForwardGpuError> {
    use super::variants::AttnKind;

    if VARIANT.attn_kind != AttnKind::Mla {
        return Err(MlaForwardGpuError::NotMlaVariant {
            kind: VARIANT.attn_kind,
        });
    }
    if pos != kv_cache.len {
        return Err(MlaForwardGpuError::PosMismatch {
            pos,
            cache_len: kv_cache.len,
        });
    }
    if (kv_cache.len as usize) >= super::variants::MAX_SEQ_LEN {
        return Err(MlaForwardGpuError::CacheFull {
            len: kv_cache.len,
            max: super::variants::MAX_SEQ_LEN,
        });
    }
    let latent_buf =
        kv_cache.latent_cache.as_ref().ok_or(MlaForwardGpuError::CacheNotReady)?;
    let rope_k_buf =
        kv_cache.rope_k_cache.as_ref().ok_or(MlaForwardGpuError::CacheNotReady)?;

    let v = VARIANT;
    let hidden_dim = v.hidden_dim as u32;
    let q_lora_rank = v.q_lora_rank as u32;
    let kv_lora_rank = v.kv_lora_rank as u32;
    let nope = v.qk_nope_head_dim as u32;
    let rope_dim = v.qk_rope_head_dim as u32;
    let qk_head_dim = nope + rope_dim;
    let v_head_dim = v.v_head_dim as u32;
    let num_heads = v.num_attn_heads as u32;
    let kv_b_per_head = nope + v_head_dim;

    // Helpers: resolve a 4-bit projection's `(weight, scales, biases)`
    // byte offsets in the shared weight buffer.
    let resolve_proj = |name: &str| -> Result<(u64, u64, u64), MlaForwardGpuError> {
        let w = format!("{name}.weight");
        let s = format!("{name}.scales");
        let b = format!("{name}.biases");
        let w_off = wf_buf
            .tensor_offset(wf, &w)
            .map_err(|_| MlaForwardGpuError::MissingTensor { name: w.clone() })?
            .ok_or(MlaForwardGpuError::MissingTensor { name: w })?;
        let s_off = wf_buf
            .tensor_offset(wf, &s)
            .map_err(|_| MlaForwardGpuError::MissingTensor { name: s.clone() })?
            .ok_or(MlaForwardGpuError::MissingTensor { name: s })?;
        let b_off = wf_buf
            .tensor_offset(wf, &b)
            .map_err(|_| MlaForwardGpuError::MissingTensor { name: b.clone() })?
            .ok_or(MlaForwardGpuError::MissingTensor { name: b })?;
        Ok((w_off, s_off, b_off))
    };
    let resolve_norm = |name: &str| -> Result<u64, MlaForwardGpuError> {
        let n = format!("{name}.weight");
        wf_buf
            .tensor_offset(wf, &n)
            .map_err(|_| MlaForwardGpuError::MissingTensor { name: n.clone() })?
            .ok_or(MlaForwardGpuError::MissingTensor { name: n })
    };

    let layer_prefix = format!("model.layers.{layer_idx}.self_attn");
    let q_a_off = resolve_proj(&format!("{layer_prefix}.q_a_proj"))?;
    let q_a_norm_off = resolve_norm(&format!("{layer_prefix}.q_a_layernorm"))?;
    let q_b_off = resolve_proj(&format!("{layer_prefix}.q_b_proj"))?;
    let kv_a_off = resolve_proj(&format!("{layer_prefix}.kv_a_proj_with_mqa"))?;
    let kv_a_norm_off = resolve_norm(&format!("{layer_prefix}.kv_a_layernorm"))?;
    let kv_b_off = resolve_proj(&format!("{layer_prefix}.kv_b_proj"))?;
    let o_off = resolve_proj(&format!("{layer_prefix}.o_proj"))?;

    // Pre-fetch pipelines as locals so the encode passes don't need
    // to borrow `metal` once we start the cmdbuf.
    let pipe_qprime = pipes.mla.q_prime.clone();
    let pipe_sdpa = pipes.mla.sdpa.clone();
    let pipe_outhead = pipes.mla.out_per_head.clone();
    let pipe_yarn = pipes.yarn_rope.clone();

    let queue = metal.queue();
    let cmdbuf = queue.new_command_buffer();

    // ---- Q chain ----
    // q_lat = q_a_proj @ pre_norm
    encode_matvec(
        cmdbuf,
        &pipes.matvec,
        wf_buf,
        &MatvecSpec {
            w_off: q_a_off.0,
            s_off: q_a_off.1,
            b_off: q_a_off.2,
            input: &bufs.pre_norm,
            output: &bufs.q_lat,
            out_dim: q_lora_rank,
            in_dim: hidden_dim,
            bits: 4,
        },
    );
    // q_lat = rms_norm(q_lat)  in place via sum_sq scratch
    encode_rms_norm_bf16_into(
        cmdbuf,
        &pipes.norms,
        &bufs.q_lat,
        wf_buf.buffer(),
        q_a_norm_off,
        &bufs.q_a_sum_sq,
        &bufs.q_lat,
        q_lora_rank,
        RMS_NORM_EPS,
    );
    // q_full = q_b_proj @ q_lat
    encode_matvec(
        cmdbuf,
        &pipes.matvec,
        wf_buf,
        &MatvecSpec {
            w_off: q_b_off.0,
            s_off: q_b_off.1,
            b_off: q_b_off.2,
            input: &bufs.q_lat,
            output: &bufs.q_full,
            out_dim: num_heads * qk_head_dim,
            in_dim: q_lora_rank,
            bits: 4,
        },
    );

    // ---- KV chain ----
    // kv_pre = kv_a_proj_with_mqa @ pre_norm
    encode_matvec(
        cmdbuf,
        &pipes.matvec,
        wf_buf,
        &MatvecSpec {
            w_off: kv_a_off.0,
            s_off: kv_a_off.1,
            b_off: kv_a_off.2,
            input: &bufs.pre_norm,
            output: &bufs.kv_pre,
            out_dim: kv_lora_rank + rope_dim,
            in_dim: hidden_dim,
            bits: 4,
        },
    );
    // kv_pre is laid out as [kv_lat | k_pe]. Copy halves out
    // in-flight so the rms_norm + RoPE kernels can target separate
    // buffers without a host bounce.
    {
        // kv_lat = kv_pre[..kv_lora_rank]; kv_lat = rms_norm(kv_lat)
        // Two encodings: a copy (use buffer-to-buffer blit) and a
        // norm. Simpler: write kv_pre's first half into kv_lat via a
        // small kernel. Cheaper: stage kv_lat / k_pe via host since
        // kv_pre is shared-storage. We use the host bounce — `kv_pre`
        // is shared so `contents()` is valid; but we have to wait
        // for the matvec to complete first. To keep this synchronous-
        // simple, we commit the chain so far, wait, then resume.
    }
    // Commit + wait so we can host-split kv_pre into kv_lat / k_pe
    // on shared-storage without racing the matvec.
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    // SAFETY: shared-storage buffers; matvec dispatch has completed.
    unsafe {
        let kv_pre_p = bufs.kv_pre.contents() as *const f32;
        let kv_lat_p = bufs.kv_lat.contents() as *mut f32;
        let k_pe_p = bufs.k_pe.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(
            kv_pre_p,
            kv_lat_p,
            v.kv_lora_rank,
        );
        std::ptr::copy_nonoverlapping(
            kv_pre_p.add(v.kv_lora_rank),
            k_pe_p,
            v.qk_rope_head_dim,
        );
        // Also extract q_pe halves out of q_full (per-head [nope|pe]).
        let q_full_p = bufs.q_full.contents() as *const f32;
        let q_pe_p = bufs.q_pe.contents() as *mut f32;
        for h in 0..v.num_attn_heads {
            std::ptr::copy_nonoverlapping(
                q_full_p.add(h * (v.qk_nope_head_dim + v.qk_rope_head_dim) + v.qk_nope_head_dim),
                q_pe_p.add(h * v.qk_rope_head_dim),
                v.qk_rope_head_dim,
            );
        }
    }

    // Resume: norms + RoPE + cache append + folded SDPA + o_proj.
    let cmdbuf = queue.new_command_buffer();

    // kv_lat = rms_norm(kv_lat)
    encode_rms_norm_bf16_into(
        cmdbuf,
        &pipes.norms,
        &bufs.kv_lat,
        wf_buf.buffer(),
        kv_a_norm_off,
        &bufs.kv_lat_sum_sq,
        &bufs.kv_lat,
        kv_lora_rank,
        RMS_NORM_EPS,
    );

    // ---- YaRN RoPE on q_pe and k_pe ----
    encode_yarn_rope_apply(
        cmdbuf,
        &pipe_yarn,
        &bufs.q_pe,
        &yarn.inv_freq,
        num_heads,
        rope_dim,
        pos,
        yarn.mscale,
    )
    .map_err(|_| MlaForwardGpuError::Metal(MetalError::NoDevice))?;
    encode_yarn_rope_apply(
        cmdbuf,
        &pipe_yarn,
        &bufs.k_pe,
        &yarn.inv_freq,
        1, // shared k_pe, broadcast-style
        rope_dim,
        pos,
        yarn.mscale,
    )
    .map_err(|_| MlaForwardGpuError::Metal(MetalError::NoDevice))?;

    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    // Write rotated q_pe back into q_full's pe slots, and append
    // (kv_lat, k_pe) to the MLA cache. Both are host-side memcpys
    // since we're already past a sync point.
    let new_idx = pos as usize;
    // SAFETY: shared-storage buffers; cmdbuf completed above.
    unsafe {
        let q_pe_p = bufs.q_pe.contents() as *const f32;
        let q_full_p = bufs.q_full.contents() as *mut f32;
        for h in 0..v.num_attn_heads {
            std::ptr::copy_nonoverlapping(
                q_pe_p.add(h * v.qk_rope_head_dim),
                q_full_p.add(h * qk_head_dim as usize + v.qk_nope_head_dim),
                v.qk_rope_head_dim,
            );
        }
        // Append kv_lat + k_pe to kv_cache row[new_idx].
        let lat_dst_p = (latent_buf.contents() as *mut f32)
            .add(new_idx * v.kv_lora_rank);
        std::ptr::copy_nonoverlapping(
            bufs.kv_lat.contents() as *const f32,
            lat_dst_p,
            v.kv_lora_rank,
        );
        let rk_dst_p = (rope_k_buf.contents() as *mut f32)
            .add(new_idx * v.qk_rope_head_dim);
        std::ptr::copy_nonoverlapping(
            bufs.k_pe.contents() as *const f32,
            rk_dst_p,
            v.qk_rope_head_dim,
        );
    }
    kv_cache.len = pos + 1;
    let cache_len = kv_cache.len as u32;

    // ---- Folded SDPA chain ----
    let cmdbuf = queue.new_command_buffer();

    // q_prime = q_nope @ kv_b_proj_K_per_head
    // Pack q_nope from q_full's per-head [nope|pe] layout into a
    // contiguous [num_heads, nope] buffer that the kernel can index
    // as `q_nope[h * nope + i]`.
    // SAFETY: previous cmdbuf is committed; both buffers are shared-
    // storage with no GPU work in flight.
    unsafe {
        let q_full_p = bufs.q_full.contents() as *const f32;
        let q_nope_p = bufs.q_nope.contents() as *mut f32;
        for h in 0..v.num_attn_heads {
            std::ptr::copy_nonoverlapping(
                q_full_p.add(h * qk_head_dim as usize),
                q_nope_p.add(h * v.qk_nope_head_dim),
                v.qk_nope_head_dim,
            );
        }
    }
    encode_mla_q_prime_4bit(
        cmdbuf,
        &pipe_qprime,
        wf_buf.buffer(),
        kv_b_off.0,
        wf_buf.buffer(),
        kv_b_off.1,
        wf_buf.buffer(),
        kv_b_off.2,
        &bufs.q_nope,
        &bufs.q_prime,
        num_heads,
        nope,
        kv_lora_rank,
        kv_b_per_head,
        64, // group_size
    );

    // Folded SDPA inner.
    let softmax_scale =
        (1.0 / (qk_head_dim as f32).sqrt()) * yarn.mscale * yarn.mscale;
    encode_mla_sdpa_folded(
        cmdbuf,
        &pipe_sdpa,
        &bufs.q_prime,
        &bufs.q_pe,
        latent_buf,
        rope_k_buf,
        &bufs.v_combine,
        num_heads,
        kv_lora_rank,
        rope_dim,
        cache_len,
        softmax_scale,
    )?;

    // out_per_head = v_combine @ kv_b_proj_V_per_head
    encode_mla_out_per_head_4bit(
        cmdbuf,
        &pipe_outhead,
        wf_buf.buffer(),
        kv_b_off.0,
        wf_buf.buffer(),
        kv_b_off.1,
        wf_buf.buffer(),
        kv_b_off.2,
        &bufs.v_combine,
        &bufs.out_per_head,
        num_heads,
        nope,
        kv_lora_rank,
        v_head_dim,
        kv_b_per_head,
        64,
    );

    // o_proj final matvec: out = o_proj @ out_per_head_flat
    encode_matvec(
        cmdbuf,
        &pipes.matvec,
        wf_buf,
        &MatvecSpec {
            w_off: o_off.0,
            s_off: o_off.1,
            b_off: o_off.2,
            input: &bufs.out_per_head,
            output: &bufs.out,
            out_dim: hidden_dim,
            in_dim: num_heads * v_head_dim,
            bits: 4,
        },
    );

    cmdbuf.commit();
    cmdbuf.wait_until_completed();
    Ok(())
}

/// Small helper: dispatch a single threadgroup of `size` threads — the
/// 1-thread-1-output pattern most of our oneshot encodes use. Exposed
/// for symmetry with the existing dispatchers.
#[allow(dead_code)]
fn dispatch_1d(
    enc: &metal::ComputeCommandEncoderRef,
    threadgroups: u64,
    threads: u64,
) {
    enc.dispatch_thread_groups(
        MTLSize::new(threadgroups, 1, 1),
        MTLSize::new(threads, 1, 1),
    );
}
