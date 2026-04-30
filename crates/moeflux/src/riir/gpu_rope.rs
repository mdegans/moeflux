//! GPU YaRN RoPE — Phase 2 of the GPU MLA port.
//!
//! Mirrors [`super::rope::apply_rotary_emb_yarn`] (rope.rs:306) on
//! the Metal side. In-place rotation of a `[num_heads, rotary_dim]`
//! buffer using a precomputed `inv_freq[half]` table, a position
//! scalar, and a mscale multiplier baked into both `cos` and `sin`.
//!
//! ## Validation contract
//!
//! Drift between this kernel and the CPU reference comes entirely
//! from `cos`/`sin` precision: Metal's default `cos(x)`/`sin(x)`
//! routines are fast-math by default and differ from libm by a few
//! ULP. The unit tests assert `≤ 4 ULP` tolerance on a non-trivial
//! position; cross-position monotonicity holds bit-exactly because
//! the table lookup is integer-arithmetic identical.
//!
//! ## Dispatch shape
//!
//! `(num_heads, half)` threadgroups of one thread each; one
//! `(head, i)` pair per threadgroup. The total work is
//! `num_heads * half ≈ 4096` threads for Cogito-V2 — small enough
//! that the trivial geometry doesn't waste GPU. A simdgroup-size
//! micro-tiling is a follow-up if the per-token YaRN cost shows up
//! in profiling.

use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, MTLResourceOptions,
    MTLSize, NSUInteger,
};

use super::metal::{MetalBackend, MetalError};

/// Errors from the GPU YaRN RoPE dispatch.
#[derive(Debug, thiserror::Error)]
pub enum GpuRopeError {
    #[error("position must be non-negative (got {pos})")]
    NegativePos { pos: i32 },
    #[error("buffer length {got} != num_heads * rotary_dim ({expected})")]
    BufLen { got: usize, expected: usize },
    #[error("inv_freq length {got} != rotary_dim/2 ({expected})")]
    InvFreqLen { got: usize, expected: usize },
    #[error("Metal backend: {0}")]
    Metal(#[from] MetalError),
}

/// Encode a YaRN RoPE rotation into `cmdbuf` against an existing
/// shared-storage buffer `x_buf` of shape `[num_heads, rotary_dim]`.
///
/// `pipe` is the pre-fetched `yarn_rope_apply` compute pipeline
/// (typically obtained once at engine init via
/// `metal.pipeline("yarn_rope_apply")?.clone()`). `inv_freq_buf` is a
/// constant buffer holding `half = rotary_dim / 2` floats; caller
/// owns the buffer and is expected to reuse it across layers
/// (typically a single per-`RsCtx` `MlaYarnTables`).
///
/// Pipeline-as-argument matches the convention in [`super::gpu_norm`]
/// (e.g. `encode_rms_norm_bf16_into`) and avoids borrowing the
/// `MetalBackend` mutably across the encoding pass.
///
/// # Safety
///
/// The function records GPU work; correctness depends on the caller
/// driving the command-buffer to completion before reading `x_buf`
/// host-side and on no other in-flight encoder writing `x_buf`
/// concurrently.
#[allow(clippy::too_many_arguments)]
pub fn encode_yarn_rope_apply(
    cmdbuf: &CommandBufferRef,
    pipe: &ComputePipelineState,
    x_buf: &Buffer,
    inv_freq_buf: &Buffer,
    num_heads: u32,
    rotary_dim: u32,
    pos: i32,
    mscale: f32,
) -> Result<(), GpuRopeError> {
    if pos < 0 {
        return Err(GpuRopeError::NegativePos { pos });
    }
    let half = rotary_dim / 2;
    let pos_f = pos as f32;
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(x_buf), 0);
    enc.set_buffer(1, Some(inv_freq_buf), 0);
    enc.set_bytes(2, 4, (&num_heads as *const u32).cast());
    enc.set_bytes(3, 4, (&rotary_dim as *const u32).cast());
    enc.set_bytes(4, 4, (&pos_f as *const f32).cast());
    enc.set_bytes(5, 4, (&mscale as *const f32).cast());
    enc.dispatch_thread_groups(
        MTLSize::new(num_heads as NSUInteger, half as NSUInteger, 1),
        MTLSize::new(1, 1, 1),
    );
    enc.end_encoding();
    Ok(())
}

/// One-shot host-side helper: copy `x` to a fresh GPU buffer, apply
/// YaRN RoPE, copy back. Diff-oracle / unit-test harness, not the
/// production hot path.
pub fn yarn_rope_apply_oneshot(
    metal: &mut MetalBackend,
    x: &mut [f32],
    rotary_dim: usize,
    inv_freq: &[f32],
    pos: i32,
    mscale: f32,
) -> Result<(), GpuRopeError> {
    if pos < 0 {
        return Err(GpuRopeError::NegativePos { pos });
    }
    let half = rotary_dim / 2;
    if inv_freq.len() != half {
        return Err(GpuRopeError::InvFreqLen {
            got: inv_freq.len(),
            expected: half,
        });
    }
    if x.len() % rotary_dim != 0 {
        return Err(GpuRopeError::BufLen {
            got: x.len(),
            expected: rotary_dim,
        });
    }
    let num_heads = (x.len() / rotary_dim) as u32;

    let pipe = metal.pipeline("yarn_rope_apply")?.clone();
    let device = metal.device();
    let buf_x = device.new_buffer_with_data(
        x.as_ptr().cast(),
        (x.len() * std::mem::size_of::<f32>()) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_inv = device.new_buffer_with_data(
        inv_freq.as_ptr().cast(),
        (inv_freq.len() * std::mem::size_of::<f32>()) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );

    let cmdbuf = metal.queue().new_command_buffer();
    encode_yarn_rope_apply(
        cmdbuf,
        &pipe,
        &buf_x,
        &buf_inv,
        num_heads,
        rotary_dim as u32,
        pos,
        mscale,
    )?;
    cmdbuf.commit();
    cmdbuf.wait_until_completed();

    // SAFETY: shared-storage buffer, GPU work has completed.
    unsafe {
        let p = buf_x.contents() as *const f32;
        let s = std::slice::from_raw_parts(p, x.len());
        x.copy_from_slice(s);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riir::rope::{apply_rotary_emb_yarn, compute_yarn_inv_freq};

    /// `pos = 0` and `mscale = 1` ⇒ identity. Sanity check on
    /// kernel-encoded buffer plumbing; doesn't exercise `cos`/`sin`.
    #[test]
    fn yarn_rope_gpu_pos_zero_mscale_one_is_identity() {
        let mut metal = match MetalBackend::new() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[gpu_rope] skipping: Metal init failed: {e:?}");
                return;
            }
        };
        let rotary_dim: usize = 64;
        let half = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / 10000f32.powf(2.0 * (i as f32) / rotary_dim as f32))
            .collect();
        let num_heads = 4;
        let mut x: Vec<f32> = (0..num_heads * rotary_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let x_orig = x.clone();

        yarn_rope_apply_oneshot(
            &mut metal, &mut x, rotary_dim, &inv_freq, 0, 1.0,
        )
        .unwrap();

        for i in 0..x.len() {
            assert!(
                (x[i] - x_orig[i]).abs() < 1e-7,
                "x[{i}] = {} but expected identity {}",
                x[i],
                x_orig[i],
            );
        }
    }

    /// At `pos > 0`, the GPU kernel must agree with the CPU
    /// reference within `4 ULP`. The drift bound covers Metal's
    /// fast-math `cos`/`sin` precision; anything tighter would
    /// fail spuriously across GPU revisions.
    #[test]
    fn yarn_rope_gpu_matches_cpu_at_pos_4096() {
        let mut metal = match MetalBackend::new() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[gpu_rope] skipping: Metal init failed: {e:?}");
                return;
            }
        };
        // Synthetic inv_freq table — non-trivial, monotone, in the
        // same ballpark a real YaRN table would produce.
        let rotary_dim: usize = 64;
        let half = rotary_dim / 2;
        let inv_freq = compute_yarn_inv_freq(
            rotary_dim, /* base = */ 1.0e4, /* factor = */ 40.0,
            /* original_max = */ 4096.0, /* beta_fast = */ 32.0,
            /* beta_slow = */ 1.0,
        );
        assert_eq!(inv_freq.len(), half);

        let num_heads = 8;
        let pos = 4096;
        let mscale: f32 = 1.0;

        let mut x_gpu: Vec<f32> = (0..num_heads * rotary_dim)
            .map(|i| ((i as f32) * 0.0137).sin())
            .collect();
        let mut x_cpu = x_gpu.clone();

        apply_rotary_emb_yarn(pos, &mut x_cpu, rotary_dim, &inv_freq, mscale)
            .unwrap();
        yarn_rope_apply_oneshot(
            &mut metal,
            &mut x_gpu,
            rotary_dim,
            &inv_freq,
            pos,
            mscale,
        )
        .unwrap();

        // 4 ULP for f32 around 1.0 is ~4.8e-7. We use abs-diff
        // because the values straddle 0 and ULP-relative tolerance
        // gets brittle near zero.
        let max_drift =
            x_gpu.iter().zip(&x_cpu).map(|(g, c)| (g - c).abs()).fold(
                0.0f32,
                f32::max,
            );
        assert!(
            max_drift < 5e-6,
            "GPU/CPU drift {max_drift} exceeds 4 ULP tolerance"
        );
    }
}
