//! Scaled dot-product attention — Rust port of `infer.m`'s
//! `cpu_sdpa` helper (extracted from `full_attention_forward`).
//!
//! Per-head, GQA: each group of `num_attn_heads / num_kv_heads` query
//! heads shares one KV head. Single query position against a
//! `kv_len`-position cache.
//!
//! ```text
//! scores[h, p] = dot(Q[h], K[p, h/heads_per_kv]) / sqrt(head_dim)
//! weights[h]   = softmax(scores[h])
//! out[h]       = sum_p weights[h, p] * V[p, h/heads_per_kv]
//! out         *= sigmoid(q_gate)
//! ```
//!
//! ## Tolerance contract — ULP-bounded, not bit-exact
//!
//! Two libm-precision calls per element: `expf` inside the softmax
//! (one per kv position per head) and `expf` again in the sigmoid
//! gate. Both sit in the same compiler-choice territory as the
//! `cosf` / `sinf` / `powf` calls in RoPE: clang at `-O3` is free to
//! auto-vectorize through Apple libm vector variants while Rust
//! `extern "C"` calls don't. On top of that, multiple sequential
//! reductions accumulate per element (Q·K dot, softmax sum, weighted
//! sum of V), each of which can be vectorized differently.
//!
//! The tolerance budget is wider than RoPE's: drift compounds across
//! kv_len and head_dim. Per-element ULP drift is typically a few
//! hundred ULPs at small kv_len, growing modestly with longer
//! contexts. We don't assert a tight bound at the kernel level —
//! the attention output enters a 4-bit-quantized `o_proj` matvec
//! immediately after, which floors precision well before any of this
//! drift would matter for downstream behavior.

use crate::riir::variants::VARIANT;

unsafe extern "C" {
    fn expf(x: f32) -> f32;
}

/// Errors specific to the SDPA port.
#[derive(Debug, thiserror::Error)]
pub enum SdpaError {
    #[error("kv_len must be > 0 (got {kv_len})")]
    EmptyCache { kv_len: i32 },
    #[error("Q buffer length {got} != num_attn_heads * head_dim ({expected})")]
    QLen { got: usize, expected: usize },
    #[error("q_gate buffer length {got} != num_attn_heads * head_dim ({expected})")]
    QGateLen { got: usize, expected: usize },
    #[error("K cache length {got} != kv_len * num_kv_heads * head_dim ({expected})")]
    KCacheLen { got: usize, expected: usize },
    #[error("V cache length {got} != kv_len * num_kv_heads * head_dim ({expected})")]
    VCacheLen { got: usize, expected: usize },
    #[error("output buffer length {got} != num_attn_heads * head_dim ({expected})")]
    OutLen { got: usize, expected: usize },
}

/// Compute scaled dot-product attention with sigmoid-gated output for
/// one query position against a `kv_len`-position cache. Mirrors
/// `cpu_sdpa` in `infer.m`. Shape comes from the active [`VARIANT`].
///
/// `out` is overwritten (not accumulated). `q`, `q_gate`, `k_cache`,
/// `v_cache` are read-only. None of the buffers may alias `out`.
pub fn sdpa_cpu(
    kv_len: i32,
    q: &[f32],
    q_gate: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
) -> Result<(), SdpaError> {
    if kv_len <= 0 {
        return Err(SdpaError::EmptyCache { kv_len });
    }
    let kv_len_u = kv_len as usize;
    let num_attn_heads = VARIANT.num_attn_heads;
    let num_kv_heads = VARIANT.num_kv_heads;
    let head_dim = VARIANT.head_dim;

    let q_dim = num_attn_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let kv_total = kv_len_u * kv_dim;

    if q.len() != q_dim {
        return Err(SdpaError::QLen {
            got: q.len(),
            expected: q_dim,
        });
    }
    if q_gate.len() != q_dim {
        return Err(SdpaError::QGateLen {
            got: q_gate.len(),
            expected: q_dim,
        });
    }
    if k_cache.len() != kv_total {
        return Err(SdpaError::KCacheLen {
            got: k_cache.len(),
            expected: kv_total,
        });
    }
    if v_cache.len() != kv_total {
        return Err(SdpaError::VCacheLen {
            got: v_cache.len(),
            expected: kv_total,
        });
    }
    if out.len() != q_dim {
        return Err(SdpaError::OutLen {
            got: out.len(),
            expected: q_dim,
        });
    }

    let heads_per_kv = num_attn_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for o in out.iter_mut() {
        *o = 0.0;
    }

    let mut scores = vec![0.0f32; kv_len_u];
    for h in 0..num_attn_heads {
        let kv_h = h / heads_per_kv;
        let qh = &q[h * head_dim..(h + 1) * head_dim];

        for p in 0..kv_len_u {
            let kp_start = p * kv_dim + kv_h * head_dim;
            let kp = &k_cache[kp_start..kp_start + head_dim];
            let mut dot: f32 = 0.0;
            for d in 0..head_dim {
                dot += qh[d] * kp[d];
            }
            scores[p] = dot * scale;
        }

        // softmax over scores[..kv_len_u]
        let mut max_val = scores[0];
        for &s in scores[1..kv_len_u].iter() {
            if s > max_val {
                max_val = s;
            }
        }
        let mut sum: f32 = 0.0;
        for s in scores[..kv_len_u].iter_mut() {
            // SAFETY: scalar libm `expf` — well-formed f32 input.
            *s = unsafe { expf(*s - max_val) };
            sum += *s;
        }
        let inv_sum = 1.0f32 / sum;
        for s in scores[..kv_len_u].iter_mut() {
            *s *= inv_sum;
        }

        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        for p in 0..kv_len_u {
            let vp_start = p * kv_dim + kv_h * head_dim;
            let vp = &v_cache[vp_start..vp_start + head_dim];
            let w = scores[p];
            for d in 0..head_dim {
                oh[d] += w * vp[d];
            }
        }
    }

    for i in 0..q_dim {
        // SAFETY: scalar libm `expf` — well-formed f32 input.
        let g = 1.0f32 / (1.0f32 + unsafe { expf(-q_gate[i]) });
        out[i] *= g;
    }

    Ok(())
}
