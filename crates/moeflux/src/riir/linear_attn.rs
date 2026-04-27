//! Linear-attention (GatedDeltaNet) primitives — Rust ports of the
//! CPU helpers from `infer.m`'s `linear_attention_forward`.
//!
//! Slice 8a covers the three single-purpose primitives:
//!
//! - [`conv1d_step`] — depthwise 1D conv with state shift + SiLU
//! - [`rms_norm_bare`] — RMSNorm with no weight (just normalize)
//! - [`rms_norm_gated`] — RMSNorm × SiLU(z) × weight
//!
//! Slice 8b adds the recurrence proper:
//!
//! - [`gated_delta_recurrence`] — per-v-head decay → kv_mem → delta →
//!   state update → output. The novel part of GatedDeltaNet.
//!
//! ## Tolerance regimes
//!
//! - `rms_norm_bare`: bit-exact. Same arithmetic shape as the existing
//!   weighted `rms_norm_cpu`; sequential sum-of-squares + sqrt + scale.
//! - `conv1d_step`: ULP-bounded. The dot-product kernel fuses
//!   `acc + state[k]*w[k]` patterns and uses `mul_add` to match clang's
//!   FMA contraction proactively (per the LM head finding); the SiLU
//!   tail introduces one `expf` per channel, which is the ULP source.
//! - `rms_norm_gated`: ULP-bounded. SiLU has one `expf` per element.

use crate::riir::embedding::bf16_to_f32;
use crate::riir::weight_file::WeightFile;

/// Errors specific to the linear-attention primitives.
#[derive(Debug, thiserror::Error)]
pub enum LinearAttnError {
    #[error("weight tensor '{name}' missing from manifest")]
    MissingTensor { name: String },
    #[error(
        "weight tensor '{name}' has {got} bytes, expected {expected} (= {elems} bf16 elements)"
    )]
    WeightSize {
        name: String,
        got: u64,
        expected: u64,
        elems: usize,
    },
    #[error("input length {got} != expected {expected}")]
    InputLen { got: usize, expected: usize },
    #[error("output length {got} != expected {expected}")]
    OutputLen { got: usize, expected: usize },
    #[error(
        "conv state length {got} != (kernel_size-1) * channels = {expected}"
    )]
    ConvStateLen { got: usize, expected: usize },
    #[error("non-positive shape: channels={channels} kernel_size={kernel_size}")]
    BadConvShape { channels: usize, kernel_size: usize },
    #[error("dim must be positive (got 0)")]
    ZeroDim,
}

/// SiLU activation in place: `x[i] = x[i] / (1 + exp(-x[i]))`. Matches
/// the C `cpu_silu` exactly. Helper for the conv1d tail.
#[inline]
fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

/// Depthwise 1D conv step with state shift + SiLU tail. For each
/// channel `c`, the dot product over `[conv_state..., new_input]`
/// against the channel's `kernel_size`-long weight row is computed,
/// then SiLU-activated.
///
/// Layout (matches `cpu_conv1d_step` in `infer.m`):
///
/// - `conv_state`: `[(kernel_size-1) * channels]`, row-major over
///   `(time, channel)` — i.e. `conv_state[k * channels + c]` is the
///   value at relative time `k` for channel `c`.
/// - `new_input`: `[channels]`, the latest time step.
/// - `weight_bf16`: `[channels * kernel_size]`, row-major over
///   `(channel, time)` — i.e. `weight[c * kernel_size + k]` for
///   channel `c` at relative time `k`. Time index `kernel_size-1`
///   is the slot for `new_input`.
/// - `out`: `[channels]`, written from scratch.
///
/// State shifting is the caller's responsibility — this function
/// reads the state but does not advance it. (`linear_attention_forward`
/// in C performs the `memmove` + `memcpy` shift outside the call.)
///
/// `mul_add` is used at the FMA contraction site to match clang's
/// AArch64 `-O3` codegen byte-for-byte; the per-channel SiLU tail is
/// the only ULP-source remaining.
pub fn conv1d_step(
    conv_state: &[f32],
    new_input: &[f32],
    weight_bf16: &[u8],
    channels: usize,
    kernel_size: usize,
    out: &mut [f32],
) -> Result<(), LinearAttnError> {
    if channels == 0 || kernel_size == 0 {
        return Err(LinearAttnError::BadConvShape {
            channels,
            kernel_size,
        });
    }
    let expected_state = (kernel_size - 1) * channels;
    if conv_state.len() != expected_state {
        return Err(LinearAttnError::ConvStateLen {
            got: conv_state.len(),
            expected: expected_state,
        });
    }
    if new_input.len() != channels {
        return Err(LinearAttnError::InputLen {
            got: new_input.len(),
            expected: channels,
        });
    }
    if out.len() != channels {
        return Err(LinearAttnError::OutputLen {
            got: out.len(),
            expected: channels,
        });
    }
    let expected_weight_bytes = (channels * kernel_size * 2) as u64;
    if (weight_bf16.len() as u64) < expected_weight_bytes {
        return Err(LinearAttnError::WeightSize {
            name: "<conv1d weight>".to_string(),
            got: weight_bf16.len() as u64,
            expected: expected_weight_bytes,
            elems: channels * kernel_size,
        });
    }

    for c in 0..channels {
        let mut acc: f32 = 0.0;
        for k in 0..kernel_size - 1 {
            let w_idx = c * kernel_size + k;
            let w_bits = u16::from_le_bytes([
                weight_bf16[w_idx * 2],
                weight_bf16[w_idx * 2 + 1],
            ]);
            let w = bf16_to_f32(w_bits);
            let s = conv_state[k * channels + c];
            acc = s.mul_add(w, acc);
        }
        let w_idx = c * kernel_size + (kernel_size - 1);
        let w_bits = u16::from_le_bytes([
            weight_bf16[w_idx * 2],
            weight_bf16[w_idx * 2 + 1],
        ]);
        let w = bf16_to_f32(w_bits);
        acc = new_input[c].mul_add(w, acc);
        out[c] = acc;
    }

    silu_inplace(out);
    Ok(())
}

/// `out[i] = x[i] / sqrt(mean(x*x) + eps)` — bare RMSNorm, no weight.
/// Bit-exact against the C `cpu_rms_norm_bare` on the same hardware.
pub fn rms_norm_bare(
    x: &[f32],
    eps: f32,
    out: &mut [f32],
) -> Result<(), LinearAttnError> {
    let dim = x.len();
    if dim == 0 {
        return Err(LinearAttnError::ZeroDim);
    }
    if out.len() != dim {
        return Err(LinearAttnError::OutputLen {
            got: out.len(),
            expected: dim,
        });
    }
    let mut sum_sq: f32 = 0.0;
    for &xi in x.iter() {
        sum_sq += xi * xi;
    }
    let inv_rms = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms;
    }
    Ok(())
}

/// `out[i] = x[i] * inv_rms(x) * w[i] * silu(z[i])`. Matches the C
/// `cpu_rms_norm_gated` exactly — same sequential reduction order in
/// the sum-of-squares loop, same per-element SiLU. ULP-bounded against
/// the C path because of the `expf` in SiLU.
pub fn rms_norm_gated(
    wf: &WeightFile,
    weight_name: &str,
    x: &[f32],
    z: &[f32],
    eps: f32,
    out: &mut [f32],
) -> Result<(), LinearAttnError> {
    let dim = x.len();
    if dim == 0 {
        return Err(LinearAttnError::ZeroDim);
    }
    if z.len() != dim {
        return Err(LinearAttnError::InputLen {
            got: z.len(),
            expected: dim,
        });
    }
    if out.len() != dim {
        return Err(LinearAttnError::OutputLen {
            got: out.len(),
            expected: dim,
        });
    }

    let bytes = wf
        .tensor_bytes(weight_name)
        .ok_or_else(|| LinearAttnError::MissingTensor {
            name: weight_name.to_string(),
        })?;
    let expected_bytes = (dim * 2) as u64;
    if bytes.len() as u64 != expected_bytes {
        return Err(LinearAttnError::WeightSize {
            name: weight_name.to_string(),
            got: bytes.len() as u64,
            expected: expected_bytes,
            elems: dim,
        });
    }

    let mut sum_sq: f32 = 0.0;
    for &xi in x.iter() {
        sum_sq += xi * xi;
    }
    let inv_rms = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        let w_bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        let w = bf16_to_f32(w_bits);
        let silu_z = z[i] / (1.0f32 + (-z[i]).exp());
        out[i] = x[i] * inv_rms * w * silu_z;
    }
    Ok(())
}

/// Gated-delta-net recurrence — Rust port of the per-v-head loop in
/// `linear_attention_forward` (lines 2693–2746 of `infer.m`).
///
/// For each value head `vh`:
///
///   g = exp(-exp(A_log[vh]) * softplus(alpha[vh] + dt_bias[vh]))
///   b_gate = sigmoid(beta[vh])
///   S = ssm_state[vh, :, :]                  // [VALUE_DIM, KEY_DIM]
///   k_h = k[vh / k_heads_per_v, :]           // GQA: 4 v-heads per k-head
///   v_h = v[vh, :]
///   q_h = q[vh / k_heads_per_v, :]
///
///   S *= g                                   // step 1: decay
///   for vi in 0..VALUE_DIM:                  // step 2: update
///     kv_mem    = Σ_ki S[vi, ki] * k_h[ki]
///     delta     = (v_h[vi] - kv_mem) * b_gate
///     S[vi,:]  += k_h * delta
///   for vi in 0..VALUE_DIM:                  // step 3: output
///     out_values[vh, vi] = Σ_ki S[vi, ki] * q_h[ki]
///
/// All FMA-shaped contractions (`kv_mem +=`, `S +=`, `sum +=`) use
/// `mul_add` to match clang's AArch64 codegen, mirroring the LM head
/// finding. The `expf`/`logf`/`sigmoid` calls in the per-head decay
/// precomputation are scalar libm; same regime as the other slices in
/// this kernel.
///
/// The state buffer is mutated in place and reflects the post-step
/// recurrence state on return; `out_values` is overwritten.
///
/// # Argument layout
///
/// - `a_log`        : `f32[v_heads]`, raw `A_log` tensor (NOT bf16).
/// - `dt_bias_bf16` : `u8[v_heads * 2]`, `dt_bias` as little-endian bf16.
/// - `alpha`        : `f32[v_heads]`, per-step alpha gate input.
/// - `beta`         : `f32[v_heads]`, per-step beta gate input.
/// - `q, k`         : `f32[k_heads * key_dim]`, per-step Q and K.
/// - `v`            : `f32[v_heads * value_dim]`, per-step V.
/// - `ssm_state`    : `f32[v_heads * value_dim * key_dim]`, mutated.
/// - `out_values`   : `f32[v_heads * value_dim]`, written.
///
/// `v_heads` must be a multiple of `k_heads` (GQA). All other shape
/// constants are `Variant::LINEAR_KEY_DIM` / `Variant::LINEAR_VALUE_DIM`
/// (passed by the caller for testability).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_recurrence(
    a_log: &[f32],
    dt_bias_bf16: &[u8],
    alpha: &[f32],
    beta: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    v_heads: usize,
    k_heads: usize,
    key_dim: usize,
    value_dim: usize,
    ssm_state: &mut [f32],
    out_values: &mut [f32],
) -> Result<(), LinearAttnError> {
    if v_heads == 0 || k_heads == 0 || key_dim == 0 || value_dim == 0 {
        return Err(LinearAttnError::ZeroDim);
    }
    if v_heads % k_heads != 0 {
        return Err(LinearAttnError::InputLen {
            got: v_heads,
            expected: k_heads,
        });
    }
    if a_log.len() != v_heads {
        return Err(LinearAttnError::InputLen {
            got: a_log.len(),
            expected: v_heads,
        });
    }
    if dt_bias_bf16.len() != v_heads * 2 {
        return Err(LinearAttnError::InputLen {
            got: dt_bias_bf16.len(),
            expected: v_heads * 2,
        });
    }
    if alpha.len() != v_heads || beta.len() != v_heads {
        return Err(LinearAttnError::InputLen {
            got: alpha.len(),
            expected: v_heads,
        });
    }
    if q.len() != k_heads * key_dim || k.len() != k_heads * key_dim {
        return Err(LinearAttnError::InputLen {
            got: q.len(),
            expected: k_heads * key_dim,
        });
    }
    if v.len() != v_heads * value_dim {
        return Err(LinearAttnError::InputLen {
            got: v.len(),
            expected: v_heads * value_dim,
        });
    }
    if ssm_state.len() != v_heads * value_dim * key_dim {
        return Err(LinearAttnError::InputLen {
            got: ssm_state.len(),
            expected: v_heads * value_dim * key_dim,
        });
    }
    if out_values.len() != v_heads * value_dim {
        return Err(LinearAttnError::OutputLen {
            got: out_values.len(),
            expected: v_heads * value_dim,
        });
    }

    let k_heads_per_v = v_heads / k_heads;

    // Per-head decay + beta_gate precomputation.
    let mut g_decay = vec![0.0f32; v_heads];
    let mut beta_gate = vec![0.0f32; v_heads];
    for vh in 0..v_heads {
        let a_val = alpha[vh];
        let dt_b = bf16_to_f32(u16::from_le_bytes([
            dt_bias_bf16[vh * 2],
            dt_bias_bf16[vh * 2 + 1],
        ]));
        let a_capital = a_log[vh].exp();
        let softplus_val = (1.0f32 + (a_val + dt_b).exp()).ln();
        g_decay[vh] = (-a_capital * softplus_val).exp();
        beta_gate[vh] = 1.0f32 / (1.0f32 + (-beta[vh]).exp());
    }

    let head_state_stride = value_dim * key_dim;
    for vh in 0..v_heads {
        let kh = vh / k_heads_per_v;
        let g = g_decay[vh];
        let b_gate = beta_gate[vh];
        let s_off = vh * head_state_stride;
        let v_off = vh * value_dim;
        let k_off = kh * key_dim;
        let q_off = kh * key_dim;
        let o_off = vh * value_dim;

        // Step 1: decay state in place.
        for s in &mut ssm_state[s_off..s_off + head_state_stride] {
            *s *= g;
        }

        // Step 2: per-vi predict-error-update.
        for vi in 0..value_dim {
            let row_off = s_off + vi * key_dim;
            let mut kv_mem: f32 = 0.0;
            for ki in 0..key_dim {
                kv_mem = ssm_state[row_off + ki].mul_add(k[k_off + ki], kv_mem);
            }
            let delta = (v[v_off + vi] - kv_mem) * b_gate;
            for ki in 0..key_dim {
                ssm_state[row_off + ki] =
                    k[k_off + ki].mul_add(delta, ssm_state[row_off + ki]);
            }
        }

        // Step 3: read-out.
        for vi in 0..value_dim {
            let row_off = s_off + vi * key_dim;
            let mut sum: f32 = 0.0;
            for ki in 0..key_dim {
                sum = ssm_state[row_off + ki].mul_add(q[q_off + ki], sum);
            }
            out_values[o_off + vi] = sum;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_bare_normalizes_unit_input() {
        let x = vec![1.0f32; 16];
        let mut out = vec![0.0f32; 16];
        rms_norm_bare(&x, 1e-6, &mut out).unwrap();
        // sqrt(mean(1) + eps) ≈ 1, so out ≈ 1 / 1 ≈ 1.
        for v in &out {
            assert!((*v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn silu_at_zero_is_zero() {
        let mut x = [0.0f32];
        silu_inplace(&mut x);
        // silu(0) = 0 / (1 + 1) = 0
        assert_eq!(x[0], 0.0);
    }

    #[test]
    fn silu_at_large_positive_approaches_input() {
        let mut x = [10.0f32];
        silu_inplace(&mut x);
        // silu(10) ≈ 10 / (1 + e^-10) ≈ 10 * (1 - tiny)
        assert!((x[0] - 10.0).abs() < 1e-3);
    }
}
