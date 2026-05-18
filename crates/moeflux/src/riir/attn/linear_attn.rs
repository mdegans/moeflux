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

use crate::riir::io::embedding::bf16_to_f32;
use crate::riir::io::weight_file::WeightFile;

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
    if v_heads == 0 {
        return Err(LinearAttnError::ZeroDim);
    }
    if alpha.len() != v_heads || beta.len() != v_heads {
        return Err(LinearAttnError::InputLen {
            got: alpha.len(),
            expected: v_heads,
        });
    }
    let mut g_decay = vec![0.0f32; v_heads];
    let mut beta_gate = vec![0.0f32; v_heads];
    compute_decay_beta_cpu(
        alpha,
        beta,
        a_log,
        dt_bias_bf16,
        &mut g_decay,
        &mut beta_gate,
    )?;
    gated_delta_recurrence_supplied(
        &g_decay, &beta_gate, q, k, v, v_heads, k_heads, key_dim, value_dim,
        ssm_state, out_values,
    )
}

/// Recurrence-only variant of [`gated_delta_recurrence`] that takes
/// pre-computed `g_decay` and `beta_gate` instead of deriving them
/// from `alpha`/`beta`/`a_log`/`dt_bias`. Used by the Graph
/// compiler's `GatedDeltaNetStepNTokens` op, where the decay/beta
/// math is a separate `ComputeDecayBetaNTokens` op upstream.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_recurrence_supplied(
    g_decay: &[f32],
    beta_gate: &[f32],
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
    if g_decay.len() != v_heads || beta_gate.len() != v_heads {
        return Err(LinearAttnError::InputLen {
            got: g_decay.len(),
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

/// Chunkwise-parallel reformulation of
/// [`gated_delta_recurrence_supplied`].
///
/// The per-token delta-rule recurrence is
/// `S_t = g_t·S_{t-1} + u_t·k_tᵀ` with
/// `u_t = β_t·(v_t − g_t·S_{t-1}·k_t)` and readout `out_t = S_t·q_t`.
/// Run token-by-token it is a sequential scan — the wall in moeflux
/// prefill. This function instead splits the `n_tokens` sequence into
/// chunks of `chunk_size` and, within each chunk, expresses the
/// recurrence as matmuls plus one lower-triangular solve. Only the
/// chunk-to-chunk state carry stays sequential (`n / chunk_size`
/// steps instead of `n`).
///
/// Algebra (per v-head, per chunk; local token index `l ∈ 0..C`,
/// incoming state `S_0`). With cumulative log-decay
/// `L_l = Σ_{j≤l} ln g_j`, `γ_l = exp(L_l)`, and decay ratio
/// `Γ_{l,i} = exp(L_l − L_i) ∈ (0,1]` for `i ≤ l`:
///
/// ```text
/// A_{l,i} = β_l·Γ_{l,i}·(k_i·k_l)          (strictly lower, i < l)
/// B_l     = β_l·v_l − β_l·γ_l·(S_0·k_l)
/// (I + A)·U = B                            (forward substitution)
/// out_l   = γ_l·(S_0·q_l) + Σ_{i≤l} Γ_{l,i}·(k_i·q_l)·U_i
/// S_C     = γ_{C-1}·S_0 + Σ_i Γ_{C-1,i}·U_i·k_iᵀ
/// ```
///
/// This is the CPU reference for the eventual Metal chunkwise kernel.
/// It is algebraically equal to the per-token oracle (and diff-tested
/// against it); it is written for legibility, not speed — the GPU
/// port carries the speed.
///
/// ## Layout
/// - `g_decay`, `beta_gate`: `[n_tokens · v_heads]`, token-major.
/// - `q`, `k`: `[n_tokens · k_heads · key_dim]`, token-major.
/// - `v`: `[n_tokens · v_heads · value_dim]`, token-major.
/// - `ssm_state`: `[v_heads · value_dim · key_dim]` — `S[vh][vi][ki]`,
///   in/out, carried across the whole call.
/// - `out_values`: `[n_tokens · v_heads · value_dim]`, token-major.
///
/// `g_decay` is `exp(…)` and therefore strictly positive, so the
/// log-space cumulative decay never sees `ln(0)`.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunkwise(
    g_decay: &[f32],
    beta_gate: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_tokens: usize,
    chunk_size: usize,
    v_heads: usize,
    k_heads: usize,
    key_dim: usize,
    value_dim: usize,
    ssm_state: &mut [f32],
    out_values: &mut [f32],
) -> Result<(), LinearAttnError> {
    if v_heads == 0
        || k_heads == 0
        || key_dim == 0
        || value_dim == 0
        || chunk_size == 0
    {
        return Err(LinearAttnError::ZeroDim);
    }
    if v_heads % k_heads != 0 {
        return Err(LinearAttnError::InputLen {
            got: v_heads,
            expected: k_heads,
        });
    }
    let key_total = k_heads * key_dim;
    let value_total = v_heads * value_dim;
    for (got, expected) in [
        (g_decay.len(), n_tokens * v_heads),
        (beta_gate.len(), n_tokens * v_heads),
        (q.len(), n_tokens * key_total),
        (k.len(), n_tokens * key_total),
        (v.len(), n_tokens * value_total),
        (out_values.len(), n_tokens * value_total),
    ] {
        if got != expected {
            return Err(LinearAttnError::InputLen { got, expected });
        }
    }
    if ssm_state.len() != v_heads * value_dim * key_dim {
        return Err(LinearAttnError::InputLen {
            got: ssm_state.len(),
            expected: v_heads * value_dim * key_dim,
        });
    }
    if n_tokens == 0 {
        return Ok(());
    }

    let dot = |a: &[f32], b: &[f32]| -> f32 {
        let mut s = 0.0f32;
        for i in 0..a.len() {
            s = a[i].mul_add(b[i], s);
        }
        s
    };

    let k_heads_per_v = v_heads / k_heads;
    let head_state_stride = value_dim * key_dim;

    for vh in 0..v_heads {
        let kh = vh / k_heads_per_v;
        let s_off = vh * head_state_stride;

        let mut chunk_start = 0usize;
        while chunk_start < n_tokens {
            let c = (n_tokens - chunk_start).min(chunk_size);

            // Gather this chunk's per-token inputs for head (vh, kh)
            // into chunk-local buffers, plus the cumulative log-decay.
            let mut kc = vec![0.0f32; c * key_dim];
            let mut qc = vec![0.0f32; c * key_dim];
            let mut vc = vec![0.0f32; c * value_dim];
            let mut log_decay = vec![0.0f32; c];
            let mut beta = vec![0.0f32; c];
            let mut acc = 0.0f32;
            for l in 0..c {
                let t = chunk_start + l;
                let kq_lo = t * key_total + kh * key_dim;
                kc[l * key_dim..(l + 1) * key_dim]
                    .copy_from_slice(&k[kq_lo..kq_lo + key_dim]);
                qc[l * key_dim..(l + 1) * key_dim]
                    .copy_from_slice(&q[kq_lo..kq_lo + key_dim]);
                let v_lo = t * value_total + vh * value_dim;
                vc[l * value_dim..(l + 1) * value_dim]
                    .copy_from_slice(&v[v_lo..v_lo + value_dim]);
                acc += g_decay[t * v_heads + vh].ln();
                log_decay[l] = acc;
                beta[l] = beta_gate[t * v_heads + vh];
            }
            let gamma_at = |l: usize| log_decay[l].exp();
            // Γ_{l,i} = exp(L_l − L_i), for i ≤ l (bounded in (0,1]).
            let decay_ratio =
                |l: usize, i: usize| (log_decay[l] - log_decay[i]).exp();
            let krow = |l: usize| &kc[l * key_dim..(l + 1) * key_dim];
            let qrow = |l: usize| &qc[l * key_dim..(l + 1) * key_dim];

            // Snapshot the incoming state — every chunk-internal read
            // uses S_0; the new state is written only at the end.
            let s0 = ssm_state[s_off..s_off + head_state_stride].to_vec();
            // (S_0·x)[vi] = Σ_ki S_0[vi][ki]·x[ki].
            let s0_apply = |x: &[f32], out: &mut [f32]| {
                for vi in 0..value_dim {
                    out[vi] =
                        dot(&s0[vi * key_dim..(vi + 1) * key_dim], x);
                }
            };

            // A: strictly-lower-triangular c×c.
            let mut a = vec![0.0f32; c * c];
            for l in 0..c {
                for i in 0..l {
                    a[l * c + i] = beta[l]
                        * decay_ratio(l, i)
                        * dot(krow(i), krow(l));
                }
            }

            // RHS B_l = β_l·v_l − β_l·γ_l·(S_0·k_l), into `u`.
            let mut u = vec![0.0f32; c * value_dim];
            let mut scratch = vec![0.0f32; value_dim];
            for l in 0..c {
                s0_apply(krow(l), &mut scratch);
                let gl = gamma_at(l);
                let row = &mut u[l * value_dim..(l + 1) * value_dim];
                for vi in 0..value_dim {
                    row[vi] = beta[l] * vc[l * value_dim + vi]
                        - beta[l] * gl * scratch[vi];
                }
            }
            // Forward substitution: U_l −= Σ_{i<l} A_{l,i}·U_i.
            for l in 0..c {
                for i in 0..l {
                    let coef = a[l * c + i];
                    if coef == 0.0 {
                        continue;
                    }
                    let (lo, hi) = u.split_at_mut(l * value_dim);
                    let ui = &lo[i * value_dim..(i + 1) * value_dim];
                    let ul = &mut hi[..value_dim];
                    for vi in 0..value_dim {
                        ul[vi] -= coef * ui[vi];
                    }
                }
            }

            // Outputs: out_l = γ_l·(S_0·q_l)
            //                  + Σ_{i≤l} Γ_{l,i}·(k_i·q_l)·U_i.
            for l in 0..c {
                let t = chunk_start + l;
                s0_apply(qrow(l), &mut scratch);
                let gl = gamma_at(l);
                let o_lo = t * value_total + vh * value_dim;
                let out_l = &mut out_values[o_lo..o_lo + value_dim];
                for vi in 0..value_dim {
                    out_l[vi] = gl * scratch[vi];
                }
                for i in 0..=l {
                    let coef =
                        decay_ratio(l, i) * dot(krow(i), qrow(l));
                    let ui = &u[i * value_dim..(i + 1) * value_dim];
                    for vi in 0..value_dim {
                        out_l[vi] = coef.mul_add(ui[vi], out_l[vi]);
                    }
                }
            }

            // New state: S_C = γ_{C-1}·S_0 + Σ_i Γ_{C-1,i}·U_i·k_iᵀ.
            let last = c - 1;
            let g_last = gamma_at(last);
            let new_state =
                &mut ssm_state[s_off..s_off + head_state_stride];
            for (dst, src) in new_state.iter_mut().zip(s0.iter()) {
                *dst = g_last * src;
            }
            for i in 0..c {
                let ratio = decay_ratio(last, i);
                let ki = krow(i);
                let ui = &u[i * value_dim..(i + 1) * value_dim];
                for vi in 0..value_dim {
                    let coef = ratio * ui[vi];
                    let row =
                        &mut new_state[vi * key_dim..(vi + 1) * key_dim];
                    for (rk, &kk) in row.iter_mut().zip(ki.iter()) {
                        *rk = coef.mul_add(kk, *rk);
                    }
                }
            }

            chunk_start += c;
        }
    }

    Ok(())
}

/// Compute the linear-attn 1c step element-wise per v-head:
///
/// ```text
/// g_decay[h]   = exp(-exp(a_log[h]) * softplus(alpha[h] + dt_bias[h]))
/// beta_gate[h] = sigmoid(beta[h])
/// ```
///
/// CPU oracle for the Metal `compute_decay_beta` kernel at
/// `shaders.metal:1831`. Note: `a_log` is read as f32 in the kernel
/// (despite living in the `wf` weight buffer alongside bf16 weights),
/// while `dt_bias` is bf16.
///
/// Lengths: all six slices must equal `num_v_heads`.
#[inline]
pub fn compute_decay_beta_cpu(
    alpha: &[f32],
    beta: &[f32],
    a_log: &[f32],
    dt_bias_bf16: &[u8],
    g_decay_out: &mut [f32],
    beta_gate_out: &mut [f32],
) -> Result<(), LinearAttnError> {
    let n = alpha.len();
    if beta.len() != n
        || a_log.len() != n
        || g_decay_out.len() != n
        || beta_gate_out.len() != n
    {
        return Err(LinearAttnError::OutputLen {
            got: beta.len(),
            expected: n,
        });
    }
    if dt_bias_bf16.len() < n * 2 {
        return Err(LinearAttnError::WeightSize {
            name: "<dt_bias>".to_string(),
            got: dt_bias_bf16.len() as u64,
            expected: (n * 2) as u64,
            elems: n,
        });
    }
    for h in 0..n {
        let dt_b_bits =
            u16::from_le_bytes([dt_bias_bf16[h * 2], dt_bias_bf16[h * 2 + 1]]);
        let dt_b = bf16_to_f32(dt_b_bits);
        let a_val = alpha[h];
        let a_decay = a_log[h].exp();
        let softplus_val = (1.0f32 + (a_val + dt_b).exp()).ln();
        g_decay_out[h] = (-a_decay * softplus_val).exp();
        beta_gate_out[h] = 1.0f32 / (1.0f32 + (-beta[h]).exp());
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
    fn compute_decay_beta_at_zero_inputs() {
        // alpha = beta = 0, a_log = 0 (so a_decay = exp(0) = 1),
        // dt_bias = 0 (bf16 0x0000).
        // softplus(0 + 0) = ln(1 + e^0) = ln(2)
        // g_decay = exp(-1 * ln(2)) = 0.5
        // beta_gate = sigmoid(0) = 0.5
        let alpha = vec![0.0f32; 4];
        let beta = vec![0.0f32; 4];
        let a_log = vec![0.0f32; 4];
        let dt_bias_bf16 = vec![0u8; 4 * 2];
        let mut g_decay = vec![0.0f32; 4];
        let mut beta_gate = vec![0.0f32; 4];
        compute_decay_beta_cpu(
            &alpha, &beta, &a_log, &dt_bias_bf16, &mut g_decay, &mut beta_gate,
        )
        .unwrap();
        for h in 0..4 {
            assert!((g_decay[h] - 0.5).abs() < 1e-6, "g_decay[{h}] = {}", g_decay[h]);
            assert!((beta_gate[h] - 0.5).abs() < 1e-6, "beta_gate[{h}] = {}", beta_gate[h]);
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

    /// LCG → f32 in `[-1, 1)`. Deterministic; seed per call site.
    fn lcg_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let raw = ((*state >> 40) as f32) / ((1u32 << 24) as f32);
        2.0 * raw - 1.0
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        if na == 0.0 && nb == 0.0 {
            return 1.0;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// The chunkwise reformulation must reproduce the sequential
    /// per-token recurrence — Phase 1's load-bearing math gate. If
    /// this fails, the chunkwise algebra is wrong; catch it here, in
    /// pure Rust, before any Metal kernel work.
    ///
    /// Small head dims keep the test fast; the algebra is
    /// dimension-agnostic so this fully exercises it. `key_dim ≠
    /// value_dim` and `k_heads_per_v = 4` catch index/head-sharing
    /// bugs. The grid spans `n < C`, exact and ragged final chunks,
    /// and the across-chunk state carry.
    #[test]
    fn gated_delta_chunkwise_matches_per_token() {
        let v_heads = 8usize;
        let k_heads = 2usize;
        let key_dim = 16usize;
        let value_dim = 12usize;
        let key_total = k_heads * key_dim;
        let value_total = v_heads * value_dim;
        let state_floats = v_heads * value_dim * key_dim;

        for &n_tokens in &[1usize, 7, 64, 65, 200] {
            // Inputs (shared by both paths). Ranges mirror the
            // existing per-token diff test's generators.
            let mut rng = 0x5DEECE66Du64 ^ (n_tokens as u64);
            let g_decay: Vec<f32> = (0..n_tokens * v_heads)
                .map(|_| 0.9 + lcg_unit(&mut rng) * 0.05)
                .collect();
            let beta_gate: Vec<f32> = (0..n_tokens * v_heads)
                .map(|_| 0.5 + lcg_unit(&mut rng) * 0.2)
                .collect();
            let q: Vec<f32> = (0..n_tokens * key_total)
                .map(|_| lcg_unit(&mut rng) * 0.5)
                .collect();
            let k: Vec<f32> = (0..n_tokens * key_total)
                .map(|_| lcg_unit(&mut rng) * 0.5)
                .collect();
            let v: Vec<f32> = (0..n_tokens * value_total)
                .map(|_| lcg_unit(&mut rng) * 0.5)
                .collect();
            let initial_state: Vec<f32> = (0..state_floats)
                .map(|_| lcg_unit(&mut rng) * 0.01)
                .collect();

            // Oracle: the sequential per-token recurrence.
            let mut oracle_state = initial_state.clone();
            let mut oracle_out = vec![0.0f32; n_tokens * value_total];
            for t in 0..n_tokens {
                let mut out_t = vec![0.0f32; value_total];
                gated_delta_recurrence_supplied(
                    &g_decay[t * v_heads..(t + 1) * v_heads],
                    &beta_gate[t * v_heads..(t + 1) * v_heads],
                    &q[t * key_total..(t + 1) * key_total],
                    &k[t * key_total..(t + 1) * key_total],
                    &v[t * value_total..(t + 1) * value_total],
                    v_heads,
                    k_heads,
                    key_dim,
                    value_dim,
                    &mut oracle_state,
                    &mut out_t,
                )
                .expect("per-token oracle");
                oracle_out[t * value_total..(t + 1) * value_total]
                    .copy_from_slice(&out_t);
            }

            for &chunk_size in &[16usize, 64] {
                let mut cw_state = initial_state.clone();
                let mut cw_out = vec![0.0f32; n_tokens * value_total];
                gated_delta_chunkwise(
                    &g_decay,
                    &beta_gate,
                    &q,
                    &k,
                    &v,
                    n_tokens,
                    chunk_size,
                    v_heads,
                    k_heads,
                    key_dim,
                    value_dim,
                    &mut cw_state,
                    &mut cw_out,
                )
                .expect("chunkwise");

                let cos_out = cosine(&oracle_out, &cw_out);
                let cos_state = cosine(&oracle_state, &cw_state);
                let mad_out = max_abs_diff(&oracle_out, &cw_out);
                let mad_state =
                    max_abs_diff(&oracle_state, &cw_state);
                eprintln!(
                    "[chunkwise] n={n_tokens} C={chunk_size}: \
                     out cos={cos_out:.9} max_abs={mad_out:.3e}; \
                     state cos={cos_state:.9} max_abs={mad_state:.3e}"
                );
                assert!(
                    cos_out >= 0.9999,
                    "n={n_tokens} C={chunk_size}: output cos={cos_out}"
                );
                assert!(
                    cos_state >= 0.9999,
                    "n={n_tokens} C={chunk_size}: state cos={cos_state}"
                );
                // Inputs are small (|out| ~ 1e-2 scale); a tight
                // absolute bound also catches a systematic offset
                // that cosine alone would miss.
                assert!(
                    mad_out < 1e-4,
                    "n={n_tokens} C={chunk_size}: output max_abs={mad_out}"
                );
                assert!(
                    mad_state < 1e-4,
                    "n={n_tokens} C={chunk_size}: state max_abs={mad_state}"
                );
            }
        }
    }
}
