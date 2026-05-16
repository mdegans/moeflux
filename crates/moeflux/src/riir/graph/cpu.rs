//! `CpuBackend` — first customer of the `Backend` trait.
//!
//! Wraps the existing CPU oracle helpers in a `Backend` impl. Encoding
//! is inline execution: `encode_op` runs the kernel and writes through
//! the pool's `RefCell`-backed buffers; `submit_and_wait` is a no-op.
//!
//! Used by S7-4's `graph_metal_matches_cpu` diff test as the reference
//! truth against which `MetalBackend`'s encoded output is compared.
//! Also stays in-tree post-S7 as a regression net for future kernel
//! work.

use std::cell::{Ref, RefCell, RefMut};

use crate::riir::cpu_matvec::{
    dequant_matvec_4bit_cpu, dequant_matvec_8bit_v3_cpu,
};
use crate::riir::cpu_ops::{cpu_sigmoid_scalar, residual_add_n_tokens_cpu};
use crate::riir::embedding::bf16_to_f32;
use crate::riir::linear_attn::{
    compute_decay_beta_cpu, conv1d_step, gated_delta_recurrence_supplied,
};
use crate::riir::moe_cpu::moe_permute_fuse_cpu;
use crate::riir::variants::{GROUP_SIZE, VARIANT};
use crate::riir::weight_file::WeightFile;

use super::{Backend, BufId, BufferPool, Graph, GraphError, Op};

/// CPU buffer pool: physical storage is `Vec<RefCell<Vec<u8>>>`,
/// indexed *indirectly* by `BufId` through `bufid_to_physical`. Pre-
/// [`BufferPool::commit_plan`], the mapping is identity (one buffer
/// per `BufId`). Post-`commit_plan`, multiple colorable `BufId`s may
/// share a single physical buffer.
///
/// `reset_transient` keeps the longest persistent prefix of BufIds
/// (producer convention: persistent allocations precede transient
/// ones) and drops everything beyond.
pub struct CpuBufferPool {
    /// Physical storage. `buffers.len() == physical_buffer_count()`.
    buffers: Vec<RefCell<Vec<u8>>>,
    /// Per-`BufId` label, byte-size, persistent flag, and physical
    /// index. All four are kept in lock-step with the BufId space.
    labels: Vec<&'static str>,
    persistent: Vec<bool>,
    byte_sizes: Vec<usize>,
    bufid_to_physical: Vec<u32>,
}

impl CpuBufferPool {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            labels: Vec::new(),
            persistent: Vec::new(),
            byte_sizes: Vec::new(),
            bufid_to_physical: Vec::new(),
        }
    }

    /// Number of physical buffers actually allocated. With identity
    /// mapping this equals the number of `BufId`s; after
    /// `commit_plan` it can be strictly less (the aliasing win).
    pub fn physical_buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

impl Default for CpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool for CpuBufferPool {
    type Handle = RefCell<Vec<u8>>;
    type Error = GraphError;

    fn alloc(
        &mut self,
        bytes: usize,
        label: &'static str,
        persistent: bool,
    ) -> Result<BufId, GraphError> {
        let id = BufId(self.bufid_to_physical.len() as u32);
        let physical = self.buffers.len() as u32;
        self.buffers.push(RefCell::new(vec![0u8; bytes]));
        self.labels.push(label);
        self.persistent.push(persistent);
        self.byte_sizes.push(bytes);
        self.bufid_to_physical.push(physical);
        Ok(id)
    }

    fn handle(&self, id: BufId) -> &RefCell<Vec<u8>> {
        let physical = self.bufid_to_physical[id.0 as usize] as usize;
        &self.buffers[physical]
    }

    fn upload(&mut self, id: BufId, host: &[u8]) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let expected = self.byte_sizes[idx];
        // Prefix semantics: `host` may be shorter than the buffer
        // (once-per-run buffers are sized at max chunk width; a
        // smaller chunk uploads only its rows). Too-large is rejected.
        if host.len() > expected {
            return Err(GraphError::SizeMismatch {
                label,
                expected,
                actual: host.len(),
            });
        }
        let physical = self.bufid_to_physical[idx] as usize;
        let mut buf_mut = self.buffers[physical].borrow_mut();
        buf_mut[..host.len()].copy_from_slice(host);
        Ok(())
    }

    fn download(&self, id: BufId, host: &mut [u8]) -> Result<(), GraphError> {
        let idx = id.0 as usize;
        let label = *self.labels.get(idx).ok_or(GraphError::BadBufId(id))?;
        let expected = self.byte_sizes[idx];
        // Prefix semantics: see `upload`.
        if host.len() > expected {
            return Err(GraphError::SizeMismatch {
                label,
                expected,
                actual: host.len(),
            });
        }
        let physical = self.bufid_to_physical[idx] as usize;
        let buf = self.buffers[physical].borrow();
        host.copy_from_slice(&buf[..host.len()]);
        Ok(())
    }

    fn reset_transient(&mut self) {
        // Keep the longest persistent prefix in BufId space and drop
        // everything past it. Producer convention: persistent allocs
        // (KV cache, hidden double-buffer, weight views) come before
        // transient intermediates. After `commit_plan`, persistents
        // retain their original physical indices (only colorable
        // BufIds are re-laid out), so the physical-buffer truncation
        // below is the persistent prefix length too.
        let mut keep_bufids = 0;
        for (i, &p) in self.persistent.iter().enumerate() {
            if p {
                keep_bufids = i + 1;
            }
        }
        self.labels.truncate(keep_bufids);
        self.persistent.truncate(keep_bufids);
        self.byte_sizes.truncate(keep_bufids);
        self.bufid_to_physical.truncate(keep_bufids);

        // Physical-buffer truncation: drop any physical buffer no
        // longer referenced by a surviving BufId. Find the highest
        // physical index still in use (+1) and truncate there.
        let max_physical = self
            .bufid_to_physical
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);
        self.buffers.truncate(max_physical);
    }

    fn label(&self, id: BufId) -> &'static str {
        self.labels
            .get(id.0 as usize)
            .copied()
            .unwrap_or("<bad-bufid>")
    }

    fn commit_plan(&mut self, graph: &Graph) {
        use super::lifetime::{analyze_lifetimes, greedy_color, ColorId};
        use std::collections::HashMap;

        let lifetimes = analyze_lifetimes(graph);
        let coloring = greedy_color(&lifetimes);

        // Filter coloring: persistent BufIds keep their dedicated
        // physical buffer (content must survive `reset_transient`).
        // Non-persistent colorable BufIds are eligible for aliasing.
        let n_bufids = self.bufid_to_physical.len();
        let aliasable: HashMap<BufId, ColorId> = coloring
            .bufid_to_color
            .iter()
            .filter(|(b, _)| !self.persistent[b.0 as usize])
            .map(|(b, c)| (*b, *c))
            .collect();

        // Phase 1: place non-aliasable BufIds (persistent + non-
        // colorable transients) in the new physical layout, moving
        // their existing buffer to preserve content.
        //
        // A prior `commit_plan` may already have aliased BufIds onto
        // a shared physical buffer, so move each physical exactly
        // once (`old_to_new`) and remap every BufId that shared it to
        // the same new slot.
        let mut new_buffers: Vec<RefCell<Vec<u8>>> = Vec::new();
        let mut new_bufid_to_physical: Vec<u32> = vec![u32::MAX; n_bufids];
        let mut old_to_new: HashMap<usize, u32> = HashMap::new();
        for bufid_idx in 0..n_bufids {
            let buf = BufId(bufid_idx as u32);
            if aliasable.contains_key(&buf) {
                continue;
            }
            let old_physical = self.bufid_to_physical[bufid_idx] as usize;
            let new_phys = *old_to_new.entry(old_physical).or_insert_with(|| {
                let old_buf = std::mem::replace(
                    &mut self.buffers[old_physical],
                    RefCell::new(Vec::new()),
                );
                let np = new_buffers.len() as u32;
                new_buffers.push(old_buf);
                np
            });
            new_bufid_to_physical[bufid_idx] = new_phys;
        }

        // Phase 2: allocate one physical buffer per color group,
        // sized to the max byte_size among the group's BufIds.
        let mut color_to_physical: HashMap<ColorId, u32> = HashMap::new();
        for color in 0..coloring.color_count {
            let max_size = aliasable
                .iter()
                .filter(|&(_, c)| *c == color)
                .map(|(b, _)| self.byte_sizes[b.0 as usize])
                .max()
                .unwrap_or(0);
            if max_size == 0 {
                // No BufIds with this color after filtering out
                // persistents — skip.
                continue;
            }
            color_to_physical.insert(color, new_buffers.len() as u32);
            new_buffers.push(RefCell::new(vec![0u8; max_size]));
        }

        // Phase 3: point each aliasable BufId at its color's slot.
        for (buf, color) in &aliasable {
            let phys = color_to_physical[color];
            new_bufid_to_physical[buf.0 as usize] = phys;
        }

        debug_assert!(new_bufid_to_physical.iter().all(|&p| p != u32::MAX));
        self.buffers = new_buffers;
        self.bufid_to_physical = new_bufid_to_physical;

        // S10b-2: pin every colored BufId. Its physical layout is now
        // frozen for the run; flipping `persistent` keeps it (and the
        // shared color buffer it points at) across `reset_transient`.
        for buf in aliasable.keys() {
            self.persistent[buf.0 as usize] = true;
        }
    }
}

/// CPU Backend implementation. Each encode_op variant runs the
/// kernel inline using the [`CpuBufferPool`]'s RefCell-backed
/// buffers; `submit_and_wait` is a no-op (execution already
/// happened).
///
/// Weight resolution: each variant that reads weights uses
/// [`WeightFile::bytes_at`] against the backend's owned mmap.
/// `WeightRef` carries `(w_off, s_off, b_off, bits)`; the encode_op
/// arm computes byte lengths from dims and slices the mmap.
pub struct CpuBackend {
    pool: CpuBufferPool,
    wf: WeightFile,
}

impl CpuBackend {
    pub fn new(wf: WeightFile) -> Self {
        Self {
            pool: CpuBufferPool::new(),
            wf,
        }
    }

    pub fn weight_file(&self) -> &WeightFile {
        &self.wf
    }

    // ------------------------------------------------------------------
    // Slice accessors over pool buffers
    // ------------------------------------------------------------------

    fn read_f32(&self, id: BufId) -> Ref<'_, [f32]> {
        Ref::map(self.pool.handle(id).borrow(), |v| bytes_as::<f32>(v))
    }

    fn write_f32(&self, id: BufId) -> RefMut<'_, [f32]> {
        RefMut::map(self.pool.handle(id).borrow_mut(), |v| {
            bytes_as_mut::<f32>(v)
        })
    }

    #[allow(dead_code)]
    fn read_i32(&self, id: BufId) -> Ref<'_, [i32]> {
        Ref::map(self.pool.handle(id).borrow(), |v| bytes_as::<i32>(v))
    }

    fn write_i32(&self, id: BufId) -> RefMut<'_, [i32]> {
        RefMut::map(self.pool.handle(id).borrow_mut(), |v| {
            bytes_as_mut::<i32>(v)
        })
    }

    fn read_bytes(&self, id: BufId) -> Ref<'_, [u8]> {
        Ref::map(self.pool.handle(id).borrow(), |v| v.as_slice())
    }

    fn write_bytes(&self, id: BufId) -> RefMut<'_, [u8]> {
        RefMut::map(self.pool.handle(id).borrow_mut(), |v| v.as_mut_slice())
    }
}

// ----------------------------------------------------------------------------
// Byte → typed-slice cast helpers. Pool stores Vec<u8>; the kernels
// want &[f32] / &[i32]. align_to panics if the head/tail of the
// alignment isn't empty — that signals a misaligned alloc, which is a
// caller bug we want loud.
// ----------------------------------------------------------------------------

/// Reinterpret a pool byte buffer as `&[T]`. Panics if the buffer is
/// not `T`-aligned — that signals a misaligned pool alloc, a caller
/// bug we want loud rather than silently wrong.
fn bytes_as<T>(b: &[u8]) -> &[T] {
    // SAFETY: `align_to` is safe by definition; the assert guarantees
    // `body` covers the whole slice (empty head/tail).
    let (head, body, tail) = unsafe { b.align_to::<T>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "pool buffer not {}-aligned (head={}, tail={})",
        std::any::type_name::<T>(),
        head.len(),
        tail.len()
    );
    body
}

/// Mutable counterpart of [`bytes_as`].
fn bytes_as_mut<T>(b: &mut [u8]) -> &mut [T] {
    // SAFETY: see [`bytes_as`].
    let (head, body, tail) = unsafe { b.align_to_mut::<T>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "pool buffer not {}-aligned (head={}, tail={})",
        std::any::type_name::<T>(),
        head.len(),
        tail.len()
    );
    body
}

// ----------------------------------------------------------------------------
// Byte-variant CPU primitives for kernels whose existing helpers take
// tensor names (rms_norm, lm_head). Inlined here rather than refactor
// every existing call site.
// ----------------------------------------------------------------------------

fn rms_norm_bf16_n_tokens_cpu(
    weight_bf16: &[u8],
    x: &[f32],
    dim: usize,
    n_tokens: usize,
    eps: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), n_tokens * dim);
    debug_assert_eq!(out.len(), n_tokens * dim);
    debug_assert!(weight_bf16.len() >= dim * 2);
    for t in 0..n_tokens {
        let xt = &x[t * dim..(t + 1) * dim];
        let ot = &mut out[t * dim..(t + 1) * dim];
        let mut sum_sq = 0.0f32;
        for &xi in xt.iter() {
            sum_sq += xi * xi;
        }
        let inv_rms = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
        for i in 0..dim {
            let w_bits = u16::from_le_bytes([
                weight_bf16[i * 2],
                weight_bf16[i * 2 + 1],
            ]);
            let w = bf16_to_f32(w_bits);
            ot[i] = xt[i] * inv_rms * w;
        }
    }
}

fn rms_norm_qk_n_tokens_cpu(
    x_inout: &mut [f32],
    num_k_heads: usize,
    key_dim: usize,
    key_offset_per_token: usize,
    per_token_total: usize,
    n_tokens: usize,
    eps: f32,
) {
    // Apply per-head RMSNorm to the q region [base..base+num_k_heads*key_dim]
    // AND the k region [base+key_offset_per_token..]. Matches the
    // Metal `rms_norm_qk` shader (shaders.metal:1773): q is scaled by
    // `inv_scale * inv_scale` (absorbs 1/sqrt(key_dim) of the attention
    // pre-softmax scaling); k is scaled by `inv_scale` once.
    let inv_scale = 1.0f32 / (key_dim as f32).sqrt();
    let q_scale = inv_scale * inv_scale;
    let k_scale = inv_scale;
    debug_assert!(per_token_total >= key_offset_per_token + num_k_heads * key_dim);
    debug_assert_eq!(x_inout.len(), n_tokens * per_token_total);
    for t in 0..n_tokens {
        let base = t * per_token_total;
        // Q region
        for h in 0..num_k_heads {
            let off = base + h * key_dim;
            let row = &mut x_inout[off..off + key_dim];
            normalize_unweighted(row, eps, q_scale);
        }
        // K region
        for h in 0..num_k_heads {
            let off = base + key_offset_per_token + h * key_dim;
            let row = &mut x_inout[off..off + key_dim];
            normalize_unweighted(row, eps, k_scale);
        }
    }
}

fn normalize_unweighted(row: &mut [f32], eps: f32, inv_scale: f32) {
    let dim = row.len();
    let mut sum_sq = 0.0f32;
    for &v in row.iter() {
        sum_sq += v * v;
    }
    let inv_rms = 1.0f32 / (sum_sq / dim as f32 + eps).sqrt();
    for v in row.iter_mut() {
        *v = *v * inv_rms * inv_scale;
    }
}

fn gated_rms_norm_n_tokens_cpu(
    values: &[f32],
    z: &[f32],
    weight_bf16: &[u8],
    output: &mut [f32],
    num_v_heads: usize,
    value_dim: usize,
    n_tokens: usize,
    eps: f32,
) {
    let per_token = num_v_heads * value_dim;
    debug_assert_eq!(values.len(), n_tokens * per_token);
    debug_assert_eq!(z.len(), n_tokens * per_token);
    debug_assert_eq!(output.len(), n_tokens * per_token);
    debug_assert!(weight_bf16.len() >= value_dim * 2);
    for t in 0..n_tokens {
        for h in 0..num_v_heads {
            let base = t * per_token + h * value_dim;
            let v = &values[base..base + value_dim];
            let zr = &z[base..base + value_dim];
            let o = &mut output[base..base + value_dim];
            let mut sum_sq = 0.0f32;
            for &vi in v.iter() {
                sum_sq += vi * vi;
            }
            let inv_rms =
                1.0f32 / (sum_sq / value_dim as f32 + eps).sqrt();
            for i in 0..value_dim {
                let normed = v[i] * inv_rms;
                let zval = zr[i];
                let gate = zval / (1.0 + (-zval).exp()); // SiLU
                let w_bits = u16::from_le_bytes([
                    weight_bf16[i * 2],
                    weight_bf16[i * 2 + 1],
                ]);
                let w = bf16_to_f32(w_bits);
                o[i] = normed * gate * w;
            }
        }
    }
}

fn swiglu_fused_cpu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..gate.len() {
        let g = gate[i];
        let silu = g / (1.0 + (-g).exp());
        out[i] = silu * up[i];
    }
}

fn moe_softmax_topk_cpu(
    logits: &[f32],
    indices_out: &mut [i32],
    weights_out: &mut [f32],
    n_tokens: usize,
    n_experts: usize,
    k: usize,
) {
    debug_assert_eq!(logits.len(), n_tokens * n_experts);
    debug_assert_eq!(indices_out.len(), n_tokens * k);
    debug_assert_eq!(weights_out.len(), n_tokens * k);
    for t in 0..n_tokens {
        let lr = &logits[t * n_experts..(t + 1) * n_experts];
        // Softmax (numerically stable).
        let mut maxv = f32::NEG_INFINITY;
        for &v in lr.iter() {
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; n_experts];
        for (i, &v) in lr.iter().enumerate() {
            let p = (v - maxv).exp();
            probs[i] = p;
            sum += p;
        }
        let inv_sum = 1.0f32 / sum;
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
        // Top-K via running-minimum selection sort: matches the
        // Metal `moe_softmax_topk` kernel's slot order exactly so
        // diff is bit-exact-per-slot (no set sort needed).
        let ir = &mut indices_out[t * k..(t + 1) * k];
        let wr = &mut weights_out[t * k..(t + 1) * k];
        for slot in 0..k {
            ir[slot] = -1;
            wr[slot] = f32::NEG_INFINITY;
        }
        for (e, &p) in probs.iter().enumerate() {
            // Find the slot with the running minimum.
            let mut min_slot = 0;
            let mut min_val = wr[0];
            for s in 1..k {
                if wr[s] < min_val {
                    min_val = wr[s];
                    min_slot = s;
                }
            }
            if p > min_val {
                ir[min_slot] = e as i32;
                wr[min_slot] = p;
            }
        }
    }
}

fn moe_normalize_weights_cpu(weights: &mut [f32], n_tokens: usize, k: usize) {
    debug_assert_eq!(weights.len(), n_tokens * k);
    for t in 0..n_tokens {
        let wr = &mut weights[t * k..(t + 1) * k];
        let sum: f32 = wr.iter().sum();
        if sum > 0.0 {
            let inv = 1.0f32 / sum;
            for w in wr.iter_mut() {
                *w *= inv;
            }
        }
    }
}

fn moe_combine_residual_n_tokens_cpu(
    h_mid: &[f32],
    moe_sum: &[f32],
    shared_out: &[f32],
    shared_gate: &[f32],
    hidden_out: &mut [f32],
    n_tokens: usize,
    dim: usize,
) {
    debug_assert_eq!(h_mid.len(), n_tokens * dim);
    debug_assert_eq!(moe_sum.len(), n_tokens * dim);
    debug_assert_eq!(shared_out.len(), n_tokens * dim);
    debug_assert_eq!(shared_gate.len(), n_tokens);
    debug_assert_eq!(hidden_out.len(), n_tokens * dim);
    for t in 0..n_tokens {
        let g = cpu_sigmoid_scalar(shared_gate[t]);
        for i in 0..dim {
            let idx = t * dim + i;
            hidden_out[idx] = h_mid[idx] + moe_sum[idx] + g * shared_out[idx];
        }
    }
}

fn sdpa_causal_tiled_n_tokens_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_out: &mut [f32],
    n_tokens: usize,
    num_heads: usize,
    heads_per_kv: usize,
    head_dim: usize,
    kv_start: usize,
    kv_len_total: usize,
    softmax_scale: f32,
) {
    // Tile the GPU kernel's running-max/denom/partial pattern with
    // a single-pass per-token compute. Causal mask: token t attends
    // to positions [0..kv_start + t + 1).
    let q_stride = num_heads * head_dim;
    let kv_dim = (num_heads / heads_per_kv) * head_dim;
    debug_assert_eq!(q.len(), n_tokens * q_stride);
    debug_assert_eq!(k.len(), kv_len_total * kv_dim);
    debug_assert_eq!(v.len(), kv_len_total * kv_dim);
    debug_assert_eq!(attn_out.len(), n_tokens * q_stride);
    for t in 0..n_tokens {
        let kv_len_t = kv_start + t + 1;
        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_off = t * q_stride + h * head_dim;
            let q_h = &q[q_off..q_off + head_dim];
            let mut scores = vec![0.0f32; kv_len_t];
            let mut max_score = f32::NEG_INFINITY;
            for pos in 0..kv_len_t {
                let k_off = pos * kv_dim + kv_head * head_dim;
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += q_h[i] * k[k_off + i];
                }
                scores[pos] = dot * softmax_scale;
                if scores[pos] > max_score {
                    max_score = scores[pos];
                }
            }
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0f32 / sum_exp;
            for s in scores.iter_mut() {
                *s *= inv_sum;
            }
            let o_off = t * q_stride + h * head_dim;
            for i in 0..head_dim {
                let mut acc = 0.0f32;
                for pos in 0..kv_len_t {
                    let v_off = pos * kv_dim + kv_head * head_dim;
                    acc += scores[pos] * v[v_off + i];
                }
                attn_out[o_off + i] = acc;
            }
        }
    }
}

fn lm_head_cpu(
    hidden: &[f32],
    last_token_row: usize,
    hidden_dim: usize,
    final_norm_weight: &[u8],
    lm_head_weight: &[u8],
    vocab_size: usize,
    logits_out: &mut [f32],
) {
    debug_assert_eq!(logits_out.len(), vocab_size);
    // Final RMSNorm on the last_token_row of hidden.
    let hr =
        &hidden[last_token_row * hidden_dim..(last_token_row + 1) * hidden_dim];
    let mut normed = vec![0.0f32; hidden_dim];
    let mut sum_sq = 0.0f32;
    for &v in hr.iter() {
        sum_sq += v * v;
    }
    let inv_rms = 1.0f32 / (sum_sq / hidden_dim as f32 + 1e-6).sqrt();
    for i in 0..hidden_dim {
        let w_bits = u16::from_le_bytes([
            final_norm_weight[i * 2],
            final_norm_weight[i * 2 + 1],
        ]);
        let w = bf16_to_f32(w_bits);
        normed[i] = hr[i] * inv_rms * w;
    }
    // lm_head matvec: BF16 weight, shape [vocab_size, hidden_dim].
    let lm_w = bytes_as::<u16>(lm_head_weight);
    for r in 0..vocab_size {
        let mut acc = 0.0f32;
        for c in 0..hidden_dim {
            let w = bf16_to_f32(lm_w[r * hidden_dim + c]);
            acc = w.mul_add(normed[c], acc);
        }
        logits_out[r] = acc;
    }
}

// ----------------------------------------------------------------------------
// CpuBackend::Backend impl
// ----------------------------------------------------------------------------

/// Construction inputs for [`CpuBackend::open`]. Carries the
/// [`WeightFile`] (mmap'd weight file); the backend takes ownership.
pub struct CpuConfig {
    pub wf: WeightFile,
}

impl Backend for CpuBackend {
    type Pool = CpuBufferPool;
    type EncodeCtx = ();
    type Config = CpuConfig;
    type Error = GraphError;

    fn open(config: CpuConfig) -> Result<Self, GraphError>
    where
        Self: Sized,
    {
        Ok(Self::new(config.wf))
    }

    fn pool(&self) -> &CpuBufferPool {
        &self.pool
    }
    fn pool_mut(&mut self) -> &mut CpuBufferPool {
        &mut self.pool
    }
    fn begin_encoding(&self) {}
    fn submit_and_wait(&self, _: ()) -> Result<(), GraphError> {
        Ok(())
    }

    fn encode_op(&self, op: &Op, _ctx: &mut ()) {
        match op {
            Op::RmsNormBf16NTokens {
                x,
                weight_off,
                out,
                dim,
                n_tokens,
                eps,
                ..
            } => {
                let dim = *dim as usize;
                let n_tokens = *n_tokens as usize;
                let weight_bytes = self
                    .wf
                    .bytes_at(*weight_off, dim * 2)
                    .expect("weight_off out of mmap");
                let x_buf = self.read_f32(*x);
                let mut out_buf = self.write_f32(*out);
                rms_norm_bf16_n_tokens_cpu(
                    weight_bytes, &x_buf, dim, n_tokens, *eps, &mut out_buf,
                );
            }
            Op::RmsNormQkNTokens {
                x,
                num_k_heads,
                key_dim,
                key_offset_per_token,
                per_token_total,
                n_tokens,
                ..
            } => {
                let mut x_buf = self.write_f32(*x);
                rms_norm_qk_n_tokens_cpu(
                    &mut x_buf,
                    *num_k_heads as usize,
                    *key_dim as usize,
                    *key_offset_per_token as usize,
                    *per_token_total as usize,
                    *n_tokens as usize,
                    1e-6,
                );
            }
            Op::ResidualAddNTokens { a, b, out, .. } => {
                let a_buf = self.read_f32(*a);
                let b_buf = self.read_f32(*b);
                let mut out_buf = self.write_f32(*out);
                residual_add_n_tokens_cpu(&a_buf, &b_buf, &mut out_buf);
            }
            Op::ZeroBuffer { buf, n_bytes, .. } => {
                let mut b = self.write_bytes(*buf);
                b[..*n_bytes as usize].fill(0);
            }
            Op::MatvecNTokens {
                weight,
                input,
                input_off,
                output,
                output_off,
                in_dim,
                out_dim,
                n_tokens,
                ..
            } => {
                let in_dim = *in_dim as usize;
                let out_dim = *out_dim as usize;
                let n_tokens = *n_tokens as usize;
                let bits = weight.bits;
                let input_buf = self.read_f32(*input);
                let mut output_buf = self.write_f32(*output);
                let in_skip = (*input_off as usize) / 4;
                let out_skip = (*output_off as usize) / 4;
                let in_packed_words = in_dim * out_dim / (if bits == 4 { 8 } else { 4 });
                let in_scales = out_dim * (in_dim / GROUP_SIZE);
                let w_bytes = self
                    .wf
                    .bytes_at(weight.w_off, in_packed_words * 4)
                    .expect("weight.w_off out of mmap");
                let s_bytes = self
                    .wf
                    .bytes_at(weight.s_off, in_scales * 2)
                    .expect("weight.s_off out of mmap");
                let b_bytes = self
                    .wf
                    .bytes_at(weight.b_off, in_scales * 2)
                    .expect("weight.b_off out of mmap");
                let packed = bytes_as::<u32>(w_bytes);
                let scales = bytes_as::<u16>(s_bytes);
                let biases = bytes_as::<u16>(b_bytes);
                for t in 0..n_tokens {
                    let x_t =
                        &input_buf[in_skip + t * in_dim..in_skip + (t + 1) * in_dim];
                    let out_t = &mut output_buf
                        [out_skip + t * out_dim..out_skip + (t + 1) * out_dim];
                    if bits == 4 {
                        dequant_matvec_4bit_cpu(
                            packed, scales, biases, in_dim, out_dim, x_t, out_t,
                        )
                        .expect("4-bit matvec");
                    } else if bits == 8 {
                        dequant_matvec_8bit_v3_cpu(
                            packed, scales, biases, in_dim, out_dim, x_t, out_t,
                        )
                        .expect("8-bit matvec");
                    } else {
                        panic!("unsupported MatvecNTokens bits={bits}");
                    }
                }
            }
            Op::SwigluFusedBatched { gate, up, out, .. } => {
                let g = self.read_f32(*gate);
                let u = self.read_f32(*up);
                let mut o = self.write_f32(*out);
                swiglu_fused_cpu(&g, &u, &mut o);
            }
            Op::SdpaCausalTiled {
                q,
                k,
                v,
                attn_out,
                n_tokens,
                num_heads,
                heads_per_kv,
                head_dim,
                kv_start,
                kv_len_total,
                softmax_scale,
                ..
            } => {
                let q_buf = self.read_f32(*q);
                let k_buf = self.read_f32(*k);
                let v_buf = self.read_f32(*v);
                let mut o_buf = self.write_f32(*attn_out);
                sdpa_causal_tiled_n_tokens_cpu(
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &mut o_buf,
                    *n_tokens as usize,
                    *num_heads as usize,
                    *heads_per_kv as usize,
                    *head_dim as usize,
                    *kv_start as usize,
                    *kv_len_total as usize,
                    *softmax_scale,
                );
            }
            Op::MoeSoftmaxTopK {
                logits,
                indices_out,
                weights_out,
                n_tokens,
                n_experts,
                k,
                ..
            } => {
                let logits_buf = self.read_f32(*logits);
                let mut idx_buf = self.write_i32(*indices_out);
                let mut w_buf = self.write_f32(*weights_out);
                moe_softmax_topk_cpu(
                    &logits_buf,
                    &mut idx_buf,
                    &mut w_buf,
                    *n_tokens as usize,
                    *n_experts as usize,
                    *k as usize,
                );
            }
            Op::MoeNormalizeWeights {
                weights, n_tokens, k, ..
            } => {
                let mut w_buf = self.write_f32(*weights);
                moe_normalize_weights_cpu(
                    &mut w_buf,
                    *n_tokens as usize,
                    *k as usize,
                );
            }
            Op::MoeBatchedPermuteFuse {
                expert_refs,
                bucket_input,
                buckets,
                out_sum,
                ..
            } => {
                // Each (BufId, offset) in expert_refs identifies a per-bucket
                // expert blob already uploaded into the pool. We hold all
                // borrows simultaneously — the buffer IDs are distinct, so
                // RefCell borrows don't collide.
                let input_buf = self.read_f32(*bucket_input);
                let mut out_buf = self.write_f32(*out_sum);
                let blobs: Vec<Ref<'_, [u8]>> = expert_refs
                    .iter()
                    .map(|(id, _off)| self.read_bytes(*id))
                    .collect();
                let blob_refs: Vec<&[u8]> =
                    blobs.iter().map(|b| &**b).collect();
                moe_permute_fuse_cpu(
                    &VARIANT, &blob_refs, &input_buf, buckets, &mut out_buf,
                )
                .expect("moe permute-fuse");
            }
            Op::MoeCombineResidualNTokens {
                h_mid,
                moe_sum,
                shared_out,
                shared_gate,
                hidden_out,
                n_tokens,
                dim,
                ..
            } => {
                let h_mid_buf = self.read_f32(*h_mid);
                let moe_sum_buf = self.read_f32(*moe_sum);
                let shared_out_buf = self.read_f32(*shared_out);
                let shared_gate_buf = self.read_f32(*shared_gate);
                let mut hidden_out_buf = self.write_f32(*hidden_out);
                moe_combine_residual_n_tokens_cpu(
                    &h_mid_buf,
                    &moe_sum_buf,
                    &shared_out_buf,
                    &shared_gate_buf,
                    &mut hidden_out_buf,
                    *n_tokens as usize,
                    *dim as usize,
                );
            }
            Op::Conv1dStepNTokens {
                qkv_in,
                conv_state,
                weight_off,
                conv_out,
                conv_dim,
                n_tokens,
                ..
            } => {
                let conv_dim = *conv_dim as usize;
                let n_tokens = *n_tokens as usize;
                // Conv1d kernel size is 4 for Qwen3.6-A3B. Hardcoded
                // here because the kernel arity isn't in the Op for
                // historical reasons (Metal kernel pulls from the
                // bf16 weight tensor size). Refine if needed.
                let kernel_size = 4;
                let weight_bytes = self
                    .wf
                    .bytes_at(*weight_off, conv_dim * kernel_size * 2)
                    .expect("conv1d weight_off out of mmap");
                let qkv_in_buf = self.read_f32(*qkv_in);
                let mut conv_state_buf = self.write_f32(*conv_state);
                let mut conv_out_buf = self.write_f32(*conv_out);
                let mut tmp_out = vec![0.0f32; conv_dim];
                for t in 0..n_tokens {
                    let input =
                        &qkv_in_buf[t * conv_dim..(t + 1) * conv_dim];
                    conv1d_step(
                        &conv_state_buf,
                        input,
                        weight_bytes,
                        conv_dim,
                        kernel_size,
                        &mut tmp_out,
                    )
                    .expect("conv1d_step");
                    conv_out_buf[t * conv_dim..(t + 1) * conv_dim]
                        .copy_from_slice(&tmp_out);
                    // Shift state forward by one step: state =
                    // [state[channels..], input]
                    let cs_len = conv_state_buf.len();
                    conv_state_buf.copy_within(conv_dim..cs_len, 0);
                    conv_state_buf[cs_len - conv_dim..].copy_from_slice(input);
                }
            }
            Op::ComputeDecayBetaNTokens {
                alpha_in,
                beta_in,
                a_log_off,
                dt_bias_off,
                g_decay_out,
                beta_gate_out,
                num_v_heads,
                n_tokens,
                ..
            } => {
                let num_v_heads = *num_v_heads as usize;
                let n_tokens = *n_tokens as usize;
                let a_log_bytes = self
                    .wf
                    .bytes_at(*a_log_off, num_v_heads * 4)
                    .expect("a_log_off out of mmap");
                let dt_bias_bytes = self
                    .wf
                    .bytes_at(*dt_bias_off, num_v_heads * 2)
                    .expect("dt_bias_off out of mmap");
                let a_log: &[f32] = bytemuck_f32(a_log_bytes);
                let alpha_buf = self.read_f32(*alpha_in);
                let beta_buf = self.read_f32(*beta_in);
                let mut g_decay_buf = self.write_f32(*g_decay_out);
                let mut beta_gate_buf = self.write_f32(*beta_gate_out);
                for t in 0..n_tokens {
                    let a =
                        &alpha_buf[t * num_v_heads..(t + 1) * num_v_heads];
                    let b = &beta_buf[t * num_v_heads..(t + 1) * num_v_heads];
                    let g = &mut g_decay_buf
                        [t * num_v_heads..(t + 1) * num_v_heads];
                    let bg = &mut beta_gate_buf
                        [t * num_v_heads..(t + 1) * num_v_heads];
                    compute_decay_beta_cpu(a, b, a_log, dt_bias_bytes, g, bg)
                        .expect("compute_decay_beta");
                }
            }
            Op::GatedDeltaNetStepNTokens {
                state,
                conv_out,
                g_decay,
                beta_gate,
                output,
                num_v_heads,
                value_dim,
                k_heads_per_v,
                n_tokens,
                ..
            } => {
                let v_heads = *num_v_heads as usize;
                let value_dim = *value_dim as usize;
                let k_heads_per_v = *k_heads_per_v as usize;
                let k_heads = v_heads / k_heads_per_v;
                let key_dim = crate::riir::variants::Variant::LINEAR_KEY_DIM;
                let n_tokens = *n_tokens as usize;
                let conv_out_buf = self.read_f32(*conv_out);
                let g_decay_buf = self.read_f32(*g_decay);
                let beta_gate_buf = self.read_f32(*beta_gate);
                let mut state_buf = self.write_f32(*state);
                let mut output_buf = self.write_f32(*output);
                let key_total = VARIANT.linear_total_key();
                let value_total = v_heads * value_dim;
                // q | k | v stacked per token: q and k each
                // `key_total` floats, v `value_total` floats. The
                // v-region size is `num_v_heads * value_dim`, NOT
                // another `key_total` — kept as a single buffer so
                // the Metal kernel can read q/k/v via byte-offset
                // bindings without separate BufIds.
                let per_token_conv = 2 * key_total + value_total;
                for t in 0..n_tokens {
                    let conv_t = &conv_out_buf
                        [t * per_token_conv..(t + 1) * per_token_conv];
                    let q = &conv_t[0..key_total];
                    let k = &conv_t[key_total..2 * key_total];
                    let v =
                        &conv_t[2 * key_total..2 * key_total + value_total];
                    let g = &g_decay_buf
                        [t * v_heads..(t + 1) * v_heads];
                    let bg = &beta_gate_buf
                        [t * v_heads..(t + 1) * v_heads];
                    let mut out_t = vec![0.0f32; value_total];
                    gated_delta_recurrence_supplied(
                        g,
                        bg,
                        q,
                        k,
                        v,
                        v_heads,
                        k_heads,
                        key_dim,
                        value_dim,
                        &mut state_buf,
                        &mut out_t,
                    )
                    .expect("delta net step");
                    output_buf[t * value_total..(t + 1) * value_total]
                        .copy_from_slice(&out_t);
                }
            }
            Op::GatedRmsNormNTokens {
                values,
                z,
                weight_off,
                output,
                num_v_heads,
                value_dim,
                n_tokens,
                eps,
                ..
            } => {
                let value_dim = *value_dim as usize;
                let weight_bytes = self
                    .wf
                    .bytes_at(*weight_off, value_dim * 2)
                    .expect("gated_rms_norm weight_off out of mmap");
                let v_buf = self.read_f32(*values);
                let z_buf = self.read_f32(*z);
                let mut o_buf = self.write_f32(*output);
                gated_rms_norm_n_tokens_cpu(
                    &v_buf,
                    &z_buf,
                    weight_bytes,
                    &mut o_buf,
                    *num_v_heads as usize,
                    value_dim,
                    *n_tokens as usize,
                    *eps,
                );
            }
            Op::LmHead {
                hidden,
                last_token_row,
                logits_out,
                ..
            } => {
                let hidden_dim = VARIANT.hidden_dim;
                let vocab_size = VARIANT.vocab_size;
                let final_norm_bytes = self
                    .wf
                    .tensor_bytes("model.norm.weight")
                    .expect("final norm tensor missing");
                let lm_head_bytes = self
                    .wf
                    .tensor_bytes("lm_head.weight")
                    .expect("lm_head tensor missing");
                let hidden_buf = self.read_f32(*hidden);
                let mut logits_buf = self.write_f32(*logits_out);
                lm_head_cpu(
                    &hidden_buf,
                    *last_token_row as usize,
                    hidden_dim,
                    final_norm_bytes,
                    lm_head_bytes,
                    vocab_size,
                    &mut logits_buf,
                );
            }
        }
    }
}

fn bytemuck_f32(b: &[u8]) -> &[f32] {
    let (head, body, tail) = unsafe { b.align_to::<f32>() };
    assert!(
        head.is_empty() && tail.is_empty(),
        "byte slice not f32-aligned (head={}, tail={})",
        head.len(),
        tail.len()
    );
    body
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn cpu_pool_alloc_returns_sequential_bufids() {
        let mut p = CpuBufferPool::new();
        let a = p.alloc(64, "a", false).unwrap();
        let b = p.alloc(128, "b", true).unwrap();
        let c = p.alloc(32, "c", false).unwrap();
        assert_eq!(a, BufId(0));
        assert_eq!(b, BufId(1));
        assert_eq!(c, BufId(2));
        assert_eq!(p.physical_buffer_count(), 3);
    }

    #[test]
    fn cpu_pool_upload_download_round_trips() {
        let mut p = CpuBufferPool::new();
        let id = p.alloc(16, "x", false).unwrap();
        let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        p.upload(id, &payload).unwrap();
        let mut out = vec![0u8; 16];
        p.download(id, &mut out).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn cpu_pool_upload_rejects_size_mismatch() {
        let mut p = CpuBufferPool::new();
        let id = p.alloc(16, "x", false).unwrap();
        let too_big = vec![0u8; 17];
        match p.upload(id, &too_big) {
            Err(GraphError::SizeMismatch { label, expected, actual }) => {
                assert_eq!(label, "x");
                assert_eq!(expected, 16);
                assert_eq!(actual, 17);
            }
            _ => panic!("expected SizeMismatch"),
        }
    }

    #[test]
    fn cpu_pool_reset_transient_keeps_persistent_prefix() {
        let mut p = CpuBufferPool::new();
        let _persistent_a = p.alloc(64, "kv_a", true).unwrap();
        let _persistent_b = p.alloc(64, "kv_b", true).unwrap();
        let _transient = p.alloc(32, "intermed", false).unwrap();
        assert_eq!(p.physical_buffer_count(), 3);
        p.reset_transient();
        assert_eq!(p.physical_buffer_count(), 2);
        // The persistent ids are still valid.
        assert_eq!(p.label(BufId(0)), "kv_a");
        assert_eq!(p.label(BufId(1)), "kv_b");
    }

    #[test]
    fn cpu_pool_handle_returns_refcell_with_zeros() {
        let mut p = CpuBufferPool::new();
        let id = p.alloc(12, "z", false).unwrap();
        let handle = p.handle(id);
        let borrowed = handle.borrow();
        assert_eq!(&*borrowed, &[0u8; 12]);
    }
}
