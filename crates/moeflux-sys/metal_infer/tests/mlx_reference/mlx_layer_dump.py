#!/usr/bin/env python3
"""MLX reference dump for Qwen3.6-35B-A3B — per-layer MLP input + gate output.

For every decoder layer, captures the MoE block's intermediate tensors at
position 0 of a 4-token prompt ("The quick brown fox") and writes one
binary file per layer in the same format as moeflux's MOEFLUX_DUMP_L0.

Usage:
  uv run --with mlx --with mlx-lm python3 mlx_l0_dump.py <out_prefix>
  -> writes <out_prefix>_l0.bin, <out_prefix>_l1.bin, ... <out_prefix>_l39.bin

Binary format per file:
  i32 HIDDEN_DIM, i32 NUM_EXPERTS, i32 K, i32 layer_idx
  f32[HIDDEN_DIM]      h_post  -- MLP input after attn+post-attn norm
  f32[NUM_EXPERTS]     gate_logits (pre-softmax)
  f32[NUM_EXPERTS]     gate_probs (post-softmax)
  i32[K]               expert_indices (sorted desc by weight)
  f32[K]               expert_weights (normalized, sorted desc by weight)
"""
import sys
import struct
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3_next, qwen3_5


def main():
    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mlx_l0_dump"
    model_path = "/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-mlx-4bit"

    print(f"[MLX_L0] loading {model_path}", flush=True)
    model, tokenizer = load(model_path)

    expected = [760, 3841, 13477, 37550]
    actual = tokenizer.encode("The quick brown fox", add_special_tokens=False)
    print(f"[MLX_L0] tokens: {actual} (expected {expected})", flush=True)

    captured_layers = {}
    layer_counter = [0]
    decoder_call_count = [0]
    layer_inputs = {}
    l0_components = {}
    orig_moe_call = qwen3_next.Qwen3NextSparseMoeBlock.__call__
    orig_decoder_call = qwen3_5.DecoderLayer.__call__

    def to_np(a, dtype=mx.float32):
        return np.asarray(a.astype(dtype))

    def patched_moe_call(self, x):
        lidx = layer_counter[0]
        layer_counter[0] += 1
        # Capture every layer (at pos=0)
        captured = {}
        captured['x_in'] = to_np(x)
        gates = self.gate(x)
        captured['gate_logits'] = to_np(gates)
        gates_sm = mx.softmax(gates, axis=-1, precise=True)
        captured['gate_probs'] = to_np(gates_sm)
        k = self.top_k
        inds = mx.argpartition(gates_sm, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates_sm, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        captured['topk_inds'] = np.asarray(inds.astype(mx.int32))
        captured['topk_weights'] = to_np(scores)
        captured['top_k'] = k
        captured_layers[lidx] = captured
        # Continue the forward exactly as MLX does
        y_per_expert = self.switch_mlp(x, inds)  # shape (batch, seq, K, hidden)
        moe_out = (y_per_expert * scores[..., None]).sum(axis=-2)
        shared_raw = self.shared_expert(x)
        gate_pre = self.shared_expert_gate(x)  # shape (1, seq, 1)
        shared_gated = mx.sigmoid(gate_pre) * shared_raw
        mlp_out = moe_out + shared_gated
        if lidx == 0:
            # Capture individual components for layer-0 component diff
            l0_components['y_per_expert'] = to_np(y_per_expert)
            l0_components['moe_out'] = to_np(moe_out)
            l0_components['shared_raw'] = to_np(shared_raw)
            l0_components['shared_gate_pre'] = to_np(gate_pre)
            l0_components['shared_gated'] = to_np(shared_gated)
            l0_components['mlp_out'] = to_np(mlp_out)
        return mlp_out

    def patched_decoder_call(self, x, mask=None, cache=None):
        lidx = decoder_call_count[0]
        decoder_call_count[0] += 1
        layer_inputs[lidx] = to_np(x)
        if lidx == 0:
            # For layer 0 specifically, capture h_mid (= x + r) and final out
            # by reproducing DecoderLayer's forward manually with capture
            if self.is_linear:
                r = self.linear_attn(self.input_layernorm(x), mask, cache)
            else:
                r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r  # h_mid
            mlp_out_here = self.mlp(self.post_attention_layernorm(h))
            out = h + mlp_out_here
            l0_components['h_mid'] = to_np(h)
            l0_components['out'] = to_np(out)
            return out
        return orig_decoder_call(self, x, mask=mask, cache=cache)

    qwen3_next.Qwen3NextSparseMoeBlock.__call__ = patched_moe_call
    qwen3_5.DecoderLayer.__call__ = patched_decoder_call

    print(f"[MLX_L0] forwarding {len(actual)} tokens...", flush=True)
    input_ids = mx.array([actual])
    logits = model(input_ids)
    mx.eval(logits)
    print(f"[MLX_L0] logits shape: {logits.shape}; captured {len(captured_layers)} MoE layers, "
          f"{len(layer_inputs)} layer inputs", flush=True)

    # Dump per-layer input hidden states (= previous layer output)
    for lidx in sorted(layer_inputs.keys()):
        x_in = layer_inputs[lidx][0, 0].astype(np.float32)  # (1, seq, hidden) -> pos=0
        HIDDEN_DIM = int(x_in.shape[-1])
        out_path = f"{out_prefix}_l{lidx}_in.bin"
        with open(out_path, "wb") as f:
            f.write(struct.pack("<2i", HIDDEN_DIM, lidx))
            f.write(x_in.tobytes())
        rms = float(np.sqrt((x_in ** 2).mean()))
        print(f"[MLX_L0] wrote {out_path}  rms={rms:.4f}", flush=True)

    # Dump layer-0 MoE components
    if l0_components:
        HIDDEN_DIM = int(l0_components['h_mid'].shape[-1])
        K = int(l0_components['y_per_expert'].shape[-2])
        comp_path = f"{out_prefix}_l0_components.bin"
        with open(comp_path, "wb") as f:
            f.write(struct.pack("<3i", HIDDEN_DIM, K, 0))
            # h_mid (pos=0)
            f.write(l0_components['h_mid'][0, 0].astype(np.float32).tobytes())
            # final out (pos=0)
            f.write(l0_components['out'][0, 0].astype(np.float32).tobytes())
            # shared_raw (pos=0)
            f.write(l0_components['shared_raw'][0, 0].astype(np.float32).tobytes())
            # per-expert outputs — MLX order is determined by inds; reorder to match
            # moeflux's expert_indices order. We'll write raw MLX order here and
            # the diff script will align by index.
            y_pe = l0_components['y_per_expert'][0, 0]  # (K, HIDDEN)
            for k in range(K):
                f.write(y_pe[k].astype(np.float32).tobytes())
            # MLX's top-K inds and weights (same as captured_layers[0], for alignment)
            mlx_inds = captured_layers[0]['topk_inds'][0, 0].astype(np.int32)
            mlx_w = captured_layers[0]['topk_weights'][0, 0].astype(np.float32)
            f.write(mlx_inds.tobytes())
            f.write(mlx_w.tobytes())
            # Shared gate pre-sigmoid (pos=0)
            gate_pre = float(l0_components['shared_gate_pre'][0, 0, 0])
            f.write(struct.pack("<f", gate_pre))
        print(f"[MLX_L0] wrote {comp_path}  gate_pre={gate_pre:.6f}", flush=True)

    for lidx in sorted(captured_layers.keys()):
        cap = captured_layers[lidx]
        HIDDEN_DIM = int(cap['x_in'].shape[-1])
        NUM_EXPERTS = int(cap['gate_probs'].shape[-1])
        K = int(cap['top_k'])

        x_in = cap['x_in'][0, 0].astype(np.float32)
        logits_p0 = cap['gate_logits'][0, 0].astype(np.float32)
        probs_p0 = cap['gate_probs'][0, 0].astype(np.float32)
        inds_p0 = cap['topk_inds'][0, 0]
        w_p0 = cap['topk_weights'][0, 0]

        order = np.argsort(w_p0)[::-1]
        sorted_inds = inds_p0[order].astype(np.int32)
        sorted_w = w_p0[order].astype(np.float32)

        out_path = f"{out_prefix}_l{lidx}.bin"
        with open(out_path, "wb") as f:
            f.write(struct.pack("<4i", HIDDEN_DIM, NUM_EXPERTS, K, lidx))
            f.write(x_in.tobytes())
            f.write(logits_p0.tobytes())
            f.write(probs_p0.tobytes())
            f.write(sorted_inds.tobytes())
            f.write(sorted_w.tobytes())

        top_str = ",".join(f"{int(sorted_inds[i])}({float(sorted_w[i]):.3f})"
                           for i in range(K))
        print(f"[MLX_L0] wrote {out_path}  top-{K}: [{top_str}]", flush=True)


if __name__ == "__main__":
    main()
