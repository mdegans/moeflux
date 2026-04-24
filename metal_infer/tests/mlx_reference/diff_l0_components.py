#!/usr/bin/env python3
"""Diff layer-0 MoE components (moeflux vs MLX)."""
import struct
import numpy as np


def load_mf(path="/tmp/moeflux_dump_l0_components.bin"):
    with open(path, "rb") as f:
        HIDDEN, K, _ = struct.unpack("<3i", f.read(12))
        hid = np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
        out = np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
        shared_raw = np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
        expert_outs = [
            np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
            for _ in range(K)
        ]
        weights = np.frombuffer(f.read(K * 4), dtype=np.float32).copy()
        gate_score = struct.unpack("<f", f.read(4))[0]
    return dict(HIDDEN=HIDDEN, K=K, h_mid=hid, out=out, shared_raw=shared_raw,
                expert_outs=expert_outs, weights=weights, gate_score=gate_score)


def load_mx(path="/tmp/mlx_dump_l0_components.bin"):
    with open(path, "rb") as f:
        HIDDEN, K, _ = struct.unpack("<3i", f.read(12))
        hid = np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
        out = np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
        shared_raw = np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
        # expert_outs in MLX order matches MLX's topk inds
        y_per_expert = [
            np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
            for _ in range(K)
        ]
        inds = np.frombuffer(f.read(K * 4), dtype=np.int32).copy()
        weights = np.frombuffer(f.read(K * 4), dtype=np.float32).copy()
        gate_pre = struct.unpack("<f", f.read(4))[0]
    return dict(HIDDEN=HIDDEN, K=K, h_mid=hid, out=out, shared_raw=shared_raw,
                y_per_expert=y_per_expert, inds=inds, weights=weights, gate_pre=gate_pre)


def rel_diff(a, b, label):
    d = np.abs(a - b)
    rms_b = np.sqrt((b ** 2).mean())
    rel = np.sqrt((d ** 2).mean()) / max(rms_b, 1e-9)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    rms_a = np.sqrt((a ** 2).mean())
    print(f"  {label:>18}: a_rms={rms_a:.4f} b_rms={rms_b:.4f} "
          f"max_diff={d.max():.4f} rel_rms={rel:.4f} cos={cos:.6f}")
    return d, rel, cos


def main():
    mf = load_mf()
    mx = load_mx()

    print("\n== component diff (moeflux = a, MLX = b) ==\n")

    # 1) h_mid — post-attn residual, input to MLP block (via norm)
    print("h_mid (post-attn residual, = attention output + embedding):")
    rel_diff(mf['h_mid'], mx['h_mid'], "h_mid")

    # 2) out — final layer-0 output (= h_mid + mlp_out)
    print("\nout (final layer-0 output):")
    rel_diff(mf['out'], mx['out'], "out")

    # 3) mlp_out = out - h_mid
    mf_mlp = mf['out'] - mf['h_mid']
    mx_mlp = mx['out'] - mx['h_mid']
    print("\nmlp_out (= out - h_mid, MLP block contribution):")
    rel_diff(mf_mlp, mx_mlp, "mlp_out")

    # 4) shared_raw — raw shared-expert output (pre-gate)
    print("\nshared_raw (shared expert output BEFORE sigmoid gate):")
    rel_diff(mf['shared_raw'], mx['shared_raw'], "shared_raw")

    # 5) shared gate pre-sigmoid scalar
    print(f"\nshared gate pre-sigmoid: moeflux={mf['gate_score']:.6f}  "
          f"MLX={mx['gate_pre']:.6f}  diff={abs(mf['gate_score'] - mx['gate_pre']):.6f}")
    mf_sig = 1.0 / (1.0 + np.exp(-mf['gate_score']))
    mx_sig = 1.0 / (1.0 + np.exp(-mx['gate_pre']))
    print(f"  after sigmoid:       moeflux={mf_sig:.6f}  MLX={mx_sig:.6f}")

    # 6) shared_gated = sigmoid(gate) * shared_raw
    mf_sg = mf_sig * mf['shared_raw']
    mx_sg = mx_sig * mx['shared_raw']
    print("\nshared_gated (sigmoid(gate) * shared_raw):")
    rel_diff(mf_sg, mx_sg, "shared_gated")

    # 7) moe_out = mlp_out - shared_gated
    mf_moe = mf_mlp - mf_sg
    mx_moe = mx_mlp - mx_sg
    print("\nmoe_out (= mlp_out - shared_gated, weighted sum of expert outputs):")
    rel_diff(mf_moe, mx_moe, "moe_out")

    # 8) Reconstruct moe_out from per-expert outputs + weights
    # Moeflux's expert order is in moeflux topk output; MLX's in MLX topk output.
    # Need to align by expert_id.
    # Moeflux weights are the normalized top-K weights (already in cpu_normalize_weights order).
    # Let's just reconstruct and check it matches.
    mf_reconstructed = np.zeros_like(mf['h_mid'])
    for k in range(mf['K']):
        mf_reconstructed += mf['weights'][k] * mf['expert_outs'][k]
    mx_reconstructed = np.zeros_like(mx['h_mid'])
    for k in range(mx['K']):
        mx_reconstructed += mx['weights'][k] * mx['y_per_expert'][k]

    print("\n== reconstructed moe_out from per-expert outputs ==")
    print("moeflux reconstructed vs moe_out (self-consistency):")
    rel_diff(mf_reconstructed, mf_moe, "mf self-check")
    print("MLX reconstructed vs moe_out (self-consistency):")
    rel_diff(mx_reconstructed, mx_moe, "mx self-check")

    print("\nmoeflux_reconstructed vs MLX_reconstructed:")
    rel_diff(mf_reconstructed, mx_reconstructed, "reconstructed")

    # 9) Align and compare per-expert outputs
    # MLX weights/inds are not in a specific order (argpartition is unsorted).
    # Moeflux's cpu_topk order might differ. We need to match by expert_id.
    # But we don't have moeflux's expert_indices in this dump (!) Let me check...
    # Hmm, we don't. Would have to re-dump that. For now, compute per-expert
    # mean and count how many are "small" vs "large".
    print("\n== per-expert output norms (sorted) ==")
    mf_norms = sorted([float(np.linalg.norm(e)) for e in mf['expert_outs']], reverse=True)
    mx_norms = sorted([float(np.linalg.norm(e)) for e in mx['y_per_expert']], reverse=True)
    print(f"  moeflux expert norms: {['%.3f' % n for n in mf_norms]}")
    print(f"  MLX expert norms:     {['%.3f' % n for n in mx_norms]}")


if __name__ == "__main__":
    main()
