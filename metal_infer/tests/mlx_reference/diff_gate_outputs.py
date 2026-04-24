#!/usr/bin/env python3
"""Diff moeflux vs MLX dumps across all 40 layers.

Summarizes the per-layer divergence: h_post RMS diff, gate_logits diff,
gate_probs diff, top-K set overlap, weight diff.

Expects files named <prefix>_l<N>.bin in both dirs.
"""
import struct
import sys
import numpy as np


def load(path):
    with open(path, "rb") as f:
        hdr = f.read(16)
        HIDDEN, NEXP, K, LIDX = struct.unpack("<4i", hdr)
        h_post = np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
        logits = np.frombuffer(f.read(NEXP * 4), dtype=np.float32).copy()
        probs = np.frombuffer(f.read(NEXP * 4), dtype=np.float32).copy()
        inds = np.frombuffer(f.read(K * 4), dtype=np.int32).copy()
        weights = np.frombuffer(f.read(K * 4), dtype=np.float32).copy()
    return dict(
        HIDDEN=HIDDEN, NEXP=NEXP, K=K, LIDX=LIDX,
        h_post=h_post, logits=logits, probs=probs,
        inds=inds, weights=weights,
    )


def main():
    NLAYERS = 40
    mf_prefix = "/tmp/moeflux_dump"
    mx_prefix = "/tmp/mlx_dump"

    print(f"{'L':>3} {'h_post max':>10} {'h_post rel':>10} "
          f"{'logit max':>10} {'probs max':>10} "
          f"{'topK∩':>6} {'w_max':>8} {'argmax_match':>13}")
    print("-" * 90)

    first_big_divergence = None
    for L in range(NLAYERS):
        mf = load(f"{mf_prefix}_l{L}.bin")
        mx = load(f"{mx_prefix}_l{L}.bin")

        h_diff = np.abs(mf['h_post'] - mx['h_post'])
        # Relative RMS: rms(diff) / rms(mlx)
        mx_rms = np.sqrt((mx['h_post'] ** 2).mean())
        rel = np.sqrt((h_diff ** 2).mean()) / max(mx_rms, 1e-9)

        logit_diff = np.abs(mf['logits'] - mx['logits']).max()
        probs_diff = np.abs(mf['probs'] - mx['probs']).max()

        mf_set = set(mf['inds'].tolist())
        mx_set = set(mx['inds'].tolist())
        overlap = len(mf_set & mx_set)

        # top-1 argmax comparison (the most likely expert)
        mf_top1 = int(mf['inds'][np.argmax(mf['weights'])])
        mx_top1 = int(mx['inds'][np.argmax(mx['weights'])])
        argmax_match = "✓" if mf_top1 == mx_top1 else f"✗ {mf_top1} vs {mx_top1}"

        # weight diff after sorting both desc
        mf_sorted_w = np.sort(mf['weights'])[::-1]
        mx_sorted_w = np.sort(mx['weights'])[::-1]
        w_max = np.abs(mf_sorted_w - mx_sorted_w).max()

        print(f"{L:>3} {h_diff.max():>10.4f} {rel:>10.4f} "
              f"{logit_diff:>10.4f} {probs_diff:>10.6f} "
              f"{overlap:>6d} {w_max:>8.4f} {argmax_match:>13}")

        if first_big_divergence is None and rel > 0.1:
            first_big_divergence = L

    if first_big_divergence is not None:
        print(f"\nFirst layer with h_post relative RMS diff > 10%: L={first_big_divergence}")
    else:
        print(f"\nAll layers have h_post relative RMS diff <= 10%")


if __name__ == "__main__":
    main()
