#!/usr/bin/env python3
"""Diff per-layer INPUT hidden states (moeflux vs MLX).

layer N input = layer N-1 output (or embedding for N=0).
"""
import struct
import numpy as np


def load(path):
    with open(path, "rb") as f:
        HIDDEN, LIDX = struct.unpack("<2i", f.read(8))
        h = np.frombuffer(f.read(HIDDEN * 4), dtype=np.float32).copy()
    return h, HIDDEN, LIDX


def main():
    NLAYERS = 40
    print(f"{'L':>3} {'mf_rms':>8} {'mx_rms':>8} {'max_diff':>10} "
          f"{'mean_diff':>10} {'rel_rms':>8} {'cosine':>8}")
    print("-" * 70)

    for L in range(NLAYERS):
        mf, _, _ = load(f"/tmp/moeflux_dump_l{L}_in.bin")
        mx, _, _ = load(f"/tmp/mlx_dump_l{L}_in.bin")

        mf_rms = np.sqrt((mf ** 2).mean())
        mx_rms = np.sqrt((mx ** 2).mean())
        d = np.abs(mf - mx)
        rel_rms = np.sqrt((d ** 2).mean()) / max(mx_rms, 1e-9)
        cos = np.dot(mf, mx) / (np.linalg.norm(mf) * np.linalg.norm(mx) + 1e-12)
        print(f"{L:>3} {mf_rms:>8.4f} {mx_rms:>8.4f} {d.max():>10.4f} "
              f"{d.mean():>10.4f} {rel_rms:>8.4f} {cos:>8.6f}")


if __name__ == "__main__":
    main()
