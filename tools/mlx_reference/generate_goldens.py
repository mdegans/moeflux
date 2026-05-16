#!/usr/bin/env python3
"""Capture MLX golden top-K logits for a variant, for use as a Rust regression
test fixture.

Output format: plain text, one line per token, pipe-separated:

    # variant=<variant> prompt=<prompt> argmax=<id> vocab=<V>
    # rank | token_id | logit
    0|33075|18.000000
    1|25174|15.375000
    ...

Human-readable, diffable, trivially version-controlled. The test checks top-K
set overlap against this file plus argmax match.

Usage:
    uv run --with mlx --with mlx-lm python3 generate_goldens.py \\
        --model /path/to/mlx-model \\
        --variant qwen3-6-35b-a3b \\
        --out   /path/to/moeflux/crates/moeflux/tests/fixtures/mlx_golden_<variant>.txt \\
        [--top 200] [--prompt "The quick brown fox"]
"""
import argparse
import sys
import numpy as np
import mlx.core as mx
from mlx_lm import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to MLX-format model directory")
    p.add_argument("--variant", required=True, help="Variant slug (e.g. qwen3-6-35b-a3b)")
    p.add_argument("--out", required=True, help="Output fixture file path")
    p.add_argument("--prompt", default="The quick brown fox")
    p.add_argument("--top", type=int, default=200)
    args = p.parse_args()

    print(f"[golden] loading {args.model}", flush=True)
    model, tokenizer = load(args.model)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f"[golden] prompt={args.prompt!r} tokens={ids}", flush=True)

    input_ids = mx.array([ids])
    logits = model(input_ids)
    mx.eval(logits)

    # Last position predicts the next token after the prompt.
    last = np.asarray(logits[0, -1].astype(mx.float32))
    VOCAB = int(last.shape[-1])
    order = np.argsort(last)[::-1][: args.top]
    argmax = int(order[0])
    print(f"[golden] vocab={VOCAB} argmax={argmax} "
          f"({tokenizer.decode([argmax])!r}) logit={last[argmax]:.4f}",
          flush=True)

    with open(args.out, "w") as f:
        f.write(f"# variant={args.variant} prompt={args.prompt!r} "
                f"argmax={argmax} vocab={VOCAB} top={args.top}\n")
        f.write(f"# tokens={ids}\n")
        f.write(f"# rank | token_id | logit\n")
        for rank, tok in enumerate(order):
            f.write(f"{rank}|{int(tok)}|{float(last[int(tok)]):.6f}\n")
    print(f"[golden] wrote {args.out} ({args.top} entries)", flush=True)


if __name__ == "__main__":
    main()
