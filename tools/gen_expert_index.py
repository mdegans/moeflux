#!/usr/bin/env python3
"""Generate expert_index.json for an MLX-quantized Qwen MoE model.

This produces the input that `repack_experts.py` consumes: a JSON map
from layer index to the absolute safetensors byte offsets of each
expert component (gate/up/down × weight/scales/biases).

It works against any `mlx_lm convert`-produced MLX model that uses
the `switch_mlp` module for its experts — i.e. `qwen3_5_moe`,
`qwen3_next`, and any other family where MLX packs all experts of
a given projection into a single tensor of shape
`[num_experts, out_dim, ...]`.

Standalone — only depends on the stdlib.

Usage:
    python gen_expert_index.py --mlx-path <dir> --out expert_index.json

The output schema matches upstream's hand-written A17B index, so
`repack_experts.py` consumes it unmodified:

    {
      "model_path": "<mlx dir>",
      "expert_reads": {
        "<layer>": {
          "gate_proj.weight": { file, abs_offset, expert_stride,
                                 expert_size, total_size, shape },
          "gate_proj.scales": { ... },
          "gate_proj.biases": { ... },
          "up_proj.weight":   { ... },
          ...
        },
        ...
      }
    }
"""

import argparse
import json
import os
import struct
import sys


COMPONENTS = (
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
    "down_proj.weight", "down_proj.scales", "down_proj.biases",
)

# Tensor-name prefix pattern used by mlx_lm for MoE experts. The "{L}"
# placeholder is replaced with the layer index. This matches Qwen's
# HF -> MLX conversion output; if a new architecture lands with a
# different prefix, add a fallback here.
TENSOR_PREFIXES = (
    "language_model.model.layers.{L}.mlp.switch_mlp.{C}",
    "model.layers.{L}.mlp.switch_mlp.{C}",  # text-only MLX outputs
)


def read_safetensors_header(path):
    """Return (data_base_offset, header_dict) for a .safetensors file."""
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"{path}: file too short to contain header length")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path}: truncated header")
    return 8 + header_len, json.loads(header_bytes)


def find_tensor_key(header, layer_idx, component):
    """Return the full tensor-name key in the safetensors header for
    (layer, component), trying each known prefix pattern."""
    for template in TENSOR_PREFIXES:
        name = template.format(L=layer_idx, C=component)
        if name in header:
            return name
    return None


def build_index(mlx_path):
    index_path = os.path.join(mlx_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        sys.exit(f"error: {index_path} not found — is this an MLX model dir?")

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Cache each shard's header (parse once per shard).
    header_cache = {}
    def header_for(shard):
        if shard not in header_cache:
            header_cache[shard] = read_safetensors_header(
                os.path.join(mlx_path, shard))
        return header_cache[shard]

    # Probe layer 0 to figure out num_experts + num_layers.
    probe_shard = None
    for tname, shard in weight_map.items():
        if ".layers.0.mlp.switch_mlp.gate_proj.weight" in tname:
            probe_shard = shard
            probe_name = tname
            break
    if probe_shard is None:
        sys.exit("error: no switch_mlp.gate_proj.weight found for layer 0 — "
                 "this does not look like an MLX MoE model.")

    data_base, header = header_for(probe_shard)
    num_experts = header[probe_name]["shape"][0]

    # Count layers by scanning tensor names.
    layer_ids = set()
    for tname in weight_map:
        if ".mlp.switch_mlp.gate_proj.weight" in tname:
            # Extract layer index. Works for both prefix patterns because
            # we just look for ".layers.<N>." substring.
            import re
            m = re.search(r"\.layers\.(\d+)\.", tname)
            if m:
                layer_ids.add(int(m.group(1)))
    num_layers = max(layer_ids) + 1 if layer_ids else 0

    print(f"detected: num_experts={num_experts}, num_layers={num_layers}",
          file=sys.stderr)

    expert_reads = {}
    for layer in range(num_layers):
        layer_entry = {}
        for component in COMPONENTS:
            data_base, header = None, None
            shard = None
            key = None
            # Try prefixes to find which shard this tensor lives in.
            for template in TENSOR_PREFIXES:
                candidate = template.format(L=layer, C=component)
                if candidate in weight_map:
                    shard = weight_map[candidate]
                    key = candidate
                    break
            if key is None:
                sys.exit(f"error: layer {layer} component {component} "
                         f"not found in weight_map.")

            data_base, header = header_for(shard)
            meta = header[key]
            ds, de = meta["data_offsets"]
            total_size = de - ds
            if total_size % num_experts != 0:
                sys.exit(f"error: {key} size {total_size} not divisible "
                         f"by {num_experts} experts")
            per_expert = total_size // num_experts

            layer_entry[component] = {
                "file": shard,
                "abs_offset": data_base + ds,
                "expert_stride": per_expert,
                "expert_size": per_expert,
                "total_size": total_size,
                "shape": list(meta["shape"]),
            }
        expert_reads[str(layer)] = layer_entry

    return {
        "model_path": mlx_path,
        "expert_reads": expert_reads,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlx-path", required=True,
                    help="Path to an mlx_lm convert output directory.")
    ap.add_argument("--out", required=True,
                    help="Where to write the generated JSON.")
    args = ap.parse_args()

    index = build_index(args.mlx_path)
    with open(args.out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {args.out}: {len(index['expert_reads'])} layers",
          file=sys.stderr)


if __name__ == "__main__":
    main()
