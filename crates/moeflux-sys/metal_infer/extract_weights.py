#!/usr/bin/env python3
"""
extract_weights.py — Extract all non-expert weights from Qwen3.5-397B-A17B-4bit
into a single binary file that the C inference engine can mmap.

Outputs:
  - model_weights.bin: binary blob containing all non-expert weight tensors
  - model_weights.json: manifest describing each tensor's location, shape, dtype

The binary format is simple:
  - Tensors are packed contiguously, 64-byte aligned
  - Each tensor is stored in its native format (U32 packed, BF16 as uint16, F32)
  - The JSON manifest maps tensor names to {offset, size, shape, dtype}

Usage:
    python extract_weights.py [--model PATH] [--output DIR]
"""

import json
import struct
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
import re
import numpy as np


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def main():
    parser = argparse.ArgumentParser(description='Extract non-expert weights to binary')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to MLX-converted model directory '
                             '(containing model.safetensors.index.json + config.json)')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for model_weights.bin and .json')
    parser.add_argument('--include-experts', action='store_true',
                        help='Also extract expert weights (huge, not recommended)')
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the weight index
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Load model config — drives the manifest's "config" block instead
    # of hardcoded A17B values. qwen3_5_moe nests the text-model config
    # under "text_config"; fall back to the root for older formats.
    config_path = model_path / 'config.json'
    if not config_path.exists():
        print(f"ERROR: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        hf_config = json.load(f)
    text_config = hf_config.get('text_config', hf_config)

    # Quantization config: resolve a per-tensor `bits` for every quantized
    # tensor. MLX stores defaults at the top level of the `quantization`
    # block and per-tensor overrides as nested dicts keyed by the full
    # (unsanitized) tensor-base name, e.g.:
    #   "quantization": {
    #     "group_size": 64, "bits": 4,
    #     "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
    #     ...
    #   }
    # A3B has 8-bit overrides on `mlp.gate` and `mlp.shared_expert_gate`;
    # everything else is 4-bit. A17B has no overrides.
    q_block = hf_config.get('quantization', {}) or {}
    default_bits = q_block.get('bits', 4)
    default_group_size = q_block.get('group_size', 64)
    # base-name (no `.weight`/`.scales`/`.biases` suffix) -> bits
    per_tensor_bits = {}
    for k, v in q_block.items():
        if isinstance(v, dict) and 'bits' in v:
            per_tensor_bits[k] = int(v['bits'])

    def lookup_bits(original_name: str) -> int:
        """Return bits for a quantized tensor. `original_name` is the
        unsanitized name as it appears in the safetensors index."""
        # Strip trailing .weight / .scales / .biases to get the base key
        base = original_name
        for suf in ('.weight', '.scales', '.biases'):
            if base.endswith(suf):
                base = base[:-len(suf)]
                break
        return per_tensor_bits.get(base, default_bits)

    # Filter: keep only language_model weights, skip vision_tower
    # Also skip expert weights (switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases})
    # unless --include-experts is set
    expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}  # name -> filename
    skipped_expert = 0
    skipped_vision = 0

    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            skipped_vision += 1
            continue
        if not args.include_experts and expert_pattern.search(name):
            skipped_expert += 1
            continue
        tensors_to_extract[name] = filename

    print(f"Model: {model_path}")
    print(f"Total weights in index: {len(weight_map)}")
    print(f"Skipped vision: {skipped_vision}")
    print(f"Skipped expert: {skipped_expert}")
    print(f"Extracting: {len(tensors_to_extract)} tensors")

    # Group by shard file for sequential I/O
    by_file = defaultdict(list)
    for name, filename in tensors_to_extract.items():
        by_file[filename].append(name)

    # Parse headers and plan layout
    print("\nParsing safetensors headers...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))

    # Sanitize tensor names: remove "language_model." prefix for the C engine
    def sanitize_name(name):
        if name.startswith("language_model."):
            return name[len("language_model."):]
        return name

    # Plan the output layout
    # Sort tensors for deterministic output
    all_tensors = []  # (sanitized_name, original_name, filename)
    for name in sorted(tensors_to_extract.keys()):
        san_name = sanitize_name(name)
        all_tensors.append((san_name, name, tensors_to_extract[name]))

    # Write binary file
    bin_path = output_dir / 'model_weights.bin'
    # Pull config from the model's HuggingFace config.json; each key
    # maps 1:1 to a field the C engine expects.
    rope_params = text_config.get('rope_parameters', {})
    cfg_out = {
        "hidden_size":                     text_config['hidden_size'],
        "num_hidden_layers":               text_config['num_hidden_layers'],
        "num_attention_heads":             text_config['num_attention_heads'],
        "num_key_value_heads":             text_config['num_key_value_heads'],
        "head_dim":                        text_config['head_dim'],
        "vocab_size":                      text_config['vocab_size'],
        "rms_norm_eps":                    text_config.get('rms_norm_eps', 1e-6),
        "num_experts":                     text_config['num_experts'],
        "num_experts_per_tok":             text_config['num_experts_per_tok'],
        "moe_intermediate_size":           text_config['moe_intermediate_size'],
        "shared_expert_intermediate_size": text_config['shared_expert_intermediate_size'],
        "full_attention_interval":         text_config['full_attention_interval'],
        "linear_num_value_heads":          text_config['linear_num_value_heads'],
        "linear_num_key_heads":            text_config['linear_num_key_heads'],
        "linear_key_head_dim":             text_config['linear_key_head_dim'],
        "linear_value_head_dim":           text_config['linear_value_head_dim'],
        "linear_conv_kernel_dim":          text_config['linear_conv_kernel_dim'],
        "partial_rotary_factor":           (rope_params.get('partial_rotary_factor')
                                            or text_config.get('partial_rotary_factor', 0.25)),
        "rope_theta":                      (rope_params.get('rope_theta')
                                            or text_config.get('rope_theta', 10000000.0)),
        "default_bits":                    default_bits,
        "default_group_size":               default_group_size,
    }
    manifest = {
        "model": str(model_path),
        "num_tensors": len(all_tensors),
        "tensors": {},
        "config": cfg_out,
    }

    # Layer type map: take it from config when present (qwen3_5_moe
    # ships an explicit layer_types list); otherwise fall back to the
    # 3-linear : 1-full pattern driven by full_attention_interval.
    if 'layer_types' in text_config:
        manifest["config"]["layer_types"] = list(text_config['layer_types'])
    else:
        interval = cfg_out["full_attention_interval"]
        manifest["config"]["layer_types"] = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(cfg_out["num_hidden_layers"])
        ]

    print(f"\nWriting {bin_path}...")
    t0 = time.time()
    offset = 0
    total_bytes = 0

    ALIGN = 64  # 64-byte alignment for Metal buffers

    with open(bin_path, 'wb') as out_f:
        for i, (san_name, orig_name, filename) in enumerate(all_tensors):
            filepath = model_path / filename
            header, data_start = header_cache[filename]

            if orig_name not in header:
                print(f"  WARNING: {orig_name} not found in {filename}, skipping")
                continue

            meta = header[orig_name]
            tensor_offsets = meta['data_offsets']
            byte_len = tensor_offsets[1] - tensor_offsets[0]
            shape = meta['shape']
            dtype = meta['dtype']

            # Align offset
            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b'\x00' * pad)
                offset += pad

            # Read tensor data from safetensors
            with open(filepath, 'rb') as sf:
                sf.seek(data_start + tensor_offsets[0])
                data = sf.read(byte_len)

            # A17B stores `linear_attn.A_log` as F32; A3B stores it as BF16.
            # moeflux's C engine reads it as `float *A_log` unconditionally,
            # so A3B gets garbage without a conversion. Promote BF16 → F32
            # here so the binary file always carries F32 for A_log and the
            # C code needs no per-variant branching.
            if san_name.endswith('.linear_attn.A_log') and dtype == 'BF16':
                # bf16 is stored high-16-bits of an f32; reconstruct by
                # shifting into the top half of a uint32.
                bf16 = np.frombuffer(data, dtype=np.uint16)
                u32 = bf16.astype(np.uint32) << 16
                data = u32.tobytes()
                byte_len = len(data)
                dtype = 'F32'

            out_f.write(data)

            tensor_entry = {
                "offset": offset,
                "size": byte_len,
                "shape": shape,
                "dtype": dtype,
            }
            # Emit `bits` only for quantized weight tensors (U32 packed).
            # Scales/biases are bf16 and not bit-packed.
            if dtype == "U32":
                tensor_entry["bits"] = lookup_bits(orig_name)
            manifest["tensors"][san_name] = tensor_entry

            offset += byte_len
            total_bytes += byte_len

            if (i + 1) % 100 == 0 or i == len(all_tensors) - 1:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes / 1e9:.2f} GB written")

    elapsed = time.time() - t0
    throughput = total_bytes / elapsed / 1e9

    print(f"\nDone: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s ({throughput:.1f} GB/s)")
    print(f"Binary: {bin_path} ({os.path.getsize(bin_path) / 1e9:.2f} GB)")

    # Write manifest
    json_path = output_dir / 'model_weights.json'
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {json_path}")

    # Print summary by category
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for san_name, info in manifest["tensors"].items():
        if "embed_tokens" in san_name:
            cat = "embedding"
        elif "norm.weight" in san_name and "layers." not in san_name:
            cat = "final_norm"
        elif "lm_head" in san_name:
            cat = "lm_head"
        elif "input_layernorm" in san_name or "post_attention_layernorm" in san_name:
            cat = "layer_norms"
        elif "linear_attn" in san_name:
            cat = "linear_attention"
        elif "self_attn" in san_name:
            cat = "full_attention"
        elif "mlp.gate." in san_name:
            cat = "routing_gate"
        elif "shared_expert." in san_name:
            cat = "shared_expert"
        elif "shared_expert_gate" in san_name:
            cat = "shared_expert_gate"
        elif "switch_mlp" in san_name:
            cat = "routed_experts"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories.keys()):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")


if __name__ == '__main__':
    main()
