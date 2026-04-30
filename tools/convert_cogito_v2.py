#!/usr/bin/env python3
"""End-to-end conversion: MLX 4-bit DeepSeek-V3 / Cogito-V2 → moeflux on-disk format.

Reads an `mlx_lm convert`-produced DeepSeek-V3 architecture model
(currently `mlx-community/cogito-v2-preview-deepseek-671B-MoE-4bit`)
and emits the canonical moeflux parent-directory layout:

    parent/
    ├── mlx/
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── special_tokens_map.json
    │   ├── chat_template.jinja
    │   └── config.json
    ├── artifacts/
    │   ├── model_weights.bin     # MLA + dense + shared-expert + gate
    │   │                           + embedding + lm_head + final norm
    │   └── model_weights.json    # tensor-name → offset/size manifest
    └── root/
        └── packed_experts/
            ├── layer_03.bin      # 256 experts × 9 components, packed
            ├── ...
            ├── layer_60.bin
            └── layout.json

Layers `< first_k_dense_replace` (0-2 for Cogito-V2-671B) have no
`switch_mlp.*` tensors — those are dense MLP layers whose weights
go into `model_weights.bin` instead of per-layer `.bin` files. The
`packed_experts/` directory therefore starts at `layer_03.bin`.

Usage:
    python convert_cogito_v2.py --src <mlx-dir> --dst <parent-dir>

Standalone — only depends on the stdlib.
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

# Canonical packing order — moeflux's expert layout assumes exactly
# this sequence. Mirrors `model_variant.h`'s {GATE,UP,DOWN}_{W,S,B}_OFF.
COMPONENT_ORDER = (
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
    "down_proj.weight", "down_proj.scales", "down_proj.biases",
)

COMPONENT_DTYPE = {
    "gate_proj.weight":  "U32",
    "gate_proj.scales":  "BF16",
    "gate_proj.biases":  "BF16",
    "up_proj.weight":    "U32",
    "up_proj.scales":    "BF16",
    "up_proj.biases":    "BF16",
    "down_proj.weight":  "U32",
    "down_proj.scales":  "BF16",
    "down_proj.biases":  "BF16",
}

# 64-byte alignment for the artifacts blob — Metal buffer requirement.
ARTIFACT_ALIGN = 64

EXPERT_TENSOR_RE = re.compile(
    r"\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


def parse_safetensors_header(filepath):
    """Return (header_dict, data_start_byte_offset)."""
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def categorize_tensors(weight_map, mlx_path):
    """Walk every tensor in the MLX index. Return (artifacts, experts):
       - artifacts: list of (name, file, abs_offset, byte_len, shape, dtype)
       - experts: dict[layer_idx] -> dict[component] -> dict (per-tensor metadata)
    """
    header_cache = {}
    def header_for(filename):
        if filename not in header_cache:
            path = os.path.join(mlx_path, filename)
            header_cache[filename] = parse_safetensors_header(path)
        return header_cache[filename]

    # Pre-parse every shard once.
    shards = sorted(set(weight_map.values()))
    for s in shards:
        header_for(s)

    artifacts = []
    experts = defaultdict(dict)

    for name in sorted(weight_map.keys()):
        filename = weight_map[name]
        header, data_start = header_for(filename)
        if name not in header:
            sys.exit(f"error: tensor {name!r} not in shard {filename}'s header")
        meta = header[name]
        ds, de = meta["data_offsets"]
        byte_len = de - ds
        abs_offset = data_start + ds
        shape = meta["shape"]
        dtype = meta["dtype"]

        m = EXPERT_TENSOR_RE.search(name)
        if m:
            # Expert tensor — needs to be per-layer-packed later.
            layer_match = re.search(r"\.layers\.(\d+)\.", name)
            if not layer_match:
                sys.exit(f"error: expert tensor without layer index: {name!r}")
            layer = int(layer_match.group(1))
            component = f"{m.group(1)}.{m.group(2)}"  # e.g. "gate_proj.weight"
            num_experts = shape[0]
            per_expert = byte_len // num_experts
            if byte_len % num_experts != 0:
                sys.exit(f"error: {name!r} size {byte_len} not divisible by "
                         f"num_experts={num_experts}")
            experts[layer][component] = {
                "file":          filename,
                "abs_offset":    abs_offset,
                "expert_stride": per_expert,
                "expert_size":   per_expert,
                "total_size":    byte_len,
                "shape":         shape,
            }
        else:
            artifacts.append((name, filename, abs_offset, byte_len, shape, dtype))

    return artifacts, experts


def lookup_bits_for(name, q_block, default_bits):
    """Resolve quantization bits per-tensor. MLX stores defaults at
    the top level of `quantization` and per-tensor overrides keyed
    by base name (no `.weight`/`.scales`/`.biases` suffix)."""
    base = name
    for suf in (".weight", ".scales", ".biases"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    override = q_block.get(base, None)
    if isinstance(override, dict) and "bits" in override:
        return int(override["bits"])
    return default_bits


def write_artifacts(artifacts, mlx_path, dst_artifacts_dir, q_block,
                    default_bits, default_group_size, hf_config):
    """Pack non-expert tensors into model_weights.bin + manifest."""
    bin_path = dst_artifacts_dir / "model_weights.bin"
    json_path = dst_artifacts_dir / "model_weights.json"

    # Group tensors by source shard for sequential reads.
    by_file = defaultdict(list)
    for tup in artifacts:
        by_file[tup[1]].append(tup)

    manifest = {
        "model": str(mlx_path),
        "num_tensors": len(artifacts),
        "config": {
            # DeepSeek-V3-shaped config block. The Rust port reads
            # only the `tensors` section; this `config` is informational
            # for tooling that wants to introspect without reading the
            # MLX config.json.
            "hidden_size":          hf_config.get("hidden_size"),
            "num_hidden_layers":    hf_config.get("num_hidden_layers"),
            "num_attention_heads":  hf_config.get("num_attention_heads"),
            "vocab_size":           hf_config.get("vocab_size"),
            "rms_norm_eps":         hf_config.get("rms_norm_eps", 1e-6),
            "n_routed_experts":     hf_config.get("n_routed_experts"),
            "n_shared_experts":     hf_config.get("n_shared_experts"),
            "num_experts_per_tok":  hf_config.get("num_experts_per_tok"),
            "n_group":              hf_config.get("n_group"),
            "topk_group":           hf_config.get("topk_group"),
            "first_k_dense_replace": hf_config.get("first_k_dense_replace"),
            "moe_intermediate_size": hf_config.get("moe_intermediate_size"),
            "intermediate_size":    hf_config.get("intermediate_size"),
            "qk_nope_head_dim":     hf_config.get("qk_nope_head_dim"),
            "qk_rope_head_dim":     hf_config.get("qk_rope_head_dim"),
            "v_head_dim":           hf_config.get("v_head_dim"),
            "q_lora_rank":          hf_config.get("q_lora_rank"),
            "kv_lora_rank":         hf_config.get("kv_lora_rank"),
            "rope_theta":           hf_config.get("rope_theta"),
            "rope_scaling":         hf_config.get("rope_scaling"),
            "routed_scaling_factor": hf_config.get("routed_scaling_factor"),
            "default_bits":          default_bits,
            "default_group_size":    default_group_size,
        },
        "tensors": {},
    }

    print(f"\nWriting {bin_path}...")
    t0 = time.time()
    offset = 0
    total_bytes = 0

    with open(bin_path, "wb") as out_f:
        # Iterate shards in order, then tensors by abs_offset within each
        # shard for sequential reads.
        for filename in sorted(by_file.keys()):
            shard_path = mlx_path / filename
            tensors_in_shard = sorted(by_file[filename], key=lambda t: t[2])
            with open(shard_path, "rb") as sf:
                for name, _, abs_offset, byte_len, shape, dtype in tensors_in_shard:
                    # Align before each tensor.
                    if offset % ARTIFACT_ALIGN != 0:
                        pad = ARTIFACT_ALIGN - (offset % ARTIFACT_ALIGN)
                        out_f.write(b"\x00" * pad)
                        offset += pad

                    sf.seek(abs_offset)
                    data = sf.read(byte_len)
                    if len(data) != byte_len:
                        sys.exit(f"short read on {name}: {len(data)} != {byte_len}")
                    out_f.write(data)

                    entry = {
                        "offset": offset,
                        "size":   byte_len,
                        "shape":  shape,
                        "dtype":  dtype,
                    }
                    if dtype == "U32":
                        entry["bits"] = lookup_bits_for(name, q_block, default_bits)
                    manifest["tensors"][name] = entry
                    offset += byte_len
                    total_bytes += byte_len

            elapsed = time.time() - t0
            mb = total_bytes / 1e6
            print(f"  shard {filename}: cumulative {mb:8.1f} MB ({elapsed:.1f}s)")

    print(f"\nDone artifacts: {total_bytes / 1e9:.2f} GB in {time.time()-t0:.1f}s")

    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {json_path}")
    return manifest


def derive_expert_layout(experts):
    """Inspect any one expert layer to derive the canonical
    {components, num_experts, expert_size}. All MoE layers must share
    these dims (DeepSeek-V3 doesn't override per-layer)."""
    if not experts:
        sys.exit("error: no expert layers found; is this an MoE model?")

    ref_layer = sorted(experts.keys())[0]
    ref = experts[ref_layer]

    missing = [c for c in COMPONENT_ORDER if c not in ref]
    if missing:
        sys.exit(f"error: layer {ref_layer} missing components: {missing}")

    num_experts = ref[COMPONENT_ORDER[0]]["shape"][0]

    components = []
    offset = 0
    for name in COMPONENT_ORDER:
        info = ref[name]
        if info["shape"][0] != num_experts:
            sys.exit(f"error: {name} shape[0]={info['shape'][0]} "
                     f"!= num_experts={num_experts}")
        size = info["expert_size"]
        components.append({
            "name":   name,
            "offset": offset,
            "size":   size,
            "dtype":  COMPONENT_DTYPE[name],
            "shape":  list(info["shape"][1:]),
        })
        offset += size
    expert_size = offset

    # Consistency check across layers.
    for layer_idx, layer in experts.items():
        for c in components:
            got = layer[c["name"]]["expert_size"]
            if got != c["size"]:
                sys.exit(f"error: layer {layer_idx} {c['name']} "
                         f"expert_size {got} != {c['size']}")

    return components, num_experts, expert_size


def write_packed_experts(experts, mlx_path, dst_root_dir, components,
                         num_experts, expert_size, num_total_layers):
    """Pack experts into per-layer .bin files. Layers without expert
    tensors (dense layers in DeepSeek-V3) are skipped — those weights
    went into the artifacts blob."""
    packed_dir = dst_root_dir / "packed_experts"
    packed_dir.mkdir(parents=True, exist_ok=True)

    layer_size = num_experts * expert_size
    layout = {
        "expert_size": expert_size,
        "num_layers":  num_total_layers,
        "num_experts": num_experts,
        "components":  components,
        "moe_layers":  sorted(experts.keys()),
    }
    layout_path = packed_dir / "layout.json"
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"\nWrote {layout_path}")

    moe_layers = sorted(experts.keys())
    print(f"Packing {len(moe_layers)} MoE layers "
          f"(layer_size={layer_size/1e9:.2f} GB each, total "
          f"{len(moe_layers)*layer_size/1e9:.1f} GB)...")

    # Free space sanity.
    total_needed = len(moe_layers) * layer_size
    stat = os.statvfs(packed_dir)
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < total_needed:
        sys.exit(f"error: not enough free space — need "
                 f"{total_needed/1e9:.1f} GB, have {free_bytes/1e9:.1f} GB")

    fds_in = {}
    def fd_for(filename):
        if filename not in fds_in:
            fds_in[filename] = os.open(str(mlx_path / filename), os.O_RDONLY)
        return fds_in[filename]

    t_total = time.time()
    bytes_total = 0
    for i, layer_idx in enumerate(moe_layers):
        out_path = packed_dir / f"layer_{layer_idx:02d}.bin"
        layer_info = experts[layer_idx]

        fd_out = os.open(str(out_path), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd_out, layer_size)

        # Build (src_fd, src_offset, dst_offset, size) plan, sort
        # by (src_fd, src_offset) for sequential reads.
        plan = []
        for expert_idx in range(num_experts):
            for comp in components:
                info = layer_info[comp["name"]]
                src_offset = info["abs_offset"] + expert_idx * info["expert_stride"]
                dst_offset = expert_idx * expert_size + comp["offset"]
                plan.append((fd_for(info["file"]), src_offset, dst_offset,
                             comp["size"]))
        plan.sort(key=lambda x: (x[0], x[1]))

        t0 = time.time()
        for src_fd, src_off, dst_off, size in plan:
            data = os.pread(src_fd, size, src_off)
            if len(data) != size:
                sys.exit(f"short read at layer {layer_idx}: "
                         f"{len(data)} != {size}")
            os.pwrite(fd_out, data, dst_off)
        os.close(fd_out)

        elapsed = time.time() - t0
        bytes_total += layer_size
        avg_throughput = bytes_total / (time.time() - t_total) / 1e9
        eta = (len(moe_layers) - i - 1) * (time.time() - t_total) / (i + 1)
        print(f"  layer_{layer_idx:02d}: {layer_size/1e9:.2f} GB in "
              f"{elapsed:.1f}s (avg {avg_throughput:.2f} GB/s, "
              f"ETA {eta:.0f}s)")

    for fd in fds_in.values():
        os.close(fd)

    print(f"\nDone packed_experts: {bytes_total/1e9:.1f} GB in "
          f"{time.time()-t_total:.1f}s")


def copy_mlx_files(mlx_path, dst_mlx_dir):
    """Copy the small MLX-side files that drama_llama / blallama need
    at runtime (tokenizer, config, chat template). Skip the
    safetensors shards — those are ~2-4 GB each and we only need to
    read them, not duplicate them."""
    dst_mlx_dir.mkdir(parents=True, exist_ok=True)
    wanted = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "config.json",
    ]
    print(f"\nCopying MLX-side metadata to {dst_mlx_dir}...")
    for fname in wanted:
        src = mlx_path / fname
        if src.exists():
            shutil.copy2(src, dst_mlx_dir / fname)
            print(f"  copied {fname}")
        else:
            print(f"  WARNING: {fname} not present in source; skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="Source MLX directory.")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Destination parent directory. Will be created if needed.")
    ap.add_argument("--skip-experts", action="store_true",
                    help="Skip the per-layer expert packing (artifacts only).")
    ap.add_argument("--skip-artifacts", action="store_true",
                    help="Skip the artifacts blob (experts only).")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"--src must be a directory: {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    dst_mlx = args.dst / "mlx"
    dst_artifacts = args.dst / "artifacts"
    dst_root = args.dst / "root"
    dst_artifacts.mkdir(parents=True, exist_ok=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Load MLX index + config.
    index_path = args.src / "model.safetensors.index.json"
    if not index_path.exists():
        sys.exit(f"--src has no model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    config_path = args.src / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)
    q_block = hf_config.get("quantization", {}) or {}
    default_bits = q_block.get("bits", 4)
    default_group_size = q_block.get("group_size", 64)
    num_total_layers = hf_config.get("num_hidden_layers", 0)

    print(f"Source: {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Total tensors in index: {len(weight_map)}")
    print(f"Quantization: {default_bits}-bit, group_size={default_group_size}")
    print(f"num_hidden_layers: {num_total_layers}")

    # Categorize.
    print("\nCategorizing tensors (parsing safetensors headers)...")
    artifacts, experts = categorize_tensors(weight_map, str(args.src))
    print(f"  artifacts (non-expert): {len(artifacts)}")
    print(f"  expert layers: {len(experts)} "
          f"(layers {sorted(experts.keys())[0]}-{sorted(experts.keys())[-1]})")

    # Copy MLX-side files always.
    copy_mlx_files(args.src, dst_mlx)

    if not args.skip_artifacts:
        write_artifacts(artifacts, args.src, dst_artifacts,
                        q_block, default_bits, default_group_size, hf_config)

    if not args.skip_experts:
        components, num_experts, expert_size = derive_expert_layout(experts)
        print(f"\nExpert layout: num_experts={num_experts}, "
              f"expert_size={expert_size:,} bytes")
        for c in components:
            print(f"  {c['name']:20s}  offset={c['offset']:>10,}  "
                  f"size={c['size']:>9,}  {c['dtype']}")
        write_packed_experts(experts, args.src, dst_root, components,
                             num_experts, expert_size, num_total_layers)

    print(f"\n{'='*60}\nConversion complete: {args.dst}")


if __name__ == "__main__":
    main()
