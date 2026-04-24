#!/usr/bin/env python3
"""Repack expert weights from scattered safetensors into contiguous per-layer binary files.

Creates one binary file per layer: <output_dir>/layer_XX.bin
Each file = NUM_EXPERTS × EXPERT_SIZE bytes.
Expert E starts at byte offset E * EXPERT_SIZE.

Within each expert block, 9 components packed in this fixed order:
  gate_proj.weight, gate_proj.scales, gate_proj.biases,
  up_proj.weight,   up_proj.scales,   up_proj.biases,
  down_proj.weight, down_proj.scales, down_proj.biases

All shape constants (NUM_EXPERTS, NUM_LAYERS, component sizes,
EXPERT_SIZE) are *derived* from expert_index.json rather than
hardcoded, so the same script works across model variants — see
`tools/gen_expert_index.py` for the index generator.

Usage:
    python repack_experts.py --index expert_index_<variant>.json
    python repack_experts.py --index <path> --layers 0-4
    python repack_experts.py --index <path> --layers 0,5,10
    python repack_experts.py --index <path> --dry-run
    python repack_experts.py --index <path> --verify-only 0
    python repack_experts.py --index <path> --output-dir /path/to/output
"""

import argparse
import json
import os
import time
import sys


# Canonical packing order — infer.m's expert layout assumes exactly
# this sequence. Do not reorder without updating model_variant.h
# (the {GATE,UP,DOWN}_{W,S,B}_OFF macros mirror this).
COMPONENT_ORDER = (
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
    "down_proj.weight", "down_proj.scales", "down_proj.biases",
)

# Per-component dtype. Derived from shape/size, but hardcoded here
# so the layout.json that downstream tools consume is self-describing
# without needing an MLX install.
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


def parse_layers(spec, num_layers):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    if spec is None or spec == 'all':
        return list(range(num_layers))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_index(index_path):
    """Load expert_index.json and return expert_reads dict + model_path."""
    with open(index_path) as f:
        idx = json.load(f)
    return idx['expert_reads'], idx['model_path']


def derive_layout(expert_reads):
    """Derive COMPONENTS list + (NUM_EXPERTS, NUM_LAYERS, EXPERT_SIZE)
    from the index. Validates consistency across layers."""
    if not expert_reads:
        sys.exit("error: expert_index has zero layers")

    # Use layer 0 as the shape reference; other layers must match.
    ref_key = next(iter(sorted(expert_reads, key=int)))
    ref = expert_reads[ref_key]

    missing = [c for c in COMPONENT_ORDER if c not in ref]
    if missing:
        sys.exit(f"error: layer {ref_key} missing components: {missing}")

    # Infer NUM_EXPERTS from the ref layer's shape (first dim).
    num_experts = ref[COMPONENT_ORDER[0]]['shape'][0]

    # Build COMPONENTS in canonical order with cumulative offsets.
    components = []
    offset = 0
    for name in COMPONENT_ORDER:
        info = ref[name]
        if info['shape'][0] != num_experts:
            sys.exit(f"error: {name} shape[0]={info['shape'][0]} "
                     f"disagrees with num_experts={num_experts}")
        size = info['expert_size']
        components.append({
            "name":   name,
            "offset": offset,
            "size":   size,
            "dtype":  COMPONENT_DTYPE[name],
            # Drop the leading num_experts dim — shape is per-expert.
            "shape":  list(info['shape'][1:]),
        })
        offset += size

    expert_size = offset
    num_layers = max(int(k) for k in expert_reads) + 1

    # Consistency check: every layer must have same per-expert sizes.
    for lk, layer in expert_reads.items():
        for c in components:
            got = layer[c['name']]['expert_size']
            if got != c['size']:
                sys.exit(f"error: layer {lk} {c['name']} size {got} != {c['size']}")

    return components, num_experts, num_layers, expert_size


def open_source_files(expert_reads, model_path, layers):
    """Open all needed safetensors files, return {filename: fd}."""
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"WARNING: layer {layer_idx} not found in expert_reads")
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds


def repack_layer(layer_idx, expert_reads, fds, output_dir,
                 components, num_experts, expert_size, layer_size,
                 dry_run=False):
    """Repack all experts for one layer into a contiguous binary file."""
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        for expert_idx in range(num_experts):
            for comp in components:
                info = layer_info[comp['name']]
                _ = info['abs_offset'] + expert_idx * info['expert_stride']
                _ = expert_idx * expert_size + comp['offset']
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {layer_size:,} bytes to {out_path}")
        return layer_size, 0.0

    t0 = time.monotonic()

    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    bytes_written = 0
    read_plan = []
    for expert_idx in range(num_experts):
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']
            read_plan.append((src_fd, src_offset, dst_offset, comp['size']))

    # Sort by (src_fd, src_offset) for sequential read locality.
    read_plan.sort(key=lambda x: (x[0], x[1]))

    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} "
                          f"at offset {src_offset}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0
    return bytes_written, elapsed


def verify_layer(layer_idx, expert_reads, fds, output_dir,
                 components, num_experts, expert_size):
    """Read back a few experts from the packed file and compare to originals."""
    layer_key = str(layer_idx)
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)
    # Spot-check: first, second, middle, last.
    probes = sorted({0, 1, num_experts // 2, num_experts - 1})

    mismatches = 0
    for expert_idx in probes:
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']

            original = os.pread(src_fd, comp['size'], src_offset)
            packed = os.pread(fd_packed, comp['size'], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)

    if mismatches == 0:
        probe_str = ", ".join(str(p) for p in probes)
        print(f"  Layer {layer_idx}: verification PASSED (experts {probe_str})")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")
    return mismatches == 0


def write_layout(output_dir, components, num_experts, num_layers, expert_size):
    """Write layout.json describing the packed format."""
    layout = {
        "expert_size": expert_size,
        "num_layers":  num_layers,
        "num_experts": num_experts,
        "components":  components,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Repack expert weights into contiguous per-layer binary files")
    parser.add_argument('--index', required=True,
                        help='Path to expert_index.json '
                             '(generate with tools/gen_expert_index.py)')
    parser.add_argument('--output-dir', default=None,
                        help='Where to write packed layer files '
                             '(default: <model_path>/packed_experts)')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify offsets without writing')
    parser.add_argument('--verify-only', type=int, default=None, metavar='LAYER',
                        help='Verify a specific layer against originals')
    args = parser.parse_args()

    print("Loading expert index...")
    expert_reads, model_path = load_index(args.index)
    print(f"Model path: {model_path}")
    print(f"Layers in index: {len(expert_reads)}")

    components, num_experts, num_layers, expert_size = derive_layout(expert_reads)
    layer_size = num_experts * expert_size

    print(f"Derived layout: num_experts={num_experts}, num_layers={num_layers}, "
          f"expert_size={expert_size:,} ({layer_size / 1024**2:.1f} MB/layer)")
    for c in components:
        print(f"  {c['name']:20s}  offset={c['offset']:>10,}  size={c['size']:>9,}  {c['dtype']}")

    output_dir = args.output_dir or os.path.join(model_path, "packed_experts")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers, num_layers)

    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * layer_size
        print(f"Total data to write: {total_bytes / (1024**3):.1f} GB")

        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes:
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB "
                  f"but only {free_gb:.1f} GB free.")
            sys.exit(1)

    fds = open_source_files(expert_reads, model_path, layers)

    if args.verify_only is not None:
        verify_layer(args.verify_only, expert_reads, fds, output_dir,
                     components, num_experts, expert_size)
        for fd in fds.values():
            os.close(fd)
        return

    write_layout(output_dir, components, num_experts, num_layers, expert_size)

    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        bytes_written, elapsed = repack_layer(
            layer_idx, expert_reads, fds, output_dir,
            components, num_experts, expert_size, layer_size,
            dry_run=args.dry_run
        )
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = (total_written / overall_elapsed / (1024**3)
                                  if overall_elapsed > 0 else 0)
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | "
                  f"Total: {total_written/1024**3:.1f}/{len(layers)*layer_size/1024**3:.1f} GB "
                  f"({overall_throughput:.1f} GB/s avg) | "
                  f"ETA: {eta:.0f}s")

            if not verify_layer(layer_idx, expert_reads, fds, output_dir,
                                components, num_experts, expert_size):
                print(f"ABORTING: verification failed for layer {layer_idx}")
                sys.exit(1)

    for fd in fds.values():
        os.close(fd)

    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*60}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers)} layers validated")


if __name__ == '__main__':
    main()
