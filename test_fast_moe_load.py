"""Test fast_moe_load C extension against fast_weight_load baseline.

Verifies:
1. prealloc_stacked creates correct shapes and dtypes
2. load_and_assemble fills buffers with correct data (bit-exact vs pread)
3. BF16 view pairing works (scales/biases returned as bfloat16)
4. Timing comparison: stacked vs old per-slot approach

Uses the real 397B packed expert files.
"""

import time
import json
import struct
import os
import numpy as np
import mlx.core as mx


# --- Layout ---
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/"
    "snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"
)
PACKED_DIR = os.path.join(MODEL_PATH, "packed_experts")
LAYOUT_PATH = os.path.join(PACKED_DIR, "layout.json")

with open(LAYOUT_PATH) as f:
    layout = json.load(f)

expert_size = layout["expert_size"]
num_layers = layout["num_layers"]
num_experts = layout["num_experts"]
components = layout["components"]
K = 4  # active experts per token

DTYPE_MAP = {
    'U32': 'uint32',
    'BF16': 'uint16',  # stored as uint16, .view(bfloat16) later
    'F32': 'float32',
    'F16': 'float16',
}

# Build component specs for C extension
fml_components = []
for comp in components:
    fml_components.append({
        'name': comp['name'],
        'offset': comp['offset'],
        'size': comp['size'],
        'shape': comp['shape'],
        'dtype': DTYPE_MAP[comp['dtype']],
        'needs_bf16_view': comp['dtype'] == 'BF16',
    })


def test_prealloc():
    """Test that prealloc_stacked creates correct shapes and dtypes."""
    import fast_moe_load

    fast_moe_load.init(num_workers=8)

    # Only allocate 2 layers for testing (not 60)
    test_layers = 2
    layer_dicts = fast_moe_load.prealloc_stacked(
        test_layers, K, fml_components, PACKED_DIR, expert_size)

    assert isinstance(layer_dicts, list), f"Expected list, got {type(layer_dicts)}"
    assert len(layer_dicts) == test_layers, f"Expected {test_layers} layers, got {len(layer_dicts)}"

    for li, layer_dict in enumerate(layer_dicts):
        assert isinstance(layer_dict, dict), f"Layer {li}: expected dict"
        assert len(layer_dict) == len(components), (
            f"Layer {li}: expected {len(components)} components, got {len(layer_dict)}")

        for comp in components:
            name = comp['name']
            assert name in layer_dict, f"Layer {li}: missing component '{name}'"

            arr = layer_dict[name]
            expected_shape = tuple([K] + comp['shape'])
            assert arr.shape == expected_shape, (
                f"Layer {li}, {name}: shape {arr.shape} != expected {expected_shape}")

            # Check dtype: BF16 components should be bfloat16 (view)
            if comp['dtype'] == 'BF16':
                assert arr.dtype == mx.bfloat16, (
                    f"Layer {li}, {name}: dtype {arr.dtype} != bfloat16")
            elif comp['dtype'] == 'U32':
                assert arr.dtype == mx.uint32, (
                    f"Layer {li}, {name}: dtype {arr.dtype} != uint32")

    print(f"[PASS] prealloc_stacked: {test_layers} layers, {len(components)} comps, K={K}")
    print(f"       Shapes verified, BF16 views verified")

    s = fast_moe_load.stats()
    print(f"       Stats: {s}")

    fast_moe_load.shutdown()
    return True


def test_load_and_assemble():
    """Test that load_and_assemble fills buffers with correct data."""
    import fast_moe_load

    fast_moe_load.init(num_workers=8)

    test_layers = 3
    layer_dicts = fast_moe_load.prealloc_stacked(
        test_layers, K, fml_components, PACKED_DIR, expert_size)

    # Pick some expert indices to load
    routing = [
        (0, [10, 42, 100, 7]),     # layer 0
        (1, [255, 0, 128, 64]),    # layer 1
        (2, [33, 77, 200, 150]),   # layer 2
    ]

    result = fast_moe_load.load_and_assemble(routing)
    assert result is layer_dicts, "load_and_assemble should return same list object"

    # Verify data by reading the same experts via direct pread
    for layer_idx, expert_indices in routing:
        packed_file = os.path.join(PACKED_DIR, f"layer_{layer_idx:02d}.bin")
        fd = os.open(packed_file, os.O_RDONLY)

        for slot, eidx in enumerate(expert_indices):
            expert_offset = eidx * expert_size

            for comp in components:
                comp_name = comp['name']
                comp_offset = comp['offset']
                comp_size = comp['size']

                # Read reference data via pread
                ref_bytes = os.pread(fd, comp_size, expert_offset + comp_offset)

                # Get the stacked array and extract slot
                stacked_arr = layer_dicts[layer_idx][comp_name]

                if comp['dtype'] == 'BF16':
                    # The stacked arr is bfloat16 view; get uint16 view for comparison
                    slot_arr = stacked_arr[slot].view(mx.uint16)
                else:
                    slot_arr = stacked_arr[slot]

                # Convert reference bytes to array for comparison
                if comp['dtype'] == 'U32':
                    ref_np = np.frombuffer(ref_bytes, dtype=np.uint32).reshape(comp['shape'])
                    ref_mx = mx.array(ref_np)
                    match = mx.array_equal(slot_arr, ref_mx)
                elif comp['dtype'] == 'BF16':
                    ref_np = np.frombuffer(ref_bytes, dtype=np.uint16).reshape(comp['shape'])
                    ref_mx = mx.array(ref_np)
                    match = mx.array_equal(slot_arr, ref_mx)
                else:
                    # Fallback
                    ref_np = np.frombuffer(ref_bytes, dtype=np.uint8)
                    slot_bytes = np.array(slot_arr, dtype=np.uint8)
                    match = np.array_equal(ref_np, slot_bytes)

                mx.eval(match)
                assert match.item(), (
                    f"Data mismatch: layer={layer_idx}, slot={slot}, "
                    f"expert={eidx}, comp={comp_name}")

        os.close(fd)

    print(f"[PASS] load_and_assemble: {len(routing)} layers, K={K}, bit-exact verified")

    s = fast_moe_load.stats()
    print(f"       Stats: {s}")

    fast_moe_load.shutdown()
    return True


def test_load_experts_direct():
    """Test the compatibility load_experts_direct function."""
    import fast_moe_load

    fast_moe_load.init(num_workers=8)

    test_layers = 2
    layer_dicts = fast_moe_load.prealloc_stacked(
        test_layers, K, fml_components, PACKED_DIR, expert_size)

    # Build load_list in the old (layer, expert, slot) format
    load_list = [
        (0, 23, 0),
        (0, 45, 1),
        (0, 120, 2),
        (0, 7, 3),
        (1, 100, 0),
        (1, 200, 1),
        (1, 50, 2),
        (1, 0, 3),
    ]

    result = fast_moe_load.load_experts_direct(load_list)
    assert result > 0, f"Expected positive result, got {result}"

    # Verify one slot
    packed_file = os.path.join(PACKED_DIR, "layer_00.bin")
    fd = os.open(packed_file, os.O_RDONLY)

    # Check layer 0, slot 0, expert 23, gate_proj.weight
    comp = components[0]  # gate_proj.weight
    ref_bytes = os.pread(fd, comp['size'], 23 * expert_size + comp['offset'])
    ref_np = np.frombuffer(ref_bytes, dtype=np.uint32).reshape(comp['shape'])
    ref_mx = mx.array(ref_np)

    slot_arr = layer_dicts[0][comp['name']][0]  # slot 0
    match = mx.array_equal(slot_arr, ref_mx)
    mx.eval(match)
    assert match.item(), "load_experts_direct data mismatch"

    os.close(fd)

    print(f"[PASS] load_experts_direct: {len(load_list)} entries, data verified")

    fast_moe_load.shutdown()
    return True


def test_timing():
    """Compare timing: stacked (new) vs per-slot + mx.stack (old)."""
    import fast_moe_load

    fast_moe_load.init(num_workers=8)

    test_layers = 60  # full model
    layer_dicts = fast_moe_load.prealloc_stacked(
        test_layers, K, fml_components, PACKED_DIR, expert_size)

    # Generate random routing for all 60 layers
    rng = np.random.default_rng(42)
    routing = []
    for li in range(test_layers):
        experts = rng.choice(num_experts, size=K, replace=False).tolist()
        routing.append((li, experts))

    # Warmup
    fast_moe_load.load_and_assemble(routing)

    # Time the new stacked approach
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        result = fast_moe_load.load_and_assemble(routing)
    t_stacked = (time.perf_counter() - t0) / N

    # Time the old approach: load_experts_direct + Python mx.stack assembly
    # Build old-style load_list
    old_load_list = []
    for li, experts in routing:
        for slot, eidx in enumerate(experts):
            old_load_list.append((li, eidx, slot))

    t0 = time.perf_counter()
    for _ in range(N):
        fast_moe_load.load_experts_direct(old_load_list)
        # Simulate the Python assembly step
        for li in range(test_layers):
            expert_tensors = {}
            for comp in components:
                comp_name = comp['name']
                slices = [layer_dicts[li][comp_name][s] for s in range(K)]
                expert_tensors[comp_name] = mx.stack(slices, axis=0)
    t_old_with_stack = (time.perf_counter() - t0) / N

    # Also time just the I/O without assembly
    t0 = time.perf_counter()
    for _ in range(N):
        fast_moe_load.load_experts_direct(old_load_list)
    t_io_only = (time.perf_counter() - t0) / N

    # Compute the assembly overhead
    t_assembly = t_old_with_stack - t_io_only

    s = fast_moe_load.stats()
    bytes_per_token = K * len(components) * test_layers
    total_data_mb = sum(c['size'] for c in components) * K * test_layers / 1e6

    print(f"\n[TIMING] 60 layers, K={K}, {len(components)} components")
    print(f"  Data per token: {total_data_mb:.1f} MB ({K} experts x {test_layers} layers)")
    print(f"  Stacked (new):           {t_stacked*1000:7.2f} ms  <- ONE C call, no Python assembly")
    print(f"  Direct + mx.stack (old): {t_old_with_stack*1000:7.2f} ms  <- C I/O + Python loop + 540 mx.stack")
    print(f"    I/O only:              {t_io_only*1000:7.2f} ms")
    print(f"    Assembly overhead:      {t_assembly*1000:7.2f} ms  <- THIS is what stacked eliminates")
    print(f"  Speedup: {t_old_with_stack/t_stacked:.2f}x")

    fast_moe_load.shutdown()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing fast_moe_load C extension")
    print("=" * 60)
    print()

    all_pass = True
    for test in [test_prealloc, test_load_and_assemble, test_load_experts_direct, test_timing]:
        try:
            test()
            print()
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False
            print()

    if all_pass:
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    else:
        print("=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
