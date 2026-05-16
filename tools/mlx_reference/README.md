# MLX reference diff harness

MLX is the trusted source-of-truth for moeflux's numerical output.
These scripts capture an MLX reference and compare moeflux against it.

## Automated regression test (`cargo test`)

The live, supported path. `generate_goldens.py` captures top-K MLX
logits for the last prompt position and writes a plain-text fixture
to `crates/moeflux/tests/fixtures/`. The Rust test
`crates/moeflux/tests/mlx_regression.rs` runs moeflux on the same
prompt and asserts top-20 set overlap ≥ 95% + argmax match. This
would have caught the A3B gate-offset bug on the first run.

Regenerate a fixture:

```bash
uv run --with mlx --with mlx-lm python3 generate_goldens.py \
  --model <mlx-model-dir> --variant <slug> \
  --out ../../crates/moeflux/tests/fixtures/mlx_golden_<slug>.txt
```

Run the test (per variant — variants are mutually exclusive Cargo
features):

```bash
# A3B
cargo test -p moeflux --features model-qwen3-6-35b-a3b \
  mlx_regression_a3b -- --ignored --nocapture

# A17B — fixture not committed; see host-RAM note below
cargo test -p moeflux --features model-qwen3-5-a17b \
  mlx_regression_a17b -- --ignored --nocapture
```

### Host-RAM requirement for A17B

MLX loads the full checkpoint into memory (unlike moeflux's streaming
experts). The A17B 4-bit MLX checkpoint is ~210 GB on disk; generating
its golden needs a host with roughly that much RAM (or slow,
SSD-thrashing swap). On a machine that can't host it, the test skips
gracefully with a "fixture not found" message — it does not fail. Any
Apple Silicon machine with ≥256 GB unified memory works, as do
sufficiently large cloud hosts (the script is CPU-only Python).

## Per-layer diff scripts (currently inert)

`diff_layer_inputs.py`, `diff_gate_outputs.py`, `diff_l0_components.py`
and `mlx_layer_dump.py` localize a divergence to a specific layer by
pairing per-layer moeflux dumps against MLX dumps. They were how the
A3B gate-offset bug was found (layer-0 per-expert outputs were all
zero → `gate_proj` matmul producing zero → hardcoded A17B offsets).

These predate the Rust rewrite: they consumed the `MOEFLUX_DUMP_L0`
env-gated dumps emitted by the old C `infer` binary, which no longer
exists. The scripts are kept for reference, but reviving the manual
workflow needs the equivalent per-layer dump hooks wired into the Rust
engine first.
