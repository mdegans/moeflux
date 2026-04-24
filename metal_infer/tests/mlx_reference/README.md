# MLX reference diff harness

These scripts compare moeflux's per-layer forward-pass intermediates against
an MLX ground-truth reference, on the same prompt tokens. They were built to
debug the A3B gate-offset bug (commit that introduced this harness) and are
kept as a regression tool for future MoE variants.

## Usage

1. **Enable moeflux's env-gated dumps.** Set `MOEFLUX_DUMP_L0=/tmp/mf` before
   running `infer`. For each prompt-token-pos=0 layer entry, writes:
   - `/tmp/mf_l{N}.bin`       — per-layer h_post + gate_logits + gate_probs + top-K
   - `/tmp/mf_l{N}_in.bin`    — per-layer input hidden state
   - `/tmp/mf_l0_components.bin` — layer-0 MoE component breakdown

   ```bash
   cd /path/to/qwen3-6-35b-a3b-artifacts
   MOEFLUX_DUMP_L0=/tmp/mf \
   MOEFLUX_SHADERS_PATH=/path/to/moeflux/metal_infer/shaders.metal \
   /path/to/moeflux/metal_infer/infer \
     --model /path/to/qwen3-6-35b-a3b-root \
     --weights model_weights.bin --manifest model_weights.json --vocab vocab.bin \
     --prompt "The quick brown fox" --tokens 2 --k 8
   ```

2. **Generate the MLX reference dump.** Writes to a matching prefix so the diff
   scripts can pair files.

   ```bash
   uv run --with mlx --with mlx-lm python3 mlx_layer_dump.py /tmp/mx
   ```

   This patches `Qwen3NextSparseMoeBlock.__call__` and `DecoderLayer.__call__`
   to capture layer inputs, gate logits, top-K picks, and layer-0 MoE
   sub-components. Model path is hardcoded; edit the script to point at your
   A3B MLX checkpoint.

3. **Diff.** Three scripts, each answering a different question:
   - `diff_layer_inputs.py`   — are layer inputs (= prior layer outputs)
                                matching at pos=0? If layer N's input matches
                                but N+1's doesn't, layer N is the bug.
   - `diff_gate_outputs.py`   — per-layer h_post / gate logits / top-K set
                                overlap. Useful for early localization.
   - `diff_l0_components.py`  — layer-0 only: breaks down h_mid, shared_raw,
                                per-expert outputs, final out. Use when you
                                know layer 0 diverges but not where inside it.

## What we learned from using this harness

The A3B bug was localized in three phases:

  Phase 1 (`diff_gate_outputs.py`): layer-0 top-8 experts matched MLX → gate
    matmul + top-K were correct.
  Phase 2 (`diff_layer_inputs.py`): layer 1's input diverged (cos 0.96 vs
    0.9999 at layer 0) → something between layer 0's gate and layer 0's
    output is wrong.
  Phase 3 (`diff_l0_components.py`): moeflux's per-expert outputs were ALL
    zero → gate_proj matmul was producing zero → hardcoded A17B offsets.

## Expected (pass) output after fix

Running the A3B prompt "The quick brown fox" through moeflux should produce
"jumps over the lazy dog" as the first five generated tokens — matching MLX's
canonical pangram completion.
