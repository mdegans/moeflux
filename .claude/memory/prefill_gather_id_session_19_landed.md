---
name: prefill-gather-id-session-19-landed
description: 2026-05-20 — session 19 landed `Op::MoeGatherIdFuse` + `gather_mm_id.metal` (port of llama.cpp's `kernel_mul_mm_id` for W4_gs64). Kernel-level diff oracle passes at production scale; engine-level integration produces degenerate output (1 token then EoT). Pipeline bug, NOT a kernel bug — investigation bounded for next session.
metadata:
  type: project
---

# What landed

Commit `5a45af3` on moeflux `main`. Two pieces of the gap-to-
llama.cpp work from `llama_cpp_moe_differentiators.md`:

## Differentiator 2: MTLResidencySet (clean win)

- `crates/moeflux-metal/src/residency_set.rs` — raw `objc::msg_send!`
  bindings, ~180 LOC including a runtime-probe + smoke test.
- `crates/moeflux/src/riir/io/expert_io.rs` — `ExpertFiles` pins
  every per-layer expert mmap buffer at `attach_to_device` time.
- **Bench:** variance 16% (iter-1→iter-2 cold jump) → 2.3% peak-
  to-peak across n=5. Throughput unchanged (333 → 334 mean).
  Outcome 2 per the plan: variance fix is real, throughput-neutral
  on isolated bench (will matter under contention).

See [[prefill_residency_set_landed]] for the full memo.

## Differentiator 1: kernel algorithm (lands but BLOCKED on bug)

- `crates/moeflux-metal/shaders/gather_mm_id.metal` — 1:1 port of
  `kernel_mul_mm_id_map0` + `kernel_mul_mm_id` with W4_gs64
  dequant inlined. One-dispatch MoE matmul, per-expert tile-Z,
  `htpe[e]` early-return, in-kernel activation gather via
  `hids[e][n]`. Also includes `combine_topk` reduction kernel.
- `crates/moeflux-metal/src/lib.rs` — `MoeIdMap0Call`,
  `MoeGatherIdCall`, `MoeCombineTopkCall` dispatch wrappers.
- `crates/moeflux/src/riir/backend/{mod,gpu,cpu}/mod.rs` — new
  `Op::MoeGatherIdFuse` arm + GPU dispatch (CPU stub `todo!()`s,
  GPU-only by design — engine-level GPU/GPU env-flag A/B is the
  correctness path).
- `crates/moeflux/src/riir/moe/expert_forward.rs` —
  `encode_moe_gather_id_fuse` encoder (map0 → gate matmul → up
  matmul → SwiGLU in-place → down matmul → combine_topk).
- `crates/moeflux/src/riir/attn/linear_attn_forward.rs` —
  `MoeGraphScratch` gains `htpe`/`hids` BufIds; reuses
  `bucket_gate`/`bucket_up`/`bucket_out` as `gate_mid`/`up_mid`/
  `down_mid` (byte-equivalent, mutually exclusive paths); producer
  env-flag selection (`MOEFLUX_MOE_GATHER_ID`); host-side
  bucket-permute conditionally skipped (correctness AND fair-bench
  requirement — running it would inflate env=on by tens of ms/
  layer of wasted work).

# Bench numbers

n=5, post-reboot, `prefill_prompt_long.txt` (15692 tokens):

| path | mean prefill tok/s | variance |
|---|---|---|
| env=off (`MoeBatchedPermuteFuse`) | ~334 | 2.3% pk-pk |
| env=on (`MoeGatherIdFuse`) | ~370 | ~1% pk-pk |

**+11% prefill speedup.** Gap-to-llama.cpp: 855/370 = **2.31×**
(was 2.56×). Closed ~10% of the relative gap.

# Correctness status: BROKEN

`max_tokens=30, temp=0, seed=42` on the same long prompt:

- env=off output: `"This is a comprehensive technical deep dive
  into the architecture, training, inference, and deployment of
  modern Large Language Models (LLMs). The discussion is
  structured"` — fully coherent.
- env=on output: `"</think>"` (1 token), then `stop_reason:
  end_turn`. Total degeneracy.

# What the bug is NOT

Eliminated systematically before pausing:

- **Not the kernel.** Five diff-oracle tests pass cosine ≥ 0.9999
  in `crates/moeflux-metal/tests/gather_mm_id_diff.rs`:
  `tiny`, `mid`, `down_shape` (ne11=k), `a3b_scale_gate` (256
  experts × K=2048), `a3b_scale_down` (256 experts × N=2048).
  Production-scale coverage in both shapes.
- **Not a stride mismatch in routing.** Subagent flagged this
  initially; I verified the routing buffer is allocated MAX_K=16
  wide but BOTH the GPU router AND my map0 kernel use stride k=8
  consistently — only the first half of the buffer is touched by
  either side. No mismatch.
- **Not `commit_plan` aliasing** of bucket_gate/bucket_up/
  bucket_out. The greedy coloring in `backend/lifetime.rs:143`
  treats overlapping `[op, op]` intervals as non-aliasable; the
  three transients all get distinct colors.
- **Not a missing host upload.** When env=on we skip
  `bucket_input` host-permute (intentional). `bucket_token_idx`,
  `bucket_weights`, `expert_indices` still get uploaded for the
  old path's needs but the new path doesn't consume them.
- **Not full_attn vs linear_attn divergence.** Both attention
  paths route through the same `moe_block_forward` (linear_attn_
  forward.rs:2521 → called from full_attn_forward.rs:978).

# What to try next session

In order of bang/buck:

1. **Map0 runtime instrumentation.** Add a debug write
   (`htpe[ide]` sum, or per-expert counts as a scratch-buffer
   sentinel) and dump after a real prefill chunk. If `sum(htpe)
   != n_tokens * k`, something is wrong with the index reads. If
   correct, map0 isn't the bug and we look further.
2. **Engine-level single-layer diff harness.** Construct a
   single-layer graph with both `Op::MoeBatchedPermuteFuse` AND
   `Op::MoeGatherIdFuse` (separate runs), download `out_sum`
   after each, compare per-token-per-channel. Catches per-layer
   divergence directly without needing 60 layers of error
   accumulation.
3. **Bisect within commit `5a45af3`.** Temporarily revert the
   `bucket_input_host` skip (force the old-path's host work to
   also run with env=on); if the bug clears, the skip dropped a
   silent data dep. If not, the bug is in the Op/encoder/kernel
   interaction.
4. **Per-row cosine on simulated production routing**. The diff
   oracle currently generates random distinct-per-token routing.
   Real GPU router output may have specific distributional
   properties (some experts get many tokens, others few) that
   stress edge cases the test misses. Feed the test a captured
   real routing log (via `MOEFLUX_LOG_HTPE=1` already in the
   tree).

# Structural hardening (after the bug is found)

Mike's framing: "when we find the bug we'll work to make it
structurally harder to reoccur." Likely candidates depending on
what the bug turns out to be:

- If routing-data flow: a Rust-side type for "router output
  buffer" with stride encoded in the type, vs a generic `BufId`.
- If commit_plan-related: stronger live-range declarations on
  `Op::MoeGatherIdFuse`, or a `single-op-RMW-cluster` marker that
  pins the buffers' physical identity.
- If integration-test gap: an engine-level diff-test pattern
  (single-layer, both paths, cosine compare) that's cheap enough
  to run as part of CI.

# Reproduction recipe

```bash
# rebuild
cd /Users/mdegans/Projects/drama_llama
cargo run --release \
    --features "moeflux-model-qwen3-6-35b-a3b,axum,cli,toml" \
    --bin blallama -- \
    '/Volumes/Temp Backup/models/blallama' \
    --port 11435 --backend moeflux --seed 42

# env=on (in a second shell):
MOEFLUX_MOE_GATHER_ID=1 cargo run --release \
    --features "moeflux-model-qwen3-6-35b-a3b,axum,cli,toml" \
    --bin blallama -- \
    '/Volumes/Temp Backup/models/blallama' \
    --port 11435 --backend moeflux --seed 42

# curl (matches both env states):
curl -s -X POST http://127.0.0.1:11435/v1/messages \
    -H "Content-Type: application/json" \
    -d "$(jq -Rs --arg m qwen3-6-a3b '{model: $m,
        messages: [{role: "user", content: .}],
        max_tokens: 30, temperature: 0.0}' \
        prefill_prompt_long.txt)" \
    --max-time 120 | jq '.content[0].text'
```

# Cross-references

- [[prefill_residency_set_landed]] — Differentiator 2 (variance
  fix, throughput-neutral).
- [[llama_cpp_moe_differentiators]] — original 3-differentiator
  framing that drove the session.
- [[gather_qmm_pivot_dead]] — diagnostic that birthed the framing.
- [[future_work_m5_tensor_ops]] — M5-only `GGML_METAL_HAS_TENSOR`
  path; revisit when the M5 Studio arrives.
- `crates/moeflux-metal/tests/gather_mm_id_diff.rs` — the 5
  passing kernel diff tests (the load-bearing correctness
  evidence for "not the kernel").
- `feedback_pivot_on_discovery.md` — applied at the bench-result-
  surprised-us moment.
- `feedback_reboot_on_gpu_weirdness.md` — applied to the cold-
  boot panic mid-session (Metal flake, second run worked).

# What the session was actually about

Mike noted mid-session: this work serves a downstream Agora
alignment-canary failsafe. The goal isn't pure benchmarking — it's
having a viable offline path for an agent council that probes
hosted models for alignment consistency. If hosted Claude becomes
unavailable or compromised, this stack needs to support a 671B-
class model on commodity hardware. Closing the prefill gap is one
load-bearing piece of that runway.
