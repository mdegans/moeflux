# Perf arc — next session plan (updated 2026-05-22, end of session 2)

## Current state

### Prefill (~closed)
- **moeflux:** 775 tok/s (a3b, ~15k tokens)
- **llama.cpp:** 810-820 tok/s
- **Gap:** ~1.05× — diminishing returns

### Generation (next focus)
- **moeflux:** ~11.5 tok/s (a3b)
- **llama.cpp:** unknown — measure first
- Decode is bandwidth-bound (matvec), not compute-bound (matmul).
  Bottleneck profile will look completely different from prefill.

## Step 1: llama.cpp generation baseline

Measure llama.cpp decode tok/s on the same model + prompt for
the gap size.

## Step 2: Metal capture on decode

Single-layer capture during decode (not prefill). Key ops to watch:
- `dequant_matvec_4bit*` — weight dequant + matvec (likely dominant)
- `gated_delta_net_step` — single-token delta-net recurrence
- MoE routing + expert dispatch
- KV cache append (SDPA layers)
- Norms, residuals, overhead

## Step 3: attack the dominant op

Decode optimization levers are different from prefill:
- Memory bandwidth utilization (are we near the ~400 GB/s ceiling?)
- Weight layout / dequant efficiency
- Command buffer batching (per-layer overhead matters more at 1 tok)
- CPU-side dispatch latency

## Prefill remaining work (lower priority)

- Promote sequential delta-net to vA (slot rotation)
- qmm_t tile-size sweep (was 29% of prefill GPU time)
- Fresh prefill GPU capture with sequential kernel

## Dead ends (don't revisit)

- GQA fold on direct-device SDPA: 3% slower
- SDPA staging kernel: 7× slower
- Chunkwise delta-net: 15% slower than sequential
- LTO + codegen-units=1: neutral at 9× build time

## Env var cheat sheet

| Var | Default | Effect |
|-----|---------|--------|
| `MOEFLUX_SDPA_VB` | OFF | Staging SDPA kernel |
| `MOEFLUX_SDPA_GQA` | OFF | GQA fold on direct-device |
| `MOEFLUX_DELTA_NET_VB` | OFF | Sequential delta-net kernel |
| `MOEFLUX_MOE_GATHER_ID` | ON | gather_mm_id MoE kernel |

## Cross-references

- [[sdpa_vb_direct_device]] — SDPA arc history
- [[delta_net_sequential_session]] — delta-net sequential findings
- [[sdpa_session_learnings]] — SDPA what worked / didn't
