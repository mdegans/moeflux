# Prefill arc — session 14 plan: full-attn GPU migration + buffer pooling

The "kernel arc" is renamed: sessions 9–13 tuned prefill *kernels*; the
session-13 profile shows the prefill bottleneck is not kernels at all.
This is the prefill arc.

## Context — the session-13 profile

`profile.py --model a3b --prompt-file prefill_prompt_long.txt
--max-tokens 1` (samply, 15,692-token prefill, 58.5 s). **Prefill is
CPU/memory-bound, not GPU-bound:**

| self-time | symbol | what it is |
|---|---|---|
| **41.1%** | `_platform_memmove` | host-side data copying |
| **26.1%** | `pread` | streaming expert weights from SSD |
| 7.9% | `rms_norm_per_head_cpu` | QK-norm **on the CPU** |
| 5.4% | `__bzero` | zeroing buffers |
| **2.5%** | `__psynch_cvwait` | **waiting on the GPU** |

Inclusive: `batched_full_attn_layer_forward` **56.9%**,
`batched_linear_attn_layer_forward` 42.5%, `apply_rotary_emb` 11.6%
(0.5% self → its cost is memmove), `newBufferWithBytes` **10.7%**
(fresh Metal buffers created per full-attn layer),
`ExpertFiles::read_expert` 6.6%.

**Diagnosis:** the full-attention layer round-trips host↔device every
layer — GPU matmul → copy to host → CPU QK-norm → CPU RoPE → re-create
Metal buffers (`MtlBuffer::with_data`) → GPU SDPA → copy back. That
churn is ~70% of prefill. The linear-attn path got the graph-mode /
GPU-resident treatment in sessions 9–12; **full-attn never did** (Mike
expected it had — first task is to find out why/what state it's in).
The kernels we tuned sessions 9–13 are fast and *starved*.

## Axis 1 (priority) — migrate `batched_full_attn_layer_forward` to GPU

Goal stated by Mike: **bring host↔device copies to zero** — allocate
once, one contiguous chunk, reuse buffers across layers; keep the
residual stream / K / V GPU-resident across the whole layer.

Scope (multi-file architectural refactor ⇒ `feedback_design_before_execute`:
design conversation → plan mode → execute):

0. **Investigate first.** Read `crates/moeflux/src/riir/attn/full_attn_forward.rs`
   (the round-trip is around the SdpaCall callsite, ~`:700–746`:
   `MtlBuffer::with_data` for q/k/v, host `copy_from_slice` stacking,
   then host-side gate). Establish the exact current GPU/CPU split and
   why the linear-attn migration didn't carry over.
1. **GPU QK-norm.** `rms_norm_per_head_cpu` → GPU. `Op::RmsNormQkNTokens`
   already exists (session 11 gave it `per_token_total`) — wire full-attn
   to it instead of the CPU path.
2. **GPU RoPE.** `apply_rotary_emb` is on CPU. Check whether a GPU RoPE
   Op exists (the cmdbuf-consolidation plan listed "GPU per-head Q/K
   norm + RoPE" as planned); build or wire it.
3. **Buffer pooling.** Stop `MtlBuffer::with_data` per layer
   (`newBufferWithBytes` 10.7%). Route through `MetalBufferPool` /
   the `BufferPool` trait — the same pool linear-attn's graph2 uses.
   **Critical, Mike-flagged:** the pool has **no free path** — a design
   issue to fix as part of this. Allocate-once + lifetime-colored reuse
   (`commit_plan`) is the destination ([[feedback_pool_everything_destination]]);
   the missing free path must be designed in, not bolted on.
4. Keep K/V + residual GPU-resident — no `to_vec` / re-upload per layer.

Verify: canary battery (`diff_oracle` 12/12, esp.
`eval_prompt_matches_per_token_oracle`) + re-run `profile.py` and
confirm memmove / `newBufferWithBytes` collapse and GPU-wait rises.

## Axis 2 (second) — expert weight streaming

`pread` 26%, `ExpertFiles::read_expert` 6.6%; the profile's
`moeflux_prefetch` event showed `prefill_hits: 0` — the expert prefetch
did nothing this run. Experts stream from SSD per chunk. A persistent /
warm expert cache. Harder — memory-capacity-bound — and partly separable
from Axis 1. Tackle after Axis 1 lands, or in parallel if Axis 1 stalls.

## Re-measure command

```bash
cd ~/Projects/drama_llama
./profile.py --model a3b --prompt-file prefill_prompt_long.txt \
  --max-tokens 1 --duration 180 --top 35
```

## Note

Don't re-investigate async-copy or the `v3` matvec — both ruled out,
see [[kernel_arc_session13_landed]]. SDPA is at its structural floor;
the GQA-fold (1.3×) and vec4 staging shipped session 13.
