---
name: moeflux-hardening-arc-plan
description: Three-session arc to harden moeflux against the producer-wiring class of bug that session 19 introduced (wrong BufId fed to Op::MoeGatherIdFuse). Sessions A (rename) / B (newtype wrappers) / C (engine-level diff harness). Independent — any order works.
metadata:
  type: project
---

# Why this arc exists

Session 19 landed a kernel + producer pair (`Op::MoeGatherIdFuse` +
`gather_mm_id.metal`) with kernel-level diff oracle green and bench
green. The producer site at `linear_attn_forward.rs:2796` passed
`h_mid: h_mid_id` (pre-RmsNorm residual) into a parameter literally
named `h_mid` on the Op — which the kernel uses as the post-norm
activation for gate/up matmuls. End-to-end output was `</think>` +
EoT. Session 20 found and fixed it (one-line swap to `h_post_id`).
See [[prefill-gather-id-session-19-landed]] (RESOLVED) for the full
trace.

The bug had three contributing factors. This arc addresses each:

1. **Naming trap.** Op field name `h_mid` coincided with
   `MoeGraphScratch::h_mid` (the pre-norm residual buffer), making
   the wrong wiring read natural at the call site.
2. **Type-system blindness.** `BufId` is just an index — no
   type-level distinction between pre-norm / post-norm / routing /
   KV-cache buffers. The compiler can't object to any wiring.
3. **Testing gap.** Kernel diff oracle covers kernel math
   (synthetic input). It is *structurally incapable* of catching
   producer wiring mistakes — no engine-level integration test ran
   in the "is this done?" gate.

# The three sessions

The three are independent — no ordering constraint. Pick by energy
and design state, not dependencies.

## Session A — rename the trap field

**Goal.** Rename `Op::MoeGatherIdFuse::h_mid` (and the
`encode_moe_gather_id_fuse` encoder param) to `mlp_in` (or whichever
unambiguous name design picks). Removes the proximate trap.

**Scope.** ~4-5 site search-replace across:
- `crates/moeflux/src/riir/backend/mod.rs` (Op definition,
  `reads()` arm, `writes()` arm? probably not the writes arm —
  `h_mid` is read-only there).
- `crates/moeflux/src/riir/backend/gpu/mod.rs` (the dispatch arm).
- `crates/moeflux/src/riir/backend/cpu/mod.rs` (the `todo!()`
  stub if it names the param).
- `crates/moeflux/src/riir/moe/expert_forward.rs`
  (`encode_moe_gather_id_fuse` param + the doc-comment that
  references it).
- `crates/moeflux/src/riir/attn/linear_attn_forward.rs:2796`
  (the producer call site).

**Success gate.** Build clean, canary 9/9, env=on coherence curl
(per [[feedback-coherence-test-before-pipeline-commit]]) still
returns "The capital of France is Paris."

**Design risk.** None. Pure refactor. Plan-agent unnecessary.

## Session B — selective BufId newtype wrappers

**Goal.** Add semantic newtype wrappers (e.g.,
`struct PostNormBuf(BufId);`) at producer/consumer boundaries
where the cost of wrong-wiring is high (MoE inputs primarily).
The newtype prevents the *class* of wiring bug — not just the
specific Op field that bit us.

**Scope.** TBD — design conversation needed. Open questions the
session-B plan-mode pass should resolve:
- Which BufIds get wrapped? (MoE input is the obvious one.
  Routing? KV-cache? Residual? Where's the line?)
- Constructor shape — `PostNormBuf::from_buf_id(BufId)` (explicit,
  ugly) vs `PostNormBuf(BufId)` (terse, but allows accidental
  wrapping)?
- Does `RmsNorm`'s output field become `PostNormBuf` (consumer
  obligation enforced at the Op level), or do we wrap only at the
  Op-input boundary?
- Backend trait impact — `pool.handle(buf: BufId)` — does the
  pool need to know about wrappers, or only the Op layer?

**Success gate.** New wrappers in place for at least the MoE-input
boundary. Building any graph with a wrong-typed BufId fails to
compile. Canary 9/9 still green.

**Design risk.** Medium. Could intrude into the `Backend` trait
surface if done over-eagerly. Plan-agent should explicitly bound
the surface area.

## Session C — engine-level diff harness

**Goal.** A test harness that runs both `Op::MoeBatchedPermuteFuse`
and `Op::MoeGatherIdFuse` against *real engine state* (real model
weights mmap'd, real routing flow) and compares per-token
per-channel output. Catches producer wiring bugs of the session-19
class structurally — not just the BufId-mix-up variant the newtypes
catch.

**Scope.** TBD — design conversation needed. Open questions the
session-C plan-mode pass should resolve:
- Does the harness load a real model file, or mock the
  ExpertFiles + LayerWeightCache with synthetic-but-shape-correct
  data?
- A/B mechanism — env-flag (matches current production path) or
  explicit-construct-two-graphs (cleaner for the test)?
- Comparison threshold — bit-exact (current production matches
  this) or cosine ≥ 0.9999?
- How many layers? Just one (cheap, isolates the bug), or N
  (accumulates error, catches subtle drift)?
- Runtime budget — should this run in CI on every PR, or be
  gated to perf/correctness branches?

**Success gate.** Harness runs in seconds; reverting the session-20
fix makes it fail with a clear per-token-per-channel error; new
test added to the standard moeflux test suite.

**Design risk.** Medium. The setup of real engine state is the
expensive part; the comparison is trivial. Plan-agent should
estimate the engine-state-setup LOC honestly.

# Order recommendation

A first — it's cheap and removes the immediate trap. After that,
B and C are roughly equivalent in value:
- B catches a *class* of bug at compile time (any future BufId
  mix-up at wrapped boundaries).
- C catches a *broader class* of bug at test time (any producer
  wiring mistake involving wrapped or unwrapped BufIds, layout
  assumptions, missing uploads, etc.).

If forced to pick one, C has the broader catch radius. But B is
the pretty-code move ([[feedback-pretty-is-a-goal]]); each
reinforces the other.

# Cross-references

- [[prefill-gather-id-session-19-landed]] — the bug this arc
  exists to prevent recurrence of (RESOLVED in session 20).
- [[feedback-coherence-test-before-pipeline-commit]] — the
  discipline memo that complements C structurally.
- [[feedback-design-before-execute]] — applied to B and C
  (plan-mode pass before implementation).
- [[feedback-pretty-is-a-goal]] — applied to B (newtype
  ergonomics).

# Status

- **2026-05-20 session 20:** arc doc landed. Session A executed
  in the same session (commit `ae8c71b` — rename `h_mid → mlp_in`
  in `Op::MoeGatherIdFuse`).
- **2026-05-20 design pass with Mike:** Session B scope expanded
  from narrow (one wrapper) to full typed-`BufId` refactor. The
  narrow plan ([[moeflux-hardening-session-b-plan]]) is
  **superseded** by [[moeflux-hardening-session-b-v2-plan]]
  — every `BufId` in the backend gets a role tag; producer-wiring
  bugs become compile errors. ~510 net LOC, one focused session.
- Session C ([[moeflux-hardening-session-c-plan]]) — engine-level
  diff harness — remains independent. Catches the residual class
  of bugs the type system can't (e.g., wrong RmsNorm output
  written into a typed slot upstream of the consumer).
