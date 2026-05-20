---
name: moeflux-hardening-session-b-plan
description: Session B plan — selective BufId newtype wrappers. Wrap exactly ONE BufId class (post-norm MoE input) via `PostNormBuf` with private constructor. ~40 net LOC, one ~1-hour session. Designed by Plan agent in session 20.
metadata:
  type: project
---

# TL;DR

Wrap exactly **one** boundary BufId class — the **post-norm MoE input** (`MoeGraphScratch::h_post`) — in a private-constructor newtype `PostNormBuf`, plumbed end-to-end from `MoeGraphScratch::new` to `Op::MoeGatherIdFuse::mlp_in`. Nothing else. The session-19 bug is the only data point we have; the wrapper API stays tiny and we add more types as future bugs justify them.

Assumes Session A has landed (the field is `mlp_in`, not `h_mid`). Session A landed as commit `ae8c71b` (2026-05-20).

# Decisions

## 1. Which BufIds get wrapped?

**Wrap exactly one: the post-norm MoE input.**

- `MoeGraphScratch::h_post` becomes typed (`PostNormBuf`).
- `Op::RmsNormBf16NTokens::out` stays `BufId` (general-purpose op; typing it would force ~10 unrelated changes).
- `Op::MoeGatherIdFuse::mlp_in: PostNormBuf` (the consumer obligation that bit us).

**Not wrapped (yet):**
- Pre-norm residual (`MoeGraphScratch::h_mid`): correctly named, single consumer, no demonstrated bug class.
- Routing buffers, KV-cache buffers, scratch buffers: no demonstrated confusion class.

**Rationale.** The session-19 bug was specifically the `h_post_id` vs `h_mid_id` swap. That one swap is the only producer-wiring failure we have evidence for; wrapping more types now is speculation. Mike's standing rule: "we can always add more later." Add types per real bug, not per imagined risk.

## 2. Wrapper API shape

**Tuple struct with a private inner field, constructed via `from_rms_norm_out`:**

```rust
pub struct PostNormBuf(BufId);  // field private to the module

impl PostNormBuf {
    /// Construct from the `out` of an `Op::RmsNormBf16NTokens` that
    /// has been (or is about to be) pushed onto the graph. The caller
    /// vouches that `out` is the BufId named in that op's `out` field
    /// — there's no runtime check (the type-system value is the
    /// barrier to *accidental* wrong-wiring, not adversarial wiring).
    pub fn from_rms_norm_out(out: BufId) -> Self {
        Self(out)
    }

    /// Escape hatch. Use only when reading the inner id for a
    /// non-Op-API consumer (pool.download, debug dump, etc.).
    pub fn raw(self) -> BufId { self.0 }
}
```

**Why:**
- **Private inner field** prevents `PostNormBuf(any_bufid)` at random call sites.
- **No `Deref`** — would silently allow `&PostNormBuf` anywhere `&BufId` is expected, defeating the purpose.
- **`Copy + Clone + Debug + Eq + Hash`** — derive all, matches `BufId`.

## 3. Op definition impact

**Single field touched.**

```rust
Op::MoeGatherIdFuse {
    ...
    mlp_in: PostNormBuf,   // ← was BufId
    ...
}
```

`Op::reads()` needs `mlp_in.raw()` in the `vec!`. Trivial.

**No other Op variants change.** `MoeBatchedPermuteFuse::bucket_input` is the same data but bucket-permuted (different semantic) — no demonstrated bug, skip. `RmsNormBf16NTokens::out` stays `BufId` (10+ consumers, churn-vs-win wrong). `MoeCombineResidualNTokens::h_mid` correctly named.

## 4. Backend trait impact

**Zero changes to `Backend` / `BufferPool` traits.**

Pool stays untyped; one `.raw()` call inside the GPU dispatch arm. Adding a generic for one wrapper would be the "abstraction to look clever" anti-pattern.

CPU backend: `Op::MoeGatherIdFuse` is `todo!()` and stays a `todo!()`.

## 5. Producer code impact

**The wrap happens in `MoeGraphScratch::new`**, not at the `Op::RmsNormBf16NTokens` push site:

```rust
let h_post = PostNormBuf::from_rms_norm_out(p(chk(hidden), "mgs.h_post"));
```

Rationale: `MoeGraphScratch::new` is allocated once per run; the semantic claim "this BufId carries the post-norm activation" is made at the boundary the consumer reads. The producer at 2547 reads `let h_post: PostNormBuf = moe.h_post;` and passes it through; never constructs a fresh wrapper, never calls `.raw()` except inside the one `Op::MoeGatherIdFuse { mlp_in: h_post, ... }` push.

**End-to-end chain enforced:** the only way to populate `Op::MoeGatherIdFuse::mlp_in` is to get a `PostNormBuf`. The only place that creates one is `MoeGraphScratch::new` (one call per run). Wrong-wiring (`h_mid` for `mlp_in`) becomes a compile error because `MoeGraphScratch::h_mid` stays `BufId` and won't coerce.

**Call sites that need `.raw()`** (approx 7 total):
- `pool.download(moe.h_post.raw(), ...)` at `linear_attn_forward.rs:2560`
- `Op::MatvecNTokens { input: moe.h_post.raw(), ... }` — shared FFN gate/up (2 sites in linear-attn `moe_block_forward`).
- Router/shared-gate matvecs in both attention producers (`linear_attn_forward.rs:2436/2452`, `full_attn_forward.rs:923/939`).
- `Op::RmsNormBf16NTokens { out: moe.h_post.raw(), .. }` at line 2423 (linear-attn) and 910 (full-attn).

## 6. `MoeGraphScratch` struct

**One field changes:** `pub h_post: BufId` → `pub h_post: PostNormBuf`. Everything else stays `BufId`.

## 7. Migration plan

Order (one session, ~1 hour):

1. Define `PostNormBuf` in `backend/mod.rs` next to `BufId`. Derive `Copy, Clone, Debug, Eq, PartialEq, Hash`. Doc-comment narrating session-19 as rationale.
2. Change `Op::MoeGatherIdFuse::mlp_in` to `PostNormBuf` in enum.
3. Fix `Op::reads()` for that variant: `*mlp_in` → `mlp_in.raw()`.
4. Change `MoeGraphScratch::h_post` field type + constructor.
5. **Build.** Compiler walks you through every site that reads `moe.h_post`. Each needs `.raw()` except the one `Op::MoeGatherIdFuse { mlp_in: moe.h_post, ... }` push at 2802 — that's the win.
6. Update GPU dispatch: `self.pool.handle(*mlp_in)` → `self.pool.handle(mlp_in.raw())`.
7. CPU dispatch unchanged (`{ label, .. }` destructure).
8. Canary 9/9 + env=on coherence curl per [[feedback-coherence-test-before-pipeline-commit]].
9. Commit.

## 8. Out of scope (explicitly)

- Phantom types `BufId<Tag>` — rejected.
- Wrapping `bucket_input`, routing buffers, KV buffers.
- Changing `Pool` / `Backend` trait surface.
- Renaming `MoeGraphScratch::h_post` (Session A-style; independent).

# Migration Steps (numbered, with file:line targets)

| # | File:Line | Change |
|---|---|---|
| 1 | `backend/mod.rs:47` (after `BufId`) | Add `pub struct PostNormBuf(BufId);` + `from_rms_norm_out` / `raw` |
| 2 | `backend/mod.rs:602` | `mlp_in: BufId` → `mlp_in: PostNormBuf` |
| 3 | `backend/mod.rs:840` | `*mlp_in` → `mlp_in.raw()` in `reads()` |
| 4 | `attn/linear_attn_forward.rs:351` | `pub h_post: BufId` → `pub h_post: PostNormBuf` |
| 5 | `attn/linear_attn_forward.rs:451` | Wrap: `let h_post = PostNormBuf::from_rms_norm_out(p(chk(hidden), "mgs.h_post"));` |
| 6 | `attn/linear_attn_forward.rs:2547` | `let h_post_id = moe.h_post;` — re-type-infers to `PostNormBuf` |
| 7 | `linear_attn_forward.rs:2423, 2436, 2452, 2560, 2732, 2748` | `.raw()` at BufId-only sites |
| 8 | `linear_attn_forward.rs:2802` | The unwrapped pass-through: `mlp_in: h_post_id,` — THIS IS THE WIN |
| 9 | `full_attn_forward.rs:910, 923, 939` | `.raw()` polish for parallel producer |
| 10 | `backend/gpu/mod.rs:1156, 1176` | `self.pool.handle(*mlp_in)` → `self.pool.handle(mlp_in.raw())` |
| 11 | Build + canary + coherence curl | |
| 12 | Commit | |

# LOC Estimate

| File | Lines changed | Lines added |
|---|---|---|
| `backend/mod.rs` | 2 | ~25 (new type + doc) |
| `attn/linear_attn_forward.rs` | ~8 | 1 |
| `attn/full_attn_forward.rs` | ~3 | 0 |
| `backend/gpu/mod.rs` | 1 | 0 |
| **Total** | **~14** | **~26** |

Net ~40 LOC.

# Risk Surface

- **Compile-time only.** Type-system change, no runtime behavior change. If it builds, it runs identically.
- Test fixtures (`backend/mod.rs` tests) construct `MoeBatchedPermuteFuse`, not `MoeGatherIdFuse` — no fixture change needed.
- `PostNormBuf` derives `Copy` — no lifetime/borrow surprises.
- Rollback: single revert reverts to pre-change state.

**What this doesn't catch:**
- Wrong producer pushing wrong RMS-norm output into `moe.h_post` field.
- Layout mistakes (size mismatch).

Session C's engine-level diff harness covers those classes.

# Open Questions for Mike

1. **Naming.** `PostNormBuf` vs `MlpInputBuf` vs `MoeInputBuf`. The constructor is `from_rms_norm_out` — Plan agent leans `PostNormBuf`. 30-second confirm.
2. **`PostNormBuf::raw()` consume self or borrow?** Both work for `Copy`. Suggest value-self (matches `BufId`'s API).
3. **Should the *pre-attention* RMS-norm output (feeds QKV projections) also be `PostNormBuf`?** Same semantic. Recommendation: NO, keep minimal — extend later if Mike wants broader use.

All three are 30-second confirmations; plan executes either way.

# Cross-references

- [[moeflux-hardening-arc-plan]] — the arc this slots into
- [[prefill-gather-id-session-19-landed]] — the bug class this wrapper catches
- [[feedback-pretty-is-a-goal]] — ergonomic shape (private constructor, no Deref)
- [[feedback-design-before-execute]] — discipline applied (Plan agent before execution)
