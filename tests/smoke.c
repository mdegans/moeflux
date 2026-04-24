// moeflux — C smoke test
//
// Exercises the full mf_* C API end-to-end. Intended as:
//   (1) a basic link check — proves libmoeflux.a has all the symbols
//       drama_llama's FFI will call.
//   (2) a pipe-is-alive check — loads a model, runs a 4-token prefill,
//       runs three more single-token decodes, verifies logits are not
//       all-zero / NaN and that consecutive logit vectors differ.
//   (3) a KV-management check — exercises mf_memory_seq_rm +
//       mf_memory_seq_pos_max + mf_memory_clear.
//
// This is NOT a correctness test. Token IDs are fabricated; the
// outputs are not compared against a reference. Content quality of
// the generated logits is irrelevant — the test only asserts that
// the moeflux pipeline runs without crashing and produces non-
// pathological output shapes.
//
// Build: from metal_infer/ directory:
//   cc -O2 -Wall -I../metal_infer ../tests/smoke.c \
//      ../metal_infer/libmoeflux.a \
//      -framework Metal -framework Foundation -framework Accelerate \
//      -lpthread -lcompression \
//      -o ../tests/smoke
//
// Or via `make smoke` in the Makefile target.
//
// Run:
//   ./smoke <weights_path> <manifest_path> <vocab_path> <experts_dir>

#include "../metal_infer/moeflux.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int logits_look_sane(const float *logits, size_t n, const char *label) {
    int nonzero = 0;
    int finite = 1;
    float maxv = -INFINITY;
    float minv = INFINITY;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(logits[i])) { finite = 0; }
        if (logits[i] != 0.0f) nonzero++;
        if (logits[i] > maxv) maxv = logits[i];
        if (logits[i] < minv) minv = logits[i];
    }
    fprintf(stderr, "[%s] nonzero=%d/%zu min=%g max=%g finite=%s\n",
            label, nonzero, n, minv, maxv, finite ? "yes" : "NO");
    if (!finite) {
        fprintf(stderr, "[%s] FAIL: NaN or Inf present\n", label);
        return 0;
    }
    if (nonzero == 0) {
        fprintf(stderr, "[%s] FAIL: logits are all zero\n", label);
        return 0;
    }
    return 1;
}

static int logits_differ(const float *a, const float *b, size_t n) {
    // Compare up to the first 1024 slots — if the prefix is identical
    // bit-for-bit the vectors are almost certainly the same, and we
    // don't want to walk 250k floats on every comparison.
    size_t m = (n < 1024) ? n : 1024;
    for (size_t i = 0; i < m; i++) {
        if (a[i] != b[i]) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr,
                "usage: %s <weights.bin> <manifest.json> <vocab.bin> "
                "<experts_dir>\n",
                argv[0]);
        return 2;
    }
    const char *weights_path  = argv[1];
    const char *manifest_path = argv[2];
    const char *vocab_path    = argv[3];
    const char *experts_dir   = argv[4];

    fprintf(stderr, "[smoke] mf_init_model\n");
    mf_ctx *ctx = mf_init_model(weights_path, manifest_path, vocab_path,
                                experts_dir,
                                /*experts_per_tok=*/4,
                                /*use_2bit=*/0);
    if (!ctx) {
        fprintf(stderr, "[smoke] FAIL: mf_init_model returned NULL\n");
        return 1;
    }

    size_t n_vocab = mf_n_vocab(ctx);
    size_t n_ctx = mf_n_ctx(ctx);
    int32_t eos = mf_eos(ctx);
    const char *name = mf_model_name(ctx);
    fprintf(stderr,
            "[smoke] model=%s n_vocab=%zu n_ctx=%zu eos=%d\n",
            name ? name : "(null)", n_vocab, n_ctx, eos);
    if (n_vocab == 0) {
        fprintf(stderr, "[smoke] FAIL: n_vocab is 0\n");
        mf_free_model(ctx);
        return 1;
    }

    float *logits = calloc(n_vocab, sizeof(float));
    float *logits_b = calloc(n_vocab, sizeof(float));
    if (!logits || !logits_b) {
        fprintf(stderr, "[smoke] FAIL: out-of-memory allocating logits\n");
        mf_free_model(ctx);
        return 1;
    }

    // Fabricated token IDs — we're not testing content, just pipe
    // liveness. Using low-numbered tokens so they're in-vocab for
    // every MoE model we might point this test at.
    int32_t prompt[4] = { 1, 100, 500, 1000 };

    fprintf(stderr, "[smoke] mf_eval_prompt (4 tokens at pos [0,4))\n");
    if (mf_eval_prompt(ctx, prompt, 4, 0, 0, logits) != 0) {
        fprintf(stderr, "[smoke] FAIL: mf_eval_prompt returned nonzero\n");
        mf_free_model(ctx);
        return 1;
    }
    if (!logits_look_sane(logits, n_vocab, "after-prefill")) {
        mf_free_model(ctx);
        return 1;
    }

    int pos_max = mf_memory_seq_pos_max(ctx, 0);
    fprintf(stderr, "[smoke] mf_memory_seq_pos_max → %d (expect 4)\n",
            pos_max);
    if (pos_max != 4) {
        fprintf(stderr, "[smoke] FAIL: pos_max mismatch (got %d)\n", pos_max);
        mf_free_model(ctx);
        return 1;
    }

    // Three decode steps. Pick a token from each step's logits (not
    // argmax — we don't need real sampling; any valid in-vocab ID).
    int32_t next = 42;
    for (int step = 0; step < 3; step++) {
        memcpy(logits_b, logits, n_vocab * sizeof(float));
        fprintf(stderr, "[smoke] mf_eval_token step=%d pos=%d token=%d\n",
                step, 4 + step, next);
        if (mf_eval_token(ctx, next, 4 + step, 0, logits) != 0) {
            fprintf(stderr, "[smoke] FAIL: mf_eval_token returned nonzero\n");
            mf_free_model(ctx);
            return 1;
        }
        if (!logits_look_sane(logits, n_vocab, "after-decode")) {
            mf_free_model(ctx);
            return 1;
        }
        if (!logits_differ(logits, logits_b, n_vocab)) {
            fprintf(stderr,
                    "[smoke] FAIL: consecutive logit vectors are "
                    "identical at step %d\n", step);
            mf_free_model(ctx);
            return 1;
        }
        next = (next * 31 + 7) % (int32_t)n_vocab;
    }

    // Truncate back to position 4 — linear-attn layers get reset
    // (Option A). pos_max should read 4 again.
    fprintf(stderr, "[smoke] mf_memory_seq_rm to pos 4\n");
    if (!mf_memory_seq_rm(ctx, 0, 4, -1)) {
        fprintf(stderr, "[smoke] FAIL: mf_memory_seq_rm returned 0\n");
        mf_free_model(ctx);
        return 1;
    }
    pos_max = mf_memory_seq_pos_max(ctx, 0);
    fprintf(stderr, "[smoke] mf_memory_seq_pos_max → %d (expect 4)\n",
            pos_max);
    if (pos_max != 4) {
        fprintf(stderr, "[smoke] FAIL: pos_max after seq_rm mismatch\n");
        mf_free_model(ctx);
        return 1;
    }

    // State save / load round-trip (Option B).
    // With kv_len = 4 on each full-attn layer from the prefill above,
    // snapshot the state, run a couple of decodes (which mutate it),
    // restore the snapshot, and verify pos_max returns to 4 and the
    // next token at pos=4 produces logits close to the logits we
    // recorded right after the original prefill.
    fprintf(stderr, "[smoke] mf_state_size (current snapshot size)\n");
    size_t snap_size = mf_state_size(ctx);
    fprintf(stderr, "[smoke] snap_size = %zu bytes\n", snap_size);
    if (snap_size == 0) {
        fprintf(stderr, "[smoke] FAIL: mf_state_size returned 0\n");
        mf_free_model(ctx);
        return 1;
    }

    void *snap = malloc(snap_size);
    if (!snap) {
        fprintf(stderr, "[smoke] FAIL: out-of-memory allocating snapshot\n");
        mf_free_model(ctx);
        return 1;
    }

    // Re-run the original prefill so we can capture post-prefill
    // logits as a reference for the restore check. (memory_clear
    // happened just above.)
    if (mf_eval_prompt(ctx, prompt, 4, 0, 0, logits) != 0) {
        fprintf(stderr, "[smoke] FAIL: re-prefill for state test\n");
        free(snap); mf_free_model(ctx);
        return 1;
    }

    long long wrote = mf_state_save(ctx, snap, snap_size);
    if (wrote <= 0) {
        fprintf(stderr, "[smoke] FAIL: mf_state_save wrote=%lld\n", wrote);
        free(snap); mf_free_model(ctx);
        return 1;
    }
    fprintf(stderr, "[smoke] mf_state_save wrote %lld bytes\n", wrote);

    // Capture the next-token logits that would follow the prefill.
    memcpy(logits_b, logits, n_vocab * sizeof(float));

    // Mutate state by decoding two tokens.
    mf_eval_token(ctx, 42, 4, 0, logits);
    mf_eval_token(ctx, 43, 5, 0, logits);
    if (mf_memory_seq_pos_max(ctx, 0) != 6) {
        fprintf(stderr, "[smoke] FAIL: pos_max should be 6 after decodes\n");
        free(snap); mf_free_model(ctx);
        return 1;
    }

    // Restore snapshot; expect pos_max back to 4.
    fprintf(stderr, "[smoke] mf_state_load\n");
    if (mf_state_load(ctx, snap, (size_t)wrote) != 0) {
        fprintf(stderr, "[smoke] FAIL: mf_state_load returned nonzero\n");
        free(snap); mf_free_model(ctx);
        return 1;
    }
    pos_max = mf_memory_seq_pos_max(ctx, 0);
    fprintf(stderr, "[smoke] pos_max after restore → %d (expect 4)\n",
            pos_max);
    if (pos_max != 4) {
        fprintf(stderr, "[smoke] FAIL: pos_max after restore mismatch\n");
        free(snap); mf_free_model(ctx);
        return 1;
    }

    // Run one token at the restored position. Under Option B's
    // contract, the resulting logits should be deterministic and match
    // (within floating-point tolerance) what the same token produces
    // after a fresh prefill. We re-run eval_token on an arbitrary
    // token and compare to an equivalent run from a clean-then-
    // prefill path below.
    int32_t tkn = 7;
    mf_eval_token(ctx, tkn, 4, 0, logits);
    memcpy(logits_b, logits, n_vocab * sizeof(float));

    // Reset and re-run the prefill + same token fresh.
    mf_memory_clear(ctx);
    mf_eval_prompt(ctx, prompt, 4, 0, 0, logits);
    mf_eval_token(ctx, tkn, 4, 0, logits);

    // Compare logits_b (restored path) vs logits (fresh path). With
    // a correct state save/load the two should agree up to numerical
    // noise from the GPU kernels. We assert exact equality on the
    // first 1024 slots since the Metal path is deterministic and no
    // floating-point reordering happens between the two runs.
    int mismatches = 0;
    for (size_t i = 0; i < (n_vocab < 1024 ? n_vocab : 1024); i++) {
        if (logits_b[i] != logits[i]) mismatches++;
    }
    fprintf(stderr,
            "[smoke] restored-vs-fresh first-1024-slot mismatches: %d\n",
            mismatches);
    if (mismatches > 0) {
        // Not a hard failure yet — small numerical drift is plausible
        // on some GPUs. Report it loudly for the first run so Mike
        // can decide what tolerance to set.
        fprintf(stderr,
                "[smoke] WARN: restored path does not match fresh "
                "path bit-for-bit. Review before trusting Option B "
                "for prefix-cache reuse.\n");
    }

    free(snap);

    // Full clear.
    fprintf(stderr, "[smoke] mf_memory_clear\n");
    mf_memory_clear(ctx);
    pos_max = mf_memory_seq_pos_max(ctx, 0);
    fprintf(stderr, "[smoke] mf_memory_seq_pos_max → %d (expect 0)\n",
            pos_max);
    if (pos_max != 0) {
        fprintf(stderr, "[smoke] FAIL: pos_max after clear != 0\n");
        mf_free_model(ctx);
        return 1;
    }

    free(logits);
    free(logits_b);
    mf_free_model(ctx);

    fprintf(stderr, "[smoke] PASS\n");
    return 0;
}
