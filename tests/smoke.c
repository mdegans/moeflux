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
