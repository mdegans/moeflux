# Makefile for moeflux/metal_infer — preserved for the differential
# regression oracle (`crates/moeflux` `diff-oracle` feature).
#
# Targets:
#   make infer    — build the standalone C/Objective-C inference engine
#   make lib      — build libmoeflux.a (linked by moeflux-sys's build.rs)
#   make clean    — remove build artifacts
#
# Phase 6 cutover (2026-04-28) moved the source-of-truth Metal kernels
# to `../crates/moeflux/shaders/shaders.metal`. The standalone targets
# below read shaders at runtime via MTLDevice newLibraryWithSource:;
# point MOEFLUX_SHADERS_PATH at the new shader location before running.
#
# Note: Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

# Model variant selection. Defaults to Qwen3.5-A17B (upstream's target);
# set MODEL=qwen3-6-35b-a3b on the command line to build for the smaller
# Qwen3.6-35B-A3B shape. See model_variant.h for the full list and
# docs/model_variants.md for how to add a new one.
MODEL ?= qwen3-5-a17b
ifeq ($(MODEL),qwen3-5-a17b)
MODEL_DEFINE = -DMOEFLUX_MODEL_QWEN3_5_A17B
else ifeq ($(MODEL),qwen3-6-35b-a3b)
MODEL_DEFINE = -DMOEFLUX_MODEL_QWEN3_6_35B_A3B
else
$(error Unknown MODEL=$(MODEL). Supported: qwen3-5-a17b, qwen3-6-35b-a3b)
endif
CFLAGS += $(MODEL_DEFINE)

# Inference engine (complete forward pass)
INFER_TARGET = infer
INFER_SRC = infer.m

# Static library for Rust FFI (drama_llama's MoefluxDecoder).
# Built with -DMOEFLUX_LIB so infer.m's main() is excluded.
LIB_TARGET = libmoeflux.a
LIB_OBJ = infer_lib.o

.PHONY: all clean infer infer-run lib

all: $(INFER_TARGET) $(LIB_TARGET)

# Build the inference engine
$(INFER_TARGET): $(INFER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(INFER_SRC) -o $(INFER_TARGET)

# Build the static library (main() excluded via -DMOEFLUX_LIB).
# Consumers link this alongside Metal, Foundation, Accelerate, and
# -lpthread -lcompression (same linker flags as the infer binary).
$(LIB_OBJ): $(INFER_SRC) moeflux.h
	$(CC) $(CFLAGS) $(FRAMEWORKS) -DMOEFLUX_LIB -c $(INFER_SRC) -o $(LIB_OBJ)

$(LIB_TARGET): $(LIB_OBJ)
	ar rcs $(LIB_TARGET) $(LIB_OBJ)

lib: $(LIB_TARGET)

# C smoke test — links against libmoeflux.a. Does not actually run
# during `make`; invoke via ./smoke <weights> <manifest> <vocab>
# <experts_dir>. See ../tests/smoke.c.
SMOKE_TARGET = smoke
SMOKE_SRC = ../tests/smoke.c

$(SMOKE_TARGET): $(SMOKE_SRC) $(LIB_TARGET) moeflux.h
	$(CC) -O2 -Wall -I. $(SMOKE_SRC) $(LIB_TARGET) $(FRAMEWORKS) $(LDFLAGS) -o $(SMOKE_TARGET)

clean:
	rm -f $(INFER_TARGET) $(LIB_TARGET) $(LIB_OBJ) $(SMOKE_TARGET)

# Inference engine targets
build-infer: $(INFER_TARGET)

infer-run: $(INFER_TARGET)
	./$(INFER_TARGET) --prompt "Hello, what is" --tokens 20 --k 4
