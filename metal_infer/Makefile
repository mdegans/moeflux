# Makefile for moeflux/metal_infer — streaming-experts MoE inference
#
# Targets:
#   make          — build metal_infer + infer binaries
#   make run      — single expert forward pass
#   make verify   — verify Metal vs CPU reference
#   make bench    — benchmark single expert (10 iterations)
#   make moe      — full MoE forward pass (K experts, single layer)
#   make moebench — benchmark MoE (10 iterations)
#   make full     — full 60-layer MoE forward pass (K=4)
#   make fullbench — benchmark full 60-layer forward (3 iterations)
#   make clean    — remove build artifacts
#
# Upstream had a `chat` HTTP/SSE client target — stripped in the
# moeflux fork since drama_llama drives generation. See NOTES.md.
#
# A `libmoeflux.a` static-library target will be added in Phase 3b.4
# once the mf_* C API surface is in place.
#
# Note: Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

TARGET = metal_infer
MAIN_SRC = main.m

# Optional: offline shader compilation (faster startup, but not required)
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = shaders.metal
SHADER_AIR = shaders.air
SHADER_LIB = shaders.metallib

# Inference engine (complete forward pass)
INFER_TARGET = infer
INFER_SRC = infer.m

# Static library for Rust FFI (drama_llama's MoefluxDecoder).
# Built with -DMOEFLUX_LIB so infer.m's main() is excluded.
LIB_TARGET = libmoeflux.a
LIB_OBJ = infer_lib.o

.PHONY: all clean run verify bench moe moebench full fullbench fast metallib infer infer-run lib

all: $(TARGET) $(INFER_TARGET)

# Build the binary (shaders compiled at runtime from source)
$(TARGET): $(MAIN_SRC) $(SHADER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(MAIN_SRC) -o $(TARGET)

# Optional: pre-compile shaders (not required — runtime compilation is the default)
metallib: $(SHADER_LIB)

$(SHADER_AIR): $(SHADER_SRC)
	$(METALC) -c $(SHADER_SRC) -o $(SHADER_AIR)

$(SHADER_LIB): $(SHADER_AIR)
	$(METALLIB_TOOL) $(SHADER_AIR) -o $(SHADER_LIB)

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
	rm -f $(TARGET) $(INFER_TARGET) $(LIB_TARGET) $(LIB_OBJ) $(SMOKE_TARGET) $(SHADER_AIR) $(SHADER_LIB)

# Run targets
run: $(TARGET)
	./$(TARGET) --layer 0 --expert 0

verify: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --verify

fast: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --verify

bench: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --benchmark

moe: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe

moebench: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe --benchmark

full: $(TARGET)
	./$(TARGET) --fast --full --k 4

fullbench: $(TARGET)
	./$(TARGET) --fast --full --k 4 --benchmark

# Inference engine targets
build-infer: $(INFER_TARGET)

infer-run: $(INFER_TARGET)
	./$(INFER_TARGET) --prompt "Hello, what is" --tokens 20 --k 4
