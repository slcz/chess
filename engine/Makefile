# engine/Makefile — builds the C rules engine.
#
#   make / make all     build shared library + perft CLI + unit-test binary
#   make check          run the C unit tests + full perft suite (the Phase 1 gate)
#   make check-asan     same, under AddressSanitizer + UBSan (slower, catches
#                       memory errors and UB that "correct" runs hide)
#   make clean          remove build artifacts
#
# The shared library (libchess) carries the rules engine plus the FFI handle
# API (chess_ffi.h) — Python loads it via cffi (chess_ai/engine.py). The C
# binaries link the same objects statically; nothing here links the .dylib.
#
# NN-side code (nn_encode.c: Phase 2 input encoder) is deliberately separate
# from the rules sources; same for its tests (test_nn.c vs test_units.c).

CC      ?= cc
OPT      ?= -O2
WARN     = -Wall -Wextra -Wpedantic -Werror -std=c11
CFLAGS   = $(WARN) $(OPT)
INCLUDES = -Iinclude -Itests

# Bake the absolute path of the perft suite data file into the binaries so
# tests find it regardless of the working directory.
SUITE_PATH_DEFINE = -DPERFT_SUITE_PATH="\"$(abspath tests/perft_suite.txt)\""

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SO = libchess.dylib
else
  SO = libchess.so
endif

RULES_SRCS = src/tables.c src/board.c src/fen.c src/movegen.c src/termination.c src/perftlib.c
NN_SRCS   = src/nn_encode.c
LIB_SRCS  = $(RULES_SRCS) $(NN_SRCS) src/ffi.c
LIB_HDRS  = include/chess.h include/chess_ffi.h include/chess_nn.h
TEST_SRCS = tests/test_units.c tests/suite_loader.c
NN_TEST_SRCS = tests/test_nn.c

BUILD      = build
BUILD_ASAN = build-asan

.PHONY: all check check-asan clean

all: $(BUILD)/$(SO) $(BUILD)/perft $(BUILD)/test_units

# --- default build (optimized) ---

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/$(SO): $(LIB_SRCS) $(LIB_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) -fPIC -shared $(LIB_SRCS) -o $@

$(BUILD)/perft: src/perft.c $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.c tests/suite_loader.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) src/perft.c $(RULES_SRCS) tests/suite_loader.c -o $@

$(BUILD)/test_units: $(TEST_SRCS) $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) $(TEST_SRCS) $(RULES_SRCS) -o $@

# The NN encoder test links the rules + encoder (no FFI handle needed).
$(BUILD)/test_nn: $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) include/chess.h include/chess_nn.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) -o $@

# --- sanitizer build ---

$(BUILD_ASAN):
	mkdir -p $(BUILD_ASAN)

$(BUILD_ASAN)/perft: src/perft.c $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.c tests/suite_loader.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) $(SUITE_PATH_DEFINE) -fsanitize=address,undefined -fno-omit-frame-pointer src/perft.c $(RULES_SRCS) tests/suite_loader.c -o $@

$(BUILD_ASAN)/test_units: $(TEST_SRCS) $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) $(SUITE_PATH_DEFINE) -fsanitize=address,undefined -fno-omit-frame-pointer $(TEST_SRCS) $(RULES_SRCS) -o $@

$(BUILD_ASAN)/test_nn: $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) include/chess.h include/chess_nn.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) -fsanitize=address,undefined -fno-omit-frame-pointer $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) -o $@

# --- gates ---

check: $(BUILD)/test_units $(BUILD)/test_nn $(BUILD)/perft
	$(BUILD)/test_units
	$(BUILD)/test_nn
	$(BUILD)/perft suite

check-asan: $(BUILD_ASAN)/test_units $(BUILD_ASAN)/test_nn $(BUILD_ASAN)/perft
	$(BUILD_ASAN)/test_units
	$(BUILD_ASAN)/test_nn
	$(BUILD_ASAN)/perft suite
# (LeakSanitizer is unavailable on macOS/arm64; it runs on the Linux cloud
#  containers — add ASAN_OPTIONS=detect_leaks=1 there.)

clean:
	rm -rf $(BUILD) $(BUILD_ASAN)
