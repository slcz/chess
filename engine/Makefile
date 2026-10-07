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
LIB_SRCS  = $(RULES_SRCS) $(NN_SRCS) src/puct.c src/ffi.c
LIB_HDRS  = include/chess.h include/chess_ffi.h include/chess_nn.h include/chess_puct.h
TEST_SRCS = tests/test_units.c tests/suite_loader.c
NN_TEST_SRCS = tests/test_nn.c
PUCT_TEST_SRCS = tests/test_puct.c
PUCT_TEST_SRCS_ALL = $(RULES_SRCS) $(NN_SRCS) src/puct.c
PUCT_TEST_DEPS = $(PUCT_TEST_SRCS_ALL) include/chess.h include/chess_nn.h include/chess_puct.h

BUILD      = build
BUILD_ASAN = build-asan
BUILD_TSAN = build-tsan

# The NN smoke embeds Python (macOS CLT Python3.framework, the venv's ABI)
# and runs the real checkpoint through the IO thread. Manual target.
PYFW := /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9
PYINCLUDE := $(PYFW)/Headers
PYLIBDIR  := $(PYFW)/lib
PURELIB   := $(shell ../.venv/bin/python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)
PYFLAGS    = -I$(PYINCLUDE) -L$(PYLIBDIR) -lpython3.9 -Wl,-rpath,$(abspath $(PYFW)/../../..) \
             -DPUCT_REPO_ROOT='"$(abspath ..)"' -DPUCT_PURELIB='"$(PURELIB)"'

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

# The PUCT search test links rules + encoder + search.
$(BUILD)/test_puct: $(PUCT_TEST_SRCS) $(PUCT_TEST_DEPS) | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(PUCT_TEST_SRCS) $(PUCT_TEST_SRCS_ALL) -o $@ -lm

# --- sanitizer build ---

$(BUILD_ASAN):
	mkdir -p $(BUILD_ASAN)

$(BUILD_TSAN):
	mkdir -p $(BUILD_TSAN)

$(BUILD_ASAN)/perft: src/perft.c $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.c tests/suite_loader.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) $(SUITE_PATH_DEFINE) -fsanitize=address,undefined -fno-omit-frame-pointer src/perft.c $(RULES_SRCS) tests/suite_loader.c -o $@

$(BUILD_ASAN)/test_units: $(TEST_SRCS) $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) $(SUITE_PATH_DEFINE) -fsanitize=address,undefined -fno-omit-frame-pointer $(TEST_SRCS) $(RULES_SRCS) -o $@

$(BUILD_ASAN)/test_nn: $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) include/chess.h include/chess_nn.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) -fsanitize=address,undefined -fno-omit-frame-pointer $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) -o $@

$(BUILD_ASAN)/test_puct: $(PUCT_TEST_SRCS) $(PUCT_TEST_DEPS) | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) -fsanitize=address,undefined -fno-omit-frame-pointer $(PUCT_TEST_SRCS) $(PUCT_TEST_SRCS_ALL) -o $@ -lm

$(BUILD_TSAN)/test_puct: $(PUCT_TEST_SRCS) $(PUCT_TEST_DEPS) | $(BUILD_TSAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) -fsanitize=thread -fno-omit-frame-pointer $(PUCT_TEST_SRCS) $(PUCT_TEST_SRCS_ALL) -o $@ -lm

# The NN smoke links the search + the embedded-Python evaluator.
$(BUILD)/test_puct_nn: tests/test_puct_nn.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) include/chess_nn_eval.h | $(BUILD)
	$(CC) -O2 -Wall -std=c11 $(INCLUDES) $(PYFLAGS) tests/test_puct_nn.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) -o $@ -lm

# Match driver: one game between two search configs (see src/match_puct.c).
$(BUILD)/match_puct: src/match_puct.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) include/chess_nn_eval.h | $(BUILD)
	$(CC) -O2 -Wall -std=c11 $(INCLUDES) $(PYFLAGS) src/match_puct.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) -o $@ -lm

# UCI adapter for external matches (vs Stockfish; see src/uci_puct.c).
$(BUILD)/uci_puct: src/uci_puct.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) include/chess_nn_eval.h | $(BUILD)
	$(CC) -O2 -Wall -std=c11 $(INCLUDES) $(PYFLAGS) src/uci_puct.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) -o $@ -lm

# Search-convergence probe: argmax move per FEN per sim level (src/puct_probe.c).
$(BUILD)/puct_probe: src/puct_probe.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) include/chess_nn_eval.h | $(BUILD)
	$(CC) -O2 -Wall -std=c11 $(INCLUDES) $(PYFLAGS) src/puct_probe.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) -o $@ -lm

# --- gates ---

check: $(BUILD)/test_units $(BUILD)/test_nn $(BUILD)/test_puct $(BUILD)/perft
	$(BUILD)/test_units
	$(BUILD)/test_nn
	$(BUILD)/test_puct
	$(BUILD)/perft suite

check-asan: $(BUILD_ASAN)/test_units $(BUILD_ASAN)/test_nn $(BUILD_ASAN)/test_puct $(BUILD_ASAN)/perft
	$(BUILD_ASAN)/test_units
	$(BUILD_ASAN)/test_nn
	$(BUILD_ASAN)/test_puct
	$(BUILD_ASAN)/perft suite
# (LeakSanitizer is unavailable on macOS/arm64; it runs on the Linux cloud
#  containers — add ASAN_OPTIONS=detect_leaks=1 there.)

clean:
	rm -rf $(BUILD) $(BUILD_ASAN) $(BUILD_TSAN)

# Concurrent match vs limited-Strength Stockfish children (one per game).
$(BUILD)/sf_gauntlet: src/sf_gauntlet.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) include/chess_nn_eval.h | $(BUILD)
	$(CC) -O2 -Wall -std=c11 $(INCLUDES) $(PYFLAGS) src/sf_gauntlet.c src/nn_eval.c $(PUCT_TEST_SRCS_ALL) -o $@ -lm
