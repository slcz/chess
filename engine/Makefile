# engine/Makefile — builds the C rules engine.
#
#   make / make all     build shared library + perft CLI + unit-test binary
#   make check          run the C unit tests + full perft suite (the Phase 1 gate)
#   make check-asan     same, under AddressSanitizer + UBSan (slower, catches
#                       memory errors and UB that "correct" runs hide)
#   make clean          remove build artifacts
#
# The shared library is not used by any binary here yet — it exists because
# Phase 0.3/3 (cffi bindings) consume it; building it from day one keeps the
# library target honest (no -Werror surprises later).

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

LIB_SRCS = src/tables.c src/board.c src/fen.c src/movegen.c src/perftlib.c
LIB_HDRS = include/chess.h
TEST_SRCS = tests/test_units.c tests/suite_loader.c

BUILD      = build
BUILD_ASAN = build-asan

.PHONY: all check check-asan clean

all: $(BUILD)/$(SO) $(BUILD)/perft $(BUILD)/test_units

# --- default build (optimized) ---

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/$(SO): $(LIB_SRCS) $(LIB_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) -fPIC -shared $(LIB_SRCS) -o $@

$(BUILD)/perft: src/perft.c $(LIB_SRCS) $(LIB_HDRS) tests/suite_loader.c tests/suite_loader.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) src/perft.c $(LIB_SRCS) tests/suite_loader.c -o $@

$(BUILD)/test_units: $(TEST_SRCS) $(LIB_SRCS) $(LIB_HDRS) tests/suite_loader.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) $(TEST_SRCS) $(LIB_SRCS) -o $@

# --- sanitizer build ---

$(BUILD_ASAN):
	mkdir -p $(BUILD_ASAN)

$(BUILD_ASAN)/perft: src/perft.c $(LIB_SRCS) $(LIB_HDRS) tests/suite_loader.c tests/suite_loader.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) $(SUITE_PATH_DEFINE) -fsanitize=address,undefined -fno-omit-frame-pointer src/perft.c $(LIB_SRCS) tests/suite_loader.c -o $@

$(BUILD_ASAN)/test_units: $(TEST_SRCS) $(LIB_SRCS) $(LIB_HDRS) tests/suite_loader.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) $(SUITE_PATH_DEFINE) -fsanitize=address,undefined -fno-omit-frame-pointer $(TEST_SRCS) $(LIB_SRCS) -o $@

# --- gates ---

check: $(BUILD)/test_units $(BUILD)/perft
	$(BUILD)/test_units
	$(BUILD)/perft suite

check-asan: $(BUILD_ASAN)/test_units $(BUILD_ASAN)/perft
	$(BUILD_ASAN)/test_units
	$(BUILD_ASAN)/perft suite
# (LeakSanitizer is unavailable on macOS/arm64; it runs on the Linux cloud
#  containers — add ASAN_OPTIONS=detect_leaks=1 there.)

clean:
	rm -rf $(BUILD) $(BUILD_ASAN)
