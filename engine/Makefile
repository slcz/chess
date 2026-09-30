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
# -ffp-contract=off: the L2 bit-parity contract requires every float
# expression to round exactly as written (Python's separate mults/adds).
# clang's arm64 -O2 default contracts a*b+c into fma — one rounding
# instead of two — which bit-differs from the reference on near-ties
# (surfaced by the torch-net L2 test with seeded weights, 2026-09-29).
# -D_DEFAULT_SOURCE: -std=c11 sets __STRICT_ANSI__, under which glibc hides
# the POSIX/BSD APIs the engine uses (clock_gettime, MAP_ANON, open's
# O_CREAT…) — the Linux cloud build fails without this; inert on Darwin,
# whose headers expose them regardless (first x86_64 build, 2026-09-30).
CFLAGS   = $(WARN) $(OPT) -ffp-contract=off -D_DEFAULT_SOURCE
INCLUDES = -Iinclude -Itests

# Bake the absolute path of the perft suite data file into the binaries so
# tests find it regardless of the working directory.
SUITE_PATH_DEFINE = -DPERFT_SUITE_PATH="\"$(abspath tests/perft_suite.txt)\""

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SO = libchess.dylib
  SO_EXT = dylib
else
  SO = libchess.so
  SO_EXT = so
endif

RULES_SRCS = src/tables.c src/board.c src/fen.c src/movegen.c src/termination.c src/perftlib.c
NN_SRCS   = src/nn_encode.c
SEARCH_SRCS = src/rng.c src/search.c
LIB_SRCS  = $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) src/ffi.c
LIB_HDRS  = include/chess.h include/chess_ffi.h include/chess_nn.h \
            include/chess_rng.h include/chess_search.h
TEST_SRCS = tests/test_units.c tests/suite_loader.c
NN_TEST_SRCS = tests/test_nn.c
SEARCH_TEST_SRCS = tests/test_search.c
SP_SRCS    = src/sp_engine.c src/sp_dump.c
SP_EVAL_SRC  = src/sp_evaluator.c
SP_LIB_SRCS  = $(LIB_SRCS) $(SP_SRCS) $(SP_EVAL_SRC)
SP_TEST_SRCS = tests/test_sp.c

# Python headers for the embedded-CPython evaluator (F8). The library is
# built for loading INTO a Python process: the interpreter's symbols are
# resolved at load time (macOS: -undefined dynamic_lookup; Linux .so
# allows undefined symbols by default), so nothing links against libpython.
# The chess_selfplay BINARY is the exception — as an executable it must
# link the interpreter explicitly (Py_Initialize needs a real libpython).
# Prefer the project venv (its Python is the host the library loads into);
# fall back to whatever python3 is on PATH.
PYTHON     ?= $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; \
                     elif [ -x ../../.venv/bin/python ]; then echo ../../.venv/bin/python; \
                     else echo python3; fi)
PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
# Locate the loadable libpython for the chess_selfplay executable link.
# sysconfig's LIBDIR alone is unreliable for framework builds (the
# CommandLineTools python reports a nonexistent Xcode.app path), so also
# try the framework layout derived from the include dir. macOS framework
# libpythons carry an @rpath install name, so the flag carries an -rpath
# at the framework's parent dir; empty → link by name (-lpythonX.Y) and
# let the system search paths speak.
PY_LDFLAGS := $(shell $(PYTHON) -c "import os,sysconfig; c=sysconfig.get_config_var; v=c('LDVERSION') or c('VERSION'); inc=sysconfig.get_paths()['include']; fw=os.path.dirname(os.path.dirname(inc)); dirs=[c('LIBDIR'),os.path.join(fw,'lib'),fw]; lib=next((f for d in dirs if d for f in [os.path.join(d,'libpython'+v+ext) for ext in ('.dylib','.so')] if os.path.exists(f)),None); rpath=lib; lib and exec('for _ in range(5): rpath=os.path.dirname(rpath)'); print((lib+' -Wl,-rpath,'+rpath) if lib else ('-lpython'+v))")

BUILD      = build
BUILD_ASAN = build-asan

.PHONY: all check check-asan clean

all: $(BUILD)/$(SO) $(BUILD)/libchess_sp.$(SO_EXT) $(BUILD)/perft $(BUILD)/test_units $(BUILD)/test_search $(BUILD)/test_sp $(BUILD)/chess_selfplay

# --- default build (optimized) ---

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/$(SO): $(LIB_SRCS) $(LIB_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) -fPIC -shared $(LIB_SRCS) -o $@

# The sp .so resolves the embedded evaluator's Python symbols at dlopen
# time. -undefined dynamic_lookup is Darwin-only ld syntax; Linux shared
# objects permit undefined symbols by default, so the flag is simply off.
ifeq ($(shell uname),Darwin)
SP_UNDEF := -undefined dynamic_lookup
endif

$(BUILD)/libchess_sp.$(SO_EXT): $(SP_LIB_SRCS) $(LIB_HDRS) include/chess_sp.h include/chess_sp_eval.h include/chess_sp_dump.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -I"$(PY_INCLUDE)" $(SUITE_PATH_DEFINE) -fPIC -shared $(SP_LIB_SRCS) -o $@ $(SP_UNDEF)

$(BUILD)/perft: src/perft.c $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.c tests/suite_loader.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) src/perft.c $(RULES_SRCS) tests/suite_loader.c -o $@

$(BUILD)/test_units: $(TEST_SRCS) $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SUITE_PATH_DEFINE) $(TEST_SRCS) $(RULES_SRCS) -o $@

# The NN encoder test links the rules + encoder (no FFI handle needed).
$(BUILD)/test_nn: $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) include/chess.h include/chess_nn.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) -o $@

# The search test links the rules + encoder + RNG + search (Phase 3.1).
# -lm: search.c's softmax uses exp; libm is implicit on Darwin, separate
# on Linux (first x86_64 build, 2026-09-30).
$(BUILD)/test_search: $(SEARCH_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) include/chess.h include/chess_nn.h include/chess_rng.h include/chess_search.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SEARCH_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) -o $@ -lm

# The parallel-engine test links everything except the FFI handle and the
# embedded evaluator (the evaluator is injected as C function pointers).
$(BUILD)/test_sp: $(SP_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) $(SP_SRCS) include/chess.h include/chess_nn.h include/chess_rng.h include/chess_search.h include/chess_sp.h include/chess_sp_eval.h include/chess_sp_dump.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) $(SP_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) $(SP_SRCS) -o $@ -lm

# The standalone self-play work-unit runner (Phase 3.4): Py_Initialize +
# the engine, one unit per process, DONE.json completion marker. Unlike
# libchess_sp (dlopen-only, interpreter symbols resolved by the host), an
# EXECUTABLE must link libpython explicitly. Its run gate is
# tests/test_selfplay_cli.py (needs a live interpreter); `make check`
# stays Python-free.
# Libpython's own dependencies must be on the executable's link line:
# since binutils ~2.22 a shared lib's DSOs are NOT pulled in transitively
# ("DSO missing from command line", first x86_64 build 2026-09-30). The
# stub libs are no-ops on macOS.
PY_SYSLIBS := -lm -ldl -lpthread

$(BUILD)/chess_selfplay: src/chess_selfplay_main.c $(SP_LIB_SRCS) $(LIB_HDRS) include/chess_sp.h include/chess_sp_eval.h include/chess_sp_dump.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDES) -I"$(PY_INCLUDE)" $(SUITE_PATH_DEFINE) src/chess_selfplay_main.c $(SP_LIB_SRCS) -o $@ $(PY_LDFLAGS) $(PY_SYSLIBS)

# --- sanitizer build ---

$(BUILD_ASAN):
	mkdir -p $(BUILD_ASAN)

$(BUILD_ASAN)/perft: src/perft.c $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.c tests/suite_loader.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) $(SUITE_PATH_DEFINE) -fsanitize=address,undefined -fno-omit-frame-pointer src/perft.c $(RULES_SRCS) tests/suite_loader.c -o $@

$(BUILD_ASAN)/test_units: $(TEST_SRCS) $(RULES_SRCS) $(LIB_HDRS) tests/suite_loader.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) $(SUITE_PATH_DEFINE) -fsanitize=address,undefined -fno-omit-frame-pointer $(TEST_SRCS) $(RULES_SRCS) -o $@

$(BUILD_ASAN)/test_nn: $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) include/chess.h include/chess_nn.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) -fsanitize=address,undefined -fno-omit-frame-pointer $(NN_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) -o $@

$(BUILD_ASAN)/test_search: $(SEARCH_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) include/chess.h include/chess_nn.h include/chess_rng.h include/chess_search.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) -fsanitize=address,undefined -fno-omit-frame-pointer $(SEARCH_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) -o $@

$(BUILD_ASAN)/test_sp: $(SP_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) $(SP_SRCS) include/chess.h include/chess_nn.h include/chess_rng.h include/chess_search.h include/chess_sp.h include/chess_sp_eval.h include/chess_sp_dump.h | $(BUILD_ASAN)
	$(CC) $(WARN) -O1 -g $(INCLUDES) -fsanitize=address,undefined -fno-omit-frame-pointer $(SP_TEST_SRCS) $(RULES_SRCS) $(NN_SRCS) $(SEARCH_SRCS) $(SP_SRCS) -o $@

# --- gates ---

check: $(BUILD)/test_units $(BUILD)/test_nn $(BUILD)/test_search $(BUILD)/test_sp $(BUILD)/perft
	$(BUILD)/test_units
	$(BUILD)/test_nn
	$(BUILD)/test_search
	$(BUILD)/test_sp
	$(BUILD)/perft suite

check-asan: $(BUILD_ASAN)/test_units $(BUILD_ASAN)/test_nn $(BUILD_ASAN)/test_search $(BUILD_ASAN)/test_sp $(BUILD_ASAN)/perft
	$(BUILD_ASAN)/test_units
	$(BUILD_ASAN)/test_nn
	$(BUILD_ASAN)/test_search
	$(BUILD_ASAN)/test_sp
	$(BUILD_ASAN)/perft suite
# (LeakSanitizer is unavailable on macOS/arm64; it runs on the Linux cloud
#  containers — add ASAN_OPTIONS=detect_leaks=1 there.)

clean:
	rm -rf $(BUILD) $(BUILD_ASAN)
