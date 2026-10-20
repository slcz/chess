#!/bin/sh
# web/build.sh — build the wasm engine into engine/build/wasm/.
#
# Sources the local emsdk (installed at ~/dev/emsdk; emcc needs Python
# >= 3.10, hence the brew python3.12 on PATH — the repo venv is 3.9).
# The Makefile target is `make -C engine wasm`; this wrapper only sets
# up the environment.
set -e
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "$HOME/dev/emsdk/emsdk_env.sh" >/dev/null 2>&1 || true
export PATH="/opt/homebrew/opt/python@3.12/bin:/opt/homebrew/bin:$PATH"
command -v emcc >/dev/null || { echo "emcc not found (emsdk at ~/dev/emsdk?)"; exit 1; }
make -C "$ROOT/engine" wasm
echo "wasm engine: engine/build/wasm/chess_engine.{js,wasm}"
