#!/bin/sh
# web/get_vendor.sh — fetch the vendored onnxruntime-web runtime into
# web/vendor/ (gitignored: the page has no committed artifacts).
#
#   sh web/get_vendor.sh
#
set -eu
cd "$(dirname "$0")"
# Browser runtime 1.30.0 (requirements.txt pins 1.19.2 for the Python
# parity gate). 1.19.2 wedged mid-session on the single-threaded wasm
# path in a classic worker (session.run's promise never settled, worker
# pinned at 100% CPU — observed on a 2000-sim think's 6th+ eval batch).
# 1.30.0 fixed the hang; same dist layout, ort_session.js unchanged.
VER=1.30.0
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -sL "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-${VER}.tgz" \
  -o "$TMP/ort.tgz"
mkdir -p vendor "$TMP/x"
tar xzf "$TMP/ort.tgz" -C "$TMP/x"
# ort.min.mjs (the ESM bundle ort_session.js imports) plus the wasm
# backends it lazily loads: jsep for the WebGPU EP, plain for the
# single-threaded wasm fallback (no SAB in this GUI).
for f in ort.min.mjs ort.all.min.mjs \
         ort-wasm-simd-threaded.jsep.mjs ort-wasm-simd-threaded.jsep.wasm \
         ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm; do
  cp "$TMP/x/package/dist/$f" vendor/
done
ls -l vendor/
