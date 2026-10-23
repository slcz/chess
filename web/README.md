# web/ — the serverless GUI (wasm-serverless branch)

One static page, no server: the C play path compiled to WASM, the net
through **onnxruntime-web** (WebGPU EP, WASM fallback), and self-play
shards opened client-side for walkthroughs. Works on any static host in
any browser — plain WASM 1.0, no SharedArrayBuffer, no cross-origin
isolation, no build tooling on the page side.

## Design: single-threaded, stepped eval

The engine is compiled WITHOUT threads. A think runs as an explicit
loop across the JS/C boundary (all in [engine/src/puct.c](../engine/src/puct.c),
under `__EMSCRIPTEN__`):

```
puct_think_start(g, sims)   → 1 = done, 0 = a bank awaits evaluation
puct_zone_pending()         → which bank (or -1)
puct_zone_bank(b, out)      → {n, planes, mask, logp, wdl} byte offsets
   ... JS runs the net on those rows (ort_session.js) ...
puct_zone_publish()         → results ready; the next poll delivers them
```

Walkers park between the steps; everything they need is heap-resident
and nothing else runs, so this is simply safe. The bank state machine,
tree reuse, and delivery are the same code the native build uses — only
the transport differs (native keeps its io thread; the wasm build's
`io_main` is compiled out). Two gates pin equivalence:

```sh
web/build.sh                        # emcc → engine/build/wasm/ (emsdk, python3.12)
node web/test/parity_node.mjs       # wasm search ≡ native, bit-for-bit
node web/test/bridge_node.mjs       # stepping protocol ≡ native, bit-for-bit
node web/test/net_node.mjs          # real net through the real loop
node web/test/page_check.mjs        # headless-Chromium acceptance run
```

The first two diff 399-line search transcripts (doubles as IEEE754 hex)
against `engine/build/parity_dump` (`make -C engine build/parity_dump`):
the dummy evaluator is deterministic, so the io timing can never change
which rows land where. The stepping protocol replaced an earlier
pthreads + BroadcastChannel + Atomics design — that one worked in Node
but hit two browser walls (Chromium silently drops channel messages
containing a SAB; COOP/COEF hosting friction), and emscripten 6's
Asyncify — the third option tried — is broken in the 6.0.x line (its
minimal case fails; deprecated in favor of JSPI). Stepping is the
honest mechanism: the async boundary sits BETWEEN C calls, where there
is no C stack to preserve.

## Layout

| Path | What |
|---|---|
| `index.html` + `app/main.js` | the page: play mode + walkthrough mode |
| `app/engine_worker.js` | owns the wasm engine on a worker (SAN, replay, thinks) |
| `app/ort_session.js` | ONNX model → async run over row offsets |
| `app/shard.js` | SPSHARD1 reader (v3 only), port of chess_ai/selfplay/shard.py |
| `app/slots.js` | policy slot ↔ UCI, port of the move-space half of features.py (unreferenced since v3 π went movegen-order; kept for the net-play spare) |
| `app/board.js` | CSS board, Unicode pieces, SVG π/visit arrows, eval bar |
| `build.sh` | env wrapper around `make -C engine wasm` |
| `serve.py` | any static server works; this one serves the repo root |
| `test/` | the four gates + screenshots |

## Models and shards

The page has no bundled artifacts: load a generation's `.onnx` (exported
by `scripts/export_onnx.py`, ~26 MB fp32; gen14's parity gate: max
|Δlogp| 1.5e-5 vs torch) and a `shard-0.spbin` (v3 — convert legacy
shards with `scripts/spbin_to_v3.py`) through the panel's file
inputs — generation checkpoints and self-play units live on HuggingFace
(`chriszhang01/chess-value-net`, `chriszhang01/chess-selfplay`). Without
a model the engine plays with the dummy evaluator (uniform + draw):
legal, and useful for UI work.

## Notes

- The net's CPU path is single-threaded here (ORT-web's threads need
  the isolation we dropped); WebGPU doesn't care. gen14 at 300 sims
  runs in ~2-4 s via WASM-fallback CPU in a headless window.
- `engine_worker.js` is a classic worker (the emscripten glue uses
  `importScripts`); it loads `ort_session.js` via dynamic `import()`.
- The walkthrough computes SAN through the engine (a port of
  chess_ai/san.py lives in the worker) — the page itself never decides
  chess truth.
