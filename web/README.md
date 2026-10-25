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
node web/test/shard3_node.mjs       # SPSHARD1 v3 parse + π ≡ engine movegen
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
| `app/shard.js` | SPSHARD1 reader (v1+v2), port of chess_ai/selfplay/shard.py |
| `app/slots.js` | policy slot ↔ UCI, port of the move-space half of features.py |
| `app/board.js` | CSS board, Unicode pieces, SVG π/visit arrows, eval bar |
| `app/hf.js` | Hugging Face Hub access: tree listing, resolve fetch + progress |
| `build.sh` | env wrapper around `make -C engine wasm` |
| `serve.py` | any static server works; this one serves the repo root |
| `get_vendor.sh` | fetch the vendored ORT runtime into `vendor/` (gitignored) |
| `test/` | the gates + screenshots |

## Models and shards

The page has no bundled artifacts and no local-file inputs: everything
loads straight from HuggingFace through the panel — `.onnx` nets from the
models repo (`chriszhang01/chess-value-net`), `.spbin` shards from the
datasets repo (`chriszhang01/chess-selfplay`, path prefix narrows the
listing). Checkpoints are converted offline and pushed to the models repo:

    .venv/bin/python scripts/export_onnx.py final.pt --out final.onnx

(SPHARD1 v3 note: its plies store π as a dense f16 vector in ENGINE
MOVEGEN order instead of (slot, prob) pairs — `shard.js` parses the
bytes, and the worker zips probs with its own replayed legal-move list
per shown ply, same as `web/test/shard3_node.mjs` proves.) Without a
model the engine plays with the dummy evaluator (uniform + draw):
legal, and useful for UI work.

## Notes

- The net prefers **WebGPU** (ort.all bundle, jsep backend; ~12× the wasm
  CPU path — a 2000-sim think runs in ~3 s on Metal vs minutes on CPU),
  falling back to the single-threaded wasm EP (ORT-web's threads need the
  isolation we dropped). Known ORT-web defect: the single-threaded wasm
  EP intermittently spins forever inside `session.run` (100% CPU, thread
  unrecoverable; seen on 1.19.2 and 1.30.0, not data- or model-dependent,
  never on node). Minimal repro: `web/test/eval_hang.html` — on
  WebGPU-capable machines the GPU path sidesteps it entirely.
- `engine_worker.js` is a classic worker (the emscripten glue uses
  `importScripts`); it loads `ort_session.js` via dynamic `import()`.
- The walkthrough computes SAN through the engine (a port of
  chess_ai/san.py lives in the worker) — the page itself never decides
  chess truth.
