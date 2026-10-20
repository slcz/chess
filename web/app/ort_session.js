/*
 * ort_session.js — an ONNX model loaded into onnxruntime-web, exposed as
 * the one function the engine's evaluator calls (the __ortRun host of
 * engine/src/wasm_eval.c):
 *
 *   run({n, planes, mask, logp, wdl}, memory)  →  Promise
 *
 * receiving BYTE OFFSETS into the engine's wasm memory — the same
 * offsets-over-shared-heap trick the native seam plays with pinned host
 * pointers. No SharedArrayBuffer, no channel, no handshake: the caller
 * is on the same thread and awaits this directly (Asyncify unwinds the
 * C search into the event loop to do so).
 *
 * The graph's contract is the export from chess_ai/net/export_onnx.py:
 *   in : tokens [n,64,117] f32, mask [n,4240] bool
 *   out: policy_logp [n,4240] f32 (illegal slots exp→ exactly 0),
 *        wdl [n,3] f32 softmax probs (the search backs up w[0] − w[2])
 *
 * Failure discipline: the C search WAITS for this promise, so run() must
 * always leave valid numbers in logp/wdl — a failed forward degrades to
 * the dummy outputs (uniform + a draw) rather than freezing the engine.
 */
import * as ort from "../vendor/ort.min.mjs";

{ // vendor dir relative to this module: a URL for the browser, a plain
  // path for Node (ort's fs-based loader rejects file:// URLs)
  const vendor = new URL("../vendor/", import.meta.url);
  ort.env.wasm.wasmPaths =
    typeof document === "undefined" ? vendor.pathname : vendor.href;
}
ort.env.wasm.numThreads = (typeof navigator !== "undefined"
  && navigator.hardwareConcurrency) || 4;

const NN_TOKENS = 64 * 117;
const POLICY = 4240;

export async function loadSession(source) {
  // source: ArrayBuffer (file picker / buffer) or URL string (fetched)
  const buf = typeof source === "string"
    ? await (await fetch(source)).arrayBuffer() : source;
  let session, ep;
  try {
    session = await ort.InferenceSession.create(buf,
      { executionProviders: ["webgpu", "wasm"] });
    ep = (typeof navigator !== "undefined" && navigator.gpu)
      ? "webgpu/wasm" : "wasm";
  } catch (e) {
    console.warn("webgpu session create failed, retrying wasm-only:", e);
    session = await ort.InferenceSession.create(
      buf, { executionProviders: ["wasm"] });
    ep = "wasm";
  }

  const run = async (req, memory) => {
    try {
      const feeds = {
        tokens: new ort.Tensor("float32",
          new Float32Array(memory, req.planes, req.n * NN_TOKENS),
          [req.n, 64, 117]),
        mask: new ort.Tensor("bool",
          new Uint8Array(memory, req.mask, req.n * POLICY),
          [req.n, POLICY]),
      };
      const out = await session.run(feeds);
      new Float32Array(memory, req.logp, req.n * POLICY)
        .set(out.policy_logp.data);
      new Float32Array(memory, req.wdl, req.n * 3).set(out.wdl.data);
    } catch (err) {           // never hang the search — see header note
      console.warn("eval run failed, serving dummy outputs:", err);
      fillDummy(memory, req.logp, req.wdl, req.n);
    }
  };
  return { run, ep };
}

function fillDummy(memory, logp, wdl, n) {
  new Float32Array(memory, logp, n * POLICY).fill(0);
  const w = new Float32Array(memory, wdl, n * 3);
  for (let r = 0; r < n; r++) {
    w[3 * r] = 0;
    w[3 * r + 1] = 1;
    w[3 * r + 2] = 0;
  }
}
