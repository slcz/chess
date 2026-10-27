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
// ort.all (not ort.min): the wasm-only bundle has no WebGPU backend, so
// the ["webgpu","wasm"] EP attempt below could never succeed — the GUI has
// been silently CPU-only. ort.all carries the jsep/WebGPU backend; wasm
// remains the fallback where GPU is unavailable.
import * as ort from "../vendor/ort.all.min.mjs";

{ // vendor dir relative to this module: a URL for the browser, a plain
  // path for Node (ort's fs-based loader rejects file:// URLs)
  const vendor = new URL("../vendor/", import.meta.url);
  ort.env.wasm.wasmPaths =
    typeof document === "undefined" ? vendor.pathname : vendor.href;
}
// Threads need SharedArrayBuffer (cross-origin isolation) — this GUI has
// none by design. Asking for threads anyway makes ORT warn and "fall
// back", and that half-configured path hangs mid-session in a worker;
// only enable threads when isolation is actually present.
ort.env.wasm.numThreads =
  (typeof crossOriginIsolated !== "undefined" && crossOriginIsolated)
    ? ((typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4)
    : 1;

const NN_TOKENS = 64 * 117;
const POLICY = 4240;
// Rows per session.run — the engine stages whole walker batches (up to 256
// rows) at fixed heap offsets; chunking keeps each await's tensor sizes
// small and the copy-in/copy-out window short. (The intermittent wasm-EP
// session.run wedge is an ORT defect, NOT fixed by chunking — see
// web/test/eval_hang.html; WebGPU sidesteps it.)
const CHUNK_ROWS = 64;

/** WebGPU is usable only if an adapter can actually be acquired: some
 * environments expose navigator.gpu but return null (headless Chrome). */
async function hasWebgpu() {
  if (typeof navigator === "undefined" || !navigator.gpu) return false;
  try {
    const adapter = await Promise.race([
      navigator.gpu.requestAdapter(),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("WebGPU adapter timeout")), 1500)),
    ]);
    return !!adapter;
  } catch {
    return false;
  }
}

export async function loadSession(source) {
  // source: ArrayBuffer (file picker / buffer) or URL string (fetched)
  const buf = typeof source === "string"
    ? await (await fetch(source)).arrayBuffer() : source;
  const gpu = await hasWebgpu();
  let session, ep;
  try {
    session = await ort.InferenceSession.create(buf,
      { executionProviders: gpu ? ["webgpu", "wasm"] : ["wasm"] });
    // ORT silently drops an unavailable EP; its proxy exposes the actual
    // set. Trust that over the preflight answer.
    const actual = session.executionProviders;
    ep = Array.isArray(actual) && actual.length
      ? (actual.includes("webgpu") ? "webgpu/wasm" : "wasm")
      : (gpu ? "webgpu/wasm" : "wasm");
  } catch (e) {
    console.warn("webgpu session create failed, retrying wasm-only:", e);
    session = await ort.InferenceSession.create(
      buf, { executionProviders: ["wasm"] });
    ep = "wasm";
  }

  const run = async (req, memory) => {
    try {
      for (let r0 = 0; r0 < req.n; r0 += CHUNK_ROWS) {
        const rows = Math.min(CHUNK_ROWS, req.n - r0);
        const feeds = {
          tokens: new ort.Tensor("float32",
            new Float32Array(memory, req.planes + r0 * NN_TOKENS * 4,
                             rows * NN_TOKENS),
            [rows, 64, 117]),
          mask: new ort.Tensor("bool",
            new Uint8Array(memory, req.mask + r0 * POLICY, rows * POLICY),
            [rows, POLICY]),
        };
        const out = await session.run(feeds);
        new Float32Array(memory, req.logp + r0 * POLICY * 4, rows * POLICY)
        new Float32Array(memory, req.logp + r0 * POLICY * 4, rows * POLICY)
          .set(out.policy_logp.data);
        new Float32Array(memory, req.wdl + r0 * 12, rows * 3)
          .set(out.wdl.data);
      }
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
