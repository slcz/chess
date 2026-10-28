/*
 * main.js — page wiring: you vs the net (or vs the dummy evaluator before
 * any model is loaded). Sims slider; eval bar after every think.
 *
 * Artifacts load straight from Hugging Face (no local files, no server):
 * the .onnx through the worker (hf.js there too — the page must not carry
 * 30 MB through postMessage). All chess truth (legality, termination, SAN,
 * replay) comes from the worker; this file only renders and routes
 * messages.
 */
import { Board, renderEvalBar, pieceGlyph } from "./board.js";
import { listFiles, parseRepo, resolveUrl } from "./hf.js";

const $ = (id) => document.getElementById(id);
const worker = new Worker(
  new URL("engine_worker.js?v=" + Date.now(), import.meta.url));
worker.onerror = (e) => {
  // a classic-worker crash is otherwise silent — surface it in the UI
  const el = $("status");
  if (el) el.textContent =
    `engine worker failed: ${e.message} (${e.filename}:${e.lineno})`;
};
let reqId = 0;
const pending = new Map();
worker.onmessage = (e) => {
  const { id, ...r } = e.data;
  if (id === undefined) {          // unsolicited: net download progress
    if (r.t === "netProgress") {
      const pct = r.total ? `${(r.loaded / r.total * 100).toFixed(0)}%`
        : `${(r.loaded / 1e6).toFixed(1)} MB`;
      $("netName").textContent = `loading ${netLabel}… ${pct}`;
    }
    return;
  }
  const p = pending.get(id);
  if (p) {
    pending.delete(id);
    r.t === "error" ? p.reject(new Error(r.error)) : p.resolve(r);
  }
};
const call = (m) => new Promise((resolve, reject) => {
  pending.set(++reqId, { resolve, reject });
  worker.postMessage({ id: reqId, ...m });
});

const board = new Board($("board"));
let engineWalkers = 0;             // set by the worker's init reply
let cpuMode = false;               // no WebGPU: 8 walkers, ≤200 sims
// dev probe: live page state for the headless acceptance run
window.__pgn = buildPgn;
window.__dbg = () => ({ thinking, humanSide, sims, temperature,
  walkers: engineWalkers,
  cpu: cpuMode,
  S: S && { fen: S.fen, side: S.side, term: S.termination },
  histLen: playHist.length, pend: pending.size });
let sel = null;                    // selected square
let S = null;                      // current engine state
let thinking = false;

/* play-mode state */
let humanSide = "w";
let sims = 1000;
let temperature = 0;               // 0 = best move; 1 = sample visits

/* CPU fallback: the single-threaded wasm EP gets a smaller batch and a
 * tighter budget than WebGPU. Applied before the first think when no
 * WebGPU adapter is available, and again if ORT reports a wasm session. */
function setSimBudget(cpu) {
  const el = $("sims");
  const prev = +el.value;
  if (cpu) {
    el.max = "200";
    // setting max clamps the value to 200 first, so restore the intended
    // CPU default explicitly unless the user already chose a lower budget
    el.value = String(prev > 200 ? 100 : prev);
  } else {
    el.max = "5000";
  }
  sims = +el.value;
  $("simsVal").textContent = sims;
}

async function useCpuFallback() {
  if (cpuMode) return;
  cpuMode = true;
  setSimBudget(true);
  try {
    const r = await call({ t: "setWalkers", walkers: 8 });
    engineWalkers = r.walkers;
  } catch { /* keep playing; the next new game picks up 8 */ }
}
window.__cpuFallback = useCpuFallback;   // dev probe for the fallback path

/** WebGPU is usable only if an adapter can actually be acquired. */
async function gpuAdapterAvailable() {
  if (!navigator.gpu) return false;
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

/* HF panel state (netLabel names the file being downloaded) */
let netLabel = "";

/* ---- rendering -------------------------------------------------------- */
function render() {
  if (!S) return;
  board.render({ ...S, sel });
  $("fen").textContent = S.fen;
  renderPlay();
}

function renderPlay() {
  const over = S.termination !== "PLAYING";
  $("status").textContent = over ? "Game over — " + termText()
    : thinking ? "engine is thinking…"
    : (S.side === humanSide ? "your move (click a piece)" : "engine to move");
  const moves = $("moves");
  moves.innerHTML = "";
  playHist.forEach((h, i) => {
    if (i % 2 === 0) {
      const num = document.createElement("span");
      num.className = "num";
      num.textContent = (i / 2 + 1) + ".";
      moves.appendChild(num);
    }
    const li = document.createElement("li");
    li.textContent = h.san;
    moves.appendChild(li);
    if (i % 2 === 0) moves.appendChild(document.createTextNode(" "));
  });
  $("bUndo").disabled = !playHist.length;
  $("bPgn").disabled = !playHist.length;
  $("bEngineMove").disabled = over || thinking || S.side === humanSide;
}

function termText() {
  const names = { CHECKMATE: "checkmate", STALEMATE: "stalemate",
    INSUFFICIENT_MATERIAL: "insufficient material",
    FIFTY_MOVES: "50-move rule", SEVENTYFIVE_MOVES: "75-move rule",
    THREEFOLD_REPETITION: "threefold repetition",
    FIVEFOLD_REPETITION: "fivefold repetition" };
  if (S.termination === "CHECKMATE")
    return (S.side === "w" ? "0–1 black" : "1–0 white") + " wins by checkmate";
  return "draw — " + (names[S.termination] || S.termination);
}

/* ---- play -------------------------------------------------------------- */
let playHist = [];

async function newGame() {
  playHist = [];
  sel = null;
  const r = await call({ t: "newGame" });
  S = r.state;
  renderEvalBar($("evalbar"), null);
  board.arrows([]);
  render();
}

board.clickCb = async (name) => {
  if (thinking || !S || S.termination !== "PLAYING" || S.side !== humanSide)
    return;
  if (sel === null) {
    if (S.legal.some((m) => m.slice(0, 2) === name) && ownPieceAt(name))
      sel = name;
  } else if (name === sel) {
    sel = null;
  } else {
    const matches = S.legal.filter(
      (m) => m.slice(0, 2) === sel && m.slice(2, 4) === name);
    if (!matches.length) sel = ownPieceAt(name) ? name : null;
    else if (matches.length === 1 && matches[0].length === 4)
      return doMove(matches[0]);
    else if (matches.some((m) => m.length === 4))
      return doMove(matches.find((m) => m.length === 4));
    else choosePromotion(matches);
  }
  render();
};

function ownPieceAt(name) {
  const flat = S.fen.split(" ")[0].replace(/[0-8]/g, (m) => ".".repeat(+m))
    .replace(/\//g, "");
  const idx = (8 - +name[1]) * 8 + "abcdefgh".indexOf(name[0]);
  const ch = flat[idx];
  return ch !== "." && (ch === ch.toUpperCase()) === (S.side === "w");
}

function choosePromotion(matches) {
  const p = $("promo");
  p.style.display = "block";
  const white = S.side === "w";
  p.querySelectorAll("button").forEach((b) => {
    b.textContent = pieceGlyph(b.dataset.p);
    b.classList.toggle("wp", white);
    b.classList.toggle("bp", !white);
    b.onclick = () => {
      p.style.display = "none";
      doMove(matches[0].slice(0, 4) + b.dataset.p);
    };
  });
}

async function doMove(uci) {
  sel = null;
  const r = await call({ t: "play", uci });
  S = r.state;
  playHist.push(r.move);
  render();
  if (S.termination === "PLAYING" && S.side !== humanSide) engineTurn();
}

async function engineTurn() {
  thinking = true;
  renderPlay();
  try {
    const r = await call({ t: "engineMove", sims, temperature });
    const mover = S.side;        // side that searched, before the move
    S = r.state;
    showThink(r.result, mover);
    playHist.push(r.move);
  } finally {
    thinking = false;
  }
  render();
}

/** Eval bar after a think: rootV is the parent's view of the pre-move
 * mover, so flip it only when that mover was White. */
function showThink(r, mover) {
  const whiteV = mover === "w" ? -r.rootV : r.rootV;
  renderEvalBar($("evalbar"), whiteV);
}

/** PGN of the current game: headers, numbered movetext, result. */
function buildPgn() {
  const result = S.termination === "CHECKMATE"
    ? (S.side === "w" ? "0-1" : "1-0")
    : S.termination === "PLAYING" ? "*" : "1/2-1/2";
  const date = new Date().toISOString().slice(0, 10).replace(/-/g, ".");
  const headers = [
    `[Event "chess wasm"]`, `[Site "local"]`, `[Date "${date}"]`,
    `[White "${humanSide === "w" ? "human" : "engine"}"]`,
    `[Black "${humanSide === "w" ? "engine" : "human"}"]`,
    `[Result "${result}"]`,
  ];
  const tokens = [];
  playHist.forEach((h, i) => {
    if (i % 2 === 0) tokens.push(`${i / 2 + 1}.`);
    tokens.push(h.san);
  });
  tokens.push(result);
  const lines = [];
  let cur = "";
  for (const tok of tokens) {
    if (cur.length + tok.length + 1 > 80) { lines.push(cur.trim()); cur = ""; }
    cur += tok + " ";
  }
  if (cur.trim()) lines.push(cur.trim());
  return headers.join("\n") + "\n\n" + lines.join("\n") + "\n";
}

$("bPgn").onclick = () => {
  if (!playHist.length) return;
  const blob = new Blob([buildPgn()], { type: "application/x-chess-pgn" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `chess-${new Date().toISOString().slice(0, 10)}.pgn`;
  a.click();
  URL.revokeObjectURL(a.href);
};

/* ---- panel wiring: the net loads from Hugging Face --------------------- */

/* the repo field accepts a bare "owner/name", "owner/name/with/path",
 * or a pasted huggingface.co URL — parseRepo normalizes all of them. */

let netListTimer = null;
let netListSeq = 0;          // drops stale listings after a repo edit
let loadedNetPath = "";      // the .onnx currently in the worker

async function listNets() {
  const seq = ++netListSeq;
  const { repo, path } = parseRepo($("netRepo").value);
  if (!repo) return;
  $("netName").textContent = `listing ${repo}…`;
  try {
    const files = (await listFiles("model", repo, path))
      .filter((f) => f.type === "file" && f.path.endsWith(".onnx"));
    if (seq !== netListSeq) return;   // superseded by a newer repo edit
    const sel$ = $("netSel");
    sel$.innerHTML = "";
    for (const f of files) {
      const o = document.createElement("option");
      o.value = f.path;
      o.textContent = `${f.path} (${(f.size / 1e6).toFixed(1)} MB)`;
      sel$.appendChild(o);
    }
    if (!files.length) {
      $("netName").textContent =
        "no .onnx in that repo — convert a checkpoint (scripts/export_onnx.py)";
      return;
    }
    // keep the loaded net selected when it is still listed; otherwise
    // auto-load the newest listed one — selecting is the whole interaction
    if (files.some((f) => f.path === loadedNetPath)) {
      sel$.value = loadedNetPath;
      $("netName").textContent = `${loadedNetPath.split("/").pop()} (loaded)`;
    } else {
      sel$.selectedIndex = 0;
      loadNet();
    }
  } catch (err) {
    if (seq === netListSeq)
      $("netName").textContent = String(err.message || err);
  }
}

async function loadNet() {
  const { repo } = parseRepo($("netRepo").value);
  const path = $("netSel").value;
  if (!path) return;
  netLabel = path.split("/").pop();
  $("netName").textContent = `loading ${netLabel}…`;
  try {
    const r = await call({ t: "loadNetHf",
                            url: resolveUrl("model", repo, path) });
    loadedNetPath = path;
    $("netName").textContent = `${netLabel} (${r.ep})`;
    if (r.ep === "wasm") await useCpuFallback();
  } catch (err) {
    $("netName").textContent = `load failed: ${err.message || err}`;
  }
}

/* selecting a net is the whole interaction — it loads immediately */
$("netSel").onchange = () => {
  if ($("netSel").value && $("netSel").value !== loadedNetPath) loadNet();
};
/* the repo field is the only input: typing (debounced) or committing the
 * field (blur/Enter) re-lists that repo/path's .onnx files */
$("netRepo").addEventListener("input", () => {
  clearTimeout(netListTimer);
  netListTimer = setTimeout(listNets, 600);
});
$("netRepo").addEventListener("change", () => {
  clearTimeout(netListTimer);
  listNets();
});

$("sims").addEventListener("input", (e) => {
  sims = +e.target.value;
  $("simsVal").textContent = sims;
});

$("temp").addEventListener("input", (e) => {
  temperature = +e.target.value;
  $("tempVal").textContent = temperature.toFixed(1);
});

$("bNew").onclick = newGame;
$("bFlip").onclick = () => {
  humanSide = humanSide === "w" ? "b" : "w";
  newGame().then(() => {
    if (S.side !== humanSide && S.termination === "PLAYING") engineTurn();
  });
};
$("bUndo").onclick = async () => {
  // rewind the engine's reply and your move (two half-moves)
  const plies = playHist.length >= 2 ? 2 : 1;
  const r = await call({ t: "undo", plies });
  S = r.state;
  playHist.length -= r.popped;
  render();
};
$("bEngineMove").onclick = () => engineTurn();

// CPU fallback: no usable WebGPU → 8 walkers and ≤200 sims (100 default).
const gpu = await gpuAdapterAvailable();
cpuMode = !gpu;
if (cpuMode) setSimBudget(true);
const init = await call({ t: "init", walkers: gpu ? 256 : 8 });
engineWalkers = init.walkers;
newGame();
listNets();                  // list the pre-filled repo after init
