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
import { Board, renderEvalBar } from "./board.js";
import { listFiles, parseRepo, resolveUrl } from "./hf.js";

const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("engine_worker.js", import.meta.url));
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
// dev probe: live page state for the headless acceptance run
window.__dbg = () => ({ thinking, humanSide, sims,
  S: S && { fen: S.fen, side: S.side, term: S.termination },
  histLen: playHist.length, pend: pending.size });
let sel = null;                    // selected square
let S = null;                      // current engine state
let thinking = false;

/* play-mode state */
let humanSide = "w";
let sims = 200;

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
  p.querySelectorAll("button").forEach((b) => b.onclick = () => {
    p.style.display = "none";
    doMove(matches[0].slice(0, 4) + b.dataset.p);
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
    const r = await call({ t: "engineMove", sims });
    S = r.state;
    showThink(r.result);
    playHist.push(r.move);
  } finally {
    thinking = false;
  }
  render();
}

/** Eval bar after a think (rootV is the parent's view of the mover;
 * show WHITE's). */
function showThink(r) {
  const whiteV = S.side === "w" ? -r.rootV : r.rootV;
  renderEvalBar($("evalbar"), whiteV);
}

/* ---- panel wiring: the net loads from Hugging Face --------------------- */

/* the repo field accepts a bare "owner/name", "owner/name/with/path",
 * or a pasted huggingface.co URL — parseRepo normalizes all of them. */

let netListTimer = null;
let loadedNetPath = "";      // the .onnx currently in the worker

async function listNets() {
  const { repo, path } = parseRepo($("netRepo").value);
  if (!repo) return;
  $("netName").textContent = `listing ${repo}…`;
  try {
    const files = (await listFiles("model", repo, path))
      .filter((f) => f.type === "file" && f.path.endsWith(".onnx"));
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
  } catch (err) {
    $("netName").textContent = `load failed: ${err.message || err}`;
  }
}

/* selecting a net is the whole interaction — it loads immediately */
$("netSel").onchange = () => {
  if ($("netSel").value && $("netSel").value !== loadedNetPath) loadNet();
};
listNets();                               // list the pre-filled repo on boot

$("sims").addEventListener("input", (e) => {
  sims = +e.target.value;
  $("simsVal").textContent = sims;
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

await call({ t: "init" });   // boot the wasm engine (dummy eval until a net loads)
newGame();
