/*
 * main.js — page wiring. Two modes over one wasm engine worker:
 *
 *   play        you vs the net (or vs the dummy evaluator before any
 *               model is loaded); sims slider; visit arrows + eval bar
 *               after every think (doubles as "show hints" on your turn)
 *   walkthrough open a .spbin shard from Hugging Face: pick a game, step
 *               through plies, π arrows + eval bar come straight from the
 *               bytes the shard recorded per ply — zero search, zero net
 *
 * Artifacts load straight from Hugging Face (no local files, no server):
 * the .onnx through the worker (hf.js there too — the page must not carry
 * 30 MB through postMessage), the .spbin here. v3 shards record π as a
 * dense f16 vector in engine movegen order; the worker replays each shown
 * ply's pre-move position and zips probs with its own legal-move list.
 * All chess truth (legality, termination, SAN, replay) comes from the
 * worker; this file only renders and routes messages.
 */
import { Board, renderEvalBar } from "./board.js";
import { parseShard, END_NAMES } from "./shard.js";
import { indexToUci } from "./slots.js";
import { listFiles, parseRepo, resolveUrl, fetchWithProgress } from "./hf.js";

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
window.__dbg = () => ({ mode, thinking, humanSide, sims,
  S: S && { fen: S.fen, side: S.side, term: S.termination },
  histLen: playHist.length, pend: pending.size });
let mode = "play";
let sel = null;                    // selected square (play mode)
let S = null;                      // current engine state
let thinking = false;

/* play-mode state */
let humanSide = "w";
let sims = 200;

/* walkthrough state */
let shard = null;
let wtGame = 0;
let wtPly = 0;
let wtSans = [];

/* HF panel state (netLabel names the file being downloaded) */
let netLabel = "";

/* ---- rendering -------------------------------------------------------- */
function boardState() {
  return S ? { ...S, sel } : {};
}

function render() {
  if (!S) return;
  board.render(boardState());
  $("fen").textContent = S.fen;
  if (mode === "play") renderPlay();
  else renderWalk();
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

function renderWalk() {
  const g = shard.games[wtGame];
  const ply = g.plies[wtPly];
  $("status").textContent =
    `game ${g.gameIndex}: ply ${wtPly}/${g.nPlies} — ` +
    `${END_NAMES[g.endKind]}, ${g.zWhite === 0.5 ? "½–½" : g.zWhite === 1 ? "1–0" : "0–1"}`;
  const moves = $("moves");
  moves.innerHTML = "";
  g.plies.forEach((p, i) => {
    if (i % 2 === 0) {
      const num = document.createElement("span");
      num.className = "num";
      num.textContent = (i / 2 + 1) + ".";
      moves.appendChild(num);
    }
    const li = document.createElement("li");
    li.textContent = wtSans[i] || p.playedUci;
    if (i === wtPly) li.classList.add("cur");
    li.onclick = () => gotoPly(i);
    moves.appendChild(li);
    if (i % 2 === 0) moves.appendChild(document.createTextNode(" "));
  });
  // eval bar + π arrows from the shard bytes (mover view → white view)
  if (ply) {
    const v = ply.mover === 0 ? ply.outlook : -ply.outlook;
    renderEvalBar($("evalbar"), v);
    board.arrows(walkArrows(ply));
  } else {
    renderEvalBar($("evalbar"), 0);
    board.arrows([]);
  }
  $("wtSlider").value = wtPly;
}

/** π overlay for one ply. v1/v2: (slot, prob) pairs → uci via the slot
 * table. v3: the worker's replay mapped the dense f16 π to ucis — ALL
 * legal moves carry a prob, so thin to the top 8 or the board drowns. */
function walkArrows(ply) {
  let pairs;
  if (ply.piUci) {
    const keep = new Set([...ply.piUci].sort((a, b) => b[1] - a[1])
      .slice(0, 8).map(([u]) => u));
    pairs = ply.piUci.filter(([u]) => keep.has(u));
  } else {
    pairs = (ply.pi || []).map(([slot, p]) => [indexToUci(slot, ply.mover), p]);
  }
  const maxP = Math.max(...pairs.map(([, p]) => p), 1e-9);
  return pairs.map(([uci, p]) =>
    ({ ...splitUci(uci), weight: p / maxP, kind: "pi" }))
    .concat([{ ...splitUci(ply.playedUci), weight: 1, kind: "played" }]);
}

const splitUci = (uci) => ({ from: uci.slice(0, 2), to: uci.slice(2, 4) });

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

/* ---- play mode -------------------------------------------------------- */
let playHist = [];

async function newGame() {
  mode = "play";
  document.body.dataset.mode = "play";
  playHist = [];
  sel = null;
  const r = await call({ t: "newGame" });
  S = r.state;
  renderEvalBar($("evalbar"), null);
  board.arrows([]);
  render();
}

board.clickCb = async (name) => {
  if (mode !== "play" || thinking || !S
      || S.termination !== "PLAYING" || S.side !== humanSide) return;
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
 * show WHITE's). Play mode draws no board arrows — that is the
 * walkthrough's job. */
function showThink(r) {
  const whiteV = S.side === "w" ? -r.rootV : r.rootV;
  renderEvalBar($("evalbar"), whiteV);
}

/* ---- walkthrough mode -------------------------------------------------- */

let wtLoadSeq = 0;      // supersedes stale game loads when the user jumps

async function loadWalkGame(i) {
  mode = "walkthrough";
  document.body.dataset.mode = "walk";
  wtGame = i;
  const g = shard.games[i];
  $("wtSlider").max = g.nPlies;
  const seq = ++wtLoadSeq;
  // one wtLoad for the full-game SAN list (replayed + rewound in C)
  const r = await call({ t: "wtLoad", fen: g.startFen, plies: [],
                         sansPlies: g.plies.map((p) => p.playedUci) });
  if (seq !== wtLoadSeq) return;   // the user already jumped elsewhere
  wtSans = r.sans;
  await gotoPly(0);
}

async function gotoPly(p) {
  const seq = ++wtLoadSeq;    // a jump supersedes any in-flight game load
  const g = shard.games[wtGame];
  wtPly = Math.max(0, Math.min(g.nPlies, p));
  const ply = g.plies[wtPly];
  // v3: the position loaded here is the ply's PRE-move one — exactly what
  // its dense π was searched from, so the worker can zip probs ↔ legal
  const r = await call({ t: "wtLoad", fen: g.startFen,
                          plies: g.plies.slice(0, wtPly).map((x) => x.playedUci),
                          piProbs: ply ? ply.probs : null });
  if (seq !== wtLoadSeq) return;
  if (ply && r.pi) ply.piUci = r.pi;
  S = r.state;
  render();
}

/* ---- panel wiring: everything loads from Hugging Face ------------------ */

/* panel repo fields accept a bare "owner/name", "owner/name/with/path",
 * or a pasted huggingface.co URL — parseRepo normalizes all of them. */

$("netList").onclick = async () => {
  const { repo, path } = parseRepo($("netRepo").value);
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
    $("netLoad").disabled = !files.length;
    $("netName").textContent = files.length
      ? "pick a net and press Load"
      : "no .onnx in that repo — convert a checkpoint (scripts/export_onnx.py)";
  } catch (err) {
    $("netName").textContent = String(err.message || err);
  }
};

$("netLoad").onclick = async () => {
  const { repo } = parseRepo($("netRepo").value);
  const path = $("netSel").value;
  if (!path) return;
  netLabel = path.split("/").pop();
  $("netName").textContent = `loading ${netLabel}…`;
  try {
    const r = await call({ t: "loadNetHf",
                            url: resolveUrl("model", repo, path) });
    $("netName").textContent = `${netLabel} (${r.ep})`;
  } catch (err) {
    $("netName").textContent = `load failed: ${err.message || err}`;
  }
};

$("shardList").onclick = async () => {
  const { repo, path: repoPath } = parseRepo($("shardRepo").value);
  const prefix = repoPath || $("shardPrefix").value.trim();
  $("shardInfo").textContent = `listing ${repo}/${prefix || "…"}…`;
  try {
    const files = (await listFiles("dataset", repo, prefix))
      .filter((f) => f.type === "file" && f.path.endsWith(".spbin"));
    const sel$ = $("shardSel");
    sel$.innerHTML = "";
    const MAX = 500;
    for (const f of files.slice(0, MAX)) {
      const o = document.createElement("option");
      o.value = f.path;
      o.textContent = f.path;
      sel$.appendChild(o);
    }
    $("shardInfo").textContent = `${files.length} shards` +
      (files.length > MAX ? ` (showing first ${MAX})` : "");
  } catch (err) {
    $("shardInfo").textContent = String(err.message || err);
  }
};

$("shardSel").onchange = loadSelectedShard;

async function loadSelectedShard() {
  const { repo } = parseRepo($("shardRepo").value);
  const path = $("shardSel").value;
  if (!path) return;
  $("shardInfo").textContent = `downloading ${path}…`;
  try {
    const buf = await fetchWithProgress(
      resolveUrl("dataset", repo, path),
      (l, t) => $("shardInfo").textContent =
        `downloading ${path}… ` +
        (t ? `${(l / t * 100).toFixed(0)}%` : `${(l / 1e6).toFixed(1)} MB`));
    shard = parseShard(buf);
    $("shardInfo").textContent =
      `v${shard.version}: ${shard.games.length} games` +
      `${shard.truncated ? ", last record truncated" : ""}`;
    const sel$ = $("wtGameSel");
    sel$.innerHTML = "";
    shard.games.forEach((g, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = `#${g.gameIndex} — ${g.nPlies} plies, ` +
        `${END_NAMES[g.endKind]}, ${g.nSims} sims`;
      sel$.appendChild(o);
    });
    sel$.onchange = () => loadWalkGame(+sel$.value);
    if (shard.games.length) loadWalkGame(0);
  } catch (err) {
    $("shardInfo").textContent = String(err.message || err);
  }
}

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
$("wtSlider").addEventListener("input", (e) => gotoPly(+e.target.value));
$("bWtPrev").onclick = () => gotoPly(wtPly - 1);
$("bWtNext").onclick = () => gotoPly(wtPly + 1);
document.addEventListener("keydown", (e) => {
  if (mode !== "walk") return;
  if (e.key === "ArrowLeft") gotoPly(wtPly - 1);
  else if (e.key === "ArrowRight") gotoPly(wtPly + 1);
});

/* tab switching: walkthrough controls vs play controls */
$("tabPlay").onclick = () => {
  if (mode !== "play") newGame();
};
$("tabWalk").onclick = () => {
  if (mode === "walk") return;
  if (shard && shard.games.length) loadWalkGame(wtGame);
  else $("shardInfo").textContent = "list a HF repo and pick a .spbin to begin";
};

await call({ t: "init" });   // boot the wasm engine (dummy eval until a net loads)
newGame();
