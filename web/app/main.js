/*
 * main.js — page wiring. Two modes over one wasm engine worker:
 *
 *   play        you vs the net (or vs the dummy evaluator before any
 *               model is loaded); sims slider; visit arrows + eval bar
 *               after every think (doubles as "show hints" on your turn)
 *   walkthrough open a .spshard (SPSHARD1): pick a game, step through
 *               plies, π arrows + eval bar come straight from the bytes
 *               the shard recorded per ply — zero search, zero net
 *
 * All chess truth (legality, termination, SAN, replay) comes from the
 * worker; this file only renders and routes messages.
 */
import { Board, renderEvalBar } from "./board.js";
import { parseShard, END_NAMES } from "./shard.js";
import { indexToUci } from "./slots.js";

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
    const maxP = Math.max(...ply.pi.map(([, p]) => p), 1e-9);
    board.arrows(
      ply.pi.map(([slot, p]) => ({
        ...splitUci(indexToUci(slot, ply.mover)),
        weight: p / maxP, kind: "pi",
      })).concat([{ ...splitUci(ply.playedUci), weight: 1, kind: "played" }]));
  } else {
    renderEvalBar($("evalbar"), 0);
    board.arrows([]);
  }
  $("wtSlider").value = wtPly;
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
$("shardFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  shard = parseShard(await f.arrayBuffer());
  $("shardInfo").textContent =
    `${shard.games.length} games (v${shard.version}` +
    `${shard.truncated ? ", last record truncated" : ""})`;
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
});

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
  const r = await call({ t: "wtLoad", fen: g.startFen,
                          plies: g.plies.slice(0, wtPly).map((x) => x.playedUci) });
  if (seq !== wtLoadSeq) return;
  S = r.state;
  render();
}

/* ---- panel wiring ------------------------------------------------------ */
$("netFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("netName").textContent = `loading ${f.name}…`;
  const r = await call({ t: "loadNet", bytes: await f.arrayBuffer() });
  $("netName").textContent = `${f.name} (${r.ep})`;
});

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
  else $("shardInfo").textContent = "open a .spshard to begin";
};

await call({ t: "init" });   // boot the wasm engine (dummy eval until a net loads)
newGame();
