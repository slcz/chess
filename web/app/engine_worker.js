/*
 * engine_worker.js — owns the wasm engine (the page's whole chess truth)
 * on a dedicated Web Worker, because puct_think blocks its calling
 * thread and the UI thread must never block.
 *
 * The worker keeps TWO C handles per play-mode game, deliberately:
 *   - a ChessGame (FFI) — the game record: fen/legal/termination, and
 *     the SAN history the move list renders from;
 *   - a PuctGame — the search tree, advanced with puct_play so the
 *     subtree survives move to move (tree reuse, the same way
 *     selfplay.c drives it).
 * Walkthrough mode replays shard games through a third, record-only
 * ChessGame — no search: the shard already contains the thinking.
 *
 * SAN (chess_ai/san.py ported) is computed here too: piece_at + legal
 * moves + push/pop all live on the handle, keeping the page itself free
 * of chess rules.
 *
 * Every message carries an id; replies echo it. The engine is the
 * SINGLE-THREADED build: a think runs as poll → eval → publish steps
 * (puct_think_start / puct_zone_publish), and the net forward —
 * ort_session.js on this same thread — fills the staged rows between
 * polls. No SharedArrayBuffer, no cross-origin isolation.
 */
"use strict";
importScripts("../../engine/build/wasm/chess_engine.js");

const TERM_NAMES = ["PLAYING", "CHECKMATE", "INSUFFICIENT_MATERIAL",
  "STALEMATE", "SEVENTYFIVE_MOVES", "FIVEFOLD_REPETITION",
  "FIFTY_MOVES", "THREEFOLD_REPETITION"];
const PIECE_LETTER = ["", "N", "B", "R", "Q", "K"];   // engine type 0..5
const EMPTY = 12;
const BLK_PER_SIM = 20 << 10; // node block: 20 KB per sim — 4 MB @ 200 sims,
                              // 40 MB @ 2000; a search that outgrows its
                              // block even after compact gives up, so the
                              // block is sized to the whole think (the
                              // block upgrades lazily in think(), tree
                              // reuse is lost on the upgrade — the budget
                              // knob was turned deliberately)
const BLK_MIN = 4 << 20;
const WALKERS = 256;          // concurrent descents per think (= zone rows:
                              // one full batch ships per net forward)

let M = null;                 // the emscripten module
let game = 0;                 // ChessGame* (play mode)
let puct = 0;                 // PuctGame*  (play mode)
let wt = 0;                   // ChessGame* (walkthrough replay)
let history = [];             // [{uci, san, fen}] — fen AFTER the move
let buf = {};                 // scratch buffers in wasm memory

function alloc() {
  for (const [k, n] of [["fen", 128], ["uci", 16], ["moves", 256 * 8],
                        ["mv", 256 * 2], ["vis", 256 * 4], ["q", 256 * 8],
                        ["rn", 4], ["rs", 8], ["prior", 256 * 4]])
    buf[k] = M._malloc(n);
}

const mem = () => M.HEAPU8.buffer;
const str = (p) => M.UTF8ToString(p);

/* Raw wasm exports take POINTERS, not strings — marshal through the
 * scratch buffer (buf.uci is never live across calls). */
function pushUci(g, uci) {
  M.stringToUTF8(uci, buf.uci, 16);
  return M._chess_push_uci(g, buf.uci);
}

function legalMoves(g) {
  const n = M._chess_legal_moves(g, buf.moves, 256);
  const bytes = new Uint8Array(mem(), buf.moves, n * 8);
  const out = [];
  for (let i = 0; i < n; i++) {
    let s = "";
    for (let j = i * 8; j < i * 8 + 8 && bytes[j]; j++)
      s += String.fromCharCode(bytes[j]);
    out.push(s);
  }
  return out;
}

/** Position facts of a ChessGame handle — no history attached; callers
 * compose move lists from their own records. */
function readState(g, lastUci) {
  M._chess_fen(g, buf.fen);
  const term = M._chess_termination(g);
  return {
    fen: str(buf.fen),
    side: M._chess_side(g) === 0 ? "w" : "b",
    legal: term === 0 ? legalMoves(g) : [],
    termination: TERM_NAMES[term],
    check: !!M._chess_in_check(g),
    ply: M._chess_ply(g),
    lastMove: lastUci || null,
  };
}

/* ---- SAN (port of chess_ai/san.py; the board is restored) -------------- */
function uciToSan(g, uci) {
  const sq = (name) =>
    (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
  const pieceAt = (name) => M._chess_piece_at(g, sq(name));
  const legal = legalMoves(g);
  const frm = uci.slice(0, 2), to = uci.slice(2, 4);
  const promo = uci[4] ? uci[4].toUpperCase() : "";
  const pc = pieceAt(frm), ptype = pc % 6;
  const isCapture = pieceAt(to) !== EMPTY ||
    (ptype === 0 && frm[0] !== to[0]);
  let core;
  if (ptype === 5 && Math.abs(to.charCodeAt(0) - frm.charCodeAt(0)) === 2) {
    core = to[0] > frm[0] ? "O-O" : "O-O-O";
  } else if (ptype === 0) {
    core = (isCapture ? frm[0] + "x" : "") + to + (promo ? "=" + promo : "");
  } else {
    const others = new Set(
      legal.filter((m) => m.slice(2, 4) === to && m.slice(0, 2) !== frm
                   && pieceAt(m.slice(0, 2)) === pc)
        .map((m) => m.slice(0, 2)));
    let dis = "";
    if (others.size) {
      if (![...others].some((s) => s[0] === frm[0])) dis = frm[0];
      else if (![...others].some((s) => s[1] === frm[1])) dis = frm[1];
      else dis = frm;
    }
    core = PIECE_LETTER[ptype] + dis + (isCapture ? "x" : "") + to;
  }
  pushUci(g, uci);
  const suffix = M._chess_termination(g) === 1 ? "#"
    : (M._chess_in_check(g) ? "+" : "");
  M._chess_pop(g);
  return core + suffix;
}

/* ---- search ------------------------------------------------------------- */
let blkBytes = 0;             // current PuctGame node block size

/** Make sure the search tree's block can sustain a whole think at `sims`
 * (an OOM after compact aborts the search — never let that be the reason
 * a think falls short). Upgrades by freeing + replanting at the current
 * position; a bigger block from an earlier, higher budget is kept. */
function ensureBlock(sims) {
  const want = Math.max(BLK_MIN, sims * BLK_PER_SIM);
  if (puct && blkBytes >= want) return;
  if (puct) M._puct_game_free(puct);
  blkBytes = want;
  puct = M._puct_game_new(blkBytes);
  M._puct_set_walkers(puct, WALKERS);
  M._chess_fen(game, buf.fen);
  M._puct_game_set_fen(puct, buf.fen);
}

async function dummyEval(req) {
  const mem = M.HEAPU8.buffer;    // fresh read: growth replaces it
  new Float32Array(mem, req.logp, req.n * 4240).fill(0);
  const w = new Float32Array(mem, req.wdl, req.n * 3);
  for (let r = 0; r < req.n; r++) {
    w[3 * r] = 0;
    w[3 * r + 1] = 1;
    w[3 * r + 2] = 0;
  }
}

async function think(sims) {
  ensureBlock(sims);
  // poll → eval → publish, until the think completes
  const out = M._malloc(5 * 4);
  try {
    while (!M._puct_think_start(puct, sims, out)) {
      const [n, planes, mask, logp, wdl] =
        new Int32Array(M.HEAPU8.buffer, out, 5);
      await (globalThis.__ortRun ?? dummyEval)(
        { n, planes, mask, logp, wdl });
      M._puct_zone_publish(puct);
    }
  } finally {
    M._free(out);
  }
  const oom = M._puct_oom_count(puct), compact = M._puct_compact_count(puct);
  if (oom || compact)
    console.warn(`think: oom=${oom} compact=${compact} blk=` +
      `${(blkBytes / 1e6).toFixed(0)}MB — the search gave up early; ` +
      `raise the block (BLK_PER_SIM)`);
  const n = M._puct_root_children(puct, buf.mv, buf.vis, buf.q, buf.prior);
  M._puct_root_stats(puct, buf.rn, buf.rs);
  // value_sum is a SUM over visits (parent view); average it into a
  // root value in [-1, 1] before display math touches it
  const rootN = new Uint32Array(mem(), buf.rn, 1)[0];
  const rootV = rootN
    ? new Float64Array(mem(), buf.rs, 1)[0] / rootN
    : 0;
  const children = [];
  const mv = new Uint16Array(mem(), buf.mv, n);
  const vis = new Uint32Array(mem(), buf.vis, n);
  const q = new Float64Array(mem(), buf.q, n);
  for (let i = 0; i < n; i++) {
    M._move_to_uci(mv[i], buf.uci);
    children.push({ uci: str(buf.uci), visits: vis[i], q: q[i] });
  }
  M._move_to_uci(M._puct_best_move(puct), buf.uci);
  return { best: str(buf.uci), children, rootV };
}

function syncPuct(uci) {
  M.stringToUTF8(uci, buf.uci, 16);
  return M._puct_play(puct, M._uci_to_move(M._puct_state(puct), buf.uci));
}

/* ---- messages ------------------------------------------------------------ */
self.onmessage = async (e) => {
  const m = e.data;
  const reply = (r) => self.postMessage({ id: m.id, ...r });
  try {
    if (m.t === "init") {
      M = await createEngine({
        // the wasm sits next to the engine glue, not next to this worker
        locateFile: (f) =>
          new URL("../../engine/build/wasm/" + f, self.location.href).href,
      });
      M._chess_init();
      alloc();
      M._puct_attach_js(WALKERS);   // zone rows — one row per walker, so a
                                    // full batch ships in a single forward
      reply({ t: "ready" });
    } else if (m.t === "loadNetHf") {
      // fetch the .onnx right here (the page must not carry 30 MB through
      // postMessage) and stream progress to the panel as it lands
      const { fetchWithProgress } = await import(
        new URL("./hf.js", self.location.href).href);
      const bytes = await fetchWithProgress(m.url, (loaded, total) =>
        self.postMessage({ t: "netProgress", loaded, total }));
      const { loadSession } = await import(
        new URL("./ort_session.js", self.location.href).href);
      const { run, ep } = await loadSession(bytes);
      // the buffer is read FRESH per call: memory growth replaces it
      globalThis.__ortRun = (req) => run(req, M.HEAPU8.buffer);
      reply({ t: "net", ep });
    } else if (m.t === "newGame") {
      if (puct) M._puct_game_free(puct);
      if (game) M._chess_game_free(game);
      game = M._chess_game_new(0);
      blkBytes = BLK_MIN;         // think() upgrades to the slider's budget
      puct = M._puct_game_new(blkBytes);
      // walkers are set ONCE, before the first think (the search creates
      // its default 8 lazily otherwise); spec-batch tradeoff: more walkers
      // fill bigger batches but pick nodes more speculatively
      M._puct_set_walkers(puct, WALKERS);
      history = [];
      if (m.fen) {
        M.stringToUTF8(m.fen, buf.fen, 128);
        M._chess_set_fen(game, buf.fen);
        M._puct_game_set_fen(puct, buf.fen);
      }
      reply({ t: "state", state: readState(game) });
    } else if (m.t === "state") {
      reply({ t: "state", state: readState(game, history.at(-1)?.uci) });
    } else if (m.t === "play") {           // a human (or scripted) move
      const san = uciToSan(game, m.uci);
      pushUci(game, m.uci);
      syncPuct(m.uci);
      M._chess_fen(game, buf.fen);
      history.push({ uci: m.uci, san, fen: str(buf.fen) });
      reply({ t: "state", state: readState(game, m.uci),
              move: { uci: m.uci, san } });
    } else if (m.t === "engineMove") {     // think + commit the argmax
      const r = await think(m.sims);
      const san = uciToSan(game, r.best);
      pushUci(game, r.best);
      syncPuct(r.best);
      M._chess_fen(game, buf.fen);
      history.push({ uci: r.best, san, fen: str(buf.fen) });
      reply({ t: "moved", result: r, state: readState(game, r.best),
              move: { uci: r.best, san } });
    } else if (m.t === "hint") {           // think, keep the tree, no commit
      reply({ t: "hint", result: await think(m.sims) });
    } else if (m.t === "undo") {           // rewind m.plies half-moves
      let steps = m.plies || 1, popped = 0;
      while (steps-- && history.length) {
        history.pop();
        popped++;
        M._chess_pop(game);
      }
      M._chess_fen(game, buf.fen);
      M._puct_game_set_fen(puct, buf.fen); // replant: the tree is history
      reply({ t: "state", state: readState(game, history.at(-1)?.uci),
              popped });
    } else if (m.t === "wtLoad") {
      // Walkthrough: position = start_fen + the first m.plies shard moves.
      // m.sansPlies (optional) computes SANs by replaying forward and
      // rewinding — asked once per game load, not per ply jump.
      // m.piProbs (optional, v3 shards) maps the ply's dense f16 π: the
      // replayed position is exactly the pre-move one, so the probs zip
      // 1:1 with this engine's legal-move list (deterministic movegen;
      // the shard's n_legal must agree — anything else gets no arrows).
      if (!wt) wt = M._chess_game_new(0);
      M.stringToUTF8(m.fen, buf.fen, 128);
      M._chess_set_fen(wt, buf.fen);
      for (const uci of m.plies || []) pushUci(wt, uci);
      const sans = [];
      for (const uci of m.sansPlies || []) {
        sans.push(uciToSan(wt, uci));
        pushUci(wt, uci);
      }
      for (let i = 0; i < (m.sansPlies || []).length; i++) M._chess_pop(wt);
      let pi = null;
      if (m.piProbs) {
        const legal = legalMoves(wt);
        if (legal.length === m.piProbs.length)
          pi = legal.map((u, i) => [u, m.piProbs[i]]);
      }
      reply({ t: "state", state: readState(wt, m.plies?.at(-1)), sans, pi });
    }
  } catch (err) {
    reply({ t: "error", error: String(err && err.stack || err) });
  }
};
