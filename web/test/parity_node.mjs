#!/usr/bin/env node
/*
 * parity_node.mjs — the wasm half of the search-parity gate.
 *
 * Drives the Emscripten module through EXACTLY the schedule of
 * engine/tests/parity_dump.c (same FENs, sims, think→play cadence) and
 * diffs the transcript against the native binary line by line. With the
 * deterministic dummy evaluator and all-double search math, native and
 * wasm must agree bit-for-bit — and the transcripts print doubles as
 * IEEE754 hex, so number formatting cannot fake it.
 *
 * Usage: node web/test/parity_node.mjs
 * Exit 0 = parity; exit 1 prints the first differing lines.
 */
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(fileURLToPath(new URL('../../', import.meta.url)));

export const createEngine = () =>
  require(path.join(ROOT, 'engine/build/wasm/chess_engine.js'))();

const FENS = [
  null, // the standard start position
  'rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2',
  '7k/8/8/8/8/8/R7/1R5K w - - 0 1',
  'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4',
];
const SIMS = [16, 64, 257];
const BLK = 4 << 20;
const MAX_MOVES = 256;

/** Drive one think through the single-threaded stepping protocol: poll
 * until a bank awaits evaluation, hand its rows to `evalFn`, publish.
 * evalFn(req) → Promise; req = {n, planes, mask, logp, wdl} byte offsets. */
export async function driveThink(M, g, sims, evalFn) {
  const out = M._malloc(5 * 4);
  try {
    while (!M._puct_think_start(g, sims, out)) {
      const [n, planes, mask, logp, wdl] =
        new Int32Array(M.HEAPU8.buffer, out, 5);
      await evalFn({ n, planes, mask, logp, wdl });
      M._puct_zone_publish(g);
    }
  } finally {
    M._free(out);
  }
}

/** The dummy evaluator in JS: uniform priors + a draw, straight into the
 * wasm heap (the native dummy_eval's twin for the stepping protocol).
 * `after` (optional) delays the reply — exercises the suspend/resume. */
export function jsDummyEval(M, after) {
  return async ({ n, logp, wdl }) => {
    if (after) await after();
    const mem = M.HEAPU8.buffer;    // fresh read: growth replaces it
    new Float32Array(mem, logp, n * 4240).fill(0);
    const w = new Float32Array(mem, wdl, n * 3);
    for (let r = 0; r < n; r++) {
      w[3 * r] = 0;
      w[3 * r + 1] = 1;
      w[3 * r + 2] = 0;
    }
  };
}

/** Run the parity_dump schedule against a live, ALREADY-SET-UP module
 * (chess_init + zone_init or attach_js done by the caller); returns the
 * transcript lines. Shared by the dummy gate and the bridge test. */
export async function runTranscript(M, evalFn) {
  const lines = [];
  const mv = M._malloc(MAX_MOVES * 2);
  const vis = M._malloc(MAX_MOVES * 4);
  const q = M._malloc(MAX_MOVES * 8);
  const rn = M._malloc(4);
  const rs = M._malloc(8);
  const fen = M._malloc(128);
  const uci = M._malloc(16);
  // Shared memory grows in place (SAB), so a view taken now stays valid
  // for everything allocated below it; read helpers re-view per access.
  const mem = M.HEAPU8.buffer;
  const f64bits = (ptr) =>
    new BigUint64Array(mem, ptr, 1)[0].toString(16).padStart(16, '0');

  const dump = (tag, sims, g) => {
    const n = M._puct_root_children(g, mv, vis, q, 0);
    M._move_to_uci(M._puct_best_move(g), uci);
    M._puct_root_stats(g, rn, rs);
    lines.push(
      `${tag} sims=${sims} best=${M.UTF8ToString(uci)} ` +
      `rootN=${new Uint32Array(mem, rn, 1)[0]} rootW=${f64bits(rs)}`);
    const v = new Uint32Array(mem, vis, n);
    for (let i = 0; i < n; i++) {
      M._move_to_uci(new Uint16Array(mem, mv, n)[i], uci);
      lines.push(`C ${M.UTF8ToString(uci)} n=${v[i]} q=${f64bits(q + 8 * i)}`);
    }
  };

  M._chess_init();
  for (const f of FENS) {
    const g = M._puct_game_new(BLK);
    if (f) {
      M.stringToUTF8(f, fen, 128);
      if (M._puct_game_set_fen(g, fen) !== 0) throw new Error(`bad fen ${f}`);
    }
    for (const sims of SIMS) {
      await driveThink(M, g, sims, evalFn);
      dump('T', sims, g);
      if (M._puct_game_over(g))
        break; // sampling ended the game (parity_dump stops here too)
      M._puct_play(g, M._puct_best_move(g));
    }
    if (!M._puct_game_over(g)) {
      await driveThink(M, g, 64, evalFn);
      dump('P', 64, g);
    }
    M._puct_game_free(g);
  }
  lines.push('DONE');
  return lines;
}

/** Diff helper shared with bridge_node.mjs. */
export function diffTranscripts(nativeLines, wasmLines, label) {
  let bad = 0;
  for (let i = 0; i < Math.max(nativeLines.length, wasmLines.length); i++) {
    if (nativeLines[i] !== wasmLines[i]) {
      console.error(`line ${i + 1} differs:\n  native: ${nativeLines[i]}\n  wasm:   ${wasmLines[i]}`);
      if (++bad >= 5) break;
    }
  }
  if (bad) {
    console.error(`${label} FAIL (${nativeLines.length} native vs ${wasmLines.length} wasm lines)`);
    process.exit(1);
  }
  console.log(`${label} OK: ${wasmLines.length} lines bit-identical to native`);
}

export const nativeTranscript = () =>
  execFileSync(path.join(ROOT, 'engine/build/parity_dump'))
    .toString().split('\n').filter((l) => l.length);

// Run the gate only when invoked directly, not when imported.
if (import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  const M = await createEngine();
  M._chess_init();
  M._puct_zone_init(8); // zone only — the eval runs in JS (jsDummyEval)
  diffTranscripts(nativeTranscript(),
                  await runTranscript(M, jsDummyEval(M)), 'parity');
}
