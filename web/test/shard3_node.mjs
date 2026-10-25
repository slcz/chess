#!/usr/bin/env node
/*
 * shard3_node.mjs — the SPSHARD1 v3 gate.
 *
 * Builds a synthetic v3 shard (the engine's own movegen supplies the
 * legal-move lists and the π order the file format promises), parses it
 * with the page's shard.js, then replays through the wasm engine and
 * asserts the whole walkthrough contract:
 *   - header/version/truncation handled (a killed last game is dropped)
 *   - per ply: file n_legal == engine legal count, probs zip 1:1 with the
 *     ENGINE MOVEGEN ORDER, played move's prob is the argmax (as the
 *     writer recorded), π sums to ~1 in f16 precision
 *   - f16 decode round-trips the encoded values
 *
 * Usage: node web/test/shard3_node.mjs
 */
import { createEngine } from './parity_node.mjs';
import { parseShard } from '../app/shard.js';

const M = await createEngine();
M._chess_init();

/* ---- f16 encode (round-to-nearest) for the synthetic writer ------------ */
function f16enc(x) {
  const buf = new ArrayBuffer(4), dv = new DataView(buf);
  // double → half via clamping the f32 bits; good enough for test data
  dv.setFloat32(0, x);
  const fbits = dv.getUint32(0);
  const sign = (fbits >> 31) & 1, exp = ((fbits >> 23) & 0xff) - 127,
        man = fbits & 0x7fffff;
  if (exp >= 16) return sign << 15 | 0x7c00;         // inf (test never needs more)
  if (exp >= -14) {
    const half = ((exp + 15) << 10) | (man >> 13);
    // round to nearest even on the dropped 13 bits
    const round = (man >> 12) & 1 && !(half === 0x7bff);
    return sign << 15 | (half + round);
  }
  // denormal
  return sign << 15 | Math.round(man / (1 << 23) * 2 ** -14 * 1024);
}
function f16dec(u) {
  const s = (u >> 15) & 1, e = (u >> 10) & 0x1f, m = u & 0x3ff;
  let v;
  if (e === 0) v = m * 2 ** -24;
  else if (e === 31) v = m ? NaN : Infinity;
  else v = (1 + m / 1024) * 2 ** (e - 15);
  return s ? -v : v;
}
for (const x of [0.5, 0.25, 0.05, 0.947, 1.0, 0.001])
  if (Math.abs(f16dec(f16enc(x)) - x) > 1e-3 * Math.max(x, 1e-3))
    { console.error(`FAIL: f16 roundtrip ${x} -> ${f16dec(f16enc(x))}`); process.exit(1); }

/* ---- engine helpers ----------------------------------------------------- */
const legalAt = (fen, ucis) => {
  const g = M._chess_game_new(0);
  const fb = M._malloc(128), ub = M._malloc(16), mb = M._malloc(2048);
  M.stringToUTF8(fen, fb, 128);
  M._chess_set_fen(g, fb);
  for (const u of ucis) { M.stringToUTF8(u, ub, 16); M._chess_push_uci(g, ub); }
  const n = M._chess_legal_moves(g, mb, 256);
  const bytes = new Uint8Array(M.HEAPU8.buffer, mb, n * 8);
  const out = [];
  for (let i = 0; i < n; i++) {
    let s = "";
    for (let j = i * 8; j < i * 8 + 8 && bytes[j]; j++)
      s += String.fromCharCode(bytes[j]);
    out.push(s);
  }
  M._chess_game_free(g); M._free(fb); M._free(ub); M._free(mb);
  return out;
};

/* ---- build a synthetic v3 shard ----------------------------------------- */
const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const SCRIPTS = [                     // [fen, [plies]] scripted openings
  [START, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"]],
  [START, ["d2d4", "d7d5", "c1f4", "c7c6"]],
];

const enc = new TextEncoder();
const parts = [];
const wbuf = (arr) => parts.push(Uint8Array.from(arr));
const wstr = (s, n) => {
  const b = enc.encode(s); wbuf(b.subarray(0, n));
  wbuf(new Array(n - Math.min(b.length, n)).fill(0));
};
const wu32 = (v) => wbuf([v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255]);
const wu16 = (v) => wbuf([v & 255, (v >> 8) & 255]);
const wf64 = (v) => { const b = new DataView(new ArrayBuffer(8));
                      b.setFloat64(0, v, true); wbuf(new Uint8Array(b.buffer)); };

wstr("SPSHARD1", 8); wu32(3); wu32(0);   // version 3, worker 0
{ const b = new DataView(new ArrayBuffer(8)); b.setBigUint64(0, 0x1234n, true);
  wbuf(new Uint8Array(b.buffer)); }      // unit_seed

let gseq = 0;
const truth = [];                        // per-ply {fen0, ucis0, uci, probs}
for (const [fen, plies] of SCRIPTS) {
  wu32(gseq++); wu32(plies.length);
  wbuf([1, 255]);                        // natural end, no resigner
  wu32(300);                             // nominal sims
  wf64(gseq % 2 ? 1.0 : 0.5);            // z_white
  wstr(fen, 96);
  for (let i = 0; i < plies.length; i++) {
    const ucis0 = plies.slice(0, i);     // position BEFORE this ply
    const legal = legalAt(fen, ucis0);
    const uci = plies[i];
    const idx = legal.indexOf(uci);
    if (idx < 0) throw new Error(`scripted ${uci} not legal at ply ${i}`);
    const probs = legal.map((_, k) => k === idx ? 0.382 : 0.618 / (legal.length - 1));
    wbuf([i % 2]); wstr(uci, 8); wu16(legal.length);
    wf64(idx === 0 ? 0.25 : -0.25);      // outlook (mover view)
    wu16(24);                            // per-ply sims (the loop1 learn cap)
    for (const p of probs) wu16(f16enc(p));
    truth.push({ fen, ucis0, uci, legal, probs });
  }
}
// a killed last game: header + partial ply, never finishes
wu32(gseq); wu32(9); wbuf([3, 255]); wu32(300); wf64(0.5); wstr(START, 96);
wbuf([0]); wstr("e2e4", 8); wu16(20); wbuf(new Array(13).fill(0));

const shardBuf = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
{ let o = 0; for (const p of parts) { shardBuf.set(p, o); o += p.length; } }
// deep-copy into a plain ArrayBuffer (parseShard takes DataView-able bytes)
const ab = shardBuf.buffer.slice(shardBuf.byteOffset,
                                 shardBuf.byteOffset + shardBuf.byteLength);

/* ---- parse and assert ---------------------------------------------------- */
const shard = parseShard(ab);
let fails = 0;
const fail = (msg) => { console.error("FAIL:", msg); fails++; };

if (shard.version !== 3) fail(`version ${shard.version} != 3`);
if (!shard.truncated) fail("truncated trailing game not detected");
if (shard.games.length !== SCRIPTS.length)
  fail(`games ${shard.games.length} != ${SCRIPTS.length} (killed one must drop)`);

let checked = 0;
for (const g of shard.games) {
  for (let i = 0; i < g.nPlies; i++) {
    const ply = g.plies[i], t = truth[checked++];
    if (ply.playedUci !== t.uci) fail(`ply ${i}: ${ply.playedUci} != ${t.uci}`);
    if (ply.pi !== null) fail("v3 ply.pi must be null (worker maps it)");
    if (ply.nLegal !== t.legal.length)
      fail(`n_legal ${ply.nLegal} != engine ${t.legal.length}`);
    const sum = ply.probs.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 0.01)
      fail(`π sums to ${sum.toFixed(4)} (f16 noise should stay ~1)`);
    const best = ply.probs.indexOf(Math.max(...ply.probs));
    const mapped = t.legal[best];        // engine gen order zip
    if (mapped !== t.uci)
      fail(`argmax prob maps to ${mapped}, played was ${t.uci}`);
  }
}
if (checked !== truth.length) fail(`parsed ${checked} plies, wrote ${truth.length}`);

console.log(fails ? `shard3 gate FAILED (${fails})` :
  `shard3 gate OK: ${shard.games.length} games, ${checked} plies, ` +
  `truncation dropped, π zips engine-movegen order`);
process.exit(fails ? 1 : 0);
