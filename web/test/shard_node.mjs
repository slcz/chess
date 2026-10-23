/*
 * shard_node.mjs — unit test for web/app/shard.js (the v3 SPSHARD1
 * parser) without a browser: hand-builds the same bytes as
 * tests/test_shard_format.py and checks the parse (dense f16 π, per-ply
 * sims, no playedSlot), the hand-rolled f16 decoder (normals,
 * subnormals, half-max), the legacy-version gate, and truncation
 * handling.
 *
 * Usage: node web/test/shard_node.mjs
 */
import assert from "node:assert/strict";
import { parseShard, f16 } from "../app/shard.js";

const enc = new TextEncoder();

function shardBytes(version = 3) {
  const game = new Uint8Array(4 + 4 + 1 + 1 + 4 + 8 + 96);
  const gdv = new DataView(game.buffer);
  gdv.setUint32(0, 7, true);
  gdv.setUint32(4, 2, true);            // n_plies
  gdv.setUint8(8, 1);                   // end_kind
  gdv.setInt8(9, -1);                   // resigner
  gdv.setUint32(10, 600, true);         // n_sims
  gdv.setFloat64(14, 1.0, true);        // z_white
  game.set(enc.encode("fen"), 22);

  function ply(mover, uci, probs, sims, outlook) {
    const b = new Uint8Array(21 + 2 * probs.length);
    const dv = new DataView(b.buffer);
    dv.setUint8(0, mover);
    b.set(enc.encode(uci), 1);
    dv.setUint16(9, probs.length, true);
    dv.setFloat64(11, outlook, true);
    dv.setUint16(19, sims, true);
    // f16 bit patterns (hand-computed, not from the decoder):
    // 0.25=0x3400 0.5=0x3800 0.75=0x3A00
    probs.forEach((bits, k) => dv.setUint16(21 + 2 * k, bits, true));
    return b;
  }
  const p0 = ply(0, "e2e4", [0x3400, 0x3a00], 50, -0.3);
  const p1 = ply(1, "e7e5", [0x3800], 600, 0.25);

  const shard = new Uint8Array(24 + game.length + p0.length + p1.length);
  const sdv = new DataView(shard.buffer);
  shard.set(enc.encode("SPSHARD1"), 0);
  sdv.setUint32(8, version, true);
  sdv.setUint32(12, 0, true);           // worker
  sdv.setBigUint64(16, 11n, true);      // unit_seed
  shard.set(game, 24);
  shard.set(p0, 24 + game.length);
  shard.set(p1, 24 + game.length + p0.length);
  return shard.buffer;                 // parseShard takes an ArrayBuffer
}

// ---- f16 decoder ---------------------------------------------------------
{
  const b = new ArrayBuffer(2);
  const dv = new DataView(b);
  const cases = [[0x3400, 0.25], [0x3800, 0.5], [0x3a00, 0.75],
                 [0x3c00, 1.0], [0x7bff, 65504],           // max normal
                 [0x0001, 5.960464477539063e-8],           // min subnormal
                 [0x03ff, 6.097555160522461e-5],           // max subnormal
                 [0x8000, -0.0]];
  for (const [bits, want] of cases) {
    dv.setUint16(0, bits, true);
    assert.ok(Math.abs(f16(dv, 0) - want) <= 1e-12 + 1e-9 * Math.abs(want),
              `f16(0x${bits.toString(16)}) = ${f16(dv, 0)}, want ${want}`);
  }
  dv.setUint16(0, 0x7c00, true);
  assert.equal(f16(dv, 0), Infinity);
}

// ---- v3 parse ------------------------------------------------------------
{
  const { version, games, truncated } = parseShard(shardBytes());
  assert.equal(version, 3);
  assert.equal(truncated, false);
  const [g] = games;
  assert.equal(g.gameIndex, 7);
  assert.equal(g.nPlies, 2);
  assert.equal(g.nSims, 600);
  assert.equal(g.zWhite, 1.0);
  assert.equal(g.startFen, "fen");
  const [p0, p1] = g.plies;
  assert.deepEqual([p0.mover, p0.playedUci, p0.nPi, p0.outlook, p0.sims],
                   [0, "e2e4", 2, -0.3, 50]);
  assert.deepEqual(p0.pi, [0.25, 0.75]);      // dense, engine order
  assert.equal("playedSlot" in p0, false);
  assert.deepEqual(p1.pi, [0.5]);
  assert.equal(p1.sims, 600);
}

// ---- legacy / unknown versions rejected ----------------------------------
for (const v of [1, 2, 4]) {
  let threw = null;
  try { parseShard(shardBytes(v)); } catch (e) { threw = e; }
  assert.ok(threw && /version/.test(threw.message), `v${v} must throw`);
}

// ---- a ply truncated mid-π drops the game, sets the flag -----------------
{
  const full = shardBytes();
  const cut = full.slice(0, full.byteLength - 1);  // mid last π prob
  const { games, truncated } = parseShard(cut);
  assert.equal(truncated, true);
  assert.equal(games.length, 0);
}

console.log("shard_node: all ok");
