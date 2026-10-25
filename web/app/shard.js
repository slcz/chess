/*
 * shard.js — SPSHARD1 self-play shard reader, the JS port of the byte
 * format documented in engine/include/chess_sp_dump.h (the Python twin
 * is chess_ai/selfplay/shard.py). Little-endian, packed:
 *
 *   shard: char[8] "SPSHARD1", u32 version (2 = per-ply sims; 3 = dense
 *          f16 π, see below), u32 worker, u64 unit_seed, then finished
 *          games back to back
 *   game:  u32 index, u32 n_plies, u8 end_kind, i8 resigner, u32 n_sims,
 *          f64 z_white, char[96] start_fen        (identical in v1/v2/v3)
 *   ply:   u8 mover, i16 played_slot, char[8] played_uci, u16 n_pi,
 *          f64 outlook (root value, MOVER's view), [v2: u16 sims],
 *          n_pi × (u16 slot, f32 prob)                     (v1/v2)
 *   ply v3:u8 mover, char[8] played_uci, u16 n_legal, f64 outlook,
 *          u16 sims, n_legal × f16 prob
 *
 * v3 (loop1-generation shards) drops played_slot and the (slot, prob)
 * pairs: π is a dense f16 vector over ALL legal moves in the ENGINE'S
 * MOVE-GENERATION ORDER, so the slot table costs nothing and the reader
 * recovers (uci, prob) by replaying the position and zipping probs with
 * its own legal-move list — the worker does that in wtLoad (the movegen
 * is deterministic, so the zip is exact; verified: the played move's
 * prob ranks first under gen order, ~14th under slot order). This file
 * only parses bytes; `pi` stays null for v3 plies.
 *
 * A run killed mid-write leaves a truncated LAST game: the reader
 * detects the short read and drops it. Per-ply FENs are not stored —
 * positions are replayed from start_fen + played_uci through the engine
 * (the worker does that part).
 */
export function parseShard(buf) {
  const dv = new DataView(buf);
  const dec = new TextDecoder();
  if (dec.decode(new Uint8Array(buf, 0, 8)) !== "SPSHARD1")
    throw new Error("not an SPSHARD1 file");
  const version = dv.getUint32(8, true);
  if (version !== 1 && version !== 2 && version !== 3)
    throw new Error(`unsupported shard version ${version}`);
  let off = 24;
  const games = [];
  let truncated = false;
  const str = (at, len) => {
    const bytes = new Uint8Array(buf, at, len);
    let end = bytes.indexOf(0);
    if (end < 0) end = len;
    return dec.decode(bytes.subarray(0, end));
  };
  const need = (n) => {
    if (off + n > buf.byteLength) { truncated = true; return false; }
    return true;
  };

  while (!truncated && off + 10 <= buf.byteLength) {
    const gameIndex = dv.getUint32(off, true);
    const nPlies = dv.getUint32(off + 4, true);
    if (!need(22 + 96 + plyBytes(nPlies, version))) break;
    const game = {
      gameIndex, nPlies,
      endKind: dv.getUint8(off + 8),
      resigner: dv.getInt8(off + 9),
      nSims: dv.getUint32(off + 10, true),
      zWhite: dv.getFloat64(off + 14, true),
      startFen: str(off + 22, 96),
      plies: [],
    };
    if (version >= 3) {
      let p = off + 118;
      for (let i = 0; i < nPlies; i++) {
        if (!need(p - off + 21)) break;
        const nLegal = dv.getUint16(p + 9, true);
        if (!need(p - off + 21 + 2 * nLegal)) break;
        const probs = new Float32Array(nLegal);
        for (let k = 0; k < nLegal; k++)
          probs[k] = f16(dv.getUint16(p + 21 + 2 * k, true));
        game.plies.push({
          mover: dv.getUint8(p),
          playedSlot: -1,
          playedUci: str(p + 1, 8),
          nLegal, probs,
          outlook: dv.getFloat64(p + 11, true),
          sims: dv.getUint16(p + 19, true),
          pi: null,          // (uci, prob) pairs come from the worker's replay
        });
        p += 21 + 2 * nLegal;
      }
      if (!truncated && game.plies.length === nPlies) games.push(game);
      off = p;
      continue;
    }
    let p = off + 22 + 96;
    for (let i = 0; i < nPlies; i++) {
      const mover = dv.getUint8(p);
      const playedSlot = dv.getInt16(p + 1, true);
      const playedUci = str(p + 3, 8);
      const nPi = dv.getUint16(p + 11, true);
      const outlook = dv.getFloat64(p + 13, true);
      let q = p + 21;
      let sims = game.nSims;
      if (version >= 2) {
        sims = dv.getUint16(q, true);
        q += 2;
      }
      const pi = [];
      for (let k = 0; k < nPi; k++)
        pi.push([dv.getUint16(q + 6 * k, true),
                 dv.getFloat32(q + 6 * k + 2, true)]);
      game.plies.push({ mover, playedSlot, playedUci, nPi, outlook, sims, pi });
      p = q + 6 * nPi;
    }
    games.push(game);
    off = p;
  }
  return { version, games, truncated };
}

function plyBytes(nPlies, version) {
  // v1/v2 per ply: 1+2+8+2+8 (+2 in v2) + 6·n_pi; v3: 21 + 2·n_legal —
  // both self-delimiting, so this is only the coarse need() check and
  // the exact boundary happens per ply below.
  return nPlies * (21 + (version >= 2 ? 2 : 0));
}

/** IEEE half → JS number (the v3 π encoding). */
function f16(u) {
  const s = (u >> 15) & 1, e = (u >> 10) & 0x1f, m = u & 0x3ff;
  let v;
  if (e === 0) v = m * 2 ** -24;
  else if (e === 31) v = m ? NaN : Infinity;
  else v = (1 + m / 1024) * 2 ** (e - 15);
  return s ? -v : v;
}

/** End-kind names (SP_END_* in chess_sp_dump.h). */
export const END_NAMES = ["playing", "natural end", "resignation", "adjudicated"];
