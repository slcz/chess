/*
 * shard.js — SPSHARD1 self-play shard reader, the JS port of the byte
 * format documented in engine/include/chess_sp_dump.h (the Python twin
 * is chess_ai/selfplay/shard.py). v3 only — legacy v1/v2 shards must be
 * converted with scripts/spbin_to_v3.py first. Little-endian, packed:
 *
 *   shard: char[8] "SPSHARD1", u32 version (3), u32 worker, u64
 *          unit_seed, then finished games back to back
 *   game:  u32 index, u32 n_plies, u8 end_kind, i8 resigner, u32 n_sims,
 *          f64 z_white, char[96] start_fen
 *   ply:   u8 mover, char[8] played_uci, u16 n_pi, f64 outlook (root
 *          value, MOVER's view), u16 sims, n_pi × f16 prob
 *
 * π is dense over the legal moves in ENGINE MOVEGEN order (n_pi == the
 * legal count): the k-th prob belongs to the k-th legal move of the
 * replayed position — main.js pairs it with the worker reply's legal
 * list, which the same engine produces in the same order.
 *
 * A run killed mid-write leaves a truncated LAST game: the reader
 * detects the short read and drops it. Per-ply FENs are not stored —
 * positions are replayed from start_fen + played_uci through the engine
 * (the worker does that part).
 */

/** IEEE f16 bits → JS number. Hand-rolled (subnormal-correct): older
 *  engines lack DataView.getFloat16. */
export function f16(dv, at) {
  const h = dv.getUint16(at, true);
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) return sign * man * 5.960464477539063e-8;  // 2^-24
  if (exp === 31) return man ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + man / 1024);
}

export function parseShard(buf) {
  const dv = new DataView(buf);
  const dec = new TextDecoder();
  if (dec.decode(new Uint8Array(buf, 0, 8)) !== "SPSHARD1")
    throw new Error("not an SPSHARD1 file");
  const version = dv.getUint32(8, true);
  if (version !== 3)
    throw new Error(`unsupported shard version ${version} `
                    + `(convert v1/v2 with scripts/spbin_to_v3.py)`);
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
    if (!need(22 + 96 + nPlies * 21)) break;   // minimum for a zero-π ply
    const game = {
      gameIndex, nPlies,
      endKind: dv.getUint8(off + 8),
      resigner: dv.getInt8(off + 9),
      nSims: dv.getUint32(off + 10, true),
      zWhite: dv.getFloat64(off + 14, true),
      startFen: str(off + 22, 96),
      plies: [],
    };
    let p = off + 22 + 96;
    for (let i = 0; i < nPlies; i++) {
      const mover = dv.getUint8(p);
      const playedUci = str(p + 1, 8);
      const nPi = dv.getUint16(p + 9, true);
      const q = p + 21;
      if (q + 2 * nPi > buf.byteLength) { truncated = true; break; }
      const outlook = dv.getFloat64(p + 11, true);
      const sims = dv.getUint16(p + 19, true);
      const pi = new Array(nPi);
      for (let k = 0; k < nPi; k++)
        pi[k] = f16(dv, q + 2 * k);
      game.plies.push({ mover, playedUci, nPi, outlook, sims, pi });
      p = q + 2 * nPi;
    }
    if (truncated) break;
    games.push(game);
    off = p;
  }
  return { version, games, truncated };
}

/** End-kind names (SP_END_* in chess_sp_dump.h). */
export const END_NAMES = ["playing", "natural end", "resignation", "adjudicated"];
