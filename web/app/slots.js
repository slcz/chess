/*
 * slots.js — policy slot ↔ UCI mapping, the JS port of the move-space
 * half of chess_ai/net/features.py (single source of truth: the C
 * engine's chess_policy_slot). Pure arithmetic, no dependencies.
 *
 * Layout (D1): 4240 = 4096 from-to (mover-relative squares, from*64+to;
 * a pair crossing into the last rank implies "=Q") + 72 own underpromos
 * (from ∈ rel-rank-7, dirs +7/+8/+9 × N/B/R) + 72 enemy underpromos
 * (from ∈ rel-rank-2, dirs −7/−8/−9 — the aux-reply target region).
 * Mover-relative = the mover sits at the bottom: for Black the board
 * flips vertically (sq ^ 56).
 */
export const POLICY_SIZE = 4240;
export const FT_SIZE = 4096;

const FILES = "abcdefgh";
const PROMO_LETTERS = { n: 0, b: 1, r: 2 };
const PROMO_DIRS = [7, 8, 9];
const PROMO_DIRS_DOWN = [-7, -8, -9];
const PROMO_FROM_BASE = 48;
const PROMO_FROM_BASE_DOWN = 8;

const relSquare = (sq, side) => (side === 0 ? sq : sq ^ 56);

export function uciToIndex(uci, side) {
  const fromAbs = FILES.indexOf(uci[0]) + (+uci[1] - 1) * 8;
  const toAbs = FILES.indexOf(uci[2]) + (+uci[3] - 1) * 8;
  const frm = relSquare(fromAbs, side), to = relSquare(toAbs, side);
  if (uci.length === 5 && uci[4] !== "q") {
    const k = PROMO_LETTERS[uci[4]];
    const dir = to - frm;
    const up = PROMO_DIRS.indexOf(dir);
    if (up >= 0)
      return FT_SIZE + (frm - PROMO_FROM_BASE) * 9 + up * 3 + k;
    const down = PROMO_DIRS_DOWN.indexOf(dir);
    return FT_SIZE + 72 + (frm - PROMO_FROM_BASE_DOWN) * 9 + down * 3 + k;
  }
  return frm * 64 + to;
}

const uciSq = (relSq, side) => {
  const sq = relSquare(relSq, side);
  return FILES[sq % 8] + ((sq >> 3) + 1);   // rank digit BEFORE concat
};

export function indexToUci(idx, side) {
  if (idx < FT_SIZE) {
    const frm = idx >> 6, to = idx % 64;   // idx >>> 6 safe (< 4096)
    let uci = uciSq(frm, side) + uciSq(to, side);
    const upPromo = frm >= 48 && frm <= 55 && to >= 56;
    const downPromo = frm >= 8 && frm <= 15 && to <= 7;
    if (upPromo || downPromo) uci += "q";  // from-to region means "=Q"
    return uci;
  }
  let k, frm, dir;
  if (idx < FT_SIZE + 72) {
    k = idx - FT_SIZE;
    frm = PROMO_FROM_BASE + (k / 9 | 0);
    dir = PROMO_DIRS[(k / 3 | 0) % 3];
  } else {
    k = idx - FT_SIZE - 72;
    frm = PROMO_FROM_BASE_DOWN + (k / 9 | 0);
    dir = PROMO_DIRS_DOWN[(k / 3 | 0) % 3];
  }
  return uciSq(frm, side) + uciSq(frm + dir, side) + "nbr"[k % 3];
}
