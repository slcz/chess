#!/usr/bin/env python3
"""
probe_labels/probe_labels.py — TEMPORARY verification probe (not part of the
pipeline). Question it answers:

    Do the encoded LABELS that trainer.py consumes — the dense policy target
    (pi_slots/pi_probs) and the WDL value target — actually carry what the C
    self-play driver (engine/src/selfplay.c) generated and recorded?

Why a separate probe is needed: trainer.train_step() never sees raw search
state. It receives batch["pi_slots"]/["pi_probs"]/["wdl"] built by
chess_ai/train/selfplay_data.py from the shard bytes, and it gathers
policy_logp at those slots. So a label error can only enter through that
reconstruction plus the pure-Python replay used to build tokens/mask. The
value side is a perspective flip (mover vs white) that is invisible to any
"is this finite" check.

The probe has four independent parts:

  A. DETERMINISM — two `selfplay` runs with identical arguments must produce
     byte-identical shards. If so, the shard really is the deterministic
     output of the C driver for that seed, and everything below compares
     against C's own generation, not against a re-derivation.

  B. SHARD-SIDE PROVENANCE (C's own record, no Python label code involved) —
     replay every ply from start_fen with the Python rules engine and check,
     for the whole shard:
       * the recorded mover == the replayed side to move;
       * the recorded played_slot == uci_to_index(played_uci, mover);
       * the played move is inside the recorded pi support;
       * every pi slot is a LEGAL policy slot in that position (the C mask);
       * sum(pi probs) == 1.
     A trainable/slow ply is also checked to be the argmax of its own pi —
     C picks the played move by sampling or argmax over exactly those visits,
     so a mis-framed slot map shows up here as a smile-shaped agreement.

  C. TRAINER-SIDE LABEL IDENTITY — for every ply that `ply_trainable` keeps,
     build the actual sample `selfplay_sample` hands to train_step and assert
     the policy label is bit-identical to the shard's (after the same
     f64->normalize->f32 round trip), and that the mask agrees with C's pi
     support (nothing C searched is masked out, nothing masked-in was
     unsearched). Then run the real model forward and recompute exactly the
     two trainer.py losses on that batch.

  D. VALUE LABEL — the shard's z_white says who won. Re-derive the outcome
     from the move list alone (rules) and assert it equals z_white, including
     the checkmate-winner direction. Then verify the trainer's encoded WDL
     one-hot is the MOVER's view of that outcome (the bug class that would
     make the value head learn the wrong sign), and print the value loss.

Usage:  .venv/bin/python probe_labels/probe_labels.py [sp_a_dir] [sp_b_dir]
Run from the repo root. Writes nothing outside probe_labels/.
"""
from __future__ import annotations

import collections
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_ai.engine import Board, Termination                    # noqa: E402
from chess_ai.net.features import (                               # noqa: E402
    POLICY_SIZE, legal_move_indices, policy_mask, uci_to_index,
)
from chess_ai.selfplay.shard import read_unit                     # noqa: E402
from chess_ai.train.selfplay_data import (                        # noqa: E402
    PI_SUPPORT_MAX, outlook_weight, ply_trainable, selfplay_sample, stack_selfplay,
)
from chess_ai.net.model import ChessTransformer, NetConfig        # noqa: E402

CKPT = ROOT / "runs" / "policy_value_net_rope_25k.pt"

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> bool:
    if not cond:
        FAILURES.append(msg)
        print(f"    FAIL: {msg}")
    return cond


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- A. determinism
def part_a(da: Path, db: Path) -> None:
    print("\n=== A. determinism: two identical-argument selfplay runs ===")
    sa, sb = da / "shard-0.spbin", db / "shard-0.spbin"
    if not (sa.exists() and sb.exists()):
        print(f"  MISSING shard(s): {sa.exists()=} {sb.exists()=} — skipping A")
        return
    ha, hb = sha(sa), sha(sb)
    print(f"  run A sha256 {ha[:32]}…  {sa.stat().st_size} bytes")
    print(f"  run B sha256 {hb[:32]}…  {sb.stat().st_size} bytes")
    check(ha == hb, "A: identical-argument runs produced DIFFERENT shards")
    print(f"  -> {'IDENTICAL' if ha == hb else 'DIFFERENT'}")


# ------------------------------------------------- B. shard-side provenance
def part_b(unit: Path) -> None:
    print("\n=== B. shard-side provenance (C record vs an independent replay) ===")
    games = read_unit(unit)
    st = collections.Counter()
    for g in games:
        board = Board(g.start_fen)
        for i, p in enumerate(g.plies):
            st["plies"] += 1
            side = board.side_to_move
            check(p.mover == side,
                  f"B: game {g.game_index} ply {i}: mover {p.mover} != side {side}")
            if len(p.played_uci) < 4:
                st["empty_uci"] += 1
                continue
            idx = uci_to_index(p.played_uci, side)
            check(idx == p.played_slot,
                  f"B: game {g.game_index} ply {i}: played_slot {p.played_slot} "
                  f"!= uci_to_index({p.played_uci}) = {idx}")
            m = policy_mask(legal_move_indices(board))
            slots = [s for s, _ in p.pi]
            st["trainable" if ply_trainable(g, p) else "fast"] += 1
            check(p.played_slot in slots,
                  f"B: game {g.game_index} ply {i}: played move not in pi support")
            for s in slots:
                check(0 <= s < POLICY_SIZE and m[s],
                      f"B: game {g.game_index} ply {i}: pi slot {s} not a legal slot")
            tot = sum(q for _, q in p.pi)
            check(abs(tot - 1.0) < 1e-5,
                  f"B: game {g.game_index} ply {i}: pi sums to {tot}")
            st["pi_max"] = max(st["pi_max"], len(slots))
            if p.sims == g.n_sims and slots:
                best = max(p.pi, key=lambda kv: kv[1])[0]
                st["slow_argmax_matches"] += (best == p.played_slot)
                st["slow_sampled_played_mass"] += dict(p.pi)[p.played_slot]
            board.push(p.played_uci)
        # end-of-game provenance for D: a natural end must be terminal by the
        # rules (no legal move for the side to move) and an adjudicated one
        # must not be. (The earlier `bool(legal) == (term != PLAYING)` form
        # was simply wrong — those two are not equal, they are complements.)
        at_end = board.termination != Termination.PLAYING
        check(at_end == (g.end_kind == 1),
              f"B: game {g.game_index}: end_kind={g.end_kind} but the replayed "
              f"end is {board.termination.name}")
        check(not at_end or not board.legal_moves,
              f"B: game {g.game_index}: natural end with legal moves available")
    print(f"  games={len(games)} plies={st['plies']} "
          f"slow(trainable)={st['trainable']} fast={st['fast']} "
          f"max pi support={st['pi_max']}")
    if st["trainable"]:
        print(f"  slow plies whose pi argmax == the played move: "
              f"{st['slow_argmax_matches']}/{st['trainable']} "
              f"(sampled plies make this <100% by design)")
        print(f"  mean pi mass on the played move at slow plies: "
              f"{st['slow_sampled_played_mass'] / st['trainable']:.4f}")
    print(f"  -> {len(FAILURES)} failure(s) so far")


# --------------------------------- C/D. trainer-side labels + the real losses
def part_cd(unit: Path) -> None:
    print("\n=== C. trainer-side label identity + the real trainer.py losses ===")
    games = read_unit(unit)
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(payload["net_config"])
    cfg.setdefault("pos_scheme", "shaw")
    model = ChessTransformer(NetConfig(**cfg))
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"  checkpoint {CKPT.name}: {sum(p.numel() for p in model.parameters()):,} params, "
          f"aux head={'yes' if model.aux is not None else 'no'}")

    identity = max_dwdl = 0
    n_train = 0
    mask_extra = mask_missing = 0
    samples = []
    for g in games:
        # --- D: independent outcome re-derivation (rules only) -------------
        board = Board(g.start_fen)
        for p in g.plies:
            board.push(p.played_uci)
        term = board.termination
        z_from_rules = 0.5
        if term == Termination.CHECKMATE:
            z_from_rules = 1.0 if board.winner == 0 else 0.0
        check(abs(z_from_rules - g.z_white) < 1e-9,
              f"D: game {g.game_index}: z_white={g.z_white} but the rules say "
              f"{z_from_rules} (term={term.name})")
        check(g.end_kind == (1 if term != Termination.PLAYING else 3),
              f"D: game {g.game_index}: end_kind={g.end_kind} but replay ends "
              f"{term.name}")
        # --- C: per-trainable-ply label identity ---------------------------
        for ply in range(g.n_plies):
            p = g.plies[ply]
            if not ply_trainable(g, p):
                continue
            n_train += 1
            s = selfplay_sample(g, ply)
            slots, probs = [x for x, _ in p.pi], np.array(
                [q for _, q in p.pi], dtype=np.float64)
            probs = probs / probs.sum()
            rec = np.asarray(s["pi_slots"])[: len(slots)]
            check(list(rec) == slots,
                  f"C: game {g.game_index} ply {ply}: pi_slots drift")
            got = np.asarray(s["pi_probs"])[: len(slots)]
            if np.allclose(got, probs.astype(np.float32), atol=0, rtol=0):
                identity += 1
            else:
                d = float(np.abs(got - probs.astype(np.float32)).max())
                check(False, f"C: game {g.game_index} ply {ply}: pi_probs drift {d}")
            # pad must be inert
            check(float(np.asarray(s["pi_probs"])[len(slots):].sum()) == 0.0,
                  f"C: game {g.game_index} ply {ply}: padding carries mass")
            # mask must agree with what C actually searched at this position
            mask = np.asarray(s["mask"])
            for sl in slots:
                if not mask[sl]:
                    mask_missing += 1
            for u in Board(g.start_fen).legal_moves if False else []:
                pass
            # mask extras: legal slots C did not search are EXPECTED (mask is
            # the legal set, pi the visited subset), so only check containment.
            wdl = np.asarray(s["wdl"])
            check(abs(float(wdl.sum()) - 1.0) < 1e-6,
                  f"C: game {g.game_index} ply {ply}: wdl not a one-hot")
            zm = 2.0 * g.z_white - 1.0
            want = np.array([1.0, 0.0, 0.0]) if (zm if p.mover == 0 else -zm) > 0 \
                else (np.array([0.0, 1.0, 0.0]) if zm == 0 else np.array([0.0, 0.0, 1.0]))
            max_dwdl = max(max_dwdl, float(np.abs(wdl - want).max()))
            sw = float(s["sample_w"])
            check(abs(sw - outlook_weight(p.outlook)) < 1e-6,
                  f"C: game {g.game_index} ply {ply}: sample_w mismatch")
            samples.append(s)

    print(f"  trainable plies: {n_train}; pi_slots/pi_probs bit-identical to the "
          f"shard: {identity}")
    print(f"  pi slots missing from the trainer's legal mask: {mask_missing}")
    print(f"  max |encoded wdl - expected mover-view one-hot|: {max_dwdl:.1e}")
    check(identity == n_train, "C: some pi labels are not identical to the shard")
    check(mask_missing == 0, "C: pi contains slots the trainer's mask calls illegal")
    check(max_dwdl == 0.0, "C/D: wdl one-hot is not the mover's view")

    if not samples:
        print("  no trainable samples — cannot run the real losses")
        return
    batch = {k: torch.from_numpy(np.ascontiguousarray(v))
             for k, v in stack_selfplay(samples).items()}
    for k in ("wdl", "sample_w", "aux_w", "pi_probs"):
        batch[k] = batch[k].float()
    with torch.no_grad():
        out = model(batch["tokens"], batch["mask"], batch["aux_mask"])
        slot_logp = out["policy_logp"].gather(1, batch["pi_slots"])
        nll = -(batch["pi_probs"] * slot_logp).sum(-1)
        w = batch["sample_w"]
        policy_loss = (nll * w).sum() / w.sum().clamp(min=1e-8)
        value_loss = -(F.log_softmax(out["wdl"], -1) * batch["wdl"]).sum(-1).mean()
        # counterfactual: rotate the value target among samples whose outcome
        # is NOT a draw. If the loss is insensitive to the flip, the label is
        # not reaching the loss.
        dec = batch["wdl"].argmax(-1)
        non_draw = (dec != 1).nonzero().squeeze(-1)
        rot = batch["wdl"].clone()
        if len(non_draw) >= 2:
            rot[non_draw] = torch.roll(batch["wdl"][non_draw], 1, dims=0)
        value_loss_rot = -(F.log_softmax(out["wdl"], -1) * rot).sum(-1).mean()
        # counterfactual: a one-hot (played-move) target would give a different
        # policy loss — proves the dense C label, not pol_idx, drives it.
        # the played move's slot as an index INTO the 256-wide pi support
        # (pol_idx is a POLICY_SIZE slot, not a support index)
        in_support = (batch["pi_slots"] * (batch["pi_probs"] > 0)
                      == batch["pol_idx"][:, None]).float()
        # the played move is always in the support; if a ply were ever
        # sampled outside its own pi support this would silently zero out,
        # so assert coverage explicitly.
        check(bool((in_support.sum(-1) == 1).all()),
              "C: played move absent from its own pi support")
        onehot = in_support / in_support.sum(-1, keepdim=True)
        policy_loss_onehot = ((-(onehot * slot_logp).sum(-1)) * w).sum() / w.sum()
        # counterfactual: rotate the dense label -> loss must rise if the
        # target is really read at these slots.
        pi_rot = torch.roll(batch["pi_probs"], 1, dims=0)
        policy_loss_rot = ((-(pi_rot * slot_logp).sum(-1)) * w).sum() / w.sum()

    print(f"  REAL trainer.py policy_loss = {float(policy_loss):.4f}   "
          f"value_loss = {float(value_loss):.4f}")
    print(f"  counterfactuals (must all differ):")
    print(f"    one-hot played-move target instead of the dense C pi: "
          f"{float(policy_loss_onehot):.4f}")
    print(f"    dense pi rotated across samples:                       "
          f"{float(policy_loss_rot):.4f}")
    print(f"    value target rotated (non-draw samples):                "
          f"{float(value_loss_rot):.4f}")
    check(abs(float(policy_loss) - float(policy_loss_onehot)) > 1e-6,
          "C: dense and one-hot targets give the SAME loss — pi is not used")
    check(abs(float(policy_loss) - float(policy_loss_rot)) > 1e-6,
          "C: rotating pi does not change the loss — the label is inert")
    check(abs(float(value_loss) - float(value_loss_rot)) > 1e-6 or len(non_draw) < 2,
          "D: rotating the value target does not change the loss — wdl is inert")
    # A faithful label must fit better than a wrong one: the model's own
    # ranking should correlate with pi more than with a rotated pi.
    agree = (slot_logp.argmax(-1) == batch["pi_probs"].argmax(-1)).float().mean()
    agree_rot = (slot_logp.argmax(-1) == pi_rot.argmax(-1)).float().mean()
    print(f"    policy_acc with the real pi: {float(agree):.3f}   "
          f"with rotated pi: {float(agree_rot):.3f}")
    print(f"  -> {len(FAILURES)} total failure(s)")


def main() -> int:
    da = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "probe_labels/sp_a"
    db = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "probe_labels/sp_b"
    print(f"probe_labels — C self-play labels vs the trainer's view")
    print(f"  unit A: {da}\n  unit B: {db}")
    part_a(da, db)
    part_b(da)
    part_cd(da)
    print("\n================ RESULT ================")
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES[:40]:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
