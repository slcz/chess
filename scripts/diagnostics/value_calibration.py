#!/usr/bin/env python3
"""
probe_labels/value_calibration.py — TEMPORARY diagnostic (not pipeline code).

Question: is the self-play fine-tune damaging the VALUE head (label
distribution drift) or the POLICY head? The repo's runs/eval/value_drift.jsonl
already shows the value head's magnitude exploding after fine-tunes
(seed mean|v| 0.12 with 0.12% of |v|>0.95, fine-tunes 0.35-0.45 with 4-7.5%).
Magnitude growth is only a symptom; what decides whether it is damage is
CALIBRATION: does v=+0.9 actually mean a +0.9 result?

For every network this measures, on the SAME self-play positions:
  * value accuracy (argmax WDL vs the game outcome, mover's view)
  * mean |v| and the confidently-wrong rate (|v|>=0.9 and the opposite result)
  * a calibration table: mean v vs mean outcome per |v| bucket
  * ECE (expected calibration error) on the scalar mover-view value
  * base rate: the mean outcome is the unconditional outcome in this dataset,
    so a "predict the base rate" model is the calibration floor
  * policy drift: KL(search pi || net prior) and the net prior's probability
    mass on the move the search actually chose (low = policy is being dragged)

Usage:
  .venv/bin/python probe_labels/value_calibration.py <shard_dir> [ckpt ...]
Default networks: the 42k supervised base, the two PCR fine-tunes, and the
rope-25k net that generated the local shards.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_ai.engine import Board                                    # noqa: E402
from chess_ai.net.model import ChessTransformer, NetConfig            # noqa: E402
from chess_ai.net.features import legal_move_indices, policy_mask, uci_to_index  # noqa: E402
from chess_ai.selfplay.shard import read_unit                         # noqa: E402

BATCH = 256
STRIDE = 4          # sample every Nth ply (speed; positions are correlated)


def load_net(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(payload["net_config"])
    cfg.setdefault("pos_scheme", "shaw")
    model = ChessTransformer(NetConfig(**cfg))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload.get("steps")


def build_positions(unit: Path, stride: int):
    """Every stride-th ply of every game.

    NOTE: the position is captured as a FEN string, not as a live Board.
    A Board is a handle into mutable C state, so keeping references and
    running the net later would evaluate every sample at the *final*
    position of its game (the classic aliasing trap). The FEN is the
    immutable snapshot.
    """
    out = []
    for g in read_unit(unit):
        board = Board(g.start_fen)
        for i, p in enumerate(g.plies):
            side = board.side_to_move
            if len(p.played_uci) < 4:            # legacy resign ply
                continue
            if i % stride == 0:
                z_white = float(g.z_white)
                out.append({
                    "fen": board.fen,
                    "side": side,
                    "pi": {s: q for s, q in p.pi},
                    "played_slot": uci_to_index(p.played_uci, side),
                    "outlook": p.outlook,
                    "z_mover": z_white if side == 0 else -z_white,
                    "sims": p.sims,
                })
            board.push(p.played_uci)
    return out


def run_net(model, positions, device):
    vals, outs, priors_on_played, kl = [], [], [], []
    for k in range(0, len(positions), BATCH):
        chunk = positions[k:k + BATCH]
        boards = [Board(c["fen"]) for c in chunk]
        toks = np.stack([b.encode() for b in boards])
        masks = np.stack([policy_mask(legal_move_indices(b)) for b in boards])
        with torch.no_grad():
            t = torch.from_numpy(toks).to(device)
            m = torch.from_numpy(masks).to(device)
            o = model(t, m, m)
            w = F.softmax(o["wdl"].float(), -1).cpu().numpy()
            logp = o["policy_logp"].float().cpu().numpy()
        vals.append(w[:, 0] - w[:, 2])
        for j, c in enumerate(chunk):
            outs.append(c["z_mover"])
            slots = list(c["pi"].keys())
            q = np.array([c["pi"][s] for s in slots], dtype=np.float64)
            q /= q.sum()
            lp = logp[j][slots]
            lq = np.log(np.clip(q, 1e-12, None))
            if not np.isfinite(lp).all():
                # would mean the search's pi references a slot the trainer's
                # legal mask calls illegal — a real framing bug, and also how
                # the aliasing trap above showed up. Do not average -1e38
                # into a "KL"; surface it instead.
                raise AssertionError(
                    f"non-finite net logp at searched slots: fen={c['fen']} "
                    f"slots={slots}")
            kl.append(float((q * (lq - lp)).sum()))         # KL(q || p_net)
            priors_on_played.append(float(np.exp(logp[j][c["played_slot"]])))
    return (np.concatenate(vals), np.array(outs),
            np.array(priors_on_played), np.array(kl))


def report(name, steps, v, z, prior_played, kl, bins) -> dict:
    acc = float(((v > 0).astype(int) == (z > 0).astype(int)).mean()
                if (z != 0).any() else np.nan)
    confident = np.abs(v) >= 0.9
    wrong = confident & (np.sign(v) != np.sign(z)) & (z != 0)
    rows = []
    for lo, hi in bins:
        sel = (np.abs(v) >= lo) & (np.abs(v) < hi) & (np.sign(v) != 0)
        if sel.sum() < 5:
            rows.append((lo, hi, sel.sum(), np.nan, np.nan))
            continue
        vv = v[sel] * np.sign(v[sel])          # magnitude of the claim
        zz = z[sel] * np.sign(v[sel])          # result in the claimed direction
        rows.append((lo, hi, int(sel.sum()), float(vv.mean()), float(zz.mean())))
    ece = 0.0
    for lo, hi in bins:
        sel = (np.abs(v) >= lo) & (np.abs(v) < hi)
        if sel.sum() < 5:
            continue
        conf = float(np.abs(v[sel]).mean())
        act = float((z[sel] * np.sign(v[sel]) > 0).mean())
        ece += sel.mean() * abs(conf - act)
    print(f"\n  {name}  (steps={steps})")
    print(f"    mean |v| = {np.abs(v).mean():.3f}   |v|>=0.9 share = "
          f"{100 * confident.mean():.2f}%   decisiveness acc = {acc:.3f}")
    print(f"    confidently wrong (|v|>=0.9, opposite decisive result) = "
          f"{int(wrong.sum())}/{int(confident.sum())}"
          + (f" = {100 * wrong.sum() / confident.sum():.1f}%" if confident.sum() else ""))
    print(f"    ECE = {ece:.3f}    policy: KL(search||net) = {kl.mean():.3f}, "
          f"mean prior on the searched move = {prior_played.mean():.3f}")
    print(f"    calibration (claim magnitude vs realised result in that direction):")
    for lo, hi, n, vv, zz in rows:
        flag = ""
        if not np.isnan(vv) and abs(vv - zz) > 0.25:
            flag = "   <-- over/under-confident"
        print(f"      |v| in [{lo:.1f},{hi:.1f}): n={n:5d}  mean|v|={vv:.3f}  "
              f"realised={zz:.3f}{flag}")
    return {"acc": acc, "ece": ece, "conf_wrong": int(wrong.sum()),
            "n_conf": int(confident.sum()), "kl": float(kl.mean()),
            "prior_played": float(prior_played.mean())}


def main() -> int:
    unit = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "probe_labels/cal"
    ckpts = [Path(p) for p in sys.argv[2:]] or [
        ROOT / "runs/gen1.pt",
        ROOT / "runs/pcr-iter1-e1.pt",
        ROOT / "runs/pcr2-iter1-e2.pt",
        ROOT / "runs/policy_value_net_rope_25k.pt",
    ]
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pos = build_positions(unit, STRIDE)
    if not pos:
        print(f"no positions in {unit}")
        return 2
    z_all = np.array([p["z_mover"] for p in pos])
    decisive = (z_all != 0)
    print(f"value_calibration on {unit} — {len(pos)} positions "
          f"({100 * decisive.mean():.1f}% decisive, draw share "
          f"{100 * (~decisive).mean():.1f}%), device {device.type}")
    print(f"  base rate: mean outcome (mover view) = {z_all.mean():+.3f}; "
          f"a 'predict the base rate' model has zero ECE by construction")
    bins = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
    summary = {}
    for ck in ckpts:
        if not ck.exists():
            print(f"  (missing {ck})")
            continue
        model, steps = load_net(ck, device)
        v, z, pp, kl = run_net(model, pos, device)
        summary[ck.name] = report(ck.name, steps, v, z, pp, kl, bins)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    print("\n=== summary (higher decisiveness acc / lower ECE / lower KL is better) ===")
    print(f"  {'net':34s} {'acc':>6s} {'ECE':>6s} {'conf-wrong':>11s} {'KL':>6s} {'prior@played':>13s}")
    for k, s in summary.items():
        print(f"  {k:34s} {s['acc']:6.3f} {s['ece']:6.3f} "
              f"{s['conf_wrong']:5d}/{s['n_conf']:<5d} {s['kl']:6.3f} {s['prior_played']:13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
