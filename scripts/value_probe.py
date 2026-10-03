"""
scripts/value_probe.py — evaluation-quality probe: handcrafted positions with
THEORETICALLY UNAMBIGUOUS values (forced wins by bare material, draws by
insufficient material) plus a sample of the run's own self-play positions,
each scored by Stockfish (the referee), the supervised seed, and the
current trained net — raw value head only (v̂ = pW − pL, mover's view).

Built 2026-10-02 after three recipes' gates all opened 0.34–0.44 vs the
seed and slid: establishes what the nets actually BELIEVE about positions
where the truth is certain (esp. draw-blindness: after 99.7%-decisive
self-play labels, does the trained net still score drawn-by-material
positions as wins?).

    python -m scripts.value_probe --trained /tmp/v3_net_iter_0010.pt
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

from chess_ai.engine import Board
from chess_ai.net.features import policy_mask, uci_to_index
from chess_ai.selfplay.ladder import NetEvaluator
from chess_ai.selfplay.spengine import _read_shard

SF = "/opt/homebrew/bin/stockfish"
SF_DEPTH = 22

# (fen, label, expected) — expected from basic theory. All positions are
# legal by construction (kings never adjacent, no castling rights).
SUITE = [
    ("4k3/8/8/8/8/8/4K3/4Q3 w - - 0 1",   "K+Q vs K",       "win"),
    ("4k3/8/8/8/8/8/4K3/4R3 w - - 0 1",   "K+R vs K",       "win"),
    ("4k3/8/8/8/8/8/3K4/3R1R2 w - - 0 1", "K+2R vs K",      "win"),
    ("4k3/8/8/8/8/8/3PK3/4R3 w - - 0 1",  "K+R+P vs K",     "win"),
    ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
                                          "queen-up start", "win"),
    ("r1bqkbnr/1ppppppp/p7/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
                                          "rook-up start",  "adv"),
    ("4k3/8/8/8/8/8/4K3/4B3 w - - 0 1",   "K+B vs K",       "draw"),
    ("4k3/8/8/8/8/8/4K3/4N3 w - - 0 1",   "K+N vs K",       "draw"),
    ("4k3/8/8/8/8/8/3K4/3N1N2 w - - 0 1", "K+N+N vs K",     "draw"),
    ("4k3/8/8/8/8/8/4K3/3B1B2 w - - 0 1", "2B same color",  "draw"),
]


def net_value(ev: NetEvaluator, fen: str) -> float:
    """v̂ (pW − pL, mover's view) for one FEN via the project's encoder."""
    board = Board(fen)
    side = board.side_to_move
    idx = [uci_to_index(u, side) for u in board.legal_moves]
    x = torch.from_numpy(board.encode().reshape(1, 64, 117).astype(np.float32))
    m = torch.zeros(1, 4240, dtype=torch.bool)
    m[0, idx] = True
    x, m = x.to(ev._device), m.to(ev._device)
    with torch.no_grad():
        out = ev.model(x, m)
    w = torch.softmax(out["wdl"].float(), dim=-1)[0].cpu()
    return float(w[0] - w[2])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.value_probe")
    ap.add_argument("--trained", required=True, help="current run snapshot")
    ap.add_argument("--seed", default="runs/policy_value_net_rope_25k.pt")
    ap.add_argument("--shards", default="/tmp/n64_audit/n64_eq9/units/iter-0001",
                    help="self-play shard dir for the real-position sample")
    ap.add_argument("--n-shard", type=int, default=40)
    ap.add_argument("--depth", type=int, default=SF_DEPTH)
    ap.add_argument("--jsonl", default="runs/eval/value_probe.jsonl")
    a = ap.parse_args(argv)

    engine = chess.engine.SimpleEngine.popen_uci(SF)

    def sf_cp(board: chess.Board) -> float:
        info = engine.analyse(board, chess.engine.Limit(depth=a.depth))
        return float(info["score"].relative.score(mate_score=10000))

    seed_ev = NetEvaluator(a.seed, device="mps")
    train_ev = NetEvaluator(a.trained, device="mps")

    print(f"=== handcrafted suite (SF depth {a.depth}) ===")
    print(f"{'position':<18} {'expect':<7} {'SF cp':>7} "
          f"{'seed v':>7} {'train v':>8}  verdict")
    rows, seed_ok, train_ok = [], 0, 0
    for fen, label, expected in SUITE:
        board = chess.Board(fen)
        cp = sf_cp(board)
        sv = net_value(seed_ev, fen)
        tv = net_value(train_ev, fen)
        want = {"win": +1, "adv": +1, "draw": 0}[expected]
        ok_s = (np.sign(sv) == want) if want else abs(sv) < 0.5
        ok_t = (np.sign(tv) == want) if want else abs(tv) < 0.5
        seed_ok += bool(ok_s)
        train_ok += bool(ok_t)
        print(f"{label:<18} {expected:<7} {cp:>7.1f} {sv:>7.3f} {tv:>8.3f}"
              f"  seed{'OK' if ok_s else 'X'} train{'OK' if ok_t else 'X'}")
        rows.append({"label": label, "fen": fen, "expected": expected,
                     "sf_cp": cp, "seed_v": round(sv, 4),
                     "train_v": round(tv, 4)})

    # real positions from the run's own self-play (training distribution)
    print(f"\n=== {a.n_shard} self-play positions (net outlook vs SF) ===")
    shards = sorted(Path(a.shards).glob("shard-w*.spbin"))
    recs = [r for w in shards for r in _read_shard(w, seed=1)]
    sample, i = [], 0
    while len(sample) < a.n_shard and recs:
        rec = recs[i % len(recs)]
        plies = [m for m in rec.moves if m.played_uci]
        if plies:
            mv = plies[(i * 7) % len(plies)]
            sample.append((mv.fen_before, mv.outlook))
        i += 13
    pairs = []
    for fen, outlook in sample:
        cp = sf_cp(chess.Board(fen)) / 100.0     # pawns, mover POV
        pairs.append((outlook, cp, net_value(train_ev, fen)))
    out_sign = sum(1 for o, c, _ in pairs if (o >= 0) == (c >= 0) and c != 0)
    t_sign = sum(1 for _, c, t in pairs if (t >= 0) == (c >= 0) and c != 0)
    overconfident = sum(1 for _, c, t in pairs
                        if abs(t) > 0.9 and abs(c) < 0.5)
    cor = statistics.correlation([c for _, c, _ in pairs],
                                 [t for _, c, t in pairs])
    print(f"outlook sign vs SF sign:   {out_sign}/{len(pairs)}")
    print(f"train v sign vs SF sign:   {t_sign}/{len(pairs)}")
    print(f"train |v|>0.9 while SF even (|cp|<0.5): {overconfident}  "
          f"<- draw-blindness smoking gun")
    print(f"pearson(SF pawns, train v): {cor:+.3f}")

    summary = {"trained": a.trained, "sf_depth": a.depth, "suite": rows,
               "suite_seed_ok": seed_ok, "suite_train_ok": train_ok,
               "suite_n": len(SUITE), "shard_n": len(pairs),
               "shard_outlook_sign_match": out_sign,
               "shard_train_sign_match": t_sign,
               "shard_pearson": round(cor, 4),
               "train_winning_when_sf_even": overconfident}
    out = Path(a.jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            **summary}) + "\n")
    engine.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
