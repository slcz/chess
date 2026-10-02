"""
scripts/az_eval.py — non-parallel, AlphaGo-Zero-style evaluation (DEBUG).

Pure-Python PUCT MCTS over the project's C bitboard engine and our net:
one NN call per new leaf (no batching, no Gumbel, no parallel engine, no
C search). Deliberately simple — the point is a clean-room second opinion
on what the trained nets believe, independent of the Gumbel machinery:

  select    argmax_a [ Q(s,a) + c_puct · P(s,a) · sqrt(ΣN)/(1+N(s,a)) ]
  expand    leaf → policy priors (masked softmax) + value v = pW − pL
  backup    v sign-flipped up the path; terminal nodes scored ±1/0 by rules
  play      argmax visits (first `sample-plies` plies sample ∝ visits for
            opening diversity, per-game RNG — the R5 protocol's AZ twin)

Modes:
  match    PUCT-vs-PUCT match between two checkpoints (default: v4 r10 vs
           the supervised seed)
  analyse  inspect ONE position: visit distribution, Q, priors, final value

    python -m scripts.az_eval match --a runs/cvisit15/net-iter-0010.pt \
        --b runs/policy_value_net_rope_25k.pt --games 4 --sims 128
    python -m scripts.az_eval analyse --ckpt <ckpt> --fen "FEN" --sims 400
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from chess_ai.engine import Board, Termination
from chess_ai.net.features import uci_to_index
from chess_ai.selfplay.ladder import NetEvaluator


class Node:
    __slots__ = ("parent", "move", "prior", "visits", "value_sum",
                 "children", "terminal_v")

    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move              # uci that led here
        self.prior = prior            # P(s,a) from the parent's policy
        self.visits = 0
        self.value_sum = 0.0          # Σ v, THIS node's mover's view
        self.children = {}            # uci → Node
        self.terminal_v = None        # set when the position is terminal

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _priors_and_value(ev: NetEvaluator, board: Board):
    """Masked policy priors + raw value for the board's mover (one NN call)."""
    side = board.side_to_move
    legal = list(board.legal_moves)
    slots = [uci_to_index(u, side) for u in legal]
    logp, wdl = ev.evaluate(board.encode(), slots, side)
    total = sum(math.exp(logp[s]) for s in slots)
    priors = {u: math.exp(logp[s]) / total for u, s in zip(legal, slots)}
    return legal, priors, float(wdl[0] - wdl[2])


def _terminal_value(board: Board):
    """Value from the side-to-move's view at a terminal position."""
    w = board.winner
    if w is None:
        return 0.0
    return 1.0 if w == board.side_to_move else -1.0


def search(ev: NetEvaluator, board: Board, sims: int, c_puct: float,
           rng: random.Random, noise: float = 0.0):
    """Run `sims` PUCT simulations from `board` (mutated via push/pop).
    Returns (root Node, root value estimate)."""
    root = Node()
    root_priors, root_v = None, None
    for _ in range(sims):
        node = root
        path = [node]
        # -- selection ----------------------------------------------------
        while node.terminal_v is None and node.children:
            total = sum(c.visits for c in node.children.values())
            best, best_score = None, -float("inf")
            for u, c in node.children.items():
                u_score = c.q + c_puct * c.prior * math.sqrt(total) / (1 + c.visits)
                if u_score > best_score:
                    best, best_score = u, u_score
            board.push(best)
            node = node.children[best]
            path.append(node)
        # -- expansion / terminal ------------------------------------------
        if node.terminal_v is not None:
            v = node.terminal_v
        else:
            legal, priors, v = _priors_and_value(ev, board)
            if board.termination != Termination.PLAYING:
                node.terminal_v = _terminal_value(board)
                v = node.terminal_v
            else:
                if node is root and noise > 0:
                    d = [rng.gammavariate(0.3, 1.0) for _ in legal]
                    s = sum(d)
                    priors = {u: (1 - noise) * priors[u] + noise * di / s
                              for u, di in zip(legal, d)}
                for u in legal:
                    node.children[u] = Node(parent=node, move=u,
                                            prior=priors[u])
                if node is root:
                    root_priors = priors
        # -- backup (sign-flip per ply) -------------------------------------
        sign = 1.0
        for n in reversed(path):
            n.visits += 1
            n.value_sum += sign * v
            sign = -sign
        # rewind the shared board to the root for the next simulation
        for _ in range(len(path) - 1):
            board.pop()
    return root, root_v if root_v is not None else root.q


def choose_move(root: Node, rng: random.Random, sample: bool):
    """Argmax visits (or ∝visits during the opening seed phase)."""
    items = sorted(root.children.items(), key=lambda kv: -kv[1].visits)
    if sample:
        total = sum(c.visits for _, c in items)
        r, acc = rng.random() * total, 0.0
        for u, c in items:
            acc += c.visits
            if r < acc:
                return u, c
    return items[0]


def play_game(eva, evb, a_white: bool, seed: int, cfg) -> dict:
    board = Board()
    rng = random.Random((seed << 1) | (0 if a_white else 1))
    ply = 0
    t0 = time.perf_counter()
    while board.termination == Termination.PLAYING and ply < cfg.max_plies:
        ev = eva if (board.side_to_move == 0) == a_white else evb
        root, _ = search(ev, board, cfg.sims, cfg.c_puct, rng)
        u, _node = choose_move(root, rng, ply < cfg.sample_plies)
        board.push(u)
        ply += 1
    w = board.winner
    kind = "natural" if board.termination != Termination.PLAYING else "adjudicated"
    return {"seed": seed, "a_white": a_white, "winner": w, "plies": ply,
            "kind": kind,
            "a_score": 0.5 if w is None else (1.0 if (w == 0) == a_white else 0.0),
            "s": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.az_eval")
    ap.add_argument("--mode", choices=["match", "analyse"], default="match")
    ap.add_argument("--a", default="runs/cvisit15/net-iter-0010.pt")
    ap.add_argument("--b", default="runs/policy_value_net_rope_25k.pt")
    ap.add_argument("--ckpt", default=None, help="analyse mode: the net")
    ap.add_argument("--fen", default=None, help="analyse mode: the position")
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--games", type=int, default=4, help="2 per seed-color pair")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first opening seed (for multi-process parallelism)")
    ap.add_argument("--sample-plies", type=int, default=10)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--jsonl", default="runs/eval/az_eval.jsonl")
    a = ap.parse_args(argv)

    if a.mode == "analyse":
        ev = NetEvaluator(a.ckpt, device=a.device)
        board = Board(a.fen)
        rng = random.Random(0)
        t0 = time.perf_counter()
        root, _ = search(ev, board, a.sims, a.c_puct, rng, a.noise)
        print(f"AZ-PUCT analysis  sims={a.sims}  c_puct={a.c_puct}  "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"position: {a.fen}")
        print(f"root value (visits-weighted): {root.q:+.3f}  "
              f"over {root.visits} sims")
        print(f"{'move':<8}{'P':>7}{'Q':>8}{'N':>6}")
        for u, c in sorted(root.children.items(), key=lambda kv: -kv[1].visits)[:8]:
            print(f"{u:<8}{c.prior:>7.3f}{c.q:>+8.3f}{c.visits:>6}")
        return 0

    eva = NetEvaluator(a.a, device=a.device)
    evb = NetEvaluator(a.b, device=a.device)
    print(f"AZ-PUCT match (no Gumbel, no batching): {a.a} vs {a.b}  "
          f"({a.games} games, {a.sims} sims/move, c_puct={a.c_puct})",
          flush=True)
    games, t0 = [], time.time()
    for seed in range(a.seed_start, a.seed_start + (a.games + 1) // 2):
        for a_white in (True, False):
            if len(games) >= a.games:
                break
            g = play_game(eva, evb, a_white, seed, a)
            games.append(g)
            print(f"  game {len(games):2d}/{a.games}  seed {g['seed']}  "
                  f"A={'white' if g['a_white'] else 'black'}  winner={g['winner']} "
                  f"({g['kind']}, {g['plies']} plies)  A-score {g['a_score']:.1f}  "
                  f"({g['s']} s)", flush=True)
    s = sum(g["a_score"] for g in games) / len(games)
    summary = {"a": Path(a.a).stem, "b": Path(a.b).stem, "mode": "az-puct",
               "sims": a.sims, "games": len(games),
               "score_a": round(s, 4),
               "draws": sum(1 for g in games if g["a_score"] == 0.5),
               "s": round(time.time() - t0, 1)}
    print(json.dumps(summary), flush=True)
    out = Path(a.jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            **summary}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
