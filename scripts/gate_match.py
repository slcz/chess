#!/usr/bin/env python3
"""gate_match.py — two-network gate matches in net-lockstep waves.

Two `uci_mux` processes (one per checkpoint, one network each) host ALL
of a wave's games through their own landing-zone mux, so every phase
batches all live games on the GPU. Wave 1 gives network A White in every
game; wave 2 swaps colors for the same book lines — each opening once
per color, the standing eval protocol. Deterministic play (argmax
visits) from the caller's book positions; claimable draws and a ply cap
adjudicated here by python-chess (test-only oracle).

Scores are from network A's perspective: 1 win, 1/2 draw, 0 loss.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import chess

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "engine/build/uci_mux"


class Engine:
    """One uci_mux process = one network hosting all of a wave's games."""

    label = "?"

    def __init__(self, ckpt, device, games, rows, walkers, seed, label="?"):
        self.label = label
        env = dict(__import__("os").environ, PUCT_CKPT=str(ckpt))
        if walkers:
            env["PUCT_WALKERS"] = str(walkers)
        argv = [str(DRIVER), str(games), str(rows), device, str(seed)]
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True,
                                  env=env)
        # the driver announces its rng stream (mux-seed=…) just before ready
        while True:
            line = self.p.stdout.readline().strip()
            if line == "ready":
                break
            if line.startswith("mux-seed="):
                self.mux_seed = line.split("=", 1)[1]
                continue
            print(f"engine {ckpt} failed to attach", file=sys.stderr)
            sys.exit(1)

    def send(self, line):
        if __import__("os").environ.get("GATE_DEBUG"):
            print(f"> [{self.label}] {line}", file=sys.stderr, flush=True)
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def read(self):
        line = self.p.stdout.readline()
        if not line:
            print(f"engine {self.label} died", file=sys.stderr)
            sys.exit(1)
        return line.strip()


def phase(eng, side, sims, games, boards, sent):
    """One think round for every live game on `side`: batched by the
    driver (all `go`s, then `run`), one bestmove per game. `sent` is THIS
    engine's per-game count of moves already fed to it — each engine has
    its own counter (A's own moves are played into A's tree by A; B only
    ever receives them as suffixes)."""
    # a game the OTHER side just mated is over but not yet swept — it
    # must not be queued for a think (the child has no legal moves)
    pending = [gid for gid in games if boards[gid].turn == side
               and not boards[gid].is_game_over(claim_draw=True)]
    for gid in pending:
        suffix = boards[gid].move_stack[sent[gid]:]
        if suffix:
            eng.send(f"game {gid} moves " + " ".join(m.uci() for m in suffix))
        eng.send(f"game {gid} go nodes {sims}")
    eng.send("run")
    for gid in pending:
        parts = eng.read().split()
        if (len(parts) != 4 or parts[0] != "game"
                or parts[2] != "bestmove" or int(parts[1]) != gid):
            print(f"protocol error: {parts}", file=sys.stderr)
            sys.exit(1)
        boards[gid].push_uci(parts[3])
        sent[gid] = len(boards[gid].move_stack)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--a-sims", type=int, default=200)
    ap.add_argument("--b-sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=16,
                    help="games PER WAVE (total = 2x; colors swap per wave)")
    ap.add_argument("--book", required=True,
                    help="FEN-per-line opening book (operator-supplied: the "
                         "gate plays every line once per colour). Generate "
                         "it with the local gen_prior_book.py.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--driver", default=str(DRIVER))
    ap.add_argument("--max-plys", type=int, default=300)
    ap.add_argument("--rows", type=int, default=0,
                    help="landing-zone rows per bank (0 = 8*games)")
    ap.add_argument("--walkers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None,
                    help="battery seed (opening sample + dice). DEFAULT: "
                         "fresh random seed per run — a fixed seed fights "
                         "the same openings every gate, understates error, "
                         "and reruns add no information. Pass an explicit "
                         "seed only to reproduce a recorded run; the "
                         "resolved seed is printed and stored in --json.")
    ap.add_argument("--dump-dir", default=None,
                    help="write one JSON per finished game (moves included) "
                         "for post-mortem inspection")
    ap.add_argument("--json", default=None,
                    help="write the machine-readable result (TOTAL + per-wave "
                         "SUMMARY) to this path — the loop's eval artifact")
    args = ap.parse_args()

    book = Path(args.book if args.book.startswith("/") else REPO / args.book)
    fens = [l.strip() for l in book.read_text().splitlines() if l.strip()]
    if len(fens) < args.games:
        print(f"book has {len(fens)} positions < games {args.games}",
              file=sys.stderr)
        sys.exit(2)
    seed = args.seed if args.seed is not None else random.randrange(1 << 31)
    print(f"GATE seed={seed} book={book.name} lines={len(fens)} "
          f"sampled={args.games} a_sims={args.a_sims} b_sims={args.b_sims}",
          flush=True)
    rng = random.Random(seed)
    wave_fens = rng.sample(fens, args.games)
    # distinct per-engine rng streams derived from the run seed: replaying a
    # recorded seed reproduces openings AND both engines' sampled dice
    seed_a, seed_b = (seed * 2 + 1) % (1 << 64), (seed * 2 + 2) % (1 << 64)

    eng_a = Engine(args.ckpt_a, args.device, args.games, args.rows,
                   args.walkers, seed_a, label="A")
    eng_b = Engine(args.ckpt_b, args.device, args.games, args.rows,
                   args.walkers, seed_b, label="B")
    games = list(range(args.games))

    t0 = time.time()
    total = {"w": 0, "d": 0, "l": 0}
    waves = []
    for wave, a_white in enumerate([True, False]):
        boards = [chess.Board(fen) for fen in wave_fens]
        sent_a = [0] * args.games
        sent_b = [0] * args.games
        alive = list(games)
        w = d = l = 0
        t_wave = time.time()
        rnd = 0
        # plant the book position in BOTH processes (fresh tree per game)
        for gid, fen in enumerate(wave_fens):
            eng_a.send(f"game {gid} fen {fen}")
            eng_b.send(f"game {gid} fen {fen}")
        while alive:
            rnd += 1
            side = chess.WHITE if a_white else chess.BLACK
            phase(eng_a, side, args.a_sims, alive, boards, sent_a)
            phase(eng_b, not side, args.b_sims, alive, boards, sent_b)
            if rnd % 10 == 1:
                print(f"# wave={wave} round={rnd} alive={len(alive)} "
                      f"ply={boards[alive[0]].ply()}", file=sys.stderr,
                      flush=True)
            still = []
            for gid in alive:
                over = boards[gid].is_game_over(claim_draw=True)
                if over or boards[gid].ply() >= args.max_plys:
                    res = (boards[gid].result(claim_draw=True)
                           if over else "1/2-1/2")
                    sc = 0.5 if res == "1/2-1/2" else (
                        1.0 if (res == "1-0") == a_white else 0.0)
                    w, d, l = (w + (sc == 1.0), d + (sc == 0.5),
                               l + (sc == 0.0))
                    print(f"GAME wave={wave} game={gid} result={res} "
                          f"plies={boards[gid].ply()}", flush=True)
                    if args.dump_dir:
                        # post-mortem inspection: the full game record
                        # (book start, every played move). gate play is
                        # deterministic, so a rerun with the same seed
                        # reproduces these games.
                        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
                        with open(Path(args.dump_dir)
                                  / f"wave{wave}-game{gid:03d}.json", "w") as f:
                            json.dump({"wave": wave, "gid": gid,
                                       "a_white": a_white, "result": res,
                                       "plies": boards[gid].ply(),
                                       "fen": wave_fens[gid],
                                       "moves": [m.uci() for m in
                                                 boards[gid].move_stack]},
                                      f)
                else:
                    still.append(gid)
            alive = still
        total["w"] += w
        total["d"] += d
        total["l"] += l
        n = w + d + l
        waves.append({"wave": wave, "a_white": a_white, "games": n,
                      "w": w, "d": d, "l": l,
                      "score_a": (w + 0.5 * d) / n if n else 0.0})
        print(f"SUMMARY wave={wave} a_white={a_white} games={n} w={w} "
              f"d={d} l={l} score_a={(w + 0.5 * d) / n if n else 0.0:.3f} "
              f"elapsed={time.time() - t_wave:.0f}s", flush=True)
    n = total["w"] + total["d"] + total["l"]
    print(f"TOTAL games={n} w={total['w']} d={total['d']} l={total['l']} "
          f"score_a={(total['w'] + 0.5 * total['d']) / n if n else 0.0:.3f} "
          f"elapsed={time.time() - t0:.0f}s")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "games": n, "w": total["w"], "d": total["d"], "l": total["l"],
            "score_a": (total["w"] + 0.5 * total["d"]) / n if n else 0.0,
            "waves": waves,
            "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
            "a_sims": args.a_sims, "b_sims": args.b_sims,
            "seed": seed, "engine_seeds": [seed_a, seed_b],
            "elapsed_s": round(time.time() - t0, 1),
        }, indent=2))
    eng_a.p.stdin.close()
    eng_b.p.stdin.close()
    eng_a.p.wait(timeout=30)
    eng_b.p.wait(timeout=30)


if __name__ == "__main__":
    main()
