"""
tests/test_supervised.py — PGN parser + dataset pack invariants (side track).

The parser is the entry point for ALL real data on this branch, so the
tests aim at the two ways it could silently poison a run:

1. SAN→UCI mismapping (wrong disambiguation, promotion, castling) —
   caught by the round-trip property: every legal move in thousands of
   random positions, converted to SAN by chess_ai.san (independently
   written for the GUI) and back by the parser, must return the exact
   original UCI.
2. Framing/filtering mistakes (dropping good games, keeping games the
   filters should reject, pack lines that don't replay) — caught by
   running preprocess over a synthetic dump containing one game of
   every kind and asserting the pack contains exactly the right ones.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pytest

from chess_ai.engine import Board, Termination
from chess_ai.san import uci_to_san
from chess_ai.supervised.pgn import (
    ParseError,
    iter_pgn_games,
    san_to_uci,
    tokenize_movetext,
)
from chess_ai.supervised.dataset import (
    PositionSampler,
    load_pack,
    preprocess,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# A realistic Lichess-shaped game: full header block, clk comments, NAGs,
# a glued move number, wrapped movetext, result token.
_GOOD_GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/xxxxxxx"]
[Result "1-0"]
[UTCDate "2013.01.01"]
[White "Alice"]
[Black "Bob"]
[WhiteElo "1700"]
[BlackElo "1690"]
[TimeControl "300+2"]
[Termination "Normal"]

1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:05:00] } 2. Nf3 { [%clk 0:04:58] }
Nc6 { [%clk 0:04:57] } 3. Bc4 $1 { [%clk 0:04:55] } Bc5 4. c3 Nf6 5. d4 exd4
6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. O-O { castles } O-O 1-0
"""


def _pgn(*games: str) -> io.StringIO:
    return io.StringIO("\n\n".join(games) + "\n")


def _hdr(result="1-0", we="1700", be="1690", tc="300+2", term="Normal",
         extra=""):
    return (f'[Result "{result}"]\n[WhiteElo "{we}"]\n[BlackElo "{be}"]\n'
            f'[TimeControl "{tc}"]\n[Termination "{term}"]\n{extra}')


def _moves(movetext="1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"):
    return "\n" + movetext + " 1-0\n"


# ---------------------------------------------------------------------------
# tokenize + SAN→UCI
# ---------------------------------------------------------------------------

def test_tokenize_strips_comments_nags_variations_numbers():
    text = ("1. e4 { [%clk 0:05:00] } e5 ; inline comment\n"
            "2. Nf3 $1 (2. f4 exf4) 2... Nc6 3.Bb5 a6 1-0")
    assert tokenize_movetext(text) == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]


def test_tokenize_result_and_star_tokens_dropped():
    assert tokenize_movetext("1. d4 * ") == ["d4"]
    assert tokenize_movetext("1/2-1/2") == []


def test_san_round_trip_over_random_games():
    """Every legal move: uci_to_san → san_to_uci must be the identity.

    uci_to_san (chess_ai/san.py, written for the GUI) and san_to_uci are
    independent implementations; agreement over whole random games covers
    disambiguation (Nbd2/R1a3/Qed4), promotions, underpromotions,
    castling through both flanks, en passant, check/mate suffixes.
    """
    rng = random.Random(7)
    checked = 0
    for g in range(6):
        board = Board(START)
        while (board.termination == Termination.PLAYING
               and board.ply < 120 and checked < 3000):
            for uci in board.legal_moves:
                assert san_to_uci(board, uci_to_san(board, uci)) == uci
                checked += 1
            board.push(rng.choice(board.legal_moves))
    assert checked >= 2000


def test_san_promotion_and_castling_specifics():
    # Promotion with capture + underpromotion, from a crafted position.
    b = Board("rnbq1bnr/ppppp1P1/8/5k2/8/8/8/K3R3 w kq - 0 1")
    legal = set(b.legal_moves)
    assert "g7g8q" in legal or "g7h8q" in legal
    for uci in legal:
        san = uci_to_san(b, uci)
        assert san_to_uci(b, san) == uci
    # Castling both spellings in a castling-legal position.
    b2 = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert san_to_uci(b2, "O-O") == "e1g1"
    assert san_to_uci(b2, "0-0-0") == "e1c1"
    assert san_to_uci(Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"), "O-O-O") == "e8c8"


def test_san_rejects_garbage():
    b = Board(START)
    with pytest.raises(ParseError):
        san_to_uci(b, "Zz9")
    with pytest.raises(ParseError):
        san_to_uci(b, "Ke9")          # legal shape, unreachable square


# ---------------------------------------------------------------------------
# Game framing + preprocessing filters
# ---------------------------------------------------------------------------

def test_iter_pgn_games_parses_full_game():
    (headers, moves), = iter_pgn_games(_pgn(_GOOD_GAME))
    assert headers["Result"] == "1-0"
    assert headers["WhiteElo"] == "1700"
    assert moves[:4] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert moves[-2:] == ["e1g1", "e8g8"]    # white O-O, then black O-O
    assert "b1d2" in moves                   # Nbxd2 disambiguation resolved


def test_iter_pgn_games_survives_bad_game_and_counts_it():
    bad = _hdr() + "\n1. e4 e5 2. Qh5 Ke7 Zz9 1-0\n"   # unparseable SAN
    errors = []
    games = list(iter_pgn_games(_pgn(_GOOD_GAME, bad, _GOOD_GAME),
                                on_error=lambda h, e: errors.append(str(e))))
    assert len(games) == 2            # stream continued past the bad one
    assert len(errors) == 1
    assert "Zz9" in errors[0]


def test_preprocess_filters(tmp_path):
    pgn = tmp_path / "dump.pgn"
    games = [
        _hdr() + _moves(),                                     # keep
        _hdr(we="1500") + _moves(),                            # low elo
        _hdr(be="?") + _moves(),                               # unrated
        _hdr(tc="60+0") + _moves(),                            # bullet
        _hdr() + "\n1. e4 1-0\n",                              # too short
        _hdr() + "\n1. e4 e5 2. Nf3 Nf6 3. Ng1 Ng8 "
        + "Nf3 Nf6 Ng1 Ng8 " * 100 + "1-0\n",                # too long (legal)
        _hdr(term="Time forfeit") + _moves(),                  # kept by default
        _hdr(extra='[SetUp "1"]\n[FEN "8/8/8/8/8/8/8/K6k w - - 0 1"]')
        + "\n1. Ka2 1-0\n",                                    # custom start
        _hdr(result="*") + _moves(),                           # no result
    ]
    pgn.write_text("\n\n".join(games) + "\n")

    pack = tmp_path / "out.pack"
    meta = preprocess(str(pgn), str(pack), min_elo=1600, min_base_sec=180,
                      min_plies=4, max_plies=300)
    lines = pack.read_text().strip().splitlines()
    assert len(lines) == 2               # good + time-forfeit
    assert lines[0].startswith("1-0\t")
    assert lines[0].split("\t")[1].startswith("e2e4 e7e5")
    s = meta["stats"]
    assert s["kept"] == 2 and s["positions"] == 12
    assert s["dropped"]["low_elo"] == 1
    assert s["dropped"]["unrated"] == 1
    assert s["dropped"]["time_control"] == 1
    assert s["dropped"]["ply_range"] == 2
    assert s["dropped"]["custom_start"] == 1
    assert s["dropped"]["no_result"] == 1
    assert json.loads(Path(str(pack) + ".stats.json").read_text())["stats"]["kept"] == 2


# ---------------------------------------------------------------------------
# Pack loading + sampler
# ---------------------------------------------------------------------------

def _tiny_pack_lines():
    # 3 short games with known outcomes and perspectives.
    return [
        "1-0\te2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7",     # scholar's mate, white
        "0-1\te2e4 d7d5 b1c3 d5d4 c3e4",               # black wins
        "1/2-1/2\td2d4 d7d5",                          # 2-ply draw
    ]


def test_load_pack_and_sampler_schema(tmp_path):
    pack = tmp_path / "p.pack"
    pack.write_text("\n".join(_tiny_pack_lines()) + "\n")
    games = load_pack(str(pack))
    assert len(games) == 3
    assert games[0][0] == 1.0 and games[2][1] == ("d2d4", "d7d5")

    s = PositionSampler(games, seed=0)
    assert s.n_positions == 7 + 5 + 2
    b1 = s.sample_batch(16)
    assert b1["tokens"].shape == (16, 64, 117)
    assert b1["mask"].shape == (16, 4240)
    assert b1["pol_idx"].dtype == np.int64
    assert b1["wdl"].shape == (16, 3)
    # every policy target inside its own legal mask (replay correctness)
    assert b1["mask"][np.arange(16), b1["pol_idx"]].all()
    # WDL one-hot rows sum to 1
    assert (b1["wdl"].sum(-1) == 1.0).all()
    # determinism at fixed seed
    s2 = PositionSampler(load_pack(str(pack)), seed=0)
    b2 = s2.sample_batch(16)
    assert (b1["pol_idx"] == b2["pol_idx"]).all()

    picks = s.fixed_positions(5)
    assert len(picks) == 5
    assert s.fixed_positions(5) == picks


def test_sampler_wdl_perspective_matches_mover():
    """In the scholar's-mate game, white's plies are W, black's are L."""
    pack_lines = [_tiny_pack_lines()[0]]
    s = PositionSampler([(1.0, tuple(pack_lines[0].split("\t")[1].split()))],
                        seed=0)
    board = Board()
    from chess_ai.supervised.dataset import replay_position
    for ply in range(7):
        _, _, _, wdl = replay_position(s.games[0], ply, board)
        expect_w = 1.0 if ply % 2 == 0 else 0.0
        assert wdl[0] == expect_w and wdl[1] == 0.0
