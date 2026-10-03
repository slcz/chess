"""
tests/test_gui_net.py — the GUI's searched 'net' controller
(chess_ai.gui.NetSearchStrategy: the gumbel.py reference search + a
checkpoint, wired as a Strategy).

Pins the pieces that make baseline-vs-selfplay-net GUI playback
trustworthy:

1. The search finds mate-in-1 even with a flat fixture evaluator — the
   terminal child's exact one-hot value dominates the halving regardless
   of the net's untrained opinions.
2. Real-checkpoint play (tiny untrained net, CPU): every chosen move is
   legal, the GUI's live Board is left exactly as found (the search
   pushes/pops and must rewind), and a given board+seed reproduces the
   move (stateless per-move search).
3. The per-side checkpoint wiring: NET_CKPT_WHITE / NET_CKPT_BLACK pick
   different files (that's the baseline-vs-v3 pairing) with NET_CKPT as
   the fallback — and a missing checkpoint fails loudly.
"""
from __future__ import annotations

import pytest

from chess_ai.engine import Board, Termination
from chess_ai.gui import NetSearchStrategy, _make_controller
from chess_ai.selfplay.gumbel import TableEvaluator
from chess_ai.net.features import uci_to_index

MATE_FEN = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"   # Re1-e8# (same as value play)


def test_net_search_finds_mate_in_one():
    """Flat priors except a nudge toward e1e8: the search must still pick
    the mating move — terminal q = +1 beats everything in the halving."""
    mover = 0
    mate_slot = uci_to_index("e1e8", mover)
    evaluator = TableEvaluator(prior={mate_slot: 5.0})
    me = NetSearchStrategy(n=100, sample_plies=0, seed=0, evaluator=evaluator)
    board = Board(MATE_FEN)
    assert me.choose_move(board) == "e1e8"
    assert board.fen == MATE_FEN                       # untouched by the search


def test_net_search_legal_and_reproducible(tmp_path):
    """Tiny untrained ckpt, CPU: legal moves only, board untouched, and the
    same board+seed asks back the same move (fresh tree per call)."""
    import torch

    from chess_ai.net.model import ChessTransformer, NetConfig
    from chess_ai.train.trainer import TrainConfig, Trainer

    torch.manual_seed(0)
    trainer = Trainer(ChessTransformer(NetConfig(n_layers=2, d_model=64,
                                                 n_heads=4, d_ff=128,
                                                 aux_reply=False)),
                      TrainConfig(batch_size=32, device="cpu"))
    ckpt = tmp_path / "tiny.pt"
    trainer.save(ckpt)

    me = NetSearchStrategy(ckpt_path=str(ckpt), n=4, sample_plies=3,
                           seed=7, device="cpu")
    board = Board()
    while board.ply < 12 and board.termination == Termination.PLAYING:
        fen_before = board.fen
        move = me.choose_move(board)
        assert move in board.legal_moves
        assert board.fen == fen_before                 # search left no residue
        assert me.choose_move(board) == move           # reproducible per seed
        board.push(move)


def test_per_side_checkpoints(tmp_path, monkeypatch):
    """NET_CKPT_WHITE / NET_CKPT_BLACK select per-side files; NET_CKPT is
    the fallback; nothing set → loud error."""
    import torch

    from chess_ai.net.model import ChessTransformer, NetConfig
    from chess_ai.train.trainer import TrainConfig, Trainer

    paths = {}
    for name, seed in (("white", 0), ("black", 1)):
        torch.manual_seed(seed)
        trainer = Trainer(ChessTransformer(NetConfig(n_layers=2, d_model=64,
                                                     n_heads=4, d_ff=128,
                                                     aux_reply=False)),
                          TrainConfig(batch_size=32, device="cpu"))
        ckpt = tmp_path / f"ckpt_{name}.pt"
        trainer.save(ckpt)
        paths[name] = str(ckpt)

    monkeypatch.setenv("NET_CKPT_WHITE", paths["white"])
    monkeypatch.setenv("NET_CKPT_BLACK", paths["black"])
    white = _make_controller("net", seed=1, movetime_ms=0, side="w")
    black = _make_controller("net", seed=2, movetime_ms=0, side="b")
    assert "ckpt_white" in white.label
    assert "ckpt_black" in black.label

    monkeypatch.delenv("NET_CKPT_WHITE")
    monkeypatch.delenv("NET_CKPT_BLACK")
    monkeypatch.setenv("NET_CKPT", paths["white"])
    both = _make_controller("net", seed=1, movetime_ms=0, side="w")
    assert "ckpt_white" in both.label

    monkeypatch.delenv("NET_CKPT")
    with pytest.raises(RuntimeError, match="checkpoint"):
        _make_controller("net", seed=1, movetime_ms=0, side="w")
