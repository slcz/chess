"""
chess_ai/match.py — game & match runner between two strategies.

The loop is deliberately boring: it is the same shape the self-play data
generator will have (play until the C engine says the game is over, record
the result), minus recording. Everything interesting lives inside the
strategies and, later, behind the batched-eval boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from chess_ai.engine import Board, Termination

# Random games can shuffle nearly forever if only mate/stalemate end them;
# the termination checks usually kick in far earlier, but keep a sane cap
# so a buggy player cannot hang a match.
DEFAULT_MAX_PLY = 600


@dataclass
class GameResult:
    moves: list[str] = field(default_factory=list)
    termination: Termination = Termination.PLAYING
    winner: int | None = None          # 0 = white, 1 = black, None = draw/cap
    plies: int = 0
    hit_ply_cap: bool = False

    def describe(self) -> str:
        if self.hit_ply_cap:
            return "ply cap reached (adjudicated draw)"
        name = self.termination.name.lower().replace("_", " ")
        if self.winner is None:
            return f"draw ({name})"
        return f"{'white' if self.winner == 0 else 'black'} wins ({name})"


def play_game(white, black, start_fen: str | None = None,
              max_ply: int = DEFAULT_MAX_PLY, verbose: bool = False) -> GameResult:
    """Play one game; `white`/`black` are Strategy callables (see strategies.py)."""
    board = Board(start_fen) if start_fen else Board()
    result = GameResult()

    while board.termination == Termination.PLAYING and board.ply < max_ply:
        player = white if board.side_to_move == 0 else black
        move = player.choose_move(board)
        if move not in board.legal_moves:
            raise RuntimeError(
                f"{getattr(player, '__class__', type(player)).__name__} "
                f"returned illegal {move!r} at {board.fen}")
        board.push(move)
        result.moves.append(move)
        if verbose:
            print(f"{board.ply:3d}. {'W' if board.side_to_move == 1 else 'B'} {move}")

    result.termination = board.termination
    result.winner = board.winner
    result.plies = board.ply
    result.hit_ply_cap = board.termination == Termination.PLAYING
    return result


def play_match(a, b, n_games: int = 10, start_fen: str | None = None,
               verbose: bool = False) -> dict:
    """Play `n_games` with colors alternating; report scores for A.

    Returns {"a_wins", "b_wins", "draws", "games": [GameResult, ...]}.
    """
    score = {"a_wins": 0, "b_wins": 0, "draws": 0, "games": []}
    for i in range(n_games):
        # i even: A takes white; i odd: A takes black.
        white, black = (a, b) if i % 2 == 0 else (b, a)
        r = play_game(white, black, start_fen)
        winner_is_a = r.winner == (0 if i % 2 == 0 else 1)
        if r.winner is None:
            score["draws"] += 1
        elif winner_is_a:
            score["a_wins"] += 1
        else:
            score["b_wins"] += 1
        score["games"].append(r)
        if verbose:
            print(f"game {i + 1:2d}: {r.describe()}, {r.plies} plies")
    return score
