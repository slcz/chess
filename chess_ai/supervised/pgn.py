"""
chess_ai/supervised/pgn.py — streaming PGN reader + SAN→UCI (side track).

Parses the Lichess monthly bulk dumps (real human games) into
(start FEN, UCI move list, outcome) games — the raw material for the
value-net validation runs. Deliberately minimal, and built ONLY on our
own C engine: python-chess stays a test-only oracle (repo rule), so the
production data path never depends on it.

Two layers:

1. GAME FRAMING (iter_pgn_games)
   Lichess bulk PGN shape: one blank line, a header block of
   `[Tag "Value"]` lines, a blank line, then movetext (possibly wrapped
   over several lines) ending in the game result token. We stream line
   by line (the dumps are gigabytes), keep only the tags we filter on
   (Result/WhiteElo/BlackElo/TimeControl/Termination + FEN/SetUp if a
   game ever starts from a custom position), and hand the movetext to
   the tokenizer.

2. MOVETEXT → UCI (tokenize_movetext + san_to_uci)
   Comments `{...}` (Lichess writes `[%clk ...]` in newer dumps), NAGs
   `$n`, variations `(...)` and move numbers are stripped; each
   remaining SAN token is matched against the CURRENT legal moves by
   structural filtering (piece letter, disambiguation file/rank,
   capture, target square, promotion) — never by generating SAN strings
   for comparison (that would be ~30× the work per move). Exactly one
   legal move must match; anything else raises ParseError and the whole
   game is dropped by the caller (counted, so dump oddities surface in
   preprocess stats instead of poisoning data silently — the §10.1
   instinct applied to someone else's data).
"""
from __future__ import annotations

import re
from typing import IO, Dict, Iterator, List, Optional, Tuple

from chess_ai.engine import Board

# Header values → white-POV outcome (same convention as GameRecord).
RESULT_SCORES = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
_RESULT_TOKENS = set(RESULT_SCORES) | {"*"}

# Headers kept per game: the filters need these; FEN/SetUp covers the
# (rare) custom-start games in the dumps.
_KEEP_TAGS = ("Result", "WhiteElo", "BlackElo", "TimeControl",
              "Termination", "FEN", "SetUp", "PlyCount")

_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
_NAG_RE = re.compile(r"\$\d+")

# SAN body: optional piece letter, optional disambiguation file and/or
# rank, optional capture x, target square, optional =PROMOTION.
_SAN_RE = re.compile(
    r"^([KQRBN])?([a-h])?([1-8])?(x)?([a-h][1-8])(?:=?([QRBN]))?$"
)
_FILES = "abcdefgh"


class ParseError(ValueError):
    """Anything in a game we cannot consume exactly; caller drops it."""


# ---------------------------------------------------------------------------
# Movetext → SAN tokens
# ---------------------------------------------------------------------------

def tokenize_movetext(text: str) -> List[str]:
    """SAN tokens from raw movetext, comments/NAGs/variations/numbers gone.

    The final result token is NOT part of the return (framing already
    knows the result from the headers; the tokenizer just must not hand
    it to san_to_uci).
    """
    # Comments first (they could contain '(', ';', '$' — remove before the
    # structural strips), then ';' to-end-of-line comments, then nested
    # variations (Lichess bulk dumps have none, but a robust reader costs
    # three lines).
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r";[^\n]*", " ", text)
    while "(" in text:
        stripped = re.sub(r"\([^()]*\)", " ", text)
        if stripped == text:            # unbalanced '(' — unusable game
            raise ParseError("unbalanced variation")
        text = stripped
    text = _NAG_RE.sub(" ", text)

    toks = []
    for tok in text.split():
        if tok in _RESULT_TOKENS:
            continue                     # end-of-game marker(s)
        tok = tok.lstrip(".").strip()
        if not tok:
            continue
        # A move number may be glued to the SAN ('12.Nf3', '3...e5'):
        while tok and (tok[0].isdigit() or tok[0] == "."):
            tok = tok.lstrip("0123456789.")
        if not tok:
            continue
        if tok in _RESULT_TOKENS:
            continue
        toks.append(tok)
    return toks


# ---------------------------------------------------------------------------
# SAN → UCI on our engine
# ---------------------------------------------------------------------------

def _sq(name: str) -> int:
    return (ord(name[1]) - ord("1")) * 8 + (ord(name[0]) - ord("a"))


def san_to_uci(board: Board, san: str) -> str:
    """One SAN token in `board`'s position → its UCI string.

    Uses only board.legal_moves + board.piece_at (both cheap FFI reads).
    Raises ParseError on ambiguity or no match — the caller drops the
    game, keeping the pack exactly-right by construction.
    """
    core = san.rstrip("+#!?")
    side = board.side_to_move

    # Castling. Tolerate the 0-0 spelling; our engine's UCI is e1g1-style.
    if core in ("O-O", "0-0", "O-O-O", "0-0-0"):
        rank = "1" if side == 0 else "8"
        to_file = "g" if core in ("O-O", "0-0") else "c"
        want_from, want_to = "e" + rank, to_file + rank
        hits = [u for u in board.legal_moves
                if u[0:2] == want_from and u[2:4] == want_to]
        if len(hits) != 1:
            raise ParseError(f"castling {san!r} not uniquely legal")
        return hits[0]

    m = _SAN_RE.match(core)
    if not m:
        raise ParseError(f"unparseable SAN {san!r}")
    piece, dfile, drank, _x, target, promo = m.groups()

    want_type = {"K": 5, "Q": 4, "R": 3, "B": 2, "N": 1}.get(piece or "", 0)

    hits = []
    for uci in board.legal_moves:
        if uci[2:4] != target:
            continue                     # ~2-4 candidates survive this
        if promo is not None:
            if uci[4:] != promo.lower():
                continue                 # explicit =PROMOTION: exact match
        elif uci[4:] != "":
            # Bare target SAN on a promotion square: tolerate only the
            # implied =Q (some exporters drop the suffix); underpromotions
            # are always explicit.
            if uci[4:] != "q":
                continue
        frm = uci[0:2]
        if dfile and frm[0] != dfile:
            continue
        if drank and frm[1] != drank:
            continue
        pc = board.piece_at(_sq(frm))
        if pc // 6 != side or pc % 6 != want_type:
            continue                     # right color & piece type from-square
        hits.append(uci)
    if len(hits) != 1:
        raise ParseError(f"SAN {san!r} matched {len(hits)} legal moves")
    return hits[0]


# ---------------------------------------------------------------------------
# Game framing (streaming over the whole dump)
# ---------------------------------------------------------------------------

def iter_pgn_games(
    stream: IO[str],
    on_error=None,
) -> Iterator[Tuple[Dict[str, str], List[str]]]:
    """Yield (headers, uci_moves) per game, streaming a PGN text stream.

    Applies NO filtering (that's preprocess's job, with stats) — but it
    DOES convert to UCI via our engine, so games whose movetext we can't
    reproduce exactly are skipped here. A generator that raised on the
    first odd game would die with it (one raise terminates the whole
    stream), so ParseError is caught per game instead; pass `on_error`
    (headers-dict, exception) to count the drops.
    """
    headers: Dict[str, str] = {}
    movetext: List[str] = []
    in_moves = False

    def finish() -> Optional[Tuple[Dict[str, str], List[str]]]:
        nonlocal headers, movetext, in_moves
        game = None
        # "*" = abandoned/unplayed game: still yielded (its movetext is
        # fine) so preprocess can count it under no_result rather than
        # silently vanishing from the seen tally.
        if headers and headers.get("Result") in set(RESULT_SCORES) | {"*"}:
            sans = tokenize_movetext(" ".join(movetext))
            board = Board(headers.get("FEN")) \
                if headers.get("SetUp") == "1" and headers.get("FEN") else Board()
            uci_moves: List[str] = []
            for san in sans:
                uci_moves.append(san_to_uci(board, san))
                board.push(uci_moves[-1])
            game = (headers, uci_moves)
        headers, movetext, in_moves = {}, [], False
        return game

    for line in stream:
        stripped = line.strip()
        if not stripped:
            continue                     # blank separators carry no state
        if not in_moves and stripped.startswith("[") and _HEADER_RE.match(stripped):
            tag, value = _HEADER_RE.match(stripped).groups()
            if tag in _KEEP_TAGS:
                headers[tag] = value
            continue
        in_moves = True                  # first non-header, non-blank line
        movetext.append(stripped)
        if stripped.endswith(tuple(_RESULT_TOKENS)):
            try:
                game = finish()
            except ParseError as exc:
                if on_error is not None:
                    on_error(dict(headers), exc)
                # finish() raised before its own reset — do it here or the
                # next game's headers would be eaten as this one's movetext.
                headers, movetext, in_moves = {}, [], False
                game = None
            if game is not None:
                yield game


def open_pgn(path: str) -> IO[str]:
    """Open a .pgn / .pgn.zst / .pgn.gz as a text stream.

    zstd via the CLI (the dumps are .zst; keeps the venv dependency-free —
    the container installs `zstd` alongside python). gzip via stdlib.
    """
    import gzip
    import subprocess
    import codecs
    if path.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-dc", path], stdout=subprocess.PIPE)
        return codecs.getreader("utf-8")(proc.stdout, errors="replace")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")
