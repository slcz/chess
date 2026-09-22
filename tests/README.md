# tests/ — pytest suite

Verification is designed alongside the code, not bolted on after (PLAN.md §10):

```sh
.venv/bin/python -m pytest tests/ -v        # single entry point (~30 s)
cd engine && make check                     # C-only fast path (~15 s)
cd engine && make check-asan                # sanitizer build (~1 min)
```

## Layers (current state)

| Layer | What | Where |
|---|---|---|
| Perft | exact node counts, published externally (chessprogramming.org + Peter Ellis Jones gist), incl. startpos d6 / Kiwipete d5 | `engine/tests/perft_suite.txt` (human-editable data file), run by `test_units` and `perft suite` |
| C unit tests | rule edge cases (castling through check, ep pin, promotions, mate/stalemate), make/unmake bit-exact restore, mailbox↔bitboard consistency, FEN round-trips, random self-play games | `engine/tests/test_units.c` |
| pytest gates | build the engine, run the two binaries above as subprocess gates, and oracle-fuzz legal-move sets + FEN serialization against python-chess over random game walks | `tests/test_engine.py` |
| Sanitizers | whole suite under ASan+UBSan | `engine`'s `make check-asan` |

python-chess is a **test-only oracle**: it never touches training data, so the zero-knowledge constraint holds. The fuzz harness takes `FUZZ_GAMES` / `FUZZ_MAX_PLIES` env knobs (defaults 40 / 300 ≈ 11k positions, ~4 s); a 400-game run covered 145k positions cleanly. The plan's full ≥10⁶-walk fuzz (Phase 1.5) runs the same harness, just bigger.

## Still ahead

- 50-move / threefold / insufficient-material termination tests (Phase 1.3)
- Perft through the FFI binding, once cffi lands (Phase 0.3)
- Net overfit test (Phase 2 gate), tiny-game smoke test (Phase 3 gate), tactic-suite fixtures in `fixtures/` (Phase 6)
