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
| C NN tests | Phase 2 input encoder: purity (State unchanged), startpos tensor, mover-perspective flip, history frames, repetition flags, castling reorder, ring push/pop, invariants over random games | `engine/tests/test_nn.c` |
| pytest gates | build the engine, run the C binaries above as subprocess gates, and oracle-fuzz legal-move sets + FEN serialization against python-chess over random game walks | `tests/test_engine.py` |
| Feature parity | C encoder vs an independent pure-Python reference (`chess_ai/net/features.py`), tensor-for-tensor over random games; move-space round-trip/injectivity incl. promotions both colors | `tests/test_nn_features.py` |
| Net contracts | parameter budget (~6.8M), illegal-move probability exactly zero + unit mass on legal, value range, aux fallback for finished games, one train step finite, MPS smoke | `tests/test_net_model.py` |
| Train pipeline | build_tensors invariants (targets inside their masks, aux frame, WDL perspective), replay-buffer cap/sampling, checkpoint save/load resumes bit-identically | `tests/test_train.py` |
| Phase 2 gate | the net *learns* a fixed structured fixture: random-vs-greedy policy labels (greedy side only) + material-sign W/D/L value labels; bars policy ≥ 0.65 / value ≥ 0.92 + masking re-asserted (see test docstring for the design rationale — uniform-random labels rote-stall) | `tests/test_overfit.py` |
| Sanitizers | whole suite under ASan+UBSan | `engine`'s `make check-asan` |

python-chess is a **test-only oracle**: it never touches training data, so the zero-knowledge constraint holds. The fuzz harness takes `FUZZ_GAMES` / `FUZZ_MAX_PLIES` env knobs (defaults 40 / 300 ≈ 11k positions, ~4 s); a 400-game run covered 145k positions cleanly. The plan's full ≥10⁶-walk fuzz (Phase 1.5) runs the same harness, just bigger.

## Still ahead

- Tiny-game smoke test (Phase 3 gate), tactic-suite fixtures in `fixtures/` (Phase 6)
