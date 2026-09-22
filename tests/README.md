# tests/ — pytest suite

Verification is designed alongside the code, not bolted on after (PLAN.md §10):

- Perft fixtures (`fixtures/`): standard positions — startpos, Kiwipete, … — with exact node counts at depth 5–6, run via the C CLI and through the FFI. The non-negotiable gate before any ML code exists.
- Rules unit tests: castling through check, en-passant pins, promotion, 50-move rule, threefold repetition, insufficient material.
- Oracle fuzzing vs. **python-chess** (test-only oracle — it never touches training data, so zero-knowledge stays intact): legal-move sets, termination calls, and FEN round-trips agree across ≥10⁶ random game walks.
- C hygiene: the pytest suite also runs against ASan/UBSan builds of the engine.
- Later phases: net overfit test (Phase 2 gate), tiny-game smoke test (Phase 3 gate).
