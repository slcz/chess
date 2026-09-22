# engine/ — the C play path

The performance-critical half of the system (same split as Lc0/KataGo): a shared library with **no globals** (hundreds of game instances per process) and **zero dynamic memory** after init — fixed-size `State` structs, stack move lists, arena-allocated search trees. This design removes most C pitfalls up front and is what lets one Python process feed a batched GPU.

Planned contents (Phase 1+):

- Bitboard board representation; make/unmake with a fixed undo stack
- FEN parse/serialize
- Full legal movegen: castling (incl. through check), en passant (incl. ep pins), promotions, check/mate/stalemate detection
- Termination: 50-move rule, threefold repetition, insufficient material
- Perft CLI (`perft`, `perft divide`) for exact node-count verification
- MCTS in C: PUCT with Dirichlet root noise + a Gumbel/sequential-halving variant, tree nodes from a node-pool reset per move
- Input-plane encoder — implemented once here, shared by self-play and training data prep

Build system lands in Phase 0.3: Makefile with default + ASan/UBSan targets. Verified by perft fixtures and python-chess oracle fuzzing before any ML code exists (Phase 1 gate) — one rules edge-case bug silently poisons all training data.
