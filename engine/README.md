# engine/ — the C play path

The C half of the system: a bitboard rules engine (strictly-legal move
generation, FEN, termination and repetition semantics matching
python-chess) plus the PUCT/MCTS walking it — batched "walkers" stepping a
muxed landing zone so one Python process can feed a GPU across many games at
once. The headers are the design authority: `include/chess.h` (rules),
`include/chess_puct.h` (search, root noise, the LEARN target),
`include/chess_nn.h` (the 64×117 input encoder) and
`include/chess_nn_eval.h` (the evaluator seam), with the drivers in
`src/selfplay.c` (PCR generation), `src/uci_mux.c` (UCI),
`src/gate_match.c` and `src/sf_gauntlet.c` (eval).
