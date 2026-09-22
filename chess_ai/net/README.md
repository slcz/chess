# chess_ai/net/ — neural network

ResNet trunk (first run: 6–10 blocks × 64–128 filters) with policy + value (tanh) heads. Built in Phase 2. Sanity gate: the net memorizes ~100 fixed games' policy/value targets, and illegal moves are masked to zero probability in tests.
