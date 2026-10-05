# PUCT MCTS — implementation plan (branch `puct_mcts`)

Five phases, one at a time (pause after each, per AGENTS.md). Decisions D1–D8
at the bottom. Built fresh on this branch; nothing ported from main.

## Phase 1 — single-thread, single-game PUCT MCTS (C)

> **Status (this round):** implemented in `engine/include/chess_puct.h`,
> `engine/src/puct.c`, `engine/tests/test_puct.c`; wired into the Makefile
> (`check` / `check-asan`). Deviations from the text below, on purpose:
> block is **64 MB**; on fill the search **packs** (fresh mmap, BFS copy of
> the root's subtree with parent/first_child remap, old block munmap'd —
> dead subtrees dropped), OOM only if even a packed fresh block can't fit
> one expansion; evaluator is the **dummy** (NULL callback ⇒ uniform priors,
> v = 0) — rollout stub and real net come next. Gate items 3 (Python parity)
> and 7 (real-net smoke) are deferred with them; slot-map coverage is
> C-only known-answers + a mask fuzz for now. Dummy-eval speed:
> ~1.5M sims/s on M4.

Files: `engine/include/chess_puct.h`, `engine/src/puct.c`,
`engine/tests/test_puct.c`. Wired into `engine/Makefile` (search joins
libchess; new `test_puct` target in `check` / `check-asan`).

### Data structures

```c
typedef struct PuctNode {     /* 48 bytes; u32 handles, no pointers      */
    uint32_t parent;          /* arena idx; 0 = none (root's parent)     */
    uint32_t first_child;     /* children contiguous; 0 = childless      */
    uint32_t n_children;
    uint32_t visits;          /* N                                       */
    double   value_sum;       /* W, stored in the PARENT-mover's view    */
    float    prior;           /* P                                       */
    float    wdl[3];          /* terminal nodes only                     */
    uint16_t move;            /* move from the parent                    */
    uint8_t  flags;           /* terminal kind / pending (Phase 2)       */
    uint8_t  pad;
} PuctNode;
```

- One opaque `PuctGame` per game: `State` + `NnHistory` ring + move log +
  search state. One fixed **8 MB block per game** (knob), bump-allocated.
  Slot 0 is a sentinel; the root lives at slot 1 (`parent = 0`).
- Children are created **eagerly at deliver**: one `gen_legal` + one
  contiguous bump of all children. An unvisited child has N = 0, Q = 0 and
  is ranked by prior through the U term.
- The tree **persists across moves**: the played child becomes the new root
  and its subtree is kept, no renormalization (W is parent-view). When the
  block fills: allocate a new block, copy the active subtree (remap
  `parent` / `first_child`), free the old. If even a fresh block cannot fit
  one move's search: give up the game, increment the OOM counter.
- Size check: one sim expands ≤ 1 leaf → n × (1 + n_legal) nodes worst
  case. n = 500 → 840 KB average, 5.2 MB worst < 8 MB.

### Algorithm

```
think(g, sims):
    if root has no children: evaluate(root)
    repeat sims times:
        leaf = descend from root by PUCT, making moves as we go:
            score(c) = Q(c) + c_puct · P(c) · √N(p) / (1 + N(c))
            Q = 0 while N = 0; c_puct default 1.25
            stop at a terminal node (back up its WDL, no eval needed)
            or at a child with N = 0
        evaluate(leaf)                # planes + mask in, logp + wdl out
        deliver(leaf):
            create children; prior_i = softmax over logp[slot(move_i)]
            v = wdl[0] − wdl[2]
        backup: for (i = leaf; i != 0; i = parent(i)) {
                    W(i) += v_signed;  N(i) += 1;  v_signed = −v_signed
                }                         # starts at −v (leaf's parent view)
        rewind to root (unmake + history drop), asserted
    pick move: argmax visits (τ-sampling knob for self-play later)
```

### Eval seam

```c
void puct_request_planes(g, float out[7488]);   /* = nn_encode            */
void puct_request_mask(g, uint8_t out[4240]);   /* gen_legal + slot map   */
int  chess_policy_slot(const State *s, Move m); /* move → 4240 policy slot;
                                                   the only new mapping   */

typedef void (*PuctEvalFn)(void *ctx, int n,
                           const float *planes,   /* n × 7488        */
                           const uint8_t *masks,  /* n × 4240        */
                           float *logp,           /* n × 4240 out    */
                           float *wdl);           /* n × 3 out       */
```

- Requests are built entirely C-side (planes + mask). Python only runs the
  forward. Deliver (gather logp by slot, max-subtracted softmax, backup) is
  all-C.
- Stub evaluator `puct_eval_rollout`: uniform priors (logp 0 at legal
  slots); value = average WDL over `n_rollout` (default 1000, knob) random
  playouts, Zobrist-seeded, ~600-ply cap with material adjudication.
  Tens of ms per eval; CI drops the count or uses a scripted evaluator.
- Real net: embedded Python (the proven `infer_worker` pattern), RoPE seed
  checkpoint, MPS (Mac) / CUDA (salad) device flag.

### Tests / gate

1. Mate-in-1/2 pins (uniform priors, search's own terminal detection);
   terminal nodes never request eval.
2. Bookkeeping: priors sum to 1; W/N consistency on a hand-built tree.
3. Python parity: pure-Python PUCT via cffi, scripted evaluator → exact
   visit vectors.
4. Memory lifecycle: compaction remaps indices correctly; subtree inherited
   intact into the next think; OOM give-up path (counter, clean abort).
5. Slot-map fuzz: `chess_policy_slot` / mask == `features.py` mapping.
6. `make check` + `check-asan`; stub-eval nps on M4.
7. Real-net smoke (MPS): seed ckpt loads, legal sane move at low sims, demo
   game vs greedy-material.

> **Status (this round):** implemented — 8 walkers (knob `puct_set_walkers`)
> descend one game's tree concurrently on the waker thread, parking on
> flagged nodes when they need an eval (virtual visit +1 N at enqueue; W
> only at delivery; parked indices live in the walker structs, and
> compaction linearly scans the walkers for flagged nodes). A dedicated IO thread runs the evaluator
> (dummy for now) over the SoA staging batch; strict mutex/condvar ping-pong,
> no data locks, tree touched only by the waker. Walker State is scratch —
> copied from the game root per sim, never rewound. Batch fires when all
> walkers are asleep; end-of-think drains in-flight rows (sims overshoot by
> < W). TSan-clean. M4 dummy-eval: 182k/271k/351k/394k sims/s at
> 1/2/4/8 walkers. Multi-game multiplexing stays Phase 3.

## Phase 2 — request pooling + virtual visits (one thread, many games)

- One thread multiplexes G games. `puct_walk(g, max_reqs)` enqueues up to
  `MAX_REQ` leaf evals per game and returns; the driver fills one combined
  batch, evals, delivers row by row.
- Virtual visit: enqueue adds +1 to the leaf's N (marked pending), W
  untouched — pending leaves read Q = 0, and the shrinking 1 + N in U
  spreads the batch's descents over siblings. Deliver expands, applies real
  W per row, backs up. Optional virtual-loss knob.
- `MAX_REQ` per game defaults to **8** — small on purpose (chess has ~35
  legal moves; large in-flight counts force descents into bad moves). Batch
  scale comes from games, not per-game requests.
- Evaluator: stub (CI) or real net via the bench transport.
- Measurement: pooling correctness (batch fill, no lost requests, duplicate
  rate), pos/s at single-thread-achievable batches, per-request CPU cost
  (descent / encode / deliver). Peak 8192-batch throughput is NOT measurable
  here — one thread is CPU-bound far below GPU peak; that moves to Phase 3.

## Phase 3 — threads

- N worker threads replicate the Phase 2 worker: private games, private
  memory, no sharing.
- **One shared pinned landing zone**: B = 8192 request rows (planes 34 KB +
  mask + small metadata each) plus a mirrored output zone (17 KB/row).
  Worker: `atomic_fetch_add` a row, fill, mark FULL. **Collector thread**
  (the only Python entrant): dump the filled prefix H2D → forward → D2H →
  flag rows DONE. Worker: read result, deliver, mark row FREE.
- Atomics only: game counter (hands out `game_index` = per-game seed
  material; stops at the game budget), row allocation, row state (own cache
  lines), per-thread record files. No barrier, no condvars, no locks.
- Batch cut: fire when the zone fills or on timeout; dump the filled
  prefix, never the whole zone at low fill. Full zone = backpressure
  (block on a free row, or switch to another of the thread's games).
- Compaction while requests are in flight: each game keeps an in-flight
  list (row ids + leaf handles) and remaps handles during its own block
  swap.
- Measurement: pos/s vs thread count and games-per-thread — 1024 games ×
  8 requests = 8192 = the peak batch. GPU idle fraction, paused time,
  games/hour. Record dump format fixed here: (position, logits, value) per
  step; the trainer applies the game result z.

## Phase 4 — device-side double buffer

- Two landing-zone / output-zone pairs. Workers fill the idle zone's cursor
  while the collector dumps and computes the other; a zone is reusable once
  every row is consumed.
- Copy-FIFO order: enqueue batch k+1's H2D before batch k's D2H (the bench's
  push-before-pull discipline).
- Measurement: pos/s delta vs Phase 3, GPU idle fraction, GPU timeline
  (`DB_TIMING`).

## Phase 5 — request caching: measure, likely skip

- Subtree keeping already recycles the dominant reuse. The NN input embeds
  the 8-ply history, so position-keyed caching is wrong, and a full-input
  key gets no hits on transpositions. Remaining repeats are thin.
- Measurement only (free): the landing zone counts duplicate leaf requests.
- If material: in-flight dedupe at enqueue (same game + leaf → merge rows,
  one eval row). Build the LRU only if telemetry justifies: one global map,
  one lock, full-input hash key, sparse legal-only entries (~1 KB) + 12 B
  WDL.

## Decisions

| #  | Decision                                                                                       |
|----|------------------------------------------------------------------------------------------------|
| D1 | Q stored in the parent-mover's view; selection is sign-free, backup flips sign per ply        |
| D2 | Real net wired in Phase 1 (embedded Python, MPS/CUDA); stub evaluators stay the CI path        |
| D4 | Requests built entirely C-side (planes + 4240 mask + slot map); Python does the forward only   |
| D6 | Fixed structure (eager children), fixed 8 MB block per game, subtree keeping, compaction-by-copy, OOM give-up counter |
| D7 | Atomics-only concurrency: game counter, landing zone + collector, private memory; no locks     |
| D8 | Request cache: telemetry-first; in-flight dedupe as fallback; LRU only if justified            |
