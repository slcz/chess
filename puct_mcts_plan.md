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
> flagged nodes when they need an eval (virtual visit +1 N on **every path
> node at descent** — intermediates, pending leaves, terminal arrivals;
> W only at delivery, ply-wall/OOM deaths retract their increments).
> Co-sleepers on one leaf **share the pending eval row** (no duplicate
> requests; one result delivers every sleeper). A dedicated IO thread runs the evaluator
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
- Virtual visit: the descent adds +1 N to **every node on its path**
  (intermediate nodes, the pending leaf, terminal arrivals), W untouched —
  pending leaves read Q = 0, and the shrinking 1 + N in U steers later
  walkers away at *every* branch point, not just the leaf's parent
  (measured on the 42k seed, n=160, 8 walkers: leaf-only increments give
  ~53% duplicate parks; full-path increments ~20%). Deliver applies real W
  along the path; deaths (ply wall, failed expand) retract. Optional
  virtual-loss knob remains the stronger spread if ~20% ever matters.
- `MAX_REQ` per game defaults to **8** — small on purpose (chess has ~35
  legal moves; large in-flight counts force descents into bad moves). Batch
  scale comes from games, not per-game requests.
- Evaluator: stub (CI) or real net via the bench transport.
- Measurement: pooling correctness (batch fill, no lost requests, duplicate
  rate), pos/s at single-thread-achievable batches, per-request CPU cost
  (descent / encode / deliver). Peak 8192-batch throughput is NOT measurable
  here — one thread is CPU-bound far below GPU peak; that moves to Phase 3.

## Phase 3 — multi-game batching + dual-bank GPU pipeline

> **Design agreed 2026-10-05** (discussion with user; supersedes the
> one-paragraph sketches below). Three entities: a **walker thread pool**
> (~nproc threads) multiplexing games; **one io thread** that is the sole
> CUDA/Python enqueue-er; the **GPU** running async compute enqueued by io.
> Implementation order (one mechanism per step, no rewrites):
>
> - **3a — one thread, many games.** Current code is one game with 8
>   walkers. First step: a single thread multiplexes G games; staging
>   moves out of `PuctGame` (it embeds ~400 KB of `planes_buf`/`mask_buf`/
>   `logp_buf` per game today) into **one shared landing zone**; eval is
>   called synchronously on the walking thread over the cross-game batch
>   when no game can progress. End-state structures (landing zone, row
>   states) are built here.
> - **3b — one walker thread + dual banks + io thread.** Two landing-zone/
>   output-zone pairs (banks A/B). The single walker thread fills the
>   active bank while io runs the strict-lockstep enqueue below; eval no
>   longer blocks the walking thread.
> - **3c — N walker threads.** Pool threads multiplex games on the same
>   structures; only the bank accounting and row allocation become atomic.
>
> Scale target: up to 2048 games (1024 per bank × 2 banks), each with a
> **4 MB node arena** (~87k nodes at 48 B/node; self-play trees fit), game
> count is a knob. Per game up to 8 in-flight evals — batch scale comes
> from games, not per-game requests. Rows: planes 74,880 B + mask 4,240 B
> in, logp 16,960 B + wdl 12 B out ≈ 96 KB/row → up to 8192 rows/bank
> (1024 × 8) ≈ 790 MB pinned per bank.
>
> > **Status — 3a done (2026-10-06, reworked after review):** staging moved
> > out of `PuctGame` into **one global landing zone** per system (file-
> > static in puct.c): SoA rows + `used` counter + **the one evaluator**
> > (`puct_set_eval(fn, ctx)`, no game/zone argument) + **the one io
> > thread** (same strict ping-pong protocol, created once at
> > `puct_zone_init(rows)`). First rejected design (a `PuctZone` object per
> > mux with per-zone eval fn and per-zone thread) was rejected by the user:
> > one zone, one evaluator, one io thread per system, ever. `puct_think`
> > split into `puct_walk` / `puct_deliver` (wrapper preserved). New
> > `PuctMux` implements the user's grow-on-sleep policy: games start one
> > at a time when every live game's walkers are parked, up to `games_max`,
> > batch capped at `rows_max` rows (sizes the global zone); one eval per
> > step; the driver plays moves and recycles finished games in place via
> > `puct_game_reset` (fresh startpos, same block/struct). Walk order
> > rotates so a late game cannot starve when the zone fills early.
> > `PuctGame` shed its embedded staging (~400 KB/game) and all staging/
> > thread fields. `puct_attach_nn(ckpt, device, rows)` sizes the global
> > zone and installs the pinned tensors + eval fn; the per-game uci/probe
> > re-attach calls were dropped (evaluator is process-global now). Tests:
> > mux parity vs single-game think (bit-exact), zone-pressure invariants
> > (root visits == inherited + parks), recycling + game/rows quota drive.
> > `make check` / `check-asan` / TSan / real-net MPS smoke clean.
> > Deferred: run-level quota lives in the future self-play driver (2-line
> > condition over `puct_mux_rows` + finished count); spawn callback for
> > non-startpos games when self-play needs seed positions; 3b becomes
> > "dual banks" = a second row buffer, not a second zone object/thread.
>
> > **Follow-up (2026-10-06, user-driven): expand-at-park.** A walker that
> > reaches a childless leaf now expands it on the spot (uniform placeholder
> > priors — softmax of zero logp) and stages its row; delivery only fills
> > in the real priors (per-child policy-slot gather; the parked walker's
> > scratch State still holds the leaf, and `chess_policy_slot` reads only
> > the side to move) and backs the value up. Consequences: co-sleeping /
> > shared eval rows disappear *by construction* (a parked leaf has children,
> > so later walkers descend through it — dup rate is structurally 0, every
> > sim extends the tree, and every batch row is a distinct position, which
> > is better for batching); `PuctWalker` sheds `ml` + `slots[224]` (the
> > move list is a transient local, the walker shrinks ~1.5 KB); delivery
> > can no longer fail (no allocation there), so `undo_path` at delivery and
> > the SLEEPING probe guard are gone; `dups` telemetry removed from
> > `puct_park_stats` and the match STATS line. Search dynamics change in
> > the pending window (descents now go through freshly-expanded uniform
> > nodes instead of parking on them) — eval-round strength numbers predate
> > this and should be re-checked at the next round.
>
> > **Status — 3b step 2 done (2026-10-06): double buffer.** The landing
> > zone now has TWO banks (rows per bank = the `rows_max` knob, clamped to
> > `PUCT_MAX_ROWS` = 8192, the plan's scale target); row storage is
> > permanent static BSS (~840 MB virtual, paged in only for used rows —
> > nothing in the zone allocates; the nn seam points a bank's pointers at
> > pinned tensors instead); row ids
> > are global across banks (bank = row / cap). Banks move through an
> > explicit state machine FREE → QUEUED → EVAL → READY → FREE under the
> > zone mutex (the EVAL state matters: without it a bank being evaluated
> > looks free and the waker stages into it mid-flight — found via the mux
> > pressure test). `zone_eval` (submit+block) is replaced by `zone_pump`:
> > ship the active bank when stuck and keep walking in the other bank;
> > reclaim + deliver an in-flight bank only when nothing can stage. Eval
> > no longer blocks the walking thread. Still one io thread, one
> > evaluator; `PuctEvalFn` gains a `bank` argument and `nn_worker` keeps
> > one pinned tensor set per bank (`forward(bank, n)`). Driver contract
> > unchanged: `puct_mux_step` ends fully delivered. Two latent bugs fixed
> > en route: (1) the mux re-walked completed games mid-step (resetting
> > per-think stats and over-searching un-played roots) — a completed think
> > now stays done until the driver plays, keyed by position; `puct_think`
> > forces a fresh budget per call. (2) `puct_mux_rows` accounting was
> > dropped in the first double-buffer draft. Measurement (`test_puct
> > benchmux`, pure CPU + dummy eval): **~520K rows/s at 64 games /
> > 64-row banks, ~490K rows/s at 256 games / 256-row banks** — 13× the
> > 37.3K pos/s the GPU needs, so a single walker thread sustains the GPU
> > with room to spare. All gates green (check / asan / tsan / MPS smoke).
>
> > **Status — 3b step 2b done (2026-10-06): async device pipeline.** The
> > io thread no longer blocks inside the forward. PuctEvalFn became a
> > three-phase protocol (PUSH = H2D + async compute launch, PULL = D2H
> > enqueue gated on the forward's event, WAIT = block on the D2H event —
> > the only GPU-blocking op), and nn_worker grew the machinery proven in
> > benchmarks/infer_worker_db.py: three CUDA streams (s_h2d/s_compute/
> > s_d2h), per-bank device input/output buffers and fp32 conversion
> > targets, per-bank events. Bank states: FREE → QUEUED → IN_QUEUE →
> > PULLED → READY → FREE. The io round pushes every queued bank, then
> > pulls every pushed bank — the copy FIFO runs H2D(A) H2D(B) D2H(A)
> > D2H(B), so no H2D head-of-line-blocks behind a D2H and the two
> > forwards run back to back. Hazard the bench doesn't have: the search's
> > waker can be starved (all games parked on one bank), so a pushed bank
> > whose peer never fills would never be pulled — zone_wait flags such a
> > bank "flush" and the io thread pulls it immediately (steady state is
> > unaffected). Synchronous backends (MPS, dummy, scripted tests) do the
> > whole forward at PUSH; PULL/WAIT are no-ops. Still one io thread, one
> > evaluator; the waker never enters Python. TSan caught deliver_bank's
> > state transition racing the now-asynchronous io_pending read — the
> > bank's return to FREE is mutex-guarded now. Device-side verification
> > (back-to-back forwards on a CUDA box) is the first salad.com task.
>
> ### io thread — strict lockstep (user's design decision)
>
> Enqueue order per round: **H2D A → launch compute A → H2D B → launch
> compute B → D2H A → D2H B**. This *is* the push-before-pull discipline
> the bench measured (22.9k → 37.3k pos/s): both computes' inputs are in
> the copy queue before any results drain, so the copy-queue FIFO never
> head-of-line-blocks H2D B behind D2H A. Rationale: dual bank implies
> interleaving A→B→A→B; while the GPU works on A, the priority is getting
> H2D B ready, not draining A's results. With ~1024 games per bank, bank
> fill rates are symmetric by the law of large numbers, and the earlier
> benchmark showed 8 CPU threads out-produce the GPU — so fill + H2D stays
> ahead of compute and lockstep never starves the GPU.
>
> - Each D2H is **pre-wired** with `cudaStreamWaitEvent` on its
>   compute-done event: D2H starts by itself the instant compute finishes
>   — no polling, no gap. (CUDA handoff mechanisms considered: blocking
>   `cudaEventSynchronize`, polling `cudaEventQuery`, GPU-side
>   `cudaStreamWaitEvent`, `cudaLaunchHostFunc` push. Chosen: pre-wired
>   D2H + one blocking event wait per round.)
> - io then blocks on the D2H-done event before waking that bank's parked
>   walkers — safe because both computes are already queued when io sleeps.
> - **Bubble counter** instrumentation: compute B enqueued time − compute A
>   finished time. Nonzero ⇒ fill lost the race ⇒ revisit (timeout tuning
>   first, per-bank FIFO ordering as documented fallback).
> - **Partial batches:** bank not full within a timeout ⇒ ship half-full.
>
> ### Other protocol points
>
> - Row state machine FREE → FULL → INFLIGHT → DONE → FREE, atomics only
>   (D7), row header on its own cache line.
> - Compaction while requests are in flight: per-game in-flight list (row
>   ids + leaf handles), remapped during the game's own block swap
>   (generalizes today's `sleep_remap`).
> - Inference stays behind `nn_eval.c`/`nn_worker.forward` (io thread is
>   the only Python entrant, GIL uncontended); `nn_worker` gains the
>   two-slot/three-stream/event machinery proven in
>   `benchmarks/infer_worker_db.py`. Per-batch Python dispatch (~50–100 µs)
>   is ~0.2% of a ~50 ms round at batch 8192 — negligible.
>
> Measurement targets unchanged: pos/s vs thread count and games, GPU idle
> fraction, `DB_TIMING`, games/hour. Record dump format fixed at 3c:
> (position, logits, value) per step; trainer applies z.

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

## Eval round 1 (2026-10-05) — 42k seed behind the search, first calibration

Tools: `engine/src/{match_puct,uci_puct,puct_probe}.c` +
`scripts/{sweep_mcts,gauntlet_sf,diversity_check,n_sweep_agreement}.py`.
All runs on the Mac (MPS, 4 parallel processes, ~1.5–2k sims/s aggregate).
Full local write-up: `doc/mcts_eval_round1.md` (doc/ is untracked).

- **Strength sweep, MCTS(n) vs raw policy** (32 games/n, both sides sample
  first 30 plys, draws ½): n=32 0.641, 64 0.734, 128 0.906, 256 0.953 —
  monotone; raw baseline = think(1) + argmax prior.
- **Node-limited Stockfish** (sims=256, 8 games/level): 0.688 / 0.625 /
  0.312 / 0.062 at SF 30 / 100 / 300 / 1000 nodes → crossover ~100–300 SF
  nodes/move ≈ rough 1200–1400 Elo territory for the 42k seed.
- **Search convergence** (79 ply-stratified self-play positions, fresh tree
  per level): raw policy agrees with n=512 on only 58%; adjacent agreement
  A(n−32,n) crosses ~0.95 at n≈160 and plateaus. Working operating point
  **n=160–256**; cost linear in n (553 sims/s single-game on MPS).
- **Diversity check**: 30-ply sampling gives 8/8 distinct 12-ply openings,
  ~4–5% first-30-ply pairwise agreement; sampling-off control is identical
  (the knob is the diversity source).
- **Pathologies**: WDL head happy to repeat → threefold draws from playable
  positions (fix in self-play later); engine not bit-reproducible run-to-run
  (walker scheduling → batch composition → MPS reduction order → late-game
  argmax flips in equal positions).
- **Next**: Phase 3 threads + landing zone for throughput; MCTS-selection
  checks; then self-play rounds at n=200, re-measuring the agreement curve
  per round (hypothesis: optimal n drifts upward as the net sharpens).

## Eval round 2 (2026-10-06) — rerun after the walker fix (87a1ac5)

Full battery rerun after full-path virtual visits + shared eval rows (dup
rate 53% → ~19–23%). Same seeds/protocol as round 1. Local write-up:
`doc/mcts_eval_round2.md` (doc/ untracked).

- **Sweep vs raw** (32 games/n): 0.734 / 0.906 / 0.906 / 0.969 at
  n=32/64/128/256 (round 1: 0.641 / 0.734 / 0.906 / 0.953). The bug hurt
  small n most — n=32 +0.09, n=64 +0.17; draws collapsed 11 → 5; mates
  came faster (13–25 ply mates at n=64–256).
- **SF gauntlet** (sims=256): 0.812 / 0.562 / 0.250 / 0.188 at SF
  30/100/300/1000 nodes (round 1: 0.688 / 0.625 / 0.312 / 0.062) —
  roughly even at SF 100 nodes now, scoring at 1000.
- **Convergence** (79 positions): adjacent agreement hits 0.975 at n≈192
  and holds 0.975–1.000 above; A(n,512) ≥ 0.95 by n≈288 (round 1: ~0.85
  at 256–288) — mid-n searches land closer to the deep answer. Raw-policy
  vs n=512: 0.582 → 0.633.
- **Operating point**: n≈192–256 (matches), 128 for cheap experiments,
  n=200 default for self-play generation — unchanged region, firmer.
