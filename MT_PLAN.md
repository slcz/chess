# Plan: 2-thread self-play (branch `mt2`, on top of `act`)

Scope: parallelise the **walk**. The evaluator stays on the one io thread;
nothing about the network, the shard format or the trainer changes.

Why: with SEARCH-style selection the walk is now a large share of self-play
time, and the walk is the only serial part left. It is also what limits
`ROWS`: bigger banks do not pay off while one thread cannot fill them.

Branch base: `act` (PUCT + noise + LEARN), not `search`, so ACT and
multi-threading can be evaluated together. The threading work touches the
zone/mux; `search` touches `select_child`/`backup`, so the two stay
separable if only one is kept.

## Design

**Games are owned by threads.** A walker mutates its game's tree, state and
history, so two threads must never walk the same `PuctGame`. Thread *t* owns
games `t, t+N, t+2N, …` (round-robin, so budgets are balanced).

**Rows are allocated atomically.** This is the piece that replaces both a
per-row lock and per-thread row partitions:

    off = atomic_fetch_add(&Z.banks[b].used, 1);
    if (off >= z_rows) { park; }        /* bank full: this thread is done */

A thread stages one row per simulation, so this is a few thousand grabs per
second — no contention worth designing around. The allocation also removes
the barrier: whoever arrives keeps walking until the bank is full, and the
last useful one does not wait for a slow peer.

Caveat: a losing race can reserve one slot past `z_rows`. Treat "used >=
z_rows" as full and accept up to N-1 wasted slots per bank.

**Ship rule.** The io thread ships when the bank is full *or* every walker is
parked (no thread can stage anything, i.e. all sims for the step are in
flight). Both are already conditions `zone_pump` knows; the change is that
"all games parked" is evaluated across all threads, not one.

**One mutex for the zone metadata**, held only around state transitions
(submit/publish/deliver), never around the walk. `Z.mu` stays.

**Step structure.** `puct_mux_step` becomes: each thread walks its slice until
the budget for the step is met or the bank fills → barrier (`pthread_barrier`)
→ the existing pump/io handoff → step ends fully delivered, as today.

**max_games** is a `_Atomic unsigned` counter; a thread stops when it takes the
last game. Trivial, as you said.

## Determinism: per-game RNG (do this first)

`engine/src/rng.c` holds one process-wide `static uint64_t state`. Any
threading makes that a data race, and it also makes results depend on
scheduling. Replace with a stream per game (`rng_seed_for(seed ^ game_index)`),
used by `select_child`'s sampler.

Payoff: **the transcript becomes independent of thread count**, which is both
the reproducibility property we want for A/B rounds and the test that proves
no shared mutable state was missed. This lands first, on its own, and is
verified with 1 thread before any thread is created.

## Concurrency audit (what is shared)

| object | owner | note |
|---|---|---|
| `PuctGame` (nodes, state, hist, walkers) | one thread | never shared |
| `Z.banks[]` rows + state | shared, `Z.mu` + atomic `used` | staging is atomic |
| `Z.eval`/`Z.eval_ctx` | io thread only | unchanged |
| RNG | per game | replaces the global stream |
| `Slots`/shard writer | main thread after each game | single writer today |

## Work breakdown

1. **Per-game RNG** (`rng.c`, `puct.c` root noise) — done. Note the gate
   changed: splitting one process-wide stream into per-game streams
   *necessarily* changes the sequence, so "bit-identical to the old binary"
   is unachievable. The gates are determinism within a build, then
   invariance to thread count.
2. **Atomic row allocator** (`puct.c`: `zone_submit`/`walker_step` staging) —
   ~40 lines. Verify: engine suite, then 1-thread equivalence again.
3. **Thread pool in the mux** (`puct.c`: `puct_mux_new(games, rows, threads)`,
   new `puct_mux_step` with the barrier) — ~120 lines, plus `max_games` atomic.
4. **`selfplay.c`**: expose threads (argv or env, TBD), keep the shard writer
   single-threaded.
5. **Validation + measurement** (below).

## Validation gates (in this order)

1. 1-thread `selfplay` transcript identical to `act` after steps 1–2.
2. **2 threads vs 1 thread, same seed → identical transcript.** This is the
   gate that proves tree isolation and RNG correctness; if it fails, the
   design is wrong, not the timing.
3. Engine suite + wasm parity (both must stay green: the wasm build has no
   threads, so it exercises the 1-thread path only).
4. Rate: 1 vs 2 threads at production sims on the box, same games, and
   `oom`/`compact` from SUMMARY.
5. Only then a GPU round.

## Risks / rollback

- The bank state machine is the heart of the pipeline; a race here corrupts
  training data silently rather than crashing. Mitigation is gate 2, not code
  review.
- If threading proves unstable, the branch is self-contained: `act` ships
  without it, and nothing in the shard format changed.
- Thread count is a startup parameter, not a data-format change, so shards
  from 1- and 2-thread runs are interchangeable.

## Decisions (reviewed)

1. **Threads are a parameter, default 2.** `puct_mux_new(games, rows, threads)`
   taking the count; `selfplay` exposes it as an argv slot. Not a compile-time
   switch, so the 1-thread path stays testable from the same binary.
2. **The io thread does not walk** — it never did. Its job stays exactly
   PUSH/PULL/deliver on the bank; the only change is that the "everything is
   parked" condition is evaluated across all walker threads.
3. **`ROWS` 2048 -> 4096 is a separate follow-up** (one line, once threading
   lands and the rate is known). Not part of this branch.

## Watch items

- `block_mb` per game at `pool=1024` is 4 GB of trees; with 2 threads the same
  total, but compaction behaviour may shift. Watch `oom`/`compact` in SUMMARY.
- Thread count is a startup parameter, not a data-format change: shards from
  1- and 2-thread runs stay interchangeable.
