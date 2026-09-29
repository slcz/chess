# Parallel Gumbel self-play — design review & pseudo-code (v4, FOR REVIEW)

**Status:** proposal for alignment. No C code yet (per AGENTS.md: align first).
Round log: **R2** worker-local buffers + all-workers barrier chosen (R1–R3 fixes);
**R3** B=256, lock-free submission path confirmed, resignation added (R4), cache demoted
(F4), eval seeding added (R5); **R4** evaluator embeds **CPython** (F8 closed), resign
default 5% confirmed, eval seeding confirmed + variance telemetry, new **R6 verification
ladder**, new **R7 NN inference interface** (`policy_logp` is already the right tensor);
**R5 (2026-09-27, after the 3.2a reference landed): the wave discipline**
— descents scheduled in per-round waves makes batched ≡ sequential exact *by
construction* (L1 upgraded from aspiration to proof), which in turn deletes the
PASS-resume mode and the round-1 fast path (F3 rewritten).
**R6 (2026-09-27, user decision): the Python reference simplified to a pure
oracle** — synchronous size-1 evals at the expansion point, no request queues,
no park/resume, no Python protocol layer (3.3a dropped; there is no Python-side
batching to share). `gumbel.py` is now literally PLAN's pseudocode plus one
narrow `Evaluator.evaluate` call. Consequences: the L1 batch-invariance test is
retired (there is no Python batching); the exact-diff gate is **L2 only**, with
the pinned order = PLAN-literal **candidate-major** rounds (in sequential
execution, wave order and candidate-major order give identical trees — sibling
candidate subtrees are disjoint — so the pin is free of subtleties). The wave
discipline survives purely as the C parallel engine's batching structure (§4.4).
Reference: `chess_ai/selfplay/gumbel.py` + `rng.py`, tests
`tests/test_gumbel_ref.py` (green; full repo suite green).

**R7 (2026-09-27, user review round — the simplification pass):** standing rule —
every line in the reference serves the algorithm or the C-comparison contract, or
it goes. Changes: (a) no k==1 shortcut — forced positions flow through the
general path (m=1, no halving rounds, π′ one-hot, outlook = v̂ via Eq. 10);
(b) training samples the played move ∝ π′ at every ply (`sample_plies=None`) —
deterministic survivor-play reserved for eval (R5); (c) the handoff is threaded
as an explicit parameter through `play()` — no game-level field, nothing can go
stale; (d) resignation is a single immediate threshold check — the sustained
streak and the no-resign telemetry games are deleted; RNG contract loses the
game-creation coin (first draw is now move 1's Gumbels); (e) `evals` and
`root_eval` telemetry fields removed from MoveRecord — records carry only
training-relevant data (position, π′, played move); evaluator-call accounting,
where a test needs it, lives in test-local evaluator wrappers.

**Scope:** the C-orchestrated parallel Gumbel self-play engine (Phase 3.3, amended)
plus the sequential Python reference. The *search math* is unchanged — the agreed Full
Gumbel of PLAN.md Phase 3 (Gumbel top-m + sequential halving at root, Eq. 14 below
root, σ = (C_VISIT + max_b N(b))·C_SCALE·q, no Dirichlet, no temperature, π′ target).

---

## 1. The scheme, restated

1. N threads: N−1 MCTS workers + 1 evaluator (PyTorch via embedded CPython).
   Reference sizing: 8 workers × B = 256 → epoch batch 2048.
2. A game lives on one thread for its whole life (tree, working State, records).
3. Workers advance games round-robin; blocked games are set aside; new games are
   started freely (atomic `games_started` < budget is the only guard).
4. Double request buffer per worker, B = 256/slot; seal on full or no-work.
5. Evaluator wakes when every *active* worker has ≥1 sealed buffer; one batched
   forward pass; per-worker result vectors back.
6. Termination at the game budget (completed counter; zero overshoot via reservation).
7. Resignation: root-level, immediate — outlook ≤ threshold (round 7: no
   streak, no telemetry games).

## 2. Verdict

Unchanged: the architecture concentrates all synchronization at the NN batch boundary;
rounds 2–4 progressively *simplified* it (no shared queue, no per-search dedup, no
default cache, near-zero locking, shorter games). All required fixes from earlier
rounds carry over. Nothing in this round reopened settled ground.

---

## 3. Review findings

### R1 (settled): seal rule and sizing at B = 256

Unbounded game pickup + no-work seal leg (startup counterexample in v2). Sizing law:
G\* ≈ B/r, r ≈ (1.3–2)·n/⌈log₂m⌉ ≈ 5–10 → ~32 games/worker in flight (~8 MB total);
startup burst ≈ B/(m+1) ≈ 15 games/worker; epoch batch 2048; straggler window ~1–2 ms
vs ~4–8 ms forward pass (measured risk; knobs: smaller B, appendix queue).

### R2 (settled): per-thread queues — submission path needs no locks

No shared structure is touched per request. The per-epoch handshake (post, swap-all,
deliver, broadcast) is ~2(N−1)+2 uncontended acquisitions per 2048 requests.
Per-worker sealed flags (never a counting semaphore — double-buffering aliases
double posts); C11-atomics option only if profiling ever cares.

### R3 (settled): end-of-run — exit, don't post

Idle workers decrement `active_workers` and exit; evaluator predicate tracks the
shrinking denominator; drains and exits at zero. Zero budget overshoot via the
`games_started` reservation.

### F3 (rewritten rounds 5–6): descent scheduling — waves for C batching, candidate-major for the reference

Within a round, the `plays` descents per candidate can be scheduled two ways that are
**result-identical in sequential execution** (sibling candidate subtrees are disjoint —
children are addressed by (parent, move slot); no transposition merging — so descents of
different candidates never read each other's nodes, and a same-candidate descent always
sees its predecessor's results resolved):

- **Candidate-major** (`for a in A: plays × descend(a)`) — PLAN's pseudocode, literal.
  This is the **Python reference's order** (`gumbel.py`) and the order the C engine's
  *sequential mode* must pin for the L2 exact diff.
- **Wave-ordered** (the w-th play of every a in A before any (w+1)-th play) — the
  **C parallel engine's batching structure**: a wave's requests batch together; the wave
  barrier (all wave-w descents backed up before wave w+1 starts) is what keeps every
  descent's decisions on *resolved* state only, so a parallel run's divergence from the
  sequential reference is bounded to fp accumulation order in shared-path backups.

Under either order, a descent parks only on its own expansion leaf (BACKUP mode, one
waiter per node); the PASS-resume machinery of rounds 1–4 is unreachable — **deleted**
(encountering a foreign PENDING node in C is a protocol bug and must fail loudly). The
round-1 "fast path" is likewise gone — wave-1 descents produce exactly the m
candidate-child requests naturally. The C engine (3.3b) needs no PASS resume, no waiter
lists (a single parked-path slot per pending node), and no dedup beyond node existence.
The worker's deferred-on-full park (R1/§4.3) is unaffected: it parks whole descents when
both buffer slots are sealed, and is the only remaining park path besides the leaf.

### F4 (settled): cache = 1-entry handoff; LRU off by default; no dedup

Within-search dedup is provably a no-op under the correct key (position + last-8-ply
*sequence*; different move order ⇒ different sequence ⇒ genuinely different eval).
The played child's evaluated node is handed to the next search — zero hashing.
`eval_cache` flag off; measure hit rate in the smoke; delete if < 5%.

### R4 (round 4; simplified round 7): resignation — root-level, immediate

Trigger at `finalize_move`: mover's `max_a q(a)` over candidates ≤ −0.90 (≈5% win
prob) → resign immediately. Label: loss for resigner. **Round-7 simplification
(user): the sustained-2-turn streak AND the 1/16 no-resign telemetry games are
deleted** — one knob (`resign_value`) remains; if false resigns show up in the
games themselves, raise the threshold. (Round-4 form was −0.90 / 2 consecutive
turns / 1/16 telemetry.) Allowed in ladder/SF matches too.

### R5 (settled round 4 + telemetry): eval determinism & diversity

Seed phase: first 10 plies sampled ∝ π′ (per-game RNG seeded from match slot id);
measurement phase deterministic (argmax survivor, no sampling); each seed played with
colors reversed; 64 seeds × 2 = 128 games per ladder point. Training self-play needs
none of this (fresh Gumbels each move + opening sampling + per-game seeds).

**Variance monitoring (added round 4):** each eval run reports — per-seed outcome
matrix (64 × 2), overall win rate with a binomial CI, and a consistency check
splitting the seeds into halves/quartiles (win rates of the splits should agree within
the CI). A pathological seed (decided by move 10 both colors) then shows up as an
outlier row instead of noise. If CIs are routinely too wide to separate checkpoints,
grow the seed set before adding games per seed; SPRT gating is the later upgrade.

### R6 (new round 4): verification ladder — comparing C against the Python reference

The user's question: "at test time you compare the parallel MCTS with the Python
reference — you need to fix the seeds." Yes — and one honest caveat shapes the answer:
**the parallel engine is schedule-nondeterministic by design**. Even with every seed
fixed, two runs of the C engine differ, because (a) thread timing decides which
descents are in flight together — descents of the same round can meet at a shared
interior node and, without virtual loss, evolve the tree differently than any serial
order; and (b) `value_sum` accumulates in delivery order (fp addition is not
associative). This is inherent to tree-parallel MCTS, not a bug; forcing determinism
on the real parallel path would mean locks/virtual-loss we deliberately rejected.

So the comparison is a three-layer ladder — exact where determinism exists,
statistical where it cannot:

- **L2 — C engine in *sequential mode* vs Python reference (exact).** The C engine
  ships a test/CI mode: one worker, synchronous size-1 inline evaluator, CPU fp32,
  candidate-major rounds — the same pinned order as `gumbel.py` (F3). That path *is*
  deterministic, so C and Python must produce identical games end-to-end (moves, visit
  counts, π′ within 1e-6). This is the port-verification gate: it exercises the C
  search, encoder parity through the real path, and the logp plumbing. Bit-exactness is
  achievable because both sides run the same torch model on CPU at batch 1 and the same
  IEEE-double tree arithmetic in the same order. (The former L1 — Python-internal
  batch-invariance — was retired with the Python protocol layer: the reference no
  longer batches, per the round-6 simplification.)
- **L3 — C parallel engine (invariants + statistical + self-golden).**
  - *Hard invariants* (schedule-independent): π′ sums to 1 over A₀ and is exactly 0
    off legal; every candidate visited ≥ 1; total visits == scheduled descents per
    round; no node requested twice; visits/q ranges sane; records schema-valid;
    resign rules held.
  - *Statistical agreement with the reference:* same seeds, K parallel runs vs the
    reference — per-game move-agreement rate (root Gumbels are seeded, so candidate
    sets match; q estimates converge to the same values, so agreement should be
    high — measure and track it rather than assert a bar), plus telemetry bands
    (game length, resign rate, evals/move).
  - *Deterministic-interleave mode (self-golden):* the worker code is a resumable
    state machine (parks/phases), so one OS thread can drive all workers in fixed
    round-robin order — full "parallel" semantics (parks, in-flight rounds,
    batching) with a deterministic schedule. Bit-reproducible run-to-run → golden
    game hashes per release. This is the race detector of last resort: any change
    in results between normal runs and interleave runs points at a data race.

**RNG discipline (makes L2/L3 possible at all):** one generator per game,
**xoshiro256++ seeded via splitmix64(work_unit_seed, game_index)** — implemented once
in C and mirrored in the Python reference (~30 lines; no dependence on numpy RNG
internals). Consumption order is fully specified: Gumbels as −log(−log(u)) in
*legal-move enumeration order* (the reference drives rules through the same C engine
via FFI, so enumeration order is shared by construction); move sampling by inverse CDF
over candidates in enumeration order. Fixed seeds therefore reproduce every random
input exactly; only the schedule (L3) varies.

**Precision policy:** oracle diffs (L2) run the net on CPU fp32 — deterministic.
Production self-play may run bf16 on the GPU (Gumbel noise ≫ bf16 error; exploration
is stochastic anyway). Eval/ladder matches run fp32 for strength and stability.
Tree arithmetic is IEEE double on both sides everywhere.

### R7 (new round 4): NN inference interface — the net already exposes the right tensor

Gumbel needs the policy in **log space** at legal moves: z(a) = logp(a) + g(a).
`ChessTransformer.forward` already returns exactly that — `"policy_logp"` [B, 4240],
log-softmax over legal slots, illegal = −inf (its hardest-since-Phase-2 contract).
No head change is needed, and this is not luck but shift-invariance: raw logits and
log-probs differ by the constant log Z, which cancels in **every** Gumbel formula —
Gumbel-top-m sampling (softmax(logits+g) = softmax(logp+g) as distributions),
halving ranking (argmax of z + σ(q): constant shift drops out), and
π′ = softmax(logp + σ(completedQ)) (softmax is shift-invariant). σ(q) reaches ~±50
at high visits vs logp ∈ ~[−25, 0]; the relative prior/value weighting is unaffected
by the shift. **Convention: search stores and computes with logp everywhere; never
mix scales.** (Node field renamed `logits` → `logp` below for this reason.)

What each search-time consumer needs, and where it comes from:

```text
quantity            source (forward dict)          notes
policy logp (legal) "policy_logp"                  sparse-extracted by the evaluator
value v = pW − pL   "value"                        already computed by the net
(pW, pD, pL)        softmax("wdl")                 one line in the evaluator
aux reply logp      —                              training-only; skipped at inference
                                                       (call trunk+policy+value directly,
                                                        or eat the head cost; decide in code)
```

One real integration requirement surfaced by this: **`forward` needs the legal mask
[B, 4240] as input, but the evaluator does zero engine work (F7).** So the *worker*
computes the mask (it owns `node.legal`) and ships a **4240-bit legal bitset (530 B)
in each request** alongside the u8 planes; the evaluator builds the bool tensor by
unpacking bitsets. Request size stays ~8 KB; the evaluator remains pure tensor work:
unpack → stack → cast → `net()` → sparse-extract legal logp + WDL probs → deliver.

### F7 (settled): C translation notes

Arenas per worker, reset per search; nodes hold no State copy (path replay via
make/unmake on the game's working State + encoder ring); u8 planes encoded by the
worker at request time; results as per-worker vectors (evaluator never writes trees);
lock/condvar happens-before; per-game RNG per R6.

### F8 (closed round 4): evaluator ↔ PyTorch = embedded CPython

**Decided: CPython.** Only the evaluator thread holds the GIL, only around
`net(batch)`; `torch.set_num_threads(1)`; workers are pure C and never touch the
interpreter. Reuses `chess_ai/net/model.py`, checkpoint loading, and the fp32/bf16
device logic as-is. ONNX/TorchScript export stays a *later, measured* optimization
(export + parity fixture per checkpoint) if the interpreter handshake ever shows up
in profiling. → record in PLAN §11 as decided.

### F9 (PLAN amendments once aligned)

§3 self-play-infra row (C orchestration now, Python as sequential oracle only —
3.3a dropped by the round-6 simplification); subtasks 3.2a / 3.1 / 3.3b;
resignation moved from "deferred to Phase 5" into Phase 3 with R4 knobs;
F8 recorded as decided (D7 or a note under D6's boundary); R6/R7 noted in the
Phase 3 test plan.

---

## 4. The protocol — pseudo-code

### 4.0 Knobs

```text
n            sims per move (self-play 2–32; eval 100–200)       # PLAN 3.2
m_cap        16                                 # m = min(n, m_cap, #legal)
C_VISIT, C_SCALE, c_fpu         = 50, 1.0, 0
sample_plies                     = None (training: sample ∝ π′ every ply);
                                   eval seed phase sets 10 + deterministic argmax
B            = 256              # per buffer slot (user, R3)
resignV = −0.90; immediate resign — no streak, no telemetry (round 7)
eval: seedPlies = 10, deterministic remainder, color-paired seeds, 64×2 games
eval_cache   off                # measure in smoke before keeping
infer dtype  bf16 self-play · fp32 eval · fp32 CPU for L1/L2 oracles
seeds        work_unit_seed → splitmix64 → per-game xoshiro256++ (R6)
threads      N (workers N−1; reference 8) · budget (games_started atomic)
```

### 4.1 Shared state (per-EPOCH handshake only)

```text
lock; cv_eval; cv_workers (broadcast)
active_workers, posted_workers; games_started (atomic), game_counter
per worker w: sealed[0..2]: SealedBuf; results: [ReqResult];
               games, arena, rng, deferred (private)
SealedBuf = { requests: [Req] }
Req  = {game, node, planes(u8 7488), legal_bits(4240), slot}
ReqResult = {slot, sparse logp:[(slot,f32)], (pW,pD,pL)}
```

### 4.2 Node / descent

```text
Node:
  state   : NEW | PENDING | EVALUATED | TERMINAL
  legal   : [Move]              # movegen, once at expansion
  logp    : {slot → f32}        # sparse log-probs (R7) — log space everywhere
  wdl     : (pW,pD,pL);  v = pW − pL
  children: {slot → Node}       # by move slot; no key map (F4)
  visits, value_sum             # mover's view; q = value_sum/visits
  waiters : [DescentCtx]        # parked descent — at most ONE per node (F3, wave
                                # discipline: only the node's own expansion leaf parks)
DescentCtx:  game, path:[Node]  # BACKUP mode only; PASS deleted (round 5)
```

### 4.3 Worker thread (× N−1)

```text
worker_main(w):
  lock: active_workers += 1
  loop:
    while a slot is unsealed with room:
        progressed = False
        for g in w.games (round-robin): progressed |= game_step(g, cur_buf)
        run w.deferred first if nonempty
        if not progressed:
            if games_started.fetch_add(1) < budget:
                w.games += new_game(seed = splitmix64(unit_seed, game_index))
            else: break
    if cur_buf nonempty:
        w.sealed.append(seal(cur_buf))
        lock: if len(w.sealed) went 0→1: posted_workers += 1; cv_eval.notify()
    if w.games empty and w.sealed empty: exit_worker(w)
    lock: while w.results empty and not exit: cv_workers.wait(lock)
          results, w.results = w.results, []
    unlock
    for r in results:
        node.fill(r.logp, r.wdl)
        for ctx in node.waiters: resume(ctx)

def game_step(g, buf) -> bool:
  if g.finished: drop; return True
  if g.search is None:
      if terminal(g.pos): finalize_game(g); return True
      g.search = new_search(g)
      if g.handoff: g.search.root = g.handoff; g.handoff = None; return True
      root.state = PENDING; buf.append(encode_req(g, root)); return True
  match g.search.phase:
    AWAIT_ROOT: return False
    ROUND:      return run_one_descent(g.search, buf)
    ROUND_END:  halving(g.search); return True
    FINAL:      finalize_move(g); return True
```

### 4.4 Descent

```text
on root applied:                        # fresh eval or handoff node
  m = min(n, m_cap, len(legal))
  for a in legal-enumeration-order: g_a = −log(−log(u()))     # R6 order
  s.z = {a: logp[a] + g_a};  s.candidates = top_m(s.z)
  s.rounds = ceil(log2 m);  start_round(s)
start_round(s):
  plays = ceil(n / (s.rounds·|A|));  waves = plays   # wave discipline (F3)
  for w in 1..plays: for a in A (z-desc): one descent, forced_first=a
  # wave barrier: wave w+1 starts only after every wave-w descent backed up;
  # no round-1 fast path needed — wave-1 descents park at their candidate
  # children and produce exactly the m candidate-child requests themselves
  run_one_descent / descend / backup:  as v3 (path replay, make/unmake; park ONLY
  at the descent's own fresh leaf — a foreign PENDING node is an assertion;
  TERMINAL one-hot inline backup; visits/value_sum with sign flip — parent reads
  a child's q NEGATED (own-mover-view stats); s.pending −= 1; ROUND_END when
  drained; after halving to 1 survivor → FINAL (while |A|>1 loop semantics))
```

### 4.5 Selection & halving (PLAN math, verbatim)

```text
select_eq14(node):
  argmax over legal a of: logp[a] + σ(completedQ(a)) − N(a)/(1+ΣN)
    completedQ(a) = child.q if N(a)>0 else node.v − c_fpu
    σ(q) = (C_VISIT + maxN)·C_SCALE·q
halving(s): keep top ⌈|A|/2⌉ by z(a) + σ(q(a)); FINAL at 1 survivor
```

### 4.6 Move finalization, resignation, targets

```text
finalize_move(s):
  π′ = softmax over original A₀ of [logp(a) + σ(completedQ(a))]
  outlook = max over a in A₀ of q(a)               # R4 resign estimate
  if outlook ≤ resignV: resign, else:
  move = sample(π′) — training samples at EVERY ply (2026-09-27: the
         deterministic survivor is the eval-measurement move, R5); eval
         seed phases sample while ply < seedPlies, then argmax survivor
  g.record.append(pos, π′, move)
  g.handoff = node_of(move) if it has a child node else None
  g.pos.push(move); g.search = None; arena_reset(keeping handoff node)
finalize_game(g, resigned=None):
  z = outcome or {resigner: L, other: W}
  emit GameRecord(..., z) → shard writer; atomic game_counter += 1
k == 1 legal moves: NO special case (round 6) — the general path handles them:
m = 1, the halving loop never runs, π′ = softmax over the single candidate =
exactly 1.0, outlook = v̂(root) − c_fpu via the Eq. 10 fallback (so resignation
works uniformly), and the move costs one root eval — usually free via the
handoff (the forced position's node exists and is evaluated after a searched
move). RNG: the forced move draws its single Gumbel like any move.
```

### 4.7 Evaluator thread (× 1, embedded CPython)

```text
evaluator_main(net):
  while True:
    lock: wait posted_workers == active_workers (or active == 0); swap all sealed
    unlock
    mask  = unpack bool[B,4240] from legal bitsets          # R7
    x     = stack(u8 planes).to(dtype)
    with gil, no_grad: out = net(x, mask)                    # trunk+policy+value;
                                                             # aux skipped
    logp_sparse = gather out["policy_logp"] at legal slots
    wdl_p = softmax(out["wdl"]);  v not needed (C recomputes pW−pL cheaply)
    lock: for w: w.results += its slice; cv_workers.notify_all()
    unlock
```

---

## 5. Liveness walkthrough

Unchanged from v3: startup (start games until B seals ≈ B/(m+1) ≈ 15/worker), steady
state (G\* ≈ B/r ≈ 32 games/worker; zero-request progress is finite — no livelock),
deferred-when-both-slots-sealed, straggler ≤ ~1–2 ms (measure at B=256), end-of-run
(exit-don't-post, drain, zero overshoot), parks are single-waiter leaf parks only
(F3), tree logic never
leaves owners.

## 6. Verification & test plan (Python reference 3.2a, C engine 3.3b)

1. **Reference self-contract (green, `tests/test_gumbel_ref.py`):** same seed ⇒
   byte-identical records (moves, visit-derived π′, outcomes);
   backup sign-flipping; π′ well-formed over A₀; forced (k==1) positions flow
   through the general path (π′ one-hot, outlook = v̂ via Eq. 10); handoff makes
   every root after move 1 free; resignation trigger; mate-in-1 found.
2. **L2 (exact):** C engine sequential mode (1 worker, B=1, inline eval, CPU fp32)
   vs Python reference, same seeds — identical games end-to-end; π′ within 1e-6.
   Includes encoder parity through the real request path (planes + legal bitsets).
3. **L3 (invariants + statistical):** C parallel engine — invariant list (R6) on
   every game; move-agreement rate vs reference over K seeded runs (tracked metric);
   telemetry bands; **deterministic-interleave mode** with golden game hashes per
   release; diff normal-vs-interleave results to expose races.
4. **RNG discipline tests:** mirrored xoshiro produces identical streams in C and
   Python; same seed → same candidate sets regardless of schedule; per-game seeds
   unique; games_started ≤ budget.
5. **Resignation units:** immediate trigger at the threshold; sign-flip fixture
   (bad leaf, winning root) never triggers; z labels correct.
6. **Eval determinism/diversity + variance telemetry (R5):** same slot id replays
   the same game; 64 slots give distinct seed phases; color pairing covers both
   orientations; per-seed outcome matrix + split-half consistency reported.
7. **Properties:** mate-in-1/2 at n=16; forced positions produce π′ one-hot with
   outlook = v̂ (Eq. 10 fallback); π′ sums to 1 over A₀, zero
   off legal; top-m without replacement; survivor == argmax(z) at equal Q.
8. **Handoff:** root-after-move-1 costs zero evals; a forced move right after a
   searched move costs zero evals (the forced position's node IS the handoff).
9. **Tiny-game smoke (PLAN 3.5):** tic-tac-toe through this protocol → perfect play.
10. **Sizing telemetry:** in-flight ≈ B/r; straggler gap per epoch; evals/move.

## 7. Open questions (round 4 — one left + closing actions)

1. **Sims accounting (carried):** keep PLAN's `plays = ceil(n/(rounds·|A|))`
   (≈1.3–2× n descents/move, round-1 coverage guaranteed) and document — or switch
   to a floor-style ≈ n budget? Recommendation: keep ceil.
2. If no objections: amend PLAN §3 per F9, update the selfplay README, and start
   **3.2a — the sequential Python Gumbel reference** (search math + L1 tests first,
   then the batched protocol layer + resignation + eval seeding).

---

## 8. R8 (2026-09-28): implementation record — 3.3b landed

The engine is implemented: `engine/src/sp_engine.c` (+ `include/chess_sp.h`,
`include/chess_sp_eval.h`), the wave-concurrent game machine in
`engine/src/search.c` (`sp_game_step_wave` / `sp_deliver_node`, the §4.3
game_step verbatim), the embedded-CPython evaluator in
`engine/src/sp_evaluator.c` behind `chess_ai/selfplay/sp_evaluator.py`, and
the Python orchestration seam `chess_ai/selfplay/spengine.py` (built as
**libchess_sp** — the pure rules library stays Python-free; only the _sp
build links the interpreter headers, symbol-resolved at dlopen time).

**L3 landed stronger than planned.** The round-5/6 simplification has a
consequence the earlier rounds could not see: with a *batch-invariant*
evaluator, the wave-concurrent engine is **fully deterministic** — each
game's descent sequence is fixed by its scheduler; candidate subtrees are
disjoint; same-candidate descents are serial behind the wave barrier; and
the only delivery-order-dependent accumulation (the root's own
`value_sum`) is never read by selection. So R6's schedule-nondeterminism
(a)/(b) — written before the simplification — no longer applies, and the
ladder gains an exact rung: **L3a: parallel ≡ sequential, bit-for-bit**
(same seeds, fixture evaluator, varying worker counts and tiny buffers
that force stragglers and deferred parks), plus **L3b: interleave goldens
== threaded runs == sequential**. Any divergence is now a data race by
definition. L3d (real net) stays statistical: batched fp32 matmuls round
differently from batch-1, measured as move-agreement (first-move
agreement 3/3 in the CI smoke; tracked metric, no bar).

Deviations from the letter of this doc (each a deliberate refinement, the
math untouched):

1. **Requests carry the legal slots in enumeration order, not a 4240-bit
   legal bitset** (R7). The bitset forces the evaluator to return
   ascending-slot logp and C to re-permute into enumeration order; the
   enum-order list is needed for the logp gather anyway, and doubles as
   the mask source (scatter). Fixed-stride requests are kept (max-size
   tail, ~8.5 KB each) so buffer arithmetic stays trivial.
2. **u8 planes with dim 116 = the raw halfmove clock** (R7's "planes(u8
   7488)" would truncate the clock/100 feature to 0 or 1). Dims 0–115 are
   exact 0/1; the un-quantizer computes `(float)clock/100.0f` —
   bit-identical to nn_encode's f32 output, so the parallel path feeds the
   net exactly what batch-1 would.
3. **The evaluator pulls nothing; the engine pushes staging.** Batch
   assembly (u8 planes + enum-order slots + offsets + mover) happens in
   the engine before the call; the Python evaluator receives memoryviews
   over the staging (zero-copy) and writes outputs in place. Workers
   deliver their own results (F7 kept).
4. **Budget reservation is a CAS** (`games_started` never exceeds the
   budget — §6.4's invariant now holds exactly, not approximately).
5. The evaluator module/spec are C-side `strdup`ed config strings; the
   interpreter is required to be live (the library loads into a Python
   process; a future standalone binary would `Py_Initialize` first).
6. **One epoch buffer of 2·B per worker instead of two B-slots** (user
   decision, 2026-09-28). In the epoch model the buffer is per-worker
   queue capacity — the evaluator takes it whole each epoch, so two slots
   with one posted flag per worker (this round's own de-aliasing, R2) are
   equivalent to a single 2·B buffer, and the classic ping-pong overlap
   never arises: the gather copies payloads out, the workers sleep through
   the forward pass (by the wave discipline every game they own is parked
   and waiting for evaluations by seal time — the pass *is* the critical
   path), and the deferred list remains the overflow beyond 2·B. Slot
   rotation would become load-bearing only with an async/pipelined
   evaluator; the appendix's shared-queue fallback stays the designed
   escape hatch. Verified behaviorally: test_sp's 1216 checks (incl.
   parallel ≡ sequential exact and interleave goldens) pass unchanged on
   both structures.
12. **Per-game multi-park prefetching: already the wave machine's shipped
   semantics — premise corrected** (2026-09-29). A design round recorded
   the concern that one park per game per epoch couples the GPU batch to
   the in-flight game count (batch = games × 1, so a ~2k batch would need
   ~2k warm games and their memory). That premise was WRONG for the
   shipped code, and the correction is worth recording verbatim: the fill
   phase sweeps REPEATEDLY and parks each game's ENTIRE WAVE before
   posting — one pending descent per surviving candidate (search.c's
   dstate/flights arrays; wave_pending counts them; SP_STEP_IDLE only
   when the whole wave is in flight) — so a game contributes up to
   |A| ≈ min(n, m_cap) ≈ 8 requests per epoch. The legal ceiling is
   exactly |A| and no more: a candidate's next descent selects on Q/N
   that the current descent's backup will change (same-candidate serial
   across waves), and below a fresh leaf the tree does not exist until
   its policy arrives. The prefetcher/walker split the round described
   (walk-and-park pre-value; delivery-side backup) is precisely how
   search.c already factors it. Consequence: the decoupling the round
   wanted — same batch from ~games/|A| warm games — SHIPS: verified
   2026-09-29 with the fixture evaluator, 1 worker, B=256: 8 warm games
   → max_batch 64 (8.0 parks/game), 4 games → 32 (8.0); and live sizing
   stands unchanged (2048-slot staging, ~2k batches from ~410-game
   pools). Locked in by test_sp's batch-fill gate (few games must fill
   a big batch — ≥2 parks/game asserted, 8.0 observed). Epochs per move
   remain O(plays) waves — the round's serial dependency chain, by
   construction not by accident. No build followed from this round; the
   only actionable residue was the gate and this correction.
11. **Collect-time shard dump: no drain, no ring** (user decision,
   2026-09-28 — supersedes deviation 10's drain MECHANICS; its
   budget/pool decoupling is kept and strengthened). A finished game is
   dumped by its OWN worker, at collect time, to that worker's shard
   file (`chess_sp_dump.h`: little-endian native format,
   `shard-w<i>.spbin`, self-delimiting records — n_plies/n_pi make a
   crashed run's truncated tail detectable), and its pool slot returns
   to the worker's free list immediately — same thread pushes and pops,
   race-free. The done ring, n_done-as-claim-cursor, SpDrain callback
   and the py_drain trampoline are all deleted. Rationale for not
   collecting straight into Python instead: the callback would put GIL
   round-trips on worker threads mid-fill (stalling the fill phase
   against the epoch's forward pass) and make the record sink shared
   mutable state; the file seam keeps ALL Python at the epoch barrier,
   and the only record cost is 1 + n_plies page-cache write()s per
   game. `failed` becomes atomic (workers set it on dump failures).
   The format's reference consumers: engine/tests/test_sp.c (parses
   shards back and diffs against sequential twins — the exact gate now
   runs through the file round trip) and spengine.py `read_records`
   (replays played_uci from start_fen — per-ply FENs are not stored,
   same rule as the in-memory records). The trainer's tensor conversion
   builds on the same reader later (3.4). Gates: test_sp 453 checks
   green incl. budget-300-from-128-slots; ASAN clean.
10. **n_done = cumulative; pool = concurrency; per-epoch drain couples
   them** (user decision, 2026-09-28 — the training-granularity model).
   The first static cut conflated two bounds: game_budget was capped by
   the pool (finished games held slots until run end, because records
   were only readable post-wait). Corrected: game_budget is the
   TRAINING-GRANULARITY total (10k, 100k, more when training demands —
   NOT memory-bound), the pool (128/worker) bounds CONCURRENT in-flight
   games only, and a per-epoch DRAIN couples them: between gather and
   batch (workers parked), the wait-caller hands every finished game to
   an SpDrain consumer (export records / hash / write a shard), then
   returns its slot to the owning worker's pool. n_done is the ring
   claim cursor AND the cumulative games-done counter; the ring
   (pool-sized) cannot overflow because every undrained completion pins
   a distinct pool slot — fewer than ring-size completions can coexist
   between drains, so no index is claimed twice before its entry is
   drained. Consequences: workers post every round, sealed OR EMPTY —
   this is R1's "seal on full or no-work" implemented exactly (an empty
   post is how a slot-starved worker still counts toward the barrier;
   the drain unblocks it, the epoch broadcast wakes it); the worker exit
   condition is pure end-of-run (budget exhausted && nothing in flight —
   an empty free list is transient under recycling); a drain consumer
   must extract what it needs before returning (the slot becomes a fresh
   game). The C tests verify at collect time (exact sequential diff per
   finished game; FNV-1a golden game hashes — R6's goldens, now real —
   for the interleave test), and the decoupling gate runs budget 300
   from one worker's 128 slots. The Python seam accumulates GameRecords
   in a RecordSink; 3.4's shard writer replaces the accumulate step with
   a serialized append — the C side is already drain-shaped for it.
   Live: 5000 games from the 1024-slot pool in 6.2 s, avg batch 1959.
9. **Zero-allocation engine** (user decision, 2026-09-28 — the full
   static-memory guideline): search.c and sp_engine.c contain no
   malloc/calloc/realloc/free. Everything is fixed-size, bounded by
   published constants, and lives in ONE caller-provided storage block:
   - Game slots: sizeof(SpGame) ≤ SP_GAME_SLOT_SIZE = 2 MB (static
     assert; actual ≈ 1.8 MB — State 24 KB + planes 29 KB + fixed
     MAX_PLY record array ~110 KB + two 768 KB tree-arena regions).
     Games are POOLED per worker (SP_ENGINE_MAX_OUTSTANDING = 128, the
     sizing law's 64-required equilibrium × 2) and re-initialized by
     sp_game_init() — never allocated.
   - Records: one per ply, bounded by MAX_PLY = 1024 (the rules engine's
     own bound); a record is ~108 B: played move + π′ over ≤ 16
     candidates as f32 + outlook, NO FEN string (the exporter replays the
     game's moves from the stored start position — same serializer, so
     byte-identical output; the L2 gate's 1e-6 tolerance is untouched by
     the f32 π′ rounding).
   - Tree arenas: two fixed 768 KB bump regions per game, ping-ponged at
     the per-move handoff reset (the subtree copy never overlaps its
     source; the dead tree dies by pointer reset). Covers eval mode
     n = 200 (peak ≈ 550 KB incl. the copy); a full region asserts.
   - Worker buffers, deferred lists (bound: outstanding × m_cap),
     result lists, batch staging, and the done list are embedded in the
     engine struct; the epoch buffer is B (the "2·B" headroom was removed
     with the double-buffer vestige — deviation 6's conclusion).
   - Storage is caller-provided: C tests reserve it with anonymous mmap
     (a multi-GB BSS segment cannot even load on macOS), Python mmaps it;
     sp_engine_init never memsets the bulk (untouched pool/staging pages
     stay uncommitted, so the ~2.3 GB bound costs only what the run
     touches). If the worst case does not fit the machine, mmap fails
     loudly at startup — malloc would not save the day.
   Consequences: (a) n_workers/B/m_cap are capped (8/256/16) — raised
   by editing one constant each; (b) finished games hold their slots only
   until the next epoch's drain (deviation 10) — the budget is NOT
   pool-bound. Fixed en route: the evaluator's free hook now fires
   exactly once, in sp_engine_destroy (run_interleave's extra call
   double-freed the Python evaluator's ctx — caught by L3b).
8. **Done list: embedded fixed array, lock-free slot claim** (user
   decision, 2026-09-28). The CAS reservation bounds started games — hence
   done entries — by the budget, and the array is embedded in the engine
   struct at compile-time capacity SP_ENGINE_MAX_GAMES = 2^20 (game_budget
   must be ≤ that cap; larger runs shard): no mutex, no realloc, no
   allocation at all, and `n_done` *is* the games-done counter. (A
   file-scope C `static` array was rejected — it would violate the
   engine's no-mutable-globals rule and break multi-engine processes.) The engine
   mutex is now used only at the per-epoch handshake (R2, exactly). List
   readers run after pthread_join, so claimed-then-written slot ordering
   is a non-issue. 3.4 note: mid-run draining (streaming records to disk)
   on top of slot-claiming needs a written-watermark cursor (reader drains
   only slots below the watermark) — or per-worker sublists; decide there.
   Memory datum for that decision: a completed game retains ~110 KB
   (63 KB SpGame + ~768 B/recorded move), so a 100k-game run holds ~11 GB
   unless drained.
7. **No dedicated evaluator thread** (user decision, 2026-09-28): all
   cfg.n_workers spawned threads are workers; the sp_engine_wait() caller
   plays the evaluator between epochs (`run_evaluator_loop`). Rationale:
   in the lock-step epoch the two roles never compute simultaneously —
   every game is parked while the batch runs — so a reserved evaluator
   thread idled through every fill phase. Merging makes the fill phase
   run on N fillers instead of N−1 and spawns one fewer thread, at zero
   cost to the eval phase. The GIL is re-acquired per batch by the wait
   caller (cffi released it at the boundary). Contract change: wait() is
   now load-bearing — the engine makes no progress unless it is running
   (documented in chess_sp.h; every seam calls start+wait together).

Gates green: `engine/tests/test_sp.c` (1216 checks; parallel ≡ sequential
exact, interleave goldens, budget/invariants, evaluator-failure path),
`tests/test_sp_l3.py` (L3a/L3b/L3c/L3d), full C `check`/`check-asan` (the
threaded engine is ASan+UBSan clean), full pytest suite. Telemetry at
reference sizing (7 workers, B=256, fixture eval): ~400k evals/s on one
M4 core set, average batch ≈ 900, max ≈ 1600 — the deferred prefetch
fills batches past the per-worker parked-wave floor, as designed.

---

## Appendix: the shared-queue alternative (not adopted)

Shared pending queue + condvars; no straggler exposure; per-request locking and a
min_batch knob. Fallback if the B=256 straggler gap (R1) ever measures badly — the
worker/evaluator split makes the swap local to the submission/delivery paths.
