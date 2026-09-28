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

## Appendix: the shared-queue alternative (not adopted)

Shared pending queue + condvars; no straggler exposure; per-request locking and a
min_batch knob. Fallback if the B=256 straggler gap (R1) ever measures badly — the
worker/evaluator split makes the swap local to the submission/delivery paths.
