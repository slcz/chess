/*
 * chess_search.h — the Gumbel self-play search core (Phase 3.1 / 3.3).
 *
 * WHAT THIS IS
 * ============
 * The C implementation of the agreed Gumbel search (PLAN Phase 3,
 * "Agreed search design" — Danihelka et al., ICLR 2022), ported from the
 * sequential Python reference chess_ai/selfplay/gumbel.py (Phase 3.2a) and
 * structured for the parallel engine (PARALLEL_DESIGN.md). Search math is
 * unchanged from the reference:
 *
 *   - root: Gumbel top-m candidates (m = min(n, m_cap, #legal)), z(a) =
 *     logp(a) + gumbel(a) with Gumbels drawn per legal move in movegen
 *     enumeration order;
 *   - sequential halving over the candidates: rounds = ceil(log2 m) fixed
 *     from the ORIGINAL m; per round plays = ceil(n / (rounds·|A|)) full
 *     descents per candidate, candidate-major (`for a in A: plays ×
 *     descend(a)` — the reference's pinned order, what the L2 exact diff
 *     requires); A = top ⌈|A|/2⌉ by z + σ(q);
 *   - below the root: Eq. 14 — argmax over legal a of
 *     logp(a) + σ(completedQ(a)) − N(a)/(1+ΣN), with completedQ = q(a) if
 *     N(a)>0 else v̂(node) − c_fpu, and σ(q) = (C_VISIT + max_b N)·C_SCALE·q;
 *   - π′ = softmax over the original candidate set A₀ of
 *     logp(a) + σ(completedQ(a)) — the training target;
 *   - resignation: outlook = max completedQ over A₀ ≤ resign_value → the
 *     mover resigns immediately (PARALLEL_DESIGN R4/round 7);
 *   - handoff: the played move's child node (always evaluated, subtree
 *     included) becomes the next move's root — every root after move 1
 *     costs zero evaluations (F4);
 *   - log space everywhere: the net's policy_logp is stored and used as-is
 *     (R7); never mixed with raw logits.
 *
 * Sign conventions (load-bearing, from the reference's _backup): a node's
 * stats are from ITS OWN mover's view — the leaf value (pW−pL) is backed up
 * with a sign flip per ply; a parent reading a child's q negates it. All
 * tree arithmetic is IEEE double, in the reference's exact evaluation
 * order, so the L2 gate (sequential C vs gumbel.py, same seed ⇒ identical
 * games) is bit-exact.
 *
 * THE STATE MACHINE SHAPE (why it looks like this)
 * ================================================
 * The search is a *resumable per-game state machine* with exactly one park
 * point — the fresh expansion leaf — instead of a recursive descent with an
 * inline evaluator. One code path serves every driver:
 *
 *   - sequential (this phase, 3.1): sp_play() loops sp_game_step() and
 *     fulfils each evaluation request synchronously through a C function
 *     pointer (fixture evaluators in the unit tests; the embedded-CPython
 *     evaluator later);
 *   - Python-driven (the L2 test): the same loop, with the request pulled
 *     over the FFI boundary and fulfilled in Python (cffi ABI mode has no
 *     callbacks — the pull model needs none);
 *   - parallel (3.3b, next): a worker loop advances many games round-robin;
 *     sp_game_step() returns SP_EVENT_NEED_EVAL, the request goes into the
 *     worker's batch buffer and the game is set aside until the evaluator
 *     delivers (PARALLEL_DESIGN §4.3 game_step verbatim). No rewrite: the
 *     park point *is* the batch request.
 *
 * Each game owns everything it touches — working State, NN history ring,
 * RNG, node arena, records — and the module has no mutable globals, so any
 * number of games can run concurrently on separate threads (the tables from
 * chess_init() are read-only). Nodes hold no position state (F7): the
 * search walks the game's working State with make/unmake, and the NN
 * history ring is kept in step with every push/pop exactly like the FFI
 * handle does, so an encoding taken while parked sees the correct 8-ply
 * window.
 *
 * MEMORY (F7, arena)
 * ==================
 * Nodes come from a per-game chunked bump arena, reset after every move:
 * a search's node count is bounded by its descent count (~1.3–2n plus the
 * handed-off subtree), so the arena is tiny at self-play sims (n ≤ 32) and
 * modest at eval sims (100–200). The reset keeps the handoff subtree by
 * deep-copying it into the fresh arena (bump allocation cannot free
 * selectively; the subtree is bounded by the previous search's size). Node
 * references are u32 arena indices (never pointers), so the copy is a
 * straight remap. Move records are kept per game until the game ends (the
 * 3.4 shard writer flushes them); at self-play sims a record is ~0.5 KB,
 * so a several-hundred-ply game is well under 1 MB.
 */
#ifndef CHESS_SEARCH_H
#define CHESS_SEARCH_H

#include <stdint.h>

#include "chess.h"   /* Move, CHESS_FEN_MAX lives in chess_ffi.h below  */
#include "chess_ffi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

/* Knobs — mirrors of gumbel.py's SelfPlayConfig (PARALLEL_DESIGN §4.0).
 * n must be ≥ 2 (m ≤ n is what gives every candidate a round-1 visit; the
 * forced k == 1 position flows through the general path with m = 1 and no
 * halving — there is no k==1 special case, round 6).
 * Field order is part of the ABI surface (Python mirrors the struct in
 * cffi declarations) — do not reorder casually. */
typedef struct {
    int    n;             /* sims per move (self-play 2–32; eval 100–200)  */
    int    m_cap;         /* candidate cap ≤ SP_MAX_CANDIDATES (design: 16) */
    double c_visit;       /* 50.0                                          */
    double c_scale;       /* 1.0                                           */
    double c_fpu;         /* Eq. 10 reduction (0 = paper default)          */
    int    sample_plies;  /* < 0 = sample ∝ π′ every ply (training);
                             else sample only on plies < sample_plies and
                             play the halving survivor afterwards (eval)   */
    double resign_value;  /* outlook ≤ this → resign (−0.90, R4)           */
} SpConfig;

/* PARALLEL_DESIGN §4.0 defaults (n=8, m_cap=16, 50/1/0, every-ply
 * sampling, −0.90). */
void sp_config_init(SpConfig *cfg);

/* The candidate cap the record layout can hold (π′ support = A₀, at most
 * m ≤ m_cap). The design pins m_cap = 16; 64 leaves headroom for sweeps. */
#define SP_MAX_CANDIDATES 64
/* Active descent depth bound: each descent creates exactly one fresh node,
 * so depth ≤ 1 + evals per search (~2n; n ≤ 200 in eval mode) — far below
 * the search's own tree bound. (Park path, not the game's move log.) */
#define SP_MAX_PATH 1024

/* ------------------------------------------------------------------ */
/* Static game storage (PARALLEL_DESIGN §8, deviation 9: no mallocs)    */
/* ------------------------------------------------------------------ */

/* Games are FIXED-SIZE structs living in caller-provided storage — the
 * engine pools them; nothing in the search allocates. The constants below
 * pin the geometry; search.c static-asserts sizeof(SpGame) ≤
 * SP_GAME_SLOT_SIZE.
 *
 *   - Records: one per ply, bounded by MAX_PLY (the rules engine's own
 *     undo-stack bound — a game cannot exceed 1024 plies). A record holds
 *     the played move + π′ over ≤ SP_REC_CANDIDATES candidates as f32 +
 *     the resignation outlook; NO FEN string (the exporter replays the
 *     game's moves from the start position — sp_rec_fen_before does this
 *     on demand). m_cap must be ≤ SP_REC_CANDIDATES (the design pins 16).
 *   - Tree arena: two fixed bump regions, ping-ponged at the per-move
 *     reset so the handoff subtree copy never overwrites its source (the
 *     copy compacts into the other region). 768 KB covers eval at
 *     n = 200 (peak ≈ 550 KB — old tree + handoff copy); self-play
 *     (n ≤ 32) peaks at ~55 KB. A full region asserts. */

#define SP_REC_CANDIDATES  16   /* π′ support per record (m_cap ≤ this) */
#define SP_GAME_ARENA_CAP  (768u * 1024u)   /* bytes per bump region    */
#define SP_GAME_MAX_NODES  4096u            /* tree nodes per move      */
#define SP_GAME_SLOT_SIZE  (2u * 1024u * 1024u)  /* ≥ sizeof(SpGame), aligned */

/* ------------------------------------------------------------------ */
/* Game lifecycle                                                      */
/* ------------------------------------------------------------------ */

typedef struct SpGame SpGame; /* opaque: everything a game needs is inside */

/* Initialize a game IN CALLER-PROVIDED STORAGE (`storage` must be
 * SP_GAME_SLOT_SIZE bytes, suitably aligned — engine pools hand out their
 * slots; standalone drivers declare static or mmap'd buffers). fen NULL or
 * "" = standard start position; NULL return = malformed FEN. `seed` is
 * the work-unit seed and `game_index` the game's slot: the per-game RNG
 * is xoshiro256++(game_seed(seed, game_index)) — the rng.py derivation —
 * so identical (seed, game_index) pairs replay identical games in both
 * engines. The first RNG draw is move 1's Gumbels (no game-creation
 * coin). There is no destroy: re-initing the storage reuses it. */
SpGame *sp_game_init(void *storage, const SpConfig *cfg, const char *fen,
                     uint64_t seed, uint32_t game_index);

/* The slot size, as a function — foreign runtimes (cffi) cannot read
 * expression macros, and this cannot drift from the static assert. */
size_t sp_game_slot_size(void);

/* ------------------------------------------------------------------ */
/* The state machine                                                   */
/* ------------------------------------------------------------------ */

/* What one sp_game_step() call did:
 *   NEED_EVAL — the game parked on a fresh expansion leaf (or on a fresh
 *               root): fulfil the pending request (sp_request_* /
 *               sp_deliver), then step again.
 *   MOVED     — a move completed (record appended, position pushed, handoff
 *               taken). Includes the resign record (played_slot = −1, no
 *               move pushed). Step again for the next move.
 *   DONE      — the game is over: inspect sp_result_* / the records. */
enum {
    SP_EVENT_NEED_EVAL = 0,
    SP_EVENT_MOVED     = 1,
    SP_EVENT_DONE      = 2,
};

/* Advance the game's state machine until it parks on an evaluation, finishes
 * a move, or the game ends. End conditions, checked at move start in this
 * order (gumbel.py play()): resignation recorded at the previous finalize →
 * done; natural termination (state_termination, claim-draw semantics) →
 * done; max_plies reached (when max_plies ≥ 0) → adjudicated done. */
int sp_game_step(SpGame *g);

/* ------------------------------------------------------------------ */
/* Evaluation requests (valid between NEED_EVAL and sp_deliver)         */
/* ------------------------------------------------------------------ */

/* The 7488-float feature tensor of the parked position (the same nn_encode
 * the FFI's chess_encode produces — mover-relative, 8-ply history). */
const float *sp_request_planes(const SpGame *g);
/* The parked position's legal policy slots, in movegen enumeration order —
 * the order the logp array below must be aligned to (it is the order the
 * RNG's Gumbels and every tie-break are specified over). */
const uint16_t *sp_request_slots(const SpGame *g);
int sp_request_count(const SpGame *g);       /* number of legal moves      */
int sp_request_mover(const SpGame *g);       /* 0 white / 1 black          */
/* 1 when the parked request is a *root* evaluation (no handoff node) —
 * telemetry for the handoff property ("every root after move 1 is free"). */
int sp_request_is_root(const SpGame *g);

/* Fulfil the pending request: logp[count] aligned to sp_request_slots()
 * (the net's policy_logp at the legal slots — double, holding widened f32
 * values from the net path), wdl[3] = (pW, pD, pL) from the mover's view.
 * The node is filled (state EVALUATED), the search backs the value up and
 * resumes; the next sp_game_step() continues the move. */
void sp_deliver(SpGame *g, const double *logp, const double *wdl);

/* Convenience evaluator for sp_play(): read the request via the accessors,
 * write logp[count] + wdl[3]. */
typedef void (*SpEvalFn)(void *ctx, const SpGame *g, double *logp,
                         double *wdl);

/* Sequential driver: play the game to natural end / resignation /
 * adjudication (max_plies < 0 = unbounded), fulfilling every evaluation
 * synchronously through (fn, ctx). This is the whole Phase 3.1 loop; the
 * parallel engine replaces it with the worker loop over sp_game_step. */
void sp_play(SpGame *g, SpEvalFn fn, void *ctx, int max_plies);

/* Adjudication bound for the sp_game_step driver (Python L2 test / eval
 * matches): games end adjudicated once ply ≥ max_plies. < 0 = unbounded
 * (the default). sp_play sets it from its argument. */
void sp_set_max_plies(SpGame *g, int max_plies);

/* ------------------------------------------------------------------ */
/* Records (gumbel.py's MoveRecord / GameRecord, field for field)       */
/* ------------------------------------------------------------------ */

int sp_record_count(const SpGame *g);        /* moves recorded so far      */
int sp_rec_ply(const SpGame *g, int i);
int sp_rec_mover(const SpGame *g, int i);    /* 0 white / 1 black          */
/* FEN before the move (≥ CHESS_FEN_MAX bytes). */
void sp_rec_fen_before(const SpGame *g, int i, char *out);
/* Played move: policy slot and UCI. A resign-terminated move records
 * played_slot = −1 and an empty UCI (R4/round 7 — the record still carries
 * π′ and the outlook that triggered the resign). */
int  sp_rec_played_slot(const SpGame *g, int i);
void sp_rec_played_uci(const SpGame *g, int i, char out[8]);
/* π′ support: the ORIGINAL candidate set A₀ in z-descending order, with
 * π′ aligned to it. */
int    sp_rec_n_pi(const SpGame *g, int i);
int    sp_rec_pi_slot(const SpGame *g, int i, int j);
double sp_rec_pi_prob(const SpGame *g, int i, int j);
/* max completedQ over A₀ — the resignation input. */
double sp_rec_outlook(const SpGame *g, int i);

/* ------------------------------------------------------------------ */
/* Game result (valid once sp_game_step returned SP_EVENT_DONE)         */
/* ------------------------------------------------------------------ */

enum {
    SP_END_PLAYING = 0,   /* game still in progress (not DONE yet)        */
    SP_END_NATURAL,       /* state_termination fired (mate / draw rules)  */
    SP_END_RESIGN,        /* outlook ≤ resign_value at finalize (R4)      */
    SP_END_ADJUDICATED,   /* max_plies reached (eval matches)             */
};

int    sp_result_kind(const SpGame *g);       /* SP_END_*                   */
int    sp_result_resigner(const SpGame *g);   /* −1 none / 0 white / 1 black*/
double sp_result_z_white(const SpGame *g);    /* +1 white win / 0 / −1      */
void   sp_result_start_fen(const SpGame *g, char *out); /* ≥ CHESS_FEN_MAX  */
int    sp_result_game_index(const SpGame *g);
int    sp_result_n_sims(const SpGame *g);     /* cfg.n                      */

/* ------------------------------------------------------------------ */
/* Parallel-engine mode (Phase 3.3b): the wave-concurrent game machine */
/* ------------------------------------------------------------------ */

/*
 * The sequential driver above parks the whole game on ONE descent at a
 * time. The parallel engine instead runs a round's descents in per-round
 * WAVES (PARALLEL_DESIGN F3/§4.4): the wave's |A| descents (one forced
 * first step per candidate) are started one game_step at a time, each
 * parking at its own expansion leaf, and wave w+1 starts only after every
 * wave-w descent was delivered and backed up (the wave barrier — what
 * keeps every descent's decisions on resolved state only). A game
 * therefore contributes up to |A| requests per fill phase before it
 * stalls on the barrier — the sizing law's r ≈ (1.3–2)·n/⌈log₂ m⌉
 * requests per game.
 *
 * The two modes share all the tree math (descend/Eq. 14/backup/halving/
 * finalize); only the scheduler differs. With a batch-invariant evaluator
 * the wave mode produces exactly the sequential mode's games: candidate
 * subtrees are disjoint, same-candidate descents are serial (the barrier
 * guarantees it), and the only delivery-order-dependent accumulation is
 * the root's own stats, which selection never reads.
 *
 * SP_REQ_MAX_LEGAL bounds the per-request legal-move list (the strict
 * generator never emits more than MAX_MOVES). */
#define SP_REQ_MAX_LEGAL 218

/* One parked descent, owned by the engine's buffer slot. The search fills
 * everything except `game` at park time (the engine stamps the owner).
 * Planes are the u8-quantized feature tensor: dims 0–115 are exactly 0/1
 * (piece one-hots, repetition flags, ep file, castling); dim 116 holds the
 * RAW halfmove clock (0–150) — the un-quantizer computes
 * (float)clock/100.0f, bit-identical to nn_encode's f32 output (a plain
 * u8 cast of the /100 feature would truncate it). Slots are the legal
 * policy slots in movegen ENUMERATION order — the order the returned logp
 * array must be aligned to (it is the RNG/tie-break order). */
typedef struct {
    SpGame  *game;                        /* owner — stamped by the engine */
    uint32_t node;                        /* parked node (delivery key)    */
    uint8_t  is_root;                     /* root evaluation request       */
    uint8_t  mover;                       /* side to move at the park      */
    uint16_t n_legal;
    uint8_t  planes_u8[64 * 117];         /* quantized nn_encode output    */
    uint16_t slots[SP_REQ_MAX_LEGAL];     /* enum order                    */
} SpReq;

/* Wave-mode game step (the worker loop's game_step): advance the game by
 * one scheduler event. Returns one of:
 *   SP_STEP_PARKED   — a descent parked; *req is filled (the caller owns
 *                      it now: stamp req->game, enqueue, count it).
 *   SP_STEP_PROGRESS — real progress without a park (halving, finalize,
 *                      move completed).
 *   SP_STEP_IDLE     — no runnable descent (wave barrier: parked descents
 *                      await delivery; or root eval outstanding).
 *   SP_STEP_DONE     — the game is over (records complete). */
enum {
    SP_STEP_IDLE = 0,
    SP_STEP_PARKED = 1,
    SP_STEP_PROGRESS = 2,
    SP_STEP_DONE = 3,
};

/* Switch the game to wave-concurrent scheduling. Must be called before the
 * first step (games created for the parallel engine; sequential games are
 * untouched and stay bit-identical to the L2 contract). */
void sp_game_set_wave_mode(SpGame *g);

/* One wave-mode step. `req` may be NULL only if the caller knows the step
 * cannot park (it can't know — always pass a slot). */
int sp_game_step_wave(SpGame *g, SpReq *req);

/* Deliver a result to ONE pending request of a wave-mode game (the wave's
 * descents park concurrently; `node` — the request's key — selects which).
 * The descent is backed up and the wave barrier re-evaluated. logp is
 * aligned to the request's slots (enumeration order), wdl is (pW,pD,pL)
 * from the parked position's mover's view. */
void sp_deliver_node(SpGame *g, uint32_t node, const double *logp,
                     const double *wdl);

/* Owner stamp for result routing (engine bookkeeping; pure data). */
void sp_game_set_owner(SpGame *g, int owner);
int  sp_game_owner(const SpGame *g);

/* True while the game has at least one undelivered request outstanding
 * (parked descent or root eval) — engine accounting. */
int sp_game_pending(const SpGame *g);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_SEARCH_H */
