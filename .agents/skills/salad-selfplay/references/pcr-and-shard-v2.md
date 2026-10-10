# PCR + shard v2 — design rationale

Recorded 2026-10-10 (branch `puct_mcts`, commits `c6bf915`, `eecc51f`).
Read this before changing the randomization, the shard format, or the
training filter — each rule below was a decision, several were user
corrections.

## Playout cap randomization (KataGo, Wu 2020 §4)

Each move's search budget is random: cheap `fast_sims` with probability
`fast_p`, else the slow `sims`. Only slow-cap plies become training
targets; every ply is still *played* from its own (cheap) search.

- Why it pays: expected cost 0.75·50 + 0.25·600 = **187.5 sims/move**,
  i.e. ~3.2× cheaper games than uniform-600 — while the *quality* of
  harvested targets is unchanged (all slow-cap). Trade: only 25% of plies
  are samples, so matching a fixed sample count needs ~4× games at ~1.25×
  the old total self-play cost. The bet (user-approved): at this stage,
  diverse games + full-depth targets beat many medium-depth targets.
  Round bar: ≥50% vs the 42k net at n=200/400 games.
- Fast plies are NOT wasted: they keep games moving realistically (100-
  or 50-sim moves are still decent), they add exploration noise, and the
  game outcome `z` only needs the game *played*.

## Why the draw is per STEP, not per move (deviation from the paper's letter)

`selfplay.c` runs a pool (default 1024) of games in lockstep on one
`PuctMux`: every `puct_mux_step(m, sims)` gives every live game one
`sims`-budget think, batched across the pool, and the step ends only when
all games are done. With per-move draws, P(no game drew slow) =
(1−fast_p)^pool ≈ 0 — every step would pay slow-step wall-clock anyway,
with the GPU draining tiny batches in the tail. Drawing ONE budget per
step for the whole pool keeps the identical per-ply marginal (each ply
independently slow w.p. 1−fast_p) and full-pool batches. This is the one
place the implementation deliberately deviates from KataGo's per-move
draws, which assume independent game threads. User approved.

## Shard format v2

Byte authority: `engine/include/chess_sp_dump.h`. v2 = v1 + one `u16
sims` per ply block (the budget that ply actually searched with), placed
after `outlook`, before the π pairs. The game-level `n_sims` stays the
slow/nominal cap. Records stay self-delimiting (killed run → truncated
trailing game, dropped with a warning).

Reader contract (`chess_ai/selfplay/shard.py`): accept v1 AND v2; for v1,
fill `SpPly.sims` from the game cap so every ply looks trainable — old
units on HF keep ingesting unchanged. Unknown versions must raise, not
guess (`tests/test_shard_format.py` pins all three behaviors with
hand-built bytes). Everything that parses shard headers must use
`VERSIONS` from shard.py (sp_pusher's `shard_seed` once asserted v1).

Trainable rule (`selfplay_data.ply_trainable`): `ply.sims ==
game.n_sims`. `selfplay_samples` walks ALL plies (positions replay
through fast moves — per-ply FENs are not stored) but emits samples only
for trainable plies; the aux-reply target of ply i is ply i+1's played
move even when that move was fast-chosen (matches KataGo).
`selfplay_sample(game, ply)` asserts trainability so a caller-side
filtering bug fails loudly. `sp_round_train.py` filters its position
list and logs "N trainable (of M recorded plies)".

## Instance identity & restart semantics (user-specified)

Assume: a restarted container has a CLEAN filesystem, and salad may or
may not reuse `SALAD_INSTANCE_ID`. Therefore identity = boot stamp
`<SALAD_INSTANCE_ID>-<epoch>`; seed = FNV-1a-64(stamp) (fits the engine's
uint64 seed); prefix `units/<RUN>-i<stamp>/`. Consequences:

- A restart always starts a FRESH unit — new games, new files; it can
  never replay the same game sequence or overwrite an earlier upload.
- Partial units already on HF stay valid. Ingest dedups by
  `(unit_seed, game_index)` globally, so even a pathological overlap is
  deduped, not corrupted.
- No marker-file resume; round state is read from HF
  (`scripts/sp_progress.py` walks `units/<RUN>-i*/part-*/DONE.json`).
- `deploy/sp_autorun.sh` refuses to run without an identity env (a shared
  seed/prefix across instances is the one unrecoverable mistake).

## The 42k base checkpoint

`runs/policy_value_net_42k.pt` (Mac-local, sha256 prefix `2a3fb75b55bc…`,
steps=42000, 8,762,659 params) is baked into the image and symlinked to
`runs/ckpt_a/net.pt`. Verified 2026-10-10: HF `policy_value/final.pt`
lists but downloads 404 (broken upload); `policy_value/latest.pt` is
steps=40000. So HF holds no true 42k net — bake the local file; do not
"helpfully" fetch a different checkpoint at boot (user was explicit).
