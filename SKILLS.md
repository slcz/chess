# SKILLS.md — runtime cheat sheet: self-play → train → eval

The full loop on one page. Every command here was exercised end-to-end in
the gen1→gen2 round (2026-10-13). Ops skill (machine-readable, salad
container pipeline): `.agents/skills/salad-selfplay/SKILL.md`.

```
selfplay (salad GPU) ──push──▶ HF dataset (units/<RUN>-i<stamp>/part-XXXX)
                                        │
             ┌────── snapshot_download ◀┘
        train (sp_round_train.py: stream-pack → cache → steps)
             │
             ├──▶ HF ckpt repo (selfplay/<name>.pt) at each epoch end
             ▼
        gate: 400 games candidate vs base (sp_round_eval.sh, ≥ 0.500)
             ▼
        arena: Stockfish Elo ladder (engine/build/sf_gauntlet)
```

## 0. Setup / auth

- Mac `.env` holds `HF_TOKEN`, `GITHUB_GHCR_KEY`, `SSH_HOST`, `SSH_PORT`
  (update HOST/PORT per instance from the salad console Connection panel).
- `scripts/salad.sh ssh|put` wraps ssh/scp: purges stale host keys, uses
  root@SSH_HOST. Instance images carry `HF_TOKEN` baked (ENV) — no token
  config on salad side, ever.
- **Detach rule**: the salad ssh gateway kills descendants when the channel
  closes. Long jobs MUST run as
  `setsid nohup <cmd> > /workspace/<name>.log 2>&1 < /dev/null &`.

## 1. Self-play generation

Autonomous (the image's CMD; a boot = a fresh unit, prefix
`units/<RUN>-i<STAMP>/`): configure in the salad console env only.

| Env (container group) | Meaning | Default in image |
|---|---|---|
| `RUN` | run name — **fresh name per generation** (monitor + trainer scope by it) | `sp-42k-pcr` |
| `GAMES` | per-instance target | 6250 |
| `SIMS` / `FAST_SIMS` / `FAST_P` | PCR slow cap / fast cap / fast prob | 300 / 50 / 0.75 |
| `SAMPLE_PLYS` | opening move-sampling window (τ=1 visits, then argmax); NOT a training filter | 10 |
| `SSH_PUBKEY` | optional shell access | — |

Manual single node (inside tmux or the container):
```bash
scripts/sp_gen.sh <RUN> <ID> [games] [seed]          # gen + pusher, exits at DONE
scripts/sp_gen_tmux.sh <RUN> <ID> <games>            # tmux wrapper (bare nodes)
scripts/sp_progress.py --run <RUN> --target <N> [--watch 300]   # from Mac
```
PCR semantics: one pool-wide budget draw per mux step; **only slow-cap
plies train** (`ply.sims == game.n_sims`, ~25% of plies). Fast plies are
replayed through and serve as aux-reply targets only. `SAMPLE_PLYS` only
affects which moves get *played* early — unrelated to what trains.

Stopping a box you want to keep training on:
- Kill the ENGINE only: `pkill -f "build/selfpla[y]"` — the pusher is the
  container job's foreground process; killing IT exits the container and
  salad reboots the box (fresh fs, lost tail <1000 games).
- A dead engine + living pusher = GPU free, container alive: the standard
  pre-training state.

## 2. Training (`scripts/sp_round_train.py`)

```bash
cd /workspace/chess && setsid nohup python3 scripts/sp_round_train.py \
    --prefix units/<RUN> --base-ckpt <ckpt.pt> --out runs/<name> \
    --epochs 2 --lr 2e-5 --batch 1024 \
    > /workspace/train-<name>.log 2>&1 < /dev/null & echo launched
```

- **Streaming ingest**: parts stream into `PackedRound` (flat numpy label
  arrays; tokens are replayed per batch, never stored). 280k games ≈ 4.5 GB
  during parse, ~7 GB steady — the ~55 GB record-object OOM is fixed.
- `--workers 8` packs parts in parallel (~11 min → ~2 min for 532 parts).
- `--cache-dir` (default `packed-cache/`, gitignored): the packed arrays
  are cached on disk keyed by the unit listing; re-launches skip the parse
  (~40 s to first step). Delete the file to force a repack.
- `--max-games N` forces the serial path (order-preserving truncation).
- Other knobs: `--value-weight/--policy-weight` (loss mix), `--save-every N`
  (snapshots), `--limit-steps N` (smoke), `--device auto`.
- Outputs: `<out>-e1.pt`, `<out>-e2.pt` locally AND pushed to
  `chriszhang01/chess-value-net` under `selfplay/` — epoch checkpoints
  survive box death.
- Healthy startup: `ingested N games, M positions from K parts` →
  `T trainable positions (of M recorded plies)` (T ≈ 25% = PCR slow-cap
  share) → `step 100/…` every ~1–2 min. The parse phase is silent unless
  pre-fix (see history); watch `ps` RSS if unsure.
- Watch: `tail -2 /workspace/train-<name>.log`.

## 3. Gate eval (candidate vs base)

```bash
scripts/sp_round_eval.sh runs/<cand>.pt runs/ckpt_a/net.pt 400
# detached:
setsid nohup scripts/sp_round_eval.sh runs/<cand>.pt runs/ckpt_a/net.pt 400 \
    > /workspace/eval-<name>.log 2>&1 < /dev/null & 
scripts/salad.sh ssh 'grep -aE "^(SUMMARY|TOTAL)" /workspace/eval-<name>.log'
```
- 400 games @ 200 sims each side, colors swapped per wave off
  `data/noob_3moves.epd` (~10 MB — NOT baked in the image yet; scp it:
  `scripts/salad.sh put data/noob_3moves.epd /workspace/chess/data/`).
- Acceptance: TOTAL `score_a ≥ 0.500`. ~14 min on a 3090.
- Raw variant with dumps: `scripts/gate_match.py --ckpt-a … --ckpt-b …
  --games 200 --dump-dir /tmp/gate`.

## 4. Stockfish arena

Two tools, two difficulty axes:

**Elo ladder (preferred; mux-batched)** — `engine/build/sf_gauntlet`
(must be built on the box: `make -C engine build/sf_gauntlet`; image does
not bake it). Needs `stockfish` on PATH — apt puts it in `/usr/games`:
`ln -sf /usr/games/stockfish /usr/local/bin/stockfish`.
```bash
export NN_BF16=1 NN_COMPILE=1
engine/build/sf_gauntlet <ckpt> 256 1350,1600,1850,2100,2350 \
    60 30 200 10 300 stockfish cuda 0 7 > /workspace/sf-<name>.log 2>&1
# args: ckpt sims elo_csv games/level pool movetime sample_plys max_plys sf device rows seed
```
Read the crossover rung (where score crosses 0.500). Reference results
(3090, SF 14.1 movetime 200, sims 256, sample_plys 10, 60 games/rung):

| SF Elo | 42k base (gen1) | iterN-e2 (gen2) |
|---|---|---|
| 1350 | 0.950 | 0.967 |
| 1600 | 0.650 | 0.783 |
| 1850 | 0.358 | 0.483 |
| 2100 | 0.142 | 0.050 |
| 2350 | 0.131 | 0.083 |

crossover ≈ 1730 (gen1) vs ≈ 1835 (gen2) ≈ **+100 Elo**. Always rerun the
base on the same box — SF builds/hardware shift absolute numbers.

**Node ladder (quick, Mac-capable)** — `scripts/gauntlet_sf.py`:
```bash
python3 scripts/gauntlet_sf.py --ckpt <ckpt> --device mps --sims 256 \
    --sample-plys 10 --games 8 --sf-nodes 30 100 300 1000
```

## 5. Checkpoints & data on HF

```bash
set -a; . ./.env; set +a                      # HF_TOKEN
# fetch dataset parts (labels only; globs: '<prefix>*' matches both layouts)
.venv/bin/hf download chriszhang01/chess-selfplay --repo-type dataset \
    --include "units/<RUN>*" --local-dir ./hf-data
# fetch/push checkpoints (model repo)
.venv/bin/hf download chriszhang01/chess-value-net selfplay/<name>.pt --local-dir .
# rename = upload new + delete old (verify upload first!)
```
- Dataset: `chriszhang01/chess-selfplay` (`units/<RUN>-i<stamp>/part-XXXX/
  {shard-0.spbin,DONE.json}`, ~8 MB per 1000 games).
- Checkpoints: `chriszhang01/chess-value-net` (`selfplay/*.pt`).
- Generation naming: **gen1 = 42k base** (`selfplay/selfplay-gen1.pt`),
  **gen2 = iterN-e2** (`selfplay/selfplay-gen2.pt`). Ingest dedups by
  (unit_seed, game_index); trainer downloads automatically.

## 6. Inspection / debugging

```bash
python3 scripts/sp_stats.py <part-dir>          # outcome/entropy/dup report
python3 scripts/sp_pusher.py <unit-dir> chriszhang01/chess-selfplay \
    units/<stamp> --every 1 --once              # salvage a killed unit's tail
python3 -m pytest tests/test_packed_round.py \
    tests/test_selfplay_roundtrip.py -q         # local data-path parity (~7 s)
```

## 7. Pitfalls (each one bit us)

1. Fresh `RUN` name per generation — monitor + `--prefix` scope by it.
2. Only slow-cap plies train. Don't "fix" the ~25% trainable share.
3. Never kill the pusher on an autonomous container — engine only.
4. Detach long jobs (`setsid nohup`); ssh drops kill foreground children.
5. Fresh container = image's code. Today's fixes reach new boxes only via
   image rebuild, or `scripts/salad.sh put` of
   `scripts/sp_round_train.py` + `chess_ai/train/selfplay_data.py`
   (+ `data/noob_3moves.epd` for evals) until then.
6. `hf download --include "units/<RUN>/**"` matches nothing on container
   rounds — use `units/<RUN>*`.
7. `torch.optim.Adam` here runs wd=0 and `load_state_dict` restores saved
   hyperparams — set `--lr` (it re-applies post-load) per run; add any new
   optimizer knob post-load too.
8. Image rebuild (after engine/deploy/script changes), from repo root:
   ```bash
   set -a; . ./.env; set +a
   echo "$GITHUB_GHCR_KEY" | docker login ghcr.io -u slcz --password-stdin
   docker buildx build --platform linux/amd64 --build-arg HF_TOKEN="$HF_TOKEN" \
       -t ghcr.io/slcz/chess-value:latest --push .
   ```
   Smoke first: one CPU game in the emulated image (see skill §rebuild).
   Rebuilt image ≠ running instances; they pick it up on recreate.
