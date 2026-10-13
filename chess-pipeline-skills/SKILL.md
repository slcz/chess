---
name: chess-pipeline-skills
description: >
  Customized skills for the deepseek harness (dsh) driving the chess
  value-net pipeline on salad.com GPU boxes: building the pipeline
  container images, operating a self-play generation round (launch,
  monitor, stall-diagnosis, stop), and running the train/evaluate loop.
---

## Pipeline invariants (shared — violating these breaks the pipeline subtly)

- **Only slow-cap plies train**: the rule is `ply.sims == game.n_sims`
  (~25% of plies at FAST_P=0.75); fast plies are replayed through as
  aux-reply targets only. Never "fix" training to include them.
- **PCR draws one budget per mux step for the WHOLE pool**, never per move
  — games think in lockstep so every step batchs the full GPU.
- **A restarted container is a clean slate.** Identity is a boot stamp
  `<SALAD_INSTANCE_ID>-<epoch>` → seed = FNV-1a-64(stamp), HF prefix
  `units/<RUN>-i<stamp>/`. Never reuse a seed or prefix; round state lives
  on HF only.
- **The baked base checkpoint is the only true 42k net**
  (`runs/policy_value_net_42k.pt` = `runs/ckpt_a/net.pt` in-image). HF's
  `policy_value/final.pt` is a broken 404 upload; `latest.pt` is 40k steps.
- **Shard format v2 carries per-ply sims** (u16; byte-layout authority:
  `engine/include/chess_sp_dump.h`). v1 shards stay readable (every ply
  reads as all-trainable). A format change needs `sp_pusher.py`,
  `chess_ai/selfplay/shard.py`, and `tests/test_shard_format.py` together.
- Design rationale (per-step draws, v2 byte layout, efficiency math):
  `.agents/skills/salad-selfplay/references/pcr-and-shard-v2.md` — read
  before changing any of it.

# Skill: create-selfplay-container

Use when a pipeline container image must be (re)built and pushed to GHCR —
after any engine, `deploy/`, `scripts/`, `chess_ai/` change, or a token
rotation. Two images share the recipe:

| Image | Dockerfile | Boots into | Baked extras |
|---|---|---|---|
| `ghcr.io/slcz/chess-value:latest` | `Dockerfile` | autonomous self-play (generates → pushes to HF → exits) | 42k checkpoint (gen1) |
| `ghcr.io/slcz/chess-train:latest` | `Dockerfile.train` | **idle** (`sleep infinity`, ssh workbench — no autorun) | 42k ckpt, `data/noob_3moves.epd`, stockfish (+PATH), tmux, `uci_puct`, `sf_gauntlet`, current trainer w/ packed ingest |

## Steps

1. Docker Desktop must be running (`open -a Docker` if not). From the repo
   root with `.env` present (`GITHUB_GHCR_KEY`, `HF_TOKEN`):
   ```bash
   set -a; . ./.env; set +a
   echo "$GITHUB_GHCR_KEY" | docker login ghcr.io -u slcz --password-stdin
   docker buildx build --platform linux/amd64 -f Dockerfile \
       --build-arg HF_TOKEN="$HF_TOKEN" \
       -t ghcr.io/slcz/chess-value:latest --push .
   # workbench variant: -f Dockerfile.train -t ghcr.io/slcz/chess-train:latest
   ```
2. Expect QEMU-emulated compile + a full torch wheel download when caches
   are cold (~10–25 min). `Dockerfile.train` keeps the base layers
   byte-identical to `Dockerfile`, so the heavy torch layer stays cached.
3. Smoke before declaring success:
   - self-play image (CPU, emulated, ~40 s; a libpython failure exits 127):
     ```bash
     docker run --rm --platform linux/amd64 --entrypoint /bin/bash \
         ghcr.io/slcz/chess-value:latest -c \
         'timeout 240 ./engine/build/selfplay runs/ckpt_a/net.pt /tmp/smoke \
          1 4 1 64 30 60 4 0.25 0.3 cpu 1 2>&1 | tail -2'
     ```
   - train/eval image:
     ```bash
     docker run --rm --platform linux/amd64 ghcr.io/slcz/chess-train:latest \
         bash -c 'ls data/noob_3moves.epd engine/build/sf_gauntlet \
                  engine/build/uci_puct >/dev/null && which stockfish \
                  && grep -q PackedRound chess_ai/train/selfplay_data.py \
                  && echo SMOKE-OK'
     ```
4. Salad deploy (console): expose TCP 22, set `SSH_PUBKEY`. `HF_TOKEN` is
   baked at build time (`--build-arg` from `.env`; GHCR package must stay
   private — the token is recoverable from image config; rotation =
   rebuild). Self-play image env knobs: `RUN` (fresh name per round!),
   `GAMES`, `SIMS`/`FAST_SIMS`/`FAST_P` (PCR, default 300/50/0.75),
   `SAMPLE_PLYS` (opening move-sampling window, default 10). The train
   image takes NO job env — it boots idle; stop the group when done (it
   bills while it sleeps).
5. Running instances keep the old image until recreated — safe (a
   recreated self-play container starts a fresh unit; ingest dedups by
   `(unit_seed, game_index)`).

## Gotchas

- `python3-dev` must stay in the apt line (Makefile calls the unversioned
  `python3-config`); build `all build/selfplay …`, not just `selfplay`
  (the engine's embedded python imports `chess_ai.*` → needs libchess).
- `.env` is dockerignored — tokens never enter the context; the 42k
  checkpoint is re-included via `.dockerignore` negations (`runs/*`,
  `data/*` patterns — docker cannot un-exclude a file whose parent dir is
  excluded).
- PCR semantics are pinned by `tests/test_selfplay_roundtrip.py`; run it
  locally after engine changes: `make -C engine build/selfplay && .venv/
  bin/python -m pytest tests/test_selfplay_roundtrip.py -q`.

---

# Skill: run-selfplay-round

Use to launch, monitor, diagnose, and stop one self-play generation round
on salad.com. Prerequisite: the chess-value image deployed per
create-selfplay-container (TCP 22, `SSH_PUBKEY`; private GHCR → the
container group also needs registry credentials for the pull).

## Launch (salad console)

- Container-group env: `RUN` (**fresh name per round** — the monitor and
  the training prefix both scope by it; a reused name mixes rounds),
  `GAMES` = ceil(total target / n_instances) per instance,
  `SIMS`/`FAST_SIMS`/`FAST_P` (default 300/50/0.75), `SAMPLE_PLYS`
  (default 10). `HF_TOKEN` is baked; set it only to override.
- On boot the container preflights (ckpt → HF_TOKEN → CUDA; failures print
  `[autorun] FATAL: …` to the container logs), then runs generation +
  pusher (parts land on HF every ~1000 games) and exits at its target.

## Monitor (from the Mac, works mid-flight)

```bash
.venv/bin/python scripts/sp_progress.py --run <RUN> --target <TOTAL> [--watch 300]
```
Reads part `DONE.json`s from HF — per-stamp games/seeds + total vs target.
No state to check on the instances themselves.

## If progress stalls

A stamp sitting at exactly `GAMES` means that unit **completed**, not
stalled. A problem is a total below target with nothing moving across two
checks minutes apart. Cheapest checks first:

- **What died?** Pusher output goes ONLY to the container logs (salad
  console) — there is no pusher log file. Generation logs are on the
  instance at `/workspace/sp/<RUN>-i<stamp>/gen.log` (ssh host/port from
  the console).
- **Pusher died, engine still alive** → nothing uploads until you act. The
  pusher is crash-safe (resumes from `pushed.json`): ssh in, install tmux
  (not in this image), rerun `scripts/sp_pusher.py <unit-dir>
  chriszhang01/chess-selfplay units/<RUN>-i<stamp> --every 1000
  --interval 60`. Loss is bounded by games since the last part.
- **Everything frozen simultaneously** → group-level event (stopped, env
  edit redeploying all containers, shared token failure). An env edit
  restarts every container with fresh stamps — safe, fresh units.
- **Recovery is always**: relaunch with the same `RUN` and
  `GAMES = ceil(remaining / n_instances)`; fresh stamps make relaunching
  safe, and ingest dedups by `(unit_seed, game_index)` regardless.
- Salvage a killed unit's tail (≤ <1000 un-pushed games) before recreating:
  `python3 scripts/sp_pusher.py <unit-dir> chriszhang01/chess-selfplay
  units/<stamp> --every 1 --once`.

## Stop

Stop the container group in the salad console when the target is met
(kill switch — it bills while alive), then hand off to train-and-eval.

---

# Skill: train-and-eval

Use to run a training round over a finished self-play generation and
evaluate the result. Prerequisites: a `chess-train` container group
deployed (see create-selfplay-container), its host/port in Mac `.env` as
`SSH_HOST`/`SSH_PORT`, and the generation's units on HF
(`chriszhang01/chess-selfplay`, prefix `units/<RUN>`).

## 1. Train

```bash
scripts/salad.sh ssh bash -c "'cd /workspace/chess && setsid nohup \
    python3 scripts/sp_round_train.py --prefix units/<RUN> \
    --base-ckpt runs/ckpt_a/net.pt --out runs/<name> \
    --epochs 2 --lr 2e-5 --batch 1024 \
    > /workspace/train-<name>.log 2>&1 < /dev/null & echo launched'"
scripts/salad.sh ssh 'tail -2 /workspace/train-<name>.log'      # monitor
```

- **Detach is mandatory** — the salad ssh gateway kills descendants when
  the channel closes.
- Ingest streams parts into flat arrays (`PackedRound`): 280k games ≈
  4.5 GB during parse, ~7 GB steady; a 31 GB box is comfortable. Parallel
  packers (`--workers 8`) + disk cache (`packed-cache/`, keyed by the unit
  listing): first launch ~2–3 min to step 1, re-launches ~40 s.
- Reference points: gen2 = 42k base, 280k PCR games (300/50 sims),
  lr 2e-5, batch 1024, 2 epochs → gate 0.639. Only slow-cap plies train
  (≈ 25% of plies — `ply.sims == game.n_sims`); that filter is by design.
- Epoch checkpoints `<out>-e1/-e2.pt` are pushed to
  `chriszhang01/chess-value-net` (`selfplay/`) at each epoch end — box
  death costs at most one epoch.
- Healthy startup lines: `ingested N games, M positions from K parts` →
  `T trainable positions (of M recorded plies)` → `step 100/…`.

## 2. Gate eval (candidate vs base)

```bash
scripts/salad.sh ssh bash -c "'cd /workspace/chess && setsid nohup \
    scripts/sp_round_eval.sh runs/<name>-e2.pt runs/ckpt_a/net.pt 400 \
    > /workspace/eval-<name>.log 2>&1 < /dev/null & echo launched'"
scripts/salad.sh ssh 'grep -aE "^(SUMMARY|TOTAL)" /workspace/eval-<name>.log'
```
- 400 games @ 200 sims, colors swapped per wave off `data/noob_3moves.epd`
  (baked in chess-train; scp it onto chess-value boxes).
- Acceptance: TOTAL `score_a ≥ 0.500`. ~14 min on a 3090. Per-wave SUMMARY
  lines should agree in direction (color artifact check).

## 3. Stockfish Elo ladder (optional, strength estimate)

Baked in chess-train (older boxes: `make -C engine build/sf_gauntlet`;
`ln -sf /usr/games/stockfish /usr/local/bin/stockfish`).
```bash
export NN_BF16=1 NN_COMPILE=1
engine/build/sf_gauntlet runs/<name>-e2.pt 256 1350,1600,1850,2100,2350 \
    60 30 200 10 300 stockfish cuda 0 7 > /workspace/sf-<name>.log 2>&1
grep '^SUMMARY' /workspace/sf-<name>.log     # score per Elo rung
```
Read the crossover rung (score crossing 0.500). Reference (3090, SF 14.1,
movetime 200, sims 256, 60 games/rung): gen1 crossover ≈ 1730, gen2 ≈
1835. **Always rerun the base net on the same box** — SF builds and
hardware shift absolute numbers.

## 4. Bookkeeping

- Fetch/checkpoint push helpers: `hf download chriszhang01/chess-value-net
  selfplay/<file>.pt --local-dir .` (token from `.env`/baked env).
- Rename on HF = upload under the new name FIRST, delete the old path
  only after the upload succeeded. Generation naming: gen1 = 42k base
  (`selfplay/selfplay-gen1.pt`), gen2 = first self-play iteration's e2
  (`selfplay/selfplay-gen2.pt`).
- Gotchas: fresh `RUN` name per generation (monitor + `--prefix` scope by
  it); a fresh container = the image's code (chess-train is current;
  chess-value boxes need `scripts/salad.sh put` of
  `scripts/sp_round_train.py` + `chess_ai/train/selfplay_data.py` +
  `data/noob_3moves.epd` until its next rebuild); `hf download --include
  "units/<RUN>/**"` matches nothing on container rounds — use
  `units/<RUN>*`; Adam here runs wd=0 and `load_state_dict` restores saved
  hyperparams — `--lr` is re-applied post-load, any new optimizer knob
  must be too.
- Local sanity loop (Mac, MPS): `make -C engine build/selfplay && .venv/
  bin/python -m pytest tests/test_packed_round.py
  tests/test_selfplay_roundtrip.py -q`.
