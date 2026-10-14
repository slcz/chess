---
name: salad-selfplay
description: >
  Operate the chess repo's autonomous self-play pipeline on salad.com:
  playout cap randomization (PCR) semantics, the boot-and-generate container
  image, launching/monitoring/stopping a round, and training on the round's
  units. Use whenever the user mentions self-play rounds, PCR / playout cap
  randomization, sp_gen / sp_pusher / sp_progress / units on HuggingFace,
  rebuilding or pushing the chess-value container image, shard v1/v2, or
  salad instances for this project — even if they just say "start another
  round" or "the image needs the new engine".
---

# Salad self-play rounds (PCR container pipeline)

One round = N salad instances each boot `ghcr.io/slcz/chess-value:latest`,
generate games (self-play + HF push) until their per-instance target, and
exit; training later samples only the slow-cap plies from the units on HF.

## Invariants (violating these breaks the pipeline subtly)

- **Only slow-cap plies train.** The rule is `ply.sims == game.n_sims`
  (`chess_ai/train/selfplay_data.ply_trainable`). Fast plies are replayed
  through (no per-ply FENs exist) but never become samples; they still
  serve as aux-reply targets. Never "fix" training to include fast plies.
- **PCR draws one budget per mux step for the WHOLE pool**, never per move.
  The games think in lockstep (`puct_mux_step`): a per-move draw would make
  every step wait on the slowest game, defeating the speedup. Per-step
  gives the identical per-ply marginal with full-pool GPU batches.
- **A restarted container is a clean slate.** Assume nothing on its
  filesystem survives. Identity is a boot stamp
  `<SALAD_INSTANCE_ID>-<epoch>` → seed = FNV-1a-64(stamp), HF prefix
  `units/<RUN>-i<stamp>/`. Never reuse a seed or prefix; never resume from
  container-local state. Round state lives on HF only.
- **The base checkpoint is the file baked into the image**
  (`runs/gen1.pt`; `runs/ckpt_a/net.pt` is a compat symlink to it —
  the pre-2026-10-13 name was `runs/policy_value_net_42k.pt`).
  It is the only true 42k net: HF's `policy_value/final.pt` is a broken
  404 upload and `policy_value/latest.pt` is only 40k steps. Don't swap
  bases without the user asking.
- **Shard format v2 carries per-ply sims** (one u16 per ply; byte layout
  authority: `engine/include/chess_sp_dump.h`). v1 shards must stay
  readable — the reader fills each ply's sims from the game cap so old
  units read as all-trainable. `sp_pusher.py`, `shard.py` accept both
  versions; a new version needs both updated plus
  `tests/test_shard_format.py`.

## Rebuild + push the image (after any engine/`deploy/`/`scripts/` change)

Docker Desktop must be running (`open -a Docker` if not). From the repo
root, with `.env` present:

```bash
source .env   # GITHUB_GHCR_KEY + HF_TOKEN (baked into the image)
echo "$GITHUB_GHCR_KEY" | docker login ghcr.io -u slcz --password-stdin
docker buildx build --platform linux/amd64 --build-arg HF_TOKEN="$HF_TOKEN" \
    -t ghcr.io/slcz/chess-value:latest --push .
```

The image carries the repo, the x86_64 engine compiled IN-image, and the
baked checkpoint. Things that have bitten before:

- `python3-dev` must stay in the apt line — the Makefile calls the
  unversioned `python3-config` (`python3.10-dev` alone doesn't provide it).
- Build `make -C engine all build/selfplay`, not just `selfplay`: the
  engine's embedded Python imports `chess_ai.*` at NN attach, which loads
  `libchess.so`.
- Expect QEMU-emulated compile + a full torch wheel re-download when the
  buildx cache is pruned (~10 min total). `.env` is dockerignored — tokens
  never enter the image. Instances pull `:latest` only at container
  creation: running instances keep the old image until the group is
  restarted/redeployed, which is safe (a restarted instance starts a
  fresh unit) but forfeits its current <1000-game part.
- Smoke with one real game through ckpt load, the eval seam and a shard
  write — validated command (CPU, in the emulated image, ~40 s):

  ```bash
  docker run --rm --platform linux/amd64 --entrypoint /bin/bash \
      ghcr.io/slcz/chess-value:latest -c \
      'timeout 240 ./engine/build/selfplay runs/gen1.pt /tmp/smoke \
       1 4 1 64 30 60 4 0.25 0.3 cpu 1 2>&1 | tail -2'
  ```

  PCR behavior itself is pinned locally by `tests/test_selfplay_roundtrip.py`.

## Launch a round (user does this in the salad console)

Image `ghcr.io/slcz/chess-value:latest`, expose TCP 22, container-group env
— **all optional**: the upload token ships baked in the image (build-time
`--build-arg` from `.env`; the GHCR package must stay private; set env
`HF_TOKEN` only to override, and note a token rotation requires a rebuild):

| Env | Meaning |
|---|---|
| `SSH_PUBKEY` | optional (no shell without it, but the job still runs) |
| `HF_TOKEN` | baked in at build; set only to override |
| `RUN` | run name, default `sp-42k-pcr` → prefix `units/<RUN>-i<stamp>/`. **Use a fresh name per round** — the monitor and the training prefix both scope by RUN, so a reused name mixes rounds and corrupts the target math |
| `GAMES` | **per-instance** target; set `ceil(total / n_instances)`, and remember the value — the monitor's per-stamp lines are only interpretable against it |
| `SIMS`/`FAST_SIMS`/`FAST_P` | PCR knobs, default `600`/`50`/`0.75` |

If the GHCR package is private, the container group also needs registry
credentials for the pull (runbook §2).

On boot the container preflights (ckpt present → HF_TOKEN → CUDA), then
runs `scripts/sp_gen.sh` (generation + pusher uploading ~1K-game parts) and
exits. A dead instance loses only its current <1000-game part; a
replacement starts a fresh unit automatically.

## Monitor (from the Mac, works mid-flight)

```bash
.venv/bin/python scripts/sp_progress.py --run sp-42k-pcr --target 25000 --watch 300
```

Reads part `DONE.json`s from HF — per-stamp games/seeds + total vs target.
No state to check on the instances themselves.

## If progress stalls

Run the monitor once without `--watch` and read the per-stamp lines
against the `GAMES` you launched with: a stamp sitting at exactly `GAMES`
means that unit **completed**, not stalled — a problem is a total below
target with nothing moving across two runs minutes apart. Where to look
next, cheapest first:

- **What died?** Pusher output goes ONLY to the container logs (salad
  console) — there is no pusher log file. Generation logs are on the
  instance at `/workspace/sp/<RUN>-i<stamp>/gen.log` (ssh host/port from
  the console; preflight failures print `[autorun] FATAL: …`).
- **Pusher died, engine still running** → nothing uploads until you act.
  The pusher is crash-safe (resumes from `pushed.json`): ssh in, install
  tmux (not in the image), rerun `scripts/sp_pusher.py` with the same
  arguments. Loss is bounded by games since the last successful part.
- **Everything frozen simultaneously** → group-level event (stopped,
  env edit redeploying all containers, shared token failure). Note an env
  edit restarts every container with fresh stamps — safe, fresh units.
- **Recovery is always**: relaunch with the same `RUN` and
  `GAMES = ceil(remaining / n_instances)`; fresh stamps make relaunching
  safe, and ingest dedups by `(unit_seed, game_index)` regardless.

## Stop + train

When the target is met: stop the container group in the salad console
(kill switch), then train on one node (code is already in the image; ssh
in, the units come from HF):

```bash
python3 scripts/sp_round_train.py --prefix units/sp-42k-pcr \
    --base-ckpt runs/gen1.pt --out runs/iterN --epochs 2 --lr 2e-5 --batch 1024
```

The trainer filters to slow-cap plies automatically and logs
"N trainable (of M recorded plies)". Round acceptance (PCR baseline):
candidate ≥ 50% vs the 42k net at n=200 over 400 games
(`scripts/gate_match.py` / `sp_round_eval.sh`).

## Local dev loop (C/Python changes)

`make -C engine build/selfplay`, then
`.venv/bin/python -m pytest tests/test_selfplay_roundtrip.py
tests/test_shard_format.py -q` — the roundtrip runs the real binary with
PCR on (fast=4@75% of slow=16) end-to-end on MPS in ~5 s.

Design rationale (why per-step draws, the v2 byte layout, the efficiency
math): read `references/pcr-and-shard-v2.md` before changing any of it.
