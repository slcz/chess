# Salad runbook — chess value-net cloud run (side track)

End-to-end flow for the supervised value-net training on a salad.com
RTX 5090 (assumed GPU; the image is CUDA-12.8 and works on 3090/4090
too). Budget guardrail agreed in PLAN §0: **$5 / 8 h kill switch**
(5090 ≈ $0.50–0.70/h → stop by ~7 h).

Churn rules (PLAN §5): nothing durable lives on the instance.
Checkpoints push to HuggingFace every `--save-every` steps
(`HF_TOKEN`), `latest.pt` allows `--resume` on a fresh box, and the
pack is reproducible from the Lichess URL. **Instances WILL die**
(first one lasted ~1 h); recovery is the 3 steps below.

## 0. Field notes — the salad ssh proxy (measured 2026-09-24)

- ssh commands run via exec, not a remote shell: multi-command strings
  need `scripts/salad.sh ssh bash -c "'...'"` form.
- **Every descendant process dies when the ssh channel closes —
  `setsid`/`nohup`/`disown` do NOT survive.** Long jobs therefore run
  under **tmux on the node** (user rule 2026-09-26: the Mac sleeping or
  dropping the channel must not kill a run): launch detached
  (`tmux new-session -d -s <name> "bash script.sh > log 2>&1"`), watch
  via the log file, attach with `tmux attach -t <name>` over a fresh
  ssh. If even tmux dies on channel close, the checkpoint/resume design
  (atomic latest.pt + `--resume`) caps the loss at one save interval —
  re-launch the same tmux line.
- `scp` works (SFTP mode). `rsync` is not installed (and the project
  rule is scp anyway). First scp after boot can fail once — retry.
- Wrong image deployed? Adapt in place if it has python+torch:
  `apt-get install -y make gcc zstd; pip3 install cffi chess pytest
  huggingface_hub` — the engine builds from source in seconds.

## 1. Build & push the image (Mac) — DONE

Image live at `ghcr.io/slcz/chess-value:latest` (linux/amd64,
sha256:3c8e7d…). Rebuild only when Dockerfile/requirements change:

    source .env   # GITHUB_GHCR_KEY
    echo "$GITHUB_GHCR_KEY" | docker login ghcr.io -u slcz --password-stdin
    docker buildx build --platform linux/amd64 \
        -t ghcr.io/slcz/chess-value:latest --push .

## 2. Deploy on salad (user, console)

Container group with image `ghcr.io/slcz/chess-value:latest`
(check package visibility — private needs registry credentials in the
group config), GPU = RTX 5090, expose **TCP 22**, env
`SSH_PUBKEY=<~/.ssh/id_ed25519.pub contents>`. Then set `SSH_HOST` /
`SSH_PORT` in `.env` on the Mac.

## 3. Fresh-node recovery (3 steps, ~2 min of agent work)

    # a) ship the code (tarball through scp; image is env-only)
    scripts/salad.sh put /tmp/chess.tgz /workspace/chess.tgz
    scripts/salad.sh ssh bash -c "'mkdir -p /workspace/chess && tar xzf /workspace/chess.tgz -C /workspace/chess'"
    # b) bootstrap: engine (x86_64) + fast tests + CUDA check
    scripts/salad.sh ssh bash -c "'cd /workspace/chess && bash deploy/bootstrap.sh'"
    # c) the long run — user's terminal, kept open (see field notes):
    scripts/salad.sh ssh bash /workspace/chess/deploy/cloud_run.sh

(Regenerate /tmp/chess.tgz first if stale: see the tar line in the old
step 3 below.)

## 4. Dataset — handled inside cloud_run.sh

Download (Lichess 2016-01, ~872 MB) + preprocess (300k games, ~22M
positions) run on the node; lichess→datacenter beats Mac→salad upload.

## 5. Train — handled inside cloud_run.sh

20k steps × batch 1024 on cuda, val every 500 steps, atomic
checkpoints every 500 steps pushed to `chriszhang01/chess-value-net`
(HF_TOKEN via the shipped `.env`). Churn mid-run = repeat step 3;
`--resume` continues from `latest.pt`.

## 6. Evaluate + play

    python3 -m chess_ai.supervised.train_supervised eval  --ckpt runs/value_cloud/final.pt --pack data/lichess_2016-01.pack
    python3 -m chess_ai.supervised.train_supervised match --ckpt runs/value_cloud/final.pt --opponent greedy --n-games 20

Pull the net to the Mac for the GUI (scp), then:

    .venv/bin/python scripts/play_gui.py --black value --value-ckpt runs/value_cloud/final.pt

## 7. Kill switch (cost cap)

Delete/stop the container group in the salad console. The only state
worth keeping — `final.pt` — is already on HuggingFace.

---

### Old detailed step 3 (kept for the tar command)

    tar --exclude ./.git --exclude ./.venv --exclude ./doc \
        --exclude ./data --exclude ./runs --exclude ./engine/build \
        --exclude ./engine/build-asan --exclude '*/__pycache__' \
        --exclude ./.pytest_cache --exclude ./Dockerfile --exclude ./.dockerignore \
        -czf /tmp/chess.tgz .
    scp -P "$SSH_PORT" /tmp/chess.tgz root@"$SSH_HOST":/workspace/chess.tgz

NOTE: the tarball includes `.env` (HF_TOKEN) — needed on the instance
for checkpoint pushes; the repo never commits it.

---

# Self-play pilot (Steps 2–3, 2026-09-30) — same node, new loop

The supervised side track's node conventions all carry over (image,
tarball, tmux, kill switch). What's new: the chess self-play loop
(`chess_ai/train/selfplay_loop.py`) runs self-play + training in ONE
process and pushes every round to HF — a fresh box reconstructs state
from remote alone (PLAN §5 rule 1).

## P1. Deploy + launch

    tar -czf /tmp/chess.tgz .   # the old step-3 tar line above (includes .env)
    scripts/salad.sh put /tmp/chess.tgz /workspace/chess.tgz
    scripts/salad.sh ssh tar -xzf /workspace/chess.tgz -C /workspace/chess
    scripts/salad.sh ssh bash /workspace/chess/deploy/pilot.sh

pilot.sh: builds the engine, runs the self-play test slice, fetches the
RoPE seed ckpt from HF, launches `tmux` session **chesspilot** with
`--device cuda --eval-device cuda --workers 8 --buffer-requests 128
--games-per-iter 128 --train-steps 96 --resign -1 --hf-repo
chriszhang01/chess-selfplay-pilot`. Passthrough flags override (later
wins), e.g. `pilot.sh --n 8 --out runs/chess_pilot_n8`.

## P2. n-sweep (one-off probes, ~3 min each — NOT under tmux)

    scripts/salad.sh ssh 'cd /workspace/chess && python3 -m chess_ai.train.selfplay_loop \
        --out /tmp/probe_n8 --iters 1 --games-per-iter 64 --n 8 \
        --device cuda --eval-device cuda --workers 8 --buffer-requests 128 \
        --resign -1 --train-steps 4'
    # repeat with --n 16, --n 32; compare games_per_hour in the it line

Pick the pilot's n from the curve (Mac pilot: n=8 ≈ 2× n=16 games/hour;
the Gumbel paper's low-sim sweet spot), then launch the real pilot with
`pilot.sh --n <picked>`. Artifacts: `--out` under /tmp — throwaway.

## P3. Monitor

    scripts/salad.sh ssh tail -30 /workspace/chess/runs/chess_pilot.out
    scripts/salad.sh ssh tail -3 /workspace/chess/runs/chess_pilot/log.jsonl

Health signals per round: `games_per_hour`, `draw_rate`, `mean_plies`,
train `value_acc`/`policy_acc` trends. (Resignation is OFF for the pilot
— the Mac pilot showed `resign_value −0.90` collapses self-play with the
supervised seed: loop-induced value-head sharpening, 29–36% false
resigns by round 8 at ANY threshold.)

## P4. Churn drill (4.4 gate) — do once the pilot is a few rounds in

    scripts/salad.sh ssh pkill -9 -f selfplay_loop     # the "instance died"
    scripts/salad.sh ssh bash /workspace/chess/deploy/pilot.sh   # relaunch

PASS = the new run logs `resumed trainer … (steps N)` + `ingested K
completed unit(s)` and continues at the next round; loss = only the
in-flight round (its games replay deterministically into the O_TRUNC'd
shards). HF repo `chriszhang01/chess-selfplay-pilot` must show
`latest.pt`, `log.jsonl`, and `units/iter-*/` updating after each round.

## P5. Cost cap + stop

Stop the same way as the side track (console kill / container delete) or
just `tmux kill-session -t chesspilot` and let the box idle while deciding.
Target: 5–20k games ≈ a few $ at 5090 rates. Everything needed to resume
or analyze is on HF (latest.pt, round snapshots every 20 rounds, unit
shards, log.jsonl).
