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
  `setsid`/`nohup`/`disown` do NOT survive.** Long jobs must run inside
  an OPEN interactive session (that's why `deploy/cloud_run.sh` is a
  foreground script: the user runs it and keeps the window open).
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

    python3 -m chess_ai.supervised.train_value eval  --ckpt runs/value_cloud/final.pt --pack data/lichess_2016-01.pack
    python3 -m chess_ai.supervised.train_value match --ckpt runs/value_cloud/final.pt --opponent greedy --n-games 20

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
