# Salad runbook — chess value-net cloud run (side track)

End-to-end flow for the supervised value-net training on a salad.com
RTX 5090 (assumed GPU; the image is CUDA-12.8 and works on 3090/4090
too). Budget guardrail agreed in PLAN §0: **$5 / 8 h kill switch**
(5090 ≈ $0.50–0.70/h → stop by ~7 h).

Churn rules (PLAN §5): nothing durable lives on the instance.
Checkpoints push to HuggingFace every `--save-every` steps
(`HF_TOKEN`), `latest.pt` allows `--resume` on a fresh box, and the
pack is reproducible from the Lichess URL.

## 1. Build & push the image (Mac)

    source .env   # GITHUB_GHCR_KEY
    echo "$GITHUB_GHCR_KEY" | docker login ghcr.io -u "$(git remote get-url origin | sed 's#.*github.com/##;s#/.*##')" --password-stdin
    docker buildx build --platform linux/amd64 \
        -t ghcr.io/<owner>/chess-value:latest --push .

`--platform linux/amd64` is required: the Mac is arm64, salad hosts are
x86_64. First build pulls multi-GB CUDA + torch layers (20–40 min under
emulation); the heavy layers cache for rebuilds.

## 2. Deploy on salad

Container group with image `ghcr.io/<owner>/chess-value:latest`,
GPU = RTX 5090 (or 4090/3090 — image is arch-agnostic within CUDA 12.8
nodes; 3090 is sm_86, covered by cu128 wheels), expose **TCP 22**, env:

    SSH_PUBKEY=<contents of ~/.ssh/id_ed25519.pub>

After start: `ssh root@<node-ip> -p <port>` (salad maps a public port).

## 3. Ship the code (Mac → instance, scp per project rule)

    tar --exclude .git --exclude .venv --exclude doc \
        --exclude engine/build --exclude engine/build-asan \
        --exclude data --exclude runs --exclude '__pycache__' \
        -czf /tmp/chess.tgz .
    scp /tmp/chess.tgz root@<node>:/workspace/
    ssh root@<node> 'mkdir -p /workspace/chess && tar xzf /workspace/chess.tgz -C /workspace/chess'

NOTE: the tarball includes `.env` (HF_TOKEN) — it is needed on the
instance for checkpoint pushes and the repo never commits it. Skip
`--exclude .env` only if you accept that copy.

## 4. Dataset + engine bootstrap (on instance)

    ssh root@<node> '/workspace/chess/deploy/bootstrap.sh'   # engine + tests + throughput

The dataset is fetched on the instance (lichess → datacenter is much
faster than Mac → salad upload):

    ssh root@<node>
    cd /workspace/chess
    python3 -m chess_ai.supervised.train_value download --month 2016-01
    python3 -m chess_ai.supervised.train_value preprocess \
        --pgn data/lichess_2016-01.pgn.zst --out data/lichess_2016-01.pack \
        --min-elo 1600 --max-games 300000

(~5 min for a 300k-game prefix on the 2016-01 dump, ~22M positions.)
Alternative for a quick first run: scp the Mac pilot pack
(`data/lichess_2013-01.pack`, 1.8M positions) and skip the download.

## 5. Train (on instance)

    cd /workspace/chess && source .env
    nohup python3 -m chess_ai.supervised.train_value train \
        --pack data/lichess_2016-01.pack --out runs/value_cloud \
        --steps 20000 --batch-size 1024 --lr 3e-4 --device cuda \
        --n-workers 4 --val-every 500 --save-every 500 --log-every 50 \
        --hf-repo <hf-user>/chess-value-net \
        > runs/value_cloud.out 2>&1 &

Expected (measured shape, MPS pilot): value acc climbs past prior within
a few hundred steps; 20k steps × batch 1024 ≈ 11 epochs of a 22M-position
pack ≈ well past it. Watch it:

    tail -f runs/value_cloud/train.log     # or: ssh ... 'tail -f ...'

Churn recovery on a fresh instance: repeat 3–4, then the same train
command with `--resume` (optimizer state + step counter ride in
`latest.pt`; the HF copy is the off-instance backup).

## 6. Evaluate + play

    python3 -m chess_ai.supervised.train_value eval  --ckpt runs/value_cloud/final.pt --pack data/lichess_2016-01.pack
    python3 -m chess_ai.supervised.train_value match --ckpt runs/value_cloud/final.pt --opponent greedy --n-games 20

Pull the net to the Mac for the GUI (scp), then:

    .venv/bin/python scripts/play_gui.py --black value --value-ckpt runs/value_cloud/final.pt

## 7. Kill switch (cost cap)

Delete/stop the container group in the salad console. The only state
worth keeping — `final.pt` — is already on HuggingFace.
