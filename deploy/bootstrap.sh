#!/bin/bash
# bootstrap.sh — one-time setup INSIDE the chess-value container, after the
# repo has been scp'd to /workspace/chess (see scripts/salad_runbook.md).
#
#   ssh root@<node> '/workspace/chess/deploy/bootstrap.sh'
#
# Three jobs:
#   1. build the C engine for x86_64 from source (the Mac dylib is arm64
#      and never ships — PLAN §4.1)
#   2. run the fast test slice (parser/dataset/player invariants)
#   3. measure the two throughputs that size the training run:
#      position-encode rate (CPU) and CUDA step time (GPU)
set -e
cd /workspace/chess

echo '== 1/3 build C engine (x86_64, from source) =='
make -C engine

echo '== 2/3 fast test slice =='
python3 -m pytest -q tests/test_supervised.py tests/test_value_play.py

echo '== 3/3 CUDA + throughput check =='
python3 - <<'EOF'
import glob
import time

import torch

print("torch", torch.__version__,
      "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
if not torch.cuda.is_available():
    raise SystemExit("CUDA not visible — check the salad GPU config")

from chess_ai.net.model import ChessTransformer
from chess_ai.supervised.dataset import PositionSampler, load_pack
from chess_ai.train.trainer import TrainConfig, Trainer, tensors_to_torch

packs = sorted(glob.glob("data/*.pack"))
if not packs:
    print("no data/*.pack yet — run download+preprocess (runbook step 4)")
    raise SystemExit(0)
games = load_pack(packs[0])
print(f"pack {packs[0]}: {len(games)} games")

s = PositionSampler(games, seed=0, n_workers=4)
s.sample_batch(256)                                   # warmup
t0 = time.time(); s.sample_batch(1024)
print(f"encode batch-1024 (4 workers): {(time.time()-t0)*1e3:.0f} ms")

tr = Trainer(ChessTransformer(),
             TrainConfig(batch_size=1024, device="cuda",
                         policy_weight=0.0, aux_reply_weight=0.0))
tb = tensors_to_torch(s.sample_batch(1024), tr.device)
tr.train_step(tb)                                     # warmup + cudnn init
t0 = time.time()
for _ in range(10):
    tr.train_step(tb)
per = (time.time() - t0) / 10 * 1e3
print(f"CUDA train_step batch-1024: {per:.0f} ms  "
      f"(-> {20000 * per / 1000 / 60:.0f} min for 20k steps)")
EOF

echo '== bootstrap done — start training per runbook step 5 =='
