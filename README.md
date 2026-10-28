# chess — a self-play chess AI

An AlphaZero-style chess engine that improves by playing against itself.
The engine is built around a small 8-layer transformer neural network
(~6.7M parameters). Its value head outputs a board value, and its policy
head outputs a probability distribution over moves. As in AlphaZero, the
engine improves by playing games against itself and using Monte Carlo
tree search guided by the value and policy heads.

The engine is designed to be resource-efficient. Starting from a randomly
initialized network, it reaches roughly 1900–2000 Elo on a single
consumer-grade GPU in about 60 hours (37 self-play iterations).

To keep self-play economical, the engine uses the playout cap
randomization idea (David J. Wu, KataGo). Given a fixed compute budget,
it generates more games by running a full search on only a small number
of plies, searching most others with a smaller budget, and training the
network only on the fully searched plies. The network is trained on the
25% of plies that receive 300 simulations each; the remaining fast plies
use 50 simulations.

Instead of AlphaZero's PUCT selection rule, the engine's MCTS search uses
**regularized policy optimization** (Grill et al., 2020): PUCT's
hand-crafted formula approximates the solution to a regularized
policy-optimization objective. Here, π̄ is computed exactly as

$$
\bar{\pi} = \arg\max_y \left[ q^T y - \lambda_N \,\mathrm{KL}(\pi_\theta, y) \right].
$$

The same π̄ is the policy-head training target, the distribution sampled
during MCTS descent, and the played move (sampled for the opening plies,
argmax afterwards). This has a useful property: there is no Dirichlet
noise, no temperature, no visit-count target, and no hand-crafted
exploration tricks. Because π̄ is computed directly from Q-values, it
also works well with playout cap randomization and the fast plies that
use a smaller search budget. AlphaZero instead uses visit counts to
estimate Q-values, which intertwines search and learning.

## Build and test

To set up the environment and build the C binaries:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

make -C engine all
make -C engine check

# native NN drivers: self-play, the gate server, and the Stockfish gauntlet
make -C engine build/selfplay build/uci_mux build/sf_gauntlet

python -m pytest tests/ -q
```

To run the self-play → training → evaluation loop:

```sh
mkdir -p data
curl -fL -o /tmp/noob_3moves.epd.zip \
  https://github.com/official-stockfish/books/raw/master/noob_3moves.epd.zip
unzip -p /tmp/noob_3moves.epd.zip > data/book.epd
rm /tmp/noob_3moves.epd.zip

python scripts/selfplay_loop.py \
  --work-dir /data/az-run \
  --book data/book.epd \
  --iterations 37 --games 30000 --device cuda
```

`--init-ckpt` is optional: if it is omitted and `best.pt` does not exist,
the loop creates a random-init checkpoint in the work directory. On
macOS, use `--device mps`. The opening book is used only for evaluation;
self-play generates games by sampling from π̄.

## Web GUI

You can try the trained network in the [web GUI](https://slcz.github.io/chess/).

```sh
# ORT-web runtime, ~42 MB, fetched once into web/vendor/
sh web/get_vendor.sh

# Emscripten (once, macOS example)
git clone https://github.com/emscripten-core/emsdk.git ~/dev/emsdk
cd ~/dev/emsdk && ./emsdk install latest && ./emsdk activate latest
cd -

# Build wasm. On macOS the wrapper handles emsdk + Python 3.12:
web/build.sh
# On Linux, source the emsdk env and call make directly:
# source ~/dev/emsdk/emsdk_env.sh && make -C engine wasm
```

To serve the web page:

```sh
python3 web/serve.py
```
