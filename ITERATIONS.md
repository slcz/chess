# Scratch lineage iterations

One YAML record per iteration, appended by `scripts/scratch_lineage.py approve`.
`elo_delta` is the new checkpoint's relative strength against the **previously
approved** net, computed from the 400-game gate score (200 sims both sides,
fixed opening book, colours swapped):

```
elo_delta = 400 * log10(s / (1 - s))        s = new net's score
sigma     = 400 / ln(10) / sqrt(n * s * (1 - s))
```

At `n = 400` that is ±17–18 Elo by binomial propagation, but our measured
gen1-vs-gen1 null control was 0.500 with waves 0.468/0.532, so the *systematic*
floor is nearer ±25–30 Elo. Treat |elo_delta| below ~40 as noise; ±40 is the
adoption bar.

```yaml
- generation: 1
  lineage: scratch
  data:
    parent: null
    games: 0
    note: randomly initialised spiral rope2d + RopePolicyHead (seed 0); no self-play
      yet
  train:
    init: random
    params: 6679569
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3)
  loss: null
  eval: null
  adopted: true
  checkpoint:
    file: selfplay-scratch/gen1.pt
    sha256: 56cae5b2a13e4973a9eee6cbb7a7140246777e02e809539fa7106b9a2563d568
- generation: 2
  lineage: scratch
  data:
    available_games: 31258
    used_games: 30000
    units: 7
    round: g1-56cae5b2-sp6k
    buffer: 30000 (first iteration, no trim)
    anchor_gate: "skipped by user \u2014 too early"
  train:
    base: selfplay-scratch/gen1.pt (random init)
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_weight: 0.15
    rows: 3824143
    steps: 934
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 4.4129
    policy: 2.9719
    value: 0.7306
    aux: 2.3681
  - step: 400
    loss: 4.2027
    policy: 2.9074
    value: 0.669
    aux: 2.0875
  - step: 700
    loss: 4.1927
    policy: 2.9055
    value: 0.6657
    aux: 2.0716
  - step: 900
    loss: 4.1891
    policy: 2.904
    value: 0.6667
    aux: 2.0612
  eval:
    games: 400
    score: 0.754
    wdl:
    - 204
    - 195
    - 1
    elo_delta: 194.6
    elo_sigma: 20.2
  adopted: true
  checkpoint:
    file: gen2.pt
    sha256: 3ddbd3cf426eab2b63630c61b55972da2e692babf757c34a8683ac350b55083b
- generation: 3
  data:
    old_round_parts_total: 30
    trimmed_parts: 8
    trimmed_games: 8130
    trim_pct: 26.0
    trim_policy: uniform random (seed 0)
    kept_old_games: 23128
    new_round_games: 32080
    buffer_games: 55208
    buffer_parts: 53
    manifest: selfplay-scratch/manifests/gen3.json
  train:
    base: selfplay-scratch/gen2.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_weight: 0.15
    rows: 8065767
    steps: 1970
    plies_per_game: 146.1
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 4.0032
    policy: 2.7726
    value: 0.6018
    aux: 2.096
  - step: 500
    loss: 3.9825
    policy: 2.7654
    value: 0.5934
    aux: 2.0788
  - step: 1000
    loss: 3.9722
    policy: 2.7626
    value: 0.5882
    aux: 2.0713
  - step: 1500
    loss: 3.9719
    policy: 2.764
    value: 0.5891
    aux: 2.0625
  - step: 1900
    loss: 3.9613
    policy: 2.7609
    value: 0.5833
    aux: 2.057
  eval:
    games: 400
    score: 0.741
    wdl:
    - 197
    - 199
    - 4
    elo_delta: 182.6
    elo_sigma: 19.8
  adopted: true
  checkpoint:
    file: gen3.pt
    sha256: b449b1580536963e78bd41724b02620fa3ff8a19cdc4283fad1196220102b1aa
```
