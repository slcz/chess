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
- generation: 4
  data:
    old_round_parts_total: 53
    trimmed_parts: 13
    trimmed_games: 13288
    trim_pct: 24.1
    trim_policy: uniform random (seed 4)
    kept_old_games: 41920
    new_round_games: 32501
    buffer_games: 74421
    buffer_parts: 71
    manifest: selfplay-scratch/manifests/gen4.json
    new_round: g3-b449b158 (32,501 games, 40.5% draws)
  train:
    base: selfplay-scratch/gen3.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    rows: 10545483
    steps: 2575
    plies_per_game: 141.7
    precision: bf16+compile
    gpu: RTX 5090
    ab_test:
      vs: value-only (val 1.0 / pol 0 / aux 0)
      score: 0.515
      elo: 10.4
      sigma: 17.4
      verdict: "tie \u2014 kept the original scheme"
  loss:
  - step: 100
    loss: 4.0328
    policy: 2.6895
    value: 0.714
    aux: 2.0976
  - step: 1000
    loss: 3.9961
    policy: 2.6767
    value: 0.7023
    aux: 2.057
  - step: 2000
    loss: 3.9888
    policy: 2.674
    value: 0.7007
    aux: 2.0468
  - step: 2500
    loss: 3.9862
    policy: 2.6769
    value: 0.6978
    aux: 2.0382
  eval:
    games: 400
    score: 0.944
    wdl:
    - 359
    - 37
    - 4
    elo_delta: 490.7
    elo_sigma: 37.8
  adopted: true
  checkpoint:
    file: gen4.pt
    sha256: 826cc156e1d5cd93a8451f52ac30189a19044c37d268a01159e027a110645198
- generation: 5
  data:
    history: "none \u2014 clean cut, first buffer of the fixed-LEARN-target era"
    round: g4-826cc156
    round_games: 32270
    parts: 30
    trim: none
    manifest: selfplay-scratch/manifests/gen5.json
    round_wdl:
      W: 47.2
      D: 7.4
      L: 45.5
      adjudicated: 0.0
      mean_plies: 105.3
    also_vs_gen3:
      score: 0.995
      wdl:
      - 398
      - 0
      - 2
      games: 400
      elo: 919.5
      sigma: 123.0
      note: 'regression check requested by user: no regression'
  train:
    base: selfplay-scratch/gen4.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    rows: 3795749
    steps: 927
    plies_per_game: 117.6
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 3.7256
    policy: 2.5669
    value: 0.5401
    aux: 2.062
  - step: 200
    loss: 3.6652
    policy: 2.5484
    value: 0.5168
    aux: 1.9999
  - step: 900
    loss: 3.6153
    policy: 2.5261
    value: 0.5114
    aux: 1.926
  eval:
    games: 400
    score: 0.985
    wdl:
    - 394
    - 0
    - 6
    elo_delta: 726.9
    elo_sigma: 71.5
  adopted: true
  checkpoint:
    file: gen5.pt
    sha256: c10652f0bd727d65995e1aabfd4482c3c69af03834f57619607623ea4fc26218
- generation: 6
  data:
    previous_buffer: round 5, 30 parts, 32270 games
    trim: "uniform random 1/3 (seed 5) \u2014 changed from 1/4 at user request"
    dropped_games: 10868
    dropped_pct: 33.7
    kept_old_games: 21402
    new_round: g5-c10652f0 (38025 games, 35 parts)
    buffer_games: 59427
    buffer_parts: 55
    manifest: selfplay-scratch/manifests/gen6.json
    sf_floor:
      '1350': 0.131
      '1600': 0.057
      '1850': 0.05
      '2100': 0.025
      '2350': 0.008
      mean_plies: 54.8
      note: first movement off the floor (gen4 was 0.008 at 1350); sweep now monotone,
        crossover near 1300-1400 SF Elo
  train:
    base: selfplay-scratch/gen5.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    rows: 6660854
    steps: 1627
    plies_per_game: 112.1
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 3.8052
    policy: 2.6002
    value: 0.5862
    aux: 2.0628
  - step: 200
    loss: 3.7846
    policy: 2.5928
    value: 0.5793
    aux: 2.0415
  - step: 1600
    loss: 3.7136
    policy: 2.5532
    value: 0.5705
    aux: 1.9661
  eval:
    games: 400
    score: 0.931
    wdl:
    - 368
    - 9
    - 23
    elo_delta: 452.0
    elo_sigma: 34.3
  adopted: true
  checkpoint:
    file: gen6.pt
    sha256: f65d9bb7379b4458cf1f156bddb070c245e642ea37c6a96f7008e24d67e737b0
```
