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
- generation: 7
  data:
    previous_buffer: gen6 buffer, 55 parts, 59427 games
    trim: uniform random 1/3 (seed 6)
    dropped_games: 19821
    dropped_pct: 33.4
    kept_old_games: 39606
    new_round: g6-f65d9bb7 (31461 games, 29 parts)
    buffer_games: 71067
    buffer_parts: 66
    manifest: selfplay-scratch/manifests/gen7.json
    sf_floor:
      '1350': 0.311
      '1600': 0.058
      '1850': 0.066
      '2100': 0.024
      '2350': 0.033
      1350_wdl:
      - 17
      - 4
      - 40
      1350_games: 61
      mean_plies: 59.1
      games_total: 306
      elo_vs_sf1350: -138
      absolute_band: "~1150-1250 on Stockfish's UCI_Elo scale (1350 cell implies ~1212;\
        \ a 1600 cell implies ~1116 \u2014 SF's own Elo settings are not evenly spaced,\
        \ so use the band)"
  train:
    base: selfplay-scratch/gen6.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    rows: 7802860
    steps: 1905
    plies_per_game: 109.8
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 3.7335
    policy: 2.5332
    value: 0.6008
    aux: 1.9986
  - step: 200
    loss: 3.7273
    policy: 2.5292
    value: 0.6015
    aux: 1.9887
  - step: 1900
    loss: 3.6777
    policy: 2.509
    value: 0.5924
    aux: 1.9213
  eval:
    games: 400
    score: 0.879
    wdl:
    - 343
    - 17
    - 40
    elo_delta: 344.5
    elo_sigma: 26.6
  adopted: true
  checkpoint:
    file: gen7.pt
    sha256: ffdbe9c778489e5db311c746e0421bf51de95cd4a356a3807ff96d83fc3b006c
- generation: 8
  data:
    previous_buffer: gen7 buffer, 66 parts, 71067 games
    trim: uniform random 1/3 (seed 7)
    dropped_games: 23803
    dropped_pct: 33.5
    kept_old_games: 47264
    new_round: g7-ffdbe9c7 (33780 games, 31 parts)
    buffer_games: 81044
    buffer_parts: 75
    manifest: selfplay-scratch/manifests/gen8.json
    sf_floor:
      '1350': 0.5
      '1600': 0.238
      '1850': 0.067
      '2100': 0.008
      '2350': 0.025
      1350_wdl:
      - 28
      - 4
      - 28
      1350_games: 60
      mean_plies: 66.6
      games_total: 302
      note: CROSSOVER at SF@1350 (exact 0.500); the 1600 cell independently implies
        ~1398, so both readings agree on ~1350-1400 on Stockfish UCI_Elo scale
  train:
    base: selfplay-scratch/gen7.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    rows: 8839120
    steps: 2158
    plies_per_game: 109.0
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 3.6784
    policy: 2.454
    value: 0.6368
    aux: 1.9587
  - step: 1000
    loss: 3.6464
    policy: 2.4396
    value: 0.6319
    aux: 1.9163
  - step: 2000
    loss: 3.625
    policy: 2.4317
    value: 0.6237
    aux: 1.8986
  - step: 2100
    loss: 3.6252
    policy: 2.4324
    value: 0.6266
    aux: 1.8872
  eval:
    games: 400
    score: 0.858
    wdl:
    - 327
    - 32
    - 41
    elo_delta: 312.5
    elo_sigma: 24.9
  adopted: true
  checkpoint:
    file: gen8.pt
    sha256: 1e0e24cd8b721c47670ff242a03df06b825da74b15a4eda04eb3c08ebc2b211e
- generation: 9
  data:
    previous_buffer: gen8 buffer, 75 parts, 81044 games
    trim: uniform random 1/3 (seed 8)
    dropped_games: 27226
    dropped_pct: 33.6
    kept_old_games: 53818
    new_round: g8-1e0e24cd (38223 games, 35 parts)
    buffer_games: 92041
    buffer_parts: 85
    manifest: selfplay-scratch/manifests/gen9.json
    sf_floor:
      '1350': 0.615
      '1600': 0.3
      '1850': 0.082
      '2100': 0.075
      '2350': 0.016
      1350_wdl:
      - 37
      - 1
      - 23
      1600_wdl:
      - 15
      - 6
      - 39
      mean_plies: 68.9
      absolute: "~1440 on Stockfish UCI_Elo scale \u2014 crossed 1350 (0.615); the\
        \ 1600 cell implies 1453, the 1350 cell implies 1431"
  train:
    base: selfplay-scratch/gen8.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    rows: 10127683
    steps: 2473
    plies_per_game: 110.0
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 3.5858
    policy: 2.3592
    value: 0.6544
    aux: 1.9071
  - step: 1200
    loss: 3.5512
    policy: 2.343
    value: 0.6468
    aux: 1.8714
  - step: 2400
    loss: 3.535
    policy: 2.3371
    value: 0.6438
    aux: 1.847
  eval:
    games: 400
    score: 0.804
    wdl:
    - 292
    - 59
    - 49
    elo_delta: 245.2
    elo_sigma: 21.9
  adopted: true
  checkpoint:
    file: gen9.pt
    sha256: c20bed8a9c0d31d61b81298121dbef1922fbd5396dea1bf9063f47929fb1d9f0
- generation: 10
  data:
    previous_buffer: gen9 buffer, 85 parts, 92041 games
    trim: uniform random 1/3 (seed 9)
    dropped_games: 30450
    dropped_pct: 33.1
    kept_old_games: 61591
    new_round: g9-c20bed8a (39774 games, 37 parts)
    buffer_games: 101365
    buffer_parts: 94
    manifest: selfplay-scratch/manifests/gen10.json
    sf_floor:
      '1350': 0.842
      '1600': 0.434
      '1850': 0.125
      '2100': 0.083
      '2350': 0.008
      1350_wdl:
      - 50
      - 1
      - 9
      1600_wdl:
      - 21
      - 11
      - 29
      1850_wdl:
      - 3
      - 9
      - 48
      mean_plies: 76.3
      absolute: "~1600 on Stockfish UCI_Elo scale \u2014 1350 implies 1641, 1600 implies\
        \ 1554, 1850 implies 1568; the three cells agree within ~80"
  train:
    base: selfplay-scratch/gen9.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    rows: 11077006
    steps: 2705
    plies_per_game: 109.3
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 3.4866
    policy: 2.2556
    value: 0.6713
    aux: 1.866
  - step: 1300
    loss: 3.4531
    policy: 2.2397
    value: 0.6649
    aux: 1.8279
  - step: 2700
    loss: 3.4375
    policy: 2.2366
    value: 0.6592
    aux: 1.8057
  eval:
    games: 400
    score: 0.771
    wdl:
    - 286
    - 45
    - 69
    elo_delta: 210.9
    elo_sigma: 20.7
  adopted: true
  checkpoint:
    file: gen10.pt
    sha256: b3ca7eadd265fbbbe5b6fdeffdb64884baa11e5df5c33ad0461b1e67346354a8
- generation: 11
  data:
    previous_buffer: gen10 buffer, 94 parts, 101365 games
    trim: uniform random 1/3 (seed 10)
    dropped_games: 33306
    dropped_pct: 32.9
    kept_old_games: 68059
    new_round: g10-b3ca7ead (45107 games, 41 parts)
    buffer_games: 113166
    buffer_parts: 104
    manifest: selfplay-scratch/manifests/gen11.json
    sf_floor:
      '1350': 0.875
      '1600': 0.45
      '1850': 0.164
      '2100': 0.1
      '2350': 0.058
      1350_wdl:
      - 52
      - 1
      - 7
      note: Elo cells imply 1565-1866 depending on rung; ~1650 +- 100
    sf_nodes:
      '1': 1.0
      '50': 0.84
      '100': 0.832
      '200': 0.835
      '300': 0.725
      '500': 0.554
      '1000': 0.19
      '2000': 0.08
      games_per_rung: 100
      crossover: "~550 nodes at 300 sims \u2014 the hardware-independent strength\
        \ statement"
  train:
    base: selfplay-scratch/gen10.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    steps: 3042
    rows_approx: 12460032
    plies_per_game: 110.1
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 3.3928
    policy: 2.1679
    value: 0.6824
    aux: 1.8086
  - step: 200
    loss: 3.3882
    policy: 2.1674
    value: 0.6791
    aux: 1.8055
  eval:
    games: 400
    score: 0.689
    wdl:
    - 244
    - 63
    - 93
    elo_delta: 138.2
    elo_sigma: 18.8
  adopted: true
  checkpoint:
    file: gen11.pt
    sha256: 9cbe5ce34d4fc0d8e775250ce12bfa28518155d81064f6b61caf1a379c535c7d
- generation: 12
  data:
    previous_buffer: gen11 buffer, 104 parts, 113166 games
    trim: uniform random 1/3 (seed 11)
    dropped_games: 38753
    dropped_pct: 34.2
    kept_old_games: 74413
    new_round: g11-9cbe5ce3 (32695 games, 30 parts)
    buffer_games: 107108
    buffer_parts: 99
    buffer_note: 'first shrinkage: round smaller than the 1/3 trim, converging to
      3x round size'
    manifest: selfplay-scratch/manifests/gen12.json
    sf_floor:
      '1350': 0.875
      '1600': 0.617
      '1850': 0.233
      '2100': 0.123
      '2350': 0.066
      1350_wdl:
      - 50
      - 5
      - 5
      1600_wdl:
      - 33
      - 8
      - 19
      note: "1600 rung crossed 0.5 for the first time; 1350 implies 1688 and 1600\
        \ implies 1684 \u2014 ~1685 on Stockfish scale"
    sf_nodes:
      '300': 0.772
      '500': 0.595
      '1000': 0.415
      '2000': 0.145
      games_per_rung: 100
      crossover: ~720 nodes at 300 sims (up from ~550 at gen11)
      note: "rebuild sf_gauntlet on the box before node runs \u2014 the image binary\
        \ predates SF_LIMIT and silently reads the CSV as Elo"
    cross_anchor:
      vs: "selfplay/hp-gen1-aux03-e3.pt \u2014 the best checkpoint of the 42k supervised\
        \ lineage"
      score: 0.476
      wdl:
      - 157
      - 67
      - 176
      games: 400
      elo: -16.7
      sigma: 17.4
      note: "statistical parity (1 sigma). gen10 scored 0.278 (-166) against the same\
        \ fixed net, so two generations bought ~150 Elo here. Caveat: the architectures\
        \ differ \u2014 gen12 uses spiral rope2d + RopePolicyHead at 6.68M params\
        \ vs the old 8.73M shaw+pooled \u2014 so parity is with 23% fewer parameters."
    cost_comparison:
      our_lineage:
        human_games: 0
        selfplay_games: 388388
        rounds: 11
        training: 11 x 1 epoch, buffer 30K -> 107K
      old_lineage_best:
        human_games: 42000
        selfplay_games: 287000
        training: 3 epochs on gen1's 287K round
      note: from-scratch matched the supervised-seeded peak with ~18% more self-play
        games and no human data; the cost paid instead is wall clock (~2x) and operational
        moving parts (11 sequential rounds vs 1 round + 3 epochs).
  train:
    base: selfplay-scratch/gen11.pt
    arch: spiral rope2d dirs=4 pairs=2 lam0=8 + RopePolicyHead + aux(0.3), no soft
    epochs: 1
    batch: 4096
    lr: 0.0002
    all_plies: true
    fast_scheme: 'A: val 0.15 / pol 0.15 / aux 0.15'
    steps: 2879
    rows_approx: 11792384
    plies_per_game: 110.1
    precision: bf16+compile
    gpu: RTX 5090
  loss:
  - step: 100
    loss: 3.333
    policy: 2.1233
    value: 0.6776
    aux: 1.7736
  - step: 1400
    loss: 3.3034
    policy: 2.1128
    value: 0.6685
    aux: 1.7403
  - step: 2800
    loss: 3.2932
    policy: 2.1118
    value: 0.6655
    aux: 1.7197
  eval:
    games: 400
    score: 0.647
    wdl:
    - 217
    - 84
    - 99
    elo_delta: 105.3
    elo_sigma: 18.2
  adopted: true
  checkpoint:
    file: gen12.pt
    sha256: d2b6f32ed8a475a75e28a936c3ba567be72a9a52b2289f7fc2b5623184d665bb
```




