# diagnostics/ — temporary probes (not pipeline code)

One-off investigation scripts from the Oct 11 PCR-round debugging,
salvaged from probe_labels/ before cleanup. Each answers one question
about net/data health; paths inside are hardcoded to that era's
checkpoints and will need adjusting before reuse.

- probe_labels.py      do the trainer-consumed labels (dense pi, WDL)
                       match what the C engine computed?
- value_calibration.py is the fine-tune damaging the value head or the
                       policy head (label-distribution drift)?
- head_swap.py         build head-swap ablation checkpoints (policy /
                       trunk / value only)
- trunk_only.py        trunk-only variant of the same ablation
- kp_vs_k_bed.py       K+P vs K exact-ground-truth feasibility bed
