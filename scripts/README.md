# scripts/ — salad.com cloud ops

Salad instances are cheap but **can vanish at any moment**; everything here treats that as normal operation, not an exception:

- Worker bootstrap (Phase 0.4): bare machine → clone → install → pull checkpoint → run one toy work unit → push result. Runs on the Mac first.
- Work-queue protocol (Phase 4): small idempotent units — "from checkpoint N, play K games, upload shard" — re-issued by the control plane when a worker dies.
- Checkpoint/push protocol: the trainer checkpoints every few minutes and pushes off-instance immediately; instances only cache, never own, state.
- Churn drill + cost meter (Phase 4): kill an instance mid-job and verify ≤ one work unit lost; track hours + $ per run with an alert threshold.

The Mac is the control plane (submitting units, watching heartbeats, replacing workers) and the fallback if salad is entirely down — everything must be runnable locally.
