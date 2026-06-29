# Orchestrator status — main

## /execute over the queued backlog — COMPLETE (2/2 done, queue drained)

1. **fluid-vs-snow** (material-variants) — done & committed (5f54c55), status `active` (awaiting Done).
2. **learned-residual** (learned-dynamics) — done & committed, status `active` (awaiting Done).
   Small grid-velocity MLP residual trained through a 320-step rollout against a drag mismatch;
   weight gradient FD-verified; held-out v0 transfers 66.7%. Training page core/05.

Queue is empty. Proposed (not queued): checkpointing-long-horizon, jacobian-norms
(long-rollout-pathologies). Awaiting user's Done calls and any new queueing.
