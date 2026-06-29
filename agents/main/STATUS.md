# Orchestrator status — main

## Current: /execute over the queued backlog (2 tasks, GPU-heavy → serialized)

Schedule: both queued tasks are GPU-heavy autodiff rollouts on the one shared GPU, so they run
**serially** (hard-won lesson: concurrent GPU workers corrupt each other).

1. **fluid-vs-snow** (material-variants) — brief `coordination/tasks/fluid-vs-snow.md`.
   Status: **active**, worker spawned and running. Elastic vs fluid vs snow under the throw task.
2. **learned-residual** (learned-dynamics) — brief `coordination/tasks/learned-residual.md`.
   Status: **queued, held** until #1 finishes (GPU serialization). Not yet flipped to active.

Next on each finish: review against Evidence discipline, render-check math + manifest, review/extend
the training page, commit (leave status `active` — Done is the user's call), then spawn the next.
