# Research directions — backlog & status

The **menu of what to work on** (distinct from `spec/`, which is *how* to write/what's wanted). A
living doc: the orchestrator and workers add/curate directions; each gets a one-line status. When the
fleet fans out (Phase 3), one direction ≈ one branch/worktree.

Status legend: `idea` · `queued` · `in-progress (<branch>)` · `done` · `parked`

## Active
- **DiffMPM baseline** — make `sim/mpm88.py` differentiable; optimize initial velocity / actuation to
  reach a target; loss curve + video + training report. — `in-progress (claude/elegant-bassi-cb7174)`

## Queued Priority Directions (set by user)
- **Contact / boundary differentiability** — the non-differentiable wall clamps (mpm88 lines 49–59, 77);
  smoothing and its effect on gradients. — `idea`
- **Material variants** — elastic vs fluid vs snow (constitutive model swaps) and how each optimizes. — `idea`

## Proposed / candidate directions
- **Optimizer comparison** — SGD vs Adam vs L-BFGS backpropagating through the rollout. — `idea`
- **Long-rollout gradient pathologies** — exploding/vanishing grads vs #steps; checkpointing trade-offs. — `idea`
- **Resolution / performance scaling** — grid & particle count vs GPU throughput and gradient memory. — `idea`
- **Toward learned dynamics** — replace part of the sim with a learned component; first step toward
  structured world models. — `idea`

> Co-author: reprioritize, add directions, and note which to fan out first once Phase 1 proves the loop. User will pull candidate directions into queued priority directions based on interest, or write new directions.
