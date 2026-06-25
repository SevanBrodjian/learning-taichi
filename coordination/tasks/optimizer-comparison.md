# Task brief — Optimizer comparison (SGD vs Adam vs L-BFGS through the rollout)

> You are a WORKER agent for task `optimizer-comparison` in direction `differentiable-control`. You are
> NOT the orchestrator. Do not spawn agents, do not start the dashboard. Read `CLAUDE.md` (Roles,
> Persistence), `spec/`, `coordination/shared_memory/working-with-sevan.md`, and `runs/README.md` first.
> Write all output to the MAIN checkout at `C:/Users/Owner/Projects/learning-taichi/` (absolute paths).
> Use the venv python at `C:/Users/Owner/Projects/learning-taichi/.venv/Scripts/python.exe`. When done,
> write everything to disk, merge to main, and STOP with a short summary. The orchestrator reviews.

## Objective
Answer, with evidence: **how much does the optimizer matter when the gradient is backpropagated through a
500+ step physics rollout?** Compare **SGD**, **Adam**, and **L-BFGS** on the existing throw-to-target
control problem (find a single initial velocity v0 so the blob's center of mass reaches the target), and
turn it into one clean, intuition-building result about the loss landscape of a differentiable rollout.

## What to build on
`sim/diffmpm.py` already has the full differentiable rollout, the loss, the `ti.ad.Tape` gradient, and an
Adam loop driven from numpy. Reuse that machinery. Prefer a **sibling script** (e.g. `sim/optimizer_compare.py`)
that imports or mirrors the existing fields/kernels rather than editing `diffmpm.py` destructively. The
baseline task must keep working.

The three optimizers:
- **SGD** — `v0 -= lr * grad`, optionally with momentum. Trivial.
- **Adam** — already implemented in `diffmpm.py`; reuse it.
- **L-BFGS** — use `scipy.optimize.minimize(method="L-BFGS-B")` with a closure that runs forward+backward
  once and returns `(loss, grad)` from the Tape (same numpy boundary as the Adam loop). You will need
  scipy: `python -m pip install scipy` and add `scipy` to `requirements.txt`.

## Critical gotcha — avoid the known NaN so the comparison is fair
Backprop through the long rollout overflows f32 once v0 grows, via near-zero grid-mass nodes (this is the
solved finding in `reports/training/core/03-failure-modes.md`). If you do not handle it, optimizers that
push v0 hard will NaN and the comparison will be about who NaNs first, not who optimizes best. So apply
**mass stabilization** in the grid step: divide grid momentum by `ti.max(m, eps)` with `eps = 1e-4`
instead of the bare `1/m`. This is a one-line change and makes all three optimizers run cleanly. Keep the
task, target, horizon, particle seed, and stabilization identical across the three optimizers so the only
variable is the optimizer.

## Fair-comparison rules
- Same task for all three: target `(0.7, 0.35)`, 512 steps, same initialization.
- Give each optimizer a reasonable learning rate / settings (do not cripple one to make another look good;
  a short lr sweep to pick a sane value per optimizer is fine and worth reporting).
- L-BFGS does multiple gradient evaluations per "iteration", so report **gradient evaluations** (or
  wall-clock), not just iteration count, alongside iterations. Be honest about what the x-axis means.

## Deliverables (the dashboard contract — schema v2, see runs/README.md)
Produce ONE polished task at `runs/differentiable-control/optimizer-comparison/`:
- `manifest.json` (schema_version "2", direction `differentiable-control`, task_id `optimizer-comparison`,
  status `active`, a clear named `title`, an `objective` and `findings` paragraph).
- **results** (every one must render):
  - A **comparison plot** of all three loss curves on one chart. The dashboard's `plot` type only renders
    a single series, so export a **matplotlib PNG** (loss vs gradient-evals, log-y, three labeled curves)
    and reference it as a `{"type":"image","src":...}` result.
  - A **table**: optimizer | final loss | iterations | grad-evals | stable? | notes.
  - Optionally a short `video` of the best run reaching the target (reuse the headless renderer in
    `diffmpm.py`).
- `training_refs`: at least `["differentiating-the-rollout"]`. A short textbook addition is optional and
  only if you find something genuinely teachable about the landscape; do not pad.

KaTeX rule for any math in prose: multiline `$$` must be the three-line form; never `\*`; brace multi-char
sub/superscripts.

## Definition of done
A clear, evidence-backed answer to "does the optimizer matter here, which wins, and what does that say
about the loss landscape of a long differentiable rollout," shown as one polished task on the dashboard
(objective + findings + the comparison plot + the table), with code committed and merged to main.
