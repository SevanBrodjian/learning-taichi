# Task brief — Optimizer comparison ACROSS several tasks (generality re-run)

> WORKER agent for `optimizer-comparison` in `differentiable-control`. You are NOT the orchestrator. Do
> not spawn agents, do not start the dashboard, and **do NOT git commit** (leave work on disk for the
> orchestrator to review). Read `CLAUDE.md` (especially **Evidence discipline**), `spec/`,
> `coordination/shared_memory/working-with-sevan.md`, and `runs/README.md` first.
>
> Write ONLY to `runs/differentiable-control/optimizer-comparison/` and a uniquely-named sibling script
> `sim/optimizer_compare_multi.py`. Do NOT edit any shared file (the training textbook, `requirements.txt`,
> other tasks). Deps you need (taichi, numpy, matplotlib, scipy, imageio) are already in the venv — only
> pip-install if something is genuinely missing, and report it rather than editing `requirements.txt`.
> venv python: `C:/Users/Owner/Projects/learning-taichi/.venv/Scripts/python.exe`. Absolute paths, on MAIN.
> Other workers may be sharing the GPU; if anything looks contaminated, note it.

## Why this is a re-run (read this)
The first version compared SGD/Adam/L-BFGS on ONE near-toy task (single blob -> single target) and
**overclaimed a general conclusion from one example**. The user's reopen note: *"examine the optimizer
comparison across a few varied tasks to make more general claims."* Generality is the entire point now.

## Objective
Run the same three optimizers (SGD+momentum, Adam, scipy L-BFGS-B, identical `(loss, grad)` tape boundary,
mass-stabilized grid so none NaN) across **several distinct control tasks**, and report how robustly the
earlier finding (L-BFGS dominates; all reach the same basin) holds. Reuse `sim/optimizer_compare.py`'s
machinery where useful.

## Tasks to compare across (at least 3-4, genuinely varied)
1. throw-to-target (original): single shared 2-D `v0`, target (0.7, 0.35).
2. A target that drives the blob INTO a wall (e.g. (0.08, 0.35)) — contact-influenced.
3. A genuinely higher-dimensional control: a per-region (or per-particle) initial velocity field instead
   of one shared `v0`, so the loss surface is higher-dim and not trivially smooth.
4. (optional) a different horizon (256 vs 512) or a two-point / via objective.
Within each task, hold task/seed/horizon/stabilization identical across the three optimizers.

## Honesty rules (mandatory)
- Score in **gradient evaluations**, not nominal iterations (L-BFGS line search).
- A claim is only as general as the tasks you actually ran. If L-BFGS wins on all, say "on these N tasks";
  if it varies, report exactly where and why it breaks. Keep the f32 loss-floor caveat where relevant.

## Deliverables (schema v2)
- `manifest.json` (status `active`): objective, **scoped findings** (per task + the honest cross-task
  pattern), a **hypothesis** (why the pattern holds, where it might fail, what would test further), and
  **limitations**.
- results: one comparison plot PER task (PNG `image` results), one cross-task summary `table` (task |
  winner | L-BFGS best | Adam best | SGD best | same v0*?), and a short `video` PER task of its best run
  (multiple `video` results are supported and encouraged).
- `training_refs`: ["differentiating-the-rollout"].

## Definition of done
An evidence-backed, honestly-scoped statement of how far the "L-BFGS dominates a smooth rollout landscape"
finding generalizes across several distinct control tasks. Leave on disk; do not commit.
