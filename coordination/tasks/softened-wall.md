# Task brief — Softened wall contact for smooth boundary gradients

> WORKER agent for `softened-wall` in `long-rollout-pathologies`. You are NOT the orchestrator. No
> spawning, no dashboard, and **do NOT git commit** (leave work on disk for review). Read `CLAUDE.md`
> (especially **Evidence discipline**), `spec/`, `coordination/shared_memory/working-with-sevan.md`,
> `runs/README.md`, and `reports/training/core/03-failure-modes.md`.
>
> Write ONLY to `runs/long-rollout-pathologies/softened-wall/` and a uniquely-named sibling script
> `sim/softened_wall.py`. Do NOT edit any shared file. Deps (taichi, numpy, matplotlib, imageio) are
> already in the venv; only install if genuinely missing and report it. venv python:
> `C:/Users/Owner/Projects/learning-taichi/.venv/Scripts/python.exe`. Absolute paths, on MAIN. Other
> workers may share the GPU; note it if results look contaminated.

## Objective
The hard wall condition (zeroing the inward-normal velocity at the boundary) is a non-smooth kink in the
gradient. Replace it with a **smooth ramp** (e.g. scale the inward-normal velocity by a smooth factor that
goes 1 -> 0 across a boundary band, or a soft penalty) and measure whether **gradient quality improves**
when particles actually touch the wall. Keep mass stabilization on (this is about contact, not the
already-fixed mass overflow).

## What to measure (hard vs soft wall, same task/seed/horizon)
- Use a task that genuinely drives the blob INTO a wall (e.g. a target near a boundary) so contact happens.
- Compare hard vs soft: convergence (loss vs iters), final loss, gradient norm / stability, and whether
  the optimizer path is smoother with the soft wall. Quantify, do not assert.
- Try at least two ramp widths / softness settings to show the trade-off (too soft distorts the physics).

## Honesty (mandatory)
This is one contact scenario; scope claims to it. The prior finding was that contact is NOT the cause of
the long-rollout NaN (mass overflow was) — so frame this as "does smoothing contact improve gradient
quality", not "contact was the problem". `hypothesis` + `limitations` required.

## Deliverables (schema v2, `runs/long-rollout-pathologies/softened-wall/`)
`manifest.json` (status `active`): objective, scoped findings, hypothesis, limitations; results: loss-curve
comparison plot(s) hard vs soft (PNG `image` or `plot`), a `table` (variant | final loss | grad-norm |
notes), and one or two short `video`s (hard vs soft run). `training_refs`: ["failure-modes"]. Leave on
disk; do not commit.
