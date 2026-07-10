# Orchestrator status — main

_Snapshot as of 2026-07-09. All work committed to main (not pushed this session). Working tree clean._

## Latest session — harness/dashboard upgrades (7-item backlog)
Shipped: **per-task effort tiers** quick/standard/deep (dashboard picker + `/api/task-effort` + `effort` in
overview/detail; consumed at spawn time by `/execute` and the task template); **live running status**
(worker writes `runs/<dir>/<task>/status.json` via `harness/tools/task_status.py`, gitignored; server
`live_statuses()` surfaces `{state,step,age}`; dashboard shows a pulsing dot + step on Active cards and the
task head); **mobile task-head fix** (title above a wrapping action row); **independent scrolling** for the
Training/Reports two-pane views (`.content-split`); **"Open in Training ↗"** deep-link from a task page's
embedded textbook section; **brevity spec** (new `spec/style_training_report.md` "Brevity and prioritization"
section, template + CLAUDE.md updated, periodic training-sweep added as an orchestrator duty). One-time
**training sweep**: merged learned-viscosity-interpolation into learned-material-interpolation (349→126 lines;
index/links/manifests cascaded) and trimmed perorations/task-numbers/code across the recent core series.
Skipped by user request: the execute-trigger mechanism (#1) — deferred pending their testing.
Deeper 30–40% cuts on the mid/foundational pages are an open follow-up (kept surgical this pass).


## Queue
**Empty.** Nothing queued. Proposed (not queued): `residual-hard-mismatch` (learned-dynamics),
`checkpointing-long-horizon` + `jacobian-norms` (long-rollout-pathologies), `shape-match-materials`
(material-variants).

## Directions (completed tasks; all `active`/`done` awaiting or given the user's Done)
- **differentiable-control** — throw-to-target; optimizer-comparison (reworked, multi-task).
- **learned-dynamics** — learned-residual (FD-verified residual through a 320-step rollout).
- **long-rollout-pathologies** — nan-root-cause; softened-wall; resolution-memory.
- **material-variants** — nondifferentiable showcase (fluid/elastic/snow); differentiable materials
  (FD-verified gradients, CFL was the real prior failure); viscosity sweep (oil→honey); learned-viscosity
  weight interpolation (sags below linear ideal); learned-material weight interpolation (degenerate interior,
  reversed a first overclaimed run); **one-nn conditioned on a 2-param descriptor** (beats weight-blend but a
  real edge-fidelity tradeoff; 5×5 grid + interactive `custom_html`).
- **realistic-rendering** — non-differentiable renderer; improve-realism (water dynamics + interior-fill
  no-holes); **gpu-accelerate-fluid-renderer** (Taichi, 130–265×, visual parity); more-realistic showcase
  (6 long 11–14 s clips + per-particle-dye color mixing on the GPU renderer).

## This session also shipped (beyond research tasks)
- **Dashboard features**: delete/reject-to-queue/propose-follow-up on tasks, phone dropdown menu, large-
  monitor scaling, nav badges (new-training / unread-notifications), shared VideoPlayer + memoized markdown
  (video pause fix), typing-lag fix, visible figure captions.
- **Server fix** (`harness/server/app.py`): `_git_commit` now scopes to its path (was committing the whole
  index → clobbered concurrent hand edits).
- **Spec hardening** (CLAUDE.md, `spec/style_training_report.md`, task template, `/execute`): workers view
  their own figures; prereq coverage + `[[link]]` resolution; parameter-effect examples; **write the manifest
  last referencing only existing media; finish the task within your turn**.
- **New skills**: `/new-direction`, `/new-task`.
- **Textbook** grew to 16 core pages + linear-algebra & SVD/polar prereqs.

## Live services
Data server (`.venv\Scripts\python.exe harness\server\app.py`) on **8732**; Vite dashboard on **5174**.
See `coordination/shared_memory/orchestration-lessons.md` for how to restart and other hard-won ops facts.
