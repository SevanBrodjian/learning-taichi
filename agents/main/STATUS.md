# STATUS — main

**Last worker:** `propose-new-rendering-for-each-of-the-four-materials` (material-variants) — finished,
results on disk, **not committed** (the orchestrator reviews and commits).

## What is on disk
- `runs/material-variants/propose-new-rendering-for-each-of-the-four-materials/` — manifest.json
  (schema v2, custom_html embedded), bespoke_page.html, render_cost.json, metrics.json, 49 mp4 clips,
  22 stills, build_manifest.py.
- `sim/material_render.py` (renderer + treatments + the baseline port), `sim/material_render_run.py`
  (the deliverable driver), `sim/material_render_cost.py` (device-time measurement).
- `reports/training/core/17-material-appearance.md`, `reports/training/prerequisites/06-filters-and-samples.md`,
  a forward link added to `core/12-fluid-rendering.md`, `reports/training/index.json` updated.
- `spec/registry/metrics.json` — `render_gpu_ms`, `render_wall_ms` registered.

## Untouched, deliberately
`harness/dashboard/src/components/DemoView.jsx`, `harness/dashboard/src/components/mpm/**`,
`sim/physics/**`. Verified with `git status`.

## Needs a human
1. **Pick an appearance** per material (each has options A and B on the task page). That is the point
   of the task and it is a taste call.
2. **The task-page media defect** described in `agents/main/LOG.md` is a harness bug affecting every
   task page, not this run's to fix.
