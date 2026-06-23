# runs/ — experiment artifacts and the dashboard data contract

Each run writes a self-describing folder; the dashboard reads these, so the schema is a **contract**.

```
runs/
  <branch>/<run-id>/
    manifest.json     # REQUIRED — described below
    metrics.json      # loss curve and any time series
    video.mp4         # optional rendered playback
    frames/           # optional frame sequence
    checkpoints/      # optional
  index.json          # GENERATED — aggregates every manifest; powers the branch-switcher
```

## manifest.json (schema_version 0)
```json
{
  "schema_version": "0",
  "run_id": "diffmpm-20260620-001",
  "branch": "claude/elegant-bassi-cb7174",
  "title": "DiffMPM: optimize initial velocity to reach a target",
  "created": "2026-06-20T12:00:00Z",
  "status": "completed",
  "summary": "One-line human summary of what this run did and showed.",
  "metrics": { "final_loss": 0.0123, "iterations": 200, "series": "metrics.json" },
  "media": { "video": "video.mp4", "frames_dir": "frames/" },
  "reports": { "training": "reports/training/diffmpm.md" },
  "params": { "lr": 0.1, "n_particles": 8192, "n_grid": 128, "steps": 1024 }
}
```
Paths inside `manifest.json` are repo-root-relative so the dashboard can resolve them uniformly.

## index.json (schema_version 0)
```json
{
  "schema_version": "0",
  "runs": [
    { "run_id": "diffmpm-20260620-001", "branch": "claude/elegant-bassi-cb7174",
      "title": "DiffMPM: optimize initial velocity to reach a target",
      "status": "completed", "created": "2026-06-20T12:00:00Z",
      "manifest": "runs/claude/elegant-bassi-cb7174/diffmpm-20260620-001/manifest.json" }
  ]
}
```
A small indexer (built in Phase 1) regenerates `index.json` by scanning `runs/*/*/manifest.json`.
