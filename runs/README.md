# runs/ — task artifacts and the dashboard data contract

A **task** is one worker's polished deliverable. Each task writes a self-describing folder; the
dashboard reads these, so the schema is a **contract**. Tasks belong to a **direction** (see
`coordination/directions/`), and the layout encodes that relationship so the Overview↔Task link is
structural, not maintained by hand.

```
runs/
  <direction-id>/<task-id>/
    manifest.json     # REQUIRED — schema_version "2", described below
    metrics.json      # optional loss curve / time series (referenced by a "plot" result)
    video.mp4         # optional rendered playback (referenced by a "video" result)
    *.png             # optional images (referenced by an "image" result)
```

## manifest.json (schema_version "2")
A task MUST surface an objective, findings, and at least one result. It SHOULD also carry a `hypothesis`
(why the result holds, and what would test its generality) and `limitations` (what was not tested).
**Scope findings to what was actually tested — no overclaiming from one example** (see `CLAUDE.md` ->
Evidence discipline). Absent results are simply omitted from the display (no placeholders).

```json
{
  "schema_version": "2",
  "task_id": "throw-to-target",
  "direction": "diffmpm-baseline",
  "title": "Throw a blob to a target by backprop",
  "tldr": "ONE sentence, no jargon, including what failed. Shown first; used to triage many tasks fast.",
  "status": "done",
  "created": "2026-06-23T04:12:24Z",
  "objective": "One paragraph: what this task set out to do.",
  "findings": "One paragraph: what was found, scoped to what was actually tested (no overclaiming).",
  "hypothesis": "Why the result holds (the mechanism), and what would test its generality on other tasks.",
  "limitations": "What was NOT tested; the scope the claims are bound to (e.g. one task, one target).",
  "results": [
    { "type": "plot",  "kind": "loss", "series": "runs/<dir>/<task>/metrics.json", "log": true, "caption": "..." },
    { "type": "video", "src": "runs/<dir>/<task>/video.mp4", "caption": "..." },
    { "type": "image", "src": "runs/<dir>/<task>/figure.png", "caption": "..." },
    { "type": "table", "columns": ["param", "value"], "rows": [["lr", "0.1"]], "caption": "..." }
  ],
  "custom_html": null,
  "training_refs": ["mls-mpm-forward"]
}
```
- **Result types**: `video` (auto-plays), `image`, `plot` (a metrics series), `table`.
- **`tldr` is required.** One sentence stating what happened, including the part that failed. It renders
  above Objective so many tasks can be scanned without opening them.
- **`custom_html` is the task's OWN page** and now *leads* the task view, with the results grid, full
  report, hypothesis and limitations collapsed into an "Evidence & detail" expander beneath it. It is a
  sandboxed iframe (scripts only, no same-origin, no network) that sizes itself to its content. Design it
  per `spec/style_task_page.md` — it is not an optional extra panel any more.
- **Metrics come from `spec/registry/`.** Use registered names; register anything new in the same run,
  with a real source file:line. The dashboard renders definitions on click/hover.
- Paths (`src`, `series`) are repo-root-relative; the server rewrites them to absolute `/api/data/...` URLs.
- `training_refs` are section ids from `reports/training/index.json`, transcluded (collapsed) under the task.
- `hypothesis` and `limitations` render as their own sections; use them to keep observation, explanation,
  and scope cleanly separate, and to seed follow-up tasks.
- `status` is `active` while a worker runs it; the user marks it `done` after review (never automatic).
- `notes` are the **user's** passive margin notes. They live in `coordination/directions/<dir>.json` (not
  here), never change status, and survive re-runs. Workers do not write them.

## How the dashboard joins it
The data server (`harness/server`) reads `coordination/directions/<id>.json` (the board source: each
direction's name, status, and task list) and matches each task to its artifact at
`runs/<direction>/<task-id>/manifest.json`. A task shows a detail iff that manifest exists; the server
also reports orphans (artifacts with no board entry). There is no generated `index.json` anymore.
