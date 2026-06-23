# harness/server — master data server

One process, run from the **main** checkout, that gives the dashboard a single live view of
runs across the main repo **and every git worktree**. No merging required to see an agent's
results, and no copying files into the dashboard. This is the bridge to a future Django API
(the JSON shapes are identical).

## Run
```
python harness/server/app.py        # http://localhost:8732
```
Set `DASHBOARD_PORT` to change the port. The dashboard points at this origin via
`harness/dashboard/src/config.js` (`DATA_BASE`).

## Endpoints
- `GET /api/index` — unified run list across all worktrees, newest first.
- `GET /api/data/{root}/{path}` — serve any run/report/media file from the worktree that owns
  it. `manifest.json` responses are rewritten so their `media` / `metrics.series` / `reports`
  paths are absolute `/api/data/...` URLs, so the dashboard never has to know about roots.

## Why a server (vs. the old static copy)
A worktree's working directory holds files written *in that worktree*, even uncommitted ones.
The server reads them directly via `git worktree list`, so parallel agents' runs appear live.
A freshly created worktree shows up on the next request with no restart.
