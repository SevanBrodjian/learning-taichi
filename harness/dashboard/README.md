# dashboard

Standalone **Vite + React** dashboard for learning-taichi runs, reports, and demos. Reuses the same
component libraries as the personal site (`react-markdown`, `katex`) so components port into
`sevanbnet-frontend` cleanly. The data layer (`src/config.js → DATA_BASE`) reads static JSON/media
locally and can later be repointed at the Django API.

## Run it
```bash
npm install
python ../tools/sync_dashboard_data.py   # copy ../runs + ../reports into public/data
npm run dev                               # http://localhost:5174
```

## Data contract
Reads `DATA_BASE/index.json` (all runs), then each run's `manifest.json`, its `metrics.json`, the
training-report markdown, and any media. Paths inside manifests are **repo-root-relative** (see
`../runs/README.md`). A committed sample fixture under `public/data/` lets the UI render before the
first real run exists.

> Installable-PWA + live-monitoring wiring is deferred to the dedicated setup pass (manifest icons,
> service worker, ntfy, auto-refresh).

## Public static export (read-only)
`npm run build:public` builds the same dashboard in a read-only mode that browses a frozen snapshot
(`snapshot/api/...`, relative base, no polling, no write controls, no Notebook/Inbox) instead of the live
server; `npm run build:demo` builds the WebGPU demo as one self-contained `index.html`. Don't run these by
hand to publish: `harness/tools/export_public.py` writes the snapshot, runs both builds, and refuses to
finish if the privacy scan finds anything (see its docstring for what is excluded and the opt-in flags).
