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
