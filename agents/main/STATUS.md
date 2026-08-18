# STATUS — main

**Last worker:** `the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page` (material-variants), finished.

The Demo tab is a live four-material MLS-MPM simulation on WebGPU: water, rubber, snow and sand on one
shared grid at 1.00x real time (9,249 particles, 20,000 substeps per simulated second, 133 fps on the
RTX 4090), with pour / grab / remove, three render modes, a speed slider, reset and empty.

On disk, not committed:
- `runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/` — manifest,
  metrics, three videos, three stills, the bespoke page, the portable `web/` bundle and the whole
  `verify/` chain.
- `harness/dashboard/src/components/DemoView.jsx` — rewritten; imports only `./mpm/`, which is
  GENERATED from the run's `web/` by `web/sync_to_dashboard.py`.
- `harness/dashboard/src/components/mpm/` — new, generated.
- `reports/training/core/16-svd-in-practice.md` + `reports/training/index.json`.
- `spec/registry/metrics.json` — two new entries (`node_mass_headroom`, `realtime_factor`).

**Note for whoever restarts the dashboard:** Vite's transform cache served a stale `DemoView.jsx` for a
while after the edit; touching the file invalidated it. If the Demo tab shows the old placeholder, touch
`harness/dashboard/src/components/DemoView.jsx` or restart the dev server on 5174.
