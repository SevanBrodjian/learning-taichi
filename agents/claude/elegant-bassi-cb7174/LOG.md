# LOG — claude/elegant-bassi-cb7174 (append-only)

- **2026-06-20** — Phase 0 scaffolding laid down on branch: conventions (CLAUDE.md), `spec/`
  constitution drafts, `sim/mpm88.py` seed, `tools/notify.py`, runs/reports/coordination READMEs.
  Verified Taichi 1.7.4 on CUDA arch; Node v20.12.1 / npm 10.5.1 present.
- **2026-06-20** — Wrote DiffMPM technical design (`diffmpm_design.md`). Dashboard deferred pending the
  user's React site folder (to match its exact stack). Next: implement `sim/diffmpm.py`.
- **2026-06-20** — Inspected the site (`SevanBnet/sevanbnet-frontend`): CRA + React 18 + JS, already
  uses `react-markdown`/`katex`/`p5`/`react-router`. Built the dashboard as a standalone **Vite +
  React** app reusing those libs (Vite over the deprecated CRA, per "avoid idiosyncrasies"). Build +
  all `/data` endpoints verified on `:5174`.
- **2026-06-20** — Implemented `sim/diffmpm.py` (time-indexed differentiable MLS-MPM, `ti.ad.Tape`,
  Adam on `v0`). First run: **loss 0.144 → 6.2e-5** by iter 29, `v0* ≈ (3.54, -1.45)`; **NaN gradient
  at iter 32** (finite forward loss, NaN backward) caught by guard — failure mode #1. Exported
  metrics + 170 KB video + manifest; `tools/index_runs.py` + `tools/sync_dashboard_data.py` pushed
  them into the dashboard. Wrote interim `reports/training/diffmpm.md`.
