# STATUS — claude/elegant-bassi-cb7174

**Phase:** 1 — first DiffMPM run complete; dashboard viewable.

**Done:**
- Phase 0 scaffolding + durable memory.
- Dashboard (Vite + React, reuses the site's `react-markdown`/`katex`) built and serving on `:5174`
  with the real run; data layer behind `DATA_BASE` + `tools/sync_dashboard_data.py`.
- `sim/diffmpm.py`: differentiable 512-step MLS-MPM, `ti.ad.Tape`, Adam on initial velocity. First run
  converged **loss 0.144 → 6.2e-5** (`v0* ≈ (3.54, -1.45)`); **NaN gradient at iter 32** caught by the
  guard — documented failure mode #1.
- Interim writeup: `reports/training/diffmpm.md`.

**Viewable now:** http://localhost:5174  (Network http://100.69.101.128:5174 for iPad).

**Awaiting from user:**
- View the dashboard → then set up **monitoring** (live refresh, ntfy, installable PWA on iPad).
- `spec/` TODO lines (calibrate the full ground-up training report).
- An ntfy topic for `NTFY_TOPIC`.

**Next:** monitoring setup; full training report; commit Phase 0/1 baseline; investigate the NaN
(gradient-pathologies direction).

_Last updated: 2026-06-20._
