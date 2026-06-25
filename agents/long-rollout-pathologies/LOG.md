# LOG — long-rollout-pathologies

## 2026-06-24 — softened-wall (worker)
- Read brief, CLAUDE.md (Evidence discipline), runs/README.md, working-with-sevan.md,
  03-failure-modes.md, sim/diffmpm.py, sim/diffmpm_pathologies.py.
- Wrote `sim/softened_wall.py`: hard `grid_op_hard` (mass-stabilised baseline) vs `grid_op_soft`
  (smoothstep gate on inward-normal velocity across a `ramp_cells`-wide band, applied to all 4 walls,
  one-sided so outward motion is never damped). MASS_EPS=1e-4 always on.
- Smoke test (4 iters / 80 steps): pipeline ran but 0% wall contact — blob never reached the wall.
- Tuned the task to force genuine contact: target [0.06, 0.5], lr 0.15, 100 iters, 400 steps →
  hard 7% / soft r=3 17% particles in the band. Contact is real.
- First FD check was measured at hard's own converged v0 → biased (each wall model converges
  elsewhere). Replaced with a SHARED 4-point v0 sweep (vx in {-4,-5,-6,-7}) so only the wall model
  differs. This flipped the apparent result: soft now clearly beats hard on FD agreement.
- Observed run-to-run variance in the FD-sweep metric (GPU atomic-add non-determinism). Added a
  3-repeat sweep, stored all repeats in metrics.json, and scoped the claim to the repeat-mean:
  hard 3.42e-2 vs soft ~2.2e-2 robustly; the two soft widths are within noise of each other.
- Final loss is the stable discriminator: soft r=3 best (1.90e-3), soft r=6 worst (4.72e-3, wide
  band distorts physics). grad-direction cosine indistinguishable across conditions.
- Wrote manifest (schema v2, status active) with objective/findings/hypothesis/limitations, 2 plots,
  a 7-col table, and 3 videos. training_refs=["failure-modes"]. Left everything on disk uncommitted.
