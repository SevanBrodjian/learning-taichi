# Worker brief: Generalize One NN across Viscosity and Surface Tension

## Effort tier: deep
This is a genuinely hard task. **Persist.** Iterate, debug, and run the calibration/training sweeps it
needs rather than stopping at the first plausible result. The bar is high: the three trained corners must
reproduce ground truth, the 5×5 interior must be smooth and physical, and the held-out corner must be
reported honestly. Do not ship a degenerate grid. Keep it in the main checkout.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `generalize-one-nn-across-viscosity-and-surface-tension`. You are **NOT
the orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to disk
under `runs/material-variants/generalize-one-nn-across-viscosity-and-surface-tension/`, extend the training
textbook, and exit. **Do not commit** — the orchestrator reviews and commits your work. Fire the pings below.

## Notifications (exactly two) + live status
Start:
```
python harness/tools/notify.py --kind started --task generalize-one-nn-across-viscosity-and-surface-tension "Starting the conditioned-net training across viscosity and surface tension."
```
When results are on disk:
```
python harness/tools/notify.py --kind finished --task generalize-one-nn-across-viscosity-and-surface-tension "<one plain sentence: what's ready to review>"
```
Use `--kind blocked` if you hit a hard stop.

**Live status (call ~5-8 times over this long run, one short phrase each):**
```
python harness/tools/task_status.py --direction material-variants --task generalize-one-nn-across-viscosity-and-surface-tension --step "<a few words: current step>"
```
Milestones: "calibrating a gentle ST range", "collecting GT training states", "training the conditioned net",
"checking edge-exactness at 3 corners", "running the 5x5 grid vs GT", "held-out corner test",
"rendering montage", "writing training page".

## Objective
Following the **conditioned-network protocol** of `one-nn-for-three-materials` (`sim/one_nn_materials.py`),
train **one shared network conditioned on a two-scalar descriptor** $m = (m_\text{visc}, m_\text{st})$ to
replicate the weakly-compressible fluid across **viscosity** and **surface tension** (from
`sim/fluid_surface_tension.py`). Train it on **three** conditions and interpolate the descriptor smoothly to
fill a **5×5 grid vs ground truth**, then test generalization to the **held-out fourth corner**.

Descriptor layout (a unit square, three trained corners, one held out):
- $m_\text{visc}$: **low → high viscosity**. $m_\text{st}$: **none → high surface tension**.
- Trained: **(low visc, low ST)** at $(0,0)$, **(high visc, low ST)** at $(1,0)$, **(low visc, high ST)** at
  $(0,1)$.
- **Held out (never trained): (high visc, high ST)** at $(1,1)$ — reported honestly as whatever it does.

## How to structure it (this is the crux — read carefully)
Mirror `one_nn_materials.py`'s split of "learned stress" vs "analytic state machinery driven by the
descriptor":
- **The network predicts the per-particle stress** (the weakly-compressible pressure + Newtonian viscous
  term) as a function of the local physical features **and the descriptor** $m$, exactly as
  `net_sigma_cond` does, feeding $m_\text{visc}, m_\text{st}$ as two extra inputs. Viscosity is a per-particle
  stress (it fits naturally; cf. the learned-viscosity work). One shared weight set; only $m$ changes.
- **Surface tension is NOT a per-particle stress — it is the grid capillary (CSF) force** in
  `fluid_surface_tension.py` (`st_accumulate` / `st_apply`: smoothed grid density → normal → curvature →
  $f=\sigma_{st}\kappa\nabla\phi$, net-zero corrected). Treat it as the **analytic "state rule" driven by the
  descriptor**: apply the real CSF force with strength $\sigma_{st}(m_\text{st})$ scheduled from the
  descriptor, the same way `one_nn` drives `iso` and the plastic band from $m$. So the learned rollout at a
  cell = (conditioned-net stress at $m$) + (analytic CSF at $\sigma_{st}(m_\text{st})$).
- **Edge-exactness must hold two ways at each trained corner** (the review bar from `one_nn`): (a) the
  conditioned net at $m=$ corner reproduces the true simulator of that condition (the replication fit), and
  (b) the grid cell AT the corner reproduces the pure-config rollout to GPU-noise level (schedule parity —
  $\sigma_{st}(m_\text{st})$ must hit the exact trained $\sigma$ at the corners). Report both before trusting
  any interior cell.

## The surface-tension range — fix the saturation (explicit reviewer feedback)
In the 3×3 predecessor the ST effect **saturated almost immediately**: a roundness sweep showed 0.69→0.91 by
$\sigma_{st}=0.1$ and essentially flat past that, so its "medium" and "high" ST columns both sat in the
saturated regime and looked the same. **Fix this** so intermediate descriptor values show intermediate
behavior:
- **Calibrate first.** Sweep $\sigma_{st}$ finely at low values (e.g. 0, 0.005, 0.01, 0.02, 0.035, 0.05, …)
  on the isolation blob and find where roundness is genuinely mid-transition (≈0.8). Pick a **low $\sigma_\max$**
  near the top of the transition (likely on the order of $\sigma_{st}\sim0.03$–$0.06$, far below the old 2–4),
  and/or map $m_\text{st}\to\sigma_{st}$ with a **smooth, gentle (e.g. nonlinear) schedule** so most of the
  $m_\text{st}$ range lives in the visible transition rather than the saturated tail.
- The success criterion: reading up the ST axis of the 5×5 grid, the droplet should round **gradually**
  across the rows, not snap to a ball by the second row.

## Experiments / deliverables
1. **Calibrate the ST schedule** (above) and pick the two viscosity levels and the $\sigma_{st}(m_\text{st})$
   map. State the trained corner values explicitly.
2. **Collect GT training states** over several varied scenes per trained condition (reuse the `train_scenes`
   pattern + mirror augmentation), and **train the shared conditioned net** (adapt `train_mlp` /
   `net_sigma_cond`). Persist until each trained corner reproduces its GT.
3. **Edge-exactness**: report the replication fit and the schedule-parity check at all three trained corners.
4. **5×5 grid vs GT (headline)**: for every $(m_\text{visc}, m_\text{st})$ cell run the conditioned rollout,
   and show it **against ground truth** (GT is available everywhere except the held-out corner — for the
   interior, GT = the analytic fluid at the interpolated $(\mu, \sigma_{st})$). A labeled montage (still) is
   required; a grid video and/or interactive `custom_html` is a strong plus. Mark the three trained corners
   and the held-out corner distinctly.
5. **Held-out corner test**: does the conditioned net generalize to **(high visc, high ST)** it never saw?
   Compare to the true (high μ, high σ) fluid and report honestly — success, partial, or failure with the
   mechanism.
6. A **diagnostic** backing the eye (e.g. per-cell trajectory RMSE vs GT as a 5×5 heatmap; roundness and
   spread trending smoothly along each axis).

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope claims to exactly what was tested (2D, one resolution, these scenes, this descriptor). Keep the three
  registers separate: observed (the grid/RMSE), hypothesized (why the interior interpolates or the held-out
  corner does/doesn't generalize), and what would test it.
- Distinguish **edge-exactness** (a fit + schedule-parity property at the corners) from **interior fidelity**
  (how close the interpolated cells are to GT) — do not let one stand in for the other.
- Honest `hypothesis` + `limitations` in the manifest. If the held-out corner fails, that is a finding, not a
  defect to hide.

## Visualization standard (graded)
- The 5×5 montage must be legible on iPad, axes labeled ($m_\text{visc}$ vs $m_\text{st}$), trained/held-out
  corners marked. Where you show NN vs GT, make the comparison visible (overlay or adjacent).
- **View every image and watch every video before writing a single finding.** Confirm the trained corners
  match GT, the ST axis rounds *gradually* (the saturation fix worked), no cell is degenerate/exploded/pinned,
  and the held-out corner shows its true behavior. Regenerate anything misleading.

## Training textbook contribution (required)
Add **one short, standalone core page** (objective voice, `spec/style_training_report.md`), e.g.
`core/18-conditioned-fluid.md`, on conditioning one network across two *fluid* axes (viscosity + surface
tension) and what the interior + held-out corner reveal. Keep it **tight, intuition-first** (the brevity
bar): the key idea, why a gentle ST schedule is needed for a visible continuum, and what the held-out corner
teaches about interpolation vs extrapolation. Register it in `reports/training/index.json`.
- Build on and link [[conditioned-material-net]] (same protocol, the descriptor-not-weights lesson),
  [[surface-tension]], [[viscosity]], and [[learned-material-interpolation]]. Contrast honestly: unlike the
  fluid/elastic/snow case where the interior left the manifold, here the two axes are both *fluid* variations,
  so ask whether the interior stays physical.
- Every `[[link]]` must resolve. If you lean on any new math, add the prerequisite first.

## Output contract
Write `runs/material-variants/generalize-one-nn-across-viscosity-and-surface-tension/manifest.json`
(schema v2 — copy the shape from `runs/material-variants/one-nn-for-three-materials/manifest.json`) plus its
media, with `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (the 5×5 montage,
grid video / `custom_html`, the RMSE heatmap, edge-exactness table, held-out comparison), and
`training_refs[]` including the new page. `direction`=`material-variants`,
`task_id`=`generalize-one-nn-across-viscosity-and-surface-tension`, `status`=`active`.
- **Keep prose fields tight** (brevity guidance). **Write the manifest LAST**; every media `src` must resolve
  to a real file on disk (a dangling ref is rejected).

## Paths & params
- Reuse `sim/one_nn_materials.py` (conditioned-net architecture, training, unified rollout, grid rendering)
  and `sim/fluid_surface_tension.py` (CSF surface tension + viscous fluid). Put new code in
  `sim/one_nn_fluids.py` (leave both parents intact).
- Run dir: `runs/material-variants/generalize-one-nn-across-viscosity-and-surface-tension/`
- Grid `n_grid=128` for the fluid (match the ST/viscosity sims); f32; `ti.init(arch=ti.gpu)`; headless
  (matplotlib Agg + imageio), **no `ti.GUI`**.
- Viscosity levels: reuse the oil→high ordering (μ up to ~1.0), per-cell stable dt (viscous limit for μ,
  capillary limit for σ). Surface tension: the gentle calibrated range above.

## Definition of done
- The three trained corners reproduce GT (replication fit + schedule-parity reported); $\sigma_{st}(m_\text{st})$
  hits the trained values exactly at the corners.
- A legible **5×5 conditioned-vs-GT grid** where the ST axis rounds **gradually** (saturation fixed), every
  interior cell finite and physical, trained + held-out corners marked; plus the **held-out (high,high)
  corner** compared to its true fluid and reported honestly.
- **The task is finished within your turn** — run all training/sims/renders to completion (block/poll within
  your turn), view every output, then finalize. Do not end the turn waiting on a background job.
- Manifest with scoped `findings`, honest `hypothesis` + `limitations`; every media `src` resolves.
- One short standalone training page added and registered; render-clean KaTeX; every `[[link]]` resolves.

## Known failures to avoid
- **Calibrate the gentle ST range on a cheap isolation blob BEFORE the full pipeline** — if σ still saturates
  by the second row the whole grid is uninformative. Verify the transition is gradual first.
- **Verify edge-exactness at each trained corner before trusting interior cells** (the schedule must reduce to
  the exact trained config at the corners) — this is the exact class of bug (`iso`/band or here σ schedule
  parity) that a predecessor got sent back for.
- Per-cell stability: high viscosity needs a small dt (viscous limit); high σ needs a small dt (capillary
  limit); the held-out (high,high) corner needs the smaller of the two. A frame with particles flung to the
  corner is instability, not a result — shrink dt.
- A learned stress is not guaranteed dissipative; keep the net small, check every rollout finite, and do not
  ship a blown-up or scattered cell as a result.
- Do not spawn a long training/render job in the background and end your turn waiting on it. Run to completion.
