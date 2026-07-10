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

## How to structure it (this is the crux — READ CAREFULLY; corrected from a first attempt)
**Both viscosity AND surface tension must be LEARNED by the network — surface tension must NOT be analytic.**
A first attempt kept surface tension as the analytic CSF force scheduled by the descriptor; the user rejected
that. The reason it happened: a per-particle stress net (as in `one_nn_materials`) only sees a particle's own
local state $(S, C, v, J_p)$, which carries **no information about the interface curvature** that surface
tension depends on, so it structurally cannot represent surface tension. The fix is to **give the network the
interface signal and have it learn the capillary force**, exactly as viscosity is learned from $C$.

Two learned pieces, one shared descriptor $m=(m_\text{visc}, m_\text{st})$:
- **A conditioned per-particle stress net** for the bulk stress (weakly-compressible pressure + Newtonian
  viscous term), from local features + $m$, as `net_sigma_cond` does. Viscosity is learned here (it fits
  naturally; cf. learned-viscosity).
- **A learned capillary-force network for surface tension.** Surface tension is a grid force set by the
  **interface curvature**, a non-local quantity, so this network reads the **local neighbourhood of the
  smoothed grid density field** (a stencil around each grid node — e.g. a $3\times3$ or $5\times5$ patch of
  the smoothed indicator $\phi$, and/or its gradient components) **plus $m_\text{st}$**, and outputs the
  capillary force at that node. **Do NOT feed it the analytic curvature $\kappa$** (that would reduce it to
  echoing the formula $f=\sigma\kappa\nabla\phi$); feed it the rawer density stencil so it must *infer* the
  curvature and produce the force itself. Train it **supervised against the analytic CSF force as the target**
  (generate targets by running `fluid_surface_tension.py`'s CSF at the trained $\sigma$ values), the same
  supervised-replication paradigm used for the viscous stress and the material stresses.
- **The learned rollout uses NO analytic surface tension.** At every cell it is: (learned stress net) +
  (learned capillary net). The analytic CSF exists ONLY to (a) generate supervised targets and (b) be the
  ground-truth the 5×5 grid is compared against. If any analytic capillary force remains in the learned
  rollout, the task is not done.
- **Edge-exactness must hold two ways at each trained corner** (the `one_nn` review bar): (a) each learned
  net at $m=$ corner reproduces its analytic target (the replication fit — report it for BOTH the stress net
  and the capillary net), and (b) the full learned rollout at the corner reproduces the true fluid (analytic
  stress + analytic CSF) to GPU-noise level. Report both before trusting any interior cell.
- **The held-out corner is now a real test of learned surface tension**: at $(1,1)$ the capillary net must
  produce the right force for an $m_\text{st}$ it only ever saw at low viscosity, combined with high
  viscosity. Whether the learned capillary law generalizes there is the headline question.

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
2. **Collect GT training states + targets** over several varied scenes per trained condition (reuse the
   `train_scenes` pattern + mirror augmentation). Train **both** learned pieces: the conditioned per-particle
   **stress net** (pressure + viscous, targets = analytic stress) and the learned **capillary-force net**
   (targets = the analytic CSF force per grid node, from `fluid_surface_tension.py`, at the trained $\sigma$).
   Persist until each net reproduces its analytic target at every trained corner.
3. **Edge-exactness**: report the replication fit for **both** nets (stress and capillary) and the full
   learned-rollout-vs-true-fluid parity at all three trained corners.
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
tension). Keep it **tight, intuition-first** (the brevity bar). The key teaching point is the one this task
turns on: **a per-particle stress net can learn viscosity (a local quantity) but NOT surface tension (a
non-local, curvature-driven interface force) — the network has to be given the interface neighbourhood to
learn a capillary law at all.** That local-vs-non-local distinction is the spine of the page. Then: the
gentle ST schedule for a visible continuum, and what the held-out corner teaches about generalizing a learned
capillary force. Register it in `reports/training/index.json`.
- Build on and link [[conditioned-material-net]] (same protocol, the descriptor-not-weights lesson),
  [[surface-tension]], [[viscosity]], and [[learned-material-interpolation]]. A draft of this page from the
  first (analytic-ST) attempt exists — **rewrite it to reflect LEARNED surface tension**, not analytic.
- Every `[[link]]` must resolve. If you lean on any new math (e.g. curvature, the divergence of a normal),
  it already lives in [[vector-calculus]] — link it; add any other prerequisite before linking.

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
- **Surface tension is LEARNED by a network, not analytic** — the learned rollout contains no analytic
  capillary force; the analytic CSF is only the supervised target + the GT comparison. (This is the whole
  point of the correction; a rollout that still calls the analytic CSF is a reject.)
- Both learned nets (stress + capillary) reproduce their analytic targets at the three trained corners, and
  the full learned rollout reproduces the true fluid there (edge-exactness reported for both).
- A legible **5×5 learned-vs-GT grid** where the ST axis rounds **gradually** (saturation fixed), every
  interior cell finite and physical, trained + held-out corners marked; plus the **held-out (high,high)
  corner** — a real test of the learned capillary law — compared to its true fluid and reported honestly.
- **The task is finished within your turn** — run all training/sims/renders to completion (block/poll within
  your turn), view every output, then finalize. Do not end the turn waiting on a background job.
- Manifest with scoped `findings`, honest `hypothesis` + `limitations`; every media `src` resolves.
- One short standalone training page added and registered; render-clean KaTeX; every `[[link]]` resolves.

## Known failures to avoid
- **Do NOT keep surface tension analytic.** The first attempt learned only the viscous stress and left the
  capillary force analytic; that was rejected. A per-particle stress net cannot see interface curvature, so
  surface tension needs its own network fed the density neighbourhood. If your learned rollout still calls the
  analytic CSF force, you have not done the task.
- **Do NOT end your turn waiting on a background job.** The first attempt repeatedly spawned a smoke test and
  ended its turn to "wait", which stalled the task three times. Run every training/sim/render **synchronously,
  in the foreground, to completion within your turn** (block on it; poll your own subprocess if you must).
  Develop with `--quick`, then run the full pipeline to completion before you finalize. A running job is
  something to wait for in-turn, never a reason to stop.
- **Calibrate the gentle ST range on a cheap isolation blob BEFORE the full pipeline** — if σ still saturates
  by the second row the whole grid is uninformative. Verify the transition is gradual first. (The prior
  calibration found σ_max≈0.079 with a nonlinear m_st schedule giving a gradual roundness ramp — reuse or
  re-derive it.)
- **Verify edge-exactness at each trained corner before trusting interior cells** (the schedule must reduce to
  the exact trained config at the corners) — this is the exact class of bug (`iso`/band or here σ schedule
  parity) that a predecessor got sent back for.
- Per-cell stability: high viscosity needs a small dt (viscous limit); high σ needs a small dt (capillary
  limit); the held-out (high,high) corner needs the smaller of the two. A frame with particles flung to the
  corner is instability, not a result — shrink dt.
- A learned stress is not guaranteed dissipative; keep the net small, check every rollout finite, and do not
  ship a blown-up or scattered cell as a result.
- Do not spawn a long training/render job in the background and end your turn waiting on it. Run to completion.
