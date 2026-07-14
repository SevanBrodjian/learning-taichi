# Worker brief: Train One NN to Mimic Viscosity and Surface Tension (whole-physics)

## Effort tier: deep
Genuinely hard — the network must replace the **entire** per-particle material update, not just the stress.
Persist: iterate, debug, run the sweeps it needs. Do not stop at the first plausible result, and do not
quietly narrow the scope back to "learn the stress" (that was the previous attempt's mistake).

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `train-one-nn-to-mimic-viscosity-and-st`. You are **NOT the
orchestrator**. Do not spawn agents. Read this brief, do the task, write all results under
`runs/material-variants/train-one-nn-to-mimic-viscosity-and-st/`, extend the training textbook, and exit.
**Do not commit.** Fire the pings + live status below.

## THE CORRECTION (why this task exists — do not repeat the mistake)
A previous attempt (`generalize-one-nn-across-viscosity-and-surface-tension`) learned only the per-particle
**stress** and kept the state rules + surface tension **analytic**. The user rejected that. This time the
network must learn the **ENTIRE material physics** — it replaces the whole per-particle material computation
(the momentum contribution scattered to the grid **and** the evolution of the particle's carried state),
keeping only the **MPM transfer scaffolding** (particle-to-grid scatter, the grid update with gravity +
boundary, grid-to-particle gather). If any part of the material physics (stress, the volume/deformation
update, or the surface-tension force) is still computed by an analytic equation inside the learned rollout,
the task is not done.

## Objective
Train **one** network conditioned on a **two-scalar descriptor** `m = (m_visc, m_st)` to reproduce a
weakly-compressible liquid across **viscosity** and **surface tension**, where the network **is** the
material (it replaces the whole per-particle update inside MPM). Train on three corners of the descriptor
square, **hold out the fourth**, and show in a **5×5 grid vs ground truth** whether the network interpolates
across conditions and generalizes to the held-out corner.

**Trained corners (CONFIRM at the approval gate — the seed contradicts itself):** the seed both lists
(low visc, low ST) as a trained condition and says to hold it out. The coherent reading, matching the prior
held-out-corner design, is: **train (low visc, low ST) = (0,0), (high visc, low ST) = (1,0),
(low visc, high ST) = (0,1); hold out (high visc, high ST) = (1,1).** Use this unless the user's approval
says otherwise.

## Ground truth (forward, no gradients)
The GT is the forward liquid with viscosity **and** surface tension:
- Viscosity + the MLS-MPM fluid: use the **canonical** `sim.physics` (`sim.physics.simulate("fluid", ...,
  mu_visc=...)`) — do NOT fork it.
- Surface tension is **not yet canonical** (see `sim/physics/PROMOTION.md`); its working implementation is
  the CSF term in `sim/fluid_surface_tension.py`. For this task, generate the ST-bearing GT with that CSF on
  top of the canonical fluid. (Promoting ST into `sim/physics` with a golden test is a good *separate*
  follow-up; do not block this task on it, and note the dependency in limitations.)
- Ground truth needs no gradients — it is a forward sim used to produce the observations the network fits and
  the reference the results are compared against.

## How to structure it (the crux)
- **The network replaces the whole per-particle material update.** Input: the particle's position-free local
  state (deformation/affine/velocity/volume as needed) **plus the non-local interface signal** surface
  tension requires (e.g. a patch of the smoothed grid density around the particle) **plus the descriptor
  `m`**. Output: the per-particle contribution the P2G scatter needs **and** the update to the particle's
  carried state for the next step. No analytic stress / volume rule / capillary force remains in the learned
  rollout.
- **Keep the MPM scaffolding canonical**: the P2G scatter, the grid momentum→velocity + gravity + Coulomb
  floor + walls, and the G2P gather are the unchanged skeleton. Only the material is learned.
- **Training approach is a real design choice — flag it.** Per-step supervised regression onto the GT's
  instantaneous per-particle target is the cheap route, but the prior task showed it **compounds error over
  the rollout** (locally-correct forces still jet/blow up when integrated). Prefer a training signal that
  sees rollout stability (e.g. short unrolled/rollout-aware training, or a stability regularizer) if
  feasible; if you use pure per-step supervision, say so and treat long-horizon stability as an open risk.
- **Gentle ST schedule** (reuse the prior calibration: `sigma_max ≈ 0.079`, a nonlinear `m_st` map giving a
  gradual roundness ramp). Verify on a cheap blob first that the ST axis rounds **gradually**, not saturating.

## Deliverables
1. **5×5 learned-vs-GT grid** (headline): for every `(m_visc, m_st)` cell, the learned rollout **shown
   against the ground-truth liquid at that cell** (GT mandatory — overlay or adjacent, as video for the
   motion, not lone final frames). Trained corners + the held-out corner clearly marked. Still montage
   required; a grid video and/or interactive `custom_html` is a strong plus.
2. **Ground-truth reference clips** included and obvious — the user must be able to see what the target
   liquids look like (this was missing last time).
3. **Edge check at the trained corners**: the learned whole-material rollout reproduces the true liquid
   there (both shape and motion).
4. **Held-out corner test**: does the learned material generalize to (high visc, high ST) it never saw?
   Report honestly — success, partial, or failure with the mechanism.
5. A **per-cell fidelity diagnostic** (e.g. trajectory RMSE heatmap) — but judge physicality from the
   **videos**, not the number (a spike and a blob can share a center of mass).

## Evidence discipline (non-negotiable — CLAUDE.md)
Scope every claim to what was tested. Distinguish edge fidelity (at trained corners) from interior fidelity
and from held-out generalization. Honest `hypothesis` + `limitations`. If a cell is degenerate (fountain,
spike, blow-up), that is a finding to report, not to hide behind a low RMSE.

## Presenting results (brief + comparative — CLAUDE.md)
- Write a tight **`summary`** (1–2 paragraphs, shown by default) and put the detail in **`full_report`**.
- **Every comparison shows both sides against each other, ground truth mandatory, in the same medium as the
  claim** (motion → video).

## Training textbook contribution (required)
Add **one short, standalone core page** (objective voice, `spec/style_training_report.md`) on learning the
**whole material** (not just the stress) within fixed MPM scaffolding, and what the interior + held-out corner
reveal about interpolating a learned simulator. Keep it tight, intuition-first. Build on and link
[[conditioned-material-net]], [[surface-tension]], [[viscosity]], [[learned-material-interpolation]],
[[differentiating-the-rollout]] (the per-step-vs-rollout training point). Register it in
`reports/training/index.json`; every `[[link]]` must resolve; KaTeX-safe.

## Output contract
Write `runs/material-variants/train-one-nn-to-mimic-viscosity-and-st/manifest.json` (schema v2) plus media,
with `objective`, tight `summary`, detailed `full_report`, scoped `findings`, `hypothesis`, `limitations`,
typed `results[]` (the 5×5 learned-vs-GT grid, GT clips, held-out comparison, RMSE heatmap), and
`training_refs[]`. Write the manifest **last**; every `src` must resolve. `direction` = `material-variants`,
`status` = `active`.

## Paths & params
- Reuse `sim/one_nn_fluids.py` (grid harness, rendering, held-out logic) and `sim/one_nn_materials.py`
  (conditioned-net + whole-material state kernel pattern) as starting points; put new code in
  `sim/one_nn_whole_fluid.py`. Ground truth via `sim.physics` (+ `fluid_surface_tension` CSF for ST).
- `n_grid=128`, f32, `ti.init(arch=ti.gpu)`, headless (matplotlib Agg + imageio), **no `ti.GUI`**.

## Definition of done
- The learned rollout contains **no analytic material physics** — stress, state update, and surface tension
  are all produced by the network; only the MPM transfer + boundary are canonical.
- Trained corners reproduce the true liquid (shape + motion); a legible **5×5 learned-vs-GT grid** with GT
  shown throughout and the held-out corner tested and reported honestly.
- **Finish within your turn** — run all training/sims/renders to completion (block/poll in-turn; do NOT
  spawn a long job and end your turn waiting on it). View every figure/video before writing findings.
- Manifest with tight `summary` + `full_report`, honest `hypothesis` + `limitations`; every media `src`
  resolves. One short training page added and registered; links resolve; KaTeX clean.

## Known failures to avoid
- **Do NOT narrow to "learn the stress."** The network must produce the whole material update (incl. the
  state evolution and the surface-tension effect). Keeping any of it analytic in the rollout = reject.
- **Do NOT end your turn waiting on a background job.** Run everything to completion in-turn (the prior
  worker stalled repeatedly this way).
- Per-cell stability: high viscosity and high ST each force a smaller dt; the held-out (high,high) corner
  needs the smaller. A frame with particles flung to the corner is instability, not a result.
- Include the **ground truth** in every comparison; a learned output with no GT reference is not evidence.
