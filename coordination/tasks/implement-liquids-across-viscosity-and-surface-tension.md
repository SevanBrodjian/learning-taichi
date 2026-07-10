# Worker brief: Implement Liquids across Viscosity and Surface Tension

## Effort tier: standard
Normal depth and iteration. Get a correct, stable surface-tension term and a clean 3×3 grid; do not
over-engineer into a research sweep. One good forward implementation, verified by eye and by a simple
diagnostic, plus a training page.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `implement-liquids-across-viscosity-and-surface-tension`. You are **NOT
the orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to disk
under `runs/material-variants/implement-liquids-across-viscosity-and-surface-tension/`, extend the training
textbook, and exit. **Do not commit** — the orchestrator reviews and commits your work. Fire the pings below.

## Notifications (exactly two) + live status
At the start:
```
python harness/tools/notify.py --kind started --task implement-liquids-across-viscosity-and-surface-tension "Starting the surface-tension implementation and the viscosity x surface-tension grid."
```
When your results are on disk:
```
python harness/tools/notify.py --kind finished --task implement-liquids-across-viscosity-and-surface-tension "<one plain sentence: what's ready to review>"
```
Use `--kind blocked` instead of `finished` if you hit a hard stop.

**Live status (call ~4-6 times over the run, one short phrase each):**
```
python harness/tools/task_status.py --direction material-variants --task implement-liquids-across-viscosity-and-surface-tension --step "<a few words: current step>"
```
Suggested milestones: "implementing surface tension", "isolation test: droplet rounds up", "calibrating
stability dt", "running the 3x3 grid", "rendering montage + videos", "writing training page".

## Objective
Add a **surface-tension** parameter to the weakly-compressible MLS-MPM fluid (building directly on
`sim/fluid_viscosity.py`), then show, forward-only (no gradients, no optimizer, no loss), how a liquid's
behavior changes across the two independent axes **viscosity × surface tension**. The headline deliverable is
a **full 3×3 grid** — one fluid for every combination of {low, medium, high} viscosity × {none, medium, high}
surface tension — from a shared starting condition, plus supporting clips from a start that makes surface
tension's signature obvious (beading / droplet cohesion / merging).

## Background: what surface tension is and how to add it
Viscosity (already implemented) resists the *rate* of shear. **Surface tension** is different: it is a force
that **minimizes the free-surface area**, pulling a blob toward a round droplet, making separated blobs bead
up and merge, and resisting spreading into a thin sheet. It is a *capillary* effect concentrated **at the
interface**, not a bulk stress.

Recommended implementation: the **continuum surface force (CSF / Brackbill)** on the grid, which fits MLS-MPM
cleanly because the grid already carries a density field.
- Give each fluid particle a color/indicator $c_p = 1$. Scatter it to the grid alongside mass and **smooth**
  it (a couple of Gaussian/box smoothing passes over the grid) to get a diffuse interface field $\phi$.
- The surface **normal** is $n = \nabla\phi / \lVert\nabla\phi\rVert$ (finite differences over neighboring
  grid cells), and the **curvature** is $\kappa = -\nabla\!\cdot n$.
- Apply the capillary force per unit volume $f = \sigma_{st}\,\kappa\,\nabla\phi$ (the $\nabla\phi$ factor
  concentrates it on the interface band), added to the grid velocity in `grid_op`
  ($v \mathrel{+}= \Delta t\, f / \rho$), or scattered to particles. $\sigma_{st}$ is the new scalar knob;
  $\sigma_{st}=0$ must recover the current viscous fluid **exactly**.

A simpler particle-cohesion approximation (pull each particle along the smoothed density gradient toward the
bulk) is acceptable **if** it passes the isolation test below convincingly; CSF is preferred. Either way,
compute everything from the grid so there is no O(n²) neighbor search.

**Stability.** Surface tension adds a capillary timestep limit roughly
$\Delta t \lesssim \sqrt{\rho\,\Delta x^{3} / (2\pi\,\sigma_{st})}$. Calibrate a stable dt per
(viscosity, surface-tension) cell the way `fluid_viscosity.py`'s `--calibrate` does for viscosity — a
blown-up, particles-flung-to-the-corner frame is a bug, not a fluid. Integrate every panel to the **same
physical time** so the grid frames are synchronized even when dt differs.

## Experiments / deliverables
1. **Isolation test (do this first, before the grid).** With gravity off (or a blob resting), a **square or
   irregular blob** must relax into a **round droplet** as $\sigma_{st}$ increases, and stay diffuse/blocky at
   $\sigma_{st}=0$. Two nearby blobs must **bead and merge** under surface tension and not under zero. This is
   the check that the term is real surface tension and not just extra damping. Report a **roundness diagnostic**
   (e.g. perimeter²/area, or 1 − aspect-ratio, or spread-width) that moves monotonically with $\sigma_{st}$.
2. **The 3×3 grid (headline).** Pick three viscosities and three surface tensions (including
   $\sigma_{st}=0$ so the top row reproduces the pure-viscosity fluid). Run the **same starting condition**
   (a drop onto the floor, or a small dam-break) at all nine cells and render a **3×3 montage** — viscosity
   along one axis, surface tension along the other, clearly labeled. A late-frame **still montage** is
   required; a **9-panel video** (or an interactive `custom_html` grid) is a strong plus.
3. **A second starting condition** that showcases the surface-tension axis specifically (e.g. a blob or two
   blobs with low gravity where beading/merging dominates), as a short clip or still.
4. A **simple diagnostic** backing the eye: show that along the surface-tension axis a roundness/beading
   measure rises and spread falls, and along the viscosity axis the familiar oil→honey ordering holds, and
   that the two axes are **visibly separable** (surface tension changes shape/cohesion; viscosity changes
   speed of spreading).

Reuse `sim/fluid_viscosity.py`'s skeleton (p2g / grid_op / g2p, seeding, headless matplotlib panels). Put the
new code in a new file, e.g. `sim/fluid_surface_tension.py`, so the viscosity demo stays intact.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope every claim to exactly what was tested (2D, one resolution, these scenes, these hand-tuned knobs).
  The mapping from $\sigma_{st}$ to a physical surface tension is **not** calibrated — labels are evocative,
  only the monotonic trend and the axis-separation are claimed.
- The manifest carries an honest `hypothesis` (why the CSF term produces rounding/beading) and `limitations`
  (weakly compressible, no true free surface, capillary dt limit, no gradients).

## Visualization standard (graded, not optional)
- The 3×3 montage must be **legible on iPad**: label the viscosity and surface-tension axes, keep panels
  large enough to read the shape. Colors may evoke the fluid.
- **Show the quantity the objective is about** — surface tension is about *shape and cohesion*, so frame the
  isolation test and the grid so rounding/beading is visible, not just "the blob moved".
- **View every image and watch every video you export before writing a single finding.** Confirm the
  $\sigma_{st}=0$ column/row matches the pure-viscosity fluid, the high-$\sigma_{st}$ blobs actually round up
  and merge, no panel is blown up / clipped / pinned in a corner, and axes/labels are right. A misleading or
  degenerate figure is regenerated, not shipped.

## Training textbook contribution (required)
Add **one short, standalone core page** (objective textbook voice, `spec/style_training_report.md`), e.g.
`core/17-surface-tension.md`, titled around "Surface tension: the force that minimizes the interface". Register
it in `reports/training/index.json`. Lead with the intuition (a force that pulls a blob round and beads
droplets, concentrated at the surface, unlike bulk viscosity), then the CSF math (normal, curvature, the
capillary force), then the one parameter's effect (rounding/beading and the capillary timestep limit), and a
short honest scope. Keep it **tight** and intuition-first per the new brevity guidance — implementation
details and exact per-cell dt values stay in the manifest, not the page.
- Contrast it cleanly with [[viscosity]] (rate-of-shear bulk stress vs interface-minimizing capillary force)
  and link [[material-showcase]] / [[constitutive-models]] for the shared solver, and [[mpm-in-context]] /
  [[linear-algebra]] for the grid gradient/divergence.
- **Over-include math prerequisites** it leans on. Curvature and the divergence of a normal field, and the
  gradient/divergence operators on the grid, should resolve to a prerequisite that actually covers them —
  extend `prerequisites/` (e.g. the math toolkit or a short vector-calculus page) **before** linking, and make
  sure every `[[link]]` you write points at content that exists.
- Embed one informative figure (the isolation-test rounding, or the 3×3 montage).

## Output contract
Write `runs/material-variants/implement-liquids-across-viscosity-and-surface-tension/manifest.json`
(schema v2 — copy the shape from `runs/material-variants/varying-liquid-viscosity/manifest.json`) plus its
media, with `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (the 3×3 montage
image, the grid video and/or `custom_html`, the isolation-test clip/still, a diagnostic plot or table), and
`training_refs[]` including your new page's id. Set `direction` = `material-variants`, `task_id` =
`implement-liquids-across-viscosity-and-surface-tension`, `status` = `active`.
- **Keep the prose fields tight** (see brevity guidance): `objective` one or two sentences, `findings` leads
  with the headline then the few points that matter.
- **Write the manifest LAST**, after every media file it references exists on disk; every `src` must resolve
  to a real file (a dangling ref is a broken dashboard tile and will be rejected).

## Paths & params
- New code: `sim/fluid_surface_tension.py` (reuse `sim/fluid_viscosity.py`).
- Run dir: `runs/material-variants/implement-liquids-across-viscosity-and-surface-tension/`
- Grid: `n_grid=128`, `dim=2`, `E=200` (match the viscosity fluid); f32; `ti.init(arch=ti.gpu)`.
- Headless only — export frames/video with matplotlib Agg + imageio; **no `ti.GUI` loop**.

## Definition of done
- A working, stable surface-tension term where $\sigma_{st}=0$ recovers the viscous fluid exactly, and the
  **isolation test passes**: higher $\sigma_{st}$ visibly rounds a blob and merges droplets, with a monotonic
  roundness diagnostic.
- A legible, labeled **3×3 viscosity × surface-tension montage** (still) plus at least one grid clip or
  interactive, and a second surface-tension-showcasing start.
- **The task is finished within your turn.** Run every sim/render to completion (block or poll within your
  turn); do not end the turn "waiting" on a background job. View every output before finalizing.
- Manifest carries scoped `findings`, honest `hypothesis` + `limitations`; every media `src` resolves.
- One short standalone training page added and registered in `index.json`, render-clean (KaTeX), standalone
  voice, **every `[[link]]` resolves**, and the math prerequisites it leans on (curvature, divergence of a
  normal) exist in the prerequisites layer.

## Known failures to avoid
- **Verify the surface-tension force sign and the isolation test on a tiny scene before running the full 3×3
  grid.** A wrong-sign capillary force *explodes* the interface instead of rounding it; catch that on one
  cheap blob, not after a long grid render.
- The explicit capillary term has its own dt limit; the high-surface-tension cells will need a smaller dt.
  A frame with particles flung into the domain corner is the stability limit violated, not a fluid — shrink dt.
- Do not spawn a long render/sim in the background and end your turn waiting on it. Run it to completion.
- GPU atomic-add is not bitwise reproducible; if a frame looks off, rerun before concluding.
