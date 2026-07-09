# Worker brief: Varying liquid viscosity (honey to oil)

> Direction: `material-variants`. Task id: `varying-liquid-viscosity`.
> A **forward** (no gradients) demonstration: take the weakly-compressible MPM fluid and add a
> **viscosity** term, then show how the same scene changes as viscosity is swept across a dramatic range,
> from a thin oil that splashes to a thick honey that oozes and coils. Primarily a learning/visual task.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `varying-liquid-viscosity`. You are **NOT the orchestrator**. Do not
spawn further agents. Read this brief, do the task, write **all** results to disk under
`runs/material-variants/varying-liquid-viscosity/`, extend the training textbook, and exit. **Do not
commit** — the orchestrator reviews and commits. Fire the two pings.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task varying-liquid-viscosity "<one plain sentence>"
python harness/tools/notify.py --kind finished --task varying-liquid-viscosity "<one plain sentence>"
```
Use `--kind blocked` on a hard stop. One human sentence, never a metrics dump.

## Objective
Add a **viscosity** to the MPM fluid and demonstrate, forward-only, how the fluid's behavior changes as
viscosity is varied **dramatically** (roughly oil-thin to honey-thick), across a few scenes. The point is to
see and teach what viscosity *does* to a liquid, backed by a simple quantitative diagnostic.

## Background: how viscosity enters an MLS-MPM fluid
The current fluid (in `sim/material_showcase.py` / `sim/mpm88.py`) is weakly compressible with a pressure
$\sigma^{\text{fluid}} = -4E\,(J-1)\,I$ and **no viscosity** — it resists compression but not shear rate, so
it is effectively inviscid and splashes freely. Viscosity is resistance to the **rate of shear**. Add a
Newtonian viscous stress proportional to the symmetric part of the velocity gradient. APIC already carries an
estimate of the velocity gradient in the affine matrix $C_p$, so a clean way to add viscosity is a stress
term
$$
\sigma^{\text{visc}}_p = \mu_{\text{visc}}\,\big(C_p + C_p^{\top}\big),
$$
added into the particle stress that P2G scatters (the symmetric part $C+C^{\top}$ is the strain **rate**;
its trace is compression, its off-diagonal is shear). Sweeping $\mu_{\text{visc}}$ over a couple of decades
takes the fluid from thin to thick. (An alternative knob is PIC/FLIP blending — more PIC is more numerically
dissipative, more viscous-looking — you may mention it, but prefer the explicit physical term above so the
sweep is a real parameter, not a numerical artifact.) **Watch stability:** an explicit viscous term is
diffusive, with its own step limit roughly $\Delta t \lesssim \rho\,\Delta x^2 / \mu_{\text{visc}}$, so the
thickest settings may need a smaller `dt`. Keep the rollout finite; a blown-up frame is a bug, not honey.

## Experiments / deliverables
- Implement the viscosity term (a forward-only script, e.g. `sim/fluid_viscosity.py`, reusing the fluid
  forward from `sim/material_showcase.py`). Confirm the low-viscosity limit still matches the inviscid fluid.
- **Sweep viscosity across a dramatic range** (at least a thin / medium / thick triple; label them like
  "oil", "syrup", "honey"). Keep everything else fixed.
- **Scenes (pick 2)** that make viscosity visible, same initial condition across the viscosity panels:
  1. **Pour / drop** — a stream or blob falls onto the floor. Thin oil splashes and spreads fast; thick
     honey lands slowly, mounds up, and may **coil** or hold a peak before relaxing.
  2. **Dam-break / column collapse** — a block of fluid released sideways. Thin runs out into a fast flat
     sheet; thick advances slowly with a steep, rounded front and keeps height far longer.
- **Quantify it**: a simple diagnostic vs viscosity that backs the eye — e.g. front position / spread width
  versus time, or the time to reach a given spread. It should be monotonic in viscosity.
- Render **labeled side-by-side** videos (thin | medium | thick) per scene, plus a still per scene.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Forward demonstration on a couple of fixed scenes at fixed resolution, one viscosity model (a Newtonian
  strain-rate term), hand-tuned stable parameters. Scope claims to that. It shows *what viscosity does to
  this fluid here*, not a validated rheology model or a claim about real honey.
- Manifest carries honest `hypothesis` (why the strain-rate term produces the slowdown) and `limitations`
  (2D, weakly compressible, explicit viscosity, tuned dt, no free-surface tension, etc.).

## Visualization standard (graded)
- Same initial condition across the viscosity panels; clear labels (viscosity value + a plain name).
- **View every video and still before writing findings.** Confirm the thick fluid is visibly slower/thicker
  and nothing blew up, went NaN, or clipped through a wall. Regenerate anything degenerate.
- Readable labels; a legible floor/domain. Prefer a clean particle render tinted by the fluid.

## Training textbook contribution (required)
Add **one short, standalone** page (suggested `reports/training/core/09-viscosity.md`, id `viscosity`) in the
impersonal textbook voice (`spec/style_training_report.md`): what viscosity *is* (resistance to shear rate,
momentum diffusion), how it enters the MPM stress as a strain-rate term built from the symmetric part of the
velocity gradient, and what the honey-to-oil spectrum looks like. Tie to `[[material-showcase]]`,
`[[constitutive-models]]`, `[[mpm-in-context]]`, and `[[linear-algebra]]` (the symmetric part / trace vs
shear) — all exist, every `[[link]]` must resolve. Embed a viewed still. Captions are plain prose (no
`$math$`). Render-check the KaTeX. Add the page to `reports/training/index.json` (core group).

## Output contract
Write `runs/material-variants/varying-liquid-viscosity/manifest.json` (schema v2 — copy the shape from
`runs/material-variants/implement-nondifferentiable-material-variants/manifest.json`) plus media, with
`objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (videos, stills, a diagnostic
`table` or `plot`), and `training_refs[]`. Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/material-variants/varying-liquid-viscosity/`
- New code: `sim/fluid_viscosity.py` (forward-only); reuse the fluid forward from `sim/material_showcase.py`.
- Suggested: `n_grid=128`, `n_particles≈8k`, fluid `E≈180–400`, base `dt≈1e-4` (smaller for the thickest
  viscosity if it destabilizes). Sweep $\mu_{\text{visc}}$ across ~2 decades for a dramatic thin→thick range.

## Definition of done
- At least two scenes, each a labeled thin|medium|thick viscosity comparison where the difference is
  **visibly clear**, plus a monotonic diagnostic backing it.
- Every video/still **viewed**; nothing degenerate ships.
- Forward-only. Training page renders (KaTeX), reads standalone, **every `[[link]]` resolves**, embeds a
  viewed figure. Manifest complete schema-v2.

## Known failures to avoid
- **Viscous stiffness**: high $\mu_{\text{visc}}$ makes the explicit term stiff; respect the diffusion step
  limit and shrink `dt` for the thick cases. A blown-up "honey" is a bug — watch the video.
- Do not fake viscosity purely with PIC damping and call it a physical sweep; use the strain-rate stress.
- Headless only (no `ti.GUI`). Forward only — no gradients/optimizer.
- Do not spawn a long background render then end the turn without viewing the outputs and confirming files.
