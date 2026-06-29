# Worker brief: Elastic vs fluid vs snow under the same control task

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `fluid-vs-snow`. You are **NOT the orchestrator**. Do not spawn
further agents. Read this brief, do the task, write **all** results to disk under
`runs/material-variants/fluid-vs-snow/`, extend the training textbook, and exit. **Do not commit** — the
orchestrator reviews and commits your work. Fire the two pings below.

## Notifications (exactly two)
At the start:
```
python harness/tools/notify.py --kind started --task fluid-vs-snow "Starting the elastic-vs-fluid-vs-snow throw comparison."
```
When your results are on disk:
```
python harness/tools/notify.py --kind finished --task fluid-vs-snow "Done; three-material throw comparison and a training page are on disk."
```
Use `--kind blocked` instead of `finished` if you hit a hard stop. **One sentence of human status, never a
metrics dump or a technical report.**

## Objective
Hold the control task fixed (throw a blob's center of mass to a target by backprop on a shared initial
velocity `v0`) and swap **only the constitutive model** — weakly-compressible **fluid**, **elastic**
(corotated / neo-Hookean), and **snow** (elastoplastic with hardening) — to answer: under an identical
optimization setup, how **controllable** is each material, and how does its **gradient behave** through
the rollout? This is a comparison across three conditions, not a claim about materials in general.

## Background — what already exists
- `sim/diffmpm.py` is the differentiable throw task. Its current constitutive law is already the
  **weakly-compressible fluid**: `stress = -dt * 4 * E * p_vol * (J - 1) * inv_dx²`, with `J` the volume
  ratio tracked per particle. Reuse this as the **fluid** condition (it is your known-good baseline).
- `sim/softened_wall.py` / `sim/diffmpm_pathologies.py` show the mass-stabilized grid op
  (`vel = grid_v_in / max(m, MASS_EPS)`). **Keep mass stabilization ON in every condition** so the
  already-fixed grid-mass overflow (`reports/training/core/03-failure-modes.md`) cannot confound the
  comparison.
- Optimizer: reuse the Adam-on-`v0` loop from `sim/diffmpm.py` (`optimize()`), identical across all three
  materials so the only thing that changes is the physics.

## What to implement
Write a new self-contained script `sim/material_variants.py` (seed from `sim/diffmpm.py`; do **not**
break the existing scripts). Implement three constitutive models behind a `--material {fluid,elastic,snow}`
switch, sharing the same p2g/grid_op/g2p skeleton and the same autodiff tape:

1. **fluid** — the existing `J`-based weakly-compressible pressure (already in `diffmpm.py`). Track `J`.
2. **elastic** — track the **deformation gradient** `F` per particle (`F[0]=I`, updated each step by
   `F_new = (I + dt·C) @ F`). Use a **corotated** (or neo-Hookean) stress:
   `P = 2μ(F − R) + λ(J−1)J·F^{−T}` with `J = det(F)`, `R` from the polar decomposition of `F`;
   feed `stress·F^T` style Cauchy stress into the affine momentum exactly as the MLS-MPM update expects.
   A 2×2 polar decomposition / SVD has a closed form — implement it inside a `@ti.func` so it is
   differentiable (Taichi's `ti.svd` is available and differentiable in 2D; prefer it if simpler).
3. **snow** — elastic as above plus **plasticity**: after the elastic update, take `F = U Σ Vᵀ`, clamp the
   singular values `Σ` into `[1−θ_c, 1+θ_s]` (compression/stretch limits), and apply **hardening** by
   scaling μ, λ with `exp(ξ(1−Jp))` where `Jp` accumulates the plastic volume change. Use the standard
   Stomakhin snow parameters as a starting point (`θ_c≈2.5e-2`, `θ_s≈7.5e-3`, `ξ≈10`, base `E≈1e2..4e2`,
   `ν≈0.2`). Snow is the most numerically delicate — **lower `dt` and/or `E` as needed for stability**
   and record what you used.

**Gradient-flow probe FIRST (mandatory, anti-degeneracy).** Before any sweep, for **each** material run a
short probe (a handful of optimizer steps) and assert the loss strictly decreases and `v0` actually moves
off `(0,0)`. A flat-loss run means no gradient reached the control — **refuse to report a material whose
probe is flat**; diagnose (often `dt`/`E` too aggressive → NaN, or a non-differentiable op in the stress)
and fix before the full run. Never trust a flat-loss run (this killed an earlier multi-task attempt).

## Experiments / deliverables
Hold fixed across all three materials: same seeded blob, same target, same horizon (steps), same Adam
`lr` and iteration budget, same seed. Vary only `--material`. Measure and record per material:
- final loss and the loss curve (vs iteration);
- the optimized `v0*` and how close the center of mass got to the target;
- a qualitative read on **gradient health** — e.g. `|∇v0 loss|` over iterations, and whether/when any run
  goes non-finite. (You already have the per-iter grad norm in the optimize loop; log it.)
- Optional but valued: one cheap measure of how **deformable/spread** the material ends up (e.g. particle
  position variance at the final frame) — it helps explain *why* controllability differs.

Pick **one primary target** for the headline comparison; if cheap, add a second target to avoid reading
too much into one geometry (this strengthens the evidence but is not required if time/stability is tight).

## Evidence discipline (non-negotiable — see CLAUDE.md)
- This is **three constitutive models on one control task (the throw), one optimizer, one (or two)
  target(s)**. Scope every claim to exactly that. "On the throw task, snow was harder to control than
  fluid under identical Adam settings" — **not** "snow is harder to optimize."
- Keep observation (the measured losses/grad norms), hypothesis (why a material's loss landscape is
  rougher — plastic clamping is non-smooth, elastic stores recoverable energy, fluid forgets deformation),
  and what-would-test-it (more tasks, more targets, contact-driven tasks) cleanly separate.
- The manifest MUST carry honest `hypothesis` and `limitations` fields.

## Visualization standard (graded, not optional)
- **A side-by-side video of the three materials under their optimized throw**, each panel overlaying the
  **center-of-mass trajectory** and the **target marker** so the viewer can *see how close each got* — not
  just watch blobs move. Reuse the matplotlib/Agg headless renderer in `diffmpm.py` (no `ti.GUI`). A
  single stacked/triptych mp4 is ideal; three separate mp4s are acceptable.
- **An overlaid loss-curve plot** (three materials, one axes, labeled, log-y) and a **grad-norm plot**.
- A small **table**: material → final loss, distance-to-target, `v0*`, any NaN step, `dt`/`E` used.
- Labeled axes, readable fonts (renders small on iPad).

## Training textbook contribution (required)
Add **one short, standalone page** under `reports/training/core/` in the objective textbook voice
(`spec/style_training_report.md`): impersonal, no first/second person, no reference to "this run/task".
Teach **how the constitutive model enters the MLS-MPM stress and why that changes both the physics and the
gradient** — fluid (pressure from volume change, forgets shear history), elastic (deformation gradient `F`,
recoverable stress, smooth), snow (SVD clamp + hardening introduces a *non-smooth* plastic projection that
roughens the loss landscape). Embed the side-by-side figure or a frame. Register the new page in
`reports/training/index.json` and link prerequisites with `[[mls-mpm-forward]]`,
`[[differentiating-the-rollout]]`, `[[failure-modes]]`.

## Output contract
Write `runs/material-variants/fluid-vs-snow/manifest.json` (**schema_version "2"** — see `runs/README.md`)
plus media: `task_id` `fluid-vs-snow`, `direction` `material-variants`, `objective`, scoped `findings`,
`hypothesis`, `limitations`, typed `results[]` (the triptych video, the loss + grad plots, the table), and
`training_refs[]` including the new page id. Leave everything on disk; **do not commit**.

## Paths & params
- Run dir: `runs/material-variants/fluid-vs-snow/`
- Code: `sim/material_variants.py` (new)
- Start from: `n_grid=64`, `n_particles=4096`, fluid `E=400`, `dt=2e-4`, ~400–512 steps, Adam `lr≈0.1`,
  ~60–100 iters, fixed seed. Snow/elastic may need smaller `dt` or `E` — **record what you actually used**.

## Definition of done
- All three materials pass the gradient-flow probe (loss decreases, control moves) — no flat-loss runs.
- Manifest schema-v2 with honest `hypothesis` + `limitations`; the triptych video overlays COM trajectory
  and target; loss/grad plots and the table present; math render-checks in KaTeX.
- One standalone training page added and registered in `index.json`.

## Known failures to avoid
- **Verify gradients move the control before any long sweep** (the probe). An earlier multi-task run
  produced flat-loss garbage because no gradient reached the control — quarantine, don't report, such runs.
- Snow's SVD-clamp and the polar decomposition are the differentiability-risky spots; if a material NaNs,
  lower `dt`/`E` and check the stress `@ti.func` is finite, rather than abandoning the material.
- Do **not** spawn a long background command and then end the turn waiting on it — run to completion in
  the foreground and write results before firing the `finished` ping.
- Keep mass stabilization ON in all three conditions; do not reintroduce the fixed overflow bug.
