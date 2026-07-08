# Worker brief: Fluids, snow, and solids as differentiable simulations

> Direction: `material-variants`. Task id: `fluids-snow-and-solids-as-differentiable-simulations`.
> Follow-up to `implement-nondifferentiable-material-variants` (the forward showcase the user liked).
> The differentiable versions of these materials were tried before and **did not work functionally** —
> the gradients were bad and the particles behaved erroneously. Your job is to build a **simple, correct**
> differentiable implementation of each material and find out **what it takes to get meaningful gradients**,
> verifying rigorously and iteratively. **An honest negative result is an acceptable outcome** — if a
> material cannot give meaningful gradients, say so plainly and explain/theorize why. Do not fake success.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `fluids-snow-and-solids-as-differentiable-simulations`. You are **NOT
the orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to
disk under `runs/material-variants/fluids-snow-and-solids-as-differentiable-simulations/`, extend the
training textbook, and exit. **Do not commit** — the orchestrator reviews and commits. Fire the two pings.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task fluids-snow-and-solids-as-differentiable-simulations "<one plain sentence>"
python harness/tools/notify.py --kind finished --task fluids-snow-and-solids-as-differentiable-simulations "<one plain sentence>"
```
Use `--kind blocked` if you hit a hard stop. One human sentence, never a metrics dump.

## Objective
Build a **simple** differentiable MLS-MPM for each of the three materials — weakly-compressible **fluid**,
corotated **elastic**, Stomakhin-style **snow** — and determine **what it takes to get meaningful gradients**
through each. "Meaningful" is a specific bar: the autodiff gradient of a scalar loss with respect to a
control must **match a finite-difference estimate** (not merely be finite), and the forward rollout must
**behave physically** (particles do not disintegrate, blow up, or clip through walls). Report honestly per
material — including, if it happens, that elastic and/or snow **cannot** be made to give meaningful gradients
here, with a grounded explanation.

## Why the prior attempt is suspect (do not repeat it)
The earlier `sim/material_variants.py` / `runs/material-variants/fluid-vs-snow` optimized a **near-ballistic
COM-to-target throw** and only ever checked that "the loss went down". Two problems it hid:
1. On a COM-throw loss the constitutive model **barely matters** (the forward showcase confirmed COM
   ballistics are nearly material-independent), so it never really exercised the solid physics.
2. It **never finite-difference-checked** the elastic/snow gradients. Elastic "descended" but its optimized
   `v0` barely left the origin — a hallmark of a gradient that is finite but **wrong or vanishing**, not
   meaningful. That is very likely the "bad gradients / erroneous particles" the user saw.
So: **verify gradients numerically**, and pick a **short horizon** and a loss that actually depends on the
material, not the 512-step throw.

## Experiments / deliverables
Build on the **verified forward physics** from `sim/material_showcase.py` (the showcase the user liked) and
the correct diffmpm baseline `sim/diffmpm.py`. Write a fresh, small script (suggested `sim/material_diff.py`).
You may reuse the Taichi autodiff **scaffolding** pattern (time-indexed `needs_grad` fields, `ti.ad.Tape`,
per-step state so nothing needed is overwritten in place) from `sim/diffmpm.py` and `sim/material_variants.py`,
but **do not assume the elastic/snow gradient paths in the old file are correct** — that is exactly what you
are checking.

1. **Start with the fluid** (known-good from diffmpm) to validate your gradient-check harness end to end.
2. **Finite-difference gradient check — the core deliverable.** For a simple scalar loss $L$ and a control
   $\theta$ (start with initial velocity $v_0$; a body-force or $E$ are fine too), compute the autodiff
   $\partial L/\partial\theta$ and a **central finite-difference** estimate, and report per material:
   autodiff value, FD value, and **relative error**. Call a gradient "meaningful" only if the rel-error is
   small (state your threshold, e.g. a few percent) AND finite. Use a **short horizon** (start ~48–128
   steps) and a modest particle count so the check is clean and attenuation is mild.
3. **Functional check by eye, iteratively.** Render the forward rollout for each material at the tested
   settings and **confirm the particles behave** (no NaN, no disintegration, nothing off-screen or through
   a wall). If a material misbehaves, that is a bug to diagnose **before** trusting its gradient. Examine
   images at each iteration — this is required, and the user asked for it explicitly.
4. **A tiny optimization** for each material whose gradient checks out: descend $L$ (e.g. nudge the COM to a
   target, or hit a target height) for a handful of iterations and show it genuinely improves — with the
   control actually moving, not stalling at the origin like before.
5. **Document what it takes.** Record which knobs made gradients meaningful vs broken: horizon length
   (attenuation through the rollout), `dt` vs stiffness (CFL $\sim 1/\sqrt{E}$), **SVD degeneracy** at
   coincident singular values (an undeformed isotropic blob has $\sigma_1=\sigma_2$, where `ti.svd`
   derivatives blow up — guard it), **hard vs softened snow clamp** (the hard clamp is $C^0$, so its
   gradient may be zero/garbage — try a smooth clamp), gradient clipping, mass stabilization.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope every claim to the exact task, horizon, and per-material settings tested. Keep **"gradient is
  finite"** and **"gradient is meaningful (FD-verified, usable)"** strictly separate — conflating them is
  the specific error the prior attempt made.
- A **negative result is valid and welcome**: "on this setup, fluid gives meaningful gradients but snow's
  are dominated by clamp kinks / SVD degeneracy" is a real finding if verified. Theorize the mechanism and
  point at what would test it. Do not overclaim a working solid if the FD check fails.
- Manifest carries honest `hypothesis` and `limitations`.

## Visualization standard (graded)
- A **gradient-check table** (per material: autodiff, finite-difference, relative error, verdict).
- A **convergence curve** for each material whose optimization runs, and a **before/after particle state
  or short optimized-rollout video** so the behavior is visible.
- **View everything you render before writing findings.** A material whose particles disintegrated is a bug
  to fix, not a video to ship. Regenerate anything degenerate. Same-scale, labeled panels.

## Training textbook contribution (required)
Add **one short, standalone** page (suggested `reports/training/core/08-differentiable-materials.md`, id
`differentiable-materials`) in the impersonal textbook voice (`spec/style_training_report.md`): what it takes
to get meaningful gradients through fluid vs elastic vs snow, where the SVD and the plastic clamp help or
hurt, and — if a material fails — why. Tie to prerequisites that exist and must resolve: `[[material-showcase]]`
(the forward behaviors), `[[differentiating-the-rollout]]`, `[[svd-polar]]` (SVD differentiability and its
degeneracy), `[[failure-modes]]` (kinks, near-zero grid mass), `[[constitutive-models]]`, `[[material-stiffness]]`.
Embed a viewed figure. Captions are plain prose (visible captions — no `$math$` inside). Render-check KaTeX.

## Output contract
Write `runs/material-variants/fluids-snow-and-solids-as-differentiable-simulations/manifest.json` (schema v2 —
copy the shape from `runs/material-variants/fluid-vs-snow/manifest.json`) plus media, with: `objective`,
scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (the gradient-check `table`, convergence
`plot`s, particle `video`/`image`s), and `training_refs[]`. Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/material-variants/fluids-snow-and-solids-as-differentiable-simulations/`
- New code: `sim/material_diff.py`. Forward physics from `sim/material_showcase.py`; autodiff scaffolding
  from `sim/diffmpm.py` (+ `sim/material_variants.py`, treated as unverified).
- Suggested start: horizon **48–128 steps**, `n_particles≈1k–4k`, `n_grid=64–128`, f32. Per-material stable
  `dt`/`E` (fluid E~180–400; elastic dt smaller; snow E~150, `dt~5e-5`, and try a **softened** clamp). Start
  short and simple, lengthen only once the gradient check passes.

## Definition of done
- A **finite-difference gradient-check table for all three materials** (or an honest, explained account of
  which materials fail the check and why).
- Each material's forward **verified by eye** (no erroneous particle behavior — or the misbehavior diagnosed
  and reported).
- Every working material shows a **real descent** on a simple task with the control actually moving.
- Training page renders (KaTeX checked), reads standalone, **every `[[link]]` resolves**, embeds a viewed
  figure. Manifest complete schema-v2. Honest negatives clearly scoped.

## Known failures to avoid
- Do **not** just check "loss went down" — finite-difference-verify the gradient. That omission is what hid
  the prior bad gradients.
- **SVD degeneracy**: at $\sigma_1=\sigma_2$ (e.g. an undeformed blob, $F=I$) the SVD derivative diverges.
  Guard it (perturb, add epsilon, or avoid differentiating exactly at the isotropic point).
- **Hard snow clamp** is non-differentiable at the boundary; its gradient can be zero or garbage. Try a
  smooth clamp and report the difference.
- Stiff elastic/snow blow up if `dt` too large for `E` (CFL $\sim 1/\sqrt{E}$). Keep `dt` stable, horizon
  short; a blown-up rollout makes any gradient meaningless.
- The forward must be correct first — reuse the verified `material_showcase.py` laws; a wrong forward makes
  the gradient check meaningless.
- Headless only (no `ti.GUI`). Do not spawn a long background run then end the turn without viewing the
  outputs and confirming the files exist.
