# Worker brief: Train material-replicating NNs (fluid, elastic, snow) and interpolate

> Direction: `material-variants`. Task id: `train-material-replicating-nns-and-interpolate`.
> Follow-up to `train-and-interpolate-nns-to-mimic-viscous-liquids`. That task learned a net per **viscosity**
> (a single linear knob) and found weight interpolation sags below the linear ideal. This task repeats the
> experiment across the three **constitutive models** — weakly-compressible fluid, corotated elastic, and
> Stomakhin snow — which are **structurally different laws, not one linear family**. So there is **no known
> function-space target** for interpolating a fluid-net with a snow-net: the interesting, open question is
> what actually emerges. **The user flagged this as long and detailed; work carefully and iteratively.**

## REWORK — the first run was sent back; fix these two things specifically
The user reviewed the first attempt and returned it. Address both head-on before anything else:
1. **Train on more than one gentle drop — especially exercise snow.** One drop barely engages snow's
   plasticity, so the snow net "hardly even acts like snow". Build **varied training data per material**
   (several initial configurations, and scenes that engage each material's characteristic behavior: for snow,
   compression / piling / an angle-of-repose slump / shearing that actually *fires the plastic clamp*; for
   elastic, squash-and-recover; for fluid, spreading). Train each net on that richer set so it captures the
   material, not one trajectory. Then **test generalization on several held-out configs**, not just one.
2. **Fix the interpolation endpoint bug.** In the first run, even at $\alpha=0.00$ — which is exactly the
   endpoint network that Question 1 showed *does* learn its material — the interpolated rollout "explodes or
   fails to mimic the GT". That is a **harness bug, not a property of interpolation**: at $\alpha=0$ and
   $\alpha=1$ the interpolated net *is* a trained endpoint net, so the rollout **must reproduce the pure
   material identically to the Question-1 replication rollout**. Make $\alpha=0$ and $\alpha=1$ reproduce the
   endpoints exactly (same scene, `dt`, initialization, and net-application path as the Q1 rollout) and debug
   why they currently diverge (most likely the interpolation rollout differs from the Q1 rollout in
   scene/`dt`/state-init/how the net is applied, or runs an unstable setting). **Only once the endpoints are
   correct does the interior sweep mean anything** — verify endpoint parity explicitly and report it.
Everything else in the brief below still applies.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `train-material-replicating-nns-and-interpolate`. You are **NOT the
orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to disk
under `runs/material-variants/train-material-replicating-nns-and-interpolate/`, add a training page, and
exit. **Do not commit.** Fire the two pings.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task train-material-replicating-nns-and-interpolate "<one plain sentence>"
python harness/tools/notify.py --kind finished --task train-material-replicating-nns-and-interpolate "<one plain sentence>"
```
`--kind blocked` on a hard stop. One human sentence, never a metrics dump.

## Objective
Train the **same small network architecture** to replicate each of the three materials (fluid, elastic,
snow) by replacing the constitutive part of the particle update with the net, verify each net reproduces its
material and **generalizes** to a new configuration, then **interpolate the weights** between materials and
characterize honestly what the interpolated dynamics do. Unlike the viscosity case there is no linear ideal,
so question three is exploratory: does interpolating a fluid-net and an elastic-net (or elastic and snow)
produce a plausible intermediate material, a broken one, or something else, and does it vary smoothly.

## The hard part: a common substrate so the weights are comparable
Weight interpolation only means something if all three nets share **one architecture, one input feature
layout, and one output**. The three materials natively carry different state (fluid tracks the scalar volume
ratio $J$; elastic and snow track the full deformation gradient $F$, and snow also an accumulated plastic
record). You must unify them:
- **Shared state & features.** Carry the full $F$ for all three materials (a fluid is the special case where
  only $\det F = J$ matters). Feed the net **position-free local features** common to all — e.g. the entries
  of $F$ and of the affine matrix $C_p$, and the velocity — exactly the discipline that made the viscosity
  and residual nets generalize instead of memorizing.
- **Shared output.** The cleanest single target is the **stress** the material writes into P2G (the "one
  slot" from `[[constitutive-models]]`): learn $g_\theta(F, C, v) \approx$ the true per-material stress.
  Train by supervised regression against each true simulator's stress (robust and cheap, as in the viscosity
  task); rollout training is optional and must not eat the budget.
- **Be explicit and honest about snow's plasticity.** Snow's plastic clamp is a **state update** (it mutates
  $F$'s singular values), not only a stress. Decide and state clearly how you handle it: either keep the
  analytic clamp and let the net learn only the stress (so the "snow net" is stress-only, plasticity shared),
  or fold the plastic effect into the learned map — pick one, justify it, and scope the interpolation claim
  to exactly what the net controls. Do not paper over this.

## Experiments / deliverables (three questions, in order)
1. **Replicate each material.** Drive the rollout with each trained net and show it reproduces its material
   on the training scene (a diagnostic + an overlay clip vs the true simulator). Report the fit and the gap.
2. **Generalize.** Run each net on a **new initial configuration / scene** it was not trained on and compare
   to the true material. Report the gap honestly (elastic/snow are harder than fluid; a partial result is
   fine if scoped).
3. **Interpolate the weights.** Sweep $\theta(\alpha) = (1-\alpha)\theta_A + \alpha\theta_B$ between material
   pairs (at least fluid↔elastic and elastic↔snow; fluid↔snow optional). For each interpolated net, run the
   rollout and **characterize the result**: measure interpretable diagnostics (e.g. how solid vs fluid it
   behaves — recovery/spread, shape retention), show clips, and say whether the transition is smooth,
   abrupt, plausible-intermediate, or degenerate. There is **no ground-truth intermediate**, so the finding
   is a careful description + hypothesis, not a pass/fail against an ideal. Connect to the viscosity task's
   result (compositional nonlinearity of the weight→behavior map) and note whether structurally different
   endpoints make interpolation worse.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope to the architecture, materials, scenes, and horizon tested. Three materials on one setup is a
  **demonstration**, not a law about learned constitutive models.
- Keep separate: replicates-its-material, generalizes, and interpolates-to-something-sensible. Report each.
- Interpolation has **no ground-truth target** here — do not invent one. Describe what emerges and hypothesize
  why. A messy or negative interpolation is a valid, expected-worth-reporting result.
- Manifest carries honest `hypothesis` and `limitations` (especially the plasticity-handling choice).

## Visualization standard (graded)
- Learned-vs-true overlay (clip + diagnostic) for each material and for the generalization test.
- The interpolation sweep: interpretable diagnostic(s) vs $\alpha$ for each material pair, plus clips along
  $\alpha$ so the morph (or its failure) is visible. Same-scale, labeled panels.
- **View every clip and plot before writing findings.** A learned rollout that blew up, froze, or scattered
  is a bug to diagnose, not a result. Regenerate anything degenerate.

## Training textbook contribution (required)
Add **one short, standalone** page (suggested `reports/training/core/13-learned-material-interpolation.md`,
id `learned-material-interpolation`) in the impersonal textbook voice: replacing the constitutive stress with
a net for three structurally different materials, the common-substrate trick that makes their weights
comparable, and what interpolating between structurally different dynamics does. Tie to
`[[learned-viscosity-interpolation]]` (the linear-knob precursor), `[[material-showcase]]`,
`[[differentiable-materials]]`, `[[constitutive-models]]`, `[[svd-polar]]`, `[[hybrid-learned-residual]]`
(all exist — every `[[link]]` must resolve). Embed a viewed figure; captions plain prose (no `$math$`).
Render-check the KaTeX. **Do NOT edit `reports/training/index.json`** — leave it untouched; the orchestrator
registers your page. In your final message, give the page **id, title, and filename**.

## Output contract
Write `runs/material-variants/train-material-replicating-nns-and-interpolate/manifest.json` (schema v2 — copy
the shape from `runs/material-variants/train-and-interpolate-nns-to-mimic-viscous-liquids/manifest.json`)
plus media, with `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]`
(replicate/generalize clips+plots, the interpolation sweep clips+plots, a table), and `training_refs[]`.
Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/material-variants/train-material-replicating-nns-and-interpolate/`
- New code: `sim/learned_materials.py`. Reuse the true constitutive physics from `sim/material_showcase.py`
  / `sim/material_diff.py` (import or copy — **do not mutate shared files in place**), following the
  learned-viscosity script `sim/learned_viscosity.py` as the pattern for the net + regression + interpolation
  harness.
- Same small architecture across all three materials; short horizons; modest particle counts. Keep per-
  material dt stable (snow/elastic need smaller dt than fluid — CFL) so no rollout blows up.

## Definition of done
- Three per-material nets (same architecture) each shown to replicate its material; a generalization test for
  each; and a **weight-interpolation study across at least two material pairs** with interpretable
  diagnostics and an honest characterization of what emerges.
- Every clip/plot **viewed**; nothing degenerate ships; the plasticity-handling choice is stated.
- Training page renders (KaTeX), reads standalone, **every `[[link]]` resolves**, embeds a viewed figure;
  `index.json` left untouched (report the page id/title/file). Manifest complete schema-v2.

## Known failures to avoid
- **Do not** let a fluid-only feature set silently exclude the solids — the shared substrate (carry $F$ for
  all, position-free features) is what makes the three nets comparable and the interpolation meaningful.
- A net that memorizes one trajectory fails generalization; test on a genuinely new config.
- Elastic/snow rollouts blow up if dt is too large for their stiffness (CFL ~ $1/\sqrt{E}$); keep each stable
  and watch the clips.
- Do not invent a ground-truth "intermediate material" for the interpolation — there is none; describe and
  hypothesize instead.
- Do not spawn a long training run then end the turn without viewing outputs and confirming files exist.
