# Worker brief: Train and interpolate NNs to mimic viscous liquids

> Direction: `material-variants`. Task id: `train-and-interpolate-nns-to-mimic-viscous-liquids`.
> Follow-up to `varying-liquid-viscosity`. The real research question: if a small network is trained to
> reproduce the particle update at each of a few viscosities, does **interpolating the trained weights**
> produce a liquid of **intermediate viscosity**, and does it vary **smoothly**? An honest negative (weight
> interpolation does not map cleanly to viscosity) is a valid, interesting result — report it either way.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `train-and-interpolate-nns-to-mimic-viscous-liquids`. You are **NOT the
orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to disk
under `runs/material-variants/train-and-interpolate-nns-to-mimic-viscous-liquids/`, add a training page, and
exit. **Do not commit.** Fire the two pings.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task train-and-interpolate-nns-to-mimic-viscous-liquids "<one plain sentence>"
python harness/tools/notify.py --kind finished --task train-and-interpolate-nns-to-mimic-viscous-liquids "<one plain sentence>"
```
`--kind blocked` on a hard stop. One human sentence, never a metrics dump.

## Objective
Take the viscous MPM fluid from `sim/fluid_viscosity.py` (the Newtonian strain-rate term
$\sigma_{\text{visc}} = \mu_{\text{visc}}(C+C^{\top})$). Pick **two or three** viscosity levels. For each,
train a small neural network (same architecture for all) that **replaces the particle update step, or a
well-defined portion of it**, so the learned step reproduces that viscosity's dynamics. Then answer three
questions, in order: (1) does each trained network reproduce its own viscosity; (2) does it **generalize**
to a new starting configuration; (3) when the weights of the per-viscosity networks are **interpolated**, is
the resulting liquid of intermediate viscosity, and does the effective viscosity vary smoothly with the
interpolation coefficient.

## Suggested design (tractable; improve on it, but keep it simple and get a real result)
- **Data.** Pick viscosities (e.g. a thin and a thick, or thin/medium/thick from the earlier sweep). Run the
  true forward sim per viscosity on one or two scenes and record, per step, the inputs and the target of the
  learned map.
- **What the NN replaces.** Replace a **portion** of the per-particle update with the network rather than the
  whole solver — the cleanest choice is the **viscous contribution**: learn a per-particle function
  $f_\theta(\text{local state}) \to$ (the velocity/affine correction that viscosity produces), from local,
  frame-invariant features (e.g. the particle's affine matrix $C_p$ / velocity gradient, its velocity, log
  mass — keep features **position-free** so the map is a local dynamics law, like the learned-residual task
  did). Same small MLP architecture for every viscosity so the weights are directly comparable for
  interpolation. Keep it small (tens–low hundreds of params) so interpolation is meaningful.
- **Training.** Supervised regression of the learned map against the true simulator's target is the robust
  primary approach (stable, cheap); training through the differentiable rollout is a fine alternative if you
  want trajectory-level fit, but do not let it eat the whole budget. Short horizons, modest particle counts.
- **Generalization.** Run each trained network as the update law on a **new initial configuration** (a
  different blob position/shape or a different scene) and compare the learned-fluid behavior to the true
  simulator at that viscosity (a spread/front diagnostic + a side-by-side clip). Report the gap honestly.
- **Weight interpolation.** Form $\theta(\alpha) = (1-\alpha)\,\theta_{\text{thin}} + \alpha\,\theta_{\text{thick}}$
  for a sweep of $\alpha \in [0,1]$, run the sim with each interpolated network, and **measure the effective
  viscosity** of the result via the same diagnostic (e.g. spread rate / front speed). Plot effective
  viscosity (or the diagnostic) vs $\alpha$. Is it monotonic? Smooth? Does it land between the endpoints? If
  it does not (e.g. interpolated weights give a broken or non-intermediate fluid), that is the finding —
  report and theorize why (weight space is not linear in behavior).

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope to exactly what was trained and tested: which viscosities, which architecture, which scenes, which
  horizon. Two or three viscosities is a **demonstration**, not a general law about learned dynamics.
- Keep separate: reproduces-its-own-viscosity, generalizes-to-new-config, and interpolates-smoothly — a
  network can pass one and fail another. Report each honestly. A negative interpolation result is welcome
  and must be clearly stated, not buried.
- Manifest carries honest `hypothesis` and `limitations`.

## Visualization standard (graded)
- Learned-vs-true comparison (clip + diagnostic) for each trained viscosity and for the generalization test.
- The **interpolation sweep**: effective-viscosity-vs-$\alpha$ plot, plus a few clips along $\alpha$ so the
  transition (or its failure) is visible.
- **View every clip and plot before writing findings.** A learned fluid that blew up, froze, or scattered is
  a bug to diagnose, not a result. Regenerate anything degenerate.

## Training textbook contribution (required)
Add **one short, standalone** page (suggested `reports/training/core/11-learned-viscosity-interpolation.md`,
id `learned-viscosity-interpolation`) in the impersonal textbook voice: learning a per-viscosity update law,
and what weight-space interpolation of dynamics networks does (or fails to do) to a physical parameter. Tie
to `[[hybrid-learned-residual]]`, `[[viscosity]]`, `[[differentiating-the-rollout]]`, `[[mpm-in-context]]`
(all exist — every `[[link]]` must resolve). Embed a viewed figure. Captions are plain prose (no `$math$`).
Render-check the KaTeX. **Do NOT edit `reports/training/index.json`** — leave it untouched; the orchestrator
will register your page (this avoids a concurrent-edit race with the other worker). In your final message,
tell the orchestrator the page **id, title, and filename** so it can add the index entry.

## Output contract
Write `runs/material-variants/train-and-interpolate-nns-to-mimic-viscous-liquids/manifest.json` (schema v2 —
copy the shape from `runs/material-variants/fluids-snow-and-solids-as-differentiable-simulations/manifest.json`)
plus media, with `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (comparison
clips/plots, the interpolation sweep, a table), and `training_refs[]`. Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/material-variants/train-and-interpolate-nns-to-mimic-viscous-liquids/`
- New code: `sim/learned_viscosity.py`. Reuse the viscous forward from `sim/fluid_viscosity.py`. **Do not
  edit shared files** (`sim/material_showcase.py`, `sim/fluid_viscosity.py` should be imported/copied from,
  not mutated in place if another use depends on them — prefer a fresh script).
- Keep networks small and identical across viscosities; short horizons; modest particle counts.

## Definition of done
- Two or three per-viscosity networks trained and each shown to reproduce its own viscosity; a generalization
  test to a new config; and a **weight-interpolation sweep with an effective-viscosity-vs-$\alpha$ result**
  (smooth-intermediate, or an honest account of why not).
- Every clip/plot **viewed**; nothing degenerate ships.
- Training page renders (KaTeX), reads standalone, **every `[[link]]` resolves**, embeds a viewed figure;
  `index.json` left untouched (report the page id/title/file). Manifest complete schema-v2.

## Known failures to avoid
- A network that ignores its input and memorizes one trajectory will fail generalization — use position-free
  local features and test on a new config, as in the learned-residual task.
- Interpolating weights of two networks that solved their tasks in **different internal coordinates** can
  give garbage even if each endpoint is fine; if so, that is the finding (and a reason to constrain/anchor
  the training). Do not fake a smooth interpolation.
- Keep the learned rollout stable (a learned update can blow up); watch the clips.
- Do not spawn a long background training run then end the turn without viewing the outputs and confirming
  the files exist.
