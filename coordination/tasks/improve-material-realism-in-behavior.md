# Worker brief: Improve material realism in behavior

## Effort tier: deep
**Persist.** This changes frozen ground truth, so it is the one kind of task where "the first thing that
looked better" is not acceptable. Iterate, measure, and gate every change behind the golden signatures.
Long is fine.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `improve-material-realism-in-behavior`. You are **NOT the
orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to disk
under `runs/material-variants/improve-material-realism-in-behavior/`, extend the training textbook, and
exit. **Do not commit** — the orchestrator reviews and commits your work.

## Notifications + live status
```
python harness/tools/notify.py --kind started  --task improve-material-realism-in-behavior "<one plain sentence>"
python harness/tools/notify.py --kind finished --task improve-material-realism-in-behavior "<one plain sentence>"
python harness/tools/task_status.py --direction material-variants --task improve-material-realism-in-behavior --step "<a few words>"
```

## Objective
Make the four canonical materials behave more like the real things they are named after, and add
**per-material density** so they interact by buoyancy. **This task is explicitly allowed — and expected —
to change `sim/physics/`.** That is the point of it. Everything must pass through the promotion gates.

## The four complaints, as reported
These are the user's observations of the current canonical materials. Treat each as a symptom to
diagnose, not a prescription:
1. **Water is too mushy and sticky.** It should flow more smoothly and freely.
2. **Rubber (elastic) compresses far too much** — a blob can end up occupying noticeably less area than it
   started with, which reads as wrong. It also **breaks apart too easily**.
3. **Snow and sand are already good.** Do not "improve" them into something else. Changes that move snow
   or sand measurably need an explicit justification.
4. **Density is missing entirely.** Sand and rubber should **sink** in water; snow should **float**.

## Leads worth checking first (verify before believing)
Do not take these as answers — they are where to look.
- **There is no density parameter at all.** `p_rho = 1.0` (`sim/physics/core.py:47`) is a single global
  constant, so every material currently has identical density and nothing can sink or float. Adding
  buoyancy is therefore a **structural change**: `MAT` gains a per-material `rho`, per-particle mass
  follows from it, and buoyancy should then **emerge** from the mass ratio in the P2G/grid transfer rather
  than being added as a special force. If you find yourself writing an explicit buoyancy force, stop and
  reconsider — that is a sign the density is not actually threaded through the transfer.
- **`NU = 0.2` is a single global Poisson ratio** shared by every solid path (`core.py:51`). A Poisson
  ratio of 0.2 is quite compressible; real rubber is nearly incompressible (≈0.5). This is a strong
  candidate for complaint 2 — but raising it globally would also change snow and sand, so consider making
  `NU` **per-material** and moving only elastic. Note that stiffness and the stable timestep are coupled
  ($dt \sim 1/\sqrt{E}$), and $\lambda$ blows up as $\nu \to 0.5$, so check what your change does to
  stability before declaring victory.
- **"Breaks too easily"** is a different axis from compressibility — it is about behaviour at large
  deformation, not volume preservation. Diagnose which one you are actually fixing.
- **Water's "stickiness"** may be the viscous term (`mu_visc`), the weak-compressibility stiffness `E`, or
  the transfer itself. Measure before tuning.

## Non-negotiable: the promotion gates
`sim/physics/` is frozen ground truth. Changing it is allowed here, but only through
`sim/physics/PROMOTION.md`'s three gates. **Read that file first** — it also documents two traps in the
module (the deliberate `VERSION` line-ending normalisation, and the `Vt` naming, which is a gauge choice
and not a bug). Do not "fix" either.
1. **It is genuinely ground truth** — a property of the simulated world, not a task's harness.
2. **The golden signatures pass.** Every existing signature in `sim/physics/signatures.py` must stay
   green, and **new canonical behaviour adds a new signature asserting its qualitative truth.** At
   minimum this task should add: sand sinks in water, rubber sinks in water, snow floats, and whatever
   qualitative statement your water and rubber fixes are actually making (e.g. an elastic blob preserves
   its area to within some tolerance; water spreads further / retains less structure than before).
3. **The version bumps.** `sim.physics.VERSION` is a content hash; a promotion changes it. That is
   expected and correct here.

**A signature must assert a QUALITATIVE truth, not pin a number you just measured.** "Rubber's settled
area is within 15% of its initial area" is a signature. "Rubber's settled area is 0.0413" is a
regression test that will break for no reason and teach nobody anything.

## What a version bump costs downstream — state it, do not fix it
Bumping the physics invalidates artifacts generated from the old version, and the reviewer needs to know
which:
- The shipped Demo MVP generates `params.js` from `sim.physics` and stamps `phys-bebeaafbe73e`
  (`harness/dashboard/src/components/mpm/params.js`, generated by that run's `web/gen_params.py` and
  `web/sync_to_dashboard.py`). After your bump it is **stale**.
- **Do NOT regenerate the demo or touch the Demo page.** The user was explicit: this task must not be
  incorporated into the demo yet. Just **say clearly in your findings that the demo needs regeneration**,
  and name the two scripts that do it.
- Run `harness/tools/sync_registry.py` so `spec/registry/materials.json` matches the new physics — that
  file is generated, never hand-edited, and it must not be allowed to disagree with the code.
- Prior runs stamped the old version. That is what stamping is for; do not retro-edit them.

## Experiments / deliverables
1. **Diagnose before tuning.** For each complaint, measure the current behaviour and identify the
   parameter or term responsible. Report the diagnosis separately from the fix.
2. **Change the minimum that fixes it**, and show the before/after for each material.
3. **Buoyancy**: a scene with each solid material dropped into a pool of water, showing sand and rubber
   sinking and snow floating. This is the headline visual.
4. **Regression check on snow and sand**: show they did not move. If they did, justify it.
5. **Report what you could not fix**, and why.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope every claim to what was tested. "Water flows better on this dam-break scene" is not "water is
  realistic".
- **"More realistic" is a judgement, not a measurement.** Say what you measured (spread width, retained
  area, settled slope, sink depth) and keep the aesthetic claim clearly separate from the number.
- Keep **observed / hypothesised / would-test** separate. `hypothesis` and `limitations` are required.
- If a change improves one scene and degrades another, **say so** rather than picking the flattering one.

## Visualization standard (graded, not optional)
- **Every change is a comparison, so show BOTH SIDES: old physics vs new, same scene, same seed, as
  video.** The old behaviour is the baseline and it is mandatory. A "new water" clip alone is not evidence.
- The buoyancy scene must make the sink/float ordering unmissable — annotate the resting depth.
- For rubber, show the **area** the claim rests on (outline or measured area over time), not just a blob.
- Labeled axes, readable fonts.
- **Open every image and watch every video before writing a finding.** Regenerate anything degenerate or
  misleading.

## TL;DR (required manifest field)
One sentence, no jargon, including what failed.

## Your task page (required — read `spec/style_task_page.md`)
Design the page; ship as `custom_html` + standalone `bespoke_page.html`. Self-contained, no CDNs/fetch.
**Find the flip**: old-physics vs new-physics on the same scene, toggled by the reader, is the obvious
strong form here. Open the rendered page and click every control before shipping.

## Training textbook contribution (required)
At least one short, standalone page under `reports/training/` in the objective voice
(`spec/style_training_report.md`). The natural subject is **density and buoyancy in MPM** — why a mass
ratio threaded through the grid transfer produces floating and sinking without any explicit buoyancy
force, and what the Poisson ratio actually controls (volume preservation) versus what stiffness controls.
Check whether `core/04-constitutive-models.md` or `core/material-stiffness` already owns part of this and
extend rather than duplicate. Over-include prerequisites, write them before linking, and **every
`[[link]]` must resolve**.

## Output contract
`runs/material-variants/improve-material-realism-in-behavior/manifest.json` (schema v2) + media:
`objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]`, `training_refs[]`,
`physics_version` (the NEW one), and a `physics_version_before` recording what it was.
- **Two layers: `summary` (shown) + `full_report` (expander).**
- **Write the manifest LAST**; every media `src` must resolve to a real file.

## Paths & params
- Run dir: `runs/material-variants/improve-material-realism-in-behavior/`
- Physics: `sim/physics/core.py`, `signatures.py`, `PROMOTION.md`; registry via `harness/tools/sync_registry.py`
- Current version: `phys-bebeaafbe73e`

## Definition of done
- Water and rubber measurably improved on the stated complaints, with before/after video.
- **Per-material density added and buoyancy demonstrated** (sand + rubber sink, snow floats).
- Snow and sand shown not to have regressed.
- **Every existing golden signature green, plus new signatures for the new canonical behaviour.**
- `VERSION` bumped; `spec/registry/materials.json` regenerated; the demo's staleness stated but NOT fixed.
- **Finished within your turn** — do not end the turn waiting on a background job.
- Manifest carries scoped findings, honest `hypothesis` and `limitations`.
- Every figure/video opened and viewed; every media `src` resolves.
- Training page renders (KaTeX), reads standalone, every `[[link]]` resolves.

## Known failures to avoid
- **Do not fork the physics into the task.** Edit `sim/physics/` itself — that is what this task is for.
  A per-task copy of "better water" is the exact drift the project forbids.
- **Do not pin measured numbers as signatures.** Assert qualitative truths.
- **Do not touch the Demo page** or regenerate its params. Report the staleness instead.
- **Do not "fix" the `Vt` naming or the `VERSION` line-ending normalisation** — see PROMOTION.md.
- Watch the CFL timestep: raising stiffness or $\nu$ can silently destabilise a material. If you change
  `dt`, say so and show stability.
- Don't quote a metric whose implementation you have not read (`spec/registry/metrics.json`).
