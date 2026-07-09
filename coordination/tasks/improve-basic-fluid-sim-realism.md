# Worker brief: Improve basic fluid sim realism (behave like water, no holes, push to photoreal)

> Direction: `realistic-rendering`. Task id: `improve-basic-fluid-sim-realism`.
> Follow-up to `non-differentiable-fluid-renderer`. The renderer looks decent but the user flagged three
> concrete problems, in their words: the liquid **does not behave like water** (moves a bit too slowly, the
> behavior is off, hard to pin down), the render has **holes inside continuous portions of the liquid**
> (looks wrong), and the ask is to **push realism further, toward photorealistic**. Fix all three.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `improve-basic-fluid-sim-realism`. You are **NOT the orchestrator**. Do
not spawn further agents. Read this brief, do the task, write **all** results to disk under
`runs/realistic-rendering/improve-basic-fluid-sim-realism/`, add/extend a training page, and exit. **Do not
commit.** Fire the two pings.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task improve-basic-fluid-sim-realism "<one plain sentence>"
python harness/tools/notify.py --kind finished --task improve-basic-fluid-sim-realism "<one plain sentence>"
```
`--kind blocked` on a hard stop. One human sentence, never a metrics dump.

## Objective
Push the fluid demo from "reads as liquid" to as close to **photorealistic** as this 2D pipeline can get, by
fixing the two concrete defects the user named and then improving the weakest realism cues. Start from
`sim/fluid_render.py` (the existing renderer + its fluid). Diagnose by **looking**, fix, and re-render.

## The three problems, and how to attack them
1. **"Does not behave like water — moves too slowly, behavior is off."** This is a **simulation/dynamics**
   problem, not a rendering one. Weakly-compressible MPM water often looks sluggish and gloopy. Likely
   levers: the fluid is too soft/compressible (low $E$ makes it squishy — a stiffer, more nearly
   incompressible fluid moves more like water, at the cost of a smaller `dt`); too much numerical damping
   (heavily PIC transfers are dissipative — more APIC/FLIP energy keeps it lively and splashy); the
   gravity-to-domain scale and the sim-time-to-playback-time mapping (if a clip spans little physical time
   or plays back slow, water looks like syrup). Tune these and **watch the clip** until the motion reads as
   real water: fast, lively, splashy, sheeting and breaking rather than oozing. Report what you changed and
   why.
2. **"Holes inside continuous portions of the liquid."** This is a **surface-reconstruction** artifact. The
   metaball density isocontour is almost certainly thresholding out low-density pockets *inside* the body
   (particle spacing dips below the isovalue), so the interior shows air holes. Fixes to consider: raise the
   splat kernel width or particle count so the interior density is solidly above the isovalue; separate
   "**is there liquid here**" (a filled interior mask, e.g. flood-fill inside the outer surface or a lower
   interior threshold / a signed-distance fill) from "**where is the surface**" (used for normals/shading),
   so interior pockets never punch through; make sure the foam/thin-detection is not carving holes in calm
   interior. The body should read as a continuous filled volume with structure only where there really is
   air (splash cavities, the barrel tube), not random interior holes.
3. **"Push realism further — photorealistic."** Improve the weakest cues the prior pass named: refraction is
   only visible at curved features because the flat-slab normal is near-uniform (consider a real
   screen-space **thickness** accumulated for absorption/refraction, a stronger/structured background that
   rewards refraction, subtle surface ripple detail); add depth/quality where it helps (caustic-like bright
   bands on the floor, better foam texture, higher render resolution, softer shadows, chromatic edges). Every
   addition is judged by eye.

## The graded loop: render, LOOK, improve (this IS the task)
Iterate explicitly and **judge by viewing your own frames**. After each change, open the frame and ask: does
the water *move* like water now? Are the interior holes gone? Is it more convincing than before? Keep going
until the ball drop and the collapse look markedly better than the previous version. Do **not** ship a
change you have not looked at.

## Deliverables
- Improved **ball drop** and **dam-break / tower collapse** (or better scenes) as mp4 + hero stills, visibly
  better than the prior renderer on all three counts.
- A **before/after** comparison (old frame vs new frame for the same scene) so the improvement is legible —
  this is strong evidence and easy to make.
- Optionally a short breakdown of the interior-fill fix (holey vs filled) since it teaches well.

## Evidence discipline (honest scope)
Still a stylized 2D side-view render, not a physical light-transport simulation. Say plainly what improved
and what remains stylized/hand-tuned. If it still stops short of true photoreal, show how far it got and name
the remaining gap. No overclaiming "indistinguishable from real"; an honest "closer, and here is what still
holds it back" is the right register.

## Visualization standard (graded)
- Same scene before/after for a fair comparison; readable, high-resolution frames.
- **View every video and still before writing findings**; the shipped frames are the best the pipeline
  reached, with the holes gone and the motion water-like. Regenerate anything degenerate.

## Training textbook contribution (required)
Extend the story from `[[fluid-rendering]]` with **one short, standalone** page (suggested
`reports/training/core/12-fluid-realism.md`, id `fluid-realism`) in the impersonal textbook voice: why
weakly-compressible MPM water can look sluggish and how stiffness / transfer damping / time-scale fix it, and
why a naive metaball isocontour punches interior holes and how a filled interior mask fixes it. Tie to
`[[fluid-rendering]]`, `[[material-showcase]]`, `[[material-stiffness]]`, `[[viscosity]]` (all exist — every
`[[link]]` must resolve). Embed a viewed before/after figure. Captions are plain prose (no `$math$`).
Render-check any KaTeX. **Do NOT edit `reports/training/index.json`** — leave it untouched; the orchestrator
will register your page (this avoids a concurrent-edit race with the other worker). In your final message,
tell the orchestrator the page **id, title, and filename**.

## Output contract
Write `runs/realistic-rendering/improve-basic-fluid-sim-realism/manifest.json` (schema v2 — copy the shape
from `runs/realistic-rendering/non-differentiable-fluid-renderer/manifest.json`) plus media, with
`objective`, `findings` (what changed and why, honestly), `hypothesis`/`limitations` (honest about what is
and isn't modeled), typed `results[]` (scene videos, hero images, the before/after), and `training_refs[]`.
Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/realistic-rendering/improve-basic-fluid-sim-realism/`
- Code: iterate on `sim/fluid_render.py` (you may refactor it or add a new module). **Do not edit shared
  physics files** (`sim/material_showcase.py`) in place — tune the fluid within the renderer's own sim setup
  or a fresh script, so other tasks that import the showcase fluid are unaffected.

## Definition of done
- Ball drop and collapse rendered with **water-like motion** (lively/splashy, not sluggish), **no interior
  holes**, and **visibly higher realism** than the prior renderer, with a before/after that shows it.
- You **viewed and iterated**; shipped frames are the best reached.
- Training page renders (KaTeX if any), reads standalone, **every `[[link]]` resolves**, embeds a viewed
  before/after; `index.json` left untouched (report the page id/title/file). Manifest complete schema-v2.

## Known failures to avoid
- Do not "fix" the slowness only by speeding up playback if the underlying motion is still gloopy — fix the
  dynamics (stiffness / damping / scale), then confirm by eye.
- A stiffer, more incompressible fluid needs a smaller `dt` (CFL $\sim 1/\sqrt{E}$); keep it stable or it
  blows up and renders as noise.
- Do not re-introduce interior holes with an over-aggressive isovalue or foam mask; verify a calm body is a
  filled body.
- Headless only (no `ti.GUI`). Non-differentiable is fine. Do not spawn a long render then end the turn
  without viewing the outputs.
