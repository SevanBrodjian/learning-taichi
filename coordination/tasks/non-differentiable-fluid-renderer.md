# Worker brief: Non-differentiable realistic fluid renderer

> Direction: `realistic-rendering`. Task id: `non-differentiable-fluid-renderer`.
> This is a **visual-quality** task, not a physics one. Under the hood run an MPM fluid sim; the deliverable
> is a **renderer** that turns the 2D particle fluid into frames that look like a real liquid — convincing
> enough to make someone double-take. It does **not** need to be fast, simple, or differentiable. The single
> graded question is: how good can you make it look. Aim high, iterate, and judge by looking.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `non-differentiable-fluid-renderer`. You are **NOT the orchestrator**. Do
not spawn further agents. Read this brief, do the task, write **all** results to disk under
`runs/realistic-rendering/non-differentiable-fluid-renderer/`, extend the training textbook, and exit. **Do
not commit** — the orchestrator reviews and commits. Fire the two pings.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task non-differentiable-fluid-renderer "<one plain sentence>"
python harness/tools/notify.py --kind finished --task non-differentiable-fluid-renderer "<one plain sentence>"
```
Use `--kind blocked` on a hard stop. One human sentence, never a metrics dump.

## Objective
Build a high-quality, **non-differentiable** renderer for the 2D MPM fluid. Take particle data from a
weakly-compressible MPM fluid sim and produce frames that read as a **real liquid** (a side-on view, like
looking at water in a glass tank), then show a couple of dynamic scenes. The bar is visual: it should look
convincingly real, not like a scatter of dots. Slowness is fine; spend the compute on looking good.

## Suggested rendering pipeline (improve on it — this is a starting point, not a spec)
The physics is the easy part; reuse the fluid forward from `sim/material_showcase.py` and **export
per-frame particle positions and velocities** to arrays. Then, for each frame, build a real image:
1. **Surface from particles (metaballs).** Splat particles into a high-resolution density field with smooth
   (e.g. Gaussian) kernels; smooth it; the liquid surface is a density isocontour. Compute a **surface
   normal** field from the density gradient. A clean, blobby surface (not individual dots) is the single
   biggest step toward realism.
2. **Shade the liquid body.** Combine several cues, each cheap and each adding realism:
   - **Absorption / depth color** (Beer-Lambert): tint by how much liquid the view ray passes through
     (thicker = deeper, more saturated color), so the body has volume, not a flat fill.
   - **Refraction of a background**: look up a background image/scene offset by the surface normal so the
     background visibly bends through the liquid — this is what most sells "real".
   - **Fresnel**: blend reflection vs refraction by the view/normal angle; add an environment/sky reflection
     that strengthens at grazing angles along the surface.
   - **Specular highlights** (Blinn-Phong) from a light, for bright glints on crests and the surface.
   - **Soft shadow / ambient occlusion** for depth where the liquid is thick or under overhangs.
3. **Foam and spray.** Where the local density is low or the motion is fast/turbulent (splash crowns,
   crests, thin sheets, isolated droplets), render bright **white foam / whitewater** and small droplet
   highlights. Real splashes are what make a drop read as water.
4. **Finish.** Anti-alias the surface edge, add a subtle **bloom** on highlights, a light vignette, and a
   pleasing background and floor. Composite and encode to mp4 at a decent resolution.
You are free to use a different or better approach (screen-space fluid, signed-distance surface, ray-marched
thickness, etc.). numpy for compositing, scipy for filtering if available (else a separable Gaussian in
numpy), imageio/ffmpeg for encoding; Taichi/GPU splatting is fine if you want the speed. Do whatever looks best.

## Scenes (at least two)
- **Fluid ball drop** — a blob of liquid falls into a shallow pool or onto the floor, throwing a splash /
  crown and droplets.
- **Fluid tower / dam-break collapse** — a tall column of liquid released sideways, producing a rolling wave
  and splashes as it hits the far wall.
Render each as an mp4 and save a **hero still**. Optionally add a small "how it's built" breakdown figure
(raw particles → density/surface → shaded) since it teaches well.

## The graded loop: render, LOOK, improve (this is the task)
This is explicitly a "how good can you make it" task, so **iterate on quality and judge by viewing your own
output** (the spec now requires viewing your figures anyway — here it *is* the deliverable). Render a frame,
open it, and critique it honestly: does it look like liquid or like dots? Is there a surface? Does the
background refract? Are there highlights and foam? Then improve the weakest thing and re-render. Keep a
couple of intermediate frames if they show the progression. Do **not** ship a plain matplotlib scatter or a
single flat blob and call it done — that fails the one thing this task is about.

## Evidence discipline (honest scope)
This is a rendering demo, not a claim about physical accuracy or real-time performance. Say plainly what the
renderer does and does not model (it is a stylized 2D side-view; no true 3D light transport, no measured
optical constants). If it does not reach photoreal, show how far it got and name the limitation — an honest
"here is the best achievable with this pipeline" is a fine outcome. No overclaiming "indistinguishable from
real".

## Training textbook contribution (required)
Add **one short, standalone** page (suggested `reports/training/core/10-fluid-rendering.md`, id
`fluid-rendering`) in the impersonal textbook voice (`spec/style_training_report.md`): how a cloud of MPM
particles becomes a believable liquid image — surface reconstruction from particles, depth/absorption color,
Fresnel reflection and refraction, specular highlights, foam. Tie to `[[mpm-in-context]]` (the particles
being rendered) and `[[material-showcase]]` (the fluid being simulated); both exist and every `[[link]]`
must resolve. Embed a viewed hero still (and the breakdown figure if you made one). Captions are plain prose
(no `$math$`). Render-check any KaTeX. Add the page to `reports/training/index.json` (core group).

## Output contract
Write `runs/realistic-rendering/non-differentiable-fluid-renderer/manifest.json` (schema v2 — copy the shape
from `runs/material-variants/implement-nondifferentiable-material-variants/manifest.json`) plus media, with
`objective`, `findings` (what the renderer does, honestly), `hypothesis`/`limitations` (or a rendering-notes
equivalent, honest about what is and isn't modeled), typed `results[]` (the scene `video`s, hero `image`s,
optional breakdown `image`), and `training_refs[]`. Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/realistic-rendering/non-differentiable-fluid-renderer/`
- New code: e.g. `sim/fluid_render.py` (the renderer) driving the fluid forward from
  `sim/material_showcase.py`. Render resolution can be generous (it does not need to be fast).

## Definition of done
- Two dynamic scenes (ball drop, tower/dam collapse) rendered to mp4 with a hero still each, at a visual
  quality that clearly reads as **liquid** — a real surface, shading with depth, refraction/reflection, and
  foam/spray — not a particle scatter.
- You **viewed and iterated** on the renders; the shipped frames are the best the pipeline reached.
- Training page renders (KaTeX if any), reads standalone, **every `[[link]]` resolves**, embeds a viewed
  figure. Manifest complete schema-v2.

## Known failures to avoid
- **Do not ship a bare particle scatter or a flat single-color blob.** The entire task is visual quality;
  a low-effort render fails it. If you are unsure, look at your frame and ask whether it could be mistaken
  for real liquid.
- Keep the underlying sim stable (a blown-up fluid renders as noise). Headless only (no `ti.GUI`).
- Non-differentiable is fine and expected; do not spend effort making it differentiable.
- Do not spawn a long background render then end the turn without viewing the outputs and confirming files.
