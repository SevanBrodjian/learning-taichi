# Worker brief: More, longer, more diverse realistic fluid sims (incl. color mixing)

> Direction: `realistic-rendering`. Task id: `more-realistic-basic-fluid-sims`.
> Follow-up to `improve-basic-fluid-sim-realism` (which produced the improved renderer `sim/fluid_render2.py`).
> The user's words: "These results look closer to the photorealistic target, but they are very short and
> limited. Generate a more diverse and extensive set of examples showcasing the capabilities, such as more
> starting conditions, different viscosities, even things like mixing liquids of different colors. Also,
> make the animations much longer, these ones are WAY too short." Deliver breadth, duration, and color mixing.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `more-realistic-basic-fluid-sims`. You are **NOT the orchestrator**. Do
not spawn further agents. Read this brief, do the task, write **all** results to disk under
`runs/realistic-rendering/more-realistic-basic-fluid-sims/`, add a training page, and exit. **Do not
commit.** Fire the two pings.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task more-realistic-basic-fluid-sims "<one plain sentence>"
python harness/tools/notify.py --kind finished --task more-realistic-basic-fluid-sims "<one plain sentence>"
```
`--kind blocked` on a hard stop. One human sentence, never a metrics dump.

## Objective
Using the improved renderer from `sim/fluid_render2.py` as the base, produce a **diverse, extensive, and much
longer** set of realistic fluid animations that show off the pipeline: several starting conditions, a couple
of viscosities, and at least one **color-mixing** scene where two differently colored liquids swirl together.
Length is a hard requirement — the previous clips were far too short; these must run much longer so the
dynamics fully play out.

## What to build (breadth, duration, color)
- **Reuse the improved renderer** in `sim/fluid_render2.py` (its filled-interior mask, distance-transform
  thickness, Beer-Lambert depth color, refraction/Fresnel/specular, foam). Import and reuse its pipeline;
  put new drivers/scenes in a fresh module (e.g. `sim/fluid_showcase_render.py`) rather than mutating the
  committed renderer in place.
- **Much longer animations.** The prior clips were ~2 s. Make each new clip **substantially longer** (aim for
  ~10-25 s of playback so a splash rises, falls, sloshes, and settles, or two liquids fully intermingle).
  Watch for slow energy drift over the long rollout (a weakly-compressible fluid can slowly gain/lose energy
  over thousands of steps); keep it stable and physical for the whole duration.
- **More starting conditions (several distinct scenes).** For example: a single ball drop, a double or
  offset multi-blob drop, a pouring stream, a dam-break, a fluid filling/overflowing, or a blob hitting an
  obstacle. Pick a handful that look good and are distinct.
- **Different viscosities.** Bring in the Newtonian viscous term from `sim/fluid_viscosity.py`
  ($\sigma_{\text{visc}} = \mu_{\text{visc}}(C+C^{\top})$) so at least one scene is a thin water-like liquid
  and one a thick honey-like liquid, both rendered with the realistic pipeline (thick pours/coils slowly,
  thin splashes).
- **Color mixing (the standout).** Give particles a **per-particle color / dye** carried along with the flow
  (advected by the sim, unchanged by the physics), initialize two or more regions with different colors, and
  extend the renderer to composite the **local particle color** into the body shading (splat per-channel
  color into fields, normalize by density, and drive the Beer-Lambert body tint by that local color instead
  of a single fixed water color). Show two colored liquids released together swirling and blending into
  intermediate colors where they mix. This is the most novel piece — make it read clearly.

## Evidence discipline (honest scope)
Still a stylized 2D side-view render, not physical light transport, graded on how it looks. The dye is a
passive color advection, not a real diffusion/mixing chemistry; say so. Keep the honest register from the
prior pages: what looks convincing and what remains stylized/hand-tuned. No overclaiming photoreal.

## Visualization standard (graded)
- A varied set of **long** mp4s (several scenes) plus a hero still each, at good resolution. The color-mixing
  clip should clearly show two colors becoming a blended one where they meet.
- **View every clip and still before writing findings** (sample frames across each long clip, since problems
  can appear late in a long rollout — energy drift, a slow blow-up, colors washing out). A degenerate or
  blown-up late frame is a bug to fix, not to ship. Regenerate anything wrong.

## Training textbook contribution (required)
Add **one short, standalone** page (suggested `reports/training/core/14-fluid-color-mixing.md`, id
`fluid-color-mixing`) in the impersonal textbook voice: how a passive per-particle color is advected by the
MPM transfers and composited in the renderer to show two liquids mixing, and a note on keeping a long
weakly-compressible rollout stable. Tie to `[[fluid-realism]]`, `[[fluid-rendering]]`, `[[viscosity]]`,
`[[mpm-in-context]]`, `[[material-showcase]]` (all exist — every `[[link]]` must resolve). Embed a viewed
still from the color-mixing scene; captions plain prose (no `$math$`). Render-check any KaTeX. **Do NOT edit
`reports/training/index.json`** — leave it untouched; the orchestrator registers your page. In your final
message, give the page **id, title, and filename**.

## Output contract
Write `runs/realistic-rendering/more-realistic-basic-fluid-sims/manifest.json` (schema v2 — copy the shape
from `runs/realistic-rendering/improve-basic-fluid-sim-realism/manifest.json`) plus media, with `objective`,
`findings` (the scenes, the color-mixing method, honest notes), `hypothesis`/`limitations`, typed `results[]`
(the several long scene videos, hero stills, the color-mixing clip), and `training_refs[]`. Leave everything
on disk; do not commit.

## Paths & params
- Run dir: `runs/realistic-rendering/more-realistic-basic-fluid-sims/`
- Code: a fresh driver (e.g. `sim/fluid_showcase_render.py`) reusing `sim/fluid_render2.py` and the viscous
  term from `sim/fluid_viscosity.py`; **do not mutate shared files in place** (`sim/material_showcase.py`,
  `sim/fluid_render2.py`, `sim/fluid_viscosity.py` — import/copy from them).
- Keep the fluid stable over long rollouts (stiffer fluid needs a small dt; check finiteness throughout).
  Render resolution generous; it does not need to be fast.

## Definition of done
- Several **distinct, much-longer** realistic fluid animations (varied starting conditions), including at
  least one thin and one thick viscosity and at least one **color-mixing** scene that clearly shows blending.
- Every clip/still **viewed** across its full length; nothing degenerate or blown-up ships.
- Training page renders (KaTeX if any), reads standalone, **every `[[link]]` resolves**, embeds a viewed
  still; `index.json` left untouched (report the page id/title/file). Manifest complete schema-v2.

## Known failures to avoid
- **Too short.** The whole point is longer, richer clips — do not ship ~2 s animations again.
- A long weakly-compressible rollout can slowly drift or blow up; sample late frames and keep it stable.
- Color washing out to grey everywhere (over-mixing/over-blur) or not mixing at all (colors stay hard-edged)
  both read wrong — tune so the mix is visible but the source colors remain legible.
- Headless only (no `ti.GUI`); non-differentiable is fine. Do not spawn a long render then end the turn
  without viewing outputs and confirming files.
