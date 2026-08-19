# Worker brief: Propose new rendering for each of the four materials

## Effort tier: deep
**Persist.** The bar is not "four different-looking blobs" — it is four materials a viewer could tell
apart with the colours removed. Iterate on appearance, look at what you made, and keep going.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `propose-new-rendering-for-each-of-the-four-materials`. You are **NOT
the orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to
disk under `runs/material-variants/propose-new-rendering-for-each-of-the-four-materials/`, extend the
training textbook, and exit. **Do not commit** — the orchestrator reviews and commits your work.

## Notifications + live status
```
python harness/tools/notify.py --kind started  --task propose-new-rendering-for-each-of-the-four-materials "<one plain sentence>"
python harness/tools/notify.py --kind finished --task propose-new-rendering-for-each-of-the-four-materials "<one plain sentence>"
python harness/tools/task_status.py --direction material-variants --task propose-new-rendering-for-each-of-the-four-materials --step "<a few words>"
```

## Objective
**Propose** a distinct visual treatment for each of the four canonical materials — water, rubber (elastic),
snow, sand — so they are told apart by **how they look**, not only by their colour. This is the first task
in a sequence; the goal is a set of proposals good enough to choose between, not a shipped page.

## HARD SCOPE — do NOT touch the Demo page
The user was explicit: **this task must not update the demo.** Do not edit
`harness/dashboard/src/components/DemoView.jsx` or anything under `harness/dashboard/src/components/mpm/`.
Integration happens in a later task, after the appearance is agreed. Build your proposals in your **own
run directory** as standalone artifacts.

Also **do not change the physics.** A concurrently-planned task owns that. If a material's *behaviour*
bothers you, note it in `limitations` and move on — your job is how it is **drawn**.

## The problem, as reported
All four materials currently render identically and differ only in hue. The current treatment — mushy,
diffuse, blobby — is *right for snow* and wrong for everything else. The user's starting thoughts, to
treat as direction rather than specification:
1. **Elastic / rubber** — wants a **strong defining border** to separate it from its surroundings, and an
   interior that is **not mushy**. It should read as one coherent solid object, not a cloud of dots.
2. **Snow** — the current mushy, diffuse, textured look is roughly correct. **Keep it as the reference
   point** and make the others diverge from it.
3. **Water** — should be **smooth and clear**, fluid-like. Right now "it looks like a smoothie."
   **There is prior art in this repo to draw on**: the fluid-rendering lineage did real work on this —
   `non-differentiable-fluid-renderer`, `improve-basic-fluid-sim-realism`, `gpu-accelerate-fluid-renderer`
   and `more-realistic-basic-fluid-sims` under `runs/realistic-rendering/`, plus the training pages
   `core/fluid-rendering` and `core/fluid-color-mixing`. **Read those before inventing anything** — the
   screen-space smoothing / iso-surface work there is very likely the answer for water, and re-deriving it
   from scratch would be a waste.
4. **Sand** — should be **granular and fine-grained**: individual grains or macro-particles with random
   variation in grain size and irregularity, not a smooth field.

## What "proposal" means here — the deliverable
For **each** of the four materials, produce:
- A **rendered result** of the actual canonical material in motion (video preferred, since these are
  dynamic materials and a still hides the thing that matters), rendered your proposed way.
- The **same scene rendered the current way**, for comparison. **Both sides, same medium — the current
  look is the mandatory baseline.**
- A short statement of **what technique** produces it and **what it would cost** to run in the live demo
  (see the budget note below). A gorgeous treatment that cannot run in real time is still a useful
  proposal, but it must be labelled as such.

Offer **more than one option where you have one** — this is a task about choosing an appearance, and two
credible alternatives for water is more useful than one.

**The decisive test, and you should build it:** show all four materials **rendered in greyscale, colour
removed**. If a viewer cannot tell which is which, the proposal has not solved the stated problem. Include
this honestly even if it is unflattering.

## Where the pixels come from
Use the canonical physics (`sim.physics`, currently `phys-bebeaafbe73e`) as the source of motion — it is a
forward sim and needs no gradients. Render offline in Python if that gets you better results faster; you
are proposing an appearance, not shipping a shader. **But** if a treatment is only achievable offline, say
so plainly, because the destination is a real-time WebGPU page.

**Real-time budget, for the cost note:** the shipped demo runs four materials at ~1.00× real time with the
solver taking ~5.2 ms of a 16.7 ms frame at 16,384 particles, so roughly **10 ms per frame is available
for drawing** at that particle count. Use this to label each proposal *plausible* / *needs work* /
*offline-only*. A rough estimate with stated reasoning is fine; a measured number is better; **a
confident number you did not measure is not acceptable.**

## Evidence discipline (non-negotiable — see CLAUDE.md)
- **Appearance is a judgement call, and that is fine — but label it as one.** Do not dress a preference up
  as a measurement. Where you *can* measure (frame cost, particle counts, grain statistics), measure.
- Scope claims to the scenes you rendered. One dam-break does not establish "water reads as water".
- `hypothesis` and `limitations` required. Limitations should name any treatment that will not survive
  contact with the real-time budget, and anything that looked good only on one scene.

## Visualization standard (graded, not optional)
This task **is** visuals, so the standard is higher than usual, not lower:
- **Every proposal shown against the current rendering, same scene, same seed, as video.**
- The four materials shown **together in one scene** as well as individually — distinctness is a property
  of the set, not of one material alone.
- The greyscale test above.
- **Open every image and watch every video before writing a single finding.** A rendering task that ships
  a figure its author did not look at is a contradiction in terms.

## TL;DR (required manifest field)
One sentence, no jargon, including what did not work.

## Your task page (required — read `spec/style_task_page.md` in full)
This page's job is to let the user **choose**. Design it as a comparison surface: the reader should be able
to flip between current and proposed for each material, and ideally between competing options, and see the
difference happen. Ship as `custom_html` + standalone `bespoke_page.html`, self-contained (no CDNs, no
fetch, inline data/CSS/JS), media by absolute `/api/data/...` path. **Open the rendered page and click
every control before shipping.**

## Training textbook contribution (required)
At least one short, standalone page under `reports/training/` in the objective voice
(`spec/style_training_report.md`). The natural subject is **how particle data becomes an image** — why a
particle splat reads as mush, and the standard ways out (screen-space iso-surface extraction and
smoothing, per-grain sprites, silhouette/border extraction), and which material each suits.
**`core/fluid-rendering` already exists** — check it first and **extend it rather than duplicating it** if
that is where this belongs. Every `[[link]]` must resolve; write prerequisites before linking.

## Output contract
`runs/material-variants/propose-new-rendering-for-each-of-the-four-materials/manifest.json` (schema v2) +
media: `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]`, `training_refs[]`,
`physics_version`.
- **Two layers: `summary` (shown) + `full_report` (expander).** The summary should let the user pick a
  direction in ~15 seconds.
- **Write the manifest LAST**; every media `src` must resolve to a real file that exists.

## Paths & params
- Run dir: `runs/material-variants/propose-new-rendering-for-each-of-the-four-materials/`
- Prior art to read first: `runs/realistic-rendering/*`, `reports/training/core/fluid-rendering*`
- Physics: `sim/physics/` @ `phys-bebeaafbe73e` (read-only for this task)
- Canonical colours: `spec/registry/materials.json`

## Definition of done
- Four materials, each with a proposed treatment, **each shown against the current rendering as video**.
- The four shown together in one scene, and the **greyscale distinctness test** included and honestly read.
- A real-time cost note per proposal, labelled plausible / needs work / offline-only, with stated reasoning.
- **The Demo page and `sim/physics/` are untouched** (`git status` clean for both).
- **Finished within your turn** — do not end the turn waiting on a background render.
- Manifest carries scoped findings, honest `hypothesis` and `limitations`.
- Every figure/video opened and viewed; every media `src` resolves.
- Training page renders (KaTeX), reads standalone, every `[[link]]` resolves.

## Known failures to avoid
- **Do not re-derive the fluid rendering work.** Read `runs/realistic-rendering/` first; that lineage
  already solved a lot of the water problem.
- **Do not touch the demo or the physics.** Both are owned elsewhere right now.
- **Do not ship a figure you did not look at**, and do not ship a comparison without the current-look
  baseline beside it.
- Do not state a frame cost you did not measure or reason through explicitly.
- Colour is not a solution here — the greyscale test exists precisely to keep that honest.
