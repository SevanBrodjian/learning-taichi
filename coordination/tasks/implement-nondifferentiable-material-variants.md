# Worker brief: Implement nondifferentiable material variants (forward demo of solid / liquid / snow)

> Direction: `material-variants`. Task id: `implement-nondifferentiable-material-variants`.
> This is a **learning / grounding** task, not a claim-making experiment. The point is a clean forward
> simulation that shows how an elastic solid, a weakly-compressible fluid, and snow (elastoplastic)
> actually move and differ, under a few settings, verified to behave as expected, and taught in the
> textbook. Keep it simple and make the visuals excellent.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `implement-nondifferentiable-material-variants`. You are **NOT the
orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to disk
under `runs/material-variants/implement-nondifferentiable-material-variants/`, extend the training textbook,
and exit. **Do not commit** — the orchestrator reviews and commits your work. Fire the two pings below.

## Notifications (exactly two)
At the start:
```
python harness/tools/notify.py --kind started --task implement-nondifferentiable-material-variants "<one plain sentence: what you're starting>"
```
When your results are on disk:
```
python harness/tools/notify.py --kind finished --task implement-nondifferentiable-material-variants "<one plain sentence: what's ready to review>"
```
Use `--kind blocked` instead of `finished` if you hit a hard stop. One sentence of human status, never a
metrics dump.

## Objective
Build a **forward-only (nondifferentiable)** MLS-MPM simulation that runs the same scene under three
constitutive models — **fluid** (weakly-compressible pressure from $J$), **elastic** (corotated stress from
$F$ via `ti.svd`), and **snow** (elastic + plastic clamp of $F$'s singular values + hardening) — and shows,
side by side, how each behaves. Confirm each material does the qualitatively expected thing, and teach the
differences. **No gradients, no autodiff tape, no optimization** anywhere in this task.

## Experiments / deliverables
- **Reuse the constitutive physics already written** in `sim/material_variants.py`: `fluid_stress`,
  `corotated_PFt` / `elastic_stress`, `snow_stress`, and the snow SVD-clamp in `g2p_snow`. You do **not**
  need the loss / `x_avg` / autodiff machinery from that file — strip it. Prefer writing a small, clean
  **forward-only** script `sim/material_showcase.py` (in the spirit of the pristine `sim/mpm88.py` forward
  seed) that shares one p2g / grid_op / g2p skeleton and switches the stress branch by material. Render
  **headless** — no `ti.GUI` loop; export frames and encode to mp4 (imageio/ffmpeg), so it runs on an
  agent with no display.
- **Scenes (pick 2, optionally 3) that make the differences obvious.** Same initial condition across all
  three materials so the comparison is fair:
  1. **Drop & splat** — a blob released above the floor falls under gravity and hits it. Expected: fluid
     spreads into a flat puddle; elastic squashes then springs back and holds a rounded, jiggling shape;
     snow crumples and packs into a static heap that keeps its dented shape.
  2. **Column / block collapse** — a tall block released from rest slumps. Expected: fluid runs out flat;
     elastic wobbles and largely recovers height; snow collapses to an angle-of-repose pile that holds.
  3. *(optional, ties to `material-stiffness`)* one material under a **parameter dial** — e.g. elastic at
     soft vs stiff $E$, or snow at brittle vs ductile $\theta_c$ — to show the knob's visible effect.
- For each scene render a **labeled side-by-side triptych** video (fluid | elastic | snow), and save one
  representative **still frame** (PNG) per scene for the manifest and the training page.
- **Verify behavior, and quantify at least one simple diagnostic** so "behaves as expected" is not just an
  assertion — e.g. final horizontal spread (width) or final pile height per material, tabulated. Fluid
  should spread most / end lowest; snow should hold shape / stay tallest; elastic in between and springy.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- This is a **forward demonstration on a few fixed scenes at fixed resolution and hand-set parameters**.
  Scope every statement to that. It shows the three models produce their expected qualitative signatures
  here; it is **not** a claim about optimization, gradients, generality, or quantitative accuracy.
- The manifest still carries an honest `hypothesis` (why each model moves the way it does, grounded in its
  stress law) and `limitations` (2D, fixed grid, qualitative, tuned for stability).

## Visualization standard (this is graded, not optional)
- **Same initial condition across the three materials** in each triptych, clearly labeled fluid / elastic /
  snow, so the only variable is the constitutive model. Readable labels and a visible floor/domain.
- Prefer clear, simple, informative demos. A particle render tinted by material, or a light mass/velocity
  cue, beats a raw dot cloud. Keep it legible on iPad.
- **View everything you export before writing a word of findings.** Read every saved frame/PNG back and
  watch every mp4. Confirm each material actually shows its expected signature and that **nothing blew up,
  went NaN, flew off-screen, or clipped through the floor**. A material that exploded is a bug to fix (see
  Known failures), not a video to ship. Regenerate anything degenerate.

## Training textbook contribution (required)
Add **one short, standalone** core page (suggested id `material-showcase`, e.g.
`reports/training/core/07-material-showcase.md`) in the objective textbook voice
(`spec/style_training_report.md`): impersonal, readable cold, no reference to "this task/run". Teach the
**qualitative behavior** of the three materials and *why* each moves as it does, tying the stress laws to
what the eye sees. Link prerequisites that already exist and must resolve: `[[constitutive-models]]`
(the three stress laws), `[[svd-polar]]` (the snow singular-value clamp), `[[material-stiffness]]` (the $E$
dial), `[[mpm-in-context]]`. Embed a triptych still or short clip. This is a forward page, so **no gradient
or optimization content** — keep it to the physics and the picture. Render-check the KaTeX and **view the
embedded figure**. Captions are plain prose (they render as visible captions — no `$math$` inside them).

## Output contract
Write `runs/material-variants/implement-nondifferentiable-material-variants/manifest.json` (schema v2 — see
`runs/README.md`) plus media, with: `objective`, scoped `findings`, `hypothesis`, `limitations`, typed
`results[]` (the triptych `video`s, still `image`s, and a small `table` of the diagnostic), and
`training_refs[]` pointing at the page you added. Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/material-variants/implement-nondifferentiable-material-variants/`
- New code: `sim/material_showcase.py` (forward-only). Reuse physics from `sim/material_variants.py`; seed
  idioms from `sim/mpm88.py`.
- Suggested params: `n_grid=128`, `n_particles≈8k–16k`, `E=400` baseline, fluid `dt=2e-4`. **Snow and
  elastic may need a smaller `dt` (or lower `E`) for stability** — the explicit CFL limit scales like
  $1/\sqrt{E}$ (see `material-stiffness`). Pick per-material stable settings and record them.

## Definition of done
- A labeled fluid|elastic|snow triptych for each chosen scene, same init, in which the three materials are
  **visibly, correctly distinct**, plus the diagnostic table backing "fluid spreads most / snow holds shape".
- **Every exported frame and video has been viewed**; nothing degenerate ships (no NaN/blowup/clip-through).
- Forward-only: no autodiff tape, no optimizer, no loss.
- Training page renders cleanly (KaTeX checked), reads standalone in the textbook voice, **every `[[link]]`
  resolves**, and embeds a viewed figure. Manifest is complete schema-v2.

## Known failures to avoid
- **Instability / NaN**: snow's SVD clamp and stiff elastic stress can blow up if `dt` is too large for `E`
  or if a singular value hits zero. Use a stable `dt`/`E`, guard the SVD (avoid div-by-zero), and **watch
  the video** — a material that flew apart is not "expected behavior". Reduce `dt` or `E` and note it.
- Do **not** run an interactive `ti.GUI` loop (headless agent, no display). Export frames → mp4.
- Do **not** build a differentiable version or optimize anything — this task is forward only.
- Keep blobs inside the domain (respect `bound`); a blob clipping through the wall is a setup bug.
- Do not spawn a long background render then end the turn assuming it finished — confirm the files exist and
  view them before firing the `finished` ping.
