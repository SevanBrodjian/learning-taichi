# Worker brief: Incorporate improved materials on the real Demo page, and improve polish  (T-027)

## Effort tier: deep
**Persist.** This is an integration task with three independent halves, and it is the first one that
changes what the user actually looks at every day. Run everything to completion inside your turn.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `incorporate-improved-materials-on-real-demo-page-and-improve-polish`.
You are **NOT the orchestrator**. Do not spawn further agents. Read this brief, do the task, write results
to `runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/`, extend
the training textbook, and exit. **Do not commit** — the orchestrator reviews and commits.

## Notifications + live status
```
python harness/tools/notify.py --kind started  --task incorporate-improved-materials-on-real-demo-page-and-improve-polish "<one plain sentence>"
python harness/tools/notify.py --kind finished --task incorporate-improved-materials-on-real-demo-page-and-improve-polish "<one plain sentence>"
python harness/tools/task_status.py --direction material-variants --task incorporate-improved-materials-on-real-demo-page-and-improve-polish --step "<a few words>"
```

## Objective
Bring the Demo page up to the current standard: the **new canonical physics** (T-021) and the **chosen
rendering treatments** (T-020), and make it **usable on a small screen**. Unlike the two tasks it
combines, this one **does** change `harness/dashboard/src/components/DemoView.jsx` and its `mpm/` bundle —
that is the point.

## Priority order — read this before planning your time
Three halves, and they are not equally risky. Do them in this order:

1. **Physics correctness.** The demo currently runs the OLD materials.
2. **Responsive layout.** Bounded, and it fixes a daily usability defect.
3. **Rendering treatments**, cheapest and most certain first: **snow → sand → water → rubber**.

The rendering is where time evaporates, and it degrades gracefully — shipping snow and sand correctly
with water still on the old renderer is real progress, and is far better than a half-finished water
shader that breaks the page. **If you run out of time, stop at a working page and say exactly what is
still on the old treatment.** Do not leave the demo broken; it is the flagship artifact.

## 1. Physics — the demo is running stale materials
Canonical physics moved to **`phys-c518316a4a05`** (T-021) and the demo's generated `params.js` still
stamps `phys-bebeaafbe73e`. It has **no `rho`, no `fric`, and one global `NU`**. Regenerating the file is
necessary but **not sufficient** — the WGSL has to actually use the new per-material quantities:

- **`rho` (per-material density).** Particle mass is `p_vol * rho`. This is what makes sand and rubber
  sink and snow float, and it must **emerge from the mass ratio** in the transfer — there is no buoyancy
  force in canonical physics and there must not be one here.
- **`nu` (per-material Poisson ratio).** Rubber is 0.45 now, everything else 0.20. It feeds the Lamé
  parameters, so it cannot stay a global constant.
- **`fric` (per-material boundary friction).** Water is 0, the rest 0.5. Canonical scatters a
  **mass-weighted friction to the grid** so a node shared by two materials gets the friction of whatever
  is sitting on it — mirror that, or interfaces will be wrong.
- The canonical **walls now separate with Coulomb friction** rather than zeroing both velocity
  components. Match it.

Regenerate with the run's own `web/gen_params.py` then `web/sync_to_dashboard.py` — **never retype a
constant into JS.** Stamp the new `physics_version`.

**Verify it, don't assume it:** a scene with all four materials should reproduce the canonical
sink/float ordering (snow floats, rubber and sand sink). If it does not, the density is not threaded
through properly.

## 2. Responsive layout — currently unusable on a phone
The page is fine on a large monitor and **crowded on an iPad or a small laptop, where the simulation is
not the centre of attention. On an iPhone it is effectively unusable.**

- **The canvas is the subject.** On a small screen the controls must shrink, collapse or move out of the
  way so the simulation dominates. The HUD readouts are secondary; the explanatory text under each
  control group is tertiary and can be hidden below a breakpoint.
- Test at least **iPhone (~390×844), iPad (~820×1180) and a small laptop (~1280×800)**, portrait and
  landscape. Use the browser tools to actually resize and look — do not infer from CSS.
- Touch targets need to be tappable; the pour/grab/remove interaction must still work by touch.
- **Do not regress the large-monitor layout**, which the user likes as it is.
- The canvas is sized to `devicePixelRatio`; keep it crisp, and watch that a phone's DPR does not blow
  up the particle count budget.

## 3. Rendering — the chosen treatments, and one that is NOT chosen
From T-020's proposals (`runs/material-variants/propose-new-rendering-for-each-of-the-four-materials/`,
implemented in `sim/material_render.py`). **Identify each option by its published description below, not
by a filename** — the internal keys are ambiguous and picking the wrong one is a silent failure.

- **Water — either new option.** The user is happy with either: **A "glass"** (background refraction,
  chromatic dispersion, deep Beer-Lambert absorption) or **B "film"** (same reconstruction, no background
  sampling, no dispersion, under half the absorption — reads as a thin lit liquid). B measured cheaper.
  **Pick one, say which, and say why.**
- **Snow — option A, "powder".** Same two passes as today minus the wet-plastic glint, with a soft powder
  fringe, brightened thin snow, darkened packed crevices and a fine crystal grain. Measured **0.307 ms
  against the current renderer's 0.309** — it is free.
- **Sand — option A, "grains over a packed body"** (not B, "loose grains"). Six irregular sprites per
  particle, each with its own hashed offset and radius, overlaps resolved by an atomic max on a random
  priority.
- **Rubber — NEITHER new option.** The user rejected both A ("border + printed grid") and B ("border,
  flat body"). **Keep the current renderer for rubber**, and make exactly two changes, **for rubber only**:
  **reduce how aggressively particles are smoothed into one continuous blob**, and give it **clearer
  borders**. This is a tuning change to the existing treatment, not a port of a new one.

**The costs above were measured in Taichi, not WGSL.** They say the treatments are affordable in
principle; they are not a WGSL measurement. Re-measure what you ship with `timestamp-query`, and keep the
whole frame inside budget: the solver takes ~5.2 ms of a 16.7 ms frame at 16,384 particles, so **roughly
10 ms is available for drawing**. Report the real frame cost of the shipped page.

## Hard constraints
- **Do not change `sim/physics/`.** Read-only at `phys-c518316a4a05`.
- **The TRANSPLANT CONTRACT holds.** `DemoView.jsx` and everything under `components/mpm/` import
  **nothing** from the harness — no `api.js`, no shared components, no app CSS. This page is going onto a
  personal site. Keep the standalone `web/demo.html` copy in sync via `sync_to_dashboard.py`.
- **Degrade gracefully without WebGPU**, and keep the three-case probe: "absent" usually means an
  **insecure origin**, not an unsupported device.
- `spec/aesthetic.md` at full strength — this is the flagship page.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Every timing is one GPU, one browser, one scene. Say so.
- **"Looks better" is a judgement; label it as one.** What gets measured is frame cost, particle budget,
  and the sink/float ordering.
- `hypothesis` and `limitations` required. Limitations must state exactly what, if anything, is still on
  the old treatment, and which screen sizes were actually tested rather than assumed.

## Visualization standard (graded)
- **Before/after, same scene, same seed, as video**, for the rendering change and for the physics change.
  The old page is the mandatory baseline.
- **Screenshots at each tested viewport** (phone / tablet / laptop / desktop), before and after, since the
  layout claim is a claim about what fits on a screen.
- The four-material scene showing the sink/float ordering, since that is the headline physics change.
- **Open every image and watch every clip before writing a finding.**

## TL;DR (required manifest field)
One sentence, no jargon, including what did not make it in.

## Your task page (required — `spec/style_task_page.md`)
`custom_html` + standalone `bespoke_page.html`, self-contained. The natural strong form is a
before/after flip per material and per viewport. Open the rendered page and click every control.

## Training textbook contribution (required)
At least one short standalone page (`spec/style_training_report.md`). The natural subject is **how a
particle field becomes a surface** — reconstruction (splat → mask → gradient → distance) as the step that
decides what a material looks like, and why the same solver output can read as water, powder or grains.
**`core/17-material-appearance.md` and `core/12-fluid-rendering.md` already exist — extend rather than
duplicate.** Every `[[link]]` must resolve.

## Output contract
`runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/manifest.json`
(schema v2) + media: `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]`,
`training_refs[]`, `physics_version` (the new one).
- **Two layers: `summary` (shown) + `full_report` (expander).**
- **Write the manifest LAST**; every media `src` must resolve.

## Definition of done
- The Demo page runs `phys-c518316a4a05` with per-material `rho`/`nu`/`fric`, and the sink/float ordering
  is demonstrated on the page's own solver.
- Snow, sand and water on their chosen treatments; rubber on the current one with reduced smoothing and
  clearer borders — or an explicit statement of which are still old.
- **Usable on a phone**, verified by actually resizing and looking, with the large-monitor layout intact.
- Transplant contract intact; `sim/physics/` untouched; standalone `web/demo.html` in sync.
- Finished within your turn; frame cost of the shipped page measured and reported.
- Manifest carries scoped findings, honest `hypothesis` and `limitations`; every media `src` resolves.
- Training page renders (KaTeX), reads standalone, every `[[link]]` resolves.

## Known failures to avoid
- **8 storage buffers per stage is the ceiling and the engine already uses 7.** Adding per-material
  buffers silently invalidates the bind group and drops every dispatch — a beautiful flat timing curve
  over all-zero data. Pack into existing buffers; assert non-zero motion before believing any timing.
- **`performance.now()` is clamped to 100 µs in Chromium** — use `timestamp-query`.
- **A wall clock nearly produced a wrong cost conclusion in T-020**: every screen-space treatment read
  ~3.3 ms host-timed, *identical at 360² and 1080²*. If your cost does not move with resolution, you are
  measuring the clock, not the GPU.
- Do not retype physics constants into JS; generate them.
- Do not leave the demo page broken. A partial, working page beats an ambitious broken one.
