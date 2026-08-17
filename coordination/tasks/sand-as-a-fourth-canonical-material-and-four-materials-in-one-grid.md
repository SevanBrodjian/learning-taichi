# Worker brief: Sand as a fourth canonical material, and four materials in one grid

> Filled contract for a cold worker. The queued task's `note` was only the seed. Storage direction:
> `material-variants`. Read `coordination/demo-mvp.md` first — this task exists to serve that target.

## Effort tier: deep
This task's intensity, set on the dashboard, drives how this worker was spawned (model + reasoning effort
+ how long to persist). The orchestrator fills this in and matches the spawn to it (see the `/execute`
skill). As the worker, treat it as the expected depth:
- **quick** — a light, cheap, bounded task. One clean pass, minimal sweeps; do not over-engineer.
- **standard** — the default. Normal depth and iteration.
- **deep** — a genuinely hard task. Persist: iterate, debug, run the sweeps it takes, and do not stop at
  the first plausible result. Long is fine; a shallow answer is not.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid`. You are **NOT the orchestrator**. Do not spawn further
agents. Read this brief, do the task, write **all** results to disk under
`runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid/`, extend the training textbook, and exit. **Do not commit** — the
orchestrator reviews and commits your work. Fire the pings below.

## Notifications (exactly two) + live status
At the start:
```
python harness/tools/notify.py --kind started --task sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid "<one plain sentence: what you're starting>"
```
When your results are on disk:
```
python harness/tools/notify.py --kind finished --task sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid "<one plain sentence: what's ready to review>"
```
Use `--kind blocked` instead of `finished` if you hit a hard stop. **One sentence of human status, never a
metrics dump or a technical report.**

**Live status (a few times, not spammed).** So the dashboard shows what step this Active task is on, call
this at each coarse milestone (starting, the main phase, rendering, wrapping up) — think 3–6 times over the
run, one short phrase each:
```
python harness/tools/task_status.py --direction material-variants --task sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid --step "<a few words: current step>"
```
It writes `runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid/status.json` (ephemeral, gitignored). Use `--state blocked` if you
stall on something the user must resolve.

## Objective
Add **sand** as a fourth canonical material in `sim/physics/`, measure the timestep it forces, and show
all four materials coexisting in a single grid for the first time. This unblocks the Demo MVP
(`coordination/demo-mvp.md`), which needs four materials in one simulation.

## Experiments / deliverables
**You have exclusive use of the GPU.** The WebGPU port task is deliberately held until you finish, so your
Taichi timings are not contended. Do not leave background GPU jobs running when you exit.

**1. Sand is a PROMOTION EVENT, not a task-local material.** `sim/physics/` freezes exactly fluid, elastic
and snow, and forking it is the defect this project most consistently punishes. Adding sand goes through
`sim/physics/PROMOTION.md`: it is ground truth, the golden signatures pass, the version bumps.
- Pick a granular constitutive model and **justify the choice** — Drucker-Prager style plasticity is the
  usual answer for MPM sand, but say why you chose what you chose.
- **Sand needs its own golden signature** in `sim/physics/signatures.py`: it should **pile and hold an
  angle of repose**, and must NOT spread flat like a fluid. Make the assertion qualitative and robust, in
  the style of the existing signatures — not a brittle exact number.
- **Every existing signature must stay green.** If adding sand perturbs fluid, elastic or snow at all,
  stop and fix that first; a silent change to a frozen material is a hard reject.

**2. Report sand's stable `dt` as a FIRST-CLASS RESULT.** This is not a footnote — it may set the frame
budget for the whole Demo. A shared grid means ONE timestep, so the stiffest material present dictates the
cost of the entire scene. For context you must not just repeat but check:
- fluid dt=1.2e-4 -> 139 substeps/frame; elastic dt=1.0e-4 -> 167; snow dt=5.0e-5 -> **333**.
- Snow's small dt is **not** explained by its E (150, the lowest of the three). It comes from hardening
  `h = exp(xi*(1-Jp))` with xi=10, which makes compacted snow roughly 3x stiffer than elastic. Snow runs at
  ~0.08 of the linear-CFL limit where fluid and elastic sit at 0.21-0.26.
- **So: if sand hardens or dilates, its dt could be as small or smaller.** Measure it, report the substeps
  per frame it implies at 60fps, and say plainly whether sand or snow is now the binding constraint.

**3. Multi-material in one grid.** Canonical `simulate()` runs one material at a time. Add a **per-particle
material id** and a step that branches on it, so a single grid can hold all four at once. Keep the branch
structure honest — each material must still take exactly its canonical path.
- **Interactions between materials may be weird or unphysical. That is explicitly fine.** The goal is to
  show four materials coexisting in one grid for the first time, not to get contact right.
- Verify the branching change did not alter single-material behaviour: a one-material run through the new
  multi-material path must match canonical `simulate` to simulator self-noise. Show that number.

**4. Visuals of all four, simple particle rendering.** Colours for distinction: **elastic red, sand yellow,
water blue, snow white**. Reuse the renderer from
`runs/material-variants/interactive-simulation-of-one-material/` as a starting point if it helps. Show each
material alone AND the four together in one grid, **as motion** — the claims here are about how materials
behave, so video, not final frames. Rendering polish is deliberately held out for a later task; dots and
colour are enough.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope every claim to exactly what was tested. One task supports a **hypothesis**, not a general truth.
- To claim generality, test generality (several tasks/conditions). Otherwise label the broad version a
  conjecture and say so.
- The manifest must carry an honest `hypothesis` (why the result holds) and `limitations` (what was not
  tested) field. Prefer "on this task, X" over "X is true".

## Visualization standard (this is graded, not optional)
Results are how the work becomes understandable, so make the visuals *informative*, not decorative.
- **Show the quantity the objective is actually about.** If the task optimizes the center of mass to a
  target, the video must overlay the center-of-mass trajectory and the target, not just show the blob
  moving. The viewer should be able to see *how close it got*, not infer it.
- **Any comparison shows both sides against each other, in the same medium as the claim.** If the result is
  an improvement, a change, or a learned output judged against a baseline or ground truth, **the ground
  truth is mandatory** and must be shown *next to or overlaid on* the result — as video if the claim is
  about motion (not two lone final frames). A learned rollout with no clear ground-truth comparison is not
  evidence. (Use the canonical `sim.physics.simulate` for the ground truth.)
- **Prefer clear, simple, informative demos over dense technical ones.** A frame with the grid drawn and a
  heatmap of mass or velocity teaches more than a raw particle splat. Annotate axes, targets, and key
  quantities.
- Every `plot` result needs labeled axes and readable fonts (the dashboard renders small on iPad).
- **View what you exported before you write a single finding.** Open every image you saved and watch every
  video (read the file back with your own tools, do not trust that it came out right). Ask of each one: does
  it actually show the quantity the objective is about? Are the axes and labels correct? Is anything
  degenerate — a control that never moved, an empty or clipped frame, a flat line, a blown-up curve, the
  wrong overlay? A number reported without looking at its picture is not evidence, and a misleading or
  broken figure is **regenerated, not shipped**. This step is mandatory, not a courtesy.
- Good visuals here are reusable: the best of them belong in the training textbook too (see below).

## TL;DR (required manifest field)
Write a **`tldr`**: one sentence, no jargon, stating what happened **including what failed**. It is the
first thing on the task page and exists so many tasks can be triaged without opening them. "Investigated
X" is not a TL;DR; "X worked where trained and fell apart everywhere else" is.

## Present findings, not conclusions
The user does the reasoning; you make the evidence legible. Show **what happened** and keep
**observed / hypothesised / would-test** visibly separate — an interpretation goes in `hypothesis`,
labelled as one, never asserted on the page as if observed. Never let an interpretation replace the
artifact it came from.

## Your task page (required — read `spec/style_task_page.md` in full)
You do **not** fill in a fixed card layout. You **design the page this result deserves** and ship it as
`custom_html` in the manifest; it renders as the lead of the task page, with the standard blocks collapsed
beneath it as the evidence layer.

- Before writing any HTML, answer in one sentence: **what is the single thing a reader must walk away
  knowing?** Build the page so that thing is unmissable.
- **Find the flip.** The strongest pages let the reader *switch between two states and watch the finding
  happen* — two metrics over the same cells, learned vs truth, trained vs held-out. A toggle the reader
  operates beats two static figures side by side. Reach for this first.
- **If a summary number can look fine while the result is wrong, that gap IS the finding** — show the
  metric that catches it right next to the flattering one.
- Fully self-contained (sandboxed iframe: no CDNs, no fetch, inline data and CSS/JS). Media by absolute
  `/api/data/learning-taichi/runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid/<file>` path. Prefer drawing from `metrics.json` data
  over embedding images. Also write it standalone to `bespoke_page.html` in your run directory.
- **Open the rendered page and click every control before you ship it.** Not the JSON — the page.
- Exemplar to match or beat: `runs/material-variants/train-one-nn-to-mimic-viscosity-and-st/bespoke_page.html`
  (built by `harness/tools/build_exemplar_page.py`).

## Training textbook contribution (required)
End the run by adding **at least one short, standalone training page** under `reports/training/` in the
objective textbook voice (`spec/style_training_report.md`): impersonal, no first or second person, no
reference to this brief or "this run", and readable cold by someone who never saw the task. Prefer **one
or two short new pages over one long one**, link prerequisites with `[[anchors]]`, and embed an
informative figure or short video where a picture beats prose. Teach the *understanding* the task
produced, not a log of what was done.
- **Lead with the intuition, keep it short, and add depth only where it earns its place.** The textbook is
  a growing corpus a person actually has to track, so a new page states the key idea and its "why" up front
  and stays skimmable. Reach for concision (`spec/style_training_report.md` → "Brevity and prioritization").
- **Keep implementation details and task-specific numbers out of the textbook.** Exact hyperparameters,
  code line ranges, this-run loss values, and one-off results belong in the manifest fields below, not in a
  timeless teaching page. Put the *understanding* in the book; put the *evidence* in the run.
- **Before adding a page, check whether the idea belongs on an existing one.** If a current page already
  owns this idea, extend or tighten it instead of adding a near-duplicate — the corpus should stay cohesive,
  not accrete overlapping pages.
- **Over-include math prerequisites** the page leans on (linear algebra especially — matrices, determinant
  and trace, SVD/polar decomposition). Write or extend the prerequisite page **before** linking to it.
- **Every `[[link]]` you write must resolve** to a section that already exists and covers what the sentence
  promises. Do not point a reference at a not-yet-written prerequisite.
- **When you introduce a material or model parameter** (E, timestep, resolution, a learning rate), show its
  **effect** (a worked example or small figure), not just its definition — see `spec/style_training_report.md`.
- **Render-check the page** (KaTeX) and **view any figure/video you embedded** in it, same as above.

## Output contract
Write `runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid/manifest.json` (schema v2 — see `runs/README.md`) plus its media,
with: `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (video / image /
plot / table), and `training_refs[]` pointing at the page(s) you added. Leave everything on disk; do not
commit.
- **Two-layer prose: `summary` (shown) + `full_report` (expander).** Put a tight, human-legible
  **`summary`** (1–2 clean paragraphs, jargon only where it earns its keep, leading with the takeaway) in the
  manifest — this is what the user reads by default. Put the full detail (the exhaustive findings, method
  notes, per-condition numbers) in **`full_report`**, shown behind a "Full report" expander. Raw numbers go
  in `metrics.json`. `objective` stays one or two sentences. A reader should get the point in ~15 seconds
  from the `summary` alone. (If a worker writes only `findings`, the dashboard shows it as the summary — so
  keep `findings` short and put depth in `full_report`.)
- **Write the manifest LAST, after every media file it references already exists on disk, and make
  `results[]` reference ONLY files that actually exist.** Never list a planned-but-unrendered scene — a
  dangling media `src` renders as a broken tile on the dashboard, and the orchestrator will reject it.
  Verify every `src` resolves to a real file before you finish.

## Paths & params
- Run dir: `runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid/`
- Canonical physics: `sim/physics/` — `core.py` (MAT, the step), `signatures.py`, `PROMOTION.md`, `VERSION`.
- **Do NOT change fluid / elastic / snow parameters.** Add sand alongside them. The WebGPU port task
  generates its constants from `sim.physics`, so an incidental change to elastic would silently move that
  task's ground truth.
- Stamp `physics_version` in the manifest, and state the version **before and after** your bump.
- 2D, grid ~128^2, particle counts in the low thousands. Forward simulation only — no gradients.

## Definition of done
- **Sand is promoted into `sim/physics/`** with its own golden signature (piles, holds an angle of repose,
  does not spread like a fluid), all pre-existing signatures still green, and the version bumped.
- **Sand's stable `dt` is measured and reported**, with the substeps/frame it implies at 60fps and an
  explicit statement of whether sand or snow now binds the Demo's budget.
- **Four materials run in one grid**, with a per-particle material id, shown as video.
- A single-material run through the new branching path **matches canonical `simulate` to self-noise**, with
  the number shown — proof the refactor changed nothing.
- Particle-rendered visuals of all four, colour-coded, each alone and all together, as motion.
- The manifest states the physics_version before and after the bump.
Always includes, regardless of task:
- **The task is finished within your turn.** Do NOT spawn a long render/sim/training job in the background
  and end your turn "waiting" on it — run it to completion (block or poll within your turn), view the
  outputs, and finalize the manifest and training page before you stop. A worker's value is finished output
  on disk, not a detached process someone else has to babysit.
- Manifest carries scoped `findings`, an honest `hypothesis`, and a `limitations` note (Evidence discipline).
- **Every exported figure/video has been opened and viewed**, shows the claimed quantity, and is not
  degenerate. Nothing misleading or broken ships.
- **Every manifest media `src` resolves to a file that exists** (no dangling references — see Output contract).
- The training page(s) render cleanly (KaTeX checked), read standalone in the textbook voice, **every
  `[[link]]` resolves**, and the **math prerequisites the page leans on exist**.

## Known failures to avoid
- **Do not fork or tweak the frozen materials.** Snow's hardening xi already drifted once (10.0 canonical
  vs 3.0 in two learned-material tasks); that is exactly the failure this rule exists to prevent.
- **Do not let the multi-material refactor silently change single-material results.** Prove it didn't.
- **Do not report sand's dt as an aside.** It is a headline number for the Demo budget.
- **Do not claim an angle of repose you have not looked at.** Watch the video; a pile that slumps flat is a
  failed signature, not a stylistic choice.
- Do not report a number without looking at its picture; do not spawn a long background job and end your
  turn waiting on it — finish inside your turn.
