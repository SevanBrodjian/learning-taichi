# Worker brief: Interactive Simulation of One Material

> Filled contract for a cold worker. The queued task's one-line `note` was only the seed; this is the
> executable version. Direction (storage): `material-variants`.

## Effort tier: standard
This task's intensity, set on the dashboard, drives how this worker was spawned (model + reasoning effort
+ how long to persist). The orchestrator fills this in and matches the spawn to it (see the `/execute`
skill). As the worker, treat it as the expected depth:
- **quick** — a light, cheap, bounded task. One clean pass, minimal sweeps; do not over-engineer.
- **standard** — the default. Normal depth and iteration.
- **deep** — a genuinely hard task. Persist: iterate, debug, run the sweeps it takes, and do not stop at
  the first plausible result. Long is fine; a shallow answer is not.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `interactive-simulation-of-one-material`. You are **NOT the orchestrator**. Do not spawn further
agents. Read this brief, do the task, write **all** results to disk under
`runs/material-variants/interactive-simulation-of-one-material/`, extend the training textbook, and exit. **Do not commit** — the
orchestrator reviews and commits your work. Fire the pings below.

## Notifications (exactly two) + live status
At the start:
```
python harness/tools/notify.py --kind started --task interactive-simulation-of-one-material "<one plain sentence: what you're starting>"
```
When your results are on disk:
```
python harness/tools/notify.py --kind finished --task interactive-simulation-of-one-material "<one plain sentence: what's ready to review>"
```
Use `--kind blocked` instead of `finished` if you hit a hard stop. **One sentence of human status, never a
metrics dump or a technical report.**

**Live status (a few times, not spammed).** So the dashboard shows what step this Active task is on, call
this at each coarse milestone (starting, the main phase, rendering, wrapping up) — think 3–6 times over the
run, one short phrase each:
```
python harness/tools/task_status.py --direction material-variants --task interactive-simulation-of-one-material --step "<a few words: current step>"
```
It writes `runs/material-variants/interactive-simulation-of-one-material/status.json` (ephemeral, gitignored). Use `--state blocked` if you
stall on something the user must resolve.

## Objective
Get **one** small MLS-MPM material running as a genuinely interactive, real-time simulation **in a web
browser**, and establish how the Taichi/CUDA setup converts to that format. The deliverable is the first
real step toward the Demo: a playable thing, not a video of one.

## Experiments / deliverables

**The hard constraint that shapes everything: it must run in a browser.** Taichi/CUDA cannot. So this task
is fundamentally a *port*, and the interesting content is how the port is done and what it costs.

**1. Port the step (analytic first).** Reimplement the 2D MLS-MPM step for **one material — `elastic`
(stiff rubber)** — so it runs in the browser at interactive rates. JS with typed arrays, or WebGL/WebGPU
compute, or WASM: **you choose, and you justify the choice with a measurement, not a preference.** Start
from the canonical step in `sim/physics/core.py`.

> **Canonical-physics rule (CLAUDE.md).** A port may reimplement the **step**; it may **not** invent the
> **parameters** or the **constitutive law**. Take `E`, `dt`, grid size and particle volume from
> `sim.physics` (`MAT["elastic"]`) and state in the manifest exactly what differs and why. A port that
> quietly picks its own stiffness is a defect and will be sent back.

**2. Verify the port against ground truth - this is the crux, not a formality.** Run the SAME initial
condition through canonical `sim.physics.simulate` and through your port, and compare. Report the
divergence honestly with a **registered metric** (`traj_rmse`, see `spec/registry/metrics.json`) and show
**both side by side, as motion** - the claim is about dynamics, so the evidence is video/animation, not two
final frames. A port that *looks* plausible but drifts from canonical is a finding, not a failure; say so.
Expect f32-vs-f64 and any substepping change to matter, and quantify that rather than hand-waving it.

**3. Make it actually interactive.** At minimum the user can **poke/drag the material** and it responds in
real time. Mouse **and** touch (this gets read on an iPad). Interaction must feel immediate - if input lag
or stutter is visible, that is the result to fix or to report honestly.

**4. Measure it, and be specific about the machine.** Report **particle count, grid resolution, substeps
per frame, and sustained FPS**, plus where the time actually goes (P2G / grid / G2P / draw). Find the
**particle budget at 60fps** - the number that tells us what a shippable demo can afford. State the
hardware and browser; this is a timing on one setup, not a universal constant.

**5. Answer the seed's open question with evidence.** The seed asks whether interactivity is easier via the
**standard equations** or via a **NN that learns the grid update**. Do **not** assume. The honest ordering
is analytic first (you cannot validate a learned net without a correct reference to compare it against),
then form a **specific, measured judgement**: what would a learned grid-update cost per frame at this
particle count, what would it buy, and is it plausible in a browser at 60fps? If you can cheaply test a
small net, do; if you cannot, say precisely why and what would settle it. **A conjecture labelled as one is
an acceptable answer here; an unlabelled guess is not.**

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
  `/api/data/learning-taichi/runs/material-variants/interactive-simulation-of-one-material/<file>` path. Prefer drawing from `metrics.json` data
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
Write `runs/material-variants/interactive-simulation-of-one-material/manifest.json` (schema v2 — see `runs/README.md`) plus its media,
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
- Run dir: `runs/material-variants/interactive-simulation-of-one-material/`
- Ground truth: `sim/physics/` (`simulate`, `MAT["elastic"]`, `VERSION`) - import, never fork.
- Web source: put the portable source in the run dir (e.g. `web/`) so it can be lifted out later. The
  demo itself ships as the task page's `custom_html` (a sandboxed `allow-scripts` iframe: **fully
  self-contained, no CDN, no fetch, inline everything**).
- Params: take from `sim.physics.MAT["elastic"]`. Grid ~64-128 squared, particles ~2-10k, 2D. Stamp
  `physics_version` in the manifest.
- **Transplant contract:** the demo must not depend on the dashboard, the data server, or the harness.
  Same promise as `harness/dashboard/src/components/DemoView.jsx` - assume it will be lifted onto a
  personal website.

## Definition of done
- A **working interactive elastic simulation runs in the browser**, embedded as the task page's
  centerpiece via `custom_html`, self-contained, at a **stated, measured** frame rate.
- It responds to **mouse and touch** input in real time.
- **Ported behaviour is checked against canonical `sim.physics`** on a matched initial condition, with the
  divergence quantified using a registered metric and shown **as motion, both sides together**.
- The parameters come from `sim.physics`; any deviation is named and justified in the manifest.
- **How the conversion was achieved is explained clearly** - that is the point of this task. The page and
  the training page must make a reader able to repeat the port: what changed from Taichi, what the browser
  forced, what the numerics cost.
- A **measured particle/FPS budget**, with hardware and browser stated.
- A **specific, evidence-backed judgement** on analytic-vs-learned grid update, with conjecture labelled.
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
- **Do not fork the physics parameters.** Snow's hardening xi already drifted (10.0 canonical vs 3.0 in two
  learned-material tasks) - that is the exact defect this rule exists to stop. Import from `sim.physics`.
- **Do not ship a demo you have not run.** Open the rendered page, click and drag it, at a normal window
  size. A page that "should work" is not evidence. Verify it interactively before writing findings.
- **Do not claim real-time without a number.** "Smooth" is not a measurement. Report sustained FPS at a
  stated particle count on stated hardware.
- **Do not let the iframe trap it.** The bespoke page sizes itself; do not build an internal scroll box.
- **Do not depend on `requestAnimationFrame` for correctness.** rAF-dependence has bitten this project
  three times (wheel zoom, node drag, an animation that latched forever when frames stopped). A render loop
  is fine - but nothing load-bearing may permanently stall if a frame never arrives.
- **Do not report a number without looking at its picture**, and do not spawn a long background job and end
  your turn waiting on it - finish inside your turn.
