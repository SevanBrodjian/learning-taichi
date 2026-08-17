# Worker brief: WebGPU port of the interactive simulation

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
You are a **worker agent** for task `webgpu-port-of-the-interactive-simulation`. You are **NOT the orchestrator**. Do not spawn further
agents. Read this brief, do the task, write **all** results to disk under
`runs/material-variants/webgpu-port-of-the-interactive-simulation/`, extend the training textbook, and exit. **Do not commit** — the
orchestrator reviews and commits your work. Fire the pings below.

## Notifications (exactly two) + live status
At the start:
```
python harness/tools/notify.py --kind started --task webgpu-port-of-the-interactive-simulation "<one plain sentence: what you're starting>"
```
When your results are on disk:
```
python harness/tools/notify.py --kind finished --task webgpu-port-of-the-interactive-simulation "<one plain sentence: what's ready to review>"
```
Use `--kind blocked` instead of `finished` if you hit a hard stop. **One sentence of human status, never a
metrics dump or a technical report.**

**Live status (a few times, not spammed).** So the dashboard shows what step this Active task is on, call
this at each coarse milestone (starting, the main phase, rendering, wrapping up) — think 3–6 times over the
run, one short phrase each:
```
python harness/tools/task_status.py --direction material-variants --task webgpu-port-of-the-interactive-simulation --step "<a few words: current step>"
```
It writes `runs/material-variants/webgpu-port-of-the-interactive-simulation/status.json` (ephemeral, gitignored). Use `--state blocked` if you
stall on something the user must resolve.

## Objective
Port the interactive elastic MLS-MPM step to a **WebGPU compute** path and measure it against the two
existing implementations, so the JS port, the Taichi/CUDA reference and WebGPU can be compared on the same
machine and the same physics.

## Experiments / deliverables
**You have exclusive use of the GPU** — the sand task is finished before you start, precisely so your
timings are not contended. Your headline deliverable is measurements; protect them.

**1. Why this task exists (do not re-derive it).** The JS port is numerically exact but capped at ~1150
particles at 60fps, because dt=1e-4 forces **167 substeps per frame**. The canonical Taichi/CUDA path is
*worse*: ~345 us/substep, **flat from 500 to 16384 particles**, because it pays a **56.4 us kernel launch
per kernel per substep** from Python. The 4090 is idle. That is a measurement of an API usage pattern, not
of the device. **The fix is to stop paying a launch per substep:** record many dispatches into ONE command
buffer and submit once per frame.

**2. WebGPU availability is CONFIRMED — do not spend time re-checking it.** Verified 2026-08-16 on the
Windows desktop (adapter nvidia/lovelace), the iPad, and a MacBook Air M4. One trap worth knowing:
`navigator.gpu` is only exposed in a **secure context**, so `localhost` works but a plain-HTTP LAN origin
hides the API entirely. The dashboard is also reachable over HTTPS at
`https://sevan-windows-home.tail9a3a96.ts.net`.

**3. The crux: WGSL has no atomic float add.** Confirmed by compilation: `atomic<f32>` fails with
*"'atomic' only supports 'i32', 'u32' or 'vec2u'"*. P2G scatters **mass and momentum as floats**.
- **Start with fixed-point integer atomics**: mass into `atomic<u32>`, momentum into `atomic<i32>` — signed,
  because momentum goes negative. Pick a scale multiplier and justify it.
- **Bounded effort.** Find a configuration that works well enough to demonstrate the route is viable. This
  task is NOT the optimisation study; a dedicated deep analysis of the atomic-add implementation is a
  planned follow-up.
- A **CAS-loop float add** (`bitcast` + `atomicCompareExchangeWeak`) also compiles and is *exact* in f32 but
  slower under contention. A quick head-to-head is cheap and would sharpen the follow-up — do it if it does
  not derail the main line, and report what you find either way.

**4. Verify against canonical, the same way the JS port did.** Fixed-point quantisation **changes the
numerics**, so this is the point of the task, not a formality.
- `traj_rmse` (registered — read its caution in `spec/registry/metrics.json`; it is a mean per-particle
  distance, not an RMS and not centre-of-mass) against canonical `sim.physics` on a matched initial
  condition, read against the **simulator's own self-noise band**, shown as **motion, both sides together**.
- The JS port's numbers are the benchmark to beat or match:
  launched-disk `traj_rmse` 1.72e-4 vs 6.63e-5 self-noise and 2.21e-4 for a one-ULP IC perturbation.

**5. Measure it properly.** Use **`timestamp-query`** (available on this adapter) for GPU-side timing rather
than wall clock — and note that `performance.now()` is clamped to 100 us in Chromium, which already
produced one badly wrong phase split in the previous task. Report particle count, grid, substeps/frame,
sustained fps, where the frame time goes, and the **particle budget at 60fps**, on stated hardware.
Put the three paths side by side: JS / Taichi-CUDA / WebGPU.

**6. Scope note.** A JS fallback is **no longer clearly required** — WebGPU is present on every device that
matters. Do not spend effort on fallback parity.

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
  `/api/data/learning-taichi/runs/material-variants/webgpu-port-of-the-interactive-simulation/<file>` path. Prefer drawing from `metrics.json` data
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
Write `runs/material-variants/webgpu-port-of-the-interactive-simulation/manifest.json` (schema v2 — see `runs/README.md`) plus its media,
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
- Run dir: `runs/material-variants/webgpu-port-of-the-interactive-simulation/`
- Start from `runs/material-variants/interactive-simulation-of-one-material/web/` — the working JS port,
  its generated `params.js`, and its verification harness in `verify/`.
- Parameters still come from `sim.physics` via a generator, never retyped.
- **The sand task runs before you and WILL bump `physics_version`.** It is instructed not to touch
  fluid/elastic/snow parameters, so elastic should be unchanged — but **regenerate your params rather than
  copying the old file, and stamp the version you actually used.** If elastic's constants moved, stop and
  report it; that would be a defect in the other task.
- The demo ships as `custom_html` (sandboxed `allow-scripts` iframe): fully self-contained, no CDN, no
  fetch. Also emit portable source in the run dir.

## Definition of done
- **An interactive elastic simulation runs in the browser on a WebGPU compute path**, embedded as the task
  page's centrepiece, responding to mouse and touch, at a **stated, measured** frame rate.
- **One command buffer per frame** (or an equivalent structure that avoids a per-substep round trip), with
  the design explained.
- **Fixed-point atomics work and their numerical cost is quantified** against canonical, shown as motion
  beside the reference, read against the self-noise band.
- **A three-way comparison** — JS / Taichi-CUDA / WebGPU — on the same hardware and physics, with the
  particle budget at 60fps for each.
- A clear explanation of **how the port was done and what WebGPU forced**, good enough to repeat.
- An honest statement of what the atomics approach costs and what the follow-up study should examine.
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
- **Do not re-litigate WebGPU availability.** It is confirmed; if `navigator.gpu` is missing, you are on an
  insecure origin, not an unsupported device.
- **Do not trust `performance.now()` for per-substep timing** — it is clamped to 100 us in Chromium and
  already produced a wrong phase split (3.5:1 measured against a true 1.2:1). Use `timestamp-query`, or
  price phases by differencing whole loops.
- **Do not let fixed-point quantisation slide by unverified.** It changes the physics; measure it.
- **Do not rabbit-hole on the atomics precision study.** Prove the route, then stop — the deep analysis is a
  separate planned task.
- **Do not make anything load-bearing depend on `requestAnimationFrame`.** rAF-dependence has bitten this
  project three times (wheel zoom, node drag, an animation that latched forever when frames stopped).
- Do not report a number without looking at its picture; finish inside your turn.
