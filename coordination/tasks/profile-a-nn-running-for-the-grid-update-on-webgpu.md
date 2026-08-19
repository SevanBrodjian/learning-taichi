# Worker brief: Profile a NN running for the grid update on WebGPU

## Effort tier: deep
Set to `deep` (the seed said `standard`). The reason is scope, not difficulty: this asks for a trained
network, a WGSL inference path, an interactive live demo, **and** a two-axis performance sweep. Any one of
those is a standard task. **Persist**, and run everything to completion inside your turn.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `profile-a-nn-running-for-the-grid-update-on-webgpu`. You are **NOT the
orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results to disk
under `runs/material-variants/profile-a-nn-running-for-the-grid-update-on-webgpu/`, extend the training
textbook, and exit. **Do not commit** — the orchestrator reviews and commits your work.

## Notifications + live status
```
python harness/tools/notify.py --kind started  --task profile-a-nn-running-for-the-grid-update-on-webgpu "<one plain sentence>"
python harness/tools/notify.py --kind finished --task profile-a-nn-running-for-the-grid-update-on-webgpu "<one plain sentence>"
python harness/tools/task_status.py --direction material-variants --task profile-a-nn-running-for-the-grid-update-on-webgpu --step "<a few words>"
```

## Objective
Answer one question with measurements: **is it viable to replace the entire analytic grid update with a
learned network on WebGPU, and at what network size and particle count?** This is a **profiling** task. The
network's accuracy is secondary; its **cost** is the deliverable.

## The seam, stated exactly
Keep the MPM scaffolding — **P2G and G2P stay analytic.** Replace the **whole grid update** kernel: the
thing that takes each node's accumulated mass and momentum and produces its velocity, including gravity and
the boundary/friction handling. In the existing engine that is the fused `grid_op` (which also zeroes the
accumulators — keep that part, it is bookkeeping, not physics).

- **Input per cell**: node mass and momentum, plus whatever the network needs to know about boundaries
  (position or a wall flag). State what you feed it.
- **Output per cell**: node velocity.
- **The network must do the whole job** — gravity and walls included. A network that outputs a velocity you
  then apply gravity and a wall clamp to has not replaced the grid update, and that is the failure mode this
  brief exists to prevent. If you end up doing that, say so explicitly rather than reporting it as success.

Train it on whatever is convenient — **a single fixed material and scene (water) is explicitly fine.**
Generate ground truth with canonical `sim.physics` (forward, no gradients needed).

## The arithmetic that frames the whole task — check it, then design around it
Work this out before writing code, because it determines what the sweep should even measure.

The grid is $128^2 = 16{,}384$ cells and a shared timestep of $5\times10^{-5}$ means $20{,}000$ substeps per
simulated second. So a dense sweep costs

$$16{,}384 \times 20{,}000 \approx 3.3\times10^{8}\ \text{cell-updates per simulated second.}$$

Two consequences, and the second is the interesting one:

1. **On paper the FLOPs are affordable.** A hidden width of 16 is ~770 FLOP/cell → ~250 GFLOP/s; width 64 is
   ~3 TFLOP/s. Against a quarter of a 4090 (~20 TFLOP/s fp32) that is ~1% and ~15% of *peak*. **Peak is not
   what you will get** — tiny per-cell MLPs are latency- and memory-bound, and achieved efficiency is
   routinely one to two orders of magnitude below peak. **The measurement is the point; the FLOP estimate is
   only there to tell you the question is not absurd.**
2. **A dense grid update costs the same no matter how many particles there are.** The cell count is fixed.
   So the "performance across particle counts" axis the seed asks for will, for a dense sweep, be *flat* for
   the NN and rising only for P2G/G2P — and if you measure that and report it, that is a real finding, not a
   null result. It also points at the one genuine design axis: **compacting to occupied cells only**, which
   makes the NN cost scale with particles instead. Whether you implement compaction or not, **address this
   explicitly** — a sweep over particle counts that does not explain why the NN curve is flat has missed
   what it measured.

## Cost anchors from the existing engine (same GPU, same physics)
The analytic WebGPU port measured, at 167 substeps/frame: **5.7 µs/substep at 500 particles rising to 14.1
µs at 16,384**, with a per-dispatch floor of **1.11 µs**. The grid update is one of three dispatches in
that. Your NN grid update has to be read against those numbers, and **the dispatch floor matters**: if the
network needs more than one dispatch per substep you are paying ~1.1 µs before any arithmetic.

**Derate for reality.** The user's instruction: assume **at most a quarter** of this machine's compute is
available in practice, since this is a high-end GPU. Report both the measured 4090 number and the derated
one, and make clear which is which.

## Experiments / deliverables
1. **Train the network.** Small, supervised, against canonical grid-update behaviour on a water scene. Report
   how well it fits — briefly. This is not an accuracy study.
2. **WGSL inference path**, weights in a buffer. Verify it produces the same numbers as the trained network
   on the host before timing anything.
3. **The sweep: network size × particle count.** Several hidden widths (something like 8/16/32/64, and say
   why you chose the set) crossed with several particle counts. Time with **`timestamp-query`**, not
   `performance.now()` — Chromium clamps that to 100 µs and it has already produced one wrong phase split in
   this project. Separate the NN grid-update cost from P2G/G2P so the reader can see which dominates.
4. **The verdict**: at a quarter of this GPU, what is the largest network that holds real time, and at what
   particle count? State it as a number with its conditions attached.
5. **An interactive WebGPU demo on YOUR TASK PAGE** showing it running. Not the Demo page.

## HARD SCOPE — do NOT touch the Demo page
Do not edit `harness/dashboard/src/components/DemoView.jsx` or anything under
`harness/dashboard/src/components/mpm/`. The live demo for this task is **embedded in your own task page**.
Do not change `sim/physics/` either — it is read-only here.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- **A timing is a claim about one GPU, one browser, one scene.** Say so. Every number is an RTX 4090 in
  Chromium unless stated.
- **Assert non-zero motion before believing any timing.** A silently-dropped dispatch produces a beautiful
  flat curve over nothing, and that has already happened once in this project (nine storage buffers over the
  8-per-stage limit → an invalid bind group → all-zero trajectories at 0.44 ms/frame).
- Keep **observed / hypothesised / would-test** separate. `hypothesis` and `limitations` required.
- If the honest answer is "not viable", **that is a perfectly good result** — report it plainly with the
  numbers that show it. Do not tune the framing to make it sound positive.

## Visualization standard (graded, not optional)
- The headline is a **cost plot**: NN grid-update time against network size and particle count, with the
  real-time budget drawn on it as a line, and the **analytic grid update as the mandatory baseline** — the
  comparison is "learned vs the formula it replaces", so the formula must be on the chart.
- Show the derated (quarter-GPU) budget as well as the raw one.
- The interactive demo must actually run and be visibly water-like; if the learned update is visibly wrong,
  **show that too** rather than hiding it behind a cost plot.
- Labeled axes, readable fonts. **Open every figure and watch every clip before writing a finding.**

## TL;DR (required manifest field)
One sentence, no jargon, including what failed or did not work.

## Your task page (required — read `spec/style_task_page.md`)
Ship as `custom_html` + standalone `bespoke_page.html`, self-contained (no CDNs, no fetch). This page hosts a
**live WebGPU demo**, so it is heavier than usual — make sure it degrades gracefully where WebGPU is absent
(`navigator.gpu` is hidden outside a secure context, so "absent" usually means an insecure origin, not an
unsupported device). **Open the rendered page and click every control before shipping.**

## Training textbook contribution (required)
At least one short, standalone page under `reports/training/` in the objective voice
(`spec/style_training_report.md`). The natural subject is **when a learned operator is worth its cost** —
that a per-cell network is evaluated once per cell per substep, so its cost is set by grid size and timestep
rather than by scene complexity, and why arithmetic intensity rather than FLOP count decides whether a tiny
network is fast. Check `core/real-time-cost` and `core/fixed-point-atomics` first and **extend rather than
duplicate** if that is where this belongs. Every `[[link]]` must resolve.

## Output contract
`runs/material-variants/profile-a-nn-running-for-the-grid-update-on-webgpu/manifest.json` (schema v2) +
media: `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]`, `training_refs[]`,
`physics_version`.
- **Two layers: `summary` (shown) + `full_report` (expander).** The summary must contain the verdict.
- **Write the manifest LAST**; every media `src` must resolve to a real file.

## Paths & params
- Run dir: `runs/material-variants/profile-a-nn-running-for-the-grid-update-on-webgpu/`
- Engine to start from: `runs/material-variants/the-demo-mvp-.../web/mpm4-webgpu.js` (four-material WGSL) or
  the simpler `runs/material-variants/webgpu-port-.../web/mpm-webgpu.js` (elastic-only, smaller to modify)
- Verification harness pattern: `runs/material-variants/webgpu-port-.../verify/` (`serve.py` — localhost is
  a secure context, which is the trick for getting `navigator.gpu` in a test harness)
- Physics: `sim/physics/` @ **`phys-c518316a4a05`** (read-only)

## Definition of done
- The **whole** grid update runs as a network in WGSL — gravity and boundaries included, or the gap stated.
- A sweep over network size × particle count, timed with `timestamp-query`, with the **analytic baseline**
  and the real-time budget on the same chart, raw and derated to a quarter GPU.
- **A stated verdict**: largest viable network, at what particle count, under what assumptions.
- A working interactive WebGPU demo **on the task page**.
- The Demo page and `sim/physics/` untouched (`git status` clean for both).
- Finished within your turn; non-zero motion asserted before any timing is believed.
- Manifest carries scoped findings, honest `hypothesis` and `limitations`.
- Every figure/clip opened and viewed; every media `src` resolves.
- Training page renders (KaTeX), reads standalone, every `[[link]]` resolves.

## Known failures to avoid
- **The 8-storage-buffers-per-stage limit.** Exceeding it silently invalidates the bind group and drops every
  dispatch — a gorgeous flat timing curve over all-zero data. Adding weight buffers makes this *more* likely.
  Keep the error scopes and the `uncapturederror` listener; assert motion first.
- **`performance.now()` is clamped to 100 µs in Chromium.** Use `timestamp-query`.
- **Do not report a network that only *partly* replaces the grid update as if it replaced all of it.**
- **Do not let a flat particle-count curve pass without explanation** — see the arithmetic section.
- Do not touch the demo page or the physics.
