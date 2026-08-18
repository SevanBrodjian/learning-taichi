# Worker brief: The Demo MVP — four materials on WebGPU, live on the Demo page

## Effort tier: deep
A genuinely hard task, and the flagship one. **Persist**: iterate, debug, and get a working page. Do not
stop at the first plausible result, and do not stop at a beautiful page whose physics is wrong. Long is
fine. But read **"What matters most"** below before you start — this task has a real risk of spending all
its time on the hardest sub-problem and shipping nothing, and that outcome is worse than a partial page.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page`. You are
**NOT the orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results
to disk under `runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/`, extend
the training textbook, and exit. **Do not commit** — the orchestrator reviews and commits your work. Fire
the pings below.

## Notifications (exactly two) + live status
```
python harness/tools/notify.py --kind started  --task the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page "<one plain sentence>"
python harness/tools/notify.py --kind finished --task the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page "<one plain sentence>"
```
Live status at each coarse milestone (3–6 times, one short phrase each):
```
python harness/tools/task_status.py --direction material-variants --task the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page --step "<a few words>"
```

## Objective
Put the first real, running simulation on the **Demo tab**: all four canonical materials — elastic, water,
sand, snow — on **one grid**, in **real time**, on **WebGPU**, that the user can add to, drag, switch
render modes on, and reset. This is the MVP of `coordination/demo-mvp.md`; **read that document first, it
is the target this task realises.**

## What matters most (read before planning your time)
This is an integration task with one hard sub-problem inside it. Priority order, highest first:

1. **A working page on the Demo tab.** Real time, WebGPU, one grid, materials addable and interacting,
   three render modes, a reset button, drag interaction. This is the deliverable.
2. **Each material is recognisably itself** — water spreads flat, sand slumps to a repose angle, snow and
   elastic hold a slope.
3. **Quantitative agreement with canonical physics** on a fixed scene.

**If time runs short, sacrifice from the bottom.** Shipping three materials, or four materials with a
documented numerical gap, and saying so plainly, beats shipping nothing. What you may **not** do is ship a
material that looks wrong and not say so. A quiet defect is the one unacceptable outcome.

## The engine work — what exists, and the one genuinely new piece
Start from `runs/material-variants/webgpu-port-of-the-interactive-simulation/web/mpm-webgpu.js` (702 lines,
working, verified). **Read its header comment block (lines 19–52) before touching anything** — it is a list
of things WebGPU forced, each of which cost real time to discover.

**It is elastic-only.** P2G computes stress through the closed-form 2D polar rotation, which is all elastic
needs, and G2P updates `F` with no plastic projection at all — no `Jp`, no material id, no singular values
(`mpm-webgpu.js:238-274`). Water works through a different path and is comparatively easy. **Snow
(Stomakhin clamp on the singular values) and sand (Drucker-Prager return mapping) both need a real SVD**,
and porting a 2×2 SVD to WGSL is the substantial new engineering in this task.

Treat the SVD as the highest-risk item and **verify it in isolation before wiring it into the sim**: feed
it a few hundred random and adversarial 2×2 matrices (near-singular, near-rotation, reflections with
negative determinant, strongly anisotropic) and check `U*diag(s)*V^T` reconstructs the input and that `U`
and `V` are genuinely orthogonal. **A subtly wrong SVD produces plausible-looking motion**, which is
exactly the failure that survives a visual check and poisons everything downstream.

### The storage-buffer trap — this WILL bite you
The device guarantees only **8 storage buffers per shader stage**, and the existing layout already uses
**7** (momentum X/Y are interleaved into one buffer and grid-velocity doubles as the display buffer,
specifically to get under the limit). Adding a per-particle **material id** and **`Jp`** as two new buffers
puts you at **9 — over the limit**.

The failure mode is the important part: exceeding it produces a silently invalid bind group, **every
dispatch is dropped, and the simulation runs at the speed of doing nothing**, which looks like a
spectacular performance result. It already produced one convincing fiction (a flat 0.44 ms/frame curve
over trajectories of pure zeros). So: **pack the new per-particle state into existing buffers** rather than
adding slots (widening an existing `vec2` to a `vec4` and using the spare lanes is the obvious route), and
**keep the error scopes and the `uncapturederror` listener** that engine already has. Assert non-zero
motion before you believe any timing number.

### Timestep, and why it is a correctness issue rather than a cost one
One grid means **one `dt`**, and snow's `5e-5` is the smallest, so a scene containing snow runs **333
substeps/frame** and everything in it pays that. **The budget is affordable — that is what the last two
tasks bought:** ~2.4 ms/frame at 2048 particles and a **measured 5.2 ms at 16,384** (31% of a 16.7 ms
frame). For contrast the JS port needed 29 ms/frame for snow at only 1000 particles, twice over budget.

But plastic materials **creep per substep rather than converging with physical time** (canonical snow holds
a 56° heap at its own `dt` and collapses to 19° at `dt/4`; elastic, which has no plastic projection, is flat
on both axes). So running sand at snow's timestep gives it *more* creep than canonical sand shows: **a
material's behaviour depends on what else is in the scene.** For an MVP that is acceptable — but the page
must not claim a mixed scene is quantitatively canonical, and your `limitations` must state it.

### Fixed-point atomics: keep the scales, RE-CHECK the range
Use `kM = 24` (mass) and `kV = 22` (momentum) — established by measurement across two scenes, don't change
them casually. **But their headroom was measured on scenes with a fixed particle count, and this demo lets
the user pile material up interactively.** `2^24` saturates at **256 particle-masses on one node**, and
overrun **wraps silently** — no NaN, no error, the block just detonates.

So: **measure the worst-case particles-per-cell your scene can actually reach** (pile all four materials
into one corner and drag them together — try to break it) and report the remaining headroom. If it is thin,
say so, and either cap the particle count or drop `kM`. The prior task's `headroom` and `density_probe`
metrics show the shape of this measurement.

## The page — what it must do
**Where it ships:** `harness/dashboard/src/components/DemoView.jsx`, replacing the placeholder content.

- **The TRANSPLANT CONTRACT at the top of that file still holds.** It imports **nothing** from the harness
  — no `api.js`, no shared components, no app CSS — because this is going onto a personal website. Keep it
  that way, styles inline. Preserving that promise is part of the definition of done.
- **Four materials in a palette** — elastic, water, sand, snow — that the user selects and adds into the
  scene, then watches interact. Canonical colours from `sim/physics/` (elastic red, water blue, sand
  yellow, snow white); take them from the registry, don't invent them.
- **Three render modes**, the same three the first interactive demo shipped
  (`runs/material-variants/interactive-simulation-of-one-material/web/demo.js:51`): **material** (`blob`),
  **grid mass** (`grid`), **particles** (`pts`). With four materials the material view is now the one that
  has to distinguish them, so it carries more weight than it did.
- **A reset button** — the one control the user named as a hard minimum.
- **Drag / interact with the scene.** The engine already has a poke/drag external body force layered on the
  grid update (off by default, never enabled during verification) — wire it up.
- Sim-speed and delete-material controls are welcome, not required.
- **Degrade gracefully with no WebGPU.** The existing probe already distinguishes the three cases; keep
  that. Note that "absent" usually means an **insecure origin** rather than an unsupported device
  (`navigator.gpu` is hidden outside a secure context), so the message must say *why*, not just "no".
- **`spec/aesthetic.md` at FULL strength.** This is the flagship page, not the dashboard. Read the
  easter-egg bar in that file carefully — the standard is that the user cannot tell whether it is an egg, a
  bug, or normal functionality. Nothing that addresses the user.

## Params come from the physics, never from your fingers
Canonical physics is `phys-bebeaafbe73e`. Generate `params.js` from `sim.physics` with a `gen_params.py`
like the two prior runs' (`.../webgpu-port-.../web/gen_params.py`) and **stamp `physics_version` into the
generated file**. Retyping a constant into JS is the exact drift `CLAUDE.md` forbids. Use
`spec/registry/materials.json` for the per-material parameters and colours.

Canonical ground truth is a **forward** sim: call `sim.physics.simulate`. Your WGSL engine is a
reimplementation of the *step* for the browser, which is allowed — but it may not reimplement the
*parameters* or the *constitutive law*, and any deviation must be stated in your findings.

## Verification — proportionate to an MVP
This is a demo, not a physics study. **Do not run a full `traj_rmse` campaign across variants.** Do run:

1. **The SVD unit check** described above. Non-negotiable; it is cheap and it is the thing most likely to
   be silently wrong.
2. **Per-material sanity, against canonical, on one fixed scene each.** The heap test the sand task used is
   the right instrument — water spreads flat, sand slumps to ~26°, snow and elastic hold the seeded slope.
   Compare against `sim.physics.simulate` on the **same initial conditions**, and judge the difference
   against the **self-noise band** (re-run canonical twice; nudge the ICs by ~1e-7) rather than against
   zero. `spec/registry/metrics.json` defines `traj_rmse` — **read the entry before quoting the number**;
   it is a mean per-particle distance, not an RMS.
3. **The density/headroom probe** described above.
4. **A real-time measurement of the actual shipped scene**, not a synthetic one: four materials present,
   snow's substep count, at the particle count you ship. Report the honest fps.

The verification harness from the previous task is reusable and you should reuse it rather than rebuild it:
`runs/material-variants/webgpu-port-of-the-interactive-simulation/verify/` has `serve.py` (localhost is a
**secure context**, which is the whole trick for getting `navigator.gpu` in a test harness), `harness.html`,
`prepare.py`, `score.py`, `baselines.py`, `render.py`. **There is no automated browser driver** — drive the
harness yourself with the in-app browser tools and let it POST its artifacts back to `serve.py`.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope every claim to exactly what was tested. Four materials on one scene supports a **hypothesis** about
  multi-material MPM in the browser, not a general truth.
- Keep **observed / hypothesised / would-test** visibly separate. An interpretation goes in `hypothesis`,
  labelled as one.
- `limitations` must state, at minimum: the shared-`dt` creep consequence, the fixed-point headroom you
  measured, anything that did not make it in, and any material whose agreement with canonical is weaker
  than the others.

## Visualization standard (graded, not optional)
- **Any comparison shows both sides against each other, in the same medium as the claim.** The claim here
  is about motion, so the browser-vs-canonical comparison is **video, side by side or overlaid** — not two
  final frames. **Ground truth is mandatory, never optional.**
- Show the quantity the objective is about. For a material identity claim, draw the measured slope/width on
  the frame; don't make the viewer infer it.
- Every `plot` needs labeled axes and readable fonts (the dashboard renders small on iPad).
- **View what you exported before you write a single finding.** Open every image, watch every video. Is it
  degenerate — a control that never moved, an empty or clipped frame, a flat or exploded curve, the wrong
  overlay? A number reported without looking at its picture is not evidence. Regenerate, don't ship.

## TL;DR (required manifest field)
One sentence, no jargon, stating what happened **including what failed**.

## Your task page (required — read `spec/style_task_page.md` in full)
Design the page this result deserves, ship it as `custom_html`, and also write it standalone to
`bespoke_page.html`. Self-contained (sandboxed iframe: no CDNs, no fetch, inline data and CSS/JS); media by
absolute `/api/data/learning-taichi/runs/material-variants/<task-id>/<file>` path.

For this task specifically: **the page is about a thing that runs**, so the strongest version lets the
reader see the four materials behaving differently side by side and switch between them, rather than
reading that they do. **Open the rendered page and click every control before you ship it.**
Exemplar to match or beat: `runs/material-variants/train-one-nn-to-mimic-viscosity-and-st/bespoke_page.html`.

## Training textbook contribution (required)
At least one short, standalone page under `reports/training/` in the objective textbook voice
(`spec/style_training_report.md`) — impersonal, no reference to this brief or "this run", readable cold.

The obvious subject is **the 2×2 SVD and what plasticity does with the singular values**: why snow's
Stomakhin clamp is a **box** (cohesion — it can stand a vertical wall) while sand's Drucker-Prager yield is
a **cone** (cohesionless — strength proportional to confining pressure, so it cannot), and why the elastic
path never needs the singular values at all. **Check `reports/training/` first** — `core/svd-polar` and
`core/constitutive-models` already exist and this may belong there as an extension rather than a new page.
Over-include the linear-algebra prerequisites you lean on, write them **before** linking, and make sure
**every `[[link]]` resolves** to a section that exists and covers what your sentence promises.

## Output contract
`runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/manifest.json`
(schema v2 — see `runs/README.md`) plus media: `objective`, scoped `findings`, `hypothesis`,
`limitations`, typed `results[]`, `training_refs[]`, and `physics_version`.
- **Two layers: `summary` (shown by default) + `full_report` (expander).** The `summary` is 1–2 clean
  paragraphs a person gets in ~15 seconds, leading with the takeaway. Depth goes in `full_report`; raw
  numbers in `metrics.json`.
- **Write the manifest LAST**, after every media file it references exists. **Every `src` must resolve** —
  a dangling reference is a broken tile and the orchestrator will reject it.

## Paths & params
- Run dir: `runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/`
- Ships to: `harness/dashboard/src/components/DemoView.jsx` (+ any files it inlines)
- Engine seed: `runs/material-variants/webgpu-port-of-the-interactive-simulation/web/mpm-webgpu.js`
- UI seed: `runs/material-variants/interactive-simulation-of-one-material/web/demo.js`
- Physics: `sim/physics/` @ `phys-bebeaafbe73e`; registry `spec/registry/materials.json`
- Key params: `n_grid=128`; snow `dt=5e-5` → 333 substeps/frame shared; `kM=24`, `kV=22`

## Definition of done
- **The Demo tab shows a running four-material WebGPU simulation** with the three render modes, a working
  reset, addable materials, and drag — or a documented, honest subset of that with the gap stated plainly.
- The transplant contract is intact (`DemoView.jsx` imports nothing from the harness).
- The SVD is unit-verified; each material is shown against canonical ground truth **as video**.
- Fixed-point headroom under interactive piling is measured and reported.
- `params.js` is **generated** from `sim.physics`, version-stamped.
- **The task is finished within your turn.** Do NOT spawn a long job in the background and end your turn
  waiting on it — run it to completion, view the outputs, finalize the manifest and training page.
- Manifest carries scoped `findings`, an honest `hypothesis`, and a `limitations` note.
- **Every exported figure/video has been opened and viewed** and is not degenerate.
- **Every manifest media `src` resolves to a file that exists.**
- Training page(s) render cleanly (KaTeX), read standalone, **every `[[link]]` resolves**, prerequisites exist.

## Known failures to avoid
- **The nine-buffer silent kill.** Over 8 storage buffers per stage → invalid bind group → every dispatch
  dropped → a beautiful flat timing curve over all-zero trajectories. Assert motion before believing timings.
- **`atomic<f32>` does not exist in WGSL.** Fixed-point integer atomics; use `round()`, not truncation —
  truncating a signed momentum biases it toward zero, i.e. a systematic numerical drag.
- **WebGPU errors are async and silent.** Keep the error scopes and the `uncapturederror` listener.
- **`navigator.gpu` needs a secure context.** `file://` and plain-HTTP LAN origins do not qualify;
  `http://localhost` does. This trap already produced one wrong "this device has no WebGPU" conclusion.
- **A plausible-looking wrong SVD.** Unit-test it against reconstruction and orthogonality before wiring it in.
- **Don't quote a metric whose implementation you haven't read.** An assumed-wrong `traj_rmse` already
  propagated a bad mechanism through a task page, a training page, and a spec.
- **Don't spawn a long render/sim in the background and end your turn.** Run it to completion.
