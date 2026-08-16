<!-- auto_run_at: 1786922689 -->
# Contract — Interactive Simulation of One Material (standard)

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/interactive-simulation-of-one-material.md`

## The one constraint that decides everything
**It has to run in a browser.** Taichi/CUDA cannot, so this task is fundamentally a **port** of the
MLS-MPM step — and *how the port is done and what it costs* is the actual content, not a footnote.

## What it builds
- **One material: `elastic` (stiff rubber)**, 2D, small grid, running **interactively in the browser** —
  something you can poke and drag, on **mouse and touch**.
- The implementation route (**JS typed arrays vs WebGL/WebGPU vs WASM**) is the worker's call, but it must
  be **justified with a measurement**, not a preference.
- Parameters come from `sim.physics.MAT["elastic"]` — the port may reimplement the **step**, never the
  **parameters** or the constitutive law. (This is the rule snow's ξ already broke: 10.0 canonical vs 3.0
  in two tasks.)

## What it proves
- **Ported vs canonical, as motion, side by side.** Same initial condition through `sim.physics.simulate`
  and through the port, divergence quantified with the registered `traj_rmse`. Drift is a *finding*, not a
  failure — f32-vs-f64 and substepping changes are expected to cost something, and the point is to say how
  much.
- **A real performance budget:** particle count, grid, substeps, sustained FPS, where the time goes, and
  the **particle budget at 60fps** — the number that tells us what a shippable demo can afford. Hardware
  and browser stated.

## Your open question, answered with evidence
You asked whether interactivity is easier via the **standard equations** or a **NN that learns the grid
update**. The brief forbids assuming. Analytic goes first (you cannot validate a learned net without a
correct reference), then a **specific measured judgement** on what a learned grid-update would cost per
frame and whether it is plausible at 60fps — with any conjecture **labelled as one**.

## ⚠️ Scope note — worth your 10 seconds
This produces **one small interactive elastic sandbox**, embedded as the task page's centerpiece. It is the
**conversion beachhead**, not the flagship Demo. If you expected this task to replace the Demo tab, reject
and say so.

## What it will NOT do
- **Not** multiple materials — one only. No fluid, no snow, no learned materials.
- **Not** 3D, and not a pretty renderer — a simple 2D grid is explicitly fine.
- **Not** differentiable, and **not** a training run.
- **Not** wired into the Demo tab. That page stays the placeholder until you have seen this work.
- **Not** a promise that a learned grid-update ships — only a measured judgement on whether it could.
