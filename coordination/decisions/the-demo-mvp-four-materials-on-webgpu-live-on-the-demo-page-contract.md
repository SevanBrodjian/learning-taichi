<!-- auto_run_at: 1787029397 -->
# Contract — The Demo MVP: four materials on WebGPU, live on the Demo page (deep)

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page.md`

## What ships
The Demo tab stops being a placeholder. One grid, real time, WebGPU, **all four materials** — elastic,
water, sand, snow — selectable from a palette, added by you, interacting. **Three render modes**
(material / grid mass / particles, the same three the first interactive demo had), a **reset** button, and
**drag** to push the scene around.

## The budget is affordable — that is what the last two tasks bought
Snow has the smallest timestep, so one shared grid means the whole scene pays **333 substeps/frame**. On
WebGPU that is ~2.4 ms/frame at 2,048 particles and a **measured 5.2 ms at 16,384** — 31% of a frame. The
JS port needed 29 ms for snow at *1,000* particles, twice over budget. Real time with all four is
genuinely reachable now.

## The one hard part
**The WebGPU engine is elastic-only.** It uses the closed-form polar rotation, which is all elastic needs,
and its G2P has no plastic projection at all. **Snow and sand both need a real SVD in WGSL** — snow clamps
the singular values, sand does a Drucker-Prager return mapping on them. That port is the substantial new
engineering here, and a subtly wrong SVD produces *plausible-looking* motion, so the brief requires
unit-testing it (reconstruction + orthogonality, on adversarial matrices) **before** it goes near the sim.

Second trap, already paid for once: the device guarantees **8 storage buffers per stage and the engine
already uses 7**. Adding material-id and `Jp` as new buffers makes 9 — which silently invalidates the bind
group, drops every dispatch, and yields a gorgeous flat timing curve over all-zero trajectories. The brief
says pack into existing buffers and assert motion before believing any number.

## Priority order, since this could eat its own budget
1. **A working page.** 2. Each material recognisably itself. 3. Quantitative agreement with canonical.
**Sacrifice from the bottom.** Shipping three materials with the gap stated plainly beats shipping nothing.
Shipping a material that looks wrong *without* saying so is the one unacceptable outcome.

## What it will NOT do
- **No full `traj_rmse` campaign.** Verification is scoped to an MVP: an SVD unit check, one heap scene per
  material against canonical ground truth, a density/headroom probe, and an honest fps for the *shipped*
  scene. No variant sweep, no atomics precision study — that stays a separate follow-up.
- **No per-material timestep.** One grid, one `dt`. Consequence, and it is a correctness issue not a cost
  one: plastic materials creep *per substep*, so sand run at snow's timestep creeps more than canonical
  sand does. **Material behaviour depends on what else is in the scene.** Acceptable for a demo; the page
  will not claim a mixed scene is quantitatively canonical.
- **No JS fallback.** WebGPU only, with a graceful message that explains *why* it is missing (usually an
  insecure origin, not an unsupported device).
- **No polish pass on rendering.** Distinct colours and the three modes, not a visual-quality study.
- **No physics changes.** `phys-bebeaafbe73e` unchanged; `params.js` generated from `sim.physics`, never
  retyped.

## Also
Aesthetic at full strength (`spec/aesthetic.md`) — this is the flagship page. The transplant contract in
`DemoView.jsx` holds: it imports nothing from the harness, so it can be lifted onto your site.
