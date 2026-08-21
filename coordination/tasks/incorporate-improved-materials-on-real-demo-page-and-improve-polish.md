# Worker brief: T-027 — REWORK. Water rendering only.

## THIS IS A REWORK, NOT A NEW TASK
T-027 already ran and shipped. Its physics, its responsive layout, and its snow, sand and rubber
treatments are **correct and must be preserved**. Sevan sent it back for **one thing**, and this is his
note, verbatim:

> **"The water did not get its rendering updated, it's still the old version. Use either of the new
> proposed appearances for water, whichever one is more efficient."**

Read `runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/manifest.json`
and the existing `web/` before changing anything. **Do not re-run the task from scratch.** Everything not
named below stays exactly as it is.

## Effort tier: standard
Narrow scope, one material, existing scaffolding. Do not expand it.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `incorporate-improved-materials-on-real-demo-page-and-improve-polish`
(T-027, rework). You are **NOT the orchestrator**. Do not spawn further agents. Update the existing run
in place, extend nothing else, and exit. **Do not commit.**

## The diagnosis — start here, it will save you the investigation
The previous attempt genuinely did port water **option B ("film")**: the shipped `mpm4.js` shade pass has
Beer-Lambert absorption, a tight specular, a Fresnel rim and surface-gated foam. That is why it was
reported as done.

**But it ported the SHADING and not the RECONSTRUCTION, and for water the reconstruction is the look.**
T-020's proposed water gets its smooth, clear, glassy surface from a screen-space iso-surface: a filled
mask that owns opacity, kept separate from the density gradient that owns normals, with a **jump-flood
distance transform** giving a speckle-free optical thickness. The shipped version reconstructs thickness
from four local neighbour taps instead, so the surface stays speckled — which is exactly the "it looks
like a smoothie" complaint the whole proposal existed to fix.

Compare these two yourself before you start, because the difference is the entire task:
- `runs/material-variants/propose-new-rendering-for-each-of-the-four-materials/still_fluid_proposed.png`
  (the target: smooth, clear, a clean surface line)
- `runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/still_render_after.png`
  (what shipped: bright, speckled, essentially the old water)

## What to do
Port the **reconstruction**, not just the shading, so the shipped water actually looks like the proposal.
Stay on **option B ("film")** — Sevan asked for whichever is cheaper and film measured 0.77 ms against
glass's 0.83 ms at 720² in Taichi. If, once implemented, the honest measurement says otherwise, say so
rather than quietly switching.

`sim/material_render.py` is the reference implementation from T-020. Read how it builds the mask, the
gradient and the distance field, and port that structure to WGSL.

**Budget:** the shipped page runs 7.06 ms of GPU in a 16.67 ms frame, so there is real headroom, but the
distance transform is not free — T-020 measured it at ~31% of the water pipeline. Measure what you ship
with `timestamp-query` and report the new frame cost. If it does not fit, say so plainly and ship the
best version that does.

## Do NOT touch
- **Snow, sand, rubber.** They are correct and Sevan is happy with them.
- **The physics.** `sim/physics/` is read-only; the demo's `params.js` is already current at
  `phys-c518316a4a05`.
- **The responsive layout.** Verified at five viewports; leave it alone.
- Any other page or component. The transplant contract still holds: `DemoView.jsx` and `components/mpm/`
  import nothing from the harness, and `web/demo.html` stays in sync via `sync_to_dashboard.py`.

## Definition of done
- Water on the shipped page **visibly matches the proposal's smooth, clear surface** — not just the
  shading model. Show it: before/after at identical particle state, plus the T-020 target beside it.
- The other three materials, the physics and the layout are demonstrably unchanged.
- New frame cost measured with `timestamp-query` and reported.
- Update the **existing** manifest and task page rather than writing a second run: amend `findings`,
  `full_report` and `limitations` to reflect what actually shipped, and correct the previous claim that
  water was updated — it was half-updated, and the record should say so.
- **Open the rendered page in the dashboard and look at the water** before you call it done. The previous
  attempt's mistake was believing the code rather than the pixels.

## Known failures to avoid
- **Do not claim the treatment shipped because the shading compiled.** That is precisely what went wrong.
- The 8-storage-buffer ceiling: adding a distance-field target may need packing rather than a new buffer.
  Assert non-zero motion before believing any timing.
- `performance.now()` is clamped to 100 µs in Chromium — use `timestamp-query`.
- A cost that does not change with resolution means you are timing the clock, not the GPU.
