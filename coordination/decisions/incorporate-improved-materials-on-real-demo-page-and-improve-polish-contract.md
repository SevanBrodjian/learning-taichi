<!-- auto_run_at: 1787276432 -->
# Contract — T-027 · Improved materials on the real Demo page, and polish (deep)

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/incorporate-improved-materials-on-real-demo-page-and-improve-polish.md`

This is the first task that changes the Demo page itself since the MVP.

## Your rendering choices, as I read them
- **Water** — either new option; the worker picks one and says why (B "film" measured cheaper).
- **Snow** — option A, "powder". Measured 0.307 ms against the current 0.309, so it is free.
- **Sand** — option A, "grains over a packed body" (not B, "loose grains").
- **Rubber** — **neither** new option. Keep the current renderer, and only for rubber: less aggressive
  smoothing into one continuous blob, clearer borders.

The brief identifies each option by its published *description* rather than a filename, because the
internal keys are ambiguous and picking the wrong one would fail silently.

## The half that is bigger than it looks
Updating the physics is **not** just regenerating `params.js`. The demo still stamps the old version and
has no `rho`, no `fric`, and one global `NU` — so the WGSL has to actually carry per-material density
(which is what makes sand and rubber sink and snow float, and must **emerge from the mass ratio**, not a
buoyancy force), per-material Poisson ratio, and mass-weighted grid friction. The pass condition is the
sink/float ordering reproducing on the page's own solver.

## Priority order, and why I put it this way
1. **Physics** — the demo is currently showing the old materials.
2. **Responsive layout** — bounded work, fixes a defect you hit daily.
3. **Rendering** — snow → sand → water → rubber, cheapest and most certain first.

I put layout ahead of rendering deliberately: rendering is the open-ended half and would otherwise eat
the whole budget, and shipping snow and sand with water still on the old treatment is real progress
whereas a half-finished water shader is a broken flagship page. Reorder at the gate if you disagree.

**Budget raised to 120 min** (from 60). Three halves, each of which is a normal task.

## What it will NOT do
- **It will not change `sim/physics/`** — read-only at `phys-c518316a4a05`.
- **It will not break the transplant contract.** `DemoView.jsx` and `mpm/` keep importing nothing from
  the harness, so the page still lifts onto your site.
- **It will not regress the large-monitor layout**, which you said you like as it is.
- **It will not claim a WGSL cost from T-020's numbers.** Those were measured in Taichi; whatever ships
  gets re-measured with `timestamp-query`.
- **It will not leave the page broken to be ambitious.** If time runs out it stops at a working page and
  states exactly which materials are still on the old treatment.

Layout gets verified by actually resizing to phone / tablet / laptop and looking, with screenshots at
each — a claim about what fits on a screen cannot be made from CSS.
