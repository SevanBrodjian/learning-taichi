<!-- auto_run_at: 1787282589 -->
# Contract — T-027 REWORK: the water rendering only

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/incorporate-improved-materials-on-real-demo-page-and-improve-polish.md`

**Held until the notebook worker finishes** — it is editing dashboard files right now and two workers in
the same build is how edits get silently lost. The countdown is so you do not have to come back; I hold
the spawn.

## Your note is the instruction
> "The water did not get its rendering updated, it's still the old version. Use either of the new
> proposed appearances for water, whichever one is more efficient."

I missed this the first time — my board read never looked at `rework_history`. Fixed in the rulebook and
the `/execute` skill, and this brief now leads with your note verbatim.

## You are right, and the worker was not lying — here is what actually happened
The shipped WGSL genuinely does contain water option B: Beer-Lambert absorption, Fresnel rim, foam. It
ported the **shading**. It did **not** port the **reconstruction** — and for water the reconstruction is
the look.

T-020's proposed water is smooth and glassy because of a screen-space iso-surface with a **jump-flood
distance transform** giving speckle-free thickness. The shipped version reconstructs from four local
neighbour taps, so the surface stays speckled — exactly the "looks like a smoothie" problem the proposal
existed to solve. Hence: worker honestly reports "option B shipped", you honestly see the old water, and
both are true.

## Scope: water only
Staying on **option B ("film")**, since you asked for whichever is cheaper and film measured 0.77 ms
against glass's 0.83 ms. The fix is porting the reconstruction, not swapping the option.

## What it will NOT do
- **It will not re-run T-027.** Physics, responsive layout, snow, sand and rubber are correct and stay
  untouched — that is ~80 minutes of good work and it is not being thrown away.
- **It will not touch `sim/physics/`** or the layout verified across five viewports.
- **It will not write a second run.** It amends the existing manifest and task page, and corrects the
  record where it claimed water was updated — it was half-updated, and that should be on the page.
- **It will not claim success from compiling code.** The brief requires opening the page and looking at
  the water, because believing the code over the pixels is exactly what went wrong.

Frame budget has room (7.06 ms of 16.67 ms), but the distance transform is not free — T-020 measured it
at ~31% of the water pipeline. It gets re-measured, and if it does not fit that gets said plainly.
