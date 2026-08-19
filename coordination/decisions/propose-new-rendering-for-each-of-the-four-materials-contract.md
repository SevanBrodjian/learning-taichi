<!-- auto_run_at: 1787102968 -->
# Contract — Propose new rendering for each of the four materials (deep) · RUNS SECOND

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/propose-new-rendering-for-each-of-the-four-materials.md`

**Runs SECOND**, after the physics task, so its water and rubber proposals are drawn against the materials
that will actually ship rather than ones about to be retuned.

## What it produces
A **proposal set** to choose from — not a shipped page. For each of water, rubber, snow and sand: a
rendered result of the real canonical material in motion, the **same scene rendered the current way beside
it**, and a note on what the technique would cost in the live demo. More than one option where there is a
credible alternative.

- **Rubber** — a strong defining border, a non-mushy interior, reading as one coherent solid.
- **Snow** — kept as-is; it is the reference the others diverge from.
- **Water** — smooth and clear rather than "a smoothie". The brief points it at the existing fluid-
  rendering lineage (`runs/realistic-rendering/`, `core/fluid-rendering`) and **forbids re-deriving it** —
  that work already solved much of this.
- **Sand** — granular and fine-grained: individual grains with random size and irregularity.

## The test that keeps it honest
All four rendered **in greyscale, colour removed**. If you cannot tell which is which, the proposal has not
solved the stated problem — since the complaint is precisely that they differ only in hue. Required to be
included even when unflattering.

## What it will NOT do
- **It will not touch the Demo page.** Explicitly out of scope; integration is a later task once you have
  picked an appearance. Proposals live in the run directory as standalone artifacts.
- **It will not change the physics.** The other task owns that. Behaviour complaints go in `limitations`.
- **It will not claim a frame cost it did not measure or explicitly reason through.** Each proposal gets
  labelled *plausible* / *needs work* / *offline-only* against the ~10 ms/frame that the shipped demo
  leaves for drawing at 16k particles. Offline-only treatments are allowed — but must be labelled.
- **It will not treat colour as the answer.** That is what the greyscale test is for.

## Note
Water shows up in both queued tasks with two different diagnoses — "looks like a smoothie" (rendering) and
"too mushy and sticky" (physics). Same symptom, two candidate causes. Running the physics first means this
task can tell how much of the water complaint was actually the rendering.
