# Contract — Train One NN to Mimic Viscosity and ST (deep)

**Approve to run, or Reject with a note.** Full brief: `coordination/tasks/train-one-nn-to-mimic-viscosity-and-st.md`.

## What the network replaces (the seam)
**The ENTIRE per-particle material update** — the momentum it scatters to the grid **and** the evolution of
its carried state (volume/deformation), **and** the surface-tension effect. Only the MPM scaffolding stays
fixed (P2G scatter, grid update + gravity + floor/walls, G2P gather). No analytic stress / state rule /
capillary force remains in the learned rollout. (This is the fix for last time, which only learned the stress.)

## What it tests
- One net conditioned on `m = (m_visc, m_st)`. Train **3 corners**, hold out the **4th**; a **5×5 grid** shows
  interpolation across conditions and generalization to the held-out corner — **each cell shown against the
  ground-truth liquid** (GT videos included, not just final frames).

## ⚠️ Needs your call — the seed contradicts itself
It lists **(low ST, low viscosity)** as a *trained* condition **and** says to *hold out* that same corner.
My planned reading (matching the prior design): **train (0,0) low/low, (1,0) high-visc/low-ST,
(0,1) low-visc/high-ST; hold out (1,1) high-visc/high-ST.** Adjust here if you meant something else. Note from Sevan: Yes, that's what I meant, good catch.

## Deliverables
5×5 learned-vs-GT grid (still + video/interactive), GT reference clips, trained-corner fidelity, held-out
corner test, a per-cell RMSE heatmap, one short training page. Tight summary + full-report split.

## What it will NOT do
- Will **not** keep any material physics analytic in the rollout (whole-material or it's a reject).
- Ground truth for surface tension uses the existing (working, not-yet-canonical) CSF; **promoting ST into
  `sim/physics` is a separate follow-up**, not part of this task.
- No differentiable canonical GT (GT is forward only); no 3D; one resolution.


**Resolution: APPROVED**
