<!-- auto_run_at: 1787113307 -->
# Contract — Profile a NN running for the grid update on WebGPU · HELD until rendering finishes

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/profile-a-nn-running-for-the-grid-update-on-webgpu.md`

**This will not spawn while the rendering task is still running.** Its entire deliverable is *timings*, and
GPU contention is exactly what corrupts those — that is the scar already recorded in `CLAUDE.md`. The
countdown is only so it does not need you again; I hold the spawn until the GPU is free.

## The seam
P2G and G2P stay analytic. The **whole grid update** becomes a network: mass + momentum per node in,
velocity out, **with gravity and the walls included**. A network whose output you then apply gravity and a
wall clamp to has not replaced the grid update — the brief calls that out as the failure mode to avoid.
Trained on one fixed material (water is fine); accuracy is secondary, **cost is the deliverable**.

## Two things worth knowing before it runs
**The arithmetic says the question is not absurd.** 16,384 cells × 20,000 substeps/simulated-second =
**3.3×10⁸ cell-updates per second**. A hidden width of 16 is ~250 GFLOP/s, width 64 ~3 TFLOP/s — about 1%
and 15% of a *quarter* 4090. But peak is not what tiny per-cell MLPs achieve; they are memory- and
latency-bound, so the measurement is the whole point.

**A dense grid update costs the same regardless of particle count** — the cell count is fixed. So the
particle-count axis you asked for will come out *flat* for the network and rising only for P2G/G2P. That is
a real finding rather than a null result, and it points at the one genuine design axis: compacting to
occupied cells only, which would make the NN cost scale with particles instead. The brief requires this to
be addressed explicitly rather than shipped as an unexplained flat line.

## Effort raised to `deep` (90 min) — adjust on the dashboard if you disagree
You set `standard`/40. The scope is a trained network **plus** a WGSL inference path **plus** an interactive
live demo **plus** a two-axis sweep — each of those is a standard task on its own. Not harder, just more.

## What it will NOT do
- **It will not touch the Demo page.** The live demo is embedded in its own task page, as you specified.
- **It will not change `sim/physics/`** — read-only at `phys-c518316a4a05`.
- **It will not turn into an accuracy study.** How well the network mimics water is reported briefly and
  is not the objective.
- **It will not spin the result positive.** "Not viable at any useful size" is a perfectly good answer and
  the brief says to report it plainly with the numbers that show it.

Every number will be labelled as one GPU, one browser, one scene — reported raw **and** derated to the
quarter-GPU assumption you asked for.


**Resolution: APPROVED**
