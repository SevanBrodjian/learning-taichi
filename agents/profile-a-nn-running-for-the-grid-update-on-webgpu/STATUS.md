# STATUS — profile-a-nn-running-for-the-grid-update-on-webgpu

**Worker finished. Everything is on disk, nothing is committed.**

## Result in one line
Replacing the whole grid update with a per-cell MLP on WebGPU costs 30x (width 8) to 1000x (width 64)
the analytic kernel, is latency-bound so neither fewer particles nor skipping empty cells helps, and the
only width that fits a quarter-GPU real-time budget (16, at ~8,000 particles) keeps water water for 5
frames out of 60.

## Deliverables against the brief's definition of done
- **Whole grid update as a network, gravity and boundaries included** — yes. Inputs are node mass, node
  momentum, four wall flags and the friction coefficient; output is the node velocity G2P gathers.
  Nothing analytic is applied afterwards. Friction is swept as an input over {0, 0.25, 0.5}, though every
  grid state came from water (stated as a limitation).
- **Sweep, network size x particle count, timed with `timestamp-query`** — widths 8/16/32/64 (trained)
  and 4-128 (untrained, cost only) crossed with 512-32,768 particles. Analytic baseline and both budget
  lines on the headline chart, raw and derated 4x.
- **Verdict** — width 16 at ~8,000 particles at a quarter GPU; width 32 only below 2,048 particles on the
  whole device; width 64 nowhere.
- **Live WebGPU demo on the task page** — analytic vs learned side by side, width and training-variant
  selectors, pause/restart, live cost and divergence readouts. Verified running inside the dashboard's
  sandboxed iframe.
- **Demo page and `sim/physics/` untouched** — `git status` clean for both.
- **Non-zero motion asserted before any timing** — particles moved 0.00136 domain lengths by frame 1;
  the analytic port matches canonical Taichi to 0.00217 against its own noise floor of 0.00189.

## Two measurement errors that were found and fixed mid-run (worth knowing)
1. **Variants must be interleaved.** Measuring all repetitions of A then all of B let clock drift look
   like a difference, and produced a negative differenced cost for the analytic kernel.
2. **Re-seed before any timed measurement that advances the simulation.** The four grid kernels shared
   one particle state; the learned ones wrecked it, and the analytic kernel was then timed on a scene
   piled into a few cells, reading 7x its true cost from atomic contention alone.

Both are recorded in the harness source and in the manifest's full report.

## For the reviewer
- The page is 5,877 px collapsed at a 400 px-wide frame, just under the dashboard's 6,000 px clamp.
  Opening one disclosure exceeds it; the iframe then scrolls internally and nothing is unreachable.
- Four new metrics registered in `spec/registry/metrics.json`: `grid_update_us`,
  `node_velocity_mae_massw`, `node_velocity_grad_rel_err`, `frames_tracked_60fps`.
- The training contribution **edits** `core/14-real-time-cost.md` rather than adding a page. That page
  already asserted, without measurement, that a per-cell MLP costs "a couple of orders of magnitude more"
  than the analytic grid update. That claim is now measured, and the section around it is rewritten.
