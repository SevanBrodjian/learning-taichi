## 2026-08-18 — profile a learned grid update on WebGPU

- Harvested 21M (node state -> node velocity) pairs by driving the canonical Taichi kernels on three
  water scenes and snapshotting the grid either side of `grid_op`. Friction swept by re-running the
  canonical kernel on each captured pre-state at three coefficients.
- Trained 8->h->h->2 ReLU MLPs at h = 8/16/32/64 in numpy (no torch in this venv). Stage 1 reached 2.7%
  mass-weighted node-velocity error at width 64 and the rollout still detonated after ~1,800 substeps.
- Diagnosed it: G2P gathers a spatial derivative of the node velocity field, and the derivative of the
  cell-wise fit was as large as the signal (7 against 9). Added stage 2, training on whole grids against
  the first differences. Rollout survival roughly doubled; the derivative error moved 0.41 -> 0.38.
- Wrote `web/mpm-nn-webgpu.js`: fluid MLS-MPM in WGSL with four selectable grid kernels (analytic, nn
  dense, nn with empty cells skipped, and a null kernel with identical memory traffic for differencing).
  Seven storage buffers, one under the guaranteed limit, by packing velocity and J into one vec4.
- Verified the WGSL MLP against the same weights on the host to 5e-6 absolute by recovering the exact
  per-cell inputs through the null kernel. Dense and sparse agree bit for bit on every occupied cell.
- Two protocol errors caught and fixed before the numbers were believed: variants must be interleaved
  within each repetition, and anything that advances the simulation must be re-seeded between variants.
- Key measurements: grid kernel 0.047 us analytic vs 1.3 / 3.4 / 41 / 48 us at widths 8/16/32/64, flat in
  particle count; a 36x cut in dispatched workgroups changes the time by 1%, so the kernel is
  latency-bound; cost is non-monotonic in width (48 cheaper than 40) across two independent passes.
- Sharpest accuracy number: gravity contributes 4.9e-4 of velocity per substep and the width-64 network's
  own error is 2.7e-2, i.e. 56x larger, so the learned fluid measurably does not fall.
