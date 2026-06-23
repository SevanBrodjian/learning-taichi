# LOG — long-rollout-pathologies

## 2026-06-22
- Session started. Read task brief, CLAUDE.md, spec/, existing 03-failure-modes.md, diffmpm.py.
- Wrote sim/diffmpm_pathologies.py: per-step grid-mass diagnostics, mass-stabilised grid_op, horizon sweep, f64 switching, contact isolation, NaN-skip.
- Ran experiments:
  - instrument (f32, 512 steps): NaN@iter7. Grid mass min = 1.115e-12.
  - horizon sweep: 128→no NaN (80 iters), 256→NaN@11, 512→NaN@7.
  - f64 (512 steps): no NaN in 80 iters, loss→4.5e-5. Confirms overflow not singularity.
  - clip+stabilize (f32, 512 steps, mass_eps=1e-4): no NaN in 100 iters, loss 0.15→9.5e-6.
  - isolate: center target NaN@3, wall target NaN@32. Rules out contact hypothesis.
- Key insight: Taichi v.grad[t] NOT stored after tape exits (only v0.grad reliable).
- Expanded reports/training/core/03-failure-modes.md: prerequisites on Jacobian products + adjoint backward, full experimental evidence, mechanistic explanation, 5 fixes with trade-offs.
- Updated coordination/directions.json (status→done, branch set).
- Committing and merging to main.
