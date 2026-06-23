# Status — long-rollout-pathologies

**Branch:** claude/awesome-chebyshev-9c6c11  
**Direction:** long-rollout-pathologies  
**Updated:** 2026-06-22

## Current phase
DONE. All experiments complete. Training report written. Committing and merging to main.

## What was done
- Wrote `sim/diffmpm_pathologies.py`: instrumented sim with per-step diagnostics, mass stabilisation, horizon sweep, precision switching, contact isolation.
- Ran 5 experiments: instrument (f32), horizon sweep (128/256/512), f64 precision, mass-stabilised clip (100 iters), contact isolation (center vs. wall target).
- Expanded `reports/training/core/03-failure-modes.md` with prerequisites (exploding gradients, adjoint backward amplification), experimental results table, mechanistic explanation, fixes and trade-offs.
- Updated `coordination/directions.json` (status→done, branch set).

## Key findings
- Root cause: near-zero grid mass (m~1e-12) × long-rollout Jacobian product → f32 overflow.
- Contact hypothesis RULED OUT: center target (no contact) fails EARLIER than wall target.
- f64 eliminates NaN (overflow not singularity).
- Mass stabilisation `max(m, 1e-4)` fully resolves: 100 clean iters, loss 0.15→9.5e-6.
- Taichi implementation insight: `v.grad[t]` is NOT stored after tape exits; only leaf grads (v0.grad) are reliable.
