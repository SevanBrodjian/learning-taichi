# DiffMPM — technical design (Phase 1 core)

Seed: `sim/mpm88.py` (forward MLS-MPM). Goal: make the rollout **differentiable** and solve an inverse
control task by gradient descent through the simulation. This note is the implementation map the
training report (`reports/training/diffmpm.md`) will expand from.

## The task
Optimize a control parameter $\theta$ to minimize a loss on the final state:
$$\mathcal{L} = \lVert \text{COM}(x_{T}) - x^{*}\rVert^2,$$
where $\text{COM}(x_T)=\frac1N\sum_p x_{T,p}$ is the blob's center of mass at the last step and $x^{*}$
is a target point. **First task:** $\theta$ = a single shared **initial velocity** $v_0\in\mathbb{R}^2$
applied to all particles — the cleanest end-to-end differentiable control demo. (Per-particle $v_0$ and
actuation are later variants in `coordination/research_directions.md`.)

## What changes vs. mpm88 (for autodiff)
1. **Time-indexed fields with `needs_grad=True`.** Reverse-mode AD needs every intermediate state, so
   state fields gain a leading time axis: `x, v, C, J` shaped `(steps, n_particles)`, and
   `grid_v, grid_m` shaped `(steps, n_grid, n_grid)`. `substep(s)` reads step `s`, writes `s+1`.
2. **`ti.Tape`.** Wrap the forward rollout + loss in `with ti.Tape(loss):` so Taichi records and
   auto-runs the backward pass, populating `v0.grad`.
3. **Keep ops differentiable.** The discrete base index `int(Xp-0.5)` is fine (gradients flow through
   the continuous weights, not the integer base). The **wall clamps** (`min`/`max`/`clamp`, mpm88
   lines 49–59, 77) are non-differentiable kinks — for the first run choose target/horizon so the blob
   does **not** hit walls (clean gradients); contact is studied deliberately later.
4. **Loss kernel** accumulates into a 0-D field with `needs_grad=True`.

## Training loop
```
init x[0]=blob, J[0]=1, C[0]=0; v[0]=v0 (the parameter, broadcast)
for it in range(n_iter):
    reset state to step 0 from current v0
    with ti.Tape(loss):
        for s in range(steps-1): substep(s)
        compute_loss(steps-1)
    v0 -= lr * v0.grad        # plain gradient descent first (Adam is a queued direction)
    record loss[it]
```

## Memory & stability (the failure-mode material)
- Memory scales as `steps × (grid + particles) × 2 (value+grad)`. Conservative **first-run config** to
  stay light and stable: `n_grid=64`, `n_particles=4096`, `steps≈512`, `dt` per CFL. Scale up after it
  works; **checkpointing** is the documented fix when the time axis gets too big.
- Anticipated failures to capture in the report: exploding/vanishing gradients over long rollouts;
  NaNs from CFL violation (explicit MPM is conditionally stable); learning-rate sensitivity; biased/
  noisy gradients once contact (wall clamps) is involved; local minima in the control objective.

## Outputs (dashboard contract)
- `runs/<branch>/<run-id>/metrics.json` — loss per iteration.
- `video.mp4` (+ optional `frames/`) — rendered rollout (before vs. after optimization). Render
  **headlessly** (particles → numpy → imageio), never `ti.GUI`.
- `manifest.json` per `runs/README.md`; regenerate `runs/index.json`.

## Decisions resolved
- First inverse task = **throw the blob to hit a target point** (shared initial velocity $v_0$).
  Per-particle actuation / walking (à la DiffTaichi) are queued variants in `coordination/research_directions.md`.

(Design docs do not hold open questions for the user. Anything that needs a decision goes to the
shared decision channel — see `coordination/` — and fires an ntfy `gate`, so nothing for you to
monitor is ever buried in a technical note.)
