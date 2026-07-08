# Differentiating the rollout

The forward step from [[mls-mpm-forward]] is already a composition of differentiable pieces. Turning a whole rollout into something you can optimize takes three deliberate changes, and one idea about memory.

## The inverse task
The first control task throws the blob. Optimize a single shared initial velocity $v_0$ so the blob's center of mass after $T$ steps lands on a target point $x^{*}$.

$$\mathcal L = \lVert \bar{x}_T - x^{*}\rVert^2,\qquad \bar{x}_T = \frac1N\sum_p x_{T,p}.$$

Nothing about the simulator changes for this. Only $v_0$ is free, and the gradient $\partial \mathcal L / \partial v_0$ has to travel back through all $T$ steps.

## Three changes versus the forward code
1. **Time-indexed state with gradients.** Reverse-mode autodiff needs every intermediate it will reuse, so each state field gains a leading time axis. In `sim/diffmpm.py` the fields `x, v, C, J` are shaped `(steps, n_particles)` and `grid_v, grid_m` are shaped `(steps, n_grid, n_grid)`, each allocated with `needs_grad=True`. Step $s$ reads slice $s$ and writes slice $s+1$, so no value the backward pass needs is ever overwritten.
2. **Split the grid velocity into two fields.** The raw P2G result and the post-gravity, post-wall velocity are stored separately (`grid_v_in` and `grid_v_out`). Writing both into one field would be an in-place update that the tape cannot differentiate cleanly.
3. **Wrap the rollout in a tape.** The full forward loop plus the loss runs inside `with ti.ad.Tape(loss):`. Taichi records the forward graph and replays it in reverse to populate `v0.grad`. An optimizer step then nudges $v_0$.

## The memory tension, stated plainly
Storing every intermediate state costs memory that scales like `steps × (grid + particles) × 2`, where the factor of two is value plus gradient. For the conservative first config this is fine, but it is the first wall you hit when scaling the horizon or resolution. The standard escape is **checkpointing**, where you keep only a sparse set of saved states and recompute the rest on demand during the backward pass, trading compute for memory. That tradeoff is its own queued research direction.

## Why gradients through physics are the interesting part
A learned policy can imitate, but a gradient through the actual dynamics tells you the exact direction in control space that improves the outcome, grounded in real physics rather than a surrogate. That is the capability a structured-generative-worlds vision leans on, since authoring a world means steering its dynamics on purpose. The catch is that the gradient is only as trustworthy as the smoothness of the step, which is exactly where [[failure-modes]] begins.
