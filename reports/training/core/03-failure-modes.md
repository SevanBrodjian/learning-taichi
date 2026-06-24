# Failure modes

The teaching value of differentiable simulation lives here, in what breaks and why. This section grows as runs surface new pathologies. It opens with the first one we actually hit.

---

## Prerequisites for this section

Two mathematical ideas run through everything below. Skim them if you already know them; they are short.

### P1 — Exploding and vanishing gradients in unrolled systems

Any system that applies the same (or similar) operation $T$ times in sequence is an *unrolled* system. Recurrent neural networks are the canonical example. MLS-MPM with $T$ timesteps is another.

Write the single-step update as a map $\phi$:

$$
s_{t+1} = \phi(s_t),
$$

where $s_t$ is the full state at step $t$ (particle positions, velocities, grid values). The chain rule for the full rollout is

$$
\frac{\partial s_T}{\partial s_0} = \underbrace{J_T \cdot J_{T-1} \cdots J_1}_{T\ \text{factors}},
$$

where $J_t = \partial \phi(s_{t-1})/\partial s_{t-1}$ evaluated on the actual forward trajectory. This is a product of $T$ Jacobian matrices.

The behavior of this product depends on the *spectral radius* $\rho(J_t)$ — roughly the largest singular value of each Jacobian. Three regimes:

- $\rho(J_t) > 1$ consistently: the product norm grows exponentially with $T$. Gradients *explode*.
- $\rho(J_t) < 1$ consistently: the product norm shrinks exponentially. Gradients *vanish*.
- $\rho(J_t) \approx 1$: gradients can still be useful, but the product is hard to control precisely.

The gradient of the loss with respect to the initial state is

$$
\frac{\partial \mathcal{L}}{\partial s_0} = \frac{\partial \mathcal{L}}{\partial s_T} \cdot J_T \cdot J_{T-1} \cdots J_1.
$$

For our task, $s_0$ contains $v_0$ (the parameter), and $\mathcal{L}$ is the squared displacement of the final center of mass. The gradient $\partial \mathcal{L}/\partial v_0$ is what Adam receives; it travels through all $T$ Jacobian factors on the way back from the loss to $v_0$.

In deep networks this problem is addressed with architecture changes (residual connections, gating) or by normalising each layer's Jacobian. In differentiable simulation the options are: shorter horizons, gradient clipping, checkpointing, or fixing whichever single step has an ill-conditioned Jacobian.

**Why this matters for the world-models vision.** A gradient through a long physics rollout is the mechanism that lets you author future state by adjusting present parameters. If those gradients blow up, the authoring mechanism breaks. Understanding and taming them is not optional.

### P2 — The adjoint method and backward amplification

In source-transformation reverse-mode autodiff (which Taichi uses), the backward pass computes

$$
\bar{x}_{\text{in}} \mathrel{+}= \bar{x}_{\text{out}} \cdot \frac{\partial f}{\partial x_{\text{in}}}
$$

for each primitive operation $x_{\text{out}} = f(x_{\text{in}})$. The bar notation ($\bar{x}$) means $\partial \mathcal{L}/\partial x$.

For the grid step, the key primitive is

$$
\mathbf{v}_{\text{out}} = \frac{\mathbf{v}_{\text{in}}}{m}.
$$

Its backward is

$$
\bar{\mathbf{v}}_{\text{in}} \mathrel{+}= \frac{\bar{\mathbf{v}}_{\text{out}}}{m}, \qquad \bar{m} \mathrel{+}= -\frac{\mathbf{v}_{\text{in}} \cdot \bar{\mathbf{v}}_{\text{out}}}{m^2}.
$$

Both terms scale as $1/m$ or $1/m^2$. When $m$ is near zero, the backward amplifies whatever gradient signal is passing through by an enormous factor. This is the near-zero mass singularity, and it is distinct from the Jacobian-product amplification described above, even though they compound.

---

## Failure mode 1 — NaN gradient after convergence

On the first DiffMPM run the optimization converged cleanly, with loss falling from $0.144$ to about $6\times10^{-5}$ by iteration 29. Then at iteration 32 the **backward pass produced a NaN while the forward loss stayed finite** at about $4.6\times10^{-4}$. A guard in the optimizer caught the non-finite gradient and stopped. The forward staying finite while the backward blows up is the tell that the problem is in the sensitivities, not the state.

Three mechanisms are the leading suspects, and they are not exclusive.

- **Out-of-domain particles hitting the wall branch.** Once $v_0$ is large, particles can reach the boundary band. The wall condition zeroes the inward normal velocity with a hard branch, and the base-index floor is non-smooth. Gradients through a branch like that are ill-defined right at the kink, so a particle sitting on the boundary can inject a meaningless sensitivity. This is the contact-differentiability story, seeded from `sim/mpm88.py` lines 49 to 59 and 77.
- **Division by near-zero grid mass.** The grid step divides momentum by node mass, and its backward sensitivity scales like $1/m_i^2$. A node barely grazed by one particle has a tiny mass, so it can amplify a gradient by a huge factor even when the forward velocity looks ordinary.
- **Long-rollout amplification.** Five hundred chained steps form a long product of Jacobians. A single ill-conditioned step can be magnified into an overflow by everything downstream of it, the simulation analogue of an exploding gradient in a deep recurrent net.

---

## What the experiments showed

Five experiments were run on the `long-rollout-pathologies` branch to turn this from a hypothesis list into an evidence-based explanation. Here is what each one found.

### Experiment 1 — Gradient instrumentation

The optimizer was run for up to 80 iterations with 512 steps at f32 precision. The NaN first appeared at **iteration 7** (iteration count varies slightly with initial particle seed; the baseline branch saw it at iterations 32 and 67 for different seeds — same mechanism, same regime).

An important implementation note surfaced here: **Taichi's reverse-mode autodiff does not persist intermediate gradients** ($v.\text{grad}[t]$, $x.\text{grad}[t]$) after the tape exits. Only the leaf-input gradient ($v_0.\text{grad}$) is reliably available. The backward kernels compute intermediate gradients transiently and discard them. Per-step gradient norms must therefore be inferred from the forward-pass state (grid mass statistics) rather than read from gradient fields after the tape.

The forward pass data tells the key story: the minimum non-zero grid-node mass at step 0 (just the first time step) was **$1.115 \times 10^{-12}$**, with 511 out of 512 steps having at least one such near-zero-mass node. The backward gradient amplification for a node with $m = 10^{-12}$ is $\bar{v}_{\text{in}} \approx \bar{v}_{\text{out}} / 10^{-12} = 10^{12} \cdot \bar{v}_{\text{out}}$. Even if $\bar{v}_{\text{out}}$ starts at $O(0.1)$, the result is $O(10^{11})$, well above the float32 maximum of $\approx 3.4 \times 10^{38}$... unless $\bar{v}_{\text{out}}$ at that node happens to be essentially zero.

The NaN does not appear in iteration 0 because at low $|v_0|$ the particles stay near the center of the domain and the node with $m = 10^{-12}$ is at the fringe of the cloud where the final-position gradient $\partial \mathcal{L} / \partial v_{\text{grid}}$ is also near zero. As the optimizer increases $|v_0|$ over successive iterations, particle trajectories sweep further, the fringe nodes start carrying real gradient signal, and eventually $10^{12} \times (\text{non-negligible gradient}) > 3.4 \times 10^{38}$.

### Experiment 2 — Horizon sweep

| Rollout length | NaN first at iteration | Final loss (if no NaN) |
|---|---|---|
| 128 steps | never (80 iters) | $3.1 \times 10^{-2}$ |
| 256 steps | iter 11 | — |
| 512 steps | iter 7 | — |

The trend is unambiguous: longer rollouts fail earlier. This is the fingerprint of long-rollout amplification (hypothesis 3). With 128 steps the Jacobian product is short enough that even a near-zero mass event at one step doesn't overflow; the gradient magnitude at $v_0$ is controlled. With 512 steps, the same event propagates backward through four times as many Jacobian factors and crosses the f32 ceiling.

### Experiment 3 — Precision comparison (f32 vs f64)

Running the same 512-step optimization in **float64** produced **no NaN in 80 iterations**, with the loss reaching $4.5 \times 10^{-5}$. The f64 threshold for overflow is $\approx 1.8 \times 10^{308}$ vs. $3.4 \times 10^{38}$ for f32 — 270 orders of magnitude larger. A gradient that overflows f32 at iteration 7 would have to grow by another $\approx 270$ orders of magnitude to overflow f64, which never happens here.

This is the decisive evidence that the NaN is a **floating-point overflow, not a true mathematical singularity**. The quantity $\bar{v}_0 = \partial \mathcal{L} / \partial v_0$ is mathematically finite; it is just larger than f32 can represent.

### Experiment 4 — Contact isolation

Two variants were compared: a *center* target $[0.5, 0.5]$ (the blob barely needs to move, particles stay away from walls), and a *wall* target $[0.08, 0.35]$ (particles driven toward the left boundary).

| Target | NaN at iteration |
|---|---|
| Center $[0.5, 0.5]$ | 3 |
| Wall $[0.08, 0.35]$ | 32 |

The center target fails **earlier**, not later. This rules out wall contact as the primary driver. If contact were the root cause, the wall target would fail first. Instead, the center target fails at iteration 3 even though particles never approach the boundary, which points back to the near-zero mass singularity as the cause.

The wall target surviving longer is likely because driving the blob leftward (away from its natural fall trajectory) requires a very different optimizer path; the particular $(v_0, \text{trajectory})$ pairs that create catastrophic near-zero mass nodes happen to be reached later.

### Experiment 5 — Mass stabilisation

The fix is surgical: replace the `if m > 0` branch in `grid_op` with a stable division:

```python
# Original (baseline):
if m > 0:
    vel = grid_v_in[f, i, j] / m

# Stabilised (sim/diffmpm_pathologies.py, grid_op_stable):
vel = grid_v_in[f, i, j] / ti.max(m, eps)   # eps = 1e-4 by default
```

With `eps = 1e-4`, the maximum backward amplification per step is $1 / 10^{-4} = 10^4$, regardless of how tiny $m$ becomes. The result: **zero NaN in 100 iterations**, with loss dropping from $0.152$ all the way to $9.5 \times 10^{-6}$, far past the old failure point.

For nodes where $m \gg \text{eps}$, the division is unchanged and the physics is exact. For barely-grazed fringe nodes where $m < \text{eps}$, the stabilisation says: *these nodes carry negligible physical content, so cap their backward influence*. This is a reasonable regularisation, not a distortion of the simulation.

---

## Mechanistic explanation

Putting the experiments together, here is the full causal chain:

1. The optimizer increases $|v_0|$ each iteration to move the blob toward the target. Larger velocity means particles travel further per timestep.

2. As particles sweep through the grid, the spline kernel $w_{ij}$ assigns most of a particle's weight to its three nearest nodes in each direction, but also assigns a small weight to nodes at the edge of the $3 \times 3$ stencil. When a particle is $\approx 1.5$ grid cells away from a node, that node receives weight $w \approx 0.5 \times (1.5 - 1.5)^2 = 0$, and the minimum non-zero weight is $w_{\min} = 0.5 \times \epsilon_{\text{interp}}^2$ for tiny fractional offsets $\epsilon_{\text{interp}}$. In practice, grid-node masses as small as $10^{-12}$ are observed from the first iteration onward (511 out of 512 steps have such nodes).

3. These near-zero mass nodes are harmless in the forward pass: they receive nearly zero momentum and produce nearly zero grid velocity. In the backward pass, however, Taichi computes $\bar{v}_{\text{in}} += \bar{v}_{\text{out}} / m$. A node with $m = 10^{-12}$ amplifies the gradient by $10^{12}$.

4. For the first several iterations, the gradient at these fringe nodes happens to be near zero too (the loss is not yet sensitive to them), so the amplification $10^{12} \times 0 \approx 0$ is harmless. As $|v_0|$ grows and trajectories shift, eventually a step exists where a near-zero mass node coincides with a node through which the loss gradient is non-negligible.

5. At that point, $\bar{v}_{\text{in}} \approx 10^{12} \times O(10^{-3}) = O(10^9)$. The backward Jacobians for the remaining steps then carry this $O(10^9)$ signal back through $\lesssim 512$ more matrix multiplications. The result overflows float32 and becomes NaN, which propagates to $\bar{v}_0$.

The gradient is not meaningless — it just exceeds the representational range of float32. The f64 experiment confirms it: the same computation in float64 gives a finite, correct gradient throughout.

---

## Fixes and trade-offs

### Fix 1 — Mass stabilisation (recommended)

Replace `if m > 0: vel = v_in / m` with `vel = v_in / max(m, eps)`. This caps per-step amplification at $1/\text{eps}$ and is cheap (one extra comparison per grid node). The trade-off: gradients through barely-grazed nodes are now approximate rather than exact. For optimization this is fine because fringe nodes have negligible physical weight anyway.

Implementation: `grid_op_stable` in `sim/diffmpm_pathologies.py`. Enable with `--stabilize`.

### Fix 2 — Float64 precision

Switching `ti.init(default_fp=ti.f64)` raises the overflow threshold by $\approx 270$ orders of magnitude, eliminating the NaN for the 80-iteration budget tested here. The cost is 2× GPU memory and slower compute. This does not fix the singularity — it just pushes the failure further away. For longer-horizon or longer-run experiments, f64 will eventually also NaN.

### Fix 3 — Shorter horizon

At 128 steps, the rollout never NaN'd in 80 iterations. Shorter horizons reduce both the number of Jacobian products and the exposure to near-zero mass nodes in the backward. The cost: the optimizer cannot plan as far into the future. This is a fundamental constraint on the class of tasks you can solve with gradient descent.

### Fix 4 — Gradient clipping with NaN-skip (partial)

Simple gradient clipping on $v_0$ after the backward does not help when the backward itself is already NaN. The NaN propagates before reaching $v_0.\text{grad}$. The NaN-skip strategy (skip the Adam update on NaN iterations) stalls optimization because $v_0$ never moves away from the NaN-producing configuration. 

The correct placement for clipping is at the **source** of amplification — the per-step grid gradient — which requires either modifying the backward kernel directly (non-trivial in Taichi's autodiff) or proxying it via mass stabilisation (Fix 1 above). Clipping $v_0.\text{grad}$ is a last resort after the backward successfully completes, not a fix for the backward itself.

### Fix 5 — Gradient checkpointing

Checkpointing reduces the memory footprint of long-horizon differentiation by recomputing some intermediate states during the backward pass rather than storing them all. It does not address the gradient magnitude problem, but it enables much longer horizons within a fixed GPU memory budget. For a $T$-step rollout split into $\sqrt{T}$ segments of $\sqrt{T}$ steps each, memory scales as $O(\sqrt{T})$ at a cost of $O(T \log T)$ compute. Implementation requires manually managing the tape, which goes beyond Taichi's `ti.ad.Tape`. The trade-off is documented here for completeness; the actual implementation is left as a next direction.

---

## Open questions

The near-zero-mass overflow is understood and fixed. What remains open is genuinely intellectual rather
than a checklist. Whether a smooth wall contact would further improve gradient quality (the isolation
experiment says contact is not the *cause* of the NaN, but a hard zero is still a non-smooth kink in the
gradient), and whether the near-zero-mass amplification re-emerges as the dominant problem at much longer
horizons once stabilisation is in place. Concrete experiments toward these live in
`coordination/directions/`, not here. This document teaches what is known; it does not plan work.
