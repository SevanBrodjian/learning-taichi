# Failure modes

The teaching value of differentiable simulation lives here, in what breaks and why. Differentiating
through a physics rollout fails in ways that ordinary forward simulation never shows, because the failure
is in the *sensitivities*, not the state. This page builds up the first and most instructive of these
pathologies from the ground: a gradient that overflows to NaN while the simulation itself stays perfectly
finite. It grows as further pathologies are understood.

## Prerequisites for this page

Two ideas run through everything below. They are short; skim them if they are already familiar.

### Exploding and vanishing gradients in unrolled systems

Any system that applies the same kind of operation $T$ times in sequence is *unrolled*. A recurrent neural
network is the canonical example, and an MLS-MPM simulation with $T$ timesteps is another. Writing the
single-step update as a map $\phi$ on the full state $s_t$ (all particle positions, velocities, and grid
values),

$$
s_{t+1} = \phi(s_t),
$$

the sensitivity of the final state to the initial state is a product of per-step Jacobians, by the chain
rule:

$$
\frac{\partial s_T}{\partial s_0} = \underbrace{J_T \, J_{T-1} \cdots J_1}_{T\ \text{factors}},
\qquad J_t = \frac{\partial \phi(s_{t-1})}{\partial s_{t-1}}.
$$

The size of this product is governed by the per-step amplification, roughly the largest singular value of
each $J_t$. When that factor stays above one the product grows exponentially in $T$ and gradients explode;
when it stays below one the product shrinks exponentially and gradients vanish; near one the product is
usable but hard to control. The gradient of a scalar loss back to the initial state rides this same
product,

$$
\frac{\partial \mathcal{L}}{\partial s_0} = \frac{\partial \mathcal{L}}{\partial s_T}\, J_T \, J_{T-1} \cdots J_1,
$$

so a single ill-conditioned step anywhere in the rollout can be magnified by everything downstream of it
on the way back. Deep learning tames this with architecture (residual connections, gating) or Jacobian
normalization; differentiable simulation has shorter horizons, gradient clipping, checkpointing, or
repairing the one step whose Jacobian is ill-conditioned. This product-of-Jacobians structure is the
reason a long physics rollout is a genuinely harder object to differentiate than a short one, and it is
the mechanism that lets present parameters author future state, which is precisely the capability a
controllable world model needs.

### Backward amplification through a division

Reverse-mode autodiff propagates a loss sensitivity backward through each primitive operation. For an
operation $x_{\text{out}} = f(x_{\text{in}})$, the backward step accumulates

$$
\bar{x}_{\text{in}} \mathrel{+}= \bar{x}_{\text{out}}\,\frac{\partial f}{\partial x_{\text{in}}},
\qquad \bar{x} \equiv \frac{\partial \mathcal{L}}{\partial x}.
$$

The grid update's defining primitive is the division of momentum by node mass, $\mathbf{v} = \mathbf{p}/m$,
whose backward is

$$
\bar{\mathbf{p}} \mathrel{+}= \frac{\bar{\mathbf{v}}}{m}, \qquad
\bar{m} \mathrel{+}= -\frac{\mathbf{p}\cdot\bar{\mathbf{v}}}{m^2}.
$$

Both terms carry a $1/m$ or $1/m^2$ factor. When a node's mass is near zero, the backward multiplies
whatever sensitivity passes through it by an enormous number, regardless of how ordinary the forward
velocity at that node looked. This near-zero-mass amplification is distinct from the Jacobian-product
amplification above, and the two compound.

## The phenomenon: a NaN gradient with a finite forward pass

The signature of this failure is sharp. Optimization through the rollout converges normally for a while,
the loss falling smoothly, and then the **backward pass produces a NaN while the forward loss is still
finite and small**. A finite forward with a non-finite backward is the tell that nothing is wrong with the
simulated state; the blow-up is purely in the sensitivities. The same computation that produces a sensible
trajectory produces an unrepresentable gradient.

The cause is the near-zero-mass division, amplified by the long rollout. The quadratic B-spline kernel
gives most of a particle's weight to its nearest nodes but assigns a vanishingly small weight to nodes at
the edge of its stencil. In a typical rollout the smallest non-zero node masses reach the order of
$10^{-12}$, and almost every step has at least one such node. In the forward pass these fringe nodes are
harmless: they receive almost no momentum and produce almost no velocity. In the backward pass the same
node multiplies the sensitivity flowing through it by $1/m \approx 10^{12}$.

That amplification is not immediately fatal, which is why the failure arrives mid-optimization rather than
on the first step. Early on, the fringe nodes that carry a huge $1/m$ factor are also nodes the loss does
not yet depend on, so the amplified quantity is $10^{12} \times (\text{almost zero})$ and stays small. As
the optimizer pushes the control harder, particle trajectories sweep further across the grid, and
eventually a step exists where a near-zero-mass node coincides with a node through which the loss gradient
is non-negligible. At that step the backward produces something like $10^{12} \times 10^{-3} = 10^{9}$,
and the remaining Jacobian factors carry that $10^{9}$ back through the rest of the rollout until it
exceeds the float32 ceiling of about $3.4 \times 10^{38}$ and becomes NaN.

## What the evidence pins down

Three independent properties of the failure separate the real cause from the plausible-looking suspects,
and together they make the diagnosis airtight.

**It depends on horizon length, in the direction the Jacobian product predicts.** A long rollout fails and
a short one does not. At 512 steps the overflow appears within a few optimizer iterations; at 256 it
appears later; at 128 it does not appear at all over a long run. Longer rollouts fail *earlier*, which is
the fingerprint of the product-of-Jacobians amplification: more factors downstream of the bad step means
more multiplication of the same near-zero-mass event.

**It is a floating-point overflow, not a mathematical singularity.** Running the identical 512-step
optimization in float64 produces no NaN at all. Float64 raises the overflow ceiling from about
$3.4 \times 10^{38}$ to about $1.8 \times 10^{308}$, and a gradient that overflows float32 would have to
grow by hundreds of additional orders of magnitude to overflow float64, which it never does. The gradient
$\partial \mathcal{L}/\partial(\text{control})$ is mathematically finite the whole time; float32 simply
cannot represent it at the moment of the spike. This is the decisive distinction, because a true
singularity would survive the change of precision and this does not.

**Wall contact is not the driver.** It is tempting to blame the non-smooth wall clamp, since a hard
velocity zeroing is a genuine kink in the gradient. An isolation test rules it out: a target at the domain
center, where particles never approach a boundary, overflows *sooner* than a target pressed against a
wall. If contact were the cause, the wall case would fail first. It fails later, which points the blame
back at the near-zero-mass division, an effect present everywhere in the domain rather than only at the
walls. The wall kink remains a real and separate gradient hazard, but it is not what produces this NaN.

## The fix and the alternatives

The cause is a $1/m$ that is allowed to grow without bound, so the surgical fix bounds it. Replacing the
raw division with a floored one,

```python
# baseline:  vel = grid_v_in[f, i, j] / m
vel = grid_v_in[f, i, j] / ti.max(m, eps)      # eps = 1e-4
```

caps the per-step backward amplification at $1/\text{eps} = 10^{4}$ no matter how small a node's mass
becomes, and it removes the NaN entirely while letting the loss fall far past the old failure point. The
floor changes nothing for ordinary nodes, where $m \gg \text{eps}$ and the division is exact. It only acts
on barely-grazed fringe nodes, which carry negligible physical content, so capping their backward
influence is a reasonable regularization of the gradient rather than a distortion of the physics. This is
the recommended fix, and it lives in the stabilized grid step of the differentiable simulator.

The other levers trade off differently and are worth knowing as a toolbox. Float64 removes the overflow by
raising the ceiling, but it doubles memory and slows compute and only pushes the failure farther away
rather than removing the cause, so a long enough run will still reach it. A shorter horizon avoids both
the amplification and the exposure to fringe nodes, at the cost of how far ahead the optimizer can plan,
which is a real limit on the class of solvable tasks rather than a free win. Gradient clipping applied to
the control parameter does not help, because the NaN forms inside the backward before it ever reaches the
control; clipping is only meaningful once the backward completes, and the correct place to bound the
amplification is at its source, the per-step division, which is exactly what the mass floor does.
Gradient checkpointing is orthogonal: it recomputes intermediate states instead of storing them, which
buys much longer horizons within a fixed memory budget but does nothing about gradient magnitude.

## What is open

The near-zero-mass overflow is understood and fixed, and what remains is genuinely intellectual rather
than a checklist. A hard wall clamp is still a non-smooth kink even though it is not the cause of this
particular NaN, so whether a smooth contact model measurably improves gradient quality is a real question.
So is whether the near-zero-mass amplification re-emerges as the dominant problem at much longer horizons
once the mass floor is in place, since the floor bounds per-step amplification but the Jacobian product
keeps growing with the horizon. These are open because they teach something about the limits of the
method, not because they are unfinished work.
