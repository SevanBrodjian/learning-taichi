# Differentiable materials: finite is not the same as meaningful

The [[material-showcase]] page shows fluid, elastic, and snow moving correctly in the forward direction.
The natural next question for a controllable world model is whether those same three materials can be
*steered* by gradients, and [[differentiating-the-rollout]] already built the machinery to try: a
time-indexed rollout wrapped in an autodiff tape, optimizing a control so a loss goes down. The trap is
that "the loss went down" is not evidence the gradient is correct. A gradient can be perfectly finite,
drive a descent that looks like progress, and still be **wrong** or so weak it is useless. Distinguishing a
gradient that is merely finite from one that is *meaningful* is the single most important habit in
differentiable simulation, and this page is about how to actually check it and what the check reveals about
the three materials.

## The bar: a gradient is meaningful only if it matches finite differences

Autodiff gives a number for $\partial L / \partial \theta$, the sensitivity of a scalar loss $L$ to a
control $\theta$. That number is trustworthy only if it agrees with an independent estimate that does not
use the backward pass at all. The independent estimate is a **central finite difference**, which perturbs
the control by a small step $h$ in each direction and reads how the loss actually moves,

$$
\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + h) - L(\theta - h)}{2h}.
$$

Here $L(\theta + h)$ means running the entire forward simulation again with the control nudged up by $h$,
and $L(\theta - h)$ with it nudged down. The two-sided (central) form is used rather than the one-sided
$\big(L(\theta+h) - L(\theta)\big)/h$ because its leading error shrinks like $h^2$ instead of $h$: the
odd-order terms of the Taylor expansion cancel, so for the same $h$ the central estimate is far closer to
the true derivative. Crucially, the finite difference touches **only the forward pass**. If it agrees with
the autodiff number, the backward pass has been validated against ground truth, because the one thing they
do not share is the adjoint. This is exactly the check the earlier material-variants attempt never ran, and
its absence is how a finite-but-wrong gradient passed for a working one.

The quality of the match is summarized by the relative error

$$
\varepsilon = \frac{\lvert g_{\text{ad}} - g_{\text{fd}} \rvert}{\lvert g_{\text{fd}} \rvert},
$$

where $g_{\text{ad}}$ is the autodiff gradient and $g_{\text{fd}}$ the finite-difference estimate. A
gradient earns the label **meaningful** only when it is finite and $\varepsilon$ is small, a few percent or
less. Everything below uses a threshold of five percent.

### The step size $h$ has a sweet spot

The finite difference is not free of error, and the error has two competing sources that make $h$ a
parameter with a genuine best value, not a "smaller is better" knob. Too **large** an $h$ and the $O(h^2)$
truncation error dominates: the secant over a wide interval is not the tangent. Too **small** an $h$ and the
subtraction $L(\theta+h) - L(\theta-h)$ cancels almost all of its own significant digits, so floating-point
roundoff and any run-to-run noise in the loss dominate. On a GPU that second floor is real and worth naming:
the atomic-add accumulation in particle-to-grid scatter is not bitwise reproducible, so evaluating $L$ twice
at the *same* control returns values that differ at roughly the $10^{-7}$ relative level. A useful $h$ sits
above that noise floor and below the curvature scale, and the honest way to report a check is to sweep a few
$h$ and confirm the estimate is stable across them rather than trusting a single value. When a control
barely affects the loss, its true gradient is tiny, and $h$ has to be pushed large just to lift the signal
above the noise, which is the situation for stiffness in a fluid below.

## What the check finds: all three materials pass

Running that check on a short rollout (a couple hundred steps, a single dropped disk of a couple thousand
particles, each material at its own stable timestep) gives a clean result. For a center-of-mass-to-target
loss, the autodiff gradient with respect to both components of the initial velocity **and** with respect to
Young's modulus $E$ matches central finite differences to well under the five-percent bar for every
material, usually to a fraction of a percent.

![Finite-difference gradient-check table for the three materials. Each row is one material and one control,
listing the autodiff gradient, the central finite-difference estimate, their relative error, and a pass or
fail verdict. Every row passes, with relative errors ranging from a few parts in ten thousand to about one
percent. The three E rows are the strict test of the solid stress, since the stiffness gradient flows
entirely through the constitutive law.](/api/data/learning-taichi/runs/material-variants/fluids-snow-and-solids-as-differentiable-simulations/gradcheck_table.png)

The velocity controls are a weak test of the constitutive model, because a blob's center of mass follows
nearly the same ballistic arc no matter what it is made of, so those gradients mostly exercise the transfer
and integration, not the stress. The **stiffness** control is the strict test. The loss depends on $E$ only
through the stress each material develops (the single slot the [[constitutive-models]] page is built
around), so $\partial L / \partial E$ has to travel back through the entire constitutive law to reach the
control. For the fluid that is a pressure term in $J$; for the elastic
solid it is the corotated stress, which is built from the singular value decomposition of the deformation
gradient (the rotation $R = UV^{\top}$ from [[svd-polar]]); for snow it is that same SVD **plus** the
plastic clamp on the singular values. Every one of those stiffness gradients finite-difference-verifies.
That is direct evidence the differentiable constitutive backward, SVD and clamp included, is correct, not
just the easy ballistic part.

The snow result deserves emphasis because it is the one most expected to break. At the tested throw, close
to two-fifths of all singular values across the rollout are pinned at the plastic band, meaning the clamp is
firing constantly, not sitting dormant. The clamp is only $C^0$, a genuine kink in the map, exactly the
non-smoothness [[svd-polar]] and [[failure-modes]] flag as a hazard. And yet the gradient still matches
finite differences. The kink is real, but on this task it does not corrupt the gradient, it just means the
autodiff engine returns a valid one-sided slope wherever a particle sits on a boundary, and the finite
difference lands on essentially the same value.

## Why the SVD and the clamp do not break it here

Two specific fears about the solid path turn out to be milder in practice than in theory, and both are worth
understanding rather than just noting.

The first is the SVD **degeneracy** at coincident singular values. When a blob is undeformed its deformation
gradient is the identity, $F = I$, so $\sigma_1 = \sigma_2$, the principal directions are not unique, and the
derivatives of the rotation factors $U$ and $V$ genuinely blow up (the mechanism is derived in
[[svd-polar]]). The mitigation is partly the library and partly the physics. Taichi's `ti.svd` backward
regularizes the degenerate case internally, so it returns a finite gradient (with an internal warning)
rather than a NaN, and evaluating the gradient at a state that has actually deformed moves off the
degenerate point entirely, since a blob in motion has $\sigma_1 \neq \sigma_2$ almost everywhere. The
degeneracy is a measure-zero set the trajectory does not linger on.

The second is the plastic **clamp** kink already discussed. A $C^0$ map has a well-defined derivative
everywhere except on the seam, autodiff picks a one-sided value on the seam, and as long as the loss is not
dominated by particles sitting exactly on the boundary, the assembled gradient is a faithful subgradient.
Stacking many such kinks across a long rollout is what would eventually roughen the loss landscape, but
roughening the landscape is a statement about how hard the *optimization* is, not about whether the gradient
at a given point is correct. Those are different claims, and the finite-difference check speaks only to the
second.

## The real failure mode is a blown-up forward, not a subtle adjoint

The way gradients actually go bad in this solver is blunt. If the timestep violates the CFL stability limit
for the material's stiffness, the forward simulation itself blows up. Elastic waves travel at speed
$c = \sqrt{E/\rho}$, and an explicit step stays stable only while it is shorter than the time a wave needs to
cross a grid cell,

$$
\Delta t \lesssim \frac{\Delta x}{c} = \frac{\Delta x}{\sqrt{E/\rho}},
$$

so the stable timestep scales like $\Delta t_{\max} \sim 1/\sqrt{E}$, smaller for stiffer materials, and
smaller still for snow whose clamp adds its own stiffness (the same scaling [[material-stiffness]] follows
into the numerics). Push $\Delta t$ past that limit and particle velocities amplify each step, positions run
off to infinity, and the quadratic interpolation stencil starts indexing grid nodes that do not exist. The
gradient of a simulation whose particles have flown off the grid is meaningless, when it is not an outright
out-of-bounds crash. This is almost certainly the "erroneous particle behavior" seen in earlier attempts:
not a defect in the differentiation, but an unstable forward that no adjoint can rescue. The fix is not a
gradient trick, it is a stable timestep chosen per material, which is why the working settings use a smaller
$\Delta t$ for the stiffer elastic solid and the smallest for snow. A correct forward is a precondition for a
meaningful gradient, and confirming the rollout stays finite and physical, by eye and by an
`isfinite` check, comes *before* trusting any sensitivity it produces. The near-zero grid-mass overflow that
[[failure-modes]] dissects is the other precondition, handled here by the same mass floor.

## Correctness is not usability, and the difference is the whole point

With verified gradients in hand, a plain Adam descent on the throw task drives the loss down and moves the
control **off the origin** for all three materials, the fluid reaching the target outright and the solid and
snow throwing toward it. This is the sharp contrast with the earlier attempt, whose elastic optimum barely
left the starting velocity and was read as a broken gradient. It was not broken. The gradient was correct
the whole time; two other things were going on. A center-of-mass loss barely depends on the constitutive
model, so the material-specific part of the gradient is small to begin with, and each solid ran at a smaller
stable timestep, so within a fixed step budget its rollout covered less physical time and the blob simply
could not be thrown as far. Both are limits on how *usable* the gradient is on this particular task, not on
whether it is *correct*.

Holding those two ideas apart is the real lesson. **Correctness** is a local property of the adjoint,
certified by a finite-difference check at a point. **Usability** is a global property of the loss landscape
and the task, how smooth it is, how strongly it depends on the control, how far a stable rollout can reach.
The interesting materials, the ones that store shape and yield and crumple, are the ones whose stress laws
carry the most state and the most non-smoothness, and the honest expectation is that their gradients stay
correct while their landscapes get rougher and their usable step sizes shrink. A controllable, differentiable
world model lives or dies on that second property, but it can only be studied once the first is nailed down,
and the only way to nail it down is to stop trusting that the loss went down and check the gradient against
finite differences.

## What is open

The check here certifies correctness on a center-of-mass loss, which is the loss least sensitive to the
constitutive model. The natural next probe is a loss that strongly excites internal deformation, a
shape-matching or contact task, where the elastic and snow stress gradients would be exercised hard and the
snow clamp would couple into the loss directly. The prediction is that those gradients stay
finite-difference-correct while the snow landscape visibly roughens, forcing a smaller step or a smoothed
clamp, so that the correctness-versus-usability split shows up as an optimization difficulty rather than a
wrong number. Whether a softened clamp meaningfully smooths that landscape without distorting the physics is
a real question, untested here only because the hard clamp already verified on this task. So is whether the
near-identity SVD degeneracy ever bites in a task that deliberately holds a blob undeformed while
differentiating through it.
