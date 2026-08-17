# Constitutive models: the stress slot and the gradient

Every material in MLS-MPM enters the simulation through **one slot**: the internal-stress term of the
affine-momentum matrix $A_p$ that P2G scatters to the grid. The [[mls-mpm-forward]] step wrote that matrix
as

$$
A_p = \underbrace{V_p\,\sigma_p}_{\text{internal stress}} \;+\; \underbrace{m_p\,C_p}_{\text{affine velocity}},
$$

where $V_p$ is the particle volume (its share of material, fixed at initialization), $C_p$ is the affine
velocity matrix carried between steps, and $\sigma_p$ is the **stress** the material develops in response
to how it has been deformed. The forward page used the simplest possible choice for $\sigma_p$, a fluid
pressure. That choice is the **constitutive model**, and it is the only thing this page changes. Swapping
it turns the same solver into water, rubber, or snow. The point worth carrying away is that the
constitutive model controls two things at once: the **physics** (how the blob holds together or falls
apart) and the **gradient** (how smooth the loss is in whatever parameter is being optimized). The second
is where differentiable simulation lives or dies, and it is the less obvious of the two.

This matters directly for controllable simulation. A world model that can be *steered* by gradients only
exposes the controls whose loss landscape is smooth enough to descend. The constitutive model decides how
smooth that landscape is, so it decides which materials are easy to control and which fight back.

Notation follows the [[math-toolkit]]: particle index $p$, deformation gradient $F_p$, its determinant
$J = \det F_p$ (the volume ratio), Young's modulus $E$ (overall stiffness, whose single-parameter effects on
the physics, the timestep, and the gradient are traced in [[material-stiffness]]), Poisson ratio $\nu$, and the
Lamé parameters $\mu$ (shear stiffness) and $\lambda$ (volumetric stiffness), built from $E$ and $\nu$ by

$$
\mu = \frac{E}{2(1+\nu)}, \qquad \lambda = \frac{E\,\nu}{(1+\nu)(1-2\nu)}.
$$

$\mu$ is the resistance to **shear** (changing shape at constant volume), $\lambda$ the resistance to
**volume change**. A fluid has effectively $\mu = 0$ (it cannot resist shear, which is why it flows); a
solid has both.

## The deformation gradient $F$, the object a material remembers

A fluid needs to remember only one number, how compressed it is, captured by $J$. A solid needs to
remember its full shape change, captured by the **deformation gradient** $F_p$, a $2\times 2$ matrix (in
2D) that maps a tiny material vector in the rest shape to what it has become now. $F = I$ means undeformed;
$\det F = J$ recovers the volume ratio; the non-symmetric and off-diagonal parts of $F$ record stretch,
shear, and rotation that a scalar $J$ throws away. Each step the affine velocity field stretches it,

$$
F^{\,t+1}_p = (I + \Delta t\, C_p)\, F^{\,t}_p,
$$

which says the local velocity gradient $C_p$ (how fast neighboring bits of material move apart) advances
the deformation over one timestep $\Delta t$. The whole difference between fluid and solid is whether the
state carries $J$ alone or the full $F$.

## Fluid: pressure from volume change, no memory of shear

The fluid stress depends only on the current volume ratio,

$$
\sigma^{\text{fluid}}_p = -k\,(J_p - 1)\, I, \qquad k = 4 E,
$$

an isotropic pressure (the same in every direction, hence the identity $I$). When $J > 1$ the material has
expanded and the stress pulls back in; when $J < 1$ it has been squeezed and the stress pushes out. The
factor $k = 4E$ sets how stiff that restoring push is. There is no $F$, so the fluid **forgets shear
history** entirely. Two parcels that arrive at the same volume develop the same stress regardless of how
they were sheared to get there. This is exactly why a fluid blob disperses freely instead of springing
back to a shape.

Because the stress is a smooth (in fact linear) function of $J$, and $J$ itself evolves smoothly through
$J^{t+1} = J^t(1 + \Delta t\,\operatorname{tr} C_p)$, the fluid's per-step map is about as smooth as MLS-MPM
gets. That smoothness is the reason the fluid throw task in the [[differentiating-the-rollout]] page
optimized so cleanly.

## Elastic: corotated stress from $F$, recoverable and still smooth

An elastic solid resists shape change and springs back, so its stress must be a function of the full $F$.
The **corotated** (fixed-corotated) model is a standard, well-behaved choice. Its first Piola–Kirchhoff
stress is

$$
P(F) = 2\mu\,(F - R) + \lambda\,(J-1)\,J\,F^{-\top},
$$

and the Cauchy-stress contribution P2G actually needs is $P\,F^{\top}$, scaled by the same MLS-MPM
prefactor as the fluid. Each term has a clean meaning. The first, $2\mu(F - R)$, is the **shear-restoring**
term: $R$ is the pure **rotation** part of $F$, and $F - R$ is everything that is *not* a rotation, i.e.
the genuine stretching and shearing. Subtracting $R$ is what makes the model *corotated*, so a particle
that merely spins develops no stress, only one that actually changes shape does. The second term,
$\lambda(J-1)J F^{-\top}$, is the **volume-restoring** term, the solid analogue of the fluid pressure,
penalizing $\det F \ne 1$.

The rotation $R$ comes from the **polar decomposition** $F = R S$, most stably obtained from the singular
value decomposition $F = U \Sigma V^{\top}$ as $R = U V^{\top}$. (In 2D both have closed forms, and
Taichi's `ti.svd` is differentiable, which is what lets the gradient flow through this stress at all. The
SVD and polar decomposition get a prerequisite of their own in [[svd-polar]], including why $R = U V^{\top}$
is the stable way to extract the rotation and where the derivative misbehaves.) The cost of all this richness:
$P(F)$ is a more sharply curved, larger-magnitude function of the state than the fluid pressure, so the
gradient of a loss back to a control parameter is **larger and more variable**. It is still smooth (the SVD
is differentiable away from coincident singular values), so the loss is still descendable, but with a
**smaller usable step size**. A learning rate that suits the fluid will overshoot here.

## Snow: a non-smooth plastic clamp that roughens the landscape

Snow is elastic until it is pushed too far, then it **breaks and stays broken**, a behavior called
plasticity. The Stomakhin snow model adds two pieces on top of the elastic stress. First, after the elastic
update, the singular values of $F$ are **clamped** into a permitted range,

$$
\hat\sigma_k = \operatorname{clamp}\big(\sigma_k,\; 1-\theta_c,\; 1+\theta_s\big), \qquad k = 1, 2,
$$

where $\sigma_k$ are the singular values from $F = U\Sigma V^{\top}$, and $\theta_c, \theta_s$ are the
**compression** and **stretch** limits (small numbers like $\theta_c \approx 2.5\times 10^{-2}$). Deformation
beyond the limit is removed from the recoverable elastic part $F \leftarrow U\hat\Sigma V^{\top}$ and pushed
into an accumulated **plastic** volume change $J_p^{\text{pl}}$. Second, the material **hardens** as it
compacts: the Lamé parameters are scaled by $\exp\!\big(\xi(1 - J_p^{\text{pl}})\big)$, so compacted snow
($J_p^{\text{pl}} < 1$) gets stiffer, with $\xi$ the hardening coefficient.

The hardening factor is smooth, but the **clamp is not**. `clamp` is piecewise: a singular value is either
inside the range (the clamp does nothing, derivative 1) or outside it (the clamp pins it, derivative 0),
and at the boundary the map has a **kink**, a point where the derivative jumps. The [[failure-modes]] page
made the general version of this argument for the hard wall: a one-sided clamp is $C^0$ but not $C^1$, and
right at the corner the gradient the autodiff returns is a poor predictor of the true loss change.
Snow applies such a clamp **per particle, per step**, so a long rollout stacks up many of these kinks, and
their net effect is a **rougher loss landscape** than the elastic or fluid case. Rougher means the gradient
direction is noisier and the largest stable learning rate is smaller still.

## Sand: a yield surface that depends on pressure

Snow and sand are both plastic, and comparing them is the fastest way to see what a plastic model
actually *is*. A plastic model is a rule for **which deformations the material refuses to remember**.
Everything else follows from the shape of that admissible set.

Snow's admissible set is a **fixed box** on the principal stretches, the clamp above. It does not depend
on how hard the material is being squeezed. That means snow has **cohesion**: an unloaded block of it
can hold a vertical wall, because at zero pressure the box is exactly as large as it is under load.

Sand has no cohesion. Its strength comes from grains rubbing against each other, and friction needs
normal force, so **the shear a granular pack can carry is proportional to the pressure it is under**. At
zero pressure it can carry nothing, which is why dry sand cannot stand a vertical wall and why a pile of
it slumps to a characteristic slope instead. Any model that gets sand right has to make the admissible
set shrink to a point as pressure goes to zero, and that is a **cone**, not a box.

Writing that cone down is where the [[svd-polar]] log strain earns its keep. Work in Hencky strain
$\varepsilon = \ln\Sigma$, split it as in [[linear-algebra]] into its trace (volume) and deviator
(shape) $\hat\varepsilon = \varepsilon - \tfrac{1}{d}(\operatorname{tr}\varepsilon) I$, and the
Drucker-Prager criterion is one line,

$$
\lVert \hat\varepsilon \rVert \;\le\; -\frac{d\lambda + 2\mu}{2\mu}\,\alpha\, \operatorname{tr}\varepsilon,
\qquad \operatorname{tr}\varepsilon \le 0 ,
$$

with $d$ the spatial dimension, $\lambda$ and $\mu$ the Lamé parameters, and

$$
\alpha \;=\; \sqrt{\tfrac{2}{3}}\;\frac{2\sin\varphi}{3 - \sin\varphi}
$$

the cone slope set by a single parameter, the **internal friction angle** $\varphi$ of the pack. Read the
inequality from the right: $-\operatorname{tr}\varepsilon$ is how compressed the material is, so the
right-hand side is the shear budget that compression buys. Turn $\varphi$ up and the cone opens, the
budget grows, and the material holds a steeper pile. Turn it to zero and $\alpha = 0$, the budget
vanishes, and what is left is a material with no shear strength at all, which moves like a liquid that
additionally cannot be pulled. The factor $(d\lambda+2\mu)/(2\mu)$ is the conversion from a condition on
stress, where the criterion is physically stated, to the condition on strain used here, and it is exactly
the ratio by which the Hencky stress law maps volumetric strain into pressure relative to shear.

The condition $\operatorname{tr}\varepsilon \le 0$ is the **no-tension** rule, and it does as much
visible work as the friction term. Grains can push on each other and cannot pull, so any particle that
finds itself in net extension has no business carrying stress at all.

The step that enforces this is a **return mapping**: integrate the deformation as if the material were
elastic, then, if the trial state sits outside the cone, put it back. Three cases, in order of how much
they change:

1. $\operatorname{tr}\varepsilon \ge 0$, the material is in tension. Project to the **tip** of the cone,
   $\varepsilon \leftarrow 0$, releasing all stored stress. The volumetric strain thrown away is
   accumulated in a running plastic record, so the same amount of compression has to be re-applied
   before the pack carries load again. Without that bookkeeping the material silently gains volume every
   time it is thrown apart and never re-packs.
2. Inside the cone. Nothing happens, the step was elastic.
3. Outside. Slide the log strain back onto the cone **along the deviatoric direction**, keeping the
   volumetric part fixed. Keeping it fixed is a deliberate modelling choice, and it says the material
   does not dilate when it shears.

Structurally this is the same kind of object as the snow clamp, a projection applied per particle per
step, so the remarks above about kinks and roughened gradient landscapes carry over. The differences are
that the projection surface is a cone rather than a box, and that the cone has a genuine **tip**, a point
where the surface is not smooth in any direction rather than merely creased along a face. Nothing here
has been measured against a gradient-based control task, so that is a structural expectation and not a
result.

What the model produces is the behaviour a person recognises immediately, a heap that stands at a finite
slope and runs out when it is pushed past it. [[material-showcase]] shows all four materials doing it
from the same starting shape.

## What changes, and what does not

The three models, applied to the same control task (throw a blob's center of mass to a target by
optimizing one shared initial velocity), separate cleanly into a part the constitutive model controls and
a part it does not.

![Three materials under one optimized throw. Fluid (green) has carried its center of mass to the target
(red +); elastic (blue) and snow (purple) have not traveled as far in the same iteration budget. Yellow
trail is the running center of mass.](/api/data/learning-taichi/runs/material-variants/fluid-vs-snow/triptych_frame.png)

What the constitutive model barely touches is the **ballistics of the center of mass**. Gravity is the
same for all three, and the initial velocity sets the trajectory of the average position, so at one shared
initial velocity all three blobs land their center of mass in nearly the same place. The center of mass is
*reachable* in every material; none of them hits a physics wall.

What the constitutive model strongly controls is **how easily gradient descent finds that reachable
target**. The smoother the stress law, the larger the learning rate the optimizer can use and the faster it
converges: the fluid descends with a large step, the elastic needs a much smaller one (its stiffer stress
makes the gradient large and Adam overshoots otherwise), and the snow needs care on both the timestep, to
keep the SVD clamp finite, and the step size. The single most transferable lesson is that **a shared
optimizer setting does not survive a change of material**: the learning rate that converges the fluid drives
the elastic case to diverge, because the same control parameter now sits on a stiffer, more curved loss
surface. Choosing the optimizer step is not separable from choosing the physics.

This is the sensitivity a controllable world model has to expose and tame: the constitutive model sets the
texture of the landscape gradients ride, smooth for a fluid, curved for an elastic solid, kinked for anything
with a plastic projection. The visually richest materials are exactly the ones whose non-smooth physics makes
them hardest to control.

## What's open

The clean separation above, ballistics reachable for all and controllability set by smoothness, is read from
a single near-ballistic task where the center of mass is essentially a point projectile. A task that forces
large **internal deformation** (shape matching, packing into a mold, sustained contact) would make the
constitutive differences bite far harder, since there the elastic energy and the plastic clamp shape the
optimized quantity directly rather than averaging out. Whether the smooth-to-kinked ordering sharpens on such
tasks, and whether a **smoothed** plastic clamp recovers a usable gradient for snow, are open, testable
questions.
