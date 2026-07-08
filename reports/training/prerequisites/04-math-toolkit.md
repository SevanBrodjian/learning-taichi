# Math toolkit

> Prerequisite, skim-friendly once you have it. Each recurring math object behind the core sections,
> built from intuition first and only then written as a formula. If you have read [[mpm-in-context]] you
> already know the *roles* these objects play. This section gives them their precise form. The general
> linear algebra these objects rest on (outer products, trace, determinant, transpose) is in
> [[linear-algebra]], and the decompositions the solid models need are in [[svd-polar]]. Notation is
> stated locally. Particle index $p$, grid node $i$, timestep $\Delta t$, cell size $\Delta x$.

## Interpolation weights, the smooth splash

The first object is the weight that decides how much of a particle's mass and momentum lands on each
nearby grid node during the particle-to-grid transfer. Recall from [[mpm-in-context]] that a particle
does not dump everything on its closest node. It spreads its contribution across a small neighborhood,
heaviest right under itself and tapering off around it. The weight is the number that says exactly how
much each node gets.

Two properties are non-negotiable, and they tell you what the weight function has to look like before you
even pick one. First, the weights around a particle must **sum to one**. If a particle deposits weight
$0.5$ here and $0.3$ there and $0.2$ over there, the total is $1$, and that is what guarantees the
particle's full mass is deposited, no more and no less. Sum-to-one is conservation of mass written as a
constraint on the weights. Second, the weights must vary **smoothly** as the particle moves. If a
particle drifting a hair to the right caused its weight on some node to jump discontinuously, the whole
transfer would be non-differentiable right there, and the gradient story in the core sections would
collapse at every such jump. Smoothness in particle position is what keeps the simulator differentiable.

The standard choice that delivers both is the **quadratic B-spline**. A B-spline is a bump-shaped weight
built from polynomial pieces, smooth where the pieces meet, and compact, meaning it is exactly zero
outside a small region. The quadratic version touches a fixed $3 \times 3$ block of nodes around each
particle in 2D, which is why every loop in the code runs over a $3 \times 3$ stencil. Within that block,
let $d$ be the particle's offset from the nearest base node, measured in cell units, so $d$ is a fractional
number telling you where inside its cell the particle sits. The three per-axis weights are

$$
w_0 = \tfrac12\left(\tfrac32 - d\right)^2,\qquad
w_1 = \tfrac34 - (d-1)^2,\qquad
w_2 = \tfrac12\left(d - \tfrac12\right)^2.
$$

Each is a piece of a parabola, which is what "quadratic" means, and you can check by hand that
$w_0 + w_1 + w_2 = 1$ for any $d$, which is the sum-to-one property falling out of the construction. The
full 2D weight for a node, written $w_{ip}$ for node $i$ and particle $p$, is the product of the per-axis
weight in $x$ and the per-axis weight in $y$. That product structure is just the statement that the
splash is independent along each axis.

In the code these are the three-element lists named `w`, and you can see the same expressions in both
`sim/mpm88.py` line 40 and `sim/diffmpm.py` line 86. The base node is found with an integer floor, which
*is* a non-smooth operation, but it only selects *which* three nodes are involved. The weights
themselves, the actual numbers that get multiplied through, stay smooth in the particle position, and
that is what the gradient needs.

## Why a single velocity is not enough, and the APIC matrix

In [[mpm-in-context]] the intuition was that a single velocity per particle throws away the local
rotation and shear of the material's motion, and APIC repairs that by storing an extra matrix. Here is
the precise object.

The thing a single velocity misses is how the velocity *varies* across the small patch of material a
particle represents. "How a vector field varies across space" is exactly what a gradient captures, so the
missing information is the **velocity gradient**, a matrix whose entries say how each component of velocity
changes as you move in each direction. A pure rotation and a pure shear both show up in this matrix and
in neither the average velocity, which is why a plain velocity loses them. APIC stores an approximation of
this velocity gradient per particle, called the affine matrix $C_p$. "Affine" because a constant plus a
linear term, velocity plus velocity-gradient-times-offset, is an affine function of position, and that
affine field is a far better local description of the motion than a single constant velocity.

On the way back from the grid, $C_p$ is reconstructed by gathering the grid velocities weighted not just
by $w_{ip}$ but by $w_{ip}$ times the offset from the particle to each node. Weighting by the offset is
what extracts a *gradient* rather than an average, because it asks how the velocity differs between nodes
on one side and nodes on the other. The reconstruction is

$$
C_p = \frac{4}{\Delta x^2}\sum_i w_{ip}\,v_i\,(x_i - x_p)^{\top},
$$

where $v_i$ is the updated node velocity, $x_i$ is the node position, $x_p$ the particle position, and the
$(x_i - x_p)^{\top}$ factor makes each term an outer product, building a full matrix rather than a vector.
The constant $4/\Delta x^2$ in front looks arbitrary but is forced. It is the inverse of the second moment
of the quadratic B-spline, the particular number that makes the reconstruction *exact* for a genuinely
affine velocity field, so that if the true motion really is "translate plus a constant gradient," APIC
recovers it with no error. The same expression appears in code as the `new_C` accumulation in
`sim/mpm88.py` line 77 and `sim/diffmpm.py` line 131.

## Deformation as one number, the volume ratio $J$

[[mpm-in-context]] introduced deformation as the material's memory of its rest shape and noted that the
first model compresses that memory down to a single scalar, the volume ratio $J_p$. Here is what that
scalar is and how it moves.

The full record of deformation is a matrix called the deformation gradient, which tracks how a small
chunk of material has been stretched and sheared along every direction. Its determinant has a clean
physical meaning, the **local volume ratio**, how much the chunk's volume has changed relative to rest.
The first model keeps only this determinant and calls it $J_p$. As stated before, $J_p = 1$ is rest
volume, $J_p < 1$ is compressed, $J_p > 1$ is expanded. Keeping only the volume and discarding the rest
of the deformation is exactly the simplification that makes this a *weakly compressible* model rather than
a full elastic solid, and it is enough for a springy blob.

The update rule for $J_p$ comes straight from what the velocity gradient means. The trace of the velocity
gradient, the sum of its diagonal, is the **local rate of volume change**, the rate at which the material
around a point is expanding or compressing. Since APIC already hands you the velocity gradient as $C_p$,
the rate of volume change is just its trace, and stepping the volume forward by one small timestep gives

$$
J_p \leftarrow (1 + \Delta t\,\operatorname{tr} C_p)\,J_p.
$$

Read it in plain words. The new volume ratio is the old one scaled by one plus the timestep times how
fast the volume is currently changing. If the material around the particle is expanding, $\operatorname{tr}
C_p$ is positive and $J_p$ grows. If compressing, it shrinks. This is the `J[p] *= 1 + dt * new_C.trace()`
line in `sim/mpm88.py` line 82.

The stress, the push-back force that this deformation generates, is then a simple function of how far
$J_p$ has departed from $1$. The further from rest volume, the harder the material pushes to restore it,
and that push is what enters the momentum a particle deposits during the next transfer. The exact
expression and where it sits in the deposited momentum are assembled in [[mls-mpm-forward]], because it is
easier to see the stress term in the context of the full step than in isolation.

## Reverse-mode autodiff over an unrolled program

This is the piece your machine-learning background already owns, stated so the core sections can lean on
it. It is also the object that makes the whole project possible, so it is worth pinning down precisely.

A simulation rollout is a long composition of differentiable functions, one per step, threaded through a
shared state. That is structurally identical to an unrolled recurrent network or a very deep residual
network with shared weights, an analogy worth keeping because every instinct you have about training deep
recurrent models transfers here. The quantity you ultimately want is a single scalar loss, some measure
of how far the final state is from a desired outcome, and you want its gradient with respect to an input
buried at the very start of the rollout, such as the initial velocity.

**Reverse-mode automatic differentiation**, the same engine as backpropagation, computes that gradient by
walking the composition backward exactly once. Each operation in the forward pass has a known local
derivative, and the chain rule says the gradient with respect to an input is the product of all the local
derivatives along the path from the loss back to that input. Reverse mode evaluates that product from the
loss end, propagating a gradient signal backward through every operation. Its defining virtue is cost. One
backward pass produces the gradient with respect to *every* input at once, at roughly the cost of one
forward pass, regardless of how many inputs there are. That is the property that makes gradient descent on
millions of parameters feasible, and it is the property that makes optimizing a simulation feasible too.

There is a catch, and it is the tension the whole [[differentiating-the-rollout]] section is built around.
The backward pass needs the intermediate states from the forward pass, because each local derivative is
evaluated at the actual forward values. So a naive differentiable rollout has to *store every intermediate
state it will reuse*, which is memory that grows with the length of the rollout. For a short horizon this
is free. For a long one it is the first wall you hit, and the escape, recomputing states on demand instead
of storing them all, is its own topic in the core.

The deeper catch is that reverse-mode autodiff faithfully differentiates whatever program you actually
wrote, kinks and all. Where the simulator has a hard branch, like a wall zeroing a velocity, or a near-zero
quantity in a denominator, like a barely-touched grid node, the chain rule keeps multiplying through
slopes that are ill-defined or enormous. The result can be a gradient that is technically what the program
computes but physically meaningless or numerically overflowing. Those are not autodiff bugs. They are the
honest consequence of differentiating a non-smooth physical step, and taking them apart is the entire job
of [[failure-modes]].
