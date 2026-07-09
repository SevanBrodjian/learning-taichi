# Learning three materials with one net, and why interpolating them breaks in the middle

[[learned-viscosity-interpolation]] asked whether a continuous control knob can be built cheaply, by
training a small network at a few settings and linearly blending its weights for the settings in between.
There the knob was viscosity, a single scalar multiplying a fixed stress form, and the answer was a
clean but bounded negative: the blended fluid was stable and plausible but systematically too thin
through the middle of the range. This page pushes the same question somewhere harder. Instead of one
functional form scaled by a knob, take three genuinely different constitutive laws from
[[constitutive-models]] and [[material-showcase]] (a weakly compressible fluid, a corotated elastic
solid, and Stomakhin snow), learn each with the same network, and interpolate the weights between them.
There is no linear ideal to undershoot here because a fluid and a solid are not two points on a line
through function space. What emerges is a sharper and more useful negative: the endpoints can be made
exact, but the interior of the interpolation is degenerate, a material that scatters into a diffuse
cloud rather than settling into anything in between.

This is the exact shape of the shortcut a controllable world model is tempted to take. A world with a
material dial (turn it from water toward putty toward snow) needs a continuous family of dynamics, and
the cheapest imaginable way to fill it in is to train a handful of materials and blend. Knowing precisely
how and why that shortcut fails is what tells a builder to spend the effort on the honest alternative
instead.

## The hard part is making the weights comparable

Blending two weight vectors only means something if the two networks are the same shape doing the same
job in the same coordinates. The three materials natively carry different state, which is the obstacle. A
fluid tracks only the scalar volume ratio $J = \det F$, where $F$ is the deformation gradient, the matrix
that maps a small material vector from its rest configuration to its current one. An elastic solid tracks
the full $F$. Snow tracks $F$ plus an accumulated plastic record. Three different state layouts cannot
share a weight vector, so the first move is to force one shared substrate onto all three.

Every net here reads the **same position-free local features** and writes the **same output**. Carry the
full $F$ for all three materials, including the fluid, for which it is the special case where only
$\det F$ matters. Feed the network not the raw $F$ but its symmetric stretch $S$ from the polar
decomposition $F = R\,S$, where $R$ is a rotation and $S$ is symmetric positive-definite (the derivation
and why it is the right thing to feed a corotated model are in [[svd-polar]]). Along with $S$ give it the
APIC affine matrix $C_p$, the velocity, and the plastic record. The output is the three independent
entries of the symmetric stress in the material frame, rotated back to the world frame by the analytic
$R$. So a single map

$$
g_\theta(S,\, C_p,\, v,\, J_p) \;\approx\; \text{material-frame stress},
$$

with only the weights $\theta$ differing between fluid, elastic, and snow, and a single shared feature
standardization and output scale so nothing about the three fits differs except $\theta$.

Feeding the stretch $S$ rather than $F$ is not cosmetic. The corotated stress lives in the small
difference $F - R$, an elastic strain of a few percent buried inside an order-one rotation. A network fed
raw $F$ would have to first undo the rotation itself, and its error on that large rotation swamps the
tiny strain that actually carries the stress. The polar decomposition hands the strain over directly.
This is the same discipline that made the learned residual in [[hybrid-learned-residual]] and the
viscosity net in [[learned-viscosity-interpolation]] generalize instead of memorize: give the network
the rotation-invariant, position-free local state and it is forced to learn a genuine local law.

## The stress is learnable, but the state rule is not in the weights

There is a piece of each material's identity that a memoryless stress network cannot hold. A fluid keeps
its $F$ purely volumetric (the shear part has no restoring force and simply is not tracked). Snow's
plastic clamp limits the singular values of $F$ into a yield band $[1-\theta_c,\, 1+\theta_s]$ each step
and pushes the excess into the plastic record. Both are **state updates**, rules for how $F$ evolves, not
stresses, so they sit outside $\theta$. The honest choice, following the differentiable-materials
treatment in [[differentiable-materials]], is to keep these state rules as shared analytic code and let
the network learn only the stress. The snow net does see the current plastic record as an input, so it
can represent the hardening as a function of the present state, but it never learns how that record
accumulates. Scoping every interpolation claim to what the weights actually control, and being explicit
that they do not control the state rule, is the whole ballgame later.

Under this substrate all three nets reproduce their material when they drive the rollout, and, trained on
a handful of varied scenes that genuinely exercise each material (a soft drop, a hard impact that fires
snow's plastic clamp on more than half of its particles, a slumping column, a settling slab, a lateral
throw), plus a reflected copy of every state since the physics is symmetric under left-right mirroring,
they transfer to held-out scenes they never saw. The picture below is the load-bearing check that
the snow net is really snow and not a soft solid: on a hard impact the three learned nets separate
cleanly.

![Three panels showing the three learned networks driving the same hard downward impact of a disk onto
the floor. The left panel, the learned fluid, has splattered into a thin sheet spread across the whole
floor. The middle panel, the learned elastic, has rebounded into a single compact rounded blob. The right
panel, the learned snow, sits as a compact crumpled heap that neither flowed flat like the fluid nor
sprang back like the elastic. The three learned materials are visibly distinct, and the snow one behaves
like snow.](/api/data/learning-taichi/runs/material-variants/train-material-replicating-nns-and-interpolate/q1b_distinct_hard_still.png)

## The endpoints must be exact before the interior means anything

The interpolation is the sweep $\theta(\alpha) = (1-\alpha)\,\theta_A + \alpha\,\theta_B$ from material
$A$ at $\alpha = 0$ to material $B$ at $\alpha = 1$. Before reading anything off the interior, the two
endpoints have to be right: at $\alpha = 0$ the blended net **is** the trained net for $A$, so its
rollout must be identical to $A$'s own replication rollout, and likewise at $\alpha = 1$. If an endpoint
does not reproduce its material, the harness is broken and the interior is meaningless.

The subtlety that makes this non-trivial is the state rule. The blended net at $\alpha = 0$ carries $A$'s
weights, but the rollout also needs $A$'s state rule (the fluid's volumetric projection, or elastic's
free deformation, or snow's clamp), and that rule is not in $\theta$. The fix is to run every rollout
through one unified state kernel with two continuous knobs, an isotropization that at its extreme keeps
$F$ volumetric for the fluid and a yield band that at its extreme is snow's clamp, and to co-interpolate
those knobs along the sweep so that each endpoint gets its own true state rule. With that, both endpoints
reproduce their material to the level of the simulator's own run-to-run noise. The isotropization also
earns its keep numerically: a fluid run with a fully free $F$ lets the untracked shear part drift without
bound, and $\det F$ computed from a wildly ill-conditioned $F$ turns to catastrophic-cancellation
garbage, which is exactly the kind of silent numerical failure that makes a learned rollout blow up for
reasons that have nothing to do with the network.

The lesson hiding in this plumbing is worth stating plainly. Part of what makes a fluid a fluid and snow
snow is not in the learnable stress at all, it is in a fixed state rule, so weight interpolation alone
cannot even connect the two endpoints. Something outside the weights has to move with $\alpha$ too.

## The interior is degenerate, and why

With the endpoints exact, the interior is a clean negative for both material pairs. Every interpolated
blend between the endpoints disperses.

![Five panels along the fluid-to-elastic interpolation. The far-left panel at coefficient zero is a clean
spreading fluid puddle sitting on its true-fluid reference, and the far-right panel at coefficient one is
a clean compact elastic blob sitting on its true-elastic reference. The three interior panels are each a
sparse spray of particles scattered across the entire domain, not a puddle and not a blob, a diffuse
cloud filling the box. The two endpoints are correct materials and every point in between is a broken
one.](/api/data/learning-taichi/runs/material-variants/train-material-replicating-nns-and-interpolate/interp_fluid_elastic_still.png)

The mechanism follows from two facts stacked together. The first, established in
[[learned-viscosity-interpolation]], is that the map from a weight vector to the function it computes is
strongly nonlinear. A one-hidden-layer network computes roughly $W_2\,\tanh(W_1 x)$, and the magnitude of
its output runs through the product of the two weight matrices, which is not linear in the matrices. A
straight line in weight space is a curved path in function space, so the midpoint weights of two distant
solutions do not compute the midpoint function.

The second fact is what makes this case qualitatively worse than the viscosity one rather than just
quantitatively worse. In the viscosity study both endpoints were the **same functional form**,
$\mu\,(C + C^{\top})$, scaled by the knob. Every point along the chord in weight space was therefore
still a valid viscous stress, merely the wrong size, which is why the interpolated fluid stayed a fluid
and only came out too thin. Here the two endpoints are **different functional forms**, a det-only
isotropic pressure and a full corotated tensor. The chord between their weight vectors passes through
networks whose output is neither, a tensor field that is not the gradient of any stored energy and
carries no guarantee of being dissipative or even sign-definite. A stress that is not dissipative injects
energy into the material every step instead of removing it, so the blob heats up and its particles fly
apart until they are the diffuse cloud in the figure. Leaving the linear family means leaving the
manifold of valid constitutive laws, and most of the chord lies off that manifold.

Co-interpolating the state rule does not rescue this, and seeing why is the point. The interior's problem
is the blended stress, which is invalid; the state rule is a separate axis. Snow's plastic clamp can be
engaged smoothly along the elastic-to-snow sweep, and the endpoints still come out as clean elastic and
clean snow, but every interior blend still scatters, because the stress driving it is off the manifold no
matter how the clamp is scheduled.

## Why this matters for controllable worlds, and what is open

The tempting shortcut for a continuous material control is to train a few materials and blend their
weights. [[learned-viscosity-interpolation]] showed the shortcut is unsafe even in the friendliest case,
a target exactly linear in the knob, where it merely gave the wrong magnitude. This page shows that when
the endpoints are structurally different, the friendliest case is gone and the shortcut does not produce
a material at all through the interior. A world model built to be trusted between its calibration points
cannot be built by interpolating separate per-material stress networks. The two routes that would work
are the same two that closed the viscosity case, scaled up to this harder setting: **condition** a single
network on a material descriptor so the entire continuum is trained on real physics rather than assumed
by averaging, and, because a material's identity also lives in state rules outside the stress, **learn or
condition those state rules too** rather than hoping they come along for free.

What stays open is the reach of the negative. The degenerate interior is argued from the endpoints lying
off one linear family and is expected to be robust to the training seed, but that was not measured across
seeds. It is untested whether a shared frozen backbone with small per-material output heads, which would
force the blend to stay closer to a common function, would connect structurally different materials any
better than it connected the viscosities. And the sharpest question for the vision is whether a single
material-conditioned network, trained across the fluid-elastic-snow family at once, produces a genuinely
smooth and physical morph where blending separate networks produces a cloud. That is the experiment this
negative result is meant to motivate.
