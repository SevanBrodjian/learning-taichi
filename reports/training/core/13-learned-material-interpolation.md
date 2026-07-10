# Interpolating learned materials, and why blending their weights breaks

A controllable world model wants a continuous material dial: turn it from water toward putty toward snow and
get a continuous family of dynamics. The cheapest imaginable way to build that dial is to train a small
network at a few settings and **linearly blend its weights** for the settings in between. This page tests
that shortcut in two cases of increasing difficulty and finds a clean negative in both. The lesson is not
that networks cannot imitate a stress law (they can); it is that a straight line in weight space is not a
straight line in behavior, so the blend is untrustworthy in exactly the interior region a dial exists to
cover.

## Case 1: viscosity, a single form scaled by a knob

The friendliest possible version fixes the functional form and varies one scalar. The fluid stress from
[[viscosity]] is a pressure plus a viscous term, and only the viscous term is handed to a per-particle
network $g_\theta$ that reads position-free local features (the affine matrix $C_p$ and velocity) and
outputs the symmetric viscous stress:

$$
g_\theta(C_p, v_p) \;\approx\; \mu_{\text{visc}}\,\big(C_p + C_p^{\top}\big).
$$

Each per-viscosity net is fit by plain supervised regression onto the true stress from forward runs, which
sidesteps the long-horizon rollout gradient entirely and makes several fits fast and stable. Position-free
features push the net to learn the local rule rather than a trajectory, and it does: each net reproduces its
fluid and transfers to an unseen dam-break.

This target is the right one to study because it is **linear in the knob**. With $f_\mu(C) = \mu\,(C+C^{\top})$,

$$
(1-\alpha)\,f_{\mu_{\text{thin}}} + \alpha\,f_{\mu_{\text{thick}}}
\;=\; \big((1-\alpha)\,\mu_{\text{thin}} + \alpha\,\mu_{\text{thick}}\big)\,\big(C + C^{\top}\big),
$$

so in *function* space the blend is exactly an intermediate fluid, with no ambiguity about the right answer.
Any failure of *weight* interpolation is then a statement about the weight-to-behavior map alone.

### Weight space is not behavior space

The map $\mathcal F(\theta)$ from a weight vector to the function it computes is nonlinear, and that is what
bites. Ignoring bias and nonlinearity, a one-hidden-layer net computes a **product** $W_2 W_1 x$, and a
product is not linear in its factors. If the thin net uses small matrices and the thick net larger ones,
interpolating both at once multiplies two half-grown factors, and the product of two half-grown factors is
smaller than the average of the two full products. A straight line in weight space is a sagging path in
output magnitude, and $\tanh$ only sharpens it. The thick viscous stress here is about fifteen times the
thin one, so the chord runs a long way and its interior undershoots.

![Effective viscosity of the interpolated fluid against the interpolation coefficient, thin net at zero to
thick net at one. The dotted line is the intermediate viscosity a smooth slider would need. Both the
independent-start (red) and warm-started (green) curves touch the ideal only at the endpoints and bow well
below it across the entire middle, so the interpolated fluid is markedly too thin at every interior point,
and warm-starting does not close the gap.](/api/data/learning-taichi/runs/material-variants/train-and-interpolate-nns-to-mimic-viscous-liquids/interp_effmu.png)

At the halfway coefficient the intended viscosity is about $0.16$ but the interpolated fluid measures roughly
half that. Crucially, the standard cure fails: **warm-starting** the thick net from the thin net's weights,
meant to remove the coordinate mismatch of linear-mode-connectivity, sags essentially as much (the green
curve). Warm-starting shares a starting point, but the thick net still has to travel a long way to represent
a stress fifteen times larger, and it is the length of the chord through a nonlinear $\mathcal F$, not the
random seed, that causes the sag. Coordinate mismatch is real but not the dominant effect here.

## Case 2: three materials, different functional forms

Now take three genuinely different constitutive laws from [[constitutive-models]] and [[material-showcase]]
(fluid, corotated elastic, Stomakhin snow), learn each with the same network, and interpolate. There is no
linear ideal to undershoot, because a fluid and a solid are not two points on a line through function space.

Blending weights only means something if the nets share a shape, job, and coordinates, so all three read the
**same position-free features** and write the **same output**. The key discipline is to feed the network the
symmetric stretch $S$ from the polar decomposition $F = R\,S$ (see [[svd-polar]]), not the raw deformation
gradient $F$: the corotated stress lives in the small difference $F - R$, an elastic strain of a few percent
buried inside an order-one rotation, and a net fed raw $F$ would spend its capacity undoing the rotation and
swamp the strain. So one map, weights differing per material:

$$
g_\theta(S,\, C_p,\, v,\, J_p) \;\approx\; \text{material-frame stress}.
$$

One subtlety controls everything downstream: part of a material's identity is **not in the stress**. A fluid
keeps $F$ purely volumetric; snow clamps the singular values of $F$ into a yield band each step and banks the
excess as plastic record. Both are **state rules**, not stresses, so they sit outside $\theta$ and must be
supplied as shared analytic code. Every interpolation claim has to be scoped to what the weights actually
control, which is the stress and not the state rule.

### The endpoints must be exact before the interior means anything

The interpolation is $\theta(\alpha) = (1-\alpha)\,\theta_A + \alpha\,\theta_B$. The endpoints have to
reproduce their materials first, or the interior is meaningless. Because the state rule is not in $\theta$,
every rollout runs through one unified state kernel with two continuous knobs (an isotropization that keeps
$F$ volumetric for the fluid, and a yield band that becomes snow's clamp), co-interpolated along $\alpha$ so
each endpoint gets its own true state rule. With that, both endpoints reproduce their material to the
simulator's own run-to-run noise. The plumbing carries a lesson: weight interpolation alone cannot even
connect the two endpoints, because something outside the weights (the state rule) has to move with $\alpha$
too.

![Five panels along the fluid-to-elastic interpolation. The far-left panel (coefficient zero) is a clean
spreading fluid puddle on its true-fluid reference, and the far-right (coefficient one) a clean compact
elastic blob on its true-elastic reference. The three interior panels are each a sparse spray of particles
scattered across the whole domain, neither puddle nor blob. Correct materials at the endpoints, a broken one
everywhere between.](/api/data/learning-taichi/runs/material-variants/train-material-replicating-nns-and-interpolate/interp_fluid_elastic_still.png)

### The interior is degenerate

With the endpoints exact, every interior blend disperses into a diffuse cloud, and the mechanism is the
weight-space nonlinearity of case 1 made qualitatively worse. In the viscosity study both endpoints were the
**same functional form**, so every point on the chord was still a valid viscous stress, merely the wrong
size. Here the endpoints are **different functional forms**, a det-only isotropic pressure and a full
corotated tensor, and the chord between their weight vectors passes through networks whose output is neither.
That output is a tensor field that is not the gradient of any stored energy and carries no guarantee of being
dissipative. A non-dissipative stress injects energy every step, so the blob heats up and its particles fly
apart. Leaving the linear family means leaving the manifold of valid constitutive laws, and most of the chord
lies off it. Co-interpolating the state rule does not rescue this: the interior's problem is the invalid
stress, a separate axis from the clamp schedule.

## The fixes, and what is open

Two routes actually work, and both point away from blending separate networks. The first is to stop carrying
the magnitude through a product: freeze a single hidden layer shared across materials and let only the
**linear output layer** differ, so weight interpolation becomes function interpolation. The second, and the
right one for a controllable model, is to never interpolate weights at all but to **condition** one network
on a material descriptor fed as an input, so the continuum is trained on real physics rather than assumed by
averaging. That is the experiment [[conditioned-material-net]] runs, and it is where a material's state rules
have to be conditioned along with its stress.

What stays open is the reach of the negative. The degenerate interior is argued from the endpoints lying off
one linear family and is expected to be robust to the training seed, but that was not measured across seeds.
Whether a shared frozen backbone with small per-material output heads connects structurally different
materials better than blending whole networks is argued for but not run here.
