# Learning viscosity, and what interpolating the weights does to it

The learned residual in [[hybrid-learned-residual]] put a small network inside the simulation step and
trained it to absorb a *missing* effect. This page keeps the network inside the step but changes its job
and pushes on a sharper question. The network now **replaces a known term**, the Newtonian viscous stress
of the fluid from [[viscosity]], and the same tiny architecture is trained once at each of several
viscosities. The point is not that a net can imitate a stress law it already knows in closed form. The
point is what happens next. Once there is a thin-fluid network and a thick-fluid network, the weights can
be **linearly interpolated**, and a natural hope is that the halfway weights produce a halfway fluid.

That hope is exactly the cheap dream of a controllable world model. A world model that wants a viscosity
slider needs a continuous family of dynamics, not a handful of trained settings, and the cheapest
imaginable way to fill in the in-between settings is to train a few and linearly blend their weights.
Whether that works is a question about the geometry of weight space. The answer found here is a clean and
useful **no**, for a reason worth understanding before anyone builds the slider that way.

## The setup, and why it isolates one thing

The fluid stress from [[viscosity]] is a pressure plus a viscous term,

$$
\sigma \;=\; \underbrace{E\,(J-1)\,I}_{\text{pressure}} \;+\; \underbrace{\mu_{\text{visc}}\,\big(C_p + C_p^{\top}\big)}_{\text{viscosity}},
$$

where $C_p$ is the APIC affine matrix carried on each particle, which is the estimate of the velocity
gradient built for free by the transfer (its construction is in [[mpm-in-context]]), $C_p + C_p^{\top}$ is
its symmetric part, the strain rate, and $\mu_{\text{visc}}$ is the single scalar that sets how thick the
fluid is. Only the viscous term is handed to a network. A per-particle multilayer perceptron $g_\theta$
reads position-free local features, the four entries of $C_p$ and the two velocity components, and outputs
the three independent entries of the symmetric viscous stress,

$$
g_\theta(C_p, v_p) \;\approx\; \mu_{\text{visc}}\,\big(C_p + C_p^{\top}\big).
$$

The features carry no absolute position, the same discipline that made the residual in
[[hybrid-learned-residual]] transfer across initial conditions rather than memorize a path. A stress law is
a local rule, so a network fed only local state is pushed to learn the rule, and it does. Each per-viscosity
net reproduces the fluid it was trained on when it drives the rollout, and the nets transfer to a dam-break
scene they were never trained on, which confirms they learned the strain-rate law rather than one
trajectory.

The training here is deliberately cheap. Rather than backpropagating a trajectory loss through hundreds of
physics steps, the way the residual in [[differentiating-the-rollout]] was trained, each net is fit by plain
supervised regression of its output onto the true viscous stress sampled from forward runs of the analytic
simulator. That sidesteps the long-horizon gradient entirely and makes the several per-viscosity fits fast
and stable, which matters because the interpolation study needs several fully trained nets, not one. The
price is that supervised regression only matches the stress pointwise and never sees the rollout, so a good
fit is checked separately by running the net as the update law and comparing the fluid to the truth.

The reason this particular target is the right one for studying interpolation is that it is **linear in the
knob**. Write the ideal per-viscosity function as $f_\mu(C) = \mu\,(C + C^{\top})$. Linear interpolation of
the thin and thick *functions* is

$$
(1-\alpha)\,f_{\mu_{\text{thin}}} + \alpha\,f_{\mu_{\text{thick}}}
\;=\; \big((1-\alpha)\,\mu_{\text{thin}} + \alpha\,\mu_{\text{thick}}\big)\,\big(C + C^{\top}\big)
\;=\; f_{\mu(\alpha)},
$$

with $\mu(\alpha) = (1-\alpha)\,\mu_{\text{thin}} + \alpha\,\mu_{\text{thick}}$. In *function* space the blend
is exactly a fluid of intermediate viscosity, with no ambiguity about what the right answer is. That is what
makes the experiment clean. Any failure of *weight* interpolation to produce that intermediate fluid cannot
be blamed on a crooked target. It is entirely a statement about the map from weights to behavior.

## Weight space is not behavior space

A network computes a function, but the map from a weight vector $\theta$ to the function it computes,
call it $\mathcal F(\theta)$, is both many-to-one and nonlinear. Many-to-one is easy to see. Permuting the
hidden units of a one-hidden-layer perceptron, and permuting the matching rows and columns of the two weight
matrices, leaves every output unchanged. Flipping the sign of a unit's incoming and outgoing weights does the
same through the oddness of $\tanh$. So a whole discrete family of symmetries $P$ satisfies

$$
\mathcal F(\theta) \;=\; \mathcal F(P\theta).
$$

Nonlinear is the part that bites here. Set the pressure aside and look at the viscous output alone. Ignoring
the bias and the nonlinearity for a moment, a one-hidden-layer net computes something like $W_2 W_1 x$, a
**product** of the two weight matrices. The magnitude of the output is carried by that product, and a product
is not linear in its factors. If the thin net uses small matrices and the thick net uses larger ones,
interpolating both matrices at once multiplies two half-grown factors, and the product of two half-grown
factors is smaller than the average of the two full products. A straight line in weight space is a curved,
sagging path in output magnitude. The $\tanh$ only sharpens this, since the thick net drives larger
pre-activations and the interpolated weights land where the nonlinearity has a different effective gain.

This predicts the specific failure. The thick viscous stress here is about fifteen times the thin one, so
the thick net's weights have to grow a long way to produce it. The chord from the small-weight thin net to
the large-weight thick net passes through intermediate weights whose composed output is **less** than the
average of the endpoints. Read through the fluid, that is an effective viscosity below the intended
intermediate. The interpolated fluid should come out too thin in the middle.

## What the fluid actually does

That is exactly what happens.

![Effective viscosity of the interpolated fluid plotted against the interpolation coefficient, which runs
from the thin network at zero to the thick network at one. The dotted line is the ideal, the intermediate
viscosity that blending the two stress functions would give and that a smooth slider would need. The red
curve interpolates two networks trained from independent random starts and the green curve interpolates a
thick network warm-started from the thin one. Both curves touch the dotted ideal only at the two endpoints
and bow well below it across the entire middle of the range, so the interpolated fluid is markedly thinner
than the intended intermediate viscosity at every interior point. The green warm-started curve is not
meaningfully closer to the ideal than the red independent one, so sharing an initialisation does not fix the
problem.](/api/data/learning-taichi/runs/material-variants/train-and-interpolate-nns-to-mimic-viscous-liquids/interp_effmu.png)

The two curves agree exactly at the endpoints, where each is a single fully trained network and no blending
happens. Everywhere in between they sag. At the halfway coefficient the intended viscosity is about $0.16$
but the interpolated fluid measures roughly $0.07$ to $0.09$, about half as thick as it should be. In the
side-by-side clips the interpolated fluid at each interior coefficient spreads visibly wider and flatter
than the true intermediate pile drawn behind it. The blend is monotone, it does thicken as the coefficient
climbs, but it thickens far too slowly and only catches up as the coefficient approaches one. Halfway weights
are not the halfway fluid.

## The fix everyone reaches for, and why it fails here

The standard explanation for interpolation failure between neural networks is **coordinate mismatch**, the
linear-mode-connectivity story. Two nets trained from independent random starts assign the same job to
different hidden units, unit three in one net and unit seven in the other, so averaging their weights
averages quantities that play unrelated roles and the midpoint is scrambled. The standard cure is to remove
the mismatch, most cheaply by training the second net from the **first one's weights** so both share an
initialisation and, the hope goes, a coordinate system.

This experiment runs that cure and watches it fail. The green curve above is the warm-started pair, the thick
net initialised from the thin net rather than from a fresh seed, and it sags essentially as much as the
independent red pair. The reason is visible in the weights. Warm-starting shares a *starting point*, but the
thick net still has to travel a long way to represent a stress fifteen times larger, and it ends up about as
far from the thin net as the independently initialised one does, in this run marginally farther. Sharing an
init does not shorten the chord, and it is the length of the chord through a nonlinear $\mathcal F$, not the
random seed, that causes the sag. Coordinate mismatch is a real effect, but this measurement shows it is not
the dominant one here. The dominant one is the compositional nonlinearity of the magnitude representation
along a long chord, and no amount of shared initialisation removes it.

Two routes actually would, and both are visible from here. The first is to stop carrying the magnitude
through a product. Freeze a single hidden layer shared across all viscosities and let only the **linear
output layer** differ between thin and thick. The viscous output is then linear in the interpolated output
weights, so weight interpolation becomes function interpolation and the intermediate viscosity comes out
exactly. The second, and the better one for a real controllable model, is to never interpolate at all.
**Condition** one network on the viscosity by feeding $\mu_{\text{visc}}$ in as an input, so the continuous
knob lives in the function from the start and the smoothness comes from training rather than from a hopeful
average of separate weights.

## Failure modes and fixes

The headline failure is the sagging curve itself, and the useful thing about it is what kind of failure it is
not. It is not an optimization failure, since each endpoint net fits the true stress to about two percent and
each endpoint fluid is right. It is not an instability, since every interpolated rollout stayed finite. It is
a pure statement about the geometry of the weight-to-behavior map, which is why it survives more training and
survives shared initialisation. The fixes are structural, freezing the shared layer or conditioning on the
parameter, not more epochs.

A second, quieter effect is worth keeping in mind, because it makes the negative look milder than it is. A
network can miss the true viscous stress by a noticeable relative margin on the rare high-strain-rate
particles and still reproduce the puddle almost perfectly, because the viscous term is a subdominant
correction to a pressure-driven flow and its per-particle errors average out over thousands of particles and
thousands of steps. This is why a modest pointwise fit still yields a faithful fluid at each endpoint, and it
is a warning that a good-looking rollout is a weak certificate of a learned law. The honest measurement pairs
a rollout diagnostic with a held-out fit error rather than trusting either alone.

A third practical point is stability. A learned stress is not guaranteed to be dissipative, and an
interpolated one even less so, so a learned viscous rollout can in principle blow up where the analytic one
would not. Keeping the network small, bounding the physical range of the viscosities so a single explicit
timestep stays under the diffusion limit from [[viscosity]], and checking every rollout for finiteness are
what keep the sweep trustworthy. Here nothing blew up, so the sag is a genuine behavior of the interpolated
fluid and not a numerical artifact.

## Why this matters for controllable worlds, and what is open

The tempting shortcut for a continuous control is to train a few settings and blend their weights between
them. The lesson here is that the shortcut is unsafe by default. Even in the friendliest possible case, a
target that is exactly linear in the knob, a tiny network, a stable rollout, and endpoints that each work
perfectly, naive weight blending delivered dynamics that were plausible but quantitatively wrong through the
entire interior, wrong in the one region a slider exists to cover. A world model built to be trusted between
its calibration points cannot be built this way without either aligning and constraining the networks or, far
better, conditioning a single network on the control so the interpolation is learned rather than assumed.

What stays open is how far even the negative generalizes. The target here is linear in the knob, which is
what made the ideal intermediate unambiguous; a control whose stress depends nonlinearly on it would not even
have an exactly intermediate function to aim for, so the whole comparison would have to be redrawn. The two
clean fixes, freezing a shared hidden layer and conditioning on the parameter, are argued for here but not
yet run, and the residual contribution of coordinate mismatch, expected to be small, could be pinned down by
an explicit neuron-alignment step before interpolating. Larger and more strongly nonlinear networks should
bow the curve further, since they carry more of the compositional nonlinearity for the chord to fall
through. The structural reason the blend undershoots is settled. Its size as the network and the physics grow
is not.
