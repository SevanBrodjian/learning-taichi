# Learned materials: building a continuous material dial

**The goal:** a controllable world model wants a knob. Turn it from water toward putty toward snow, or from
oil toward honey, and get a continuous family of *correct* dynamics. This page is the whole arc of trying to
build that knob — three designs, each motivated by the failure of the one before it, ending at an honest
partial result and one unresolved crux.

Read it as a single argument, not a history. The short version:

1. **Blending the weights of per-material networks fails**, and fails worse the more different the materials.
2. **Conditioning one network on a descriptor input works** — but costs accuracy and is only smooth where the
   requested material is physically real.
3. **Some physics a per-particle network structurally cannot learn**, no matter its size. The fix is inputs,
   not capacity.
4. **Learning the whole material is achievable at the training points**, but per-step supervision does not buy
   long-horizon stability — and that gap is still open.

## Attempt 1: blend the weights. It doesn't work.

The cheapest imaginable dial: train a small network at a few settings, linearly blend its weights for the
settings in between, $\theta(\alpha) = (1-\alpha)\theta_A + \alpha\theta_B$.

Test it in the friendliest possible case first — **viscosity**, where the functional form is fixed and one
scalar varies. The viscous stress is $f_\mu(C) = \mu\,(C+C^{\top})$, which is *linear in the knob*, so in
**function** space the correct blend is unambiguous:

$$(1-\alpha)\,f_{\mu_{\text{thin}}} + \alpha\,f_{\mu_{\text{thick}}} = \big((1-\alpha)\mu_{\text{thin}} + \alpha\mu_{\text{thick}}\big)(C+C^{\top}).$$

Any failure is therefore a statement about the **weight-to-behavior map alone**. And it fails:

![Effective viscosity of the interpolated fluid against the interpolation coefficient. Both the independent-start and warm-started curves touch the ideal dotted line only at the endpoints and bow well below it across the entire middle.](/api/data/learning-taichi/runs/material-variants/train-and-interpolate-nns-to-mimic-viscous-liquids/interp_effmu.png)

At the halfway point the intended viscosity is about $0.16$; the interpolated fluid measures roughly half
that. **Why:** a network's output runs through a *product* of weight matrices, and a product is not linear in
its factors. Interpolating two matrices at once multiplies two half-grown factors, and that product is
smaller than the average of the two full products. A straight line in weight space is a **sagging** path in
output magnitude. Warm-starting the thick net from the thin net's weights — the standard cure for coordinate
mismatch — sags essentially as much, because it is the *length of the chord* through a nonlinear map, not the
random seed, that causes the sag.

Now the hard case: three structurally **different** constitutive laws (fluid, corotated elastic, snow). Here
there is no linear ideal to undershoot, because a fluid and a solid are not two points on a line through
function space.

![Five panels along the fluid-to-elastic interpolation. The endpoints are a clean fluid puddle and a clean compact elastic blob. The three interior panels are each a sparse spray of particles scattered across the domain — neither puddle nor blob.](/api/data/learning-taichi/runs/material-variants/train-material-replicating-nns-and-interpolate/interp_fluid_elastic_still.png)

The endpoints are exact and every interior blend **disperses into a diffuse cloud**. The mechanism is the
sag made qualitatively worse: with the same functional form, every point on the chord was still a *valid*
viscous stress, merely the wrong size. With different functional forms, the chord passes through networks
whose output is neither — a tensor field that is not the gradient of any stored energy and carries no
guarantee of being dissipative. **A non-dissipative stress injects energy every step**, so the blob heats up
and flies apart. Leaving the linear family means leaving the manifold of valid constitutive laws, and most of
the chord lies off it.

> **The transferable lesson:** interpolating parameters of a model is not interpolating the behavior of the
> model. This is not specific to physics — it is a property of any nonlinear parameterization.

## Attempt 2: condition one network on a descriptor

Stop interpolating weights. Use **one** weight set $\theta$ and feed the material identity as an *input*:

$$g_\theta(\text{features},\, m) \approx \text{stress},$$

where $m$ is a small descriptor vector. Now the stress at an interior $m$ is the network's own output as a
smooth function of its inputs, **on the manifold it actually learned**. Moving $m$ moves the stress the way
moving the strain does — smoothly and on-distribution. Interpolate the input, not the weights.

**Part of a material's identity is not in the stress.** A fluid keeps its deformation gradient $F$ purely
volumetric; snow clamps the singular values of $F$ into a yield band each step. Those are **state rules**,
not stresses. So the descriptor must drive them too, via a shared analytic kernel with knobs that $m$ sets —
an isotropization $\operatorname{iso}(m_1) = 1-m_1$ and a plastic clamp band interpolated in inverse-band
space. Stress and state evolution move together as $m$ sweeps.

![A 5x5 grid of simulation panels, one per descriptor value. The right column grades cleanly from a springy elastic blob to a crumpled snow heap. The left and middle show wide fluid-like spreads. The bottom-left region scatters into a diffuse cloud.](/api/data/learning-taichi/runs/material-variants/one-nn-for-three-materials/grid_montage.png)

**What it buys, honestly — three distinct behaviors:**

- **Where the physics is well posed, it works.** The full-solidity column sweeps smoothly from elastic blob to
  crumpling snow heap. Every cell is a compact settled pile. This is the morph weight-blending could not build.
- **Where a state rule collides with stability, it is abrupt.** The material stays fluid-like across most of the
  solidity range and snaps to solid only near $m_1=1$. The fluid needs near-full isotropization to stay
  numerically alive, and isotropization removes exactly the shear resistance that makes a solid hold shape.
  **Some of this abruptness is intrinsic to the physics, not a tuning failure.**
- **Where the requested material is ill-posed, it degenerates.** A fluid carrying a yield surface is not any
  real material and appears in no training data; the untrained corner extrapolates into a stress that flings
  particles apart.

**And it costs accuracy.** At each trained corner the shared network tracks the true simulator to about
one to two percent, notably worse than per-material specialist nets at a tenth of a percent. That is the
**capacity cost** of one shared weight set: the shared parameters that let it interpolate are the ones it can
no longer specialize.

> Conditioning is the right primitive for a material dial — but design the descriptor so its **interior stays
> inside the manifold of real materials**. No ill-posed corners.

## The locality lesson: some laws a per-particle net cannot learn

Push the idea onto a liquid with two knobs, **viscosity** and **surface tension**, and you hit the single
most important structural distinction in learning constitutive physics.

**Viscosity is local.** The Newtonian viscous stress $\sigma = \mu(C_p + C_p^{\top})$ is a genuinely pointwise
function of the particle's own affine matrix. A per-particle network has every input it needs, and learns the
viscosity dial effortlessly.

**Surface tension is not.** It is a capillary force whose strength is set by the **curvature** of the
interface:

$$f = \sigma_{st}\,\kappa\,\nabla\phi, \qquad \kappa = -\nabla\cdot n, \qquad n = \frac{\nabla\phi}{\lVert\nabla\phi\rVert}.$$

Curvature is a **second derivative of the density field**, computed across several grid cells (see
[[vector-calculus]]). A particle's own $(J, C_p, v_p)$ carries none of that — it has no input that even
*correlates* with interface curvature. **A bigger network does not help.** It is not a capacity problem; the
information is absent.

The fix is to give a network the right **inputs**: a second net reads a $5\times5$ patch of the smoothed grid
density $\phi$ and outputs the capillary force. The window size is not arbitrary — the analytic curvature
stencil reaches two cells out, so $5\times5$ is exactly the support the force depends on. Crucially the net
is **not** handed $\kappa$; it must infer curvature itself, or it would merely be echoing the formula.

![The learned capillary force plotted against the analytic force. At the trained strength both components lie on the identity line; at an untrained intermediate strength the network still matches, showing it learned the correct linear-in-strength law from two endpoints.](/api/data/learning-taichi/runs/material-variants/generalize-one-nn-across-viscosity-and-surface-tension/capillary_fit.png)

It fits well because $\kappa\nabla\phi$ is a smooth, low-order function of a smoothed field once the network
can see the patch — and it generalizes across strength because the true force is exactly **linear** in
$\sigma_{st}$, so two endpoints pin the line.

> **The transferable lesson:** before scaling a network, ask whether its inputs contain the information the
> target depends on. Locality of the law must match locality of the features.

**A practical corollary on descriptor design.** Surface tension rounds a blob fast and then **saturates**:
past a modest strength, higher is visually identical. If a descriptor axis maps into that saturated tail,
every high-$m_{st}$ row of a sweep looks the same and the axis carries no information. Calibrate the physical
range *first*, then pick a gentle schedule $\sigma_{st}(m) = \sigma_{\max} m^{p},\ p>1$ that keeps the range
where the effect is still visible. **A descriptor axis is only useful over its unsaturated range.**

## Attempt 3: learn the *whole* material, not just the stress

Everything above still left most of the material analytic — a hand-written volume update, a bolted-on
capillary formula. That is a hybrid, not a learned material. The honest version: inside a fixed MPM solver,
replace the **entire** per-particle material with networks and keep only the transfer machinery.

A material lives in exactly **three** places in an MLS-MPM step (see [[mls-mpm-forward]]):

1. **The stress**, at particle-to-grid.
2. **The carried-state update**, at grid-to-particle — where the material *remembers* deformation ($J$ for a
   liquid, $F$ for a solid).
3. **Surface tension**, the non-local interface force.

Everything else — B-spline weights, scatter/gather, mass-normalize, gravity, the floor, advection — is
**transfer and boundary scaffolding**, identical for water, sand, or rubber. So "learn the whole material"
has a precise meaning: learn those three, import the rest unchanged. **If any of the three is still an
analytic equation in the rollout, the material is not learned — it is decorated.**

Learning the state update is the part that matters most for the future. For a liquid it is nearly trivial
($\dot J = J\operatorname{tr}C$). For a **solid** the carried state is $F$, and its update is where
plasticity, hardening, and every history-dependent effect live. A method that can only learn the stress is
stuck with hand-written state rules forever.

**Verifying the scaffolding.** A subtle failure mode of "keep the transfer fixed" is quietly forking it. The
custom step, with surface tension off, is run against the canonical frozen `simulate` and matches to a
trajectory RMSE around $10^{-6}$ — GPU noise. Only then is the scaffolding trustworthy. This is what the
frozen physics library is *for*.

![Grid of the whole learned material in cyan over the ground-truth liquid in grey at each descriptor cell, three trained corners starred and the top-right corner held out.](/api/data/learning-taichi/runs/material-variants/train-one-nn-to-mimic-viscosity-and-st/grid_overlay_montage.png)

**The result is a genuine partial.** At the three trained corners the whole learned material is **edge-exact**
— stress, capillary force, and volume rate each reproduce their targets and the rollout tracks the true
liquid. The **interior and the held-out corner are stable but only partially physical**: the thick-plus-high-ST
composition under-rounds and throws upward particle jets rather than settling.

**A trap worth internalizing: one distance scalar cannot certify a cell as physical.** Position error and
shape error measure different things, and on this grid they agree only moderately (Spearman $\rho \approx
0.55$). At the held-out corner both scream, so nothing is hidden there. But there are **interior cells whose
position error sits within a factor of two of the trained corners while their shape is badly wrong** — a
drop that stays roughly where it should while being far too round, or not round at all.

The deeper problem is that a distance number has **no interpretable scale**. Told that a cell scores $0.12$,
you cannot say whether that is "slightly off" or "not a liquid" without a reference. Roundness can be read
directly against the ground truth's own roundness. **Judge a learned simulator by shape and motion against
ground truth; use a distance scalar to rank, never to certify.**

## Identity is not history, and mixing them is the design mistake

Conditioning works, so the next question is what the conditioning input should be. A learned material has
**two** things that vary per particle, and they are easy to conflate because both look like "extra state
the network reads".

**Identity** answers "which material is this". It is one value per material, shared by every particle of
that material, and it never changes. **History** answers "what has this particle been through". It is one
value per particle, and it changes every substep. Snow illustrates both at once: every snow particle
carries the same identity, but a snow particle that has been compacted is stiffer than one that has not,
and that stiffening is carried in a plastic record that the constitutive law reads and updates.

Keeping them in separate inputs is not tidiness, it decides what can be trained. An identity code is a
**fixed input** the network can be trained against by supervised regression, one sample at a time. A
history variable in a **known** parameterisation, like the symmetric stretch $S$ and the plastic record
$J_p$, is also trainable one sample at a time, because the correct update is an analytic function of the
current state and the target is available without ever rolling forward. A **free learned latent state**,
one whose meaning the network discovers, is a different animal entirely: nothing supervises it at any
single step, so the only signal comes from whether the rollout ends up in the right place, which means
backpropagating through a long rollout and inheriting every pathology in [[failure-modes]].

The practical rule that falls out: **use a known parameterisation for history whenever one exists.** A
learned latent state is what you reach for when the physics has memory you cannot name.

### The latent space is a label, not an axis

For a family generated by a continuous knob, like viscosity, the descriptor is a real physical axis and a
value between two trained settings has a ground truth you can check against. For a set of structurally
unrelated materials, it does not. Halfway between water and snow is not a material; there is nothing to
compare against, so there is nothing to validate and no claim to make. The code is an index that happens
to be written as a vector.

Two things still make the vector worth using instead of an integer. It puts the four materials at a
fixed, well-separated geometry (the corners of a regular simplex, so no pair is closer than any other, and
centred at the origin so no material's code dominates the first layer's scale). And **jittering it during
training** — perturbing each code by a small fraction of the separation on every batch — forces the network
to learn a neighbourhood around each code rather than four point lookups. Without the jitter, a network is
free to build a lookup table keyed on four exact vectors, which is a conditioning mechanism in name only
and behaves arbitrarily the moment the code is slightly off.

### What one shared network across four unrelated materials actually achieves

Extending the descriptor idea from three related materials to the four canonical ones — fluid, rubber,
snow, sand — with the seam widened to include the **plastic state update**, so snow's clamp and sand's
Drucker-Prager return map are learned rather than applied analytically afterwards, sharpens the tradeoff
rather than removing it.

The one-step fit behaves the way capacity arguments predict. Held-out error, measured per material in
units of that material's own spread, falls steadily with width, and a second hidden layer at width 64
beats a single hidden layer at width 128 on every material. The interesting part is *which* material is
hardest, and it is consistently **snow**, by a factor of two to three, at every width.

The likely mechanism is a scale collision, and it is worth understanding because it will recur in any
shared-weight multi-material model. Snow's plastic clamp confines its principal stretches to a band about
three percent wide, while rubber's deformation gradient roams over tens of percent. A single input
normalisation shared by all four materials therefore hands the network an almost constant input for snow
while still demanding a varying output. Nothing forbids fitting that — the first layer can simply use
larger weights in the relevant directions — but those weights are shared with the material that needs the
opposite, and the conditioning code has to gate between them.

And the rollout still fails. Every material stays finite and none of them blows up, but over a full
simulation all four expand into indistinguishable dust that fills the domain, where canonical fluid,
rubber, snow and sand end at spread widths of 0.40, 0.18, 0.35 and 0.66. Fitting each step to within
ten percent of its own scale is not the same as reproducing the material, for exactly the reason the next
section gives.

## The crux: per-step supervision does not buy rollout stability

Every result above is fit by **per-step supervised regression** — at states drawn from ground-truth rollouts,
match the analytic target. This makes each piece **locally** accurate, which is why trained corners look
excellent. It does nothing about the **rollout**, because it never sees one.

The failure mechanism is **covariate shift**: the training distribution is the *ground truth's* trajectory;
the test distribution is the *learned rollout's* trajectory. A force that is right at every state the ground
truth visits can be wrong at the slightly-off states the learned rollout drifts into — and those errors
**accumulate** over hundreds of integration steps into a jet or a blow-up.

Two cheap countermeasures: **input noise** during training, and **DAgger** (roll out the current material,
collect the off-distribution states it actually visits, relabel with analytic targets, retrain). The honest
catch is that aggregation is only as good as the rollout it samples — if the current material is bad, folding
in its garbage states makes the fit *worse*, which is what happened here. The safeguard is **rollout-based
model selection**: keep the round whose rollout best matches ground truth, so aggregation can only help.

**Neither is the real answer.** The real answer is training **through** the rollout under a trajectory loss —
the differentiable-simulation route of [[differentiating-the-rollout]] — which puts error accumulation
directly in the objective and buys long-horizon stability at the price of a much harder gradient.

> **The recurring lesson of this entire page:** fitting instantaneous material laws is cheap and local.
> Keeping the resulting simulator faithful over a long rollout is a **global** property that a per-step loss
> does not buy. For a controllable world model — where a learned material must stay stable under a controller
> pushing it into unfamiliar states — that gap is the whole game.

## What is open

- **Solids.** Learning the state update has now been taken to the deformation gradient with plasticity,
  across four unrelated materials at once. The one-step fit succeeds and the rollout does not, so the
  open question moved from "can the plastic update be fitted" to "what supervision keeps it faithful".
- **Training through the rollout.** The crux above is diagnosed, not solved. No result here trains a material
  under a trajectory loss.
- **Resolution.** The capillary net is tied to the grid resolution and smoothing count its density patch was
  built at. A resolution sweep is untested.
- **Seeds and scope.** The weight-blend negative was not measured across training seeds. Every result here is
  one 2-D material family with linear schedules and supervised fits.

---

**Code:** `sim/learned_materials.py`, `sim/learned_viscosity.py`, `sim/one_nn_materials.py`,
`sim/one_nn_fluids.py`, `sim/one_nn_whole_fluid.py`.
**What it costs to deploy one of these:** [[networks-inside-a-kernel]].
**Prerequisites:** [[constitutive-models]], [[svd-polar]], [[viscosity]], [[surface-tension]].
**Next:** [[differentiating-the-rollout]] — the route that closes the stability gap.
