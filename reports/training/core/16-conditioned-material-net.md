# Conditioning one network on a material descriptor, and the fidelity it costs

[[learned-material-interpolation]] ended on a prediction. Training a separate network for each of three
materials and linearly blending their weights gives exact endpoints but a degenerate interior: the chord
between two distant weight vectors leaves the manifold of valid constitutive laws, and every blended
material in between scatters into a diffuse cloud. The fix it argued for was to stop interpolating weights
and instead **condition one network on a small material descriptor**, so the whole continuum is trained on
real physics, and then interpolate the **descriptor** (an input) rather than the weights. This page builds
exactly that and reports what it actually buys. The short version: conditioning does defeat the
weight-blend's universal explosion, but it is not free, and it is not uniformly smooth. It trades
per-material accuracy at the calibration points for an always-finite morph that is genuinely physical only
where the underlying material combination is physically well posed.

## Interpolating an input versus interpolating weights

[[learned-material-interpolation]] showed why weight-blending fails: a net's output runs through the
**product** of its weight matrices, so a straight line in weight space is a curved path in function space,
and between structurally different stress laws the midpoint weights compute a tensor field that is neither,
non-dissipative, and blows the material apart. Conditioning sidesteps this. There is **one** weight set
$\theta$, used coherently at every descriptor value, as a single map

$$
g_\theta(\text{features},\, m) \;\approx\; \text{material-frame stress},
$$

where $m$ is a small vector of descriptor scalars fed alongside the position-free physical features (the
polar stretch $S$, the affine matrix, velocity, the plastic record; same rotation-invariant features as
[[constitutive-models]] and [[svd-polar]]). The stress at an interior $m$ is the one network's own output as
a smooth function of its inputs, on the manifold it actually learned, so moving $m$ moves the stress the way
moving the strain does: smoothly and on-distribution. That is why the conditioned cells stay finite where the
weight-blend cells exploded.

## A two-parameter descriptor, and conditioning the state rule too

Three structurally different materials are the target: a weakly compressible fluid, a corotated elastic
solid, and Stomakhin snow, the same three from [[material-showcase]]. The descriptor is two scalars,
chosen so the three materials sit at corners of a square:

- $m_1$ is **solidity**. At $m_1 = 0$ the material is a fluid; at $m_1 = 1$ it is a solid.
- $m_2$ is **plasticity**. At $m_2 = 0$ the solid is elastic; at $m_2 = 1$ it is snow.

So the trained materials are fluid at $(0,0)$, elastic at $(1,0)$, and snow at $(1,1)$. The fourth corner
$(0,1)$ is never trained, and it matters later.

A descriptor that only fed the stress network would not be enough, because part of what makes a fluid a
fluid and snow snow is **not in the stress at all**. It lives in a state rule outside the weights, exactly
as [[differentiable-materials]] and the precursor spelled out. A fluid keeps its deformation gradient $F$
purely volumetric, because the shear part has no restoring force and, left free, drifts until $\det F$ (all
the fluid stress depends on) turns to catastrophic-cancellation garbage. Snow clamps the singular values of
$F$ into a yield band each step and pushes the excess into its plastic record. Both are rules for how $F$
evolves, not stresses, so the descriptor has to drive them too. The single unified state kernel carries two
knobs, and $m$ sets them:

$$
\operatorname{iso}(m_1) = 1 - m_1, \qquad
\frac{1}{\theta_c(m_2)} = \frac{1-m_2}{\theta_{\text{off}}} + \frac{m_2}{\theta_{c,\text{snow}}}.
$$

Here $\operatorname{iso}$ is the isotropization that at $\operatorname{iso}=1$ forces $F$ volumetric (the
fluid) and at $\operatorname{iso}=0$ leaves it free (the solids), and $\theta_c$ is the plastic clamp
half-band, interpolated in inverse-band space so $m_2 = 0$ never clamps (elastic) and $m_2 = 1$ is snow's
band. Each symbol: $m_1, m_2$ are the two descriptor scalars; $\theta_{\text{off}}$ is a band so wide the
clamp never fires; $\theta_{c,\text{snow}}$ is snow's actual band. One network drives the stress at every
$m$, and this shared analytic kernel drives the state evolution, both moving together as $m$ sweeps the
square. The single net is trained jointly on all three materials at once, each training state tagged with
its material's $m$, over the same varied signature-exercising scenes and mirror augmentation the precursor
used.

## What it actually buys: the edges are close, not exact

At each trained corner the one shared network follows the true simulator to about one to two percent on the
rollout (the fluid worst, the snow best), close but **not exact**, and notably worse than the precursor's
separate per-material nets that sat near a tenth of a percent. This gap is the headline, not a defect to
hide: it is the **capacity cost** of one shared weight set. A single small network holding three
structurally different stress laws cannot fit any one as tightly as a network free to spend all its
parameters on that material alone; the shared parameters that let it interpolate are the ones it can no
longer specialize. The fluid is the worst corner because its near-incompressible det-only pressure is the
hardest to reproduce precisely and a small pressure error shows up fast in a thin spreading sheet.

One check must be kept distinct from edge accuracy: driving the state kernel from the descriptor schedule
reduces to the exact pure-material state rule at each corner. That verifies the **state rule** is scheduled
correctly (the harness bug that broke the precursor's first attempt), but it is not a claim that the
material itself is reproduced exactly. It is reproduced only to the one-to-two-percent fidelity above.

## The 2-D grid: physical where the physics is well posed

Sweeping the descriptor across the unit square on a grid and running the conditioned simulation at each cell
gives the picture below. It has to be read honestly, cell by cell, because a diagnostic that trends smoothly
is not proof that a cell is a physical material.

![A five by five grid of small simulation panels, one per descriptor value, all dropping the same disk.
Horizontal axis is solidity from fluid on the left to solid on the right; vertical axis is plasticity with
elastic along the top row and snow along the bottom row. The trained materials are marked at three corners:
fluid top-left, elastic top-right, snow bottom-right. The right column is a clean column of compact settled
blobs grading from a springy elastic shape at the top to a crumpled snow heap at the bottom. Most of the
left and middle of the grid shows wide flat fluid-like spreads sitting low on the floor. The bottom-left
region, low solidity with plasticity engaged, shows particles scattered up into a diffuse cloud rather than
a settled pile, worst at the untrained bottom-left corner.](/api/data/learning-taichi/runs/material-variants/one-nn-for-three-materials/grid_montage.png)

Every one of the twenty-five cells stays finite. That alone is the win over the weight-blend, whose every
interior cell exploded. But the morph is not uniformly smooth, and three distinct behaviors show up.

The **clean, physical morph is the well-posed axis**: the full-solidity right column, where plasticity
sweeps the material from a springy elastic blob into a crumpling, holding snow heap. Every cell there is a
compact settled pile, the pile height falls smoothly as plasticity engages, and nothing disperses. This is
the elastic-to-snow morph the precursor could not build by blending weights, and here it is a genuine
continuous family.

The **solidity axis is abrupt, not gradual**. For most of the $m_1$ range the material stays a wide flat
fluid-like spread and only snaps into a compact solid near $m_1 = 1$. This is not a fit failure, it is
structural. The fluid corner needs near-full isotropization for stability, and isotropization removes shear
resistance, which is exactly what makes a solid hold its shape. The state rule the fluid needs to stay
numerically alive is the same rule that fluidizes, so anywhere solidity is less than nearly maximal the
material is forced to behave fluid-like regardless of the stress the network would like to apply. Some
abruptness on this axis is intrinsic to the physics, not a knob that can simply be tuned away.

The **fluid-plus-plasticity region degenerates**. In the bottom-left, low solidity with plasticity engaged,
the material disperses into a scattered spray; the airborne particle fraction climbs to about $0.66$ at the
untrained corner $(0,1)$, against at most a few hundredths everywhere else on the square. This is honest and
expected: a fluid carrying a yield surface is a physically ill-posed combination that is not any real
material and appears in no training data, so the network extrapolates into a stress that flings the
particles apart. The untrained corner is not a material and must not be read as one.

So the honest result is a mix, and stating it as anything cleaner would be an overclaim. Conditioning gives
an always-finite morph that is smooth and physical where the requested material is physically realizable
(the elastic-to-snow axis, the settled lower-right of the square), abrupt where a state rule and stability
collide (the solidity axis), and degenerate where the requested material is ill-posed (the plastic fluid).
The degeneracy is **confined to one quadrant** rather than filling the whole interior, which is the concrete
sense in which conditioning beats weight-blending.

## Why this matters for controllable worlds

The two ways to build a continuous material dial are now both measured. Blending separate per-material
networks ([[learned-material-interpolation]]) fails universally, the interior leaving the manifold of valid
physics and exploding. Conditioning one network on a descriptor is the better route and produces a usable
family, with two caveats a builder must plan around. One shared network buys smoothness at the cost of
exactness at the calibration points, so a knob that must hit its named settings precisely needs more
capacity, per-material output heads, or an explicit correction there. And the morph is only smooth where the
requested material is physically realizable, so the descriptor has to be **designed to keep its interior
inside the manifold of real materials** (no ill-posed corners like a plastic fluid), with the state rules
conditioned alongside the stress. Conditioning is the right primitive for a material dial, not an automatic
guarantee of a smooth physical continuum.
