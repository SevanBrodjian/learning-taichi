# Two liquids in one solver: advecting a color and compositing the mix

The [[fluid-rendering]] and [[fluid-realism]] pages turn a single MPM fluid into a believable body of water,
but they only ever draw one liquid, one fixed water color for the whole field. A great deal of what makes a
fluid demo read as alive is more than one liquid at once: cream folded into coffee, two paints swirling, a
dye plume unrolling in clear water. This page adds exactly that, and the striking part is how little it
takes. No second material, no reaction chemistry, no extra solver. One passive number carried on each
particle, and a small change in the renderer, is enough to show two colored liquids braid together and
blend to an intermediate hue where they meet.

This is the cleanest possible example of a **passive tracer**, a quantity the flow carries but never acts
back on. A world model that wants to paint, label, or track material as it moves, which region came from
where, what concentration a parcel holds, needs exactly this: a field advected by the dynamics that does not
disturb the dynamics. Color is the tracer that happens to be visible, so it is the easiest to debug by eye,
but the machinery is identical for any advected attribute.

It is worth being precise about how this differs from a genuine second material. In [[material-showcase]]
three materials share one solver, but there each material carries its own constitutive law and the physics
reads that law every step, so fluid, elastic, and snow move differently. A dye is the opposite. Two dyed
liquids are the *same* material wearing two coats of paint; the color changes nothing the solver computes.
That is exactly why the separation stays clean and why the blend is purely geometric. A passive tracer is
the degenerate, zero-feedback end of the material-label spectrum, and it is the right tool whenever the goal
is to watch where stuff goes rather than to make different stuff behave differently.

## Intuition: the particles already carry identity

The whole trick rests on one fact about a particle method that [[mpm-in-context]] states plainly. A particle
is a persistent lump of material that keeps its identity for the entire simulation. Node values are scratch,
rebuilt from zero every step, but particle index $p$ always refers to the same lump of stuff, and whatever
attributes it carries, position, velocity, the affine matrix, ride along with it wherever the flow takes it.

That is already an advection scheme, and a very good one. To advect a dye is to move it along with the fluid,
and a Lagrangian particle does that for free: attach a color to particle $p$ once, at seeding, and it is
automatically transported, because the particle it is glued to is transported. Nothing needs to be
recomputed, diffused, or re-gridded. A red particle is red for all time; it simply ends up wherever the
velocity field carries it. Two populations seeded with two colors will therefore interpenetrate exactly as
the two bodies of fluid interpenetrate, and the color at any place is just the mixture of whichever
particles have arrived there.

This is worth contrasting with a grid-based (Eulerian) simulator, where a dye is a concentration field on the
mesh and advecting it means solving a transport equation every step, an operation that numerically smears the
dye a little each time (numerical diffusion). The particle carries its color with zero smearing. The blending
seen at the end is not the tracer diffusing; it is two sharply-colored populations becoming spatially
interleaved, and the eye reading their local average as a blend.

## The one rule: color is inert

For the tracer to be **passive**, the physics must never read it. The velocity update, the pressure, the
viscous stress, the grid transfers, all proceed exactly as in the single-fluid solver, blind to what color a
particle happens to be. If the color fed back into the dynamics, say red particles were made heavier or
stickier, the two liquids would be two different materials and the clean separation would be lost. Here they
are the same water with a coat of paint, so:

$$
\text{color}_p(t) = \text{color}_p(0) \quad \text{for all } t.
$$

The color is a constant per particle. It is set once and never updated inside the step. Everything
interesting in the final image, the swirls, the purple band, the plume, comes entirely from where the
particles move, which is decided by the ordinary dynamics, and from how the renderer averages their colors
locally. That division is the point: the sim moves the paint, the renderer looks at the paint.

## Compositing the local color into the body shade

The renderer in [[fluid-rendering]] tints the liquid body with a single deep water color and a single pale
shallow color, blended by optical thickness through the Beer-Lambert law. To let the body show the local
dye, that one fixed pair of colors is replaced by a **per-pixel** color read off the particles nearby. This
extension lives entirely in the GPU renderer of [[gpu-rendering]], and it reuses machinery that page already
built. The per-channel color splat is the same atomic scatter as the density splat, just carrying an RGB
value instead of a bare count; the blur is the same separable Gaussian; and the normalization divides by the
particle count blurred to the same width. Adding a visible dye costs a few small on-device kernels and a
handful of extra fields, and nothing crosses the host boundary that did not already.

The construction is a weighted local average, built from the same splatting the density uses. Splat each
color channel onto the render grid, weighting every particle by its own value in that channel, and separately
splat the plain particle count (the density). For channel $c$ (red, green, or blue), the local dye color at a
pixel $u$ is

$$
\bar{c}(u) = \frac{G_\sigma * \big(\sum_p \text{color}_{p,c}\,\delta(u - x_p)\big)}{G_\sigma * \big(\sum_p \delta(u - x_p)\big)},
$$

and each symbol earns its place. The numerator is the particle color splatted into the grid and blurred: it
is large where many strongly-colored particles of channel $c$ sit. The denominator is the particle density
splatted and blurred with the **same** Gaussian $G_\sigma$ of width $\sigma$ (the $*$ is convolution, the
blur). Dividing the two turns a raw sum into an average, so the result is the mean color of the particles in
the neighborhood of $u$, not merely a count of them. $x_p$ is the position of particle $p$ and
$\text{color}_{p,c}$ its fixed value in channel $c$; the delta $\delta(u - x_p)$ is the ideal point splat that
the histogram-and-blur implements in practice. Where only red particles are present the average is red; where
only blue, blue; where the two populations overlap in a neighborhood, the average lands in between, and that
in-between is the visible blend.

The blur width $\sigma$ is the one knob that sets how the mix reads, and it deserves a sentence of its own.
Make it too small and the average is taken over too few particles, so the mixed zone is a speckled noisy
seam. Make it too large and color bleeds far past where the liquids actually meet, washing the whole body
toward a uniform muddy tone. There is a subtlety worth stating plainly. Because an equal-density,
non-diffusive fluid tends to hold a fairly sharp interface between the two dyed populations (the physics
never mixes them at the sub-particle scale), a color blur matched exactly to the density blur leaves a seam
that is technically correct but reads as almost hard-edged. Setting the color blur a bit wider than the
density blur, a couple of times the density sigma here, deliberately softens that interface into a blend
band a handful of particle-spacings wide, wide enough to read as a smooth transition and narrow enough that
the two source colors stay legible on either side. The denominator count is blurred at that same wider
width so the ratio is still a proper local mean and does not distort the pure regions. The honest failure
modes are the two ends of this knob: a hard seam that never blends, and a gray wash that blends everything.

That local color then drives the depth tint directly. The deep-body color becomes a saturated version of
$\bar{c}(u)$ and the thin-film color a pale version of the same local hue, and the existing Beer-Lambert
blend by thickness is left untouched. So a thick slug of red water still darkens to a deep saturated red and
a thin sheet of it stays pale pink, exactly as the single-color renderer did, but now the hue is whatever the
local paint says rather than a global constant. Refraction, Fresnel, specular, and foam all sit on top
unchanged, because none of them care what color the body is.

![Two faucets, one dyed red and one dyed blue, angled toward each other so their streams converge and pour
together into a single tank. Each particle keeps its color for the whole run, so the streams stay clearly red
and blue where they fall, and a band of purple grows in the churn between them where the two populations
interleave. The color is never touched by the physics; only the motion of the particles and the renderer's
local averaging produce the
blend.](/api/data/learning-taichi/runs/realistic-rendering/more-realistic-basic-fluid-sims/color_faucets_hero.png)

## Keeping a long weakly-compressible rollout honest

Showing two liquids actually mixing takes time. A splash that is over in a second barely interleaves the two
populations, so these scenes run for many thousands of steps rather than a few hundred, and a long rollout
raises a stability worry the short clips never had. A weakly-compressible fluid stores its incompressibility
as a stiff pressure that resists changes in the tracked volume ratio $J$, and over a long run two slow errors
can accumulate. Energy can drift, because the particle-to-grid and grid-to-particle transfers are not exactly
energy-conserving and a FLIP admixture (the re-injection of particle velocity described in [[fluid-realism]])
adds energy on purpose to fight numerical damping. And the volume ratio $J$ carried on each particle,
integrated step after step from the velocity divergence, can wander far from one if the flow is never quite
divergence-free. Either can turn a calm late frame into grid-scale noise or a slow blow-up.

Three cheap choices keep the long rollout physical, and each is a small lesson in its own right. A **thin
viscosity** is kept even on the water-like scenes, the same $\mu_{\text{visc}}(C + C^\top)$ strain-rate stress
from [[viscosity]] at a small coefficient; it costs almost nothing visually but drains the grid-scale
velocity noise that would otherwise seed a blow-up, acting as a stabilizer as much as a material property.
The **FLIP fraction is kept modest**, high enough to keep splashes lively but low enough that the energy it
injects does not outrun what viscosity and boundaries remove, so the fluid can actually settle instead of
buzzing forever. And every rollout is **checked for finiteness and sampled at late frames**, because the only
reliable way to catch a slow drift is to look at the end, not the middle. A degenerate or blown-up final
frame is a bug to fix, by shrinking the timestep, lowering the FLIP fraction, or adding a touch more
viscosity, not a result to ship.

There is a deeper reason the settling itself matters here. A single blob released in a closed box loses its
energy to these same dissipative transfers within about a second and then sits flat, which is fine for a
short clip but leaves a long one mostly dead. Sustained motion, and therefore sustained mixing, needs
sustained forcing: a faucet that keeps feeding fresh fluid, or an inherently slow material like a thick
[[viscosity]] honey whose own sluggishness stretches the motion over the whole clip. The mixing scenes use
the faucet, two colored inlets switched on a few particles at a time so a stream falls continuously, and the
continual arrival of new red and blue is what keeps stretching and folding the interface long enough for a
real blend to develop.

## What this does and does not model

The register is the honest one [[fluid-realism]] insists on: this is a **passive color advection**, not a
model of how real liquids mix. Real mixing runs on molecular diffusion and turbulent stirring, none of which
is present; the blend here is purely two particle populations interleaving by the resolved flow, plus the
renderer averaging their colors over a hand-tuned blur width. Nothing crosses the interface at the particle
level, so at infinite resolution the blend would resolve back into interleaved red and blue rather than a
true continuum. What it demonstrates cleanly is the tracer pattern: attach a quantity to the particles, let
the dynamics move it and never read it back, and recover a smooth field by local averaging at render time.
The same three lines carry any advected attribute a controllable world needs to track.
