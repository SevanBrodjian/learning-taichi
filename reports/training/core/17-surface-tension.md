# Surface tension: the force that minimizes the interface

The fluid in [[material-showcase]] splashes, and [[viscosity]] made it ooze from oil to honey by resisting
the *rate* of shear. Both of those are **bulk** effects, felt throughout the body of the fluid. There is a
second, entirely different knob a liquid has, and it lives only at the **surface**: surface tension, the
force that pulls a blob into a round droplet, beads scattered fluid into drops, and lets a stretched sheet
retract into a ball. Where viscosity changes how *fast* a liquid moves, surface tension changes what
*shape* it settles into. This page adds it to the same solver and watches a square relax into a droplet.

![A square blob of fluid with gravity switched off, at three surface tensions. At sigma zero (left) it
stays a blocky square. At medium surface tension (middle) it has pulled its corners in and rounded. At
high surface tension (right) it is a clean disk. Only the surface-tension parameter differs across the
three panels.](/api/data/learning-taichi/runs/material-variants/implement-liquids-across-viscosity-and-surface-tension/isolation_round_still.png)

Surface tension is the signature that reads a fluid as **droplets and cohesion** rather than a formless
splash, and it enters the solver through a different door than every bulk stress so far: viscosity slotted
into the particle stress, while surface tension is computed from the **grid's density field** and pushes only
on the interface band. That they are different mechanisms, not two dials on one, is most of the value here.

## Intuition: a skin that wants to be small

A liquid holds together because its molecules attract their neighbours. Deep inside the fluid a molecule is
pulled equally in all directions and feels no net force. A molecule at the surface has neighbours on the
inside only, so it is pulled inward, and the collective effect is that the surface behaves like a stretched
elastic skin that is always trying to **shrink its own area**. The shape that encloses a given amount of
fluid with the least surface is a sphere (a circle in 2D), so left alone the skin pulls the blob toward
round. A square has excess perimeter bunched at its corners, so the skin pulls the corners in until the
perimeter is uniform, which is a disk. Two nearby drops have more total surface than the single larger drop
they could form, so the skin draws them together and merges them. A long thin bar has enormous surface for
its volume, so it retracts lengthwise into a compact droplet. Every one of these is the same sentence:
**minimize the interface**.

The force is concentrated **where the surface is curved**, and it is stronger the sharper the curve. That
is the physical content of the Young-Laplace law: a curved interface supports a pressure jump proportional
to its curvature, so a tightly curved spot (a corner, a small droplet) is squeezed harder than a gently
curved one. A flat interface has no curvature and feels no surface-tension force at all, which is why the
effect is invisible in the bulk and shows up only at the free boundary.

## The math: a continuum surface force from the density field

To turn "minimize the interface" into a force the grid can apply, the interface first has to be *located*,
and MPM hands over the material needed for free: the grid already carries a density field after the
particle-to-grid scatter. The **continuum surface force** (CSF) method builds everything from it.

Give the fluid a **color** that is $1$ where there is fluid and $0$ where there is not. Scattering it to the
grid and normalising by the packed-interior density gives a field $\phi$ that is near $1$ inside the fluid
and falls to $0$ across the free surface. A raw $\phi$ drops from $1$ to $0$ over a single cell, too sharp
for a stable derivative, so it is **smoothed** over a few cells into a diffuse band, which is what makes
this a *continuum* (spread-out) rather than a sharp-interface force. The three operators from
[[vector-calculus]] then read the surface geometry straight off $\phi$:

$$
n = \frac{\nabla\phi}{\lVert\nabla\phi\rVert}, \qquad
\kappa = -\nabla\cdot n, \qquad
f = \sigma_{st}\,\kappa\,\nabla\phi.
$$

Each factor earns its place. The **normal** $n$ is the unit gradient of $\phi$; because $\phi$ increases
into the fluid, $n$ points inward across the interface, giving the surface a well-defined orientation with
no boundary tracking. The **curvature** $\kappa = -\nabla\cdot n$ is the divergence of that normal field
(derived in [[vector-calculus]]): for a convex droplet of radius $R$ it equals $1/R$, so it is large at a
tight corner and small along a flat edge. The **force per unit volume** $f = \sigma_{st}\,\kappa\,\nabla\phi$
multiplies the curvature by the strength $\sigma_{st}$ and by $\nabla\phi$, and that last factor is the key
to *where* the force acts: $\nabla\phi$ is nonzero only in the thin band where $\phi$ is changing, so the
force is switched on exactly at the interface and is zero everywhere in the bulk. This is the mathematical
statement that surface tension is a skin effect. The force is added to the grid velocity as an acceleration,
$v \mathrel{+}= \Delta t\, f / \rho$, in the same grid update that applies gravity.

Setting $\sigma_{st} = 0$ removes the term entirely and recovers the viscous fluid exactly, which is both
the sanity check and the top row of the grid below.

The one subtlety that must be handled is that surface tension is an **internal** force: it can reshape a
droplet but must not shove the whole droplet across the domain, because it exerts no net force on the fluid
as a whole. The discretised $f$ leaks a small spurious net force (from the smoothed, finite-difference
curvature not being perfectly symmetric), so the mass-weighted mean of the capillary acceleration is
subtracted from every fluid node. That subtraction removes the bulk drift exactly while leaving the
shape-changing part intact. Skipping it lets a rounding droplet wander off like a thruster fired sideways, a
tell-tale sign the force was not made net-zero.

## Contrast with viscosity: shape versus speed

It is worth pinning down precisely how this differs from [[viscosity]], because both are "add a term to the
fluid" and they are easy to conflate. Viscosity is a **bulk stress** built from the strain rate
$C_p + C_p^\top$ carried on every particle; it dissipates the rate of shear *everywhere* in the fluid and
changes how *fast* the fluid moves, draining kinetic energy as momentum diffusion. Surface tension is a **conservative
capillary force** built from the curvature of the free boundary; it acts *only* at the interface, adds and
removes no bulk energy, and changes the *shape* the fluid settles into. Honey is viscosity: it moves slowly
but still splashes flat given time. A water droplet beading on wax is surface tension: it moves freely but
holds a round shape. One is a dashpot spread through the body, the other is a skin on the boundary.

## What the parameter does: rounding, beading, and a capillary timestep

The two axes are genuinely independent, which the grid below makes visible. It drops the same blob onto the
floor at three viscosities (rows) and three surface tensions (columns).

![A 3 by 3 grid of the same dropped blob. Rows increase viscosity from top to bottom, columns increase
surface tension from left to right. The left column has no surface tension: the low-viscosity oil spreads
into a flat wide puddle, the medium into a lower mound, the high viscosity into a compact dome. Every panel
with surface tension, the middle and right columns, has pulled the fluid into a compact rounded droplet
sitting on the floor. Viscosity orders the left column by how far it spreads; surface tension rounds every
row.](/api/data/learning-taichi/runs/material-variants/implement-liquids-across-viscosity-and-surface-tension/grid_still.png)

Read down the surface-tension-free left column and the familiar viscosity ordering appears: oil spreads
widest, honey barely spreads. Read across any row and surface tension pulls the puddle up into a rounded
droplet and cuts how far it spreads. The two knobs move the fluid along different axes, one setting speed
of spreading, the other setting roundness and cohesion, and that separability is the whole point of putting
them on one grid.

There is a second thing $\sigma_{st}$ does, and like viscosity's diffusion limit it lives in the numerics.
The explicit capillary force supports **capillary waves** on the interface, and resolving them stably caps
the timestep at roughly

$$
\Delta t \;\lesssim\; \sqrt{\frac{\rho\,\Delta x^{3}}{2\pi\,\sigma_{st}}},
$$

where $\rho$ is density and $\Delta x$ the grid spacing. The reading that matters is the inverse-square-root
dependence on $\sigma_{st}$: a stiffer interface carries faster capillary waves, so a stronger surface
tension forces a smaller step, and the high-$\sigma_{st}$ cells of the grid run more substeps to reach the
same physical time. This mirrors [[viscosity]] exactly in spirit, where a stronger diffusion tightened the
step through $\mu/\Delta x^2$. A single scalar sets both a visible physical behavior and a numerical
stability limit, and reading a parameter in both registers at once is most of what it means to understand a
simulation knob. A frame with particles flung to the domain corner is this capillary limit being violated,
not a fluid.

## What's open

This is a forward demonstration of one surface-tension model at one resolution, so it shows what the
parameter does, not a calibrated capillarity. The map from $\sigma_{st}$ to a physical surface tension is
not established, so the labels are evocative and only the monotonic rounding and the axis separation are
claimed. The curvature comes from a smoothed indicator, so the interface has a finite thickness of a few
cells rather than being sharp, and the measured curvature depends on the smoothing and the grid; a genuine
sharp free surface would need a reconstruction the diffuse-interface method deliberately avoids. The honest
open questions are whether $\sigma_{st}$ can be calibrated against the Young-Laplace pressure jump of a
static droplet, and whether the more dramatic capillary phenomena (a jet breaking into a regular train of
drops, a thin sheet retracting) survive a resolution sweep or are limited by this diffuse band.
