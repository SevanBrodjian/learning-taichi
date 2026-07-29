# Viscosity: resistance to the rate of shear

The fluid in [[material-showcase]] splashes freely. Dropped on the floor it spreads into a wide flat
puddle almost at once, and released as a column it runs out into a thin sheet. That is the behavior of a
liquid with **no viscosity**, and it is the honest default of the weakly-compressible model, whose only
stress is a pressure that resists compression and nothing else. Real liquids are not all like that. Water
is close, but oil is thicker, syrup thicker still, and honey barely moves. The single property that
separates them is **viscosity**, the resistance a fluid offers to being sheared, and this page adds it to
the same solver and watches the fluid slow from oil to honey.

The reason this belongs in a book about controllable, differentiable worlds is that viscosity is one of
the cheapest, most recognizable knobs a fluid has. A world model that wants a liquid to read as *this*
liquid, a slow ooze rather than a splash, needs exactly one extra term in the stress, and that term turns
out to be the rate-analogue of the elastic shear stress the solids already use. It is a clean example of
how much visible behavior a single well-chosen term buys.

## Intuition: shear versus the rate of shear

The elastic solid in [[constitutive-models]] resists **how much** it has been deformed. Push it out of
shape and it pushes back with a force that grows with the size of the deformation, and it holds that force
for as long as the shape is held. A viscous fluid is different in one word. It resists **how fast** it is
being deformed. Stop moving and a viscous fluid stops pushing back. It has no memory of shape and no rest
configuration to spring toward. What it has is internal friction, a resistance that exists only while the
material is in motion and that is proportional to the *rate* at which neighboring parcels slide past each
other.

That is the whole physical idea. Honey poured from a spoon resists the fast shearing near the spoon's edge
and so pours slowly; left in a jar it eventually levels, because a slow enough shear meets almost no
resistance. Stir water and it swirls on; stir honey and the motion dies almost as soon as the spoon stops,
because the viscous stress has drained the shearing energy into heat. The technical name for that draining
is **momentum diffusion**. A viscous stress takes momentum from a fast-moving parcel and hands it to its
slower neighbor, smearing out sharp differences in velocity the same way heat conduction smears out sharp
differences in temperature. A thick fluid diffuses momentum quickly, so velocity differences cannot
survive, and the fluid moves as a sluggish whole rather than splashing into fast thin sheets.

## The math: a stress built from the strain rate

To turn "resistance to the rate of shear" into a stress, the object needed is the **velocity gradient**,
the matrix that records how the velocity changes from place to place. Write it $\nabla v$, a $2\times 2$
matrix (in 2D) whose entry in row $a$, column $b$ is $\partial v_a / \partial x_b$, the rate at which the
$a$-component of velocity changes as one moves in the $b$-direction. A pure translation has
$\nabla v = 0$ (every parcel moves together, no shearing), and any nonzero $\nabla v$ means neighboring
parcels are moving relative to each other.

Not every part of $\nabla v$ is a genuine deformation rate. A fluid rotating rigidly, like a spinning
disk, has a nonzero velocity gradient, but nothing is actually being stretched or sheared, so a physically
sensible viscosity must develop no stress from it. The way to strip the rotation out is to split the matrix
into its symmetric and antisymmetric parts,

$$
\nabla v = \underbrace{\tfrac{1}{2}\big(\nabla v + \nabla v^{\top}\big)}_{D,\ \text{strain rate}}
        + \underbrace{\tfrac{1}{2}\big(\nabla v - \nabla v^{\top}\big)}_{W,\ \text{spin}}.
$$

The antisymmetric part $W$ is the local rotation rate and carries no shape change, so it is discarded. The
symmetric part $D$ is the **strain rate**, and it is exactly the object a Newtonian viscosity acts on.
Reading $D$ through [[linear-algebra]] makes its two jobs concrete. Its **trace** $\operatorname{tr} D$ is
the divergence of the velocity, the rate at which the parcel is expanding or compressing, and its
**off-diagonal** entries are the rate of shearing, neighboring parcels sliding past one another. A
symmetric matrix is pure stretch along perpendicular directions, so $D$ says precisely how fast the
material is being stretched and sheared, with the spinning thrown away.

A **Newtonian** fluid is defined by the simplest possible law relating stress to that rate. The viscous
stress is linear in the strain rate,

$$
\sigma^{\text{visc}} = 2\,\mu_{\text{visc}}\,D = \mu_{\text{visc}}\,\big(\nabla v + \nabla v^{\top}\big),
$$

where the scalar $\mu_{\text{visc}}$ is the **dynamic viscosity**, the one number that says how thick the
fluid is. Water has a small $\mu_{\text{visc}}$, honey a large one. The parallel with the elastic solid is
worth pausing on, because it is the cleanest way to remember the whole thing. The corotated solid develops
a shear stress $2\mu\,(F-R)$ proportional to the shear **displacement** (how far it has been pushed out of
shape); the Newtonian fluid develops a shear stress $2\mu_{\text{visc}}\,D$ proportional to the shear
**rate** (how fast it is being pushed). One is a spring, the other is a dashpot. Same slot in the solver,
same symmetric strain-like object, one built from position and one from velocity.

The total fluid stress is then the old pressure plus the new viscous term,

$$
\sigma = \underbrace{E\,(J-1)\,I}_{\text{pressure, resists volume change}}
       + \underbrace{\mu_{\text{visc}}\,\big(\nabla v + \nabla v^{\top}\big)}_{\text{viscosity, resists shear rate}},
$$

where $J$ is the volume ratio, $E$ the weakly-compressible stiffness, and $I$ the identity. The pressure
term is unchanged from [[constitutive-models]]. Setting $\mu_{\text{visc}} = 0$ recovers the inviscid fluid
exactly, which is both the sanity check and the reason the low-viscosity limit must look like the old
splashy fluid.

## Implementation: the affine matrix already carries the velocity gradient

The clean surprise is that MLS-MPM already computes the velocity gradient it needs, for free, every step.
The APIC affine matrix $C_p$ carried on each particle is precisely an estimate of $\nabla v$ around that
particle (derived in [[mpm-in-context]] and [[linear-algebra]]). So the viscous stress needs no new field
and no finite differences. It is just

$$
\sigma^{\text{visc}}_p = \mu_{\text{visc}}\,\big(C_p + C_p^{\top}\big),
$$

the symmetric part of the affine matrix already on the particle, added straight onto the pressure and
scattered to the grid by the ordinary particle-to-grid step. Viscosity costs one matrix add and changes
nothing else in the pipeline. This is the payoff of the "one stress slot" view from [[constitutive-models]]:
a whole material property enters through the same door as the pressure.

## What the parameter does: oil to honey, and the price in timestep

Turning the single scalar $\mu_{\text{visc}}$ up across two decades walks the fluid from a thin oil to a
thick honey. On a dam-break, an identical column of fluid released against a wall, the leading front tells
the story cleanly.

![Late frame of a dam-break at three viscosities, oil on the left, syrup in the middle, honey on the
right, from the same released column. The thin oil has run out into a long low sheet reaching most of the
way across the floor. The syrup has advanced only partway. The thick honey has barely moved from the wall
and still stands as a tall rounded block. Only the viscosity differs across the three panels.](/api/data/learning-taichi/runs/material-variants/varying-liquid-viscosity/dam_still.png)

The thin oil collapses and runs its front nearly across the domain; the syrup advances partway; the honey
creeps out with a steep rounded front and keeps most of its height. Measured as the front position over
time, the three curves are monotonic in viscosity and never cross, which is the quantitative face of what
the eye sees. The mechanism is exactly the momentum diffusion above. A larger $\mu_{\text{visc}}$ drains
the shearing motion faster, so at any fixed physical time a thicker fluid has spent more of its energy
fighting its own internal friction and less on spreading.

There is a second, less obvious thing the parameter does, and it lives entirely in the numerics. The
viscous term is a **diffusion**, and an explicit (forward-in-time) diffusion is only stable when a parcel
cannot diffuse more than about one grid cell in a single step. That caps the timestep at roughly

$$
\Delta t \;\lesssim\; \frac{\rho\,\Delta x^{2}}{\mu_{\text{visc}}},
$$

where $\rho$ is the density and $\Delta x$ the grid spacing. The important reading of this formula is the
inverse dependence on $\mu_{\text{visc}}$. Making the fluid a hundred times thicker forces the stable
timestep down by a comparable factor, so honey costs far more substeps to reach the same physical time
than oil does. This is not a tuning nuisance to hide, it is the same viscous term showing its face in the
solver: the stronger the momentum diffusion physically, the tighter the stability limit numerically. In
practice the thin case runs at a timestep of about $10^{-4}$ while the thick case needs a few times
$10^{-6}$, and pushing the viscosity higher still would make an explicit scheme impractical, which is the
standard motivation for solving viscosity implicitly. A blown-up honey frame, particles flung into the
domain corner, is this stability limit being violated, not a property of the fluid.

The pattern is the familiar one from [[material-stiffness]]. A single scalar sets a visible physical
behavior and, through the same term, a numerical stability limit, and the two are not independent choices.
Stiffness tied wave speed to the timestep through $\sqrt{E}$; viscosity ties momentum diffusion to the
timestep through $\mu_{\text{visc}}/\Delta x^{2}$. Learning to read a parameter in both registers at once,
what it does to the picture and what it does to the stable step, is most of what it means to understand a
simulation knob.

## What's open

This is a forward demonstration with a single Newtonian model, so it shows what viscosity does to this
fluid, not a calibrated rheology: the map from $\mu_{\text{visc}}$ to a physical viscosity is not
established, so the labels are evocative and only the monotonic ordering is claimed. The genuinely open
questions are whether an **implicit** viscosity solve can reach honey-and-beyond without the punishing
explicit timestep, and whether the more dramatic viscous phenomena (a falling stream coiling into a rope)
need surface tension and a real free-surface reconstruction rather than just more viscosity.
