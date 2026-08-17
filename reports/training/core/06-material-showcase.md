# Four materials, one solver: fluid, elastic, snow, sand

The [[constitutive-models]] page makes an abstract claim: every material enters MLS-MPM through a single
slot, the particle stress the P2G scatter writes into grid momentum, and swapping the formula in that
slot turns the same solver into water, rubber, snow, or sand. This page cashes that claim out visually.
One scene, four stress laws, everything else held fixed.

The one idea to take away is that **a material is defined less by how hard it pushes back than by which
deformations it refuses to remember.** Stiffness is a scalar dial that changes how emphatic a material
is; the *set of deformations it permanently forgets* is what changes what it **is**. Fluid forgets all
shear instantly. Elastic forgets nothing. Snow forgets any stretch past a fixed limit. Sand forgets any
shear it is not being squeezed hard enough to hold. Those four sentences predict everything below.

The recognition matters because a controllable, differentiable world model is only worth building if its
dynamics look like the world. Before trusting gradients to steer such a thing, it is worth seeing the
forward physics produce the right *character* first.

## The setup: same start, four stress laws

Each scene begins from an identical initial condition, the same particle positions and zero initial
velocity, and is integrated forward under one of four constitutive models. The transfer skeleton
(particle-to-grid, grid update, grid-to-particle) is the shared MLS-MPM step from [[mpm-in-context]];
only the stress branch and the state each material carries change:

- **Fluid** carries a single scalar, the volume ratio $J$ (how compressed the material is), and develops
  an isotropic pressure $\sigma^{\text{fluid}} = -k\,(J-1)\,I$ with stiffness $k = 4E$.
- **Elastic** carries the full deformation gradient $F$ (a $2\times 2$ matrix recording stretch, shear,
  and rotation) and develops the corotated stress $2\mu\,(F-R) + \lambda\,(J-1)\,J\,F^{-\top}$, where $R$
  is the rotation extracted from $F$.
- **Snow** carries $F$ as well, but each step clamps its principal stretches back into a fixed band and
  pushes the excess into a permanent plastic record, so it yields and stays yielded.
- **Sand** carries $F$ and a plastic record too, but its admissible set is a **cone** rather than a box:
  the shear it can hold is proportional to the pressure it is under, and it can carry no tension at all.

Every material is integrated to the **same physical time**, so the panels are synchronized clocks rather
than matched step counts. That matters because the materials run at different timesteps for stability
(see [[material-stiffness]]), and comparing at equal step count would quietly show each one a different
moment in its own motion.

## Drop and splat

A disk is released above the floor, falls under gravity, and hits it. All four fall identically (gravity
does not care about stress), and then the stress law takes over at impact.

![Four materials dropped from the same disk, played side by side, water then elastic then snow then sand.
Water spreads into a wide flat puddle; elastic springs back into a compact rounded blob; snow lands as a
low crumpled heap; sand splats out into a broad low deposit that is wider than snow but keeps a visible
mound rather than levelling like the water.](/api/data/learning-taichi/runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid/drop_four_alone.mp4)

**Fluid spreads into a flat puddle.** Its stress depends only on $J$, the current compression, and
nothing in it remembers shape. Two parcels of fluid that reach the same volume feel the same stress no
matter how they were sheared to get there, so there is no restoring force back toward a blob. Shear
resistance in a solid comes from the $\mu$ term, and a fluid has effectively $\mu = 0$, which is the
one-line reason a liquid cannot hold a shape.

**Elastic squashes, then springs back.** Its stress is a function of the whole deformation gradient $F$,
and the shear term $2\mu\,(F-R)$ stores energy in any part of the deformation that is not a pure
rotation. On impact the disk flattens, loading that term like a spring, and the stored energy pushes it
back toward its original round shape, overshooting into a damped jiggle before settling. The rotation
$R$ is what lets a piece of the blob tumble without generating stress, and it comes from the SVD of $F$
as $R = UV^{\top}$; [[svd-polar]] derives why that is the stable way to pull a rotation out of a
deformation.

**Snow crumples and holds the dent.** Snow uses the same elastic stress but, every step, clamps the
singular values of $F$ (its principal stretches) back into a narrow band,

$$
\hat\sigma_k = \operatorname{clamp}\big(\sigma_k,\; 1-\theta_c,\; 1+\theta_s\big), \qquad k = 1, 2,
$$

where $\theta_c$ is the compression limit and $\theta_s$ the stretch limit, and moves whatever it clamped
off into a permanent plastic record. Deformation past the limit is therefore *not* recoverable. What
lands is a crumpled heap that keeps its dent.

**Sand splats and stays down.** It travels much further than snow because a granular pack that has been
thrown apart is under almost no pressure, and with no pressure it has no shear strength, so it flows
freely while it is spreading. It stops well short of the water because once it has piled up, its own
weight provides the confinement it needs to lock. That combination, mobile while dispersed and strong
once packed, is the whole character of a granular material.

## The over-steep heap: the test that separates all four

The drop scene is a poor discriminator, because everything looks like a splat at impact. The test that
actually separates the four is to seed a pile **steeper than any of them was ever going to hold** and let
it relax. Whatever slope is left at the end is the slope the material genuinely supports.

![The same 60 degree triangular heap released from rest under each of the four materials, played side by
side with the measured surface slope on each panel. Water immediately runs out flat to near zero slope.
Elastic and snow both keep the full seeded triangle essentially unchanged. Sand slumps partway and stops,
settling into a lower, wider heap that still holds a clear
slope.](/api/data/learning-taichi/runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid/heap_four_alone.mp4)

Water flattens, as it must. **Elastic and snow both keep the whole seeded slope**, and that is the
result worth pausing on, because it says something about snow that the drop scene hides. Snow's yield
criterion is a fixed box on the elastic stretch. It does not shrink as the confining pressure drops, so
snow at zero pressure is still strong: it has **cohesion**, and a cohesive material can stand a vertical
wall.

**Sand relaxes to a finite slope and stops.** That slope is the **angle of repose**, and having one is
the definition of a cohesionless granular material. Sand's admissible set is a cone whose width is
proportional to pressure, so near the free surface, where the pressure is nearly zero, it can carry
almost no shear and the grains slide. Deeper in the pile, weight supplies confinement and the material
locks. The slope where those two effects balance is the angle of repose, and it emerges from the model
rather than being prescribed anywhere in it.

A pile can only hold a slope if something under it resists sliding, so the floor carries Coulomb
friction. Raising that floor friction twentyfold barely changes the settled angle, though, which says the
limit is the sand's own internal strength and not the boundary condition.

## The friction angle is not the angle of repose

Sand's cone has exactly one shape parameter, the internal friction angle $\varphi$. It is tempting to
read it as "the slope the material will hold". It is not, and the gap is large: the angle a settled heap
actually holds comes out roughly half the $\varphi$ that produced it.

![Measured angle of repose against the Drucker-Prager friction angle parameter, with error bars over four
random seeds and three pile sizes. The measured angle rises smoothly and tightly with the parameter up to
about 50 degrees, always well below it, and past about 52 degrees the error bars explode and the curve
stops being reproducible.](/api/data/learning-taichi/runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid/phi_calibration.png)

Two lessons sit in that figure, and the second is the more valuable one.

The first is that a **model parameter is not the observable it is named after**. $\varphi$ sets a yield
surface in stress space; the angle of repose is an emergent property of a whole settled pile, produced by
that surface together with the flow rule, the resolution, and the dynamics of how the pile got there. The
only honest way to state a material's repose angle is to measure it.

The second is visible on the right of the plot. Past a certain friction angle the measured slope stops
being reproducible: it scatters by several degrees between random seeds and, worse, it changes
systematically with the **size of the pile in grid cells**, a smaller pile reading steeper. A quantity
that depends on how many cells the object spans is not a material property, it is a discretisation
artifact wearing one's clothes. The usable range is the range where the measurement is flat in both, and
the honest place to freeze a canonical parameter is the far end of *that*, not wherever the largest
number happens to appear.

## The stiffness dial, seen directly

One more knob is worth watching move. Holding the material fixed as elastic and turning only Young's
modulus $E$ changes how hard it resists deformation. Dropping the same disk at $E = 50$, $400$, and
$1600$ shows the effect at the instant of impact.

![Peak-impact frame of the same elastic disk dropped at three stiffnesses. The soft blob on the left has
pancaked almost flat against the floor. The middle blob has squashed moderately. The stiff blob on the
right has barely dented and holds a nearly round shape.](/api/data/learning-taichi/runs/material-variants/implement-nondifferentiable-material-variants/stiffness_still.png)

The soft blob pancakes flat, because a small $E$ means a small restoring stress at the same strain, so it
takes a large deformation to balance the impact. The stiff blob barely dents and rings back fast. This is
the visible half of the story [[material-stiffness]] tells in full, where the same parameter also sets the
oscillation frequency, the stable timestep, and, once gradients enter, the usable learning rate.

Notice that this dial does **not** move a material across the four categories. No value of $E$ turns
elastic into sand. Stiffness is quantitative; the yield set is categorical.

## What's open

The settled shape of the three non-elastic materials is not converged in the timestep. Elastic gives an
identical settled heap across a thirty-fold range of timestep, and fluid, snow, and sand all slump
further as the timestep is made smaller, with snow the worst affected. [[real-time-cost]] works through
what is happening and why it means a plastic material's "strength" has to be quoted with the timestep it
was measured at. Until that is resolved, an angle of repose from this solver is a number about a
discretisation as much as about a material.

Separately, four materials sharing one grid share one velocity field per node, which means two different
materials meeting at a node exchange momentum as though the node held a single blended material. That is
enough to put them in the same scene and it is not a contact model. What sand does at a water interface
is, in this solver, an artifact rather than a prediction.
