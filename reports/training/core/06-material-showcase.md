# Three materials, one solver: what fluid, elastic, and snow look like when they move

The [[constitutive-models]] page makes an abstract claim: every material enters MLS-MPM through a single
slot, the particle stress the P2G scatter writes into grid momentum, and swapping the formula in that slot
turns the same solver into water, rubber, or snow. This page cashes that claim out visually. It takes one
scene, runs it forward under three stress laws with everything else held fixed, and watches what each does.
No gradients, no optimization, no loss appear here. This is the physics before any of that, the part a
person can just look at and recognize.

The recognition is the point. A controllable, differentiable world model is only worth building if its
dynamics look like the world, and the constitutive model is the knob that decides whether a blob reads as
water, jelly, or packed snow. Before trusting gradients to steer such a thing, it is worth seeing the
forward physics produce the right *character* first. It does, and the reasons are exactly the three stress
laws.

## The setup: same start, three stress laws

Each scene begins from an identical initial condition, the same particle positions and zero initial
velocity, and is then integrated forward under one of three constitutive models. The transfer skeleton
(particle-to-grid, grid update, grid-to-particle) is the shared MLS-MPM step from [[mpm-in-context]]; only
the stress branch and the state each material carries change:

- **Fluid** carries a single scalar, the volume ratio $J$ (how compressed the material is), and develops an
  isotropic pressure $\sigma^{\text{fluid}} = -k\,(J-1)\,I$ with stiffness $k = 4E$.
- **Elastic** carries the full deformation gradient $F$ (a $2\times 2$ matrix recording stretch, shear, and
  rotation) and develops the corotated stress $2\mu\,(F-R) + \lambda\,(J-1)\,J\,F^{-\top}$, where $R$ is the
  rotation extracted from $F$.
- **Snow** carries $F$ as well, but each step projects it back into an allowed range and pushes the excess
  into a permanent plastic record, so it can yield and stay yielded.

Every material is integrated to the **same physical time**, so the panels below are synchronized clocks, not
just the same number of steps. That matters because snow runs at a smaller timestep for stability (its stress
law, like any stiff or non-smooth one, caps the stable step; see [[material-stiffness]]), and comparing at
equal step count rather than equal time would quietly show snow a different moment in its own motion.

## Drop and splat

A disk is released above the floor, falls under gravity, and hits it. The three materials fall identically
(gravity does not care about stress), and then the stress law takes over at impact.

![Settled frame of the drop-and-splat scene, one panel per material from the same dropped disk. Fluid on the
left has spread into a wide flat puddle across the floor. Elastic in the middle has sprung back into a
compact rounded blob. Snow on the right sits as a low crumpled heap that neither flowed flat like the fluid
nor bounced back like the elastic.](/api/data/learning-taichi/runs/material-variants/implement-nondifferentiable-material-variants/drop_still.png)

**Fluid spreads into a flat puddle.** Its stress depends only on $J$, the current compression, and nothing
in it remembers shape. Two parcels of fluid that reach the same volume feel the same stress no matter how
they were sheared to get there, so there is no restoring force back toward a blob. The material simply flows
downhill until it is as flat and level as the floor and walls allow. Shear resistance in a solid comes from
the $\mu$ term, and a fluid has effectively $\mu = 0$, which is the one-line reason a liquid cannot hold a
shape.

**Elastic squashes, then springs back.** Its stress is a function of the whole deformation gradient $F$,
and the shear term $2\mu\,(F-R)$ stores energy in any part of the deformation that is not a pure rotation.
On impact the disk flattens, loading that term like a spring, and the stored energy then pushes it back
toward its original round shape, overshooting into a damped jiggle before settling. The blob that lands is
almost the disk that fell, only wobbling. The rotation $R$ in that term is what lets a piece of the blob
tumble without generating stress, and it is computed from the SVD of $F$ as $R = UV^{\top}$; the
[[svd-polar]] page derives why that is the stable way to pull a rotation out of a deformation.

**Snow crumples and holds the dent.** Snow uses the same elastic stress but, every step, clamps the singular
values of $F$ (its principal stretches) back into a narrow band,

$$
\hat\sigma_k = \operatorname{clamp}\big(\sigma_k,\; 1-\theta_c,\; 1+\theta_s\big), \qquad k = 1, 2,
$$

where $\theta_c$ is the compression limit and $\theta_s$ the stretch limit, and moves whatever it clamped off
into a permanent plastic record. Deformation past the limit is therefore *not* recoverable. On impact snow
compacts, the clamp fires, and the packed-down shape is locked in rather than sprung back. What lands is a
crumpled heap that keeps its dent, exactly the behavior that makes snow (and mud, and foam, and wet sand)
read as itself and not as either water or rubber. The clamp is surgery on the singular values, and again
[[svd-polar]] is where that surgery is set up.

## Column collapse

The complement to a drop is a slump. A tall block is stood on the floor and released from rest, and again the
stress law decides the outcome. Fluid runs out into a flat sheet, because it cannot support its own shear and
flows until level. Elastic barely notices; it is a stiff spring, so it compresses a little, rebounds, and
stands nearly its full original height, wobbling. Snow is the interesting one: it slumps partway, the clamp
locks in the yielded shape, and the block settles into a stable pile with sloped sides, an **angle of
repose**, the way a real pile of granular material holds a slope instead of spreading to a puddle or
standing like a solid. (A pile holds a slope only if there is friction under it; the demo puts Coulomb
friction on the floor so snow can build its repose angle rather than sliding out on a frictionless sheet.)

## Snow is the in-between material

The three signatures also order cleanly on a simple diagnostic. Measuring final horizontal spread and pile
height on both scenes, fluid is always the widest and lowest, elastic the narrowest and tallest (it recovers
its shape rather than losing it), and snow sits between the two on both measures. That is the numerical face
of what the eye already saw: plasticity is almost definitionally the middle ground between fluid flow and
elastic recovery, and it shows up as an actual midpoint. (Hand-tuned 2D scenes at one resolution, snow on
softer settings, so the ordering is the takeaway, not the exact figures.)

## The stiffness dial, seen directly

One more knob is worth watching move. Holding the material fixed as elastic and turning only Young's modulus
$E$ changes how hard it resists deformation. Dropping the same disk at $E = 50$, $400$, and $1600$ shows the
effect at the instant of impact.

![Peak-impact frame of the same elastic disk dropped at three stiffnesses. The soft blob on the left has
pancaked almost flat against the floor. The middle blob has squashed moderately. The stiff blob on the right
has barely dented and holds a nearly round shape. All three spring back toward a disk afterward, so the
difference is sharpest at the moment of maximum squash.](/api/data/learning-taichi/runs/material-variants/implement-nondifferentiable-material-variants/stiffness_still.png)

The soft blob pancakes flat, because a small $E$ means a small restoring stress at the same strain, so it
takes a large deformation to balance the impact. The stiff blob barely dents and rings back fast, because a
large $E$ answers the same impact with a much larger stress and stores less deformation to do it. This is the
visible half of the story [[material-stiffness]] tells in full, where the same single parameter also sets the
oscillation frequency, the stable timestep, and, once gradients enter, the usable learning rate. Here it is
enough to see that one scalar slides a material smoothly from gel to hard rubber without changing anything
else about the solver.

## Why this matters

The visually richest materials, the ones that crumple, pack, and hold a shape, are exactly the ones whose
stress laws carry the most state and non-smoothness: a full deformation gradient for the solid, a per-step
plastic projection for snow. That richness is what makes them look real and, as [[constitutive-models]] and
[[svd-polar]] flag, the hard part to differentiate through. Seeing the forward behavior first is the honest
order of operations: a world model worth steering has to move like the world before it is worth asking
whether it can be steered.
