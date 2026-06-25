# MPM in context

> Prerequisite, but read it straight through the first time. This section builds the mental model of the
> Material Point Method from nothing. It assumes you have read the motivation group ([[where-mpm-sits]]
> in particular) and otherwise assumes no simulation background. Every term is defined before it is used.
> No equations live here. The math they rest on is in [[math-toolkit]], and the assembled step is
> [[mls-mpm-forward]].

## The cast of characters

MPM has exactly two kinds of object, and everything that happens is a conversation between them.

**Particles** are the material. Picture a blob of soft jelly as a few thousand dots scattered through
the region the jelly occupies. Each dot is a small lump of the material, and it stays a lump for the
whole simulation. It never disappears, never connects to its neighbors by any rigid link, and carries
with it a little packet of numbers describing the lump's current condition. The particles are the only
thing in the simulation that *persists*. If you froze time and wanted to redraw the scene or restart it,
the particles are all you would need. They are the source of truth.

**The grid** is a sheet of graph paper laid over the same region. A regular lattice of points, evenly
spaced, that does not move. In the code it is a $128 \times 128$ array of nodes in 2D. The grid is *not*
the material. No lump of jelly lives on the grid. The grid is a scratchpad, a temporary surface used for
one calculation per step and then wiped clean. This bears repeating because it is the thing newcomers
get wrong most often. Between steps, the grid holds nothing. All memory of the material lives on the
particles.

So why have a grid at all if it forgets everything. Because the calculation that has to happen each step,
working out the forces between neighboring bits of material, is painful to do directly on a scattered
cloud of particles and easy to do on a regular lattice. The grid exists purely to give the force
calculation a clean, regular place to happen. That is its entire job, explained more fully in
[[where-mpm-sits]].

## What a particle carries, and why

Each particle carries a small bundle of numbers. Take them one at a time, because each one answers a
specific physical question, and understanding *why* a particle stores each piece is most of the battle.

**Position.** Where this lump of material is right now. Obvious, but worth stating, because position is
the thing you actually see when you render the simulation, and it is the thing the control task in the
core sections is trying to steer.

**Mass.** How much material this lump represents. Mass is the bookkeeping that makes the physics
conserve the right quantities. In this simulator every particle has the same fixed mass, set once at the
start and never changed, so you can mostly treat it as a constant. Its job is to weight everything. A
heavier lump pushes harder and is harder to push.

**Velocity.** How fast this lump is moving and in what direction. This is the one that deserves a real
explanation, because *why a particle carries a velocity at all* is the question the task brief flagged as
unexplained, and it is genuinely the crux.

Here is the reasoning. The physics you want to simulate is Newton's law. Force changes momentum, momentum
is mass times velocity, and velocity changes position. To step a lump of material forward in time you
need to know how fast it is going so you can move it, and you need to know its momentum so that when it
interacts with its neighbors, the interaction conserves the total momentum the way real physics does.
Momentum is the conserved currency of motion. If a lump carries no velocity, it carries no momentum, and
there is nothing to conserve and nothing to move. So velocity is not an optional decoration. It is half
of what it means to be a moving piece of material. Position says where you are, velocity says where you
are going, and the simulation is the story of velocity bending under forces and dragging position along
behind it.

**Deformation.** How stretched, squeezed, or sheared this lump has become relative to its relaxed shape.
This is the one piece that is unfamiliar if your background is pure machine learning, so the next section
unpacks it on its own. The short version is that solid material *remembers its rest shape* and pushes
back when you deform it, the way a stretched rubber band pulls back, and the deformation record is what
lets each lump know how hard and which way to push. In this first model the entire deformation record is
compressed down to a single number per particle, the volume ratio, covered below.

That is the particle. Position, mass, velocity, deformation. A few thousand of these, each a tiny moving
witness to the material's history.

## Deformation, stress, and why solids push back

Spend a moment here because this is the physical heart of the method and the part with no machine-learning
analogue.

A fluid does not care about its shape. Pour water from one glass to another and it has no memory of the
first glass. A solid is different. A block of rubber has a *rest shape*, the shape it relaxes to when
nothing is pushing on it, and when you deform it away from that rest shape it stores energy and pushes
back to recover it. Squeeze it and it pushes out. Stretch it and it pulls in. Let go and it springs
toward rest. That push-back is what makes a solid a solid.

To simulate that, each lump of material has to know how far it currently is from its own rest shape, so
it can work out how hard to push back. **Deformation** is the name for "how far from rest, and in what
way," and **stress** is the name for the internal push-back force that the deformation generates. The
relationship between them, the rule that converts "how deformed" into "how much force and which way," is
called the **constitutive model**, and it is the place where the identity of the material lives. Soft
jelly and stiff steel run the exact same MPM machinery and differ only in their constitutive model. It is
the material's personality.

The full deformation of a lump is a matrix, because material can be stretched differently along different
directions and can be sheared, not just uniformly squeezed. The math toolkit builds that up properly. But
the first model in this project makes a deliberate simplification. It throws away everything about
deformation *except overall volume change* and tracks a single scalar per particle, the **volume ratio**,
usually written $J$. A value of $1$ means the lump is at its rest volume. Less than $1$ means it has been
compressed into a smaller volume, and the material wants to push back out. Greater than $1$ means it has
been expanded, and the material wants to pull back in. The constitutive model in this first slice is just
"the further $J$ is from $1$, the harder the lump pushes to restore its volume," which is enough to make
a weakly springy, weakly compressible blob. The precise update for $J$ and the stress it produces are in
[[math-toolkit]]. The point to carry forward is that deformation is the material's memory of its rest
shape, stress is the force that memory generates, and $J$ is the bare-minimum version of that memory.

## Transfers, the central idea

Now the word the brief singled out. A **transfer** is the act of moving information between the particles
and the grid. There are exactly two, one in each direction, and together they are the engine of MPM.

Remember the situation. The particles hold all the real state, but the force calculation wants to happen
on the regular grid. So every single step has to do a round trip. First, take what the particles are
carrying and lay it onto the grid so the grid has something to compute with. Then do the physics on the
grid. Then take the grid's answer and hand it back to the particles, which is the only place state is
allowed to persist. Those two handoffs are the transfers.

**Particle to grid**, written **P2G**, is the first handoff. Each particle deposits its mass and its
momentum onto the grid nodes near it. "Near it" matters. A particle does not dump everything onto the
single closest node. It spreads its contribution across a small neighborhood of surrounding nodes, giving
more to the nodes it is closest to and less to the ones further away, according to a smooth weighting.
That smooth spreading is what the **B-spline weights** in [[math-toolkit]] compute, and the smoothness is
not cosmetic. It is exactly what makes the whole transfer differentiable, because it means a particle
sliding a hair to the side changes the weights smoothly rather than snapping its allegiance from one node
to the next. Hold that thought, it pays off when gradients enter the story. Think of P2G as each particle
gently splashing its mass and momentum onto the graph paper underneath it, the splash widest right under
the particle and tapering off around it.

Once every particle has splashed, each grid node holds the *total* mass and *total* momentum it received
from all the particles that reached it. Now the grid has a complete, regular picture of the material's
mass and momentum, and the physics can run. That is the **grid update**. Convert each node's momentum
into a velocity by dividing out its mass, apply gravity so everything falls, and apply the boundary
conditions that stop material from flowing through the walls of the domain. After this the grid holds, at
each node, the new velocity that the physics says that patch of space should have.

**Grid to particle**, written **G2P**, is the second handoff and the reverse of the first. Each particle
looks at the same small neighborhood of grid nodes it splashed onto, reads back their freshly updated
velocities, and combines them with the same smooth weights to get its own new velocity. Then it uses that
velocity to move itself, updating its position, and to update its deformation record. After G2P the
particles once again hold all the state, now advanced by one step, the grid has done its job, and it is
wiped clean for the next round.

That round trip, P2G then grid update then G2P then discard, is one step of MPM. Run it in a loop and the
blob moves. Everything else in the method is detail hung on this skeleton.

## Why older transfers were not enough, and what APIC fixes

The transfers as just described, plain mass and plain velocity, are the original 1990s form of MPM, and
they have a real defect that is worth understanding because it explains a piece of machinery you will
otherwise wonder about.

The problem is that a single velocity per particle is a lossy summary of how the material is actually
moving near that particle. Material does not just translate, all moving the same direction at once. It
can be *rotating* locally, like a tiny whirlpool, or *shearing*, with one side sliding past the other.
A plain particle velocity captures only the average translation and throws away the local rotation and
shear. Every time you transfer to the grid and back, you re-summarize the local motion as a single
velocity and lose the rotational and shearing part of it again. Across many steps this lost information
shows up as the simulation quietly bleeding energy and going mushy, with spinning material slowing down
for no physical reason.

**APIC**, the affine particle-in-cell transfer, is the fix, and the idea is simple once the problem is
clear. Alongside its single velocity, each particle additionally stores a small matrix that records the
*local velocity gradient*, which is to say how the velocity varies across the little patch of material
the particle represents. That matrix captures exactly the local rotation and shear that a single velocity
throws away. On the way back from the grid, the particle gathers not just an average velocity but this
affine matrix too, and on the way out to the grid it deposits momentum that accounts for it. The result
is that the local rotation and shear survive the round trip instead of being discarded, the energy
bleeding stops, and the simulation stays crisp. The matrix is usually called $C$, and its precise
definition and the reason for the particular constant in front of it are in [[math-toolkit]] under the
APIC affine state. For now the takeaway is only this. A particle carries a velocity because it is a
moving lump of material, and APIC adds an affine matrix on top because a single velocity cannot describe
local rotation and shear, which matters once you transfer back and forth many times.

This is also the line from the original draft that triggered this rewrite, "older transfers store only a
particle velocity and lose the local rotation and shear of the velocity field." It should now read as a
plain statement of a problem you understand rather than a wall of unexplained jargon.

## The vocabulary, collected

With the ideas in place, here are the names in one spot for reference.

- **Particle.** A persistent lump of material carrying position, mass, velocity, and deformation. The
  source of truth.
- **Grid.** A fixed regular lattice used as a temporary scratchpad for the force calculation, wiped and
  rebuilt every step. Holds no state between steps.
- **Transfer.** Moving information between particles and grid. There are two, one each direction.
- **P2G, particle to grid.** Particles splash their mass and momentum onto nearby grid nodes through
  smooth weights.
- **Grid update.** On the grid, convert momentum to velocity, apply gravity, apply wall boundaries.
- **G2P, grid to particle.** Particles read updated velocities back from the grid, then move and update
  their deformation.
- **Mass.** How much material a lump represents. Fixed per particle here. The weight that makes
  conservation work.
- **Momentum.** Mass times velocity. The conserved currency of motion that the transfers preserve.
- **Deformation.** A lump's record of how far it is from its rest shape. The material's memory.
- **Volume ratio $J$.** The single-number version of deformation in the first model. $1$ is rest, below
  $1$ is compressed, above $1$ is expanded.
- **Stress.** The internal push-back force that deformation generates.
- **Constitutive model.** The rule converting deformation into stress. The material's personality.
- **APIC.** The transfer rule that also carries a per-particle affine matrix $C$, preserving local
  rotation and shear that a plain velocity loses.

## Why a *differentiable* MPM is the actual goal

Everything above describes the forward simulator, the thing that answers "what happens next." The reason
this project cares about MPM is the inverse question from [[why-differentiable-physics]], "what input
produces the outcome I want," and the answer is gradients flowing backward through the whole rollout.

The smoothness baked into the transfers is what makes this possible at all. Because each particle splashes
onto the grid through smooth weights rather than snapping to the nearest node, a small change in a
particle's position produces a small, well-defined change in everything downstream, which is the
definition of differentiable. Chain that step-to-step and the gradient of a final loss with respect to
the very first input can be computed in one backward pass, exactly as in [[math-toolkit]].

That inversion is the bridge to the broader goal. A world model that is both generative and *controllable*
needs dynamics you can author against, and gradients are the most direct handle for authoring. The places
where the smooth story breaks, the division by a barely-touched grid node and the hard wall branches, are
where gradients through physics stop being trustworthy, and learning exactly where that happens is the
real prize. That investigation is the spine of the [[core]] sections, and it begins in earnest in
[[failure-modes]].
