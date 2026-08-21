# Scattering a material property to a shared grid

**The key idea:** a grid node in MLS-MPM does not belong to a material. It holds whatever happened to be
nearby, which at an interface is a bit of two substances at once. So any per-material number the grid
update needs, like a friction coefficient, cannot be *looked up*. It has to be **scattered from the
particles and averaged at the node**, in exactly the same way velocity already is. The pattern is one
line of arithmetic and it is the difference between an interface that behaves and an interface that
invents forces out of nowhere.

This page assumes the transfer skeleton from [[mls-mpm-forward]] and the B-spline weights it uses, the
material dials from [[material-stiffness]] (which is where the density $\rho$ and the Poisson ratio $\nu$
come from), and, for the GPU half, the fixed-point scatter from [[fixed-point-atomics]].

## Why a node cannot own a material

A particle scatters to the $3 \times 3$ nodes around it with quadratic B-spline weights $w_{ip}$ that sum
to one over the stencil. Two particles of different materials sitting within a cell of each other both
write into the same node. After the scatter that node holds

$$
m_i = \sum_p w_{ip}\, m_p , \qquad
(m v)_i = \sum_p w_{ip}\left( m_p v_p + \mathbf{A}_p (x_i - x_p) \right),
$$

and the update recovers a velocity as $v_i = (mv)_i / m_i$. Nothing in that pair of sums records *which*
material contributed. That is not an oversight, it is the whole point of a background grid, and it is why
one grid can carry four materials at once without a contact model.

The consequence is uncomfortable the first time it is noticed. Suppose water is frictionless against the
floor and sand is not, and a node on the floor holds some of each. There is no material id to branch on.
Picking one material's coefficient is arbitrary, and picking a global constant is what produced water that
dragged along a smooth floor as if it were sandpaper.

## The pattern: scatter the numerator, scatter the denominator, divide at the node

Velocity already solves this problem. The node does not store a velocity, it stores a **momentum** and a
**mass**, and the division at the end produces a mass-weighted average of the particle velocities. The
same construction works for any per-particle scalar $q_p$. Scatter $w_{ip} m_p q_p$ into a second
accumulator alongside the mass, then divide.

$$
Q_i \;=\; \sum_p w_{ip}\, m_p\, q_p , \qquad
\bar q_i \;=\; \frac{Q_i}{m_i} \;=\; \frac{\sum_p w_{ip}\, m_p\, q_p}{\sum_p w_{ip}\, m_p}.
$$

$\bar q_i$ is a genuine convex combination of the $q_p$ that reached the node, weighted by how much mass
each one delivered. It lies between the smallest and largest contributing value, it reduces to $q$ exactly
when only one material is present, and it moves smoothly as material flows across the boundary. For
friction that is the right physical statement. A floor node under a heap of sand with a film of water on
it grips like sand, because sand is what is pressing on it.

Nothing about the construction is specific to friction. Any material property the *grid* has to apply,
rather than the particle, wants this treatment. The two properties that do **not** want it are worth
naming, because the distinction is the useful part:

- **Constitutive parameters stay on the particle.** Stiffness, Poisson ratio and the plastic parameters
  enter through the stress $\mathbf{P}\mathbf{F}^\top$, which is computed per particle *before* the
  scatter. Averaging those at a node would be averaging two different constitutive models, which is
  meaningless. They never touch the grid.
- **Density is already handled, invisibly.** Mass $m_p = V_p \rho_p$ is the weight in every sum above, so
  a heavy material automatically counts for more in every node average. Density does not need a channel
  of its own because it *is* the channel.

That last point is the mechanism behind buoyancy in a shared grid, derived in [[material-stiffness]].
Gravity is applied to the node velocity, so it accelerates every node equally regardless of mass, while
the surrounding fluid's pressure arrives as an impulse that gets divided by $m_i$. A light node is moved
more by the same impulse than a heavy one. Sinking and floating fall out of the mass ratio with no
buoyancy term written anywhere, which is a small, pleasing example of a general principle worth carrying
into learned world models. **Emergent behaviour is cheaper and more robust than behaviour that is
special-cased, and it is also the only kind that transfers to configurations nobody enumerated.** A
hand-written buoyancy force would need to know what counts as "the fluid" and what counts as "the body".
The mass ratio needs to know nothing.

## The GPU version, where the trick has to survive a budget

On the CPU, adding a second accumulator is adding a field. On a GPU it runs into two constraints that
change the design.

**Storage buffers are scarce.** WebGPU guarantees only **8 storage buffers per shader stage**, and a
four-material MLS-MPM step already needs seven for positions, velocity, the affine matrix, the deformation
gradient, node mass, node momentum and node velocity. Exceeding the limit does not raise an error at
dispatch time. It invalidates the bind group, every dispatch is silently dropped, and the simulation
"runs" over trajectories of pure zeros while producing a beautiful flat timing curve. The correct response
is to **widen an accumulator rather than add one**. Node mass is a single `u32` per cell; making it two
`u32` per cell, mass and mass-times-friction, costs one array index and no bind-group slot at all. The
general form of the rule is that a per-node quantity indexed by cell is a *layout* choice, not a
*resource* choice, and layout is free.

**Fixed-point scatter needs a fixed unit.** WGSL has no atomic float add, so the scatter accumulates into
integers, with a scale chosen so that an integer of $2^{k}$ means one particle mass (see
[[fixed-point-atomics]] for why the exponent is set by measurement and not by guessing). The moment
density becomes per-material, "one particle mass" stops being a single number, and two materials writing
into the same accumulator on two different scales produce a node mass that is not any mass at all. The fix
is to define the scale against a fixed **reference** mass, $m_{\text{ref}} = V_p \rho_{\text{ref}}$ with
$\rho_{\text{ref}} = 1$, and let every material scale against it.

That has a cost, and it is worth stating precisely because it is easy to miss. An unsigned 32-bit
accumulator at $2^{24}$ quanta per reference mass saturates at $2^{8} = 256$ reference masses on one node,
**and wraps silently**. With every material at $\rho = 1$ the ceiling is 256 particles' worth. With the
heaviest material at $\rho = 1.6$ it is 160 of *its* particles. Introducing per-material density therefore
eats into the headroom of a mechanism that fails without any error at all, so the headroom has to be
re-measured under deliberate piling rather than inherited.

## Failure modes

**A global coefficient at an interface.** The symptom is a material behaving like its neighbour at a
boundary. Water that cannot slide down a wall it was thrown against reads to a viewer as *sticky*, and the
instinct is to reach for a surface-tension or adhesion term. The actual cause is a boundary condition
that zeroes both velocity components instead of separating in the normal direction and applying Coulomb
friction on the tangent, using a coefficient that never came from the material.

**Averaging the wrong thing.** Scattering $w_{ip} q_p$ without the mass factor and dividing by
$\sum_p w_{ip}$ gives a *volume*-weighted average instead of a mass-weighted one. It looks almost right
and is wrong exactly where it matters, at an interface between materials of different density, because it
gives a sparse light material the same say as a dense one. Weight by the same quantity the momentum is
weighted by, or the average is inconsistent with the velocity it is meant to modify.

**Assuming a port inherited the change.** Constants can be regenerated from a frozen physics module into a
generated header with no hand-typed numbers, and the port can still be running the old behaviour, because
a constant that no kernel reads changes nothing. Emitting $\rho$, $\nu$ and the friction coefficient is
necessary and not sufficient. The check that catches it is behavioural rather than textual. If snow does
not rise and sand does not sink, the density is not reaching the transfer, whatever the generated file
says.

## What is open

The node average is a *mixture* rule, not a *contact* model. It says a node holding half water and half
sand behaves like the mass-weighted blend, which is a reasonable default and is not a calibrated model of
what happens where two substances actually touch. Real contact has a discontinuity in the velocity field
at the interface, separate normal and tangential responses, and the possibility of the two phases sliding
past each other. A single shared velocity per node cannot represent any of that, which is why multi-phase
MPM schemes carry per-material grid velocities and reconcile them explicitly. That is a genuine modelling
limit, not a tuning problem, and it is the thing to reach for if an interface ever needs to be right
rather than plausible.

---

**Code:** `sim/physics/core.py` (`p2g_multi` scatters the mass-weighted friction, `grid_op` divides by the
node mass).
**Related:** [[mls-mpm-forward]] for the transfer and its weights, [[material-stiffness]] for where $\rho$,
$\nu$ and the buoyancy result come from, [[fixed-point-atomics]] for the integer scatter and its silent
wrap, [[material-showcase]] for what the four canonical materials do, [[real-time-cost]] for the dispatch
accounting the GPU section assumes.
