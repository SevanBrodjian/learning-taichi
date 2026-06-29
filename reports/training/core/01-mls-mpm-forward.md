# The MLS-MPM forward step

This is the engine every later idea sits on. One step moves information particle to grid (P2G), updates
the grid, then moves it back grid to particle (G2P), and the background grid is rebuilt from scratch each
step. The concepts behind every term, what a transfer is, why a particle carries a velocity and an affine
matrix, what the grid is for, and the meaning of mass, momentum, the volume ratio $J$, and stress, are
built up from zero in [[mpm-in-context]], with the formulas in [[math-toolkit]]. This page assembles
those pieces into the actual step and shows where the gradient hazards hide. Notation follows the
[[math-toolkit]]: particle index $p$, grid node $i$, interpolation weight $w_{ip}$, timestep $\Delta t$,
cell size $\Delta x$, and Young's modulus $E$, the stiffness constant that scales how hard stress pushes
back.

Two facts about the symbols carry the whole step. The grid is a fixed background lattice that exists only
to let particles talk to their neighbors; it holds no permanent state. Each particle, by contrast, is the
material, and it carries four things between steps: position $x_p$, velocity $v_p$, an affine velocity
matrix $C_p$ (introduced below), and a volume ratio $J_p$ that records how much it has been compressed or
expanded relative to its rest state.

## Particle to grid

P2G scatters each particle's mass and momentum onto the grid nodes near it. Mass and momentum are
accumulated in the **same** scatter, with the same weights, and both sums must be written out because the
grid update needs both.

$$
m_i \mathrel{+}= \sum_p w_{ip}\, m_p, \qquad
(m v)_i \mathrel{+}= \sum_p w_{ip}\big(m_p v_p + A_p\,(x_i - x_p)\big).
$$

The node mass $m_i$ is just the weighted pile-up of nearby particle masses, and it is exactly the quantity
the grid update divides by later. It is easy to read the momentum line and forget that mass is gathered in
lockstep beside it, so it is written explicitly here: every node that receives momentum also receives the
mass that goes with it.

The weight $w_{ip}$ and the offset $(x_i - x_p)$ play **two different roles**, and conflating them is the
most common point of confusion. The weight $w_{ip}$ is a scalar from the quadratic B-spline kernel; it
sets *how much* of particle $p$ reaches node $i$, and the weights of a particle sum to one across its
$3\times 3$ stencil (a partition of unity). The offset $(x_i - x_p)$ is a vector; it encodes *where* node
$i$ sits relative to the particle center, and it is the lever arm that lets the matrix $A_p$ deposit a
momentum that **varies linearly across the particle's neighborhood** instead of stamping the same value
on every node. Weighting alone would copy the particle's velocity to its neighbors; multiplying by the
offset lets the deposit carry a gradient, which is what stress and local rotation require.

### The affine-momentum matrix $A_p$

The matrix $A_p$ folds two physical effects into one object so that a single matrix-vector product
$A_p (x_i - x_p)$ delivers both to the grid:

$$
A_p = \underbrace{-\frac{4\,\Delta t\,E\,V_p\,(J_p - 1)}{\Delta x^2}\,I}_{\text{internal stress}}
      \;+\; \underbrace{m_p\,C_p}_{\text{affine velocity}}.
$$

$V_p$ is the **particle volume**: the small chunk of material volume that particle $p$ represents, set at
initialization (the domain volume split evenly among particles). It appears because stress is a force per
area and what the grid needs is a force, so the stress must be multiplied by the volume the particle
occupies to become an actual contribution to momentum. A larger particle carries more material and so
pushes harder for the same stress.

The first term is the **internal elastic stress**, and its sign and structure are the intuition worth
keeping. The factor $(J_p - 1)$ measures how far the particle's volume has departed from rest: $J_p > 1$
is expansion, $J_p < 1$ is compression. The leading minus sign means compression ($J_p < 1$, so
$J_p - 1 < 0$) produces a positive outward push, which is exactly how a squeezed elastic material behaves.
Young's modulus $E$ scales the stiffness, $\Delta t$ converts the force into an impulse over one step, and
the $1/\Delta x^2$ is the bookkeeping that comes out of the MLS-MPM derivation relating the kernel to a
force. The identity $I$ makes this stress isotropic in this minimal model: it pushes equally in all
directions, which is why a blob of this material behaves like a simple compressible solid rather than
something with shear-dependent response.

The second term, $m_p C_p$, is the **APIC affine state**, and it is why the offset $(x_i - x_p)$ matters.
A plain particle transfer assumes the velocity is constant over the particle's little region, which throws
away local rotation and shear and makes the simulation lose angular momentum and smear out. $C_p$ is a
$2\times 2$ matrix that stores the **local velocity gradient** around the particle, the first-order
description of how the velocity field twists and stretches near $x_p$. Multiplying it by the offset
reconstructs that linear velocity field at each neighboring node, so node $i$ receives not just "the
particle's velocity" but "the particle's velocity as it would be *at node $i$* given the local rotation
and shear". This is what keeps spinning and shearing material crisp instead of diffusing, and it is the
reason the affine term and the position offset always travel together.

## Grid update

Once every particle has scattered, each node holds a total mass $m_i$ and a total momentum $(m v)_i$. The
update converts momentum to velocity, applies gravity, then applies the wall condition.

$$
v_i \leftarrow (m v)_i / m_i, \qquad
v_{i,y} \mathrel{-}= \Delta t\,g, \qquad
v_i \leftarrow \operatorname{wall}(v_i).
$$

The division by $m_i$ is the answer to "where does $m_i$ come from": it is the mass accumulated in the P2G
scatter above, and dividing the accumulated momentum by it recovers a velocity. It is also the **first
quiet hazard**. A node barely grazed by a single particle has a tiny $m_i$, and while the forward division
is harmless there, its backward sensitivity scales like $1/m_i^2$ and can amplify a gradient by an
enormous factor. That near-zero-mass amplification is the central story of [[failure-modes]]. The wall
step zeroes the inward normal velocity at the domain boundary, a hard non-smooth branch that the same page
returns to as the contact-differentiability hazard.

## Grid to particle

G2P gathers a fresh velocity and a fresh affine matrix back onto each particle, advects it, and updates
its volume.

$$
v_p = \sum_i w_{ip}\, v_i, \qquad
C_p = \frac{4}{\Delta x^2}\sum_i w_{ip}\, v_i\,(x_i - x_p)^\top, \qquad
x_p \mathrel{+}= \Delta t\,v_p, \qquad
J_p \mathrel{*}= \big(1 + \Delta t\,\operatorname{tr} C_p\big).
$$

The new velocity is the weighted average of nearby node velocities. The new affine matrix $C_p$ is the
mirror image of the deposit: where P2G used $A_p(x_i - x_p)$ to *spread* a linear field onto the grid, G2P
uses $v_i (x_i - x_p)^\top$ to *measure* the linear field back off the grid, recovering the local velocity
gradient that the next step will redeposit. The volume update uses $\operatorname{tr} C_p$, the divergence
of the local velocity field: a positive trace means the material is locally expanding, so $J_p$ grows, and
a negative trace means it is compressing, so $J_p$ shrinks. That closes the loop, because $J_p$ is exactly
what the stress term in $A_p$ reads on the next step. After this the grid is discarded and the next step
rebuilds it from scratch.

The seed forward code is `sim/mpm88.py`, the canonical 88-line MLS-MPM, and `sim/diffmpm.py` is the
time-indexed version built for gradients. The wall clamps live at `sim/mpm88.py` lines 49 to 59 and 77.

## Why this is the right altitude for the project

The whole step is a handful of smooth sums plus two rough spots, the mass division and the wall branch.
That is a small enough surface that the gradient behavior is understandable rather than mysterious, which
is the point of starting here. A controllable world model needs dynamics whose sensitivities can be
reasoned about, and a clean explicit step like this one is where that reasoning gets its footing before
scaling up.
