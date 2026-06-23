# The MLS-MPM forward step

This is the engine every later idea sits on. One step moves information particle to grid (P2G), updates the grid, then moves it back grid to particle (G2P). The background grid is rebuilt from scratch each step. Notation follows the [[math-toolkit]], with particle index $p$, grid node $i$, weight $w_{ip}$, timestep $\Delta t$, cell size $\Delta x$, and Young's modulus $E$.

## Particle to grid
Each particle deposits mass and momentum onto its surrounding nodes. The momentum it deposits carries both its plain velocity and an internal-stress plus affine term, folded into one matrix $A_p$.

$$
(m v)_i \mathrel{+}= \sum_p w_{ip}\big(m_p v_p + A_p (x_i - x_p)\big),\qquad
A_p = -\frac{4\,\Delta t\,E\,V_p (J_p-1)}{\Delta x^2}\,I + m_p C_p.
$$

The first part of $A_p$ is the elastic stress. It grows with how far the volume ratio $J_p$ has departed from rest, so compressed material pushes outward. The second part is the APIC affine state $C_p$ that preserves local rotation and shear.

## Grid update
Once every particle has deposited, each node holds total mass $m_i$ and total momentum $(m v)_i$. Convert to velocity, apply gravity, then apply wall conditions.

$$v_i \leftarrow (m v)_i / m_i,\qquad v_{i,y} \mathrel{-}= \Delta t\,g,\qquad v_i \leftarrow \text{wall}(v_i).$$

The division by $m_i$ is the first quiet hazard, since a node barely touched by any particle has a tiny mass, and its backward sensitivity scales like $1/m_i^2$. The wall step zeroes the inward normal velocity at the domain boundary, which is the non-smooth contact branch the [[failure-modes]] section returns to.

## Grid to particle
Each particle gathers a fresh velocity and a fresh affine matrix from its nodes, advects, and updates its volume.

$$
v_p = \sum_i w_{ip} v_i,\qquad
C_p = \frac{4}{\Delta x^2}\sum_i w_{ip} v_i (x_i - x_p)^\top,\qquad
x_p \mathrel{+}= \Delta t\,v_p,\qquad
J_p \mathrel{*}= (1 + \Delta t\,\operatorname{tr} C_p).
$$

After this the grid is discarded and the next step rebuilds it. The seed forward code is `sim/mpm88.py`, the canonical 88-line MLS-MPM, and `sim/diffmpm.py` is the time-indexed version built for gradients. The wall clamps live at `sim/mpm88.py` lines 49 to 59 and 77.

## Why this is the right altitude for the project
The whole step is a handful of smooth sums plus two rough spots, the mass division and the wall branch. That is a small enough surface that the gradient behavior is actually understandable rather than mysterious, which is the point of starting here. A controllable world model needs dynamics whose sensitivities you can reason about, and a clean explicit step like this one is where that reasoning gets its footing before scaling up.
