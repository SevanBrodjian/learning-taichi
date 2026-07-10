# Vector calculus on a grid: gradient, divergence, curvature

> Prerequisite, skim-friendly. The differential operators that turn a scalar field on the MPM grid into a
> surface geometry: the gradient of a scalar, the divergence of a vector field, and the curvature of a
> level set written as the divergence of a unit normal. [[linear-algebra]] already gives the trace and the
> Jacobian these rest on, and [[math-toolkit]] uses the trace of the velocity gradient as a divergence.
> This page collects the operators themselves and their finite-difference form on a regular grid, so
> [[surface-tension]] can lean on them. Notation is local. A scalar field is $\phi(x)$, a vector field is
> $F(x)$, the grid spacing is $\Delta x$.

The setting throughout is a field defined on the simulation grid: a number (or a vector) stored at every
node $(i,j)$. Three questions get asked of such a field, and each is one differential operator. Which way
is the field increasing, and how fast? That is the **gradient**. Is the field's flow spreading out or
piling up at a point? That is the **divergence**. And, applied to the direction a surface faces, how
sharply is the surface bending? That is the **curvature**, and the clean surprise is that it is just a
divergence in disguise.

## The gradient: steepest ascent of a scalar

Let $\phi(x)$ be a scalar field, one number per point (in the surface-tension use, a smoothed indicator
that is near $1$ inside the fluid and $0$ outside). The **gradient** $\nabla\phi$ collects its partial
derivatives into a vector,

$$
\nabla\phi = \left(\frac{\partial\phi}{\partial x},\ \frac{\partial\phi}{\partial y}\right).
$$

Each component is the rate at which $\phi$ changes as one moves along that axis. The vector they form has
a clean geometric meaning that is worth carrying: $\nabla\phi$ **points in the direction of steepest
increase** of $\phi$, and its length $\lVert\nabla\phi\rVert$ is the slope in that direction. Two
consequences matter later. First, because it points toward larger $\phi$, at the edge of a fluid blob
(where $\phi$ climbs from $0$ outside to $1$ inside) the gradient points **inward**, from the empty side
into the material. Second, $\nabla\phi$ is perpendicular to the level sets of $\phi$, the curves along
which $\phi$ is constant. The boundary of the blob is exactly such a level set, so $\nabla\phi$ is
**normal to the surface**. Dividing it by its own length gives the **unit normal**,

$$
n = \frac{\nabla\phi}{\lVert\nabla\phi\rVert},
$$

a vector of length one pointing straight across the interface. This is how a mere density field gives a
surface a well-defined orientation with no explicit boundary tracking.

## The divergence: net outflow of a vector field

Now let $F(x) = (F_x, F_y)$ be a vector field, an arrow at every point. The **divergence** is the scalar

$$
\nabla\cdot F = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y},
$$

the sum of each component's derivative along its own axis. Its meaning is **net outflow per unit area**.
Picture a tiny box around a point. If the arrows leaving the box outweigh those entering, the field is
spreading out there and the divergence is positive, a source. If arrows converge inward, it is negative, a
sink. If as much flows in as out, the divergence is zero. This is the same object [[math-toolkit]] uses
when the trace of the velocity gradient $\operatorname{tr} C_p = \nabla\cdot v$ measures the local rate of
volume change: a positive velocity divergence is material expanding, a negative one is material
compressing. Divergence is the trace of a Jacobian, and reading it as "net outflow" is the geometric face
of that algebra from [[linear-algebra]].

## Curvature as the divergence of the normal

Here is the operator surface tension actually needs, and it is built from the two above. A curve's
**curvature** $\kappa$ measures how fast its direction turns per unit length travelled along it. A
straight line never turns, so $\kappa = 0$. A circle of radius $R$ turns at a constant rate and has
$\kappa = 1/R$, so a tight little circle (small $R$) is sharply curved and a huge one is nearly flat. The
sharp corner of a square is the extreme case: the direction swings through a right angle over almost no
distance, so the curvature there is very large.

The key identity is that the curvature of a level set equals the **negative divergence of its unit normal
field**,

$$
\kappa = -\nabla\cdot n, \qquad n = \frac{\nabla\phi}{\lVert\nabla\phi\rVert}.
$$

The reason this works is worth seeing rather than memorising. Extend the unit normal $n$ off the surface
so that it points outward from the shape everywhere nearby. For a filled disk of radius $R$, that outward
normal at a point is the radial unit vector $\hat r$ (pointing away from the centre). A standard fact in
2D is that $\nabla\cdot\hat r = 1/R$ at radius $R$: the radial field fans out, so it has positive
divergence, and the fanning is gentler the farther out one goes. If instead $n$ is taken to point
**inward** (as $\nabla\phi$ does for a blob, since $\phi$ grows inward), then $n = -\hat r$ and
$\nabla\cdot n = -1/R$, so $-\nabla\cdot n = +1/R = \kappa$. The minus sign in the identity is just the
bookkeeping that makes a convex droplet come out with **positive** curvature regardless of which way the
normal was defined. The magnitude is what carries the geometry: $\kappa$ is large where the surface is
tightly curved and small where it is flat, so on a square it spikes at the four corners and nearly
vanishes along the flat sides. That spatial pattern is exactly what a surface-tension force will exploit
to pull the corners in.

## Doing it on a grid: central differences

All three operators are continuous derivatives, and the fields here live at discrete grid nodes, so each
derivative is replaced by a **finite difference** between neighbouring nodes. The standard, second-order
choice is the **central difference**: to estimate a derivative at node $i$, subtract the value one node to
the left from the value one node to the right and divide by the distance between them,

$$
\frac{\partial\phi}{\partial x}\Big|_{i,j} \approx \frac{\phi_{i+1,j} - \phi_{i-1,j}}{2\,\Delta x}.
$$

The numerator is the change in $\phi$ across two cells and the denominator $2\,\Delta x$ is the distance
spanned, so the ratio is a slope. Centring it (using $i+1$ and $i-1$ symmetrically rather than $i$ and
$i+1$) cancels the leading error term and makes the estimate second-order accurate, which is why it is
preferred over a one-sided difference. The grid gradient is the pair of these central differences in $x$
and $y$; the grid divergence of a vector field applies the same formula to $F_x$ in $x$ and $F_y$ in $y$
and adds them; and the grid curvature computes $n$ at every node from the gradient, then takes the central-
difference divergence of that normal field, $\kappa = -\nabla\cdot n$. Nothing beyond subtracting
neighbours and dividing by a spacing is involved, which is what makes these operators cheap to evaluate
inside a GPU kernel.

Two practical cautions, both of which bite in practice. The unit normal $n = \nabla\phi/\lVert\nabla\phi
\rVert$ divides by the gradient's length, which is near zero away from any interface, so the division is
guarded with a small floor to avoid amplifying noise where there is no real surface. And a curvature
estimated from a raw one-cell-thick jump in $\phi$ is dominated by grid staircase noise, so the indicator
field is **smoothed** first (a few local averaging passes) to spread the interface over a band a few cells
wide, giving the finite differences something continuous to bite on. That smoothing is why the surface
force in [[surface-tension]] is a *continuum* surface force rather than a sharp-interface one.
