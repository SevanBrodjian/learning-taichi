# Math toolkit

> Prerequisite, skim-level. The recurring math objects behind the core sections, stated so you can reference them and move on. Derivations deepen here over time.

## Quadratic B-spline weights
Each particle splashes onto the grid through a smooth, compact weight. MLS-MPM uses the quadratic B-spline, which touches a fixed $3\times3$ block of nodes around each particle in 2D. With a particle at continuous grid coordinate $x_p$ and the nearest base node, the per-axis weights for the three nodes are

$$w_0=\tfrac12\left(\tfrac32-d\right)^2,\quad w_1=\tfrac34-(d-1)^2,\quad w_2=\tfrac12\left(d-\tfrac12\right)^2,$$

where $d$ is the particle offset from the base node in cell units. The 2D weight $w_{ip}$ is the outer product of the per-axis weights. Two properties matter. The weights sum to one, so mass is conserved in transfer, and they are smooth in $x_p$, so the transfer is differentiable in particle position even though the base node index is found with a floor.

## The APIC affine state
Older transfers store only a particle velocity and lose the local rotation and shear of the velocity field, which bleeds energy. APIC repairs this by storing a per-particle affine matrix $C_p$ that captures the local velocity gradient. On the way back from the grid it is gathered as

$$C_p=\frac{4}{\Delta x^2}\sum_i w_{ip}\,v_i\,(x_i-x_p)^\top,$$

and on the way out it contributes an affine term to the momentum a particle deposits. The factor $4/\Delta x^2$ is the inverse of the quadratic B-spline second moment, which is the constant that makes the reconstruction exact.

## Deformation and the volume ratio $J$
The material remembers how much it has been compressed or stretched. The first model tracks this with a single scalar $J_p$, the determinant of the deformation gradient, which is the local volume ratio. $J=1$ is rest volume, $J<1$ is compressed, $J>1$ is expanded. It evolves by

$$J_p \leftarrow (1+\Delta t\,\operatorname{tr} C_p)\,J_p,$$

since the trace of the velocity gradient is the local rate of volume change. A weakly compressible elastic stress is then a simple function of $J_p$, which is what pushes compressed material back apart.

## Reverse-mode autodiff over an unrolled program
A simulation rollout is just a long composition of differentiable functions, one per step. Reverse-mode autodiff (the same engine as backprop) computes the gradient of a scalar loss with respect to every input by walking that composition backward once, multiplying by each step's local Jacobian transpose. The cost is one backward pass of the same order as the forward pass, but it needs every intermediate state it will reuse, which is the memory tension the [[differentiating-the-rollout]] section is built around. The wall conditions and other non-smooth branches are where the clean chain-rule story develops cracks, which the [[failure-modes]] section takes apart.
