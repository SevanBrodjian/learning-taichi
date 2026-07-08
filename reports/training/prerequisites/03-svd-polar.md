# SVD and polar decomposition

> Prerequisite, skim-friendly, but worth reading in full once because it is the one piece of linear
> algebra the elastic and snow constitutive models cannot do without. Assumes [[linear-algebra]] (columns
> as maps, symmetric matrices as pure stretch, orthogonal matrices as rotations, determinant as area
> scaling). This page builds the singular value decomposition and its cousin the polar decomposition, then
> connects them to the corotated stress and the snow clamp in [[constitutive-models]] and to the gradient
> pathologies in [[failure-modes]].

## Every matrix is rotate, stretch, rotate

The eigen-decomposition in [[linear-algebra]] only works for symmetric matrices. A deformation gradient
$F$ is not symmetric in general, because real deformation mixes stretching with rotation. The **singular
value decomposition** is the tool that handles *any* matrix, and its statement is remarkably clean. Every
$2 \times 2$ matrix factors as

$$
F = U \Sigma V^{\top},
$$

where $U$ and $V$ are rotations (orthogonal matrices) and $\Sigma$ is diagonal with non-negative entries
$\sigma_1 \ge \sigma_2 \ge 0$ called the **singular values**. Read right to left, the geometry is a
three-step recipe that any linear map obeys. First $V^{\top}$ rotates space to line up a special pair of
perpendicular input directions with the axes. Then $\Sigma$ stretches along those axes, by $\sigma_1$ along
one and $\sigma_2$ along the other. Then $U$ rotates the stretched result into its final orientation. So the
seemingly complicated action of an arbitrary matrix is always just **rotate, stretch along perpendicular
axes, rotate again**. There are no other possibilities. That is the whole content of the SVD, and it is why
it shows up wherever a deformation needs to be understood rather than merely applied.

The singular values are the **principal stretches**, the factors by which the material is stretched along
its two principal directions. $\sigma_k = 1$ is no stretch along direction $k$, $\sigma_k < 1$ is
compression, $\sigma_k > 1$ is extension. Their product is the area change, $\sigma_1 \sigma_2 = \det \Sigma
= \det F = J$ (the rotations contribute a determinant of $1$ each and drop out), which reconnects the SVD to
the volume ratio $J$ from [[linear-algebra]]. So the singular values refine $J$. Where $J$ says only "total
area changed by this much," the pair $(\sigma_1, \sigma_2)$ says "and here is how much along each principal
direction," which is exactly the extra information a solid needs and a fluid throws away.

## Polar decomposition, the rotation-then-stretch reading

Group the SVD factors differently and a second decomposition falls out for free, the one the corotated model
actually names. Insert $V^{\top} V = I$ between $U$ and $\Sigma$,

$$
F = U V^{\top} \, V \Sigma V^{\top} = R S, \qquad R = U V^{\top}, \quad S = V \Sigma V^{\top}.
$$

This is the **polar decomposition** $F = R S$. Here $R = U V^{\top}$ is a rotation (a product of two
rotations is a rotation), and $S = V \Sigma V^{\top}$ is symmetric and positive (it is a symmetric matrix in
the spectral form from [[linear-algebra]], with the singular values as its eigenvalues). The reading is
physical and exactly the one a solid wants. **Any deformation is a pure stretch $S$ followed by a rigid
rotation $R$.** The stretch $S$ carries all the genuine shape change, the part that stores elastic energy,
and the rotation $R$ carries the part that stores none, since spinning a body without deforming it costs no
energy.

That split is why $R = U V^{\top}$ appears in the corotated stress $2\mu (F - R)$ in [[constitutive-models]].
Subtracting the rotation $R$ from $F$ leaves only the non-rotational part, the actual stretching and
shearing, so a particle that merely spins ($F = R$, $S = I$) develops zero stress. Computing $R$ straight
from the polar decomposition is possible but ill-conditioned near $\det F = 0$. Going through the SVD and
forming $U V^{\top}$ is the numerically stable route, which is why the code (and the model) reaches for
`ti.svd`.

## The snow clamp is surgery on the singular values

The singular values are also the natural place to impose plasticity, which is what the snow model in
[[constitutive-models]] does. Snow stays elastic only while its principal stretches stay within a permitted
band. Push a stretch past the limit and the excess becomes permanent. Expressed on the singular values,

$$
\hat\sigma_k = \operatorname{clamp}\big(\sigma_k,\; 1-\theta_c,\; 1+\theta_s\big), \qquad k = 1, 2,
$$

with $\theta_c$ the compression limit and $\theta_s$ the stretch limit. The clamped stretches rebuild a
smaller *recoverable* deformation $F \leftarrow U \hat\Sigma V^{\top}$, and whatever was clamped off is
moved into an accumulated plastic record. Working on the singular values is what makes this well defined,
because they are the coordinate-free measure of stretch. Clamping them means "allow this much elastic
stretch and no more, in every direction equally," independent of how the material happens to be oriented.

## Where the gradient breaks, and why it matters here

The SVD is the reason gradients can flow through elastic and plastic stress at all, and also the reason they
sometimes flow badly. Taichi's `ti.svd` is **differentiable**, so the whole chain from a control parameter,
through the deformation, through $U \Sigma V^{\top}$, into the stress and out to the loss, has a derivative
the autodiff engine can walk backward. Without that, the corotated and snow stresses would be black boxes
and none of the control tasks in the core would be possible.

The catch is that the SVD is smooth *almost* everywhere, not everywhere, and the two failure points are
both physical. First, when the two singular values **coincide** ($\sigma_1 = \sigma_2$), the principal
directions are no longer unique (a uniformly scaled disk has no special axes), and the derivatives of $U$
and $V$ blow up even though $F$ itself is perfectly nice. Second, the snow **clamp** is only $C^0$, not
$C^1$. A singular value inside the band passes through untouched (derivative $1$), one outside is pinned
(derivative $0$), and right at the limit the map has a **kink** where the derivative jumps. This is the same
one-sided-clamp pathology [[failure-modes]] dissects for the hard wall, now applied per particle, per step,
to the singular values, and stacking many such kinks across a long rollout is what roughens the snow loss
landscape.

This is the exact spot where the abstract linear algebra meets the concrete objective of the project. A
controllable, differentiable world model needs its dynamics to expose smooth gradients, and the SVD is a
perfect microcosm of the difficulty. It is the operation that unlocks rich material behavior (rotation-aware
elasticity, plastic yielding) and simultaneously the operation whose non-smoothness, at coincident singular
values and at plastic kinks, is where gradient-based control gets hardest. Understanding the SVD is
therefore not just prerequisite bookkeeping. It is understanding, in miniature, why the interesting
materials are the hard ones to steer.
