# Linear algebra for MPM

> Prerequisite, skim-friendly. The standing linear algebra the core sections lean on, built from
> intuition first and only then written down. If you have read [[mpm-in-context]] you already know the
> *roles* some of these objects play (the affine matrix, the deformation matrix). This page gives the
> algebra its precise form and, more importantly, the geometric reading that makes each operation
> memorable. The one heavier tool, the SVD, gets its own page in [[svd-polar]]. Notation is local. A
> matrix is a capital letter $A$, a vector a lowercase $v$, the identity $I$.

The whole method is a story about matrices acting on little vectors of material. A grid velocity is a
vector, the affine state $C_p$ is a matrix, the deformation $F_p$ is a matrix, and every constitutive
model is a formula that eats the deformation matrix and returns a stress matrix. So the recurring
question underneath everything is the same one linear algebra was invented to answer. What does it mean
for a matrix to *act* on space, and how do we read off the geometry it encodes. Almost every symbol in
the core sections is a specific answer to that question.

## A matrix is a linear map, read column by column

A vector in 2D is a pair of numbers, an arrow from the origin. A matrix is a rule that takes a vector in
and hands a vector out, and the only rule it is allowed to be is a **linear** one, meaning it respects
addition and scaling. Push two arrows through and you get the sum of what each would have given
($A(u+v) = Au + Av$), and stretch an arrow before pushing it through and the output stretches the same
way ($A(cv) = c\,Av$). Every such rule in 2D is exactly a $2 \times 2$ block of numbers.

The fastest way to read a matrix is by its **columns**, because the columns are just where the matrix
sends the two basis arrows. The vector $e_1 = (1,0)$ comes out as the first column of $A$, and
$e_2 = (0,1)$ comes out as the second. A general vector $v = (v_1, v_2)$ is $v_1 e_1 + v_2 e_2$, and by
linearity its image is

$$
Av = v_1\,(\text{column 1 of }A) + v_2\,(\text{column 2 of }A).
$$

So "apply a matrix" means "take the two columns as the new homes of the two axes, and rebuild the vector
from those." A matrix is a picture of a deformed grid. Its columns are where the unit square's two edges
end up. That single image, the unit square getting sheared and stretched into a parallelogram, is the
mental model to carry through every matrix in this project. The deformation gradient $F_p$ is *literally*
this picture applied to a chunk of material, which is why it is called a gradient of the deformation and
why $F = I$ (columns still the plain axes) means undeformed.

## Dot product and outer product, the two ways to multiply vectors

Two vectors combine in two opposite ways, and MPM uses both.

The **dot product** $u^{\top} v = u_1 v_1 + u_2 v_2$ takes two vectors and returns a single number,
collapsing them down. Its meaning is alignment, $u^{\top} v = \lVert u \rVert \lVert v \rVert \cos\theta$,
positive when the arrows point the same way, zero when they are perpendicular. The small superscript
$\top$ is the transpose, defined properly below, and here it is just the bookkeeping that turns the column
$u$ into a row so the multiplication lines up.

The **outer product** $u\,v^{\top}$ goes the other direction. It takes two vectors and returns a whole
**matrix**, the block whose $(i,j)$ entry is $u_i v_j$. Where the dot product asks "how aligned," the
outer product builds "the simplest matrix that sends $v$'s direction onto $u$'s direction." This is not a
curiosity. It is exactly how the affine state is assembled in [[math-toolkit]]. The APIC reconstruction

$$
C_p = \frac{4}{\Delta x^2}\sum_i w_{ip}\,v_i\,(x_i - x_p)^{\top}
$$

is a weighted sum of outer products, each one $v_i (x_i - x_p)^{\top}$, a matrix built from the node
velocity $v_i$ and the offset $x_i - x_p$ from particle to node. Reading it as outer products is what makes
it obvious that $C_p$ is a *matrix* estimating how velocity changes across the neighborhood, not just an
averaged vector. An outer product is the atom that velocity gradients are built from.

## Transpose, and why the inverse-transpose keeps appearing

The **transpose** $A^{\top}$ flips a matrix across its diagonal, turning rows into columns. For a vector it
turns a column into a row, which is the notation that makes $u^{\top} v$ a dot product and $u v^{\top}$ an
outer product. Geometrically the transpose is the map that slides across a dot product,
$(Au)^{\top}v = u^{\top}(A^{\top}v)$, moving $A$ from one factor to the other, but the property
that matters most here is more prosaic. The transpose of a product reverses the order,
$(AB)^{\top} = B^{\top} A^{\top}$, and a rotation's transpose is its inverse (below).

The reason to pin this down is that the corotated stress in [[constitutive-models]] carries a factor
$F^{-\top}$, shorthand for $(F^{-1})^{\top} = (F^{\top})^{-1}$ (the two are equal, so the order of inverse
and transpose does not matter). It looks intimidating but has a clean role. When a force or an area vector
is measured in the *rest* shape and needs to be re-expressed in the *deformed* shape, the object that
correctly carries it across is $F^{-\top}$, because areas and normals transform by the inverse-transpose of
the map that transforms lengths. Every time $F^{-\top}$ appears, it is doing this one job, translating a
rest-frame quantity into the deformed frame the simulation currently lives in.

## Trace, the invariant that reads off volume rate

The **trace** $\operatorname{tr} A = A_{11} + A_{22}$ is the sum of the diagonal entries. On its face that
looks like an arbitrary thing to add up, but the trace is special because it does not depend on the
coordinate frame. Rotate the axes and the individual entries scramble, yet their diagonal sum is
unchanged, so the trace measures something intrinsic to the map rather than to the way it was written
down. That intrinsic something is the sum of the eigenvalues (below), and for a velocity gradient it is
the **divergence**, the local rate at which the material is expanding.

This is the whole reason [[math-toolkit]] can update the volume ratio with a trace. The affine matrix
$C_p$ is an estimate of the velocity gradient, and $\operatorname{tr} C_p$ is therefore the rate of volume
change of the material around particle $p$, which is what drives

$$
J_p \leftarrow (1 + \Delta t\,\operatorname{tr} C_p)\,J_p.
$$

The trace turns a full matrix of local motion into the single number the weakly compressible model needs,
and it is the right single number precisely because it is frame-independent.

## Determinant, signed area and the meaning of $J$

The **determinant** of a $2 \times 2$ matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ is
$\det A = ad - bc$. The definition is worth reading geometrically, because $\det A$ is the **signed area**
of the parallelogram spanned by the two columns of $A$, which is to say the factor by which $A$ scales
area. A determinant of $2$ doubles every area, a determinant of $1$ preserves area, a determinant between
$0$ and $1$ shrinks it, and a negative determinant means the map also flipped orientation (turned the
plane over). A determinant of exactly $0$ means the columns are parallel, the unit square is crushed onto
a line, and the map is not invertible because area was destroyed.

This is the exact content of the volume ratio $J = \det F_p$ from [[mpm-in-context]]. The deformation
gradient $F_p$ maps a rest-shape chunk to its current shape, so $\det F_p$ is how much that chunk's volume
(area in 2D) has changed. $J = 1$ is volume-preserving, $J < 1$ is compressed, $J > 1$ is expanded. When
the constitutive model penalizes $\det F \ne 1$, it is penalizing exactly this area change, and when a
fluid tracks only $J$ it is keeping only the determinant of the full deformation and discarding the rest.

## Inverse, and when a matrix has one

The **inverse** $A^{-1}$ undoes $A$, so $A^{-1}A = A A^{-1} = I$. A matrix has an inverse exactly when its
determinant is nonzero, which is the algebraic echo of the geometric fact above. If $A$ crushes area to
zero it has thrown information away and cannot be undone, and if it preserves some area it can. The inverse
scales area by $1/\det A$, the reciprocal, which is why a near-singular matrix (tiny determinant) has a
huge inverse and is the numerical danger to watch. That danger is the same one the failure-modes story
keeps returning to, a barely-touched grid node with near-zero mass whose reciprocal blows up.

## Symmetric matrices and the eigen-decomposition

A **symmetric** matrix satisfies $A = A^{\top}$, its entries mirrored across the diagonal. Symmetric
matrices are the well-behaved ones, and they are common here because stress and stretch are symmetric by
their physics (the push-back a material develops has no preferred handedness). The reason symmetry is a
gift is the **spectral theorem**, which says every symmetric matrix can be written

$$
A = Q \Lambda Q^{\top},
$$

where $\Lambda$ is diagonal, holding the **eigenvalues** $\lambda_1, \lambda_2$, and $Q$ is a rotation
whose columns are the **eigenvectors**, the special directions the matrix merely stretches without turning.
Read right to left, $A$ first rotates space so the eigenvectors line up with the axes ($Q^{\top}$), then
stretches along each axis by its eigenvalue ($\Lambda$), then rotates back ($Q$). So a symmetric matrix is
"pure stretch along some perpendicular set of directions." The trace is $\lambda_1 + \lambda_2$ and the
determinant is $\lambda_1 \lambda_2$, which is why the trace reads total expansion rate and the determinant
reads area scaling. Both are just summaries of the stretches.

## Orthogonal matrices and rotations

An **orthogonal** matrix $Q$ satisfies $Q^{\top} Q = I$, meaning its transpose is its inverse. Geometrically
that is the exact condition for a **rigid** map, one that preserves every length and angle, so orthogonal
matrices are the rotations and reflections. Its columns are perpendicular unit vectors, a rotated copy of
the axes. A **rotation** is the orientation-preserving case, $\det Q = +1$ (a reflection has
$\det Q = -1$, it flips the plane). Rotations matter because a spinning blob should develop no stress from
the spin alone, only from genuine shape change, and separating the rotation out of a deformation is the
job of the two decompositions below. The fact that $Q^{-1} = Q^{\top}$ for a rotation is also why undoing a
rotation is free, no matrix inverse needed, just a transpose.

## Two readings to carry forward

Every matrix in this project is read through the same small vocabulary. Its **columns** say where the axes
go, its **determinant** says how it scales area (and defines $J$), its **trace** says the expansion rate
(and updates $J$), its **transpose** and **inverse-transpose** carry rest-frame quantities into the
deformed frame, and if it is **symmetric** it is pure stretch along perpendicular directions while if it is
**orthogonal** it is a rigid rotation. The one remaining question, how to pull an arbitrary deformation
apart into a rotation and a stretch so the constitutive model can act on the stretch alone, is exactly what
the singular value decomposition answers, and it earns its own page in [[svd-polar]].

The reason all of this is prerequisite rather than trivia is the gradient. Differentiable simulation lives
or dies on whether these operations, determinant, trace, inverse, decomposition, have well-defined
derivatives, because a constitutive model is a composition of them and the loss gradient has to travel back
through every one. Most are smooth everywhere. The decompositions are the subtle case, smooth almost
everywhere but not at special configurations, and that subtlety is a direct cause of the roughness
[[failure-modes]] catalogues, which is why the next page treats them with care.
