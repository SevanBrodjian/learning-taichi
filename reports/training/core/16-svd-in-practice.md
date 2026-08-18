# Computing the 2×2 SVD, and proving it before trusting it

> Assumes [[linear-algebra]] (orthogonal matrices, determinant as signed area, the eigen-decomposition
> of a symmetric matrix) and [[svd-polar]] (what the SVD *is*, the closed-form 2D polar rotation, the
> snow clamp and the log strain as surgery on $\Sigma$). This page is about the other half: how the
> factorisation is actually **computed**, which of its outputs are physics and which are merely
> convention, and why a numerical kernel whose wrong answers still look like plausible motion has to be
> proved in isolation rather than judged by watching the simulation.

## The key idea

A plastic material is a **rule about singular values**. Snow clamps them into a box, sand projects their
logarithm onto a cone ([[constitutive-models]]). Neither rule can be evaluated without the singular
values, so a solver that carries a plastic material needs a real SVD, per particle, per substep. An
elastic solver does not: the corotated stress asks only for the polar rotation $R$, which in two
dimensions has the closed form derived in [[svd-polar]] and costs two additions and a reciprocal square
root. This is the single largest structural difference between an elastic-only solver and one that
carries the full material set, and it is why "add snow and sand" is engineering rather than a parameter
change.

What makes the SVD unusually dangerous to implement is the **shape of its failure**. A wrong matrix
inverse produces $\infty$. A wrong interpolation weight produces mass that visibly fails to conserve. A
wrong SVD produces **motion that looks right**: snow still crumbles, sand still slumps, a pile still
stands at a plausible angle, and nothing on the screen reports that the principal stretches feeding the
yield criterion are the wrong numbers. Anything downstream of it — a measured angle of repose, a learned
material fitted against it, a claim that a port reproduces a reference — inherits the error silently.
The discipline that follows from this is simple and non-negotiable: **the factorisation is verified as a
standalone numerical routine, against an independent implementation, on inputs chosen to break it,
before it is allowed anywhere near a simulation.**

## What actually consumes what

Being precise about which factor each material touches is what makes the verification tractable, because
it says exactly which outputs have to be right and which are free.

| material | needs | from the SVD |
| --- | --- | --- |
| fluid | the volume ratio $J$ alone | nothing |
| elastic | the polar rotation $R$ | nothing — closed form |
| snow | $\sigma_1, \sigma_2$ to clamp, then a reconstruction | $\Sigma$, and $U, V$ to rebuild $F$ |
| sand | $\ln \sigma_k$ to project onto the cone, and $U$ for the stress | $\Sigma$ and $U$; $V$ only to rebuild $F$ |

Two things follow. First, the **singular values are the only outputs both plastic laws read**, so they
carry almost the whole correctness burden. Second, sand's Hencky stress is $U \operatorname{diag}(\tau_k)
U^{\top}$ and snow's hardening depends only on $\sigma_1 \sigma_2$, so **$V$ never enters a stress**. It
appears only when a modified $\Sigma$ is assembled back into a deformation gradient. That asymmetry turns
out to matter, and the next section says why.

## Computing it: polar first, then one rotation

The 2×2 algorithm is two steps and no iteration. The derivation is worth following because each step is
a thing already understood, assembled.

**Step 1 — peel off the rotation.** Polar-decompose $F = R S$ with $R$ orthogonal and $S$ symmetric
positive semi-definite. [[svd-polar]] derives the $\det F > 0$ case: with $x = F_{11} + F_{22}$ and
$y = F_{21} - F_{12}$,

$$
R = \frac{1}{\sqrt{x^2 + y^2}} \begin{bmatrix} x & -y \\ y & x \end{bmatrix}.
$$

A robust implementation needs the other case too. When $\det F < 0$ the material has **inverted** — it has
been turned inside out, which an explicit solver can genuinely produce under violent compression — and
the closest *orthogonal* matrix is then a reflection, not a rotation. Repeating the same
maximise-the-trace argument over orthogonal matrices of determinant $-1$ gives the companion formula,
built from $x' = F_{11} - F_{22}$ and $y' = F_{21} + F_{12}$:

$$
R = \frac{1}{\sqrt{x'^2 + y'^2}} \begin{bmatrix} x' & y' \\ y' & -x' \end{bmatrix}, \qquad \det R = -1 .
$$

An implementation that omits this branch does not fail loudly. It returns a rotation where a reflection
was required, which flips the sign of one singular value's contribution and quietly reports an inverted
particle as a merely rotated one. The symmetric factor is then $S = R^{\top} F$, which can be written
without forming $R^{\top} F$ explicitly as $S = (F^{\top} F + |\det F|\, I) / \sqrt{|\det B|}$ where $B$
is the numerator matrix above — the same expression for both branches, which is a small but real saving
inside a kernel that runs once per particle per substep.

**Step 2 — diagonalise the symmetric leftover.** $S$ is symmetric, so by the spectral theorem in
[[linear-algebra]] it is $S = V \Sigma V^{\top}$ for a rotation $V$ and a diagonal $\Sigma$. In two
dimensions that rotation can be written down in closed form as well. Parameterise $V$ by $(c, s)$ with
$c^2 + s^2 = 1$ and require the off-diagonal of $V^{\top} S V$ to vanish. Writing $t = s/c$ for the
tangent of the rotation angle, that condition is a quadratic,

$$
t^2 + 2 \tau t - 1 = 0, \qquad \tau = \frac{S_{11} - S_{22}}{2 S_{12}},
$$

whose two roots are the two ways to assign the axes. The numerically sound choice is the root of smaller
magnitude, obtained without cancellation as

$$
t = \frac{S_{12}}{\tau' \pm w}, \qquad \tau' = \tfrac{1}{2}(S_{11} - S_{22}), \quad
w = \sqrt{\tau'^2 + S_{12}^2},
$$

taking $+$ when $\tau' > 0$ and $-$ otherwise, then $c = 1/\sqrt{1+t^2}$ and $s = -tc$. This is one step
of the **Jacobi eigenvalue method**, which in general iterates over off-diagonal entries; in $2 \times 2$
there is only one off-diagonal entry, so a single step is exact. Assembling,

$$
F = R S = R\, V \Sigma V^{\top} = (RV)\, \Sigma\, V^{\top} = U \Sigma V^{\top}, \qquad U = R V .
$$

Finally the two singular values are sorted so that $\sigma_1 \ge \sigma_2$, swapping the columns of $V$
when they are out of order. **The ordering is not cosmetic.** Both plastic laws index the singular values
positionally, and any code that assumes the first one is the larger — a heuristic on the compressed
direction, a plot of the principal stretch, a comparison against a reference — silently pairs up the
wrong quantities if the ordering convention differs.

That is the whole routine: about thirty lines, no loops, no iteration count to tune. It is worth
appreciating how much of the difficulty of the plastic models this small function absorbs.

## The part that is convention, not physics

Here is the subtlety that decides how the routine can be tested. Reconstruction after a plastic
projection is written $F \leftarrow U \Sigma' V^{\top}$. Suppose an implementation instead assembles
$U \Sigma' V$, or reorders factors, or returns $V^{\top}$ where another returns $V$ — a family of
mistakes that are easy to make and, in most numerical code, immediately fatal.

For an **isotropic** constitutive model they are not fatal. They are not even observable.

The reason is a symmetry worth stating in general, because it recurs across continuum mechanics. Let $Q$
be any rotation and consider replacing $F$ by $FQ$. Its SVD is $FQ = U \Sigma V^{\top} Q = U \Sigma
(Q^{\top} V)^{\top}$, so $U$ and $\Sigma$ are **unchanged** and only $V$ absorbs the rotation. Now check
what the stresses do. The corotated stress contracted with $F^{\top}$ is $2\mu (F - R) F^{\top} +
\lambda(J-1) J I$; under $F \mapsto FQ$ the polar rotation becomes $R Q$, and

$$
(FQ - RQ)(FQ)^{\top} = (F - R) Q Q^{\top} F^{\top} = (F - R) F^{\top},
$$

while $\det(FQ) = \det F$. The elastic stress is **exactly invariant**. The Hencky stress is
$U \operatorname{diag}(2\mu \varepsilon_k + \lambda \operatorname{tr}\varepsilon) U^{\top}$ with
$\varepsilon = \ln \Sigma$, and neither $U$ nor $\Sigma$ moved, so it is invariant too. The snow clamp
and the Drucker-Prager return map read only $\Sigma$, so they are invariant as well.

The physical content is that $Q$ is a **rotation of the reference configuration** — a relabelling of the
undeformed material, like deciding to measure a rubber block from a different corner. An isotropic
material has no preferred material directions, so it cannot possibly depend on that choice. Only an
anisotropic model, one with a fibre direction or a grain, would notice.

Three practical consequences follow, and they are the reason this section exists.

1. **Two implementations can carry visibly different deformation gradients and be doing identical
   physics.** Comparing $F$ between a reference and a port is therefore the *wrong* test. It reports
   differences that no measurement can detect.
2. **The right comparison targets are the singular values and the observables** — positions,
   velocities, stresses, settled shapes. Those are gauge-invariant, and they are what a claim of
   equivalence should actually be about.
3. **A port should nevertheless copy the reference's reconstruction verbatim.** Not because the physics
   requires it, but because two runs that agree state-for-state can be compared bit for bit, whereas two
   runs that agree only observable-for-observable diverge in floating point at a slightly different rate
   and force every comparison to argue about noise bands. Matching an arbitrary convention is cheap;
   proving that a difference does not matter, over and over, is not.

## Proving it: four checks that fail differently

A single check cannot establish that a factorisation routine is right, because the useful properties are
independent. Four are needed, and it is worth being explicit about what each one alone would miss.

1. **Reconstruction.** $\lVert U \operatorname{diag}(\sigma) V^{\top} - A\rVert / \lVert A \rVert$ small.
   Catches swapped factors, dropped transposes and sign errors. Passes happily on a $U$ that is not
   orthogonal, as long as the product happens to come out right.
2. **Orthogonality.** $\lVert U^{\top}U - I\rVert$ and $\lVert V^{\top}V - I\rVert$ small. Catches a
   normalisation that was skipped or a degenerate branch that returned garbage. Says nothing about
   whether the factorisation is of the input matrix at all.
3. **Ordering.** $\sigma_1 \ge \sigma_2$, always. Catches a missing swap, which reconstruction and
   orthogonality both wave through — the factorisation is still perfectly valid, just not the one the
   plastic laws are indexed against.
4. **Agreement with an independent implementation, on the singular values.** Catches the case where all
   three structural checks pass and the routine is nonetheless computing *a different valid
   factorisation* than the reference. By the gauge argument above, $U$ and $V$ may legitimately differ;
   $\Sigma$ may not, because $\Sigma$ is unique. Checking $\Sigma$ against a reference is what turns "this
   is *an* SVD" into "this is *the* SVD the downstream physics expects".

The **input set** matters as much as the checks. Random well-conditioned matrices establish a floor and
prove nothing, because every branch of the routine that is dangerous is a branch random matrices almost
never take. The families worth constructing deliberately:

- **near-rotations**, where $S \approx I$ and the off-diagonal falls below whatever threshold the
  implementation uses to skip the Jacobi step — the branch that exists precisely so a division by a
  vanishing $S_{12}$ never happens;
- **near-singular**, one $\sigma \to 0$, where the reciprocal square root in the polar step approaches
  its own degeneracy;
- **negative determinant**, exercising the reflection branch that is invisible on any well-behaved input;
- **strongly anisotropic**, condition numbers to $10^4$, where cancellation in $\tau'$ is worst;
- **exact zeros, exact identities, and exact clamp boundaries**, which land on branch conditions rather
  than near them;
- and, most valuable of all, **deformation gradients sampled from a real rollout of each plastic
  material**, so the test set contains the distribution the routine will actually be fed rather than a
  guess about it.

A run of a few thousand such matrices costs milliseconds and settles the question permanently. The
alternative — inferring correctness from whether the simulation looks right — cannot settle it at all,
because looking right is exactly what a broken SVD does.

## Failure modes

**Division by a determinant that rounds to zero.** The polar step scales by $1/\sqrt{|\det B|}$, and the
standard argument that $\det B \ne 0$ for any non-zero $F$ is an *exact-arithmetic* argument. In float32
a sufficiently degenerate $F$ rounds it to zero and the particle receives $\infty$, which then spreads
through the transfer into every node in its stencil. Flooring the denominator at a tiny positive constant
costs nothing and converts a whole-simulation detonation into one particle with a meaningless but finite
rotation.

**Coincident singular values.** When $\sigma_1 = \sigma_2$ the principal directions are genuinely not
unique — a uniformly scaled disc has no special axes — so $U$ and $V$ are arbitrary within a rotation
even though $F$ is perfectly well behaved. The *values* remain unique and well conditioned, which is why
every downstream consumer that reads only $\Sigma$ is safe, and why the gauge argument above is not
merely a curiosity: near-degeneracy is a place where two correct implementations *will* return different
$U$ and $V$. It is also, as [[svd-polar]] notes, where the derivatives of $U$ and $V$ blow up, which is
the differentiability version of the same statement.

**A branch threshold that is not scale-free.** Skipping the Jacobi step when $|S_{12}| < \epsilon$ uses an
absolute comparison against a matrix whose scale depends on how $S$ was normalised. It is a decision
about "close enough to diagonal already", and it silently assumes the inputs are $\mathcal{O}(1)$ —
which, for a deformation gradient hovering near the identity, they are. Feeding the same routine matrices
scaled by $10^{-6}$ would take the shortcut branch every time and return a diagonal that is not one. Any
port that reuses such a routine outside the deformation-gradient regime it was written for inherits the
assumption without being told about it.

**Trusting float32 reconstruction error as a bound on physical error.** The reconstruction residual for a
$2\times2$ product in single precision sits around $10^{-5}$ relative. That is not an error in the
singular values; it is accumulated rounding in the *product*. The singular values themselves typically
agree with a double-precision reference to a few $10^{-7}$. Reporting the reconstruction residual as
"the accuracy of the SVD" overstates the error on the only quantity the physics reads by two orders of
magnitude.

## What's open

The verification described here establishes that a routine computes the same factorisation as a
reference on a chosen input distribution. It does not establish **stability under composition**: a
plastic solver applies this map once per particle per substep, tens of thousands of times per simulated
second, and a systematic bias far below any single-call tolerance could still accumulate. Whether the
per-call agreement measured on a static test set predicts the drift of a long rollout is not settled by
the test, and separating that drift from the genuine chaos of the underlying dynamics is the same hard
problem [[real-time-cost]] runs into when comparing a port against a non-deterministic reference.

A second open question is whether the branch structure is worth its cost on a wide GPU. Every branch
above — the determinant sign, the Jacobi shortcut, the ordering swap — is a divergence point, and within
a group of threads executing in lockstep the cost of a branch is the cost of *every* path any thread in
the group takes. A branchless formulation using masked selects would be uniformly slower per particle
and uniformly predictable; which wins depends on how correlated neighbouring particles' deformation
states are, and that correlation is a property of the scene rather than of the algorithm.
