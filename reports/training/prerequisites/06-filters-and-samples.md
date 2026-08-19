# Filters and point samples: convolution, separability, and the noise you get for free

**The key idea:** almost everything done to an image in this project is one of three operations —
*blur it*, *ask how far away something is*, or *threshold it* — and the reason they keep appearing is
that a simulation gives you a **random scatter of points**, while a picture needs a **field**. Turning
one into the other is a filtering problem, and filtering has a noise floor you do not get to opt out of.

Nothing here is specific to MPM. It is the small pile of signal-processing facts that
[[fluid-rendering]] and [[material-appearance]] both stand on.

## Convolution: a weighted average that slides

A **filter** replaces every sample by a weighted average of its neighbours. In one dimension, with
weights $k$,

$$(k * f)(i) \;=\; \sum_{m} k(m)\, f(i-m).$$

Read it as: *to get the output at $i$, lay the weight stencil down centred on $i$, multiply, and add.*
The weights are the whole design. Equal weights over a window give a **box blur** (cheap, and it rings).
Weights falling off as a Gaussian,

$$k(m) \;\propto\; \exp\!\left(-\frac{m^{2}}{2\sigma^{2}}\right),$$

give a **Gaussian blur**, which is the default because it is the only kernel that is smooth, positive,
and free of the ringing a hard-edged window causes. The single parameter $\sigma$ is a *length in
pixels*: it says how far information is allowed to spread. The kernel is truncated at about $4\sigma$,
beyond which the weights are negligible, so the stencil has radius $r \approx 4\sigma$.

Two properties matter and are worth memorising:

**Convolution is linear.** $k * (f + g) = k*f + k*g$. So you can blur a sum of things by blurring the
things — which is exactly why splatting particles into a histogram and blurring **once** gives the same
field as summing a Gaussian centred on every particle, at a small fraction of the cost.

**A Gaussian is separable.** The 2-D Gaussian factorises:

$$\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right) = \exp\!\left(-\frac{x^2}{2\sigma^2}\right)\exp\!\left(-\frac{y^2}{2\sigma^2}\right).$$

So a 2-D blur is a 1-D blur along rows followed by a 1-D blur along columns. The cost per pixel drops
from $O(r^2)$ taps to $O(r)$. At $\sigma = 7$, $r = 28$: a $57\times57$ window is about 3,200
multiply-adds, and two 57-tap passes are 114 — a **28-fold** reduction before any parallelism. Whenever
a 2-D filter is separable, doing it in two passes is not an optimisation, it is the implementation.

## The noise floor of a random point sample

Here is the fact that explains a whole family of visual defects.

Scatter $N$ points uniformly at random into a region, then count how many land in one small patch of
area $a$. That count $C$ is a random variable. For small $a$ it is Poisson-like, with

$$
\mathbb{E}[C] = \lambda, \qquad \operatorname{Var}(C) = \lambda, \qquad
\frac{\operatorname{sd}(C)}{\mathbb{E}[C]} = \frac{1}{\sqrt{\lambda}},
$$

where $\lambda$ is the expected count in that patch. The mean is right; the *relative* fluctuation is
$1/\sqrt{\lambda}$ and it is large exactly where you are looking most closely. A patch expecting
$\lambda = 4$ points fluctuates by $\pm 50\%$. **The material is uniform and the measured density is
not.**

Blurring helps, and you can say by how much. Blurring with $\sigma$ averages over an effective area of
roughly $4\pi\sigma^2$, raising $\lambda$ by that factor, so the relative noise falls like $1/\sigma$.
Halving the speckle costs twice the blur radius — and twice the blur radius also smears away any real
detail finer than that. **Smoothness against detail is not a tuning preference; it is the same knob.**

The consequence for rendering: if a threshold is applied to a blurred count field, the threshold will
be crossed *at random* wherever the field sits near it. Deep inside a body that is supposed to be solid,
the density ripples, dips below the isovalue, and that spot is declared empty. That is where the
interior holes in [[fluid-rendering]] come from, and it is a property of *sampling*, not of the
simulation.

## Thresholds, and repairing what a threshold does

Thresholding turns a field into a set: $B = \{u : d(u) > \tau\}$. It is the cheapest possible decision
and it throws away everything except membership. Two standard repairs:

**Morphological closing.** *Dilation* replaces each pixel by the maximum over a disk of radius $R$
around it; *erosion* replaces it by the minimum. Dilation then erosion is a **closing**: it fills gaps
narrower than about $2R$ and leaves everything wider untouched, because a gap wide enough to survive the
dilation is restored by the erosion. Closing repairs by *width*, which is the right notion for
sub-spacing pinholes and the wrong notion for a genuine cavity that happens to be thin.

**Blur-and-rethreshold.** Blurring the 0/1 mask and thresholding the result at $0.5$ is a low-pass
filter on the *boundary*: curvature finer than $\sigma$ is removed and the silhouette becomes one smooth
closed curve. This is how a lumpy cluster of particles becomes an object with an outline.

## Distance transforms: how far to the nearest of a set

Given a set $S$ of pixels, the **Euclidean distance transform** gives every pixel

$$D(u) \;=\; \min_{s \in S} \lVert u - s \rVert .$$

Take $S$ to be the *outside* of a body and $D$ becomes "how deep inside am I" — zero on the boundary,
growing smoothly toward the core, and **speckle-free by construction**, because it depends only on the
geometry of the mask and not on any count. That is why it is the right source for an optical thickness
and the wrong thing to compute from a noisy density.

It is also the source of a *constant-width* border: the set $\{u : D(u) < w\}$ is exactly the band of
width $w$ just inside the boundary, whatever shape the boundary has. A border drawn from a density
gradient thins wherever the body thins; a border drawn from $D$ does not.

The exact transform is naturally sequential, which makes it awkward on a GPU. **Jump flooding** computes
it in $O(\log n)$ parallel sweeps instead: seed every pixel of $S$ with its own coordinate, then for
step sizes $2^k, \dots, 2, 1$ have each pixel look at the nine neighbours that far away and adopt
whichever carries a closer seed. Eleven full-image passes at $1080^2$, each embarrassingly parallel.
The general move it illustrates is worth keeping: *when a favourite CPU routine is sequential, look for
the algorithm that computes the same answer in a logarithmic number of parallel sweeps.*

## The gradient of a filtered field

Shading needs a surface normal, and a normal is a gradient. Central differences on a grid,

$$\partial_x d(i,j) \approx \tfrac{1}{2}\big(d(i{+}1,j) - d(i{-}1,j)\big),$$

are covered in [[vector-calculus]], including why $\nabla d$ points into the dense region and why it is
perpendicular to the level sets. The filtering point to add here is that **differentiation amplifies
noise**: a difference of two neighbouring samples has the noise of both and the signal of neither, so
the relative error of a gradient is worse than that of the field it came from. Always take a gradient of
something you have already smoothed, and be aware that the smoothing that makes the gradient usable is
the same smoothing that removes the detail you wanted the gradient to see.

---

**Used by:** [[fluid-rendering]] (metaball density, interior mask, thickness),
[[material-appearance]] (silhouette low-pass, constant-width borders, per-grain sprites).
**Depends on:** [[vector-calculus]] for gradients on a grid.
