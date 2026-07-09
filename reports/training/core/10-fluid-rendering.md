# From particles to a believable liquid: rendering the MPM fluid

The [[mpm-in-context]] page makes a quiet but important claim: the particles are the only thing the
simulation truly stores, and their positions are "the thing you actually see when you render." That leaves
a gap. A weakly compressible MPM fluid is a few thousand dots drifting through a box, and a scatter plot of
those dots does not look like water. It looks like static. This page is about closing that gap on the
visual side: how to take the same particle cloud the [[material-showcase]] fluid produces and turn it into a
frame that reads as a real liquid seen side-on, like water in a glass tank.

This matters to the larger goal for a blunt reason. A controllable, differentiable world model is only
compelling if what it produces looks like the world. The physics can be perfect and the demo still falls
flat if the output is a point cloud. Rendering is not decoration here, it is the last stage that decides
whether a viewer believes the simulated world at all. The rendering below is deliberately
**non-differentiable** and offline, which is the honest trade: none of it needs to survive a gradient, so it
can reach for whatever makes the image convincing rather than whatever is smooth. That is a different regime
from the rest of this book, and the contrast is the lesson. When the goal is a believable picture rather
than a trainable one, the constraints change completely.

![The dam-break scene at the moment the returning wave curls over and traps a pocket of air, the classic
breaking-wave tube. The body carries a depth gradient from bright shallow cyan at the thin top edge to deep
blue where the water is thick, the curling sheet catches a bright specular edge, and aerated whitewater
marks the fast crest. This single frame uses every stage described below.](/api/data/learning-taichi/runs/realistic-rendering/non-differentiable-fluid-renderer/dambreak_hero.png)

## The pipeline in one line

A particle cloud becomes an image in four moves: reconstruct a **surface** from the particles, shade the
**body** of the liquid with depth color and a bent background, add **reflection and highlights** at that
surface, and mark **foam** where the water is torn up. Each stage is cheap and each one adds a specific,
nameable cue that the eye uses to recognize water. The breakdown figure walks the first and last stages on
one real frame.

![How one frame is built, left to right. First the raw MPM particles as a scatter, which reads as dots not
liquid. Second the density field they splat into, a smooth blob field where bright means many nearby
particles. Third the surface reconstructed from that field, shown as a clay model lit only by its
reconstructed normals so the recovered shape is visible with no color. Fourth the final shaded frame with
depth color, refraction, reflection, and foam composited
on.](/api/data/learning-taichi/runs/realistic-rendering/non-differentiable-fluid-renderer/breakdown.png)

## Surface reconstruction: metaballs from particles

The single biggest step from "dots" to "liquid" is giving the fluid a **surface**. The standard trick is
metaballs. Treat each particle as a soft blob of influence and add the blobs up into a smooth field, then
declare the surface to be a level set of that field.

Concretely, splat every particle position $x_p$ into a high-resolution grid with a Gaussian kernel and sum,

$$
D(u) = \sum_{p} \exp\!\left(-\frac{\lVert u - x_p \rVert^{2}}{2\sigma^{2}}\right),
$$

where $u$ is a pixel location on the render grid, $x_p$ is the position of particle $p$, and $\sigma$ is the
kernel width in pixels. $D(u)$ is a **density field**: it is large where many particles sit close together
and falls off smoothly into empty space. The width $\sigma$ is the one knob that sets how the fluid reads.
Turn $\sigma$ up and neighboring particles merge into one smooth mass but fine droplets smear away; turn it
down and the surface gets lumpy and starts to show the individual particles again. It trades smoothness
against detail directly.

A useful implementation note: summing a Gaussian centered on every particle is the same as taking the raw
count of particles per pixel and blurring it once with a Gaussian. So in practice $D$ is computed by
histogramming the particles into pixels and running a single Gaussian blur, which is far cheaper than
evaluating one kernel per particle per pixel and gives the identical field.

The **surface** is then a contour of constant density, $\{\,u : D(u) = c\,\}$, for a chosen isovalue $c$.
Inside the contour the density is above $c$ and there is liquid; outside it is below and there is air. To
avoid a jagged one-pixel-wide edge, the hard test is softened into a smooth ramp across a narrow band around
$c$, which gives an anti-aliased liquid mask with a soft transparent fringe instead of a staircase.

The surface also needs a **normal**, the direction the surface faces, because every lighting cue below
depends on it. The gradient $\nabla D$ points in the direction the density increases fastest, which is
straight into the liquid, so the outward surface normal in the image plane is the normalized negative
gradient,

$$
n_{\text{2D}} = -\frac{\nabla D}{\lVert \nabla D \rVert}.
$$

That gives a normal that lies in the plane of the image. For shading, this side view is treated as a
gently rounded surface bulging toward the viewer by lifting the normal into a third dimension,

$$
N = \operatorname{normalize}\big(-\partial_x D,\; -\partial_y D,\; k\big),
$$

where $\partial_x D$ and $\partial_y D$ are the horizontal and vertical slopes of the density field and $k$
is a positive constant. The role of $k$ is worth stating plainly. Where the density is flat, deep in the
body of the fluid, the slopes vanish and $N$ points straight at the viewer, so the interior faces forward.
Near the surface edge the density drops off steeply, the slope terms dominate, and $N$ tilts to graze along
the boundary. Small $k$ exaggerates that tilt and makes the whole surface look curved and glassy; large $k$
flattens it toward a pane of glass. This one field, the reconstructed normal, is what the reflection,
refraction, and specular stages all read from, and the third clay panel of the breakdown figure shows the
shape it recovers from nothing but dots.

## Depth color: Beer-Lambert absorption gives the body volume

A flat fill of one blue is the giveaway of a fake liquid. Real water gets deeper in color the more of it a
sightline passes through, because water absorbs light along the path. That is the Beer-Lambert law: the
fraction of background light that survives a passage through thickness $T$ of an absorbing medium is

$$
\mathcal{T} = e^{-\sigma_a T},
$$

where $T$ is how much liquid the view ray crosses and $\sigma_a$ is an absorption coefficient. In this side
view the reconstructed density $D$ doubles as a proxy for that thickness: a pixel where the fluid is thick
has a large $D$ and a thin sheet or a lone droplet has a small one. The shaded body color blends a bright
shallow tint against a dark saturated deep tint using $\mathcal{T}$ as the mix,

$$
c_{\text{body}} = \mathcal{T}\, c_{\text{shallow}} + (1 - \mathcal{T})\, c_{\text{deep}},
$$

so thin water stays pale and see-through while thick water turns a deep saturated blue. Raising $\sigma_a$
pulls the crossover toward thinner water and makes the fluid read as more strongly colored and less
transparent; lowering it makes a more watery, translucent look. This single term is what gives the body
volume instead of a cutout, and in the hero frame it is the whole reason the crest reads as shallow and the
base reads as deep in the same pool.

## Refraction and reflection: the cues that sell it

Two surface-optics cues do most of the remaining work, and of the two, refraction is the one that most
makes a viewer believe the liquid is really there.

**Refraction.** Light bends as it crosses the water surface, so the background seen through the liquid is
displaced. It is faked by sampling the background image not at the pixel itself but at an offset driven by
the surface normal and the thickness,

$$
c_{\text{refr}} = \text{background}\big(u + s\, T\, n_{\text{2D}}\big),
$$

where $s$ scales the bend. The effect is only visible if the background actually has structure to displace,
which is why the tank sits in front of soft light sources and a horizon rather than a flat wall. Where the
surface curves, the background visibly warps, and that warp is a cue no scatter plot can fake.

**Reflection and Fresnel.** A water surface is partly a mirror, and how mirror-like depends on the viewing
angle. Looking straight down into a pool it is nearly clear; looking across it at a grazing angle it turns
to a bright sheet. That angle dependence is the Fresnel effect, and the Schlick approximation captures it
cheaply,

$$
F = F_0 + (1 - F_0)\,(1 - \cos\theta)^{5},
$$

where $\cos\theta$ is the angle between the surface normal and the view direction and $F_0$ is the
reflectance when looking straight on. For a water-air interface $F_0 \approx 0.02$, meaning only about two
percent of light reflects at normal incidence, which is why water is clear from above. The $(1-\cos\theta)^5$
term climbs sharply toward one as the angle grazes, which is why the rim of every blob and the far edge of
every wave lights up. In the reconstructed normal field, $\cos\theta$ is exactly the third component of
$N$: it is near one in the flat interior, giving low reflectance and a clear view through the body, and near
zero at the grazing rim, giving a bright reflective edge. The final color mixes the refracted body against a
reflected environment by $F$, so the interior shows the bent background and the edges show the bright
surroundings, the same split a real water surface makes.

## Specular highlights and foam: the finishing cues

**Specular.** A sharp glint where the surface faces exactly between the light and the eye is the signature
of a wet, smooth surface. The Blinn-Phong model places it with a half vector $H = \operatorname{normalize}(L
+ V)$, the direction bisecting the light direction $L$ and the view direction $V$, and a highlight
brightness of $(N \cdot H)^{\alpha}$. The exponent $\alpha$ sets how tight the glint is: large $\alpha$ gives
a small hard sparkle, small $\alpha$ a broad soft sheen. On a flat interior the highlight is weak and even,
but on a crest or a ripple, where the normal swings to face $H$, it flares, which is why the moving edges of
a splash catch bright points of light.

**Foam.** Fast, torn-up water entrains air and turns white, and marking that whitewater is what makes a
splash read as violent rather than gelatinous. Foam is painted where either of two things is true: the
fluid is moving fast at the surface, found by splatting particle speed into a field and looking for high
values along the surface band, or the fluid is genuinely thin and broken, found where the reconstructed
thickness is well below the level of solid body. The first catches crests and splash crowns; the second
catches spray, sheet edges, and isolated droplets. Gating foam to the surface band matters, because without
it the discreteness of the particles speckles white flecks through the calm interior, which reads as noise
rather than aeration. A final light blur and a soft bloom on the bright pixels lets the foam and the glints
glow the way wet highlights do.

## What this does and does not model

The result reads convincingly as liquid, but honesty about the register is part of the point. This is a
**stylized 2D side view**, not a physical light-transport simulation. The thickness that drives both the
absorption and the refraction is a proxy read off a 2D density field, not a real optical path length through
a 3D volume. There is no true refraction with a measured index, no caustics cast onto the floor, no
inter-reflection between parts of the surface, and no measured absorption spectrum for water; the colors and
coefficients are chosen by eye. The surface normal is reconstructed by lifting a 2D gradient into a fake
third dimension, so the sense of roundness is an artistic choice rather than recovered geometry. What the
pipeline demonstrates is how far a stack of cheap, well-chosen cues, a smooth surface, a depth gradient, a
bent background, an angle-dependent reflection, a glint, and some foam, can carry a raw particle cloud
toward looking real. It gets most of the way to a double-take and stops short of photoreal, and the gap that
remains is exactly the 3D light transport it never attempts.

The connection back to the spine of this project is the trade it makes. Everywhere else in this book the
simulator is built to be differentiated, and that constraint shapes every choice, from the constitutive
stress in [[material-showcase]] to the smoothness of the loss. Here the constraint is lifted, and the moment
it is, the renderer reaches for hard thresholds, per-pixel branches, and image samples that no gradient
could flow through, because the only goal is the picture. A controllable world model eventually has to serve
both masters at once, a forward pass that looks like the world and a backward pass that can steer it, and
seeing the visual-quality regime on its own makes clear how much of what sells an image is exactly the
non-smooth, non-differentiable machinery the rest of the project has to work around.
