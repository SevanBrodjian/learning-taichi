# Four materials, one shader: why they all look the same, and what to do about it

**The key idea:** a particle renderer draws whatever the *reconstruction* says is there, and if every
material goes through the same reconstruction then every material has the same shape language. Colour is
the only degree of freedom left, so colour is the only difference you see. **Material identity lives in
how the surface is recovered from the points, not in the shading applied afterwards.**

This page is the general version of [[fluid-rendering]], which took one material — water — all the way
to a believable image. Read that first for the water pipeline in detail; this page is about what changes
when four different substances have to share one screen and be told apart.

## The failure, stated precisely

The standard way to draw particles is a **splat**: give each particle a small disc, accumulate a smooth
weight $w$ and a weighted colour into an image, divide out the weight, and shade the result as an
iso-surface of $w$. Concretely, with a compact kernel $w(r) = (1-r^2)^2$ over a disc of radius $R$:

$$
A(u) = \sum_p w\!\left(\frac{\lVert u - x_p\rVert}{R}\right), \qquad
c(u) = \frac{1}{A(u)}\sum_p c_p\, w\!\left(\frac{\lVert u - x_p\rVert}{R}\right),
$$

then keep the pixels where $A > \tau$ and light them with a normal built from $\nabla A$.

Every material rendered this way inherits the same three visual properties, because they are properties
of $A$ and not of the material:

1. **A bumpy surface at the particle scale.** $A$ is a sum of finitely many bumps, so its level set is
   lumpy with a wavelength set by the particle spacing, not by anything physical.
2. **A soft, ambiguous edge.** The iso-contour of a smooth kernel sum fades out over roughly $R$.
3. **A uniform interior.** After dividing by $A$, the inside is one flat colour with a little shading
   noise on top.

That combination *is* the look of snow — a soft, diffuse, lumpy pile — which is why snow is the one
material this treatment gets right. Water should be smooth and see-through, rubber should be one
coherent object with a hard edge, and sand should be visibly made of grains. None of those is reachable
by changing $c_p$.

> **The diagnostic:** if you cannot tell two materials apart with the colour channel removed, you have
> not drawn two materials. Render everything with the same albedo and convert to luminance. Whatever
> survives is real; whatever disappears was hue.

## Four reconstructions, four identities

There are only a handful of ways to get from a point cloud to a picture, and each one has a shape
language that suits some materials and not others.

### 1. Iso-surface of a splat — soft, lumpy, uniform

The baseline above. Two passes, essentially free. Its identity is *soft mush*, which is correct for
powder and wrong for everything else. When a material genuinely is a loose aggregate seen from far
enough away that grains are sub-pixel, this is the right answer and the cheapest one.

What it does have going for it, and it matters: **it never loses an isolated particle**. A single
droplet of spray still deposits a visible blob. Every reconstruction that thresholds a density will
throw that droplet away.

### 2. Screen-space filled surface — smooth, deep, see-through

This is the fluid pipeline. Splat into a density, blur it, then **stop using one field for two jobs**:
a low threshold plus a morphological repair gives a *filled mask* that decides opacity, while the
density gradient — kept unsmoothed enough to retain slope detail — decides the *normal*. Optical
thickness comes from a distance transform of the mask, so it is smooth by construction rather than a
noisy count. Beer–Lambert absorption on that thickness gives the one cue nothing else in the set has:
**the inside of the body has a value gradient**, dark where it is deep and pale where it is thin.

The mechanics of the mask, the closing, the distance transform and the separable blur are all in
[[filters-and-samples]]; the shading stack — Fresnel, refraction, foam — is in [[fluid-rendering]].

Its identity is *smooth, continuous, and transparent*. Suits liquids and glass. Its cost is
pixel-bound, not particle-bound, and its characteristic failure is the one noted above: thin spray
falls under the threshold and vanishes.

### 3. Silhouette and material coordinates — hard-edged, coherent, deforming

For a solid, the thing to communicate is that the particles are **one object**. Two moves do almost all
of it.

**A constant-width border.** Take the filled mask, low-pass the *silhouette* (blur the 0/1 mask and
re-threshold at $0.5$) so the boundary is one smooth closed curve, then compute the distance $D$ to the
outside and darken the band $D < w$. Because $D$ is a distance and not a density, the border is $w$
pixels wide everywhere — it does not thin where the body thins. A closed line of constant width is read
by the eye as *the outline of a thing*, which is precisely the percept a cloud of dots lacks.

**A texture in material coordinates.** Each particle carries its rest position $X_p$ as well as its
current position $x_p$. Splat $X_p$ with the same kernel and divide by the weight, and you have a smooth
field $X(u)$ — the inverse of the deformation map, sampled on screen. Evaluate any pattern in $X$ rather
than in $u$ and the pattern is *painted on the material*: it translates, rotates, stretches and shears
with the body, because it is a function of the material point that is currently at that pixel. A coarse
grid drawn this way turns an opaque blob into an object whose deformation you can read directly, and it
is a rendering trick with no physical content at all — $X_p$ is already sitting in memory.

Its identity is *flat interior inside a hard closed outline*, optionally with a pattern that deforms.
Suits rubber, jelly, any elastic solid.

### 4. Per-grain sprites — granular, matte, rough-edged

For sand the point is the opposite of a surface: you want the eye to resolve **individual grains**.
Draw several small sprites per particle, each with its own hashed offset, radius, ellipse aspect and
rotation, so the pack has a size distribution instead of one grain size repeated. Skew the radius
distribution small (many fines, few coarse) because that is what a real grading looks like.

The implementation wrinkle is depth: with no depth buffer, overlapping grains have to be resolved. An
atomic *max* on a per-grain random priority picks a winner per pixel — and if the grain's shade is
defined as a deterministic function of that same priority, the winner's shade is recoverable in the
resolve pass without storing a second buffer. One atomic scalar does the whole job.

Its identity is *high-frequency speckle and a ragged boundary*. Suits any granular pack. Its cost is
**geometric**, not pixel-bound: $K$ sprites per particle is $K$ times the instance count, which is the
one treatment here whose price grows with how much material is on screen.

## What this costs, and which number to trust

Measured on one RTX 4090 at the demo's own canvas size and particle count (16,384 particles,
$456\times456$), as summed device time over every rendering kernel:

| treatment | GPU ms / frame | × the splat baseline |
| --- | --- | --- |
| splat iso-surface (the baseline) | 0.309 | 1.0 |
| the same, with matte shading (snow) | 0.307 | 1.0 |
| filled surface, no background sampling | 0.542 | 1.8 |
| filled surface with refraction | 0.585 | 1.9 |
| silhouette + material-coordinate texture | 0.753 | 2.4 |
| six sprites per particle | 1.118 | 3.6 |

Three things in that table are worth carrying away.

**Changing only the shading arithmetic is free.** Snow's treatment removes a specular term, softens the
edge and adds a grain — and lands within measurement noise of the baseline, because the pass structure
did not change. A surprising fraction of "make it look different" is exactly this, and costs nothing.

**Reconstruction costs pixels; sprites cost particles.** The screen-space treatments are dominated by
full-image passes and barely move when the particle count changes. The sprite treatment scales with
$K\times N$. Which of the two is expensive therefore depends entirely on the resolution-to-particle
ratio of your scene, and the ranking flips: at $1080^2$ the *baseline* splat costs more than the water
treatment, because its splat radius is a fixed fraction of the canvas and its pixel footprint grows
quadratically.

**The distance transform is the biggest single item.** In the water pipeline the jump-flood passes plus
their seeding are about 31% of device time, and their only product is a smooth optical thickness. That
is the first thing to approximate if a budget bites.

### The number *not* to trust

The same treatments, timed with a host wall clock instead of the device profiler, come out at 1.0–5.2 ms
— and give **the same 3.3 ms at $360^2$ as at $1080^2$**, a nine-fold change in pixels. A cost that does
not move with the thing it is supposed to be proportional to is not measuring that thing. What it is
measuring is 25–30 Python-side kernel launches per frame.

This is the same effect [[real-time-cost]] and [[fixed-point-atomics]] hit on the solver, and it has a
measured constant: a Taichi kernel launched from Python has a dispatch floor of about **55.6 µs**, while
a WGSL dispatch recorded inside a WebGPU command buffer costs about **1.11 µs** on the same device — a
50-fold difference that has nothing to do with the work being done. So a multi-pass renderer is cheap in
a browser **if** the passes are recorded once into a command buffer, and ruinous if each is submitted
separately. When you are deciding whether a pipeline fits a frame budget, count the passes and multiply
by the dispatch floor of the API you are actually going to ship on, then add the device time — and never
quote a Python-driven wall clock as the cost of an algorithm.

## Scope

This is a 2-D side view, and several of the identities above lean on that. There is no depth buffer, so
"which grain is in front" is decided by a hash rather than by geometry; there is no volumetric light
transport, so thickness is a screen-space proxy; and the material-coordinate texture is a 2-D map that
would need real UVs in 3-D. The measurements are one GPU, one API, one canvas size and one particle
count. What generalises is the structural claim — that the *reconstruction*, not the shading, is where
material identity is decided — and the greyscale test that checks it.

---

**Code:** `sim/material_render.py` (treatments and the baseline port), `sim/material_render_cost.py`
(the device-time measurement).
**Related:** [[fluid-rendering]] for the water pipeline in full, [[filters-and-samples]] for the
filtering and sampling facts underneath all of it, [[real-time-cost]] and [[fixed-point-atomics]] for
the dispatch-overhead result the cost section leans on, [[material-showcase]] for what the four
canonical materials actually do.
