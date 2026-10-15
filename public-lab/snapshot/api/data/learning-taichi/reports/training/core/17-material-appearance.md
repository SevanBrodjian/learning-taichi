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

## The same treatments, re-measured in the API that ships them

The table below is Taichi device time, and it ranks the treatments correctly *for Taichi*. Shipping the
same four treatments as WGSL fragment shaders inside a browser's render passes changes the numbers by an
order of magnitude, and changes which part of the pipeline is worth worrying about.

| | splat + one resolve | four treatments + grains |
| --- | --- | --- |
| GPU ms at $512^2$ | 0.034 | 0.087 |
| GPU ms at $1024^2$ | 0.097 | 0.147 |

Two readings matter. First, the whole drawing stage is now **under 1% of a 60 Hz frame**, against a solver
that costs about 7 ms of the same frame at 16,384 particles. Rendering stopped being the thing to budget
for the moment the passes were recorded into a command buffer instead of launched from Python, which is
the dispatch-floor result [[real-time-cost]] measures. Second, the *increase* is almost entirely a fixed
+0.05 ms that does not grow with resolution, because it is the per-grain sprite pass, whose cost is
$K \times N$ instances and not pixels. The resolution-dependent part of the resolve barely moved (0.063 ms
before, 0.060 ms after) even though it now carries four different shading models, because the extra work
is arithmetic on values that were already loaded.

That is worth stating as a rule, because it generalises past this renderer. **Branching per material
inside one full-screen pass is close to free; adding a pass, or adding geometry, is not.** Shading is
arithmetic on data already in registers. Structure is memory traffic and launches.

### Four materials in one resolve, without a fourth pass

The pass-count rule creates an obvious problem. Four treatments implies either four resolve passes or a
way to tell, per pixel, which material is there. Storing a material id per pixel is a second render
target; running four passes multiplies the fill.

The trick that avoids both is to change what the accumulation buffer *means*. The standard splat
accumulates premultiplied colour and weight, $(c_p w, w)$, in four channels. Accumulating the
**per-material weight** instead,

$$
A(u) = \left(A_{\mathrm{fluid}},\; A_{\mathrm{elastic}},\; A_{\mathrm{snow}},\; A_{\mathrm{sand}}\right),
$$

uses the same four channels, the same format and the same fill, and carries strictly more information.
The total weight is $\sum_m A_m$, which is what the iso-surface and its gradient need. The colour is
recoverable as $\left(\sum_m A_m c_m\right) / \sum_m A_m$, which is exactly what the old buffer stored.
And the material identity of a pixel is $\operatorname*{argmax}_m A_m$, which the old buffer had thrown
away. One resolve pass then branches to the right treatment with no extra target, no extra pass and no
extra bandwidth.

The general form is that **a premultiplied colour is a lossy projection of a composition**. Any time a
renderer accumulates $c \cdot w$ and later wishes it knew what the material was, the fix is usually to
accumulate the composition and defer the colour, because the palette is a cheap function evaluated once
per pixel rather than data that has to be carried per fragment.

The cost is a hard switch at interfaces. A pixel takes one treatment, the dominant one, so a water and
sand boundary is a one-pixel-wide change of shading model rather than a blend. At the scale these
materials are drawn it is not visible, but it is a real approximation and it would become visible with a
much larger splat radius.

## Porting a treatment: the shading is not the treatment

The claim at the top of this page — that identity lives in the reconstruction — has a practical
consequence that is easy to agree with in the abstract and easy to violate in practice. Here is the
violation, because it is worth recognising by sight.

A water treatment consists of a reconstruction and a shading model. The shading model is the part that
*looks* like the material: Beer–Lambert absorption, a Fresnel-weighted sky, a grazing rim, a tight
specular, foam. So that is the part a port copies. Port all of it, faithfully, and light the *old*
reconstruction with it, and the result looks like the old water. Not similar to it — essentially
identical, because:

$$
\text{colour}(u) = \text{shade}\big(\underbrace{t(u)}_{\text{thickness}},\; \underbrace{n(u)}_{\text{normal}},\; \underbrace{\alpha(u)}_{\text{opacity}}\big)
$$

and every argument is a **field**. If $t$ is the local splat sum $A(u)$ then $t$ is lumpy at the particle
spacing, so $e^{-\sigma t}$ is lumpy; if $n$ is $\nabla A$ over four neighbour taps then $n$ wobbles per
particle, so a $\cos^{70}$ specular becomes thousands of little highlights. The shading was correct. It
was correctly lighting the wrong surface.

What makes this failure worth a section is that **it is silent in every way a review can check.** The
code compiles. The constants match the proposal. Every term of the lighting model is present and can be
pointed at line by line. The commit message is true. Nothing is wrong except the picture, and the picture
is the only place it shows.

> **The diagnostic:** put your render next to the render you were copying, at the same time, and look.
> Not the code next to the code. If you cannot produce that pair, you have not verified the port.

The generalisable form: **when you port a look, port the fields first and the lighting last.** A shading
model is a function; a reconstruction is what the function is a function *of*. Copying a function without
its domain gets you the old picture with new arithmetic.

### What the reconstruction costs when it has to run in a browser

The reconstruction is the expensive half — [[fluid-rendering]] measures the distance transform alone at
about 31% of the water pipeline — so the temptation to skip it is real. Three things make it affordable
in a real-time frame, and they are all structural rather than clever:

**Run it at half resolution.** Optical thickness is genuinely low-frequency: it is the distance from a
pixel to the surface, and that quantity has no particle-scale detail in it by construction. Halving the
resolution of the blur, the threshold, the jump flood and the distance pass costs a quarter of the
pixels on every one of them, and the resolve upsamples the distance field bilinearly at no visible cost.
The *silhouette* still comes out crisp, because opacity is a near-hard function of that distance rather
than a blur of the density.

**Count passes, not pixels, at small sizes.** Measured on one RTX 4090 in Chromium, the eleven extra
render passes cost 0.028 ms at $480^2$ and 0.052 ms at $1080^2$ — a factor of 1.9 for a factor of 5.1 in
pixels. Fitting the two gives roughly a **fixed 0.020 ms plus 0.025 ms per megapixel**: at demo
resolutions the fixed term, which is attachment setup for a dozen render passes, dominates. That is the
opposite of the intuition "screen-space passes cost pixels", and it means the first optimisation to
reach for is *merging passes*, not shrinking them.

**Let the alpha carry the transmission.** A body of water is see-through, and the naive way to draw that
is to sample the background and mix. Under **premultiplied** alpha blending — $\text{out} = c_{\text{src}}
+ (1-\alpha)\,c_{\text{dst}}$ — you can instead emit $\alpha = 1 - e^{-\sigma t}$ and let the framebuffer
do it, so Beer–Lambert's transmission *is* the alpha channel and the shader never reads what is behind
it. Opaque materials drawn in the same pass are unaffected: emitting $c\alpha$ under `srcFactor = one` is
the same arithmetic as emitting $c$ under `srcFactor = src-alpha`.

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

### The number that is not a number

The device clock has its own version of this trap, and it is nastier because the readings look precise.
Browsers deliberately **quantise** GPU timestamps, because a fine-grained clock shared across origins is
a side channel. In Chromium the quantum observed here was $2^{15}$ ns $\approx 32.8$ µs, and disabling
the quantisation feature only reduced it. A screen-space pass that costs 30 µs is therefore *entirely
inside one tick*: every measurement of it comes back as $0$, one quantum, or two, and the ratios between
those are meaningless.

Two habits fix it, and using both is the point — they are independent, so agreement is evidence.

1. **Difference against a matched control.** Time the frame with the stage and without it, in the same
   run, on the same particle state. Both readings are quantised, but the *difference* is a difference of
   two nearby quantised values and is right to within one quantum.
2. **Amplify and take the slope.** Run the stage $K$ times inside one timed region and fit
   $T(K) = T_0 + K c$. At $K$ large enough the total leaves the quantum far behind. Keep $K$ modest —
   at hundreds of passes per submission you stop measuring the frame you actually draw and the slope
   inflates.

> **The tell:** if every timing you have is an exact multiple of the same number, that number is the
> quantum and you have measured nothing.

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
**Related:** [[fluid-rendering]] for the water pipeline in full, [[scattering-material-properties]] for the other half of making four materials share one pipeline (the solver half), [[filters-and-samples]] for the
filtering and sampling facts underneath all of it, [[real-time-cost]] and [[fixed-point-atomics]] for
the dispatch-overhead result the cost section leans on, [[material-showcase]] for what the four
canonical materials actually do.
