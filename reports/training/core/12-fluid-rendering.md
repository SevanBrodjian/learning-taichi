# Rendering the MPM fluid: from particles to a believable liquid

**The key idea:** a weakly compressible MPM fluid is a few thousand dots drifting through a box, and a
scatter plot of those dots does not look like water — it looks like static. Closing that gap takes a stack
of cheap, well-chosen visual cues, none of which is physics.

This page is deliberately the **non-differentiable** corner of the project. Nothing here needs to survive a
gradient, so it reaches freely for hard thresholds, per-pixel branches, and image samples. That freedom is
itself the lesson: a controllable world model eventually has to serve both masters — a forward pass that
looks like the world and a backward pass that can steer it — and seeing the visual-quality regime on its own
shows how much of what sells an image is exactly the machinery a trainable core has to work around.

Three things are worth your time here, and the second and third are the ones that generalize:

1. **The pipeline** — four stages that turn a point cloud into a liquid.
2. **Which layer a visual defect lives in** — the single most useful diagnostic skill on this page.
3. **The render, not the physics, is the bottleneck** — and the GPU port that fixed it.

## The pipeline in four moves

Reconstruct a **surface** from the particles, shade the **body** with depth color and a bent background, add
**reflection and highlights** at that surface, and mark **foam** where the water is torn up.

![How one frame is built, left to right: raw MPM particles as a scatter (reads as dots, not liquid); the density field they splat into; the surface reconstructed from that field, shown as a clay model lit only by its normals; and the final shaded frame with depth color, refraction, reflection, and foam composited on.](/api/data/learning-taichi/runs/realistic-rendering/non-differentiable-fluid-renderer/breakdown.png)

### Surface: metaballs from particles

The biggest single step from "dots" to "liquid" is giving the fluid a **surface**. Treat each particle as a
soft blob of influence, sum the blobs into a smooth field, and declare the surface a level set of it:

$$D(u) = \sum_{p} \exp\!\left(-\frac{\lVert u - x_p \rVert^{2}}{2\sigma^{2}}\right).$$

$D$ is a **density field** — large where particles cluster, falling smoothly to zero in empty space. The
width $\sigma$ is the one knob that sets how the fluid reads: turn it up and neighbors merge into one smooth
mass but fine droplets smear away; turn it down and the surface gets lumpy and individual particles show.
Smoothness against detail, directly.

*Implementation note worth keeping:* summing a Gaussian centered on every particle is identical to
histogramming particles into pixels and blurring **once**. Far cheaper, same field.

The surface needs a **normal**, because every lighting cue depends on it. $\nabla D$ points into the liquid,
so the outward in-plane normal is $-\nabla D / \lVert\nabla D\rVert$. For shading, that 2-D normal is lifted
into a fake third dimension:

$$N = \operatorname{normalize}\big(-\partial_x D,\; -\partial_y D,\; k\big).$$

Where density is flat (deep interior) the slopes vanish and $N$ faces the viewer. Near the edge the slopes
dominate and $N$ tilts to graze the boundary. Small $k$ exaggerates the tilt into a curved, glassy look;
large $k$ flattens it toward a pane of glass.

### Body: Beer-Lambert absorption gives volume

A flat fill of one blue is the giveaway of a fake liquid. Real water darkens with the amount of it a
sightline crosses:

$$\mathcal{T} = e^{-\sigma_a T}, \qquad c_{\text{body}} = \mathcal{T}\, c_{\text{shallow}} + (1 - \mathcal{T})\, c_{\text{deep}}.$$

Thin water stays pale and see-through; thick water turns deep saturated blue. This single term is what gives
the body volume instead of a cutout.

### Surface optics: refraction and Fresnel

**Refraction** is the cue that most makes a viewer believe the liquid is really there — sample the background
at an offset driven by the normal and thickness, $c_{\text{refr}} = \text{background}(u + s\,T\,n_{2D})$. It
only works if the background *has structure to displace*, which is why the tank sits in front of soft lights
and a horizon rather than a flat wall.

**Fresnel** makes the surface partly a mirror, depending on angle — clear looking straight down, bright and
sheet-like at a grazing angle. Schlick's approximation captures it cheaply:

$$F = F_0 + (1 - F_0)\,(1 - \cos\theta)^{5}, \qquad F_0 \approx 0.02 \text{ for water-air}.$$

Conveniently, $\cos\theta$ is exactly the third component of $N$: near one in the flat interior (clear), near
zero at the rim (bright reflective edge). That is why every blob's rim and every wave's far edge lights up.

### Finishing: specular and foam

A **specular** glint via Blinn-Phong, $(N\cdot H)^{\alpha}$ with $H$ bisecting light and view, flares exactly
where a crest swings to face the light. **Foam** marks water that is fast (splat particle speed, look for
high values at the surface) or genuinely thin and torn. Gate foam to the surface band — without gating, the
discreteness of the particles speckles white flecks through calm interior and reads as noise.

## The diagnostic skill: which layer is the defect in?

Two defects held this renderer back, and they are textbook cases of problems a newcomer misattributes.
**The liquid did not move like water. The calm body showed air holes inside.** The first is a *simulation*
problem that no shader fixes; the second is a *reconstruction* problem that no solver tuning removes.

> **Naming which layer a visual defect lives in — dynamics, reconstruction, or shading — before reaching for
> a fix is the whole skill.** It separates a real correction from optical spackle over a dynamics bug.

### Sluggish motion is a physics problem

The fluid's whole constitutive law is a pressure resisting volume change, $p = -E(J-1)$. Three separate
things make it look like syrup:

**Compressibility.** Real water is nearly incompressible, so a real splash snaps into sheets and jets. Low
$E$ makes the fluid squishy — it visibly compresses, stores the blow, and oozes. Raise $E$ and motion gets
crisper. **The cost is stability.** The pressure wave travels at $c = \sqrt{E/\rho}$, and an explicit step is
stable only while a wave cannot cross a cell in one step (the CFL condition):

$$\Delta t \;\lesssim\; \frac{\Delta x}{c} \;=\; \Delta x\,\sqrt{\frac{\rho}{E}}.$$

**Memorize this scaling: $\Delta t \sim 1/\sqrt{E}$.** Four times stiffer — noticeably more water-like —
halves the timestep and doubles the substeps for the same clip. Stiffer water is more expensive water, and
pushing $E$ without shrinking $\Delta t$ blows the sim into non-finite noise that renders as static.

**Numerical damping.** The P2G/G2P transfers are slightly lossy. A pure PIC transfer, which overwrites each
particle's velocity with the interpolated grid value, is the most dissipative and quietly bleeds energy, so
splashes settle too fast. The affine (APIC) transfer used here is far less lossy; re-injecting a fraction of
each particle's previous velocity (the FLIP correction) returns more. This is the same axis [[viscosity]]
controls on purpose — viscosity is resistance added deliberately, numerical damping is resistance smuggled
in by the scheme, and both make a fluid look thicker than intended.

**Playback rate.** A perfectly tuned splash still looks like syrup played at a fifth of real speed. Map
simulated time back to playback near real time. But do **not** paper over sluggish dynamics by speeding up
the video — if the motion is gloopy because the fluid is too soft or too damped, faster playback just gives
fast gloop. Fix the dynamics first.

### Interior holes are a reconstruction problem

The naive reconstruction asks **one threshold to do two jobs**: decide *where there is liquid at all*
(opacity) and, through its gradient, decide *which way the surface faces* (shading). Those jobs have
conflicting needs.

The particles are a discrete, random sample of a body that is supposed to be solid. Counts in any small patch
fluctuate the way independent random counts do, so even deep inside a calm body the blurred density
**ripples** — and wherever a ripple dips below the isovalue, that spot is declared air. The body speckles with
holes, worst exactly where it should look most solid. Raising the isovalue carves more; lowering it swallows
thin sheets. **No single threshold works**, because the interior wants a generous threshold and the surface
detail wants a strict one.

The fix is to stop using one field for both jobs:

- **A filled interior mask answers "is there liquid here."** Threshold low to capture the body with margin,
  then repair the discreteness — a morphological closing seals pinholes, and an enclosed-pocket fill closes
  interior gaps *below a size cap*. The cap is the important part: it fills the small Poisson holes that
  should not exist while leaving genuine cavities (a splash crater, the air tube inside a breaking wave)
  alone. This mask decides opacity.
- **The density gradient still answers "where is the surface."** Shading normals are unchanged — the fine
  slope information is what makes the surface read as curved. It simply no longer gets a vote on opacity.

**Bonus:** optical thickness now comes from a *distance transform* of the filled mask — distance from each
interior pixel to the nearest air. Zero at the edge, growing smoothly toward the core, speckle-free by
construction. The depth color rides on clean geometry instead of a random particle count.

![Left, the prior renderer: the liquid body speckled with dark interior holes wherever particle density randomly dipped below one isovalue, worst in the calm thick regions. Right: a filled interior mask decides opacity while the density gradient is kept only for surface normals, so the body reads as a continuous solid volume with air only where there is genuinely air.](/api/data/learning-taichi/runs/realistic-rendering/improve-basic-fluid-sim-realism/before_after_dambreak.png)

## The render was the bottleneck, not the physics

Here is the result that most changes how you think about simulated worlds.

The MPM step — tens of thousands of particles on a $128^2$ grid — is a **sub-millisecond** job on a modern
GPU. The renderer that makes those particles look like water splats, blurs, distance-transforms, samples a
refracted background, and layers Fresnel, specular, foam, caustics, bloom, and a tone map, all at $1080^2$.
In single-threaded numpy that costs **~1.7 seconds per frame** — while the GPU that just ran the physics sits
idle.

![Measured render time per 1080-squared frame on the same 30000-particle dam-break frame, log scale. CPU renderer about 1686 ms. Full GPU renderer including per-frame particle upload and image read-back, 13 ms. GPU render compute alone with data already resident, 6.4 ms.](/api/data/learning-taichi/runs/realistic-rendering/gpu-accelerate-fluid-renderer/benchmark.png)

| renderer | ms / frame | frames / s | speedup | 130-frame scene |
| --- | --- | --- | --- | --- |
| CPU (numpy/scipy) | 1686.3 | 0.59 | 1x | 219 s (~3.7 min) |
| GPU (Taichi, full) | 13.0 | 77 | 130x | 1.3 s wall |
| GPU (Taichi, device-only) | 6.4 | 157 | 265x | 0.83 s compute |

**Scope this honestly:** one representative frame, one scene, one resolution and particle count, one 4090
running uncontended, JIT compile excluded and reported separately. It is a timing on this setup, not a
universal constant.

### Three ports worth understanding

**Splat becomes an atomic scatter.** Every particle, in parallel, adds into the cell it lands in; parallel
writes collide, so the add must be atomic. **This is exactly the P2G primitive from [[mls-mpm-forward]]** —
the same operation that moves physics onto the grid moves particles onto the screen.

**The Gaussian blur becomes separable.** A direct 2-D convolution costs $O(r^2)$ per pixel. A Gaussian is
separable — the 2-D kernel is an outer product of two 1-D kernels — so blurring rows then columns costs
$O(r)$. At $\sigma=7$, $r=28$, that turns a 57×57 window (~3200 taps) into two 57-tap passes: a **28x**
reduction *before any parallelism*.

**The distance transform becomes jump flooding.** scipy's exact transform is inherently sequential with no
GPU analogue. Jump flooding (JFA) instead propagates nearest-seed candidates in passes with halving step
sizes $2^k, \dots, 2, 1$; each pixel adopts whichever neighbor carries a closer seed. Cost is
$O(\log_2 n)$ full-image passes — about eleven at $1080^2$ — each embarrassingly parallel. **This is the
general move when a favorite CPU routine is sequential:** find the parallel algorithm that computes the same
thing in a logarithmic number of parallel sweeps.

*One regression risk:* the no-holes fix relied on connected-component labeling, which is awkward on a GPU.
The port swaps it for a **bounded morphological closing** — dilation then erosion over a disk of radius $R$,
which fills gaps narrower than $\approx 2R$ and leaves wider ones untouched. It fills by *width* rather than
*area*, so a narrow-but-large enclosed pocket would be sealed where the CPU version kept it open. These
scenes never produce one; another could.

![The dam-break breaking-wave frame rendered by the CPU renderer on the left and the GPU renderer on the right, from the same particles. Visually indistinguishable: a single continuous depth-graded volume in both with no interior holes, and the large dark air pocket under the curling wave preserved in both.](/api/data/learning-taichi/runs/realistic-rendering/gpu-accelerate-fluid-renderer/cpu_vs_gpu_dambreak.png)

### Keep it on the device

The device-only render is 6.4 ms but the full per-frame time is 13 ms, and **the entire difference is the two
host-device transfers**. A naive port that shuttled each stage out to numpy and back would pay that cost a
dozen times per frame and throw away most of the speedup. On a GPU the arithmetic is nearly free and the
memory traffic is the budget: cross the host boundary as few times as possible, ideally twice.

> **The lesson:** the instinct is to blame the simulation. Here the physics was already free and the render
> was the whole cost. A world that cannot be rendered quickly cannot be watched, iterated on, or folded into
> a training loop that scores frames — a slow renderer is a slow feedback loop. Cheap physics and an
> expensive render is the normal shape of the problem, not a quirk of this pipeline.

## What this does and does not model

This is a **stylized 2-D side view, not light transport.** The thickness driving absorption and refraction is
a screen-space proxy, not a real optical path through a 3-D volume. There is no true refraction with a
measured index, no caustics (they are painted), no inter-reflection, and no measured absorption spectrum —
the colors and coefficients are chosen by eye. The surface normal is a 2-D gradient lifted into a fake third
dimension, so the roundness is an artistic choice rather than recovered geometry.

What the pipeline demonstrates is how far a stack of cheap cues carries a raw particle cloud toward looking
real. It gets most of the way to a double-take and stops short of photoreal, and the gap that remains is
exactly the 3-D light transport it never attempts. Making it fast did not make it light transport.

---

**Code:** `sim/fluid_render.py`, `sim/fluid_render2.py`, `sim/fluid_render_gpu.py`, `sim/fluid_showcase_gpu.py`.
**Related:** [[fluid-color-mixing]] — advecting a per-particle dye through the solver.
