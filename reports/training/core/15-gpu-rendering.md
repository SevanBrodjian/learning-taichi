# Rendering on the GPU: the render was the bottleneck, not the physics

The [[fluid-rendering]] and [[fluid-realism]] pages build a screen-space renderer that turns an MPM
particle cloud into a frame that reads as liquid. That renderer is written in numpy and scipy and runs on
the CPU, one full-image pass after another. It is also, measured honestly, the slowest thing in the whole
pipeline by two orders of magnitude. This page is about moving it onto the GPU with Taichi, what each stage
looks like when it becomes a parallel kernel, and the broader lesson the timing teaches: for a simulated
world that has to be watched, the cost is almost never the physics.

## Motivation: the physics is free, the picture is expensive

A weakly compressible MPM fluid at the scale used here is a few tens of thousands of particles pushed
through a $128 \times 128$ grid ([[mpm-in-context]] sets up why the particles are the only state that
matters). On a modern GPU that forward step is a sub-millisecond job. The renderer that makes those
particles look like water is a different animal. Per frame it splats the particles into a high-resolution
grid, blurs that grid with a wide Gaussian, runs a distance transform for optical thickness, samples a
refracted background at every pixel, and layers Fresnel, specular, foam, caustics, bloom, and a tone map on
top. At $1080^2$ that is a stack of million-pixel passes, and in single-threaded numpy it costs on the
order of **1.7 seconds per frame**. A two-scene, few-hundred-frame showcase therefore takes tens of minutes,
almost all of it spent in the renderer while the GPU that just ran the physics sits idle.

This matters beyond one demo. A world that cannot be rendered quickly cannot be watched, iterated on, or
folded into a training loop that scores frames, so a slow renderer is a slow feedback loop. Cheap physics
and an expensive render is the normal shape of the problem, not a quirk of this pipeline.

![Measured render time per 1080-squared frame on the same 30000-particle dam-break frame, on a log scale.
The CPU renderer takes about 1686 milliseconds, roughly one frame every 1.7 seconds. The full GPU renderer,
counting the per-frame particle upload and the final image read-back, takes 13 milliseconds, about 77 frames
per second. The GPU render compute on its own, with the particle data already resident on the device, takes
6.4 milliseconds, about 157 frames per second. The gap between the two GPU bars is the host-device transfer,
not the rendering.](/api/data/learning-taichi/runs/realistic-rendering/gpu-accelerate-fluid-renderer/benchmark.png)

## The measurement

The two renderers were timed on the same particle data, at $1080^2$, on one 4090 running the job alone so
the numbers are not corrupted by contention. The GPU kernels were compiled and warmed once before timing,
so the one-time JIT compile (about 5 seconds) is excluded from the per-frame figure and reported separately.

| renderer | ms / frame | frames / s | speedup | 130-frame scene |
| --- | --- | --- | --- | --- |
| CPU (numpy/scipy) | 1686.3 | 0.59 | 1x | 219 s (about 3.7 min) |
| GPU (Taichi, full) | 13.0 | 77 | 130x | 1.3 s wall |
| GPU (Taichi, device-only) | 6.4 | 157 | 265x | 0.83 s compute |

The headline is a **130x** speedup end-to-end and a **265x** speedup on the render compute itself. The
whole-scene number is the one that changes how the work feels: a scene that took nearly four minutes to
render now takes a little over a second. Video encoding to mp4 is separate, is bound by the codec and disk
rather than the render, and adds well under half a second per scene, so it no longer dominates anything.

Two honest caveats keep the claim scoped. The speedup is measured on one representative frame of one scene
at one resolution and particle count; it is a timing on this setup, not a universal constant. And the "full"
number includes a wasteful per-frame upload of the particle buffer, which is why it is twice the device-only
number; the device-only figure is the render compute the GPU actually does.

## How each stage becomes a kernel

The port keeps the pipeline identical and changes only where and how it computes. Every intermediate lives
in a Taichi field on the device, and only the final uint8 image is copied back to the host. The interesting
stages are the four that are not simply "do the same arithmetic per pixel in parallel".

### Splat by atomic scatter

The metaball splat sums each particle's contribution into a render grid. On the CPU this is a histogram. On
the GPU it is a **scatter**: every particle, in parallel, adds into the grid cell it lands in. Parallel
writes to the same cell collide, so the add must be atomic,

$$
D_{i,j} \leftarrow D_{i,j} + 1 \quad \text{for the cell } (i,j) \text{ under each particle},
$$

where $D_{i,j}$ is the accumulated count in grid cell $(i,j)$ and the updates run over all particles at once.
This is exactly the accumulation pattern of the particle-to-grid transfer in the MLS-MPM forward step
([[mls-mpm-forward]]), where each particle atomically scatters mass and momentum into the grid nodes it
touches. The same primitive that moves the physics onto the grid also moves the particles onto the screen.
A speed-weighted copy of the same scatter, accumulating each particle's speed, gives the field the foam
stage needs, at the cost of one extra atomic add per particle.

### Separable Gaussian blur

The splatted counts are noisy, so they are smoothed with a Gaussian of width $\sigma$. A direct 2D Gaussian
convolution touches every pixel in a $(2r+1) \times (2r+1)$ window, where $r \approx 4\sigma$ is the kernel
radius, so its cost per output pixel is $O(r^2)$. A Gaussian is **separable**: the 2D kernel is the outer
product of two 1D kernels, so blurring along the rows and then along the columns gives the same result at
cost $O(r)$ per pixel. For the widest blur here, $\sigma = 7$ and $r = 28$, separability turns a
57-by-57 window (about 3200 taps) into two passes of 57 taps, roughly a **28x** reduction in work before any
parallelism. Each 1D pass is a Taichi kernel that reads a shared row of precomputed weights, and the weight
rows for every distinct $\sigma$ are built once during warm-up and reused, so no kernel-weight upload
happens on the steady per-frame path.

### The distance transform by jump flooding

Optical thickness in [[fluid-realism]] comes from a Euclidean distance transform of the filled body: each
interior pixel is colored by how far it sits from the nearest air, which grows smoothly from zero at the
surface into the core and so drives clean depth color with no dependence on particle count. On the CPU this
is scipy's `distance_transform_edt`. That routine is inherently sequential and has no direct GPU analogue,
so the port uses **jump flooding** (JFA) instead.

JFA computes, for every pixel, the coordinates of the nearest seed pixel (here, the nearest air pixel), and
the distance is then just the length of that offset. It works by propagating candidate seeds in passes with
halving step sizes $s = 2^{k}, 2^{k-1}, \dots, 2, 1$. In each pass a pixel looks at the nine neighbors
offset by the current step, and adopts whichever carries a closer seed than the one it already holds. Large
steps carry a seed across the image in a few hops, and successively smaller steps refine the answer locally.
The whole transform therefore costs $O(\log_2 n)$ full-image passes for an $n \times n$ image, about eleven
passes at $1080^2$, each a simple parallel kernel. Ping-ponging between two seed buffers avoids a copy after
every pass. This is the general move when a favorite CPU routine is sequential: find the parallel algorithm
that computes the same thing in a logarithmic number of embarrassingly parallel sweeps.

### Preserving the no-holes property

The single most important thing the port must not regress is the filled-interior fix from [[fluid-realism]].
A metaball density thresholded at one isovalue punches air holes through a calm body wherever Poisson
fluctuations in particle count dip the interior below the threshold; the fix separates opacity (a filled
mask) from surface shading (the density gradient) and repairs the holes. The CPU does the repair with a
morphological closing plus a connected-component fill that plugs small enclosed pockets while leaving genuine
large cavities open.

The connected-component labeling is the awkward part on a GPU. The port replaces it with a **bounded
morphological closing**, which needs no connectivity analysis and is exactly parallel. A closing is a
dilation (a max over a disk) followed by an erosion (a min over the same disk); with a disk of radius $R$ it
fills any gap narrower than about $2R$ and leaves anything wider untouched. Choosing $R$ large enough to
swallow the sub-particle-spacing pinholes but smaller than a real cavity reproduces the fix by construction:
a calm body seals into a solid, speckle-free volume, while the genuine cavities in these scenes, a splash
crater and a breaking-wave barrel, are wide and survive. The side-by-side below confirms the property holds.

![The dam-break breaking-wave frame rendered by the CPU renderer on the left and the GPU renderer on the
right, from the same particles with the same formulas. The two images are visually indistinguishable. The
liquid body is a single continuous depth-graded volume in both, with no interior holes, and the large dark
air pocket under the curling wave, the barrel, is preserved in both rather than being filled in. Depth color,
the refracted studio background, the bright foam fringe on the crest, and the specular edges all match. The
GPU frame is produced about 130 times
faster.](/api/data/learning-taichi/runs/realistic-rendering/gpu-accelerate-fluid-renderer/cpu_vs_gpu_dambreak.png)

## Keep it on the device

The rest of the pipeline, surface normals from the density gradient, Beer-Lambert absorption, background
refraction, Fresnel, rim light, specular, foam, caustics, contact shadow, floor reflection, bloom, and the
tone map, is per-pixel arithmetic, and each stage is a straightforward parallel kernel with the same formula
as the CPU version. The one discipline that makes the whole thing fast is refusing to copy intermediates
back to the host. Every field, from the splatted counts to the composited image, stays on the device; the
particle positions go up once per frame and the finished uint8 image comes down once per frame, and nothing
else crosses the bus.

The benchmark shows why this is not optional. The device-only render is 6.4 ms, but the full per-frame time
is 13 ms, and the entire difference is the two transfers that do cross the bus. A version that shuttled each
stage out to numpy and back, as a naive port might, would pay that transfer cost a dozen times per frame and
would throw away most of the speedup. On a GPU the arithmetic is nearly free and the memory traffic is the
budget, so the rule is to move data across the host boundary as few times as possible, ideally twice.

## What is open, and the lesson

The speedup is a timing on one scene, frame, resolution, and GPU, run uncontended, not a claim that every
renderer sees 130x. The bounded morphological closing fills by width rather than area, so a narrow-but-large
enclosed pocket would be sealed where the CPU's area cap kept it open; these scenes never produce one, but
another could. And the renderer is still the stylized, non-differentiable 2D one of [[fluid-realism]]; making
it fast does not make it light transport. The lesson worth keeping is the diagnosis: the instinct is to blame
the simulation, but here the physics was already free and the render was the whole cost. Naming the expensive
layer before optimizing it is what turned a four-minute scene into a one-second one.
