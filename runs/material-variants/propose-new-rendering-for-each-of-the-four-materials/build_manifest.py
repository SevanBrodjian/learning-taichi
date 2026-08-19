"""Assemble manifest.json for this run: prose here, numbers from metrics.json, page from disk.

Kept as a script rather than a hand-written JSON blob so the manifest can never drift from the
bespoke page or from the measured costs, and so re-running after an edit is one command.
"""
import io
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
REL = "runs/material-variants/propose-new-rendering-for-each-of-the-four-materials/"
M = json.load(io.open(os.path.join(D, "metrics.json"), encoding="utf-8"))
C456 = {k: v["gpu_ms"] for k, v in M["render_cost_by_res"]["456"].items()}
HTML = io.open(os.path.join(D, "bespoke_page.html"), encoding="utf-8").read()


def v(name, cap):
    return {"type": "video", "src": REL + name, "caption": cap}


def im(name, cap):
    return {"type": "image", "src": REL + name, "caption": cap}


SUMMARY = """
The four materials now differ by **how they are drawn**, not by hue. The decisive evidence is the
greyscale test: force every material to the same neutral albedo and convert the output to luminance.
Today that produces four literally identical mushy blobs. With the proposed treatments it produces a
smooth dark see-through body (water), a flat solid inside a hard constant-width outline with a grid
that stretches as it deforms (rubber), a fine speckle of individual grains (sand), and a soft matte
pile (snow). Each material also gets a second option so there is something to choose between, and
every clip is shown against the current renderer on the same scene, same seed, as video.

Two things did not work, and both matter for the decision. **Snow is nearly a no-op** — that was the
brief's instruction, since the current look is right for powder, but the consequence is that snow is
identified *by elimination* rather than by a cue of its own. And the water reconstruction **loses the
airborne spray** the current splat keeps: an isolated droplet does not clear the "is there water here"
threshold and disappears. On cost, measured device time at the demo's own canvas and particle count
puts snow at 1.0x the current renderer (free), water and flat rubber at 1.7-1.9x, and the sand grains
at 3.6x -- all comfortably inside a ~10 ms drawing budget on a 4090, with sand the one to watch on the
iPad the demo is actually pinned to.
""".strip()

FULL = """
## What was built

`sim/material_render.py` is a standalone Taichi renderer holding five treatments plus a faithful port of
the demo's own shader. Nothing in `harness/dashboard/src/components/mpm/` or `sim/physics/` was touched;
motion comes from `sim.physics` unchanged (a forward sim, no gradients anywhere).

**The baseline is a line-by-line port** of the demo's `fs_splat` + `fs_resolve` pair in `mpm4.js`: the
same compact kernel $w=(1-r^2)^2$, the same additive (colour*w, w) accumulation, the same
central-difference normal with $n_z = 1.6\\,\\text{iso}$, the same diffuse/specular/edge terms, the same
rim brightening, and the demo's own constants (radius 0.034 NDC = 0.017 of the unit domain, iso 2.6).

**Every scene is seeded at the demo's own areal particle density** -- `DENSITY = 500/(pi*0.075^2)`
= 28,294 particles per unit area, about 1.7 per grid cell, straight out of `demo4.js`. This is
load-bearing: the accumulated weight the iso threshold is compared against scales with that density, so
a proposal that only looks good at eight times the particle count is not a proposal for this demo. It
is also why the solo scenes are small (1,076 particles for `slam`) -- that is what the demo shows.

## The treatments

**Water (options A "glass" / B "tinted film").** A port of this repo's own screen-space iso-surface
work (`sim/fluid_render2.py`, `runs/realistic-rendering/*`), not a re-derivation. Density splat ->
separable blur -> a *filled interior mask* that owns opacity, kept separate from the density gradient
that owns normals -> jump-flood distance transform for a speckle-free optical thickness ->
Beer-Lambert depth colour over a refracted background, Fresnel, rim, Blinn-Phong, surface-gated foam.
Option B keeps the reconstruction, drops the background sampling and the chromatic dispersion entirely,
and runs less than half the absorption, so it reads as a thin lit liquid rather than a deep body.

**Rubber (A "border + printed grid" / B "border, flat body").** Three moves. (1) The silhouette is
low-passed -- blur the binary mask, re-threshold at 0.5 -- so the boundary is one smooth closed curve
instead of a lumpy cluster. (2) A **constant-width** dark border from the unblurred distance-to-outside,
so the outline does not thin where the body thins; a closed line of constant width is what the eye reads
as "the edge of a thing". (3) Option A additionally splats each particle's **rest position** with the
same kernel and divides by the weight, giving a smooth material-coordinate field; a grid evaluated in
those coordinates is painted *on the material* and therefore stretches and shears with it. The rest
positions are already in memory, so this costs one extra splat and no physics.

**Snow (A "powder" / B the current treatment, unchanged).** Deliberately the same two passes as today.
The tight Blinn-Phong glint is removed (it reads as wet plastic), the hard iso cut becomes a soft powder
fringe, thin snow is brightened, packed crevices are darkened, and a fine crystal grain plus a sparse
sparkle is gated to the surface. Measured cost: 0.307 ms against the current renderer's 0.309 ms.

**Sand (A "grains over a packed body" / B "loose grains").** Six irregular sprites per particle, each
with its own hashed offset, radius (skewed small -- many fines, few coarse), ellipse aspect and
rotation. Overlaps are resolved with an atomic max on the grain's random *priority*; because the grain's
shade is a deterministic function of that same priority, the winner's shade is recovered in the resolve
pass without a second buffer. Option A composites the grains over an opaque packed body, B leaves the
gaps open. A first attempt rendered vertical corduroy through the slab, because the grain jitter is
keyed to the particle index and the slab had been seeded on a lattice; granular slabs are now seeded
with `seed_box` and `seed_lattice` is kept for the fluid it was written for.

## Cost, and the number that is not the cost

Timing was measured twice on purpose. A wall clock around `ti.sync()` reports 1.0-5.2 ms per frame --
**and the same 3.3 ms at 360x360 as at 1080x1080**, a nine-fold change in pixels. A cost that does not
move with the thing it should be proportional to is not measuring that thing; it is measuring 25-30
Python-side kernel launches. So the reported numbers come from Taichi's kernel profiler (CUDA events),
summing device time over every rendering kernel in the frame, with warm-up excluded. Both are in
`render_cost.json` so the gap is visible rather than hidden.

At 456x456 (the demo's canvas) and 16,384 particles on one RTX 4090:
current 0.309 ms, snow 0.307, rubber-flat 0.516, water-film 0.542, water-glass 0.585, rubber-grid
0.753, sand-loose 1.082, sand-packed 1.118. A per-kernel breakdown of the water pipeline puts the
**distance transform at 31% of its device time** (19.7% + 9.6% for the jump-flood passes, 2.1% for
seeding), and it exists only to produce a smooth optical thickness -- the first thing to approximate if
a budget bites.

The scaling is the more useful result. The screen-space treatments are pixel-bound and barely move with
particle count; the sand treatment is geometry-bound (six sprites per particle is 98,304 instances at
the demo's cap). And the *current* renderer scales fastest of all with canvas size, because its splat
radius is a fixed fraction of the canvas and its pixel footprint grows quadratically -- at 1080x1080 it
costs more than the water treatment it is being compared against.

For projecting onto the browser, the one thing that transfers cleanly is the dispatch count: the
screen-space treatments issue 25-30 passes per frame against the current renderer's 2, and this repo
already measured the dispatch floor at 1.11 us inside a recorded WebGPU command buffer versus 55.6 us
for a Taichi kernel launched from Python (`dispatch_floor_us`, from
`runs/material-variants/webgpu-port-of-the-interactive-simulation`). Recorded once into a command
buffer, the pass count is nearly free; submitted one at a time it is fatal.

## Scenes

Two solo scenes per material. `slam` is canonical (`sim.physics.scene`): a disk released at y=0.60 with
a 6 m/s downward kick. `fill` is **task-local and declared as such**: a slab of the material filling the
bottom of the tank plus a disk dropped into it, seeded with the canonical helpers but with a geometry
that is not in `sim.physics.scene`. It exists because the canonical scenes hold 1-2.5k particles at the
demo's density, and at that volume a fluid ends its life as a one-cell-thick film -- a fair test of the
current look and a useless test of a water treatment whose whole claim is about depth. `fill` is the
scene the demo actually shows once a visitor paints for a few seconds. All four also run together on one
shared grid (`simulate_multi`), and the canonical `scene_pool` buoyancy setup provides water with a
rubber body submerged in it.
""".strip()

FINDINGS = """
Scoped to what was rendered: four materials, two solo scenes each, one four-material shared-grid scene,
one canonical buoyancy pool, all at the demo's areal particle density, all on physics
`phys-c518316a4a05`, all offline in Taichi at 720x720. No WGSL was written and the demo was not touched.

1. **The greyscale test separates all four with the proposed treatments and none of them today.**
   Rendering every material with an identical neutral albedo and converting to luminance, the current
   renderer produces four visually interchangeable mushy blobs across the whole clip. The proposed set
   produces four different shape languages: a smooth continuous body with an interior value gradient and
   a bright meniscus (water), a flat interior inside a hard closed outline carrying a deforming grid
   (rubber), a pixel-scale speckle with a ragged boundary (sand), a soft fuzzy matte pile (snow). This
   is a judgement about appearance and is labelled as one -- what is *measured* is only that the four
   pictures are no longer the same picture.
2. **Snow's identity is negative.** Its proposed treatment is within measurement noise of the current
   one both in cost (0.307 vs 0.309 ms) and in appearance, because the brief asked for snow to stay the
   reference. The consequence, visible in the greyscale tiles, is that snow reads as "the mushy one"
   only because the other three stopped being mushy. If snow ever needs a positive cue, this run did not
   find one.
3. **The water reconstruction trades spray for surface.** In the `fill` clip the current renderer throws
   a visible fan of droplets on impact; the proposed one loses most of them. This is structural, not a
   tuning miss: any treatment that thresholds a density to decide "is there material here" discards
   isolated particles, and lowering the threshold to keep them starts inflating the body. The splat
   renderer's one genuine advantage is that it never loses a particle.
4. **Changing only the shading arithmetic is free; changing the reconstruction is not.** Snow measured
   at 1.0x the current renderer, and the screen-space treatments at 1.7-2.4x, with per-grain sand at
   3.6x. All are inside a ~10 ms drawing budget on the measured GPU, so on this hardware the choice is
   not cost-limited.
5. **Which treatment is "expensive" depends on the resolution-to-particle ratio, and the ranking flips.**
   The screen-space treatments are pixel-bound; the sand sprites are geometry-bound; and the *current*
   splat is the fastest-growing of all in canvas size, because its radius is a fixed fraction of the
   canvas. At 1080x1080 the current renderer (1.501 ms) costs more than the full glass water treatment
   (1.268 ms).
6. **A wall clock is the wrong instrument here.** Timed on the host, every screen-space treatment reports
   ~3.3 ms at 360x360 and ~3.3 ms at 1080x1080. The measurement is dominated by Python-side kernel
   launches and would have supported a completely wrong conclusion about what these treatments cost.
""".strip()

HYPOTHESIS = """
**Mechanism proposed for the observed result:** material identity in a particle renderer is decided by
the *reconstruction* -- the operator that turns points into a field and a field into a region -- and not
by the shading applied afterwards. One reconstruction imposes one shape language (here: a lumpy
particle-scale surface, a soft ambiguous edge, a uniform interior), and everything downstream can only
recolour it. On that account the four treatments work because they are four different reconstructions
(iso-surface of a splat; filled mask plus distance field; low-passed silhouette plus material
coordinates; per-grain sprites), and the greyscale test works as a diagnostic precisely because it
strips away the one channel that survives a shared reconstruction.

This is a hypothesis, not a demonstration. It predicts three things this run did **not** test: (a) that
two materials given the *same* reconstruction and different shading will still fail the greyscale test
however different the shading is; (b) that a fifth material (smoke, glass, metal) is separable exactly
to the extent that a distinct reconstruction exists for it; (c) that the ordering of these treatments by
"how different they look" is stable across scenes and particle counts, which was checked on two scenes
and is otherwise assumed.

A cheap test of (a): render rubber and snow through the identical splat reconstruction but with
maximally different shading (matte vs glossy, dark vs bright at equal albedo) and run the greyscale
test. If they separate, the reconstruction claim is too strong and shading is doing more work than
claimed.
""".strip()

LIMITATIONS = """
* **These are proposals, not an implementation.** Everything is offline Taichi at 720x720. No WGSL
  exists, the demo page and `sim/physics/` are untouched, and the treatments have never met the demo's
  interaction: the poke/grab tool deforming a body mid-frame, the erase tool, or two materials mixing at
  a shared grid node. The compositing here is per-material layers stacked in a fixed order (solids, then
  water), so where two materials genuinely interpenetrate there is a 1-2 px seam artefact.
* **Cost numbers are CUDA on one RTX 4090, not WebGPU and not the iPad the demo is pinned to.** They are
  a floor for the browser cost of the same passes. The sand treatment (3.6x, 98,304 sprite instances at
  the demo's cap) is the one most likely to break first on a weak device, and no weak device was tested.
  `render_wall_ms` is reported but must not be quoted as a cost -- it measures this harness.
* **Snow's proposal is close enough to the current look that a reader may reasonably call it no change.**
  Reported as such rather than dressed up.
* **Water loses spray**, and on a dark stage a clear liquid is easy to lose entirely, because a
  see-through material shows you what is behind it and behind it is black. Both are visible in the
  clips. Neither was solved.
* **Water options A and B are close on these scenes.** A's extra cost buys background refraction, and
  the backdrop here is deliberately near-flat; the difference is clearest in the pool scene, where the
  refracted "background" is a submerged solid. On a lighter or busier stage they would separate more.
* **Scenes are not centralised in this repo, and one of the two solo scenes is task-local.** `fill` is
  defined in `sim/material_render.py`, not in `sim.physics.scene`, and is declared in the contract and
  on the page. Comparisons to other tasks' scenes are not apples-to-apples.
* **The `slam` scene at the demo's density (1,076 particles) tears the material into a thin shell.**
  That is the physics at that particle count, not the renderer; the scene is included as a robustness
  check across regimes, not as a showcase.
* **The baseline's fidelity is established by code correspondence, not by a pixel diff.** The port
  matches `mpm4.js` term by term with the demo's own constants, but the live WebGPU demo could not be
  captured in this environment (the browser pane would not composite a frame from it), so no
  side-by-side against the real thing exists.
* **Appearance is a judgement call.** Every claim about what a treatment "reads as" is exactly that. The
  only measured quantities here are frame costs, particle counts and pass counts.
""".strip()

RESULTS = [
    v("cmp_grey_tiles.mp4",
      "THE DECISIVE TEST. Every material forced to the SAME neutral albedo, output converted to "
      "luminance, four solo runs tiled. Left, the current renderer: four visually interchangeable "
      "mushy blobs for the whole clip -- with the hue removed there is nothing left, which is the "
      "reported problem made visible. Right, the proposed set: water is a smooth dark body that "
      "lightens toward its edge with a bright surface line, rubber is a flat solid inside a hard "
      "constant-width outline with a grid that stretches as it deforms, sand is a field of individual "
      "grains, snow is the soft matte pile. Snow is the weak one: it is identified by elimination."),
    v("cmp_fill_fluid.mp4",
      "WATER, current vs proposed, same scene and seed. Left the current splat iso-surface: an opaque "
      "pale-blue crust, lumpy at the particle scale. Right the screen-space iso-surface: a continuous "
      "body that is dark where it is deep and pale and see-through where it is thin, with one smooth "
      "surface line. Watch the impact: the current panel throws a fan of airborne droplets that the "
      "proposed one loses, which is the honest cost of thresholding a density."),
    v("cmp_fill_elastic.mp4",
      "RUBBER, current vs proposed. Left, a cloud of dots that happens to be ball-shaped. Right, one "
      "coherent solid: a smoothed silhouette, a dark border of constant pixel width, a flat interior "
      "with no particle-scale variation, and a grid painted in material coordinates that squashes and "
      "shears with the body as the ball lands on the slab."),
    v("cmp_fill_snow.mp4",
      "SNOW, current vs proposed -- deliberately the smallest change in the set, since the current "
      "mushy look is right for powder. What changes: the tight specular glint is gone (it read as wet "
      "plastic), the edge is a soft powder fringe rather than a hard iso cut, thin snow is brighter, "
      "and a fine crystal grain and sparse sparkle sit on the surface. Measured at 1.0x the current "
      "renderer's cost."),
    v("cmp_fill_sand.mp4",
      "SAND, current vs proposed. Left, a smooth yellow mound. Right, six irregular hashed grains per "
      "particle over an opaque packed body: the heap keeps the same angle of repose but is now visibly "
      "made of grains, with a rough grain-scale boundary."),
    v("cmp_fourup.mp4",
      "All four on ONE shared MLS-MPM grid, released together -- water, rubber, snow, sand, left to "
      "right. Distinctness is a property of the set, and this is the set. The proposed panel also shows "
      "the honest weak point: a clear liquid on a dark stage is the hardest of the four to see, because "
      "it shows you what is behind it."),
    v("cmp_pool.mp4",
      "The canonical scene_pool buoyancy setup on the new physics: a rubber disk submerged in water. "
      "Solids are composited first and the water absorbs and refracts THEM rather than an empty "
      "background, which is what makes the ball read as being in the pool rather than pasted on it. "
      "The current panel is the same scene as an opaque light-blue slab."),
    v("opt_fill_fluid.mp4",
      "WATER, the choice: current, option A (glass -- full reconstruction with background refraction "
      "and chromatic dispersion, 1.9x the current cost), option B (tinted film -- same reconstruction, "
      "no background sampling at all and less than half the absorption, 1.8x). On this deliberately "
      "near-flat dark backdrop A and B are close; the difference is clearest where there is something "
      "behind the water to bend, as in the pool scene."),
    v("opt_fill_elastic.mp4",
      "RUBBER, the choice: current, option A (border plus the material-coordinate grid, which makes the "
      "deformation legible), option B (border plus a completely flat body, the cleaner graphic read). "
      "A costs one extra splat over B."),
    v("opt_fill_sand.mp4",
      "SAND, the choice: current, option A (grains composited over an opaque packed body), option B "
      "(the same grains with the gaps between them left open, so the background shows through the "
      "pack). Both measure the same; A reads as a dense pack, B as a loose scatter."),
    v("cmp_slam_sand.mp4",
      "The SECOND scene, so no claim rests on one: the canonical `slam` disk driven into the floor at "
      "6 m/s. At the demo's density this is only 1,076 particles and the material tears into a thin "
      "shell -- that is the physics at that particle count, not the renderer. Included because a sparse, "
      "violently scattered regime is where the treatments could have fallen apart, and the grain "
      "treatment in fact handles scattered material more gracefully than the blob does."),
    im("contact_sheet.png",
       "Every material against its two options and the current look, one frame from the `fill` scene. "
       "The whole proposal on one sheet."),
    im("grey_tiles_current.png",
       "The greyscale test, current renderer, single frame. Four materials, identical albedo, "
       "luminance output -- and four identical pictures."),
    im("grey_tiles_proposed.png",
       "The greyscale test, proposed treatments, same frame. Four different shape languages with no "
       "hue information available."),
    im("breakdown_water.png",
       "How the point cloud becomes a surface, left to right on one water frame. The blurred particle "
       "density is visibly speckled everywhere (a random point sample has relative count noise "
       "1/sqrt(lambda)); the filled interior mask thresholds low and morphologically closes that into a "
       "solid body with no holes, and owns opacity; the distance transform gives a smooth, speckle-free "
       "thickness; the surface normal (in-plane components amplified 5x for display) is flat and "
       "viewer-facing across the whole interior with a tilted band exactly along the free surface -- "
       "which is why Fresnel keeps the body clear and lights only the rim."),
    {"type": "table", "src": REL + "render_cost.json",
     "caption": "Measured render_gpu_ms and render_wall_ms per treatment at 456x456, 720x720 and "
                "1080x1080, 16,384 particles, RTX 4090. GPU time from Taichi's kernel profiler (CUDA "
                "events); wall time from the host for the identical loop. The gap between them is "
                "Python-side kernel launch overhead and is a property of this harness, not of the "
                "algorithms."},
]

man = {
    "schema_version": 2,
    "task_id": "propose-new-rendering-for-each-of-the-four-materials",
    "direction": "material-variants",
    "title": "A distinct look for each of the four materials",
    "tldr": ("Gave water a see-through screen-space surface, rubber a hard constant-width border with a "
             "grid that deforms with it, and sand real grains, so the four materials survive having "
             "their colour deleted -- but snow ended up a near-no-op that only reads by elimination, "
             "and the water surface eats the airborne spray the current splat keeps."),
    "status": "active",
    "physics_version": M["physics_version"],
    "objective": (
        "Propose a distinct visual treatment for each of the four canonical materials -- water, rubber "
        "(elastic), snow, sand -- so that a viewer tells them apart by HOW THEY LOOK rather than by "
        "hue, and give the user enough to choose between. Every proposal is shown against the demo's "
        "current renderer on the same scene and seed, as video; the four are shown together on one "
        "shared grid; and the whole set is subjected to a greyscale test with identical albedos, which "
        "is the only honest check of the original complaint. Each proposal carries a measured "
        "per-frame cost. The demo page and sim/physics are strictly read-only for this task."),
    "summary": SUMMARY,
    "findings": FINDINGS,
    "full_report": FULL,
    "hypothesis": HYPOTHESIS,
    "limitations": LIMITATIONS,
    "results": RESULTS,
    "custom_html": HTML,
    "training_refs": [
        {"id": "material-appearance",
         "title": "Four materials, one shader: reconstruction is what makes them different",
         "file": "reports/training/core/17-material-appearance.md"},
        {"id": "filters-and-samples",
         "title": "Filters and point samples: convolution, separability, sampling noise",
         "file": "reports/training/prerequisites/06-filters-and-samples.md"},
        {"id": "fluid-rendering",
         "title": "Rendering the MPM fluid (extended with a forward link)",
         "file": "reports/training/core/12-fluid-rendering.md"},
    ],
    "metrics_used": ["render_gpu_ms", "render_wall_ms", "dispatch_floor_us", "physics_version"],
    "device": M["gpu"],
    "code": ["sim/material_render.py", "sim/material_render_run.py", "sim/material_render_cost.py"],
    "cost_at_456_gpu_ms": C456,
    "cost_over_current_at_456": M["over_current_at_456"],
    "scenes": M["scenes"],
}

with io.open(os.path.join(D, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(man, fh, indent=2, ensure_ascii=False)
print("manifest.json written:", os.path.getsize(os.path.join(D, "manifest.json")), "bytes")

missing = [r["src"] for r in RESULTS
           if not os.path.exists(os.path.join(D, os.path.basename(r["src"])))]
print("dangling media:", missing if missing else "NONE")
