"""Write manifest.json LAST, and refuse to write it if any media src is missing.

    .venv/Scripts/python.exe runs/.../write_manifest.py
"""
import json
import pathlib
import time

RUN = pathlib.Path(__file__).resolve().parent
REPO = RUN.parents[2]
REL = "runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/"
M = json.loads((RUN / "metrics.json").read_text())

B = M["frame_budget"]
O = M["ordering_pass"]["mean_y_end_webgpu"]
T3 = M["buoyancy"]["pool_three"]
LP = M["layout"]["before_phone_portrait"], M["layout"]["after_phone_portrait"]
LL = M["layout"]["before_phone_landscape"], M["layout"]["after_phone_landscape"]
W = M["water_reconstruction"]
W48, W72, W108 = (W["by_res"][k] for k in ("480", "720", "1080"))
WTAB = ("| resolution | pixels | before | after | chain (direct) | chain (amplified) |\n"
        "|---|---|---|---|---|---|\n" + "".join(
    "| %s | %s | %.4f | %.4f | %.4f | %.4f |\n" % (
        lab, format(v["pixels"], ","), v["before_gpu_ms"], v["after_gpu_ms"],
        v["chain_gpu_ms_direct"], v["chain_gpu_ms_slope"])
    for lab, v in (("480^2", W48), ("720^2", W72), ("1080^2", W108))) + "\n")
WFRAME = ("%.2f ms of GPU work against a 16.67 ms budget, up from %.2f."
          % (B["total_gpu_ms_water_rework"], B["total_gpu_ms"]))

results = [
    {"type": "video", "src": REL + "cmp_water.mp4",
     "caption": "THE WATER REWORK, one variable, on the demo's own opening scene. Both halves are the "
                "same build, the same physics, the same particle positions and the same snow, sand and "
                "rubber; the only difference is whether the water reads its optical thickness and its "
                "normal from a screen-space iso-surface or from four local taps of the raw splat "
                "accumulation. Left is what this task originally shipped."},
    {"type": "video", "src": REL + "cmp_water_pool.mp4",
     "caption": "The same one-variable comparison on water alone plus one rubber ball, so the surface "
                "and the spray are both in frame. Watch the surface line and the interior when the ball "
                "lands."},
    {"type": "image", "src": REL + "cmp_water_target.png",
     "caption": "Did it land where the proposal was. Left: T-020's own Taichi render of option B "
                "('film'), its own dam-break scene at 720px. Middle: the demo page now, WebGPU, its own "
                "pool at 600px. Right: the water this task originally shipped. DIFFERENT SCENES and "
                "different APIs, so this compares the treatment and not the pixels. Interior colour at "
                "mid-depth: T-020 (50,90,112), the page now (51,90,109); nothing was tuned against that "
                "number. Read that as a check on the colour pipeline, not as proof the two images are "
                "equivalent -- T-020's pool is deeper (optical depth ~3.2 against ~1.7 here) and the "
                "absorption curve is flat across that range."},
    {"type": "image", "src": REL + "still_water_after.png",
     "caption": "One frame of the demo's opening scene with the reconstruction on: a smooth interior, a "
                "clean surface line, and faint spray that still reads as spray."},
    {"type": "image", "src": REL + "still_water_before.png",
     "caption": "The identical particle state as it originally shipped: T-020's water shading applied to "
                "a thickness reconstructed from four local taps, which is lumpy at the particle scale."},
    {"type": "video", "src": REL + "cmp_buoyancy.mp4",
     "caption": "THE PHYSICS CHANGE, one variable. Snow, rubber and sand released at rest side by side "
                "at the same depth in one pool, identical initial conditions and identical substep "
                "schedule, and both halves drawn with the OLD shading so nothing but the material "
                "parameters differs. Left: phys-bebeaafbe73e, where every material has the same mass and "
                "all three blobs simply stay where they were put. Right: phys-c518316a4a05, where snow "
                "(rho 0.3) rises to the surface, rubber (1.2) sinks and sand (1.6) sinks further. There "
                "is no buoyancy force on either side."},
    {"type": "video", "src": REL + "cmp_render.mp4",
     "caption": "THE RENDERING CHANGE, one variable. The same build, the same physics and the same "
                "particle positions in both halves; only the resolve pass differs. Left: the single "
                "treatment the page shipped with. Right: water on T-020's option B (film), snow on "
                "option A (powder), sand on option A (grains over a packed body), rubber on the OLD "
                "treatment with a smaller splat kernel and an added border."},
    {"type": "video", "src": REL + "cmp_page.mp4",
     "caption": "Both changes at once, which is what a visitor actually sees. Kept separate from the two "
                "single-variable clips above on purpose."},
    {"type": "video", "src": REL + "demo_buoyancy_after.mp4",
     "caption": "The buoyancy scene as it now appears on the page, with the new treatments: snow floats "
                "as powder, sand sinks as grains, rubber sinks as a bordered body."},
    {"type": "video", "src": REL + "demo_render_after.mp4",
     "caption": "The demo's own opening scene on the shipped build: pool, rubber ball, snow block and "
                "sand heap, all four treatments, 333 substeps per frame at dt = 5e-5."},
    {"type": "image", "src": REL + "still_buoyancy_before.png",
     "caption": "Final frame, old physics: the three blobs have not moved relative to the water. Density "
                "did not exist, so there was nothing to separate them."},
    {"type": "image", "src": REL + "still_buoyancy_after.png",
     "caption": "Final frame, new physics, same shading: snow at the surface and partly out of the water, "
                "rubber submerged, sand on the bottom. Ordered by density."},
    {"type": "image", "src": REL + "still_render_before.png",
     "caption": "Final frame with the previously shipped single treatment. Sand and snow are the same "
                "glossy material in two colours."},
    {"type": "image", "src": REL + "still_render_after.png",
     "caption": "The identical particle state with the four treatments as they now ship: granular sand, "
                "matte powder snow, water on the screen-space iso-surface, and rubber with a visible "
                "border. Re-shot after the water rework -- the earlier version of this still showed the "
                "half-ported water."},
    {"type": "image", "src": REL + "shots/before_phone_landscape.png",
     "caption": "iPhone landscape, before: the control bar takes 251 px of a 390 px viewport and leaves "
                "the field a 141 px square, with the HUD chips covering most of what is left."},
    {"type": "image", "src": REL + "shots/after_phone_landscape.png",
     "caption": "iPhone landscape, after: the controls move to a side column and the field takes the full "
                "390 px height. 2.8x wider, 7.7x the area."},
    {"type": "image", "src": REL + "shots/before_phone_portrait.png",
     "caption": "iPhone portrait, before: the field is 392x676. The canvas backing store is square and the "
                "domain is the unit square, so this is not a crop, it is a 42% vertical stretch of the "
                "simulation."},
    {"type": "image", "src": REL + "shots/after_phone_portrait.png",
     "caption": "iPhone portrait, after: 390x390 and square, controls in three dense rows at 40 px tall."},
    {"type": "image", "src": REL + "shots/before_laptop.png",
     "caption": "Small laptop, before (1280x800). Included as the control: this layout is the one the user "
                "likes and it must not move."},
    {"type": "image", "src": REL + "shots/after_laptop.png",
     "caption": "Small laptop, after. Field 660 -> 658 px (the 2 px is the border moving inside the box) "
                "and the control bar is unchanged at 142 px. Not regressed."},
]

missing = [r["src"] for r in results if not (REPO / r["src"]).exists()]
assert not missing, "MISSING MEDIA: %s" % missing

manifest = {
    "schema_version": 2,
    "task_id": "incorporate-improved-materials-on-real-demo-page-and-improve-polish",
    "direction": "material-variants",
    "title": "Incorporate improved materials on the real Demo page, and improve polish",
    "tldr": ("The Demo page now runs the current canonical physics, so snow floats and sand sinks on "
             "the page's own solver instead of every material weighing the same; snow, sand and water "
             "got their new looks and rubber got two tweaks to its old one; and the simulation is finally "
             "square and usable on a phone, where it used to be stretched 42% tall. Water took two "
             "attempts: the first shipped T-020's water SHADING on the old lumpy reconstruction and "
             "still looked like the old water, and the reconstruction is what has now been ported. "
             "Nothing was done about the timestep, so a phone still runs it in labelled slow motion."),
    "status": "active",
    "created": time.strftime("%Y-%m-%d"),
    "physics_version": M["physics_version"],

    "objective": ("Bring the flagship Demo page up to the current standard: the new canonical physics "
                  "(phys-c518316a4a05, with per-material density, Poisson ratio and wall friction), the "
                  "rendering treatments chosen from T-020's proposals, and a layout that is usable on a "
                  "phone. Unlike the two tasks it combines, this one changes what the user looks at."),

    "summary": (
        "The Demo page was running materials with no concept of density, and now it is not. Every "
        "material carries its own density, Poisson ratio and wall friction, and the sink/float ordering "
        "falls out of the mass ratio alone on the page's own WGSL solver: on a three-blob pool scene, "
        "snow ends at mean height %.3f, rubber at %.3f and sand at %.3f, ordered by density, matching "
        "canonical Taichi to within canonical's own run-to-run noise on every scene tested. There is no "
        "buoyancy force anywhere and there must not be one. Water, snow and sand are on the treatments "
        "the user chose, and rubber keeps its old treatment with a smaller splat kernel and a real "
        "border. The whole drawing stage costs %.2f ms of a 16.7 ms frame at 16,384 particles against a "
        "solver that costs %.1f ms, so rendering is not what limits this page.\n\n"
        "Water needed a second attempt and the reason is worth stating plainly: the first version of "
        "this task ported T-020's water SHADING (Beer-Lambert absorption, a tight specular, a Fresnel "
        "rim, gated foam) without the screen-space RECONSTRUCTION that shading reads, so every quantity "
        "it lit came from four neighbour taps of the raw splat accumulation and the water stayed as "
        "speckled as before. The reconstruction is now in WGSL as eleven half-resolution render passes "
        "-- blur, threshold to a binary body, a jump-flood distance transform for optical thickness -- "
        "and the pool's interior colour lands within 3/255 of T-020's own render of the same treatment "
        "without anything being tuned to it (a check on the colour pipeline, not a proof of "
        "equivalence -- the two pools sit at different optical depths on a locally flat part of the "
        "absorption curve). It costs %.3f ms at 480 squared and %.3f ms at 1080 "
        "squared, measured both as a difference against a matched control and as an amplified slope, "
        "against a solver that costs %.1f ms.\n\n"
        "The layout had a bug worth naming: the field was sized so that on any viewport taller than it "
        "is wide the height won and the width clamped, which does not crop a square simulation domain, "
        "it stretches it. An iPhone in portrait was showing the physics 42%% too tall. It is now square "
        "at every viewport tested, controls are 40 px tall instead of 29, and a phone in landscape went "
        "from a 141 px field to a 390 px one. The large-monitor layout is untouched. Not addressed: the "
        "shared timestep still forces 333 substeps per frame whenever snow is present, so a phone runs "
        "this in honest slow motion rather than at 60 fps."
    ) % (O["snow"], O["elastic"], O["sand"], B["render_gpu_ms_1024_water_rework"], B["solver_gpu_ms"],
         W48["chain_gpu_ms_direct"], W108["chain_gpu_ms_direct"], B["solver_gpu_ms"]),

    "findings": (
        "1. PHYSICS. The demo's WGSL step now computes each particle's mass as p_vol*rho[mid] from a "
        "generated constants file, scatters a mass-weighted friction into a widened node-mass "
        "accumulator, and separates at all four walls with Coulomb friction on the tangent. On the "
        "three-solids-in-one-pool scene the ordering reproduces exactly: snow rises "
        "(delta rest_depth %+.4f, submerged_fraction %.3f), rubber and sand sink (%+.4f, %+.4f), and "
        "every one of the six verification scenes lands between 0.81x and 5.25x canonical's own "
        "self-noise band.\n"
        "2. RENDERING. Four treatments now share one resolve pass because the accumulation buffer's four "
        "channels carry per-material weight instead of premultiplied colour. Cost at 1024^2 went from "
        "%.3f ms to %.3f ms; almost all of the increase is a fixed +0.05 ms from the sand grain pass "
        "(6n instances, geometry-bound) and the per-pixel part of the resolve barely moved even though "
        "it now carries four shading models.\n"
        "3. WATER, and this is a CORRECTION to the first version of this finding. The shading was "
        "ported and the reconstruction was not, and for water the reconstruction is the look. It now "
        "runs T-020's build_masks in WGSL: a separable blur (whose sigma is chosen so the splat "
        "kernel's own width plus the blur equals the smoothing T-020 applied to a point histogram), a "
        "threshold at a fixed 0.24 of full packing, a jump-flood distance transform, and optical "
        "thickness read off that distance rather than off a local density count. Eleven extra render "
        "passes at half resolution. Measured cost of the chain alone, before-vs-after in the same run "
        "with the same instrument: %.4f ms at 480^2, %.4f at 720^2, %.4f at 1080^2, cross-checked "
        "against an amplified slope that agrees to within the timestamp quantum. Sub-linear in pixels "
        "(x1.9 for x5.1 the pixels) because a fixed ~0.020 ms of twelve-render-pass setup dominates at "
        "these sizes. Snow, sand, rubber, the physics and the layout are unchanged and were re-measured "
        "to confirm it: the layout re-shoots byte-for-byte identical at all five viewports.\n"
        "4. LAYOUT. Measured from the live layout at five viewports, not from CSS. The field was "
        "42%% non-square on an iPhone in portrait and 141 px across in landscape; it is now square at "
        "all five and 390 px in landscape. Smallest control 29 px -> 40 px. Laptop and desktop "
        "unchanged."
    ) % (T3["snow"]["webgpu"]["rest_depth_change"], T3["snow"]["webgpu"]["submerged_fraction"],
         T3["elastic"]["webgpu"]["rest_depth_change"], T3["sand"]["webgpu"]["rest_depth_change"],
         B["render_gpu_ms_1024_before"], B["render_gpu_ms_1024"],
         W48["chain_gpu_ms_direct"], W72["chain_gpu_ms_direct"], W108["chain_gpu_ms_direct"]),

    "hypothesis": (
        "HYPOTHESIS, not observation. The reason regenerating params.js was necessary but nowhere near "
        "sufficient is that a generated constant only changes behaviour if a kernel reads it, and the "
        "three quantities that moved each needed a different mechanism to reach the solver: density had "
        "to become a per-particle mass inside P2G, the Poisson ratio had to be folded into per-material "
        "Lame constants before emission, and friction had to be scattered to the grid because a node at "
        "an interface holds two materials at once and has no id to branch on. The general form, which "
        "this task tests on exactly one port, is that porting a parameter is a data change but porting a "
        "PER-MATERIAL parameter is a structural change, and the structure differs by where in the "
        "transfer the parameter is consumed.\n\n"
        "On the rendering side the hypothesis is that shading is nearly free and structure is not: "
        "branching four ways inside an existing full-screen pass is arithmetic on values already in "
        "registers, while the one treatment that added geometry (sand's grains) is the one that showed "
        "up in the timing, and showed up as a resolution-independent constant. That is consistent with "
        "the two measured resolutions here but is a claim about one renderer on one GPU.\n\n"
        "The water rework adds a sharper version of the same hypothesis, and this one has a piece of "
        "evidence behind it: a material's identity lives in the RECONSTRUCTION, not in the shading, so "
        "porting a lighting model without the fields it reads reproduces the old look with new "
        "arithmetic. What makes it more than a slogan here is that the failure was silent in every way "
        "a code review can check -- the WGSL compiled, the constants matched the proposal, the shading "
        "terms were all present, and the result was reported as done. The only thing that would have "
        "caught it is putting the rendered frame next to the proposal's frame and looking. Scope: one "
        "material, one renderer; the three treatments whose look IS mostly shading (snow, sand, rubber) "
        "transferred correctly the first time, which is consistent with the hypothesis and is also why "
        "the failure was easy to miss.\n\n"
        "WOULD TEST: run the same four treatments at 2160^2 and at 65,536 particles, where the "
        "fill-bound and geometry-bound terms should cross over; and check the buoyancy ordering at a "
        "second particle density and blob radius, since the current result is one scene."
    ),

    "limitations": (
        "SCOPE. Every timing is one GPU (nvidia lovelace / RTX 4090 class), one browser (Chromium, "
        "WebGPU), one scene. Repeat runs of the identical solver build differed by up to ~15% at 8,192 "
        "particles, so differences under that are noise.\n\n"
        "PHYSICS. The buoyancy result is ONE scene at one particle density (500 particles in a disc of "
        "radius 0.075) with one blob radius and one pool depth. It shows that density now reaches the "
        "grid and produces canonical's ordering; it does not show that the page has a calibrated "
        "multi-phase contact model, and it does not. A node holding two materials gets the "
        "mass-weighted blend, which is a mixture rule, not contact. The pool_fluid control's agreement "
        "ratio of 5.25x is the widest and is an artefact of a tiny denominator, not a disagreement worth "
        "5.25 of anything (absolute traj_rmse 0.000045 domain lengths).\n\n"
        "WATER. The reconstruction is a port of T-020's structure, not of its code: the blur is 13 "
        "bilinear taps rather than a full separable kernel, the morphological grey-close is dropped "
        "(the wider blur seals the pinholes it existed for), the distance field runs at HALF resolution "
        "and is upsampled bilinearly, and the foam's motion gate reads the GRID velocity under the "
        "pixel instead of a mass-weighted splat of particle speed (the grid buffer was already bound "
        "to the fragment stage, so that cue is free; a separate speed splat would have cost an extra "
        "render target on the heaviest pass). Each of those is a place the image could differ from "
        "T-020's, and 'looks like the proposal' is judged from the three-way still and from the "
        "sampled interior colour, not proven. The tone curve is applied to WATER ONLY, because water "
        "is the only one of the four whose colour is a computed radiance rather than a palette tint -- "
        "that is a deliberate inconsistency and it is the reason the water sits in a different value "
        "range from the other three. Every timing is at ONE particle count and ONE scene; the chain's "
        "cost is pixel-bound and should be independent of particle count, but that was not tested.\n\n"
        "A CONSEQUENCE WORTH KNOWING BEFORE ACCEPTING THIS. Option B makes SHALLOW water nearly "
        "invisible, and that is not a bug in the port, it is what the treatment is: transmission goes "
        "as exp(-absorb*t), so a thin sheet transmits almost everything and the demo's background is "
        "nearly black. Found by driving the real page: pouring water into an empty scene gives a "
        "sheet that reads about (16,36,44) against a (6,9,13) background until enough depth "
        "accumulates. The default pool (0.155 of the domain deep) is well clear of this and reads "
        "strongly; a user's own small pour does not, until it pools. The previously shipped water had "
        "the opposite failing -- it was equally bright at every depth, which is why it had no depth "
        "cue at all. If the shallow case matters more than the deep one, the knob is `absorb` (0.52) "
        "or a small ambient floor, and neither was touched here because neither is T-020's.\n\n"
        "RENDERING. Interfaces take the DOMINANT material's treatment, so a water/sand boundary is a "
        "one-pixel switch of shading model rather than a blend. Not visible at this splat radius; it "
        "would be at a larger one. 'Looks better' is a judgement and is not measured anywhere here; what "
        "was measured is frame cost. Snow, sand and rubber are WGSL reinterpretations of T-020's Taichi "
        "proposals, matched to their published descriptions rather than ported line for line, so they "
        "are the chosen treatments in character and not pixel-identical to T-020's images. Water is now "
        "the exception: its palette, absorption, Fresnel, rim, specular and tone curve are T-020's "
        "constants, and its reconstruction is T-020's structure, with the deviations listed above.\n\n"
        "LAYOUT. NO PHYSICAL DEVICE WAS TESTED. The five viewports are device-metric overrides in a "
        "desktop Chromium (390x844, 844x390, 820x1180, 1280x800, 1920x1080), which establishes what "
        "fits and what the layout does, and says nothing about a real phone's touch handling, its "
        "thermal behaviour, or Safari. Touch targets were measured (40 px) but not tapped with a finger.\n\n"
        "NOT DONE. The shared timestep is untouched: any scene containing snow or water runs at "
        "dt = 5e-5 and 333 substeps per frame, so a phone will show labelled slow motion rather than "
        "60 fps. Scene geometry (drop heights, blob radii, pool depth) is still specified per task and "
        "not centralised, which is a known project-wide gap."
    ),

    "full_report": (
        "## What changed in the solver\n\n"
        "`params.js` is regenerated by this run's `web/gen_params.py` straight from `sim.physics`, and now "
        "emits `rho`, `nu` and `fric` per material, with `mu` and `la` computed from each material's OWN "
        "`nu` rather than the module-level default. Emitting them is where the previous version of this "
        "work would have stopped, and it would have changed nothing, because a constant no kernel reads "
        "does not exist. Three separate mechanisms carry them into the WGSL step:\n\n"
        "- **Density.** `pm = PR.pVol * matRho(mid)` inside P2G, with `matRho` a compile-time constant "
        "selected by the material id already packed into `vel.w`. The mass, the momentum and the affine "
        "term all use it. Nothing adds an upward force to anything.\n"
        "- **Poisson ratio.** Folded into the emitted Lame constants. No kernel sees a Poisson ratio.\n"
        "- **Friction.** Canonical scatters a mass-weighted friction to the grid so a node shared by two "
        "materials gets the friction of what is sitting on it. That is a fourth accumulator, and a fourth "
        "storage buffer would be the eighth of eight. Instead `gm` was WIDENED to two u32 per cell: "
        "`gm[2i]` is the fixed-point mass and `gm[2i+1]` the fixed-point mass*friction, and the node "
        "coefficient is their ratio. Same bind group, still seven storage buffers.\n\n"
        "One consequence had to be re-measured rather than inherited. The fixed-point scale used to be in "
        "units of 'one particle mass', which stops being a single number once density is per-material, so "
        "it is now in units of a fixed REFERENCE mass (rho = 1). At kM = 24 the u32 mass accumulator "
        "saturates at 256 reference masses on one node and WRAPS SILENTLY, so the heaviest material "
        "(sand, rho 1.6) reaches the ceiling at 160 of its own particles. Measured under deliberate "
        "piling with the interaction force crushing everything into one corner: headroom x4.60 at 8,192 "
        "particles and x2.81 at 16,384, against x5.6 at 12,288 before density existed. Still safe, "
        "with less margin.\n\n"
        "All four boundaries now separate in the normal direction and Coulomb-limit the tangent, matching "
        "canonical. The side walls previously zeroed both components, which glued material to them.\n\n"
        "## Verification\n\n"
        "Six scenes, each run three times on canonical (reference, repeat, and a 1e-7 nudge of the "
        "initial positions) to establish a self-noise band, and once on the demo's own WGSL solver from "
        "bit-identical float32 initial conditions with an identical substep schedule. Agreement as a "
        "ratio to that band: pool_snow 1.08, pool_elastic 2.84, pool_sand 1.08, pool_fluid 5.25, "
        "pool_three 1.44, mixed4 0.81. Absolute `traj_rmse` never exceeds 0.00091 domain lengths.\n\n"
        "The pass condition was behavioural, not numerical, and it passes: on the demo's solver snow "
        "rises and ends 25.4% submerged while rubber and sand are fully submerged, with final mean "
        "heights ordered snow > rubber > sand.\n\n"
        "## What changed in the renderer\n\n"
        "The accumulation target's four channels now carry PER-MATERIAL weight instead of premultiplied "
        "colour and total weight. Same texture, same rgba16float format, same fill, same four neighbour "
        "taps, and strictly more information: total weight is the channel sum, the colour is recoverable "
        "as the weighted mean of the frozen palette, and the material identity of a pixel is the argmax, "
        "which the old layout threw away. One resolve pass then branches to the right treatment with no "
        "second render target and no second pass.\n\n"
        "- **Water: T-020 option B, 'film'.** Chosen over option A ('glass') because the demo's "
        "background is a flat dark gradient, so option A's background sampling and three-tap chromatic "
        "dispersion would be paying to refract an almost constant colour. B was also the cheaper of the "
        "two in T-020's measurement. This is the treatment that took two attempts -- see 'The water "
        "rework' below.\n"
        "- **Snow: option A, 'powder'.** The specular is gone entirely (the glint is what read as wet "
        "plastic), replaced with wrap lighting, thin snow brightened, crevices darkened by the Laplacian "
        "of the weight field, a fine hashed crystal grain, and a translucent bright fringe.\n"
        "- **Sand: option A, 'grains over a packed body'** (not B, 'loose grains'). A matte, "
        "contact-occluded body from the resolve, plus one pass of six hashed grains per particle drawn as "
        "n*6 instances with per-grain offset, radius and brightness. T-020 resolved grain overlap with an "
        "atomic max on a random priority; a raster pass has no such primitive here, so overlap is "
        "resolved by near-opaque per-grain alpha, which produces the same speckle and is an "
        "approximation.\n"
        "- **Rubber: NEITHER new option.** The shipped treatment, with exactly two changes and only for "
        "rubber: the splat kernel is 0.78x the radius (with the weight scaled by 1/0.78^2 so one global "
        "iso still means the same thing for all four materials), which makes particles merge into one "
        "continuous blob far less readily; and a dark border band in the outer shell of the silhouette.\n\n"
        "`draw({treatment: 'mvp'})` selects the previously shipped single treatment on the same build. "
        "That exists so a before/after can change one thing at a time, and it is why the buoyancy clip "
        "is shaded identically on both sides and the rendering clip has identical particle positions on "
        "both sides.\n\n"
        "## Cost\n\n"
        "All device time, from `timestamp-query` across both render passes, at 16,384 particles with all "
        "four materials present, measured at TWO resolutions on purpose. blob: 0.034 -> 0.087 ms at "
        "512^2 and 0.097 -> 0.147 ms at 1024^2. The resolution-dependent part of the cost went 0.063 -> "
        "0.060 ms, i.e. did not grow, while a fixed +0.05 ms appeared that does not scale with pixels. "
        "That constant is the sand pass. The `pts` view reads 0.011 ms at both resolutions, which has the "
        "shape of the wall-clock trap but is the correct answer here: it draws 16,384 tiny hard squares "
        "and is geometry-bound. The two fill-bound views do move with resolution, which is what says the "
        "instrument works.\n\n"
        "Solver: 6.91 ms at 16,384 particles and 333 substeps (999 dispatches). Total GPU work 7.06 ms "
        "of a 16.67 ms frame. Drawing is 0.9% of the budget and the solver is 41%.\n\n"
        "## The water rework\n\n"
        "This part of the task shipped wrong and was sent back. The correction is worth recording "
        "precisely, because the failure mode is invisible to code review.\n\n"
        "**What was ported the first time.** T-020's `shade_water` for option B, essentially line for "
        "line: Beer-Lambert absorption on the optical thickness, the shallow-to-deep palette blend, a "
        "Fresnel-weighted sky, a grazing rim, a tight Blinn-Phong glint, foam gated to the fast and "
        "thin surface band. All of it compiled, all of it ran, and the water looked exactly like the "
        "water it was supposed to replace.\n\n"
        "**Why that was not enough.** Every input to that shading is a FIELD, and the fields came from "
        "the wrong place. `th` was `a / iso` -- the local accumulated splat weight -- and `nrm` was the "
        "gradient of the same `a` over four neighbour taps. A splat accumulation is a sum of a few "
        "thousand overlapping compact kernels dropped at a Poisson sample of positions, so it is lumpy "
        "at the particle spacing by construction. Beer-Lambert on a lumpy thickness gives a lumpy "
        "colour; a specular on a lumpy normal gives thousands of little highlights. That is the "
        "'smoothie'. **The shading was never the treatment.**\n\n"
        "**What actually makes T-020's water look like water** is the separation in `build_masks`: one "
        "threshold cannot both decide *is there water here* (which wants a generous cut and pinholes "
        "sealed) and *which way does the surface face* (which wants fine slope), so the filled body "
        "owns opacity and thickness while a wide band around its iso-surface owns the normal. Density "
        "noise then reaches neither. Optical thickness comes from a **distance transform** of the "
        "binary body: distance to the nearest non-water pixel. A distance field cannot carry "
        "particle-scale noise because it does not know where the particles are, only where the surface "
        "is.\n\n"
        "**The WGSL chain**, eleven passes between the existing splat and the existing resolve, all at "
        "half resolution:\n\n"
        "1. **Separable Gaussian, 2 passes.** The horizontal pass reads the full-resolution "
        "accumulation and writes half resolution, so the downsample is free. Its sigma is derived, not "
        "guessed: T-020 blurs a POINT histogram to sigma 6.5 px at 720 (0.00903 of the frame), the "
        "demo's disc splat is already worth sigma 0.354*rpx, so only the remainder in quadrature is "
        "added.\n"
        "2. **Threshold and seed, 1 pass.** The cut is 0.24 of FULL PACKING, and full packing is "
        "computed by the host from the particle density and the splat radius -- a particle deposits "
        "(pi/3)*rpx^2 of weight, a packed region holds 1/pVol particles per unit area. A fixed physical "
        "reference rather than a per-frame percentile is what keeps a thin sheet of spray reading as "
        "thin. Every pixel outside the body seeds itself with its own coordinate.\n"
        "3. **Jump flood, 6-7 passes.** Doubling steps from a start chosen so the flood's reach covers "
        "the depth at which the absorption saturates; anything past that is clamped anyway.\n"
        "4. **Seeds to distance, 1 pass**, with a 3x3 box, because a distance field off a thresholded "
        "mask is quantised in whole pixels and Beer-Lambert turns quantisation into banding.\n\n"
        "Three implementation details were forced by the platform rather than chosen. **Per-pass "
        "arguments** (blur direction, flood step) go through one uniform buffer bound at a DYNAMIC "
        "OFFSET, because a `queue.writeBuffer` between two `beginRenderPass` calls is ordered on the "
        "queue and would apply to every pass in the submission. **The distance field is rg16float**: "
        "f16 is exact on integers to 2048 so a seed coordinate survives it, and the 16-bit float "
        "formats are filterable where 32-bit float is not, which the resolve needs to bilinearly "
        "upsample. **The resolve became premultiplied.** Water transmits, and transmission has to be "
        "independent of how much colour the water adds; under premultiplied alpha `1 - alpha` carries "
        "exp(-absorb*t) exactly, so the water is see-through without the resolve ever sampling what is "
        "behind it. Snow, sand and rubber are bit-identical under the change -- `col*alp` with "
        "srcFactor `one` is the same arithmetic as `col` with srcFactor `src-alpha`.\n\n"
        "One thing was added that is not in T-020's structure and one was left out. Added: T-020's "
        "**tone curve** is applied to the water and only to the water. T-020's pipeline tonemaps the "
        "whole frame; the demo does not, and writing a Beer-Lambert radiance straight to an 8-bit "
        "non-sRGB swapchain is what produced a near-black pool on the first build of this chain. Water "
        "is the only one of the four whose colour is a computed radiance rather than a palette tint, so "
        "it is the only one that needs a transfer. Left out: the **speed splat**. T-020 gates foam on a "
        "mass-weighted splat of particle speed; that would need a second render target on the heaviest "
        "pass in the frame, and the grid velocity buffer is already bound to the fragment stage for the "
        "grid view, so the gate reads `gv` under the pixel instead. It was checked rather than assumed "
        "-- 237 whitewater pixels at the splash peak against 109 before.\n\n"
        "**Where it landed.** The pool's interior at mid-depth reads (51, 90, 109) against T-020's "
        "(50, 90, 112) on the equivalent depth of its own render. Nothing was fitted to that; it falls "
        "out of using T-020's palette, absorption coefficient and tone curve.\n\n"
        "**Cost**, from `timestamp-query` across the whole blob draw at 16,384 particles, `before` "
        "being this task exactly as it originally shipped and `after` the same build with the chain on "
        "-- same run, same instrument, same frame:\n\n"
        + WTAB +
        "It is priced twice because once was not trustworthy. **Chromium quantises "
        "`timestamp-query`**, and with `--disable-dawn-features=timestamp_quantization` the residual "
        "granularity was still 16-33 us, which is the same size as the thing being measured. So the "
        "chain is measured as a difference against a matched control AND as the slope of running it K "
        "times inside one timed region; the two agree to within the quantum. A first pass at this, "
        "before the flag went on, returned exact multiples of 32,768 ns for everything -- that number "
        "is the quantum, not a cost, and it is now written into the `render_gpu_ms` registry entry so "
        "the next task does not rediscover it.\n\n"
        "The cost is **sub-linear in pixels** (x1.9 for x5.1 the pixels). That is the honest shape of "
        "twelve render passes: a fixed ~0.020 ms of attachment setup dominates at demo resolutions and "
        "~0.025 ms per megapixel is the marginal cost. It does move with resolution, which is what says "
        "the instrument is reading the GPU rather than the clock. Whole frame: "
        + WFRAME + "\n\n"
        "## Layout\n\n"
        "The failure was a distortion, not crowding. `.frame` was `aspect-ratio:1/1; height:100%; "
        "max-width:100%`, and when max-width binds, the explicit height wins and the aspect ratio does "
        "not shrink it. On a viewport taller than it is wide the frame therefore became a rectangle, and "
        "since the canvas backing store is square and the domain is the unit square, the simulation was "
        "stretched rather than cropped. It is now `width: min(100%, 100cqh)` with `height: auto` on a "
        "size container, which picks the smaller of the stage's two dimensions.\n\n"
        "Three breakpoints beyond that, in the order the controls give up space: notes, then group "
        "labels, then HUD chips, then padding. A short-and-wide viewport (max-height 620px, landscape) "
        "moves the control bar to a right-hand COLUMN, because stacking a bar under a square field takes "
        "the space off the field's width as well as its height. Buttons get a 38-40 px min-height below "
        "720 px, and the root gets `touch-action: manipulation` and `overscroll-behavior: contain` so a "
        "vertical pour drag is not claimed by the page scroller.\n\n"
        "Measured before -> after: iPhone portrait field 392x676 (42% stretched) -> 390x390 square; "
        "iPhone landscape 141x141 -> 390x390; iPad portrait 822x930 (12% stretched) -> 820x820; small "
        "laptop 660x660 -> 658x658 (unchanged, the 2 px is the border moving inside the box); desktop "
        "940x940 -> 938x938. Smallest control 29 px -> 40 px on a phone. No horizontal overflow at any "
        "size, before or after.\n\n"
        "## Contract\n\n"
        "`DemoView.jsx` imports only React and its own `mpm/` bundle. The bundle imports nothing. The "
        "standalone `web/demo.html` is the source of truth and the dashboard copy is generated from it by "
        "`web/sync_to_dashboard.py`. `sim/physics/` was not touched."
    ),

    "results": results,
    "custom_html": (RUN / "bespoke_page.html").read_text(encoding="utf-8"),
    "training_refs": [
        {"id": "scattering-material-properties",
         "title": "Scattering a material property to a shared grid",
         "file": "reports/training/core/18-scattering-material-properties.md",
         "why": "New page. Why a grid node cannot own a material, the mass-weighted scatter-and-divide "
                "pattern that follows, and what it costs to carry that into a fixed-point GPU port with "
                "no spare storage buffers."},
        {"id": "material-appearance",
         "title": "Four materials, one shader (extended)",
         "file": "reports/training/core/17-material-appearance.md",
         "why": "Extended, not duplicated. Adds the WGSL re-measurement of the same treatments (shading "
                "is nearly free, structure is not), the four-channel accumulation trick that lets one "
                "resolve pass shade four materials, and -- from this rework -- the section on why "
                "porting a shading model without its reconstruction reproduces the old look, and what "
                "the reconstruction costs when it is done as render passes in a browser."},
    ],
    "metrics_used": ["rest_depth", "submerged_fraction", "traj_rmse", "self_noise", "render_gpu_ms",
                     "frame_ms", "us_per_substep", "node_mass_headroom", "substeps_per_frame",
                     "field_aspect_error", "field_viewport_share"],
    "device": M["device"],
    "code": [
        REL + "web/gen_params.py",
        REL + "web/mpm4-webgpu.js",
        REL + "web/demo4.js",
        REL + "web/demo4.css",
        REL + "web/sync_to_dashboard.py",
        REL + "verify/prepare.py",
        REL + "verify/harness.html",
        REL + "verify/score.py",
        REL + "verify/cap.html",
        REL + "verify/shots.py",
        REL + "verify/water.html",
        REL + "verify/water.py",
        REL + "verify/assemble_water.py",
        REL + "verify/merge_water_metrics.py",
        "harness/dashboard/src/components/DemoView.jsx",
    ],
}

(RUN / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("wrote manifest.json;", len(results), "media, all present")
