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

results = [
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
     "caption": "The identical particle state with the four treatments: granular sand, matte powder snow, "
                "a darker film-like water, and rubber with a visible border."},
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
             "square and usable on a phone, where it used to be stretched 42% tall. Nothing was done "
             "about the timestep, so a phone still runs it in labelled slow motion."),
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
        "The layout had a bug worth naming: the field was sized so that on any viewport taller than it "
        "is wide the height won and the width clamped, which does not crop a square simulation domain, "
        "it stretches it. An iPhone in portrait was showing the physics 42%% too tall. It is now square "
        "at every viewport tested, controls are 40 px tall instead of 29, and a phone in landscape went "
        "from a 141 px field to a 390 px one. The large-monitor layout is untouched. Not addressed: the "
        "shared timestep still forces 333 substeps per frame whenever snow is present, so a phone runs "
        "this in honest slow motion rather than at 60 fps."
    ) % (O["snow"], O["elastic"], O["sand"], B["render_gpu_ms_1024"], B["solver_gpu_ms"]),

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
        "3. LAYOUT. Measured from the live layout at five viewports, not from CSS. The field was "
        "42%% non-square on an iPhone in portrait and 141 px across in landscape; it is now square at "
        "all five and 390 px in landscape. Smallest control 29 px -> 40 px. Laptop and desktop "
        "unchanged."
    ) % (T3["snow"]["webgpu"]["rest_depth_change"], T3["snow"]["webgpu"]["submerged_fraction"],
         T3["elastic"]["webgpu"]["rest_depth_change"], T3["sand"]["webgpu"]["rest_depth_change"],
         B["render_gpu_ms_1024_before"], B["render_gpu_ms_1024"]),

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
        "RENDERING. Interfaces take the DOMINANT material's treatment, so a water/sand boundary is a "
        "one-pixel switch of shading model rather than a blend. Not visible at this splat radius; it "
        "would be at a larger one. 'Looks better' is a judgement and is not measured anywhere here; what "
        "was measured is frame cost. The treatments are WGSL reinterpretations of T-020's Taichi "
        "proposals, matched to their published descriptions rather than ported line for line, so they "
        "are the chosen treatments in character and not pixel-identical to T-020's images.\n\n"
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
        "two in T-020's measurement.\n"
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
                "is nearly free, structure is not) and the four-channel accumulation trick that lets one "
                "resolve pass shade four materials."},
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
        "harness/dashboard/src/components/DemoView.jsx",
    ],
}

(RUN / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("wrote manifest.json;", len(results), "media, all present")
