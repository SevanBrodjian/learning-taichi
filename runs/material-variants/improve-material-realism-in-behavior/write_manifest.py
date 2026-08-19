"""Write the task manifest. Run LAST -- every media src it lists must already exist on disk."""
import json, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "runs/material-variants/improve-material-realism-in-behavior"

db = json.load(open(os.path.join(HERE, "diag_before.json")))
da = json.load(open(os.path.join(HERE, "diag_after.json")))
vb = json.load(open(os.path.join(HERE, "volcurve_before.json")))
va = json.load(open(os.path.join(HERE, "volcurve_after.json")))
ab = json.load(open(os.path.join(HERE, "ablation.json")))
bu = json.load(open(os.path.join(HERE, "buoyancy_after.json")))
st = json.load(open(os.path.join(HERE, "stability.json")))
wf = json.load(open(os.path.join(HERE, "wall_film.json")))
sig = open(os.path.join(HERE, "signatures_output.txt"), encoding="utf-8").read()
n_pass = sig.count("[PASS]")
n_fail = sig.count("[FAIL]")


def q(d, job, key="mean", fn=min):
    return fn(d["curves"][job][key])


MAT_ROWS = []
for m in ("fluid", "elastic", "snow", "sand"):
    o, n = db["MAT"][m], da["MAT"][m]
    MAT_ROWS.append([
        m,
        f"{o['E']:g}  ->  {n['E']:g}",
        f"1.0  ->  {n['rho']:g}",
        f"{db['globals']['NU']:g}  ->  {n['nu']:g}",
        f"{db['globals']['FRICTION']:g}  ->  {n['fric']:g}",
        f"{o['dt']:g}  ->  {n['dt']:g}",
        f"{o['E'] / 1.0:g}  ->  {n['E'] / n['rho']:g}",
    ])

HEAD = [
    ["what was measured", "old physics", "new physics"],
    ["rubber, body area at peak of a hard floor impact",
     f"{100 * q(vb, 'elastic/slam'):.1f}% of its own area", f"{100 * q(va, 'elastic/slam'):.1f}%"],
    ["rubber, most-crushed particle (det F) in that impact",
     f"{q(vb, 'elastic/slam', 'min'):.3f}", f"{q(va, 'elastic/slam', 'min'):.3f}"],
    ["rubber, body area at peak of a plain drop from rest",
     f"{100 * q(vb, 'elastic/drop'):.1f}%", f"{100 * q(va, 'elastic/drop'):.1f}%"],
    ["water, most-squashed particle (J) on the drop",
     f"{q(vb, 'fluid/drop', 'min'):.3f}", f"{q(va, 'fluid/drop', 'min'):.3f}"],
    ["water, 1st-percentile J on the drop (the crushed tail)",
     f"{q(vb, 'fluid/drop', 'p01'):.3f}", f"{q(va, 'fluid/drop', 'p01'):.3f}"],
    ["water, dam-break front speed while running free",
     f"{ab['water']['revert both']['front_speed']:.2f} dom/s",
     f"{ab['water']['new (canonical)']['front_speed']:.2f} dom/s"],
    ["snow, settled slope on the over-steep heap",
     f"{db['runs']['heap/snow']['repose_angle']:.1f} deg",
     f"{da['runs']['heap/snow']['repose_angle']:.1f} deg"],
    ["sand, angle of repose on the over-steep heap",
     f"{db['runs']['heap/sand']['repose_angle']:.1f} deg",
     f"{da['runs']['heap/sand']['repose_angle']:.1f} deg"],
    ["snow / sand / rubber resting depth in a pool of water",
     "no density exists; nothing sinks or floats",
     f"{bu['runs']['mat_snow']['rest_depth_final']:+.3f} / "
     f"{bu['runs']['mat_sand']['rest_depth_final']:+.3f} / "
     f"{bu['runs']['mat_elastic']['rest_depth_final']:+.3f}"],
]

M = {
    "schema_version": "2",
    "task_id": "improve-material-realism-in-behavior",
    "direction": "material-variants",
    "title": "Making the four materials behave like the things they are named after",
    "tldr": ("Water particles could be squashed to half their volume and a rubber blob really did lose "
             "a tenth of its area every time it hit the floor; one stiffness and one Poisson ratio fix "
             "both, per-material density now makes sand and rubber sink while snow floats with no "
             "buoyancy force anywhere, and snow and sand are provably untouched -- but only half the "
             "stickiness complaint improved, because the same change that stops water dragging on the "
             "floor also lets it slide further UP the side walls, and 'rubber breaks too easily' could "
             "not be reproduced at all."),
    "status": "active",
    "created": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "physics_version": da["physics_version"],
    "physics_version_before": db["physics_version"],

    "objective": (
        "Four complaints about the canonical materials, treated as symptoms to diagnose rather than "
        "instructions to follow: water looks mushy and sticky, rubber compresses far too much and "
        "breaks too easily, snow and sand are already right and must not move, and nothing sinks or "
        "floats because there is no density. This is one of the rare tasks licensed to edit "
        "sim/physics/ itself, so the work had to pass the three promotion gates -- it is ground truth, "
        "every existing golden signature stays green and new behaviour brings new signatures, and the "
        "version hash bumps."),

    "summary": (
        "Both of the specific complaints turned out to be a single named parameter each, and both were "
        "measurable before anything was touched. The elastic solid shared one global Poisson ratio "
        "nu = 0.2 with every other solid, which is very compressible, so a blob thrown at the floor "
        "genuinely occupied "
        f"{100 * q(vb, 'elastic/slam'):.0f}% of its own area at the moment of impact with its worst "
        f"particle crushed to {q(vb, 'elastic/slam', 'min'):.2f} of rest volume -- the blob really did "
        "come back smaller, and that state is what the eye sees. Giving rubber its own nu = 0.45 takes "
        f"the same numbers to {100 * q(va, 'elastic/slam'):.0f}% and "
        f"{q(va, 'elastic/slam', 'min'):.2f}. Water's mushiness was the weak-compressibility stiffness: "
        f"at E/rho = 180 a water particle on the drop scene reached J = "
        f"{q(vb, 'fluid/drop', 'min'):.2f}, half its volume, which is a gel and not a liquid; at 900 "
        f"the same particle bottoms out at {q(va, 'fluid/drop', 'min'):.2f}. Water's stickiness was "
        "two separate things, floor friction shared with the granular materials and side walls that "
        "zeroed BOTH velocity components (glue, not a wall); water now carries fric = 0 and every "
        "boundary is separating with Coulomb friction on the tangent, which makes its dam-break front "
        f"run {100 * (ab['water']['new (canonical)']['front_speed'] / ab['water']['revert both']['front_speed'] - 1):.0f}% "
        "faster. Density is a structural addition rather than a tuning one: every material now carries "
        "rho, particle mass is p_vol*rho, and sinking and floating come out of the mass ratio in the "
        "transfer with no buoyancy force in the code at all. Snow floats with "
        f"{100 * bu['runs']['mat_snow']['submerged_final']:.0f}% of itself under water, against the 30% "
        "a rigid float of that density would show, and rubber and sand both reach the floor."
        "\n\n"
        "Snow and sand did not move, and not by luck. A single material is exactly invariant under "
        "(rho, E) -> (k rho, k E), because the momentum balance only ever contains E/rho, so each "
        "material's density was introduced with its stiffness scaled to match and its solo behaviour is "
        "unchanged to the simulator's own run-to-run noise -- now asserted as a golden signature. What "
        "is NOT fixed: the sticky complaint splits in two and only one half improved. Against the floor "
        "water is genuinely less draggy, and the ablation puts essentially all of that on the friction "
        "change. Against the side WALLS it got worse -- thin sheets ride further up a wall than they "
        f"used to (peak wall film {wf['dam']['before']['peak']:.2f} -> "
        f"{wf['dam']['after']['peak']:.2f} on the dam break) -- and the ablation says the cause is the "
        "same friction change, not the stiffness. Frictionless water slides UP a wall as freely as it "
        "slides along a floor. The 'breaks too easily' half of the rubber complaint was never "
        "reproduced on any scene tried, so there is nothing here that can be claimed to have fixed it."),

    "full_report": (
        "DIAGNOSIS (measured on the old physics, before any change)\n"
        "  * Poisson ratio. Sweeping nu on an elastic disk dropped from 0.78 with a downward kick, "
        "everything else frozen (nu_sweep.jsonl): the most-crushed particle's det(F) at the end of the "
        "roll runs 0.31 (nu=0.2), 0.37 (0.3), 0.51 (0.4), 0.63 (0.45), 0.72 (0.47), and the smallest "
        "footprint the blob reaches runs 0.876, 0.920, 0.930, 0.959, 0.972 of its starting footprint. "
        "The complaint maps onto one parameter with no ambiguity.\n"
        "  * Fluid stiffness. The spread of J across the fluid (99th minus 1st percentile) on the drop "
        "runs 0.30 at E=180, 0.16 at 400, 0.09 at 800. A weakly-compressible fluid's density variation "
        "goes as rho v^2 / E, so this is the stiffness and nothing else.\n"
        "  * Friction and walls. Coulomb friction at the floor only bites while a node is still moving "
        "downward, so it is a weak effect on a spreading sheet; the side walls, which zeroed both "
        "velocity components for any material pushing into them, are the stronger artefact. Both were "
        "shared globals with no per-material value.\n"
        "  * Breakage. Elastic never fragmented on drop, column, heap, slam, or two blobs driven "
        "head-on at 3 domain lengths per second. The complaint could not be reproduced, so it was not "
        "targeted.\n\n"
        "WHAT CHANGED IN sim/physics/core.py\n"
        "  * MAT gains rho, nu and fric per material. E is rescaled with rho for snow, sand and elastic "
        "so E/rho is preserved exactly (snow 150, sand 300 as before); the fluid's E/rho is deliberately "
        "raised 180 -> 900.\n"
        "  * nu is threaded through elastic_stress / snow_stress / sand_stress and the sand return map "
        "as a runtime argument instead of the module constant NU, and through the multi-material path "
        "as m_nu.\n"
        "  * A mass-weighted friction field grid_fr is scattered alongside grid_m in P2G, so a node "
        "shared by two materials gets the friction of whatever is on it. grid_op reads grid_fr/grid_m.\n"
        "  * All four boundaries are separating with Coulomb friction on the tangent. The side walls "
        "previously zeroed both components.\n"
        "  * simulate() gains nu, rho and fric overrides; simulate_multi() takes a per-group rho "
        "override so a buoyancy result can be shown to depend on density and nothing else.\n"
        "  * New canonical scenes: slam (a hard floor impact, where a solid's volumetric response is "
        "actually visible), dam (a one-sided dam break, which measures runout rather than sloshing), "
        "and scene_pool (a solid released at rest, fully submerged, in a pool). New diagnostics: "
        "waterline, submerged_fraction, rest_depth. New seeder: seed_lattice.\n\n"
        "TIMESTEP AND COST\n"
        "  The fluid's dt falls 1.2e-4 -> 5e-5 and elastic's 1.0e-4 -> 5e-5, both because their wave "
        "speeds rose (the fluid's from the stiffness, rubber's from lambda growing as nu -> 1/2). Snow "
        "and sand are unchanged, because their E/rho is unchanged. shared_dt over all four materials "
        "was already 5e-5 (snow's), so a scene containing several materials costs exactly what it did "
        "before; only fluid-only and elastic-only runs get about twice as expensive.\n\n"
        "A SEEDING BUG FOUND ALONG THE WAY\n"
        "  A pool of water seeded by uniform random sampling compacts as it settles: its free surface "
        "fell 25% over 2.2 s while the model's own J stayed at 1.00. The weakly-compressible fluid takes "
        "its pressure from the ADVECTED volume ratio, not from the actual particle packing, so nothing "
        "resists a random pack tightening up. Seeding on a jittered lattice (seed_lattice) starts near "
        "rest density and more than halves the drift. This mattered: with the random pool, the falling "
        "free surface hid the rubber blob sinking entirely.\n\n"
        "DOWNSTREAM STALENESS -- REPORTED, NOT FIXED\n"
        f"  The physics version moved from {db['physics_version']} to {da['physics_version']}. The "
        "shipped WebGPU demo's constants file harness/dashboard/src/components/mpm/params.js is "
        "generated from sim.physics and still stamps the old hash, so it is now stale: it carries the "
        "old E and dt for every material, a single global NU, no rho and no fric, and it will not "
        "reproduce the new materials. Regenerating it means running the demo run's web/gen_params.py "
        "and then web/sync_to_dashboard.py, and the WGSL itself would need the per-material nu, rho and "
        "friction plumbed through. That was deliberately NOT done here. Separately, "
        "runs/material-variants/interactive-simulation-of-one-material/verify/gpu_bench.py calls "
        "core.p2g / core.g2p directly and no longer matches their signatures; it is a completed run's "
        "benchmark script and was left alone. spec/registry/materials.json WAS regenerated with "
        "harness/tools/sync_registry.py, and four metrics (rest_depth, submerged_fraction, "
        "volume_ratio, retained_area) were registered in spec/registry/metrics.json.\n\n"
        f"GOLDEN SIGNATURES: {n_pass} pass, {n_fail} fail. Every one of the 15 pre-existing signatures "
        "stayed green. Nine new ones were added: snow floats / rubber sinks / sand sinks / resting "
        "depth is ordered by density / the same blob orders by density alone, the (rho, E) gauge "
        "invariance for snow, sand and elastic, rubber holds its volume through an impact, water is "
        "nearly incompressible, and friction is per-material."),

    "findings": (
        "Scope first: everything here is measured on five two-dimensional scenes at n_grid = 128 "
        "(the canonical drop, column and over-steep heap, plus a hard slam and a one-sided dam break) "
        "with 7000 particles, and the buoyancy result on one pool scene with one blob size at four "
        "densities. These are single scenes, not a survey.\n\n"
        "OBSERVED. (1) The rubber complaint is the global Poisson ratio. At nu = 0.2 an elastic blob "
        f"slammed into the floor holds {100 * q(vb, 'elastic/slam'):.1f}% of its area at peak with its "
        f"worst particle at det F = {q(vb, 'elastic/slam', 'min'):.3f}; at nu = 0.45 the same scene "
        f"gives {100 * q(va, 'elastic/slam'):.1f}% and {q(va, 'elastic/slam', 'min'):.3f}. Reverting "
        "only nu on the new physics recovers most of the old behaviour, so nu carries about two thirds "
        "of the improvement and the stiffness change the rest. (2) Water's squashiness is the "
        f"weak-compressibility stiffness: worst-particle J on the drop goes "
        f"{q(vb, 'fluid/drop', 'min'):.3f} -> {q(va, 'fluid/drop', 'min'):.3f} and the 1st-percentile "
        f"floor {q(vb, 'fluid/drop', 'p01'):.3f} -> {q(va, 'fluid/drop', 'p01'):.3f}. (3) Water's "
        "runout is the friction, not the stiffness: reverting friction alone drops the dam-break front "
        f"speed from {ab['water']['new (canonical)']['front_speed']:.3f} to "
        f"{ab['water']['revert friction to 0.5']['front_speed']:.3f} domain/s, while reverting the "
        f"stiffness alone leaves it at {ab['water']['revert stiffness to E/rho=180']['front_speed']:.3f}. "
        "For reference the ideal-fluid limit for this dam is "
        f"{ab['ritter_front_speed']:.3f} domain/s, so the new water is still well short of it. "
        "(4) Buoyancy works and follows density alone: the same rubber blob at rho = 0.3, 0.6, 1.0 and "
        "1.6 settles monotonically deeper, and its submerged fraction reads "
        + ", ".join(f"{bu['runs'][k]['submerged_final']:.2f}"
                    for k in ("rho_0.3", "rho_0.6", "rho_1.0", "rho_1.6"))
        + " against the Archimedes prediction of 0.30, 0.60, 1.00, 1.00. (5) Snow and sand are "
        "unchanged. On the drop, column, heap and slam scenes their spread widths agree to three "
        f"decimal places (snow's heap {db['runs']['heap/snow']['spread_width']:.3f} -> "
        f"{da['runs']['heap/snow']['spread_width']:.3f}, sand's heap "
        f"{db['runs']['heap/sand']['spread_width']:.3f} -> "
        f"{da['runs']['heap/sand']['spread_width']:.3f}, sand's angle of repose "
        f"{db['runs']['heap/sand']['repose_angle']:.1f} -> "
        f"{da['runs']['heap/sand']['repose_angle']:.1f} degrees). Two exceptions, both stated rather "
        f"than smoothed over: snow's dam-break spread moved "
        f"{db['runs']['dam/snow']['spread_width']:.3f} -> "
        f"{da['runs']['dam/snow']['spread_width']:.3f}, which is real and is the wall treatment (that "
        "scene starts the material pressed against a side wall); and sand's measured repose angle on "
        f"the slam scene reads {db['runs']['slam/sand']['repose_angle']:.1f} against "
        f"{da['runs']['slam/sand']['repose_angle']:.1f} degrees, which is not a material change -- the "
        "slam splatters sand across the whole domain and a line fit to a nearly flat splat is an "
        "unstable quantity. Its width and height on that scene moved by under 2%.\n\n"
        "NOT REPRODUCED. Elastic never broke apart on any scene tried, including two blobs driven "
        "head-on at 3 domain lengths per second, so the 'breaks too easily' complaint has no measurement "
        "behind it here and nothing was changed for it.\n\n"
        "WENT THE WRONG WAY. Water clings to the side WALLS more than it used to, not less. Measuring "
        "the fraction of water within five cells of a side wall and above the bulk surface -- thin "
        "sheets riding up the wall -- the peak during the splash rises from "
        f"{wf['drop']['before']['peak']:.2f} to {wf['drop']['after']['peak']:.2f} on the drop and "
        f"{wf['dam']['before']['peak']:.2f} to {wf['dam']['after']['peak']:.2f} on the dam break, and "
        f"the amount still hanging there at the end of the dam break rises from "
        f"{wf['dam']['before']['residual']:.4f} to {wf['dam']['after']['residual']:.4f}. The ablation "
        "names the cause, and it is not the one that looks guilty: reverting the stiffness moves the "
        f"peak by about a percent ({wf['_attribution']['drop']['new (canonical)']['peak']:.3f} -> "
        f"{wf['_attribution']['drop']['revert stiffness to E/rho=180']['peak']:.3f} on the drop), while "
        "reverting the friction alone removes most of the increase "
        f"({wf['_attribution']['drop']['revert friction to 0.5']['peak']:.3f} on the drop, "
        f"{wf['_attribution']['dam']['revert friction to 0.5']['peak']:.3f} on the dam). Frictionless "
        "water slides UP a wall exactly as freely as it slides along a floor. Less mechanical drag and "
        "more visual clinging are the same change, and a closed two-dimensional box -- with no third "
        "dimension for a sheet to break up into -- is where it looks worst.\n\n"
        "TIMESTEP MARGIN. Each material's canonical dt was checked against the largest multiple of "
        "itself at which a hard scene still gives a sane rollout, judging on the VELOCITY and the "
        "deformation rather than on positions (core.simulate clamps positions into the domain, so a "
        "diverging run still returns finite in-range x and a check on x alone passes on a material that "
        "has already exploded): "
        + ", ".join(f"{m} {st['runs'][m]['stable_margin']:g}x" for m in
                    ("fluid", "elastic", "snow", "sand"))
        + ". The settled shape still matches the canonical-dt reference to within 15% over the same "
        "range in every case. The two materials whose wave speed rose are the two with a 3x margin; "
        "for elastic the measured wall brackets the CFL estimate dx/c_p = 1.8e-4 between 1.5e-4 and "
        "2e-4, which is the argument confirming itself."),

    "hypothesis": (
        "The mechanism for the volume complaints is that a corotated (or Hencky) solid resists a change "
        "of volume through lambda = E nu / ((1+nu)(1-2nu)), which at nu = 0.2 is only two thirds of the "
        "shear modulus. Impact energy has to go somewhere, and the cheapest place is volumetric strain: "
        "equating kinetic energy density to the volumetric spring gives a peak strain of about "
        "v sqrt(rho/K), which is why the squash is invisible when a blob is set down gently and obvious "
        "when it is thrown. The same argument explains the fluid, where the bulk modulus IS E.\n\n"
        "The mechanism for buoyancy is that gravity is applied to the grid VELOCITY (so it is "
        "mass-blind) while the surrounding fluid's pressure gradient arrives as an IMPULSE that the "
        "grid divides by the node mass. The net acceleration is therefore g (1 - rho_fluid/rho_solid), "
        "which is Archimedes, and it needs no buoyancy term. The density-ladder control is the evidence "
        "for this reading rather than for some property of the elastic model.\n\n"
        "WHAT WOULD TEST THE GENERALITY. The nu claim is a claim about one blob shape at one impact "
        "speed; a sweep over impact speed and blob size would show whether the peak strain really "
        "scales as v sqrt(rho/K) rather than merely improving. The buoyancy claim rests on one blob "
        "radius in one pool depth; the submerged fraction should equal the density ratio for a rigid "
        f"float, and the measured excess "
        f"({bu['runs']['rho_0.3']['submerged_final']:.2f} against 0.30 at rho = 0.3 for the stiff "
        f"rubber blob, and {bu['runs']['rho_0.6']['submerged_final']:.2f} against 0.60 at rho = 0.6) "
        "predicts that the error grows with how much the body deforms -- varying the solid's stiffness "
        "at fixed density would test that directly. The claim that friction owns "
        "the runout is one dam geometry; a slope-angle sweep would say whether the effect is the "
        "Coulomb term or the boundary treatment."),

    "limitations": (
        "* Two dimensions, n_grid = 128, five scenes, one particle count. Nothing here is a survey of "
        "materials or of scene geometry, and scenes are still not centralised across tasks.\n"
        "* 'More realistic' is not measured anywhere. What is measured is retained volume, front speed, "
        "settled slope and resting depth. Whether the result LOOKS better is the reader's call, and the "
        "before/after clips are there so it can be made.\n"
        "* Material-to-material contact is still not modelled. A shared grid gives every material at a "
        "node one velocity, so the interface exchanges momentum as if the node held one blended "
        "substance. Bulk buoyancy survives this because a blob many cells wide is dominated by its "
        "interior, but the interfacial drag is artificial and large: it is why a rubber blob at "
        "rho = 1.2 takes about a second to fall a tenth of the domain.\n"
        "* The submerged fraction is only qualitatively Archimedean. A deformable body spreads and "
        "pushes a bump of water up around itself, so the reading drifts with how much the body has "
        f"deformed and with the measurement window: canonical snow settles at "
        f"{bu['runs']['mat_snow']['submerged_final']:.2f} on the long roll and the rubber blob at the "
        f"same density at {bu['runs']['rho_0.3']['submerged_final']:.2f}, against a prediction of 0.30 "
        "for both. Use the ordering, not the number.\n"
        "* A pool of water still loses free-surface height slowly (about 12% over 2.2 s even with "
        "lattice seeding, against 28% with the random seeding it replaced) because the fluid's pressure "
        "comes from an advected J rather than the actual packing. This is a pre-existing property of the "
        "canonical weakly-compressible fluid, it was measured rather than fixed here, and it has a "
        "visible consequence: a NEUTRALLY buoyant blob does not hang perfectly still, it drifts slowly "
        "upward, because the water around it is compacting while the solid is not. The ordering across "
        "densities is unaffected and is what the result rests on; the neutral case should be read as "
        "'nearly stationary', not as an equilibrium.\n"
        "* 'Breaks too easily' was never reproduced, so it is unaddressed rather than fixed.\n"
        "* The change was NOT propagated to the WebGPU demo, deliberately. The demo's generated "
        "constants are stale and will not reproduce these materials until they are regenerated.\n"
        "* Snow's dam-break spread moved by about 9% because that one scene starts the material against "
        "a side wall whose boundary condition changed. Every other snow and sand measurement is "
        "unchanged to within run-to-run noise, with one metric artifact worth naming: sand's measured "
        "angle of repose on the slam scene jumps, on a deposit that is splattered flat across the whole "
        "domain, where a line fit to the free surface has nothing to grip. Its width and height there "
        "moved by under 2%."),

    "results": [
        {"type": "video", "src": f"{REL}/buoyancy_three.mp4",
         "caption": "The headline. Snow (rho 0.3), rubber (1.2) and sand (1.6) released at rest, fully "
                    "submerged, in identical pools. Snow rises and rides the surface; rubber and sand "
                    "fall to the floor. The readout is the resting depth below the water's own "
                    "surface. No buoyancy force exists in the solver."},
        {"type": "video", "src": f"{REL}/density_ladder.mp4",
         "caption": "The control: ONE material (rubber), one stiffness, one scene, four densities. The "
                    "outcome follows rho and nothing else, which is what makes this buoyancy rather "
                    "than a quirk of the elastic model."},
        {"type": "image", "src": f"{REL}/density_ladder.png",
         "caption": "Left, resting depth against time for the same blob at four densities. Right, the "
                    "settled depth against density, annotated with the fraction of the body under "
                    "water. Archimedes predicts that fraction to be the density ratio."},
        {"type": "image", "src": f"{REL}/buoyancy_three.png",
         "caption": "Settled state of the three canonical solids in a pool, with the waterline marked."},
        {"type": "video", "src": f"{REL}/rubber_slam.mp4",
         "caption": "Rubber, old physics against new, same scene and seed: a hard floor impact. The "
                    "readout under each panel is the footprint the blob still covers as a percentage "
                    "of the area it started with."},
        {"type": "image", "src": f"{REL}/rubber_volume.png",
         "caption": "The quantity the rubber claim rests on: det(F), the model's own volume ratio, "
                    "over time. Solid lines are the body average (its true area), dashed lines the "
                    "first-percentile particle. Left a hard impact, right a plain drop from rest."},
        {"type": "video", "src": f"{REL}/water_dam.mp4",
         "caption": "Water, old against new, on a one-sided dam break. The new front runs faster and "
                    "the deposit does not cling to the far wall in the same way."},
        {"type": "image", "src": f"{REL}/dam_front.png",
         "caption": "Dam-break front position against time, old and new, with the ideal-fluid limit "
                    "2 sqrt(g h0) for reference. Both are well short of it; the new one is 16% closer."},
        {"type": "image", "src": f"{REL}/water_volume.png",
         "caption": "The quantity the mushiness claim rests on: the volume ratio J of the water on the "
                    "drop scene. The old fluid crushed its worst particles to about half volume."},
        {"type": "video", "src": f"{REL}/water_drop.mp4",
         "caption": "Water, old against new, on the canonical drop. This is where the change that went "
                    "the wrong way is visible: the new water slides further up the side walls, because "
                    "frictionless water climbs a wall as freely as it runs along a floor."},
        {"type": "video", "src": f"{REL}/regress_snow_heap.mp4",
         "caption": "Snow on the over-steep heap, old physics against new. Its density went from 1.0 "
                    "to 0.3 and its stiffness fell with it, a change the material provably cannot feel."},
        {"type": "video", "src": f"{REL}/regress_sand_heap.mp4",
         "caption": "Sand on the over-steep heap, old against new. The angle of repose is unchanged."},
        {"type": "video", "src": f"{REL}/regress_snow_drop.mp4",
         "caption": "Snow on the canonical drop, old against new."},
        {"type": "video", "src": f"{REL}/regress_sand_drop.mp4",
         "caption": "Sand on the canonical drop, old against new."},
        {"type": "image", "src": f"{REL}/regression.png",
         "caption": "Every material on every scene, old physics on the x axis and new on the y. On the "
                    "dashed line means unchanged. Snow and sand sit on it; water and rubber are the "
                    "points that leave it, which is the intended change."},
        {"type": "table", "columns": ["what was measured", "old physics", "new physics"],
         "rows": HEAD[1:], "caption": "The headline numbers, old against new."},
        {"type": "table",
         "columns": ["material", "E", "rho", "nu", "floor/wall friction", "dt", "E/rho"],
         "rows": MAT_ROWS,
         "caption": "The frozen parameter table, before and after. E/rho is the only combination a "
                    "material simulated on its own can see, which is why snow and sand did not move."},
        {"type": "table",
         "columns": ["configuration", "wall film peak (drop)", "wall film peak (dam)"],
         "rows": [[k, f"{wf['_attribution']['drop'][k]['peak']:.3f}",
                   f"{wf['_attribution']['dam'][k]['peak']:.3f}"]
                  for k in wf["_attribution"]["drop"]]
                 + [["true OLD physics", f"{wf['drop']['before']['peak']:.3f}",
                     f"{wf['dam']['before']['peak']:.3f}"]],
         "caption": "The change that went the wrong way, attributed. Fraction of water within five "
                    "cells of a side wall and above the bulk surface, at its peak. Reverting the "
                    "stiffness does nothing; reverting the friction removes most of the increase."},
        {"type": "table",
         "columns": ["material", "canonical dt", "bounded up to", "shape still matches up to"],
         "rows": [[m, f"{st['runs'][m]['dt_canonical']:g}",
                   f"{st['runs'][m]['stable_margin']:g}x  ({st['runs'][m]['dt_stable_max']:g})",
                   f"{st['runs'][m]['faithful_margin']:g}x"]
                  for m in ("fluid", "elastic", "snow", "sand")],
         "caption": "Timestep margin on a hard scene, judged on velocity and deformation rather than "
                    "on positions (positions are clamped into the domain, so they stay finite through "
                    "a blow-up). The two materials whose wave speed rose are the two at 3x."},
    ],

    "custom_html": open(os.path.join(HERE, "bespoke_page.html"), encoding="utf-8").read(),
    "training_refs": ["material-stiffness", "math-toolkit"],
}

dst = os.path.join(HERE, "manifest.json")
json.dump(M, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote manifest.json", os.path.getsize(dst) // 1024, "KB")

missing = [r["src"] for r in M["results"]
           if r.get("src") and not os.path.exists(os.path.join(HERE, "..", "..", "..", r["src"]))]
print("MISSING MEDIA:", missing if missing else "none")
