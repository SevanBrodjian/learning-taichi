"""Write manifest.json LAST, from the run's own JSON, and refuse to reference a file that is missing."""
import datetime
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
DIR, TASK = "material-variants", "sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid"
REL = f"runs/{DIR}/{TASK}"

M = json.load(open(os.path.join(D, "metrics.json")))
SW = json.load(open(os.path.join(D, "dt_sweep2.json")))
S1 = json.load(open(os.path.join(D, "dt_sweep.json")))
CR = json.load(open(os.path.join(D, "creep.json")))
FZ = json.load(open(os.path.join(D, "frozen_materials_check.json")))
EQJ = json.load(open(os.path.join(D, "equivalence.json")))
SS = json.load(open(os.path.join(D, "snow_stiffness.json")))
HTML = open(os.path.join(D, "bespoke_page.html"), encoding="utf-8").read()

MATS = ["fluid", "elastic", "snow", "sand"]
LAB = {"fluid": "water", "elastic": "elastic", "snow": "snow", "sand": "sand"}
h = M["heap"]["final"]
eq = EQJ["rows"]
cd = CR["collapse_diagnostic"]
sand_dt = M["materials"]["sand"]["dt"]
sand_spf = round((1 / 60) / sand_dt)
snow_spf = round((1 / 60) / M["materials"]["snow"]["dt"])
hard = S1["snow_hardening"]

TLDR = ("Sand is canonical physics now and it is the cheap one, so snow still sets the demo's frame "
        "budget at 333 substeps per frame; chasing sand's angle of repose is what exposed the failure, "
        "which is that the slope a plastic pile holds decays with the number of substeps taken rather "
        "than converging, and canonical snow's collapses from 56 degrees to 19 when the timestep is "
        "refined fourfold.")

OBJECTIVE = ("Promote sand into the frozen canonical physics with its own golden signature, measure the "
             "timestep it forces, and run all four materials in a single shared grid for the first "
             "time, so the Demo MVP has a fourth material and a real number for what it costs.")

# NOTE ON FORMATTING: the dashboard renders every manifest prose field as PLAIN TEXT inside a single
# <p> with white-space:normal, so markdown is shown literally and newlines collapse. No asterisks, no
# backticks, no headings, no pipe tables, no bullet lists. Emphasis is CAPS, enumeration is (1) (2) (3),
# and every table lives in results[] as type "table" instead.

SUMMARY = (
    "SAND IS IN, AND IT IS NOT THE EXPENSIVE ONE. It entered sim/physics as a Drucker-Prager granular "
    "material -- Hencky log-strain elasticity plus a return mapping onto a yield cone whose width is "
    f"proportional to confining pressure -- with four new golden signatures, all ten pre-existing "
    f"signatures still green, and the frozen materials proved untouched by a distributional test "
    f"(across-version trajectory distances sit at {FZ['worst_ratio_of_means']:.2f}x the within-version "
    f"mean at worst, i.e. inside the simulator's own run-to-run spread). Sand runs at dt = {sand_dt:.0e} "
    f"s, which is {sand_spf} substeps per frame at 60 fps, exactly what elastic costs. SNOW REMAINS THE "
    f"BINDING CONSTRAINT AT {snow_spf}, so adding sand to a scene that already contains snow costs "
    "nothing at all, and the material to design the Demo around is still snow. (Sand alone would take a "
    "water-only scene from 139 to 167.) Four materials now share one grid through a per-particle "
    "material id and a runtime branch, and a single material pushed through that path lands where "
    "canonical simulate lands, below the effect of nudging its initial positions by one float32 "
    "rounding unit. "
    "WHAT DID NOT WORK IS THE ANGLE OF REPOSE ITSELF. Sand settles a 57 degree heap to "
    f"{h['sand']['repose_angle']:.0f} degrees where water runs flat to {h['fluid']['repose_angle']:.0f} "
    "and cohesive snow and elastic keep the whole seeded slope, which is the signature and it holds. But "
    "that number is not converged. Held for 4 s and then refined fourfold in the timestep, sand's "
    f"settled slope goes from {cd['sand']['final_angle_by_dt']['1']:.0f} degrees to "
    f"{cd['sand']['final_angle_by_dt']['0.25']:.0f}, while canonical SNOW'S COLLAPSES FROM "
    f"{cd['snow']['final_angle_by_dt']['1']:.0f} TO {cd['snow']['final_angle_by_dt']['0.25']:.0f} over "
    "the same change. "
    "Plotting the same runs against cumulative substep count instead of physical time collapses snow's "
    f"spread from {cd['snow']['spread_at_equal_time']:.1f} degrees to "
    f"{cd['snow']['spread_at_equal_substeps']:.1f}, and elastic, which carries no plastic projection, is "
    "flat on both axes. So the fine-timestep run is the corrupted one rather than the converged one, and "
    "a plastic material's strength has to be quoted with the timestep and the duration it was measured "
    "at. A standing explanation also failed on the way: snow's small timestep is NOT forced by its "
    f"hardening. The hardening is real and large (compacted snow reaches {SS['E_eff_p95']:.0f} at the "
    "95th percentile against elastic's 400) but snow's measured stability wall is "
    f"{SW['summary']['snow']['dt_stable_max'] / 5e-5:.0f}x its canonical timestep, so that timestep was "
    "never set by stability at all."
)

FULL = (
    "1. SAND AS CANONICAL PHYSICS. "
    "Model choice: Drucker-Prager elastoplasticity (Klar et al. 2016) on a Hencky, i.e. log-strain, "
    "elastic law. The reason is that sand's defining property is cohesionlessness -- its shear strength "
    "comes from friction between grains, so it is proportional to confining pressure and vanishes at a "
    "free surface. That is a CONE in principal stress space, where snow's Stomakhin clamp is a fixed BOX "
    "that does not shrink with pressure, which is exactly why snow can stand a vertical wall and sand "
    "cannot. Log strain is the natural coordinate because the trace of ln(Sigma) is exactly ln(J), so "
    "the volume/shape split of the yield condition is exact and the projection onto the cone has a "
    "closed form. The alternative considered was a mu(I) viscoplastic rheology, which is more faithful "
    "for rapidly flowing sand but is rate-dependent, adds parameters, and would have forced a smaller "
    "timestep for no gain on the pile behaviour this task is judged on. The flow rule is non-dilatant "
    "(the projection is purely deviatoric), so this sand does not expand when sheared. "
    f"Parameters: E = {M['materials']['sand']['E']:.0f}, dt = {sand_dt:.0e}, "
    f"phi = {M['materials']['sand']['phi']:.0f} degrees (alpha = {M['dp_alpha_sand']:.3f}), "
    "render colour #ffd24d. "
    "WHY PHI = 50: the friction angle is a model parameter and NOT the angle of repose; canonical sand "
    f"runs phi = 50 and measures about {h['sand']['repose_angle']:.0f}. It was chosen by sweeping phi "
    "over 4 seeds x 3 pile sizes and taking the largest value still in the reproducible regime. Up to "
    "phi = 50 the measured angle has a standard deviation of 0.3 to 0.4 degrees and is independent of "
    "pile size; at phi = 55 the spread reaches 1.4 and size-dependence appears; by phi = 58 to 60 the "
    "spread is 3.8 to 5.1 and the angle tracks how many grid cells the pile spans, which makes it a "
    "discretisation artifact rather than a material property. Picking the largest angle available "
    "(phi = 60, about 34 degrees) would have looked more like real dry sand and meant nothing. "
    "SIGNATURES: four new ones on a new canonical heap scene (a 60 degree triangle released from rest, "
    "the textbook over-steep-pile relaxation) asserting that sand is stable, that sand holds an angle of "
    "repose where fluid does not, that sand does not spread flat like a fluid, and that sand yields "
    "where cohesive snow and elastic keep the seeded slope. Four more assert that the multi-material "
    "path matches canonical for each material. All fourteen signatures pass. "
    "FROZEN MATERIALS UNTOUCHED, TESTED PROPERLY: a single before/after comparison proves nothing here, "
    "because the reference simulator is nondeterministic (GPU atomic scatter order) and the collapse "
    "scenes are chaotic, so two runs of the SAME code already differ by up to 1.15e-2 on the fluid "
    "column. A first single-sample check duly produced a false alarm on exactly that cell. The real test "
    "restores the pre-promotion physics package out of git, runs "
    f"{FZ['reps_per_side']} repeats of every configuration under each version, and compares all pairwise "
    "distances within a version against all distances across versions (see the frozen-materials table "
    f"below). Every across-code range sits inside its within-code range. physics_version went "
    f"{FZ['version_before']} to {FZ['version_after']}. "

    "2. THE TIMESTEP, WHICH IS THE NUMBER THE DEMO NEEDS. "
    "Sand costs the same as elastic and 20 percent more than water; see the canonical-parameters table "
    "below for every material's dt, substeps per frame, fraction of its own linear CFL limit, and "
    "measured stability wall. SNOW IS STILL THE BINDING MATERIAL. A shared grid runs at min(dt) over the "
    "materials present, so a water scene at 139 substeps per frame becomes 333 the moment one snowball "
    "enters it, and all the water pays too. Sand joins fluid and elastic at 0.21 to 0.26 of its linear "
    "CFL limit (0.222) while snow sits far below at 0.078. "
    "An aside worth recording: the two PLASTIC materials have far higher stability walls than the "
    "elastic and fluid ones (sand is stable to 16x its canonical dt, snow to 8x, against 2 to 3x for "
    "water and elastic). The plausible reason is that a return mapping caps the elastic strain and "
    "therefore caps the stress, while a fluid's pressure E(J-1) and an elastic solid's corotated stress "
    "can both grow without bound. This was not tested as a mechanism. "
    "THE STANDING EXPLANATION FOR SNOW'S DT DOES NOT SURVIVE. The claim on record was that snow's "
    "dt = 5e-5 is forced by hardening h = exp(xi(1-Jp)) making compacted snow about 3x stiffer than "
    "elastic. The premise checks out, and is more interesting than stated: the effective stiffness after "
    "impact is strongly BIMODAL across four decades, because the same law that stiffens compacted snow "
    f"(Jp below 1) softens expanded snow (Jp above 1). About "
    f"{SS['frac_softer_than_nominal'] * 100:.0f} percent of particles end up below snow's own nominal E "
    f"of 150, about {SS['frac_stiffer_than_elastic'] * 100:.0f} percent end up above elastic's 400, the "
    f"stiff lobe peaks near 1000 and the 95th percentile is {SS['E_eff_p95']:.0f}. Compacted snow really "
    f"is by far the stiffest material in the library, and the median ({SS['E_eff_median']:.0f}) "
    "describes neither lobe. The CONCLUSION does not survive, though. Snow's measured stability wall is "
    f"{SW['summary']['snow']['dt_stable_max']:.1e}, which is "
    f"{SW['summary']['snow']['dt_stable_max'] / 5e-5:.0f}x its canonical timestep, so 5e-5 is nowhere "
    "near a stability limit and cannot have been set by one. Turning hardening off entirely (xi = 0) "
    "also leaves the limit unchanged over the range that experiment swept, to 4x canonical, where "
    "neither the hardened nor the unhardened material diverged. Given the creep result below, being "
    "conservative here is not obviously wrong, but the STATED justification is not the real one, and "
    "anything that reasoned from it (including the Demo's frame-budget argument) was reasoning from a "
    "mechanism that is not doing the work. "

    "3. THE FAILURE: A PLASTIC PILE'S SLOPE TRACKS SUBSTEPS, NOT SECONDS. "
    "The dt sweep showed every non-elastic material settling LOWER as the timestep was refined, while "
    "elastic gave an identical settled shape across a 30x range of dt. Two readings were possible: "
    "either the canonical timesteps are too coarse to resolve plastic flow, or the fine runs accumulate "
    "artificial yielding once per substep. These are distinguishable, because the first predicts the "
    "curves collapse against physical time and the second predicts they collapse against substep count. "
    "THEY COLLAPSE AGAINST SUBSTEP COUNT (see the creep table below). Snow is the extreme case: a "
    f"{cd['snow']['spread_at_equal_time']:.0f} degree spread at equal physical time becomes "
    f"{cd['snow']['spread_at_equal_substeps']:.1f} at equal substep count. Sand is affected but only "
    f"about half as much ({cd['sand']['spread_at_equal_time']:.1f} to "
    f"{cd['sand']['spread_at_equal_substeps']:.1f}), so roughly half of sand's dt-dependence is genuine "
    "resolution and half is the artifact. Elastic, the control, is 0.0 on both axes, and fluid shows no "
    "effect either way. "
    "The practical consequence is that CANONICAL SNOW'S COHESION DECAYS WITH THE NUMBER OF SUBSTEPS "
    f"TAKEN. Held at its canonical dt for 4 s (80k substeps) a snow heap keeps its "
    f"{cd['snow']['final_angle_by_dt']['1']:.0f} degree slope; at dt/4 for the same 4 s (320k substeps) "
    f"it has slumped to {cd['snow']['final_angle_by_dt']['0.25']:.0f}. Nothing in this task changes "
    "snow, and short runs at the canonical timestep are in the clean regime, but any long rollout or any "
    "refinement of dt is being handed a progressively weaker material. It also means the dt_faithful_max "
    "numbers from the sweep, which are defined against a dt/8 reference, must not be read as 'the "
    "timestep needed for convergence' for the plastic materials, because that reference is not a fixed "
    "point. "

    "4. FOUR MATERIALS, ONE GRID. "
    "simulate_multi takes a list of material groups, concatenates them, sets a per-particle mat_id, and "
    "branches at runtime in P2G and G2P. Per-material parameters are uploaded from the same frozen MAT "
    "table into small device fields, and per-particle volume and mass are carried per particle because "
    "the groups have different densities. dt defaults to shared_dt(materials), the min over the "
    f"materials present. The clip runs {M['multi_columns']['n_particles']} particles, four blocks "
    f"released from rest, shared dt = {M['multi_columns']['dt_shared']:.1e} set by "
    f"{LAB[M['multi_columns']['binding_material']]}. "
    "PROOF THE REFACTOR CHANGED NOTHING: equality is the wrong bar twice over. The reference is "
    "nondeterministic, and the two paths compile DIFFERENT KERNELS (a compile-time branch against a "
    "runtime one) which can order identical arithmetic differently at the last bit; a chaotic rollout "
    "then amplifies that exponentially. A single comparison against a single self-noise number is "
    "therefore a coin flip, and it behaved like one -- an early single-sample pass put elastic at 2.0x "
    "self-noise and the next at 0.9x, on identical code. The bar used instead is a bracket of "
    "disagreements that provably carry no information: run-to-run nondeterminism at the bottom, and "
    "re-running canonical with initial positions nudged by one float32 rounding unit (1e-7) at the top. "
    f"Over {EQJ['reps_per_path']} repeats of each path, every material's refactor disagreement is at or "
    "BELOW the one-ulp nudge (see the equivalence table below), i.e. smaller than the effect of "
    "perturbing the initial condition by a rounding error. The per-frame curves are the real evidence: "
    "they start at rounding scale (1e-11 to 1e-9), grow exponentially, and saturate into the same "
    "plateau as the no-information band, which is the shape of chaos rather than of bias (a bias appears "
    "immediately and grows linearly). The same four comparisons are also asserted as golden signatures "
    "at a looser threshold, so the refactor cannot silently regress. "
    "Contact between different materials is whatever a shared node velocity produces: two materials "
    "meeting at a node exchange momentum as if the node held one blended material. That is coexistence, "
    "not a calibrated multi-phase contact model, and this task makes no claim about a sand-water "
    "interface."
)

HYPOTHESIS = (
    "WHY THE SETTLED SLOPE TRACKS SUBSTEPS (the main mechanism, hypothesised). The particle-grid round "
    "trip returns a velocity gradient with a small quadrature error, so each substep the trial "
    "deformation carries noise. A plastic return mapping is ONE-SIDED: it can move a state from outside "
    "the admissible set onto the boundary and can never move one back out. Symmetric noise through a "
    "one-way valve becomes a drift, the drift accrues once per projection, and the projection runs once "
    "per substep, so total artificial yielding is proportional to substep count. Elastic stores and "
    "returns the same noise instead of rectifying it, which is exactly why it is the flat control here. "
    "This is the mechanism the data is most consistent with, not one that has been isolated. Two tests "
    "would settle it: sweep grid resolution, since transfer noise should scale with dx and therefore so "
    "should the creep rate; and replace the per-substep return mapping with one applied on a coarser "
    "cadence or solved implicitly, which should remove the substep dependence outright if the mechanism "
    "is right. "
    "WHY SNOW IS AFFECTED ROUGHLY TEN TIMES MORE THAN SAND (hypothesised). Snow's clamp fires on ANY "
    "strain outside a narrow fixed band, including at zero pressure, so noise near the free surface is "
    "rectified everywhere in the pile. Sand's cone is wide wherever the material is confined, so noise "
    "in the interior of a pile is usually inside the admissible set and passes through unrectified; only "
    "the low-pressure surface layer ratchets. That predicts the creep rate should scale with the "
    "fraction of particles sitting at their yield surface, which is directly measurable and was not "
    "measured here. "
    "WHY SAND IS CHEAP WHILE SNOW IS NOT (hypothesised). Sand's plastic projection bounds the elastic "
    "strain and therefore the stress, so there is no runaway mode of the kind a fluid's unbounded "
    "E(J-1) or an elastic solid's corotated stress provides. That is consistent with sand having the "
    "highest measured stability wall of the four (16x canonical), but the mechanism was not isolated. "
    "Note this does not explain snow's small canonical dt, which the xi = 0 experiment shows is not a "
    "stability limit at all; snow's timestep appears to be conservative by choice rather than by "
    "necessity, and re-deriving it is separate work with consequences for every task that used it. "
    "WHY THE MEASURED REPOSE ANGLE IS ABOUT HALF THE FRICTION-ANGLE PARAMETER (hypothesised). The alpha "
    "formula is the 3D Drucker-Prager cone applied in 2D plane strain, and a settled heap's slope is an "
    "emergent property of the yield surface together with the flow rule, the resolution and the collapse "
    "dynamics. Testing this would mean comparing against the plane-strain form of alpha and against a "
    "heap built by pouring rather than by relaxation."
)

LIMITATIONS = (
    "2D, one grid resolution (128x128), one floor friction (0.5), particle counts in the low thousands, "
    "forward simulation only, no gradients. Every claim is scoped to that. "
    "(1) THE ANGLE OF REPOSE IS ONE SCENE FAMILY. All repose numbers come from the heap scene (an "
    "over-steep triangle relaxing) at 3000 to 6000 particles. Column-collapse scenes give systematically "
    "lower apparent slopes, as expected, and are not interchangeable with these numbers. "
    f"(2) THE REPOSE ANGLE IS DT-DEPENDENT AND DURATION-DEPENDENT. {h['sand']['repose_angle']:.0f} "
    f"degrees is sand's value at its canonical dt after {M['heap']['T']} s. It is not a converged "
    "material property and should never be quoted without both. "
    "(3) THE CREEP RESULT IS FOUR TIMESTEPS PER MATERIAL ON ONE SCENE AT ONE RESOLUTION. The "
    f"substep-count collapse is a strong signal (snow {cd['snow']['spread_at_equal_time']:.0f} degrees "
    f"to {cd['snow']['spread_at_equal_substeps']:.1f}) but generality across scenes, resolutions and "
    "materials is a conjecture, not a result, and the proposed mechanism is explicitly a hypothesis. "
    "(4) SAND'S CONSTITUTIVE MODEL IS ONE CHOICE. No alternative granular model (a mu(I) rheology, a "
    "cohesive-frictional variant, a dilatant flow rule) was run against it, so nothing here says "
    "Drucker-Prager is better, only that this one behaves like sand on these tests. "
    "(5) NO MULTI-MATERIAL CONTACT MODEL. Materials sharing a grid node blend their momentum. Whatever "
    "sand does at a water interface in these clips is an artifact of that, not a prediction. "
    "(6) THE STABILITY WALLS COME FROM A DISCRETE MULTIPLICATIVE SWEEP (11 timesteps spanning 1/8x to "
    "16x) on two scenes only, so they are bracketed to within one sweep step. "
    "(7) THE XI = 0 EXPERIMENT SHOWS HARDENING DOES NOT SET SNOW'S STABILITY LIMIT. It does not show "
    "what does, because no divergence was found for snow inside the range that experiment swept. "
    "(8) Timings were taken with exclusive use of the GPU, but no wall-clock performance claim is made "
    "beyond substep counts, which are arithmetic rather than measured."
)

FINDINGS = (
    f"Sand is canonical physics with its own golden signature and costs {sand_spf} substeps per frame, "
    f"the same as elastic, so snow at {snow_spf} still binds the Demo's budget. Four materials now share "
    "one grid and the refactor is provably a no-op. The angle of repose, however, is not a converged "
    "quantity: for every plastic material it decays with the number of substeps taken rather than with "
    "physical time, and canonical snow's collapses from 56 degrees to 19 when the timestep is refined "
    "fourfold."
)


def results():
    out = []

    def add(kind, src, cap, **kw):
        p = os.path.join(D, src)
        if not os.path.exists(p):
            print("!! MISSING, dropping from manifest:", src)
            return
        out.append({"type": kind, "src": f"{REL}/{src}", "caption": cap, **kw})

    add("video", "heap_four_alone.mp4",
        "THE SIGNATURE. The same 60-degree heap released from rest under each material, with each "
        "panel's own free-surface slope shown as it evolves. Water runs flat, elastic and snow keep the "
        "whole seeded slope because they are cohesive, sand relaxes to the slope it can actually hold.")
    add("video", "four_in_one_grid.mp4",
        f"FOUR MATERIALS IN ONE GRID. {M['multi_columns']['n_particles']} particles, four blocks released "
        f"from rest, one shared timestep of {M['multi_columns']['dt_shared']:.1e} s forced on all of them "
        f"by {LAB[M['multi_columns']['binding_material']]}.")
    add("video", "four_heaps_one_grid.mp4",
        "The heap signature again, with all four materials running simultaneously in one simulation "
        "rather than four separate ones. The water pooling against the sand's flank, and the sand "
        "appearing to dam it, is an artifact of two materials sharing a node velocity; it is not a "
        "validated contact model and should not be read as one.")
    add("video", "drop_four_alone.mp4",
        "The drop test, which is a poor discriminator: everything looks like a splat at impact. Sand "
        "travels much further than snow because a dispersed granular pack is under no pressure and so "
        "has no shear strength, and stops well short of the water once its own weight confines it.")
    add("video", "equivalence_sand.mp4",
        "Sand through canonical simulate() and through the new multi-material branching path, side by "
        "side. The number that matters is in the equivalence table; this is the same claim as motion.")
    add("image", "repose_profile.png",
        "The picture behind the repose number: the binned free surface (white) and the least-squares "
        "flank fits (dashed) the reported angle is computed from, for all four materials.")
    add("image", "phi_calibration.png",
        "Measured angle of repose against the Drucker-Prager friction-angle parameter, over 4 seeds and "
        "3 pile sizes. Tight and size-independent up to about 50 degrees; past that the error bars "
        "explode and the angle starts tracking how many grid cells the pile spans.")
    add("image", "repose_vs_time.png",
        "Free-surface slope against time for the four materials on the heap scene, each at its own "
        "canonical timestep.")
    add("image", "dt_budget.png",
        "Left: substeps per frame at 60 fps for each material alone. Right: what a shared grid bills as "
        "materials are added, with each bar coloured by the material that set it.")
    add("image", "dt_faithfulness.png",
        "Settled-shape drift against a dt/8 run, as the timestep is scaled. Red crosses are runs that "
        "diverged. Elastic is flat; the others disagree with the finer run badly at their own canonical "
        "timesteps, which the creep result then shows is the finer run's fault rather than theirs.")
    add("image", "snow_stiffness.png",
        "Distribution of snow's effective stiffness E*exp(xi(1-Jp)) after impact, on a log axis, against "
        "the nominal E of the other three materials. It is strongly bimodal and spans four decades: the "
        "same hardening law that stiffens compacted snow SOFTENS expanded snow, so about 44 percent of "
        "particles end up below snow's own nominal 150 and about 50 percent above elastic's 400. The "
        "hardening is real and large. It is still not what sets snow's timestep.")
    add("image", "equivalence.png",
        "Multi-material path against canonical (solid) and canonical against itself (dotted), per frame. "
        "The curves lie on top of each other, which is what no change looks like against a nondeterministic "
        "reference.")

    out.append({"type": "table", "columns": ["material", "E", "dt", "substeps/frame @60fps",
                                             "stability wall", "measured repose (heap, canonical dt)"],
                "rows": [[LAB[m], f"{M['materials'][m]['E']:.0f}", f"{M['materials'][m]['dt']:.1e}",
                          str(round((1 / 60) / M['materials'][m]['dt'])),
                          f"{SW['summary'][m]['dt_stable_max']:.1e}",
                          f"{h[m]['repose_angle']:.1f} deg"] for m in MATS],
                "caption": "Canonical parameters after the promotion. Snow, not sand, binds the frame "
                           "budget. Repose angles are at each material's own canonical dt and are not "
                           "converged in dt for anything but elastic."})
    out.append({"type": "table",
                "columns": ["material", "run-to-run self-noise", "one-ulp nudge",
                            "multi vs canonical", "vs the nudge"],
                "rows": [[LAB[m], f"{eq[m]['within_path_mean']:.2e}",
                          f"{eq[m]['rounding_nudge_mean']:.2e}", f"{eq[m]['across_path_mean']:.2e}",
                          f"{eq[m]['ratio_to_nudge']:.2f}x"] for m in MATS],
                "caption": "traj_rmse (mean per-particle distance over the whole rollout, domain "
                           "lengths) over 3 repeats of each path. The refactor's disagreement is at or "
                           "below the effect of nudging the initial positions by one float32 rounding "
                           "unit, for every material. Asserted as golden signatures too."})
    out.append({"type": "table",
                "columns": ["scene / material", "within-code range", "across-code range",
                            "ratio of means"],
                "rows": [[f"{r['scene']} / {LAB[r['material']]}",
                          f"{r['within_code_min']:.1e} - {r['within_code_max']:.1e}",
                          f"{r['across_code_min']:.1e} - {r['across_code_max']:.1e}",
                          f"{r['ratio_of_means']:.2f}"] for r in FZ["rows"]],
                "caption": "FROZEN MATERIALS UNCHANGED BY THE PROMOTION. Within-code distances are the "
                           "simulator disagreeing with itself over 4 repeats; across-code are the "
                           "pre-promotion physics (restored from git) against the new one. Every "
                           "across-code range sits inside its within-code range, which is the only "
                           "meaningful statement when the reference is nondeterministic and the scenes "
                           "are chaotic."})
    out.append({"type": "table",
                "columns": ["material", "spread across dt at equal TIME", "at equal SUBSTEP COUNT"],
                "rows": [[LAB[m], f"{cd[m]['spread_at_equal_time']:.1f} deg",
                          f"{cd[m]['spread_at_equal_substeps']:.1f} deg"] for m in MATS],
                "caption": "Settled free-surface slope across four timesteps, compared two ways. "
                           "Collapsing on the substep axis and not the time axis is what says the slump "
                           "is bookkeeping rather than physics. Elastic is the control."})
    return out


man = {
    "schema_version": "2",
    "task_id": TASK,
    "direction": DIR,
    "title": "Sand as a fourth canonical material, and four materials in one grid",
    "tldr": TLDR,
    "status": "active",
    "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "physics_version": M["physics_version"],
    "physics_version_before": FZ["version_before"],
    "objective": OBJECTIVE,
    "summary": SUMMARY,
    "findings": FINDINGS,
    "full_report": FULL,
    "hypothesis": HYPOTHESIS,
    "limitations": LIMITATIONS,
    "results": results(),
    "custom_html": HTML,
    "training_refs": ["material-showcase", "constitutive-models", "real-time-cost", "svd-polar",
                      "linear-algebra"],
}

json.dump(man, open(os.path.join(D, "manifest.json"), "w", encoding="utf-8"), indent=1)
print("wrote manifest.json with", len(man["results"]), "results,", len(HTML), "bytes of custom_html")

# hard check: every media src resolves
missing = [r["src"] for r in man["results"] if "src" in r
           and not os.path.exists(os.path.join(D, os.path.basename(r["src"])))]
print("DANGLING SRCS:", missing if missing else "none")
sys.exit(1 if missing else 0)
