"""Write manifest.json LAST, from the measured numbers, with every media src checked to exist.

    .venv/Scripts/python.exe runs/.../write_manifest.py
"""
import datetime
import json
import pathlib

RUN = pathlib.Path(__file__).resolve().parent
ROOT = RUN.parents[2]
REL = "runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/"

M = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
H = M["per_material_heap"]
MIX = M["mixed_four_material_scene"]
RT = M["realtime"]
PILE = M["fixed_point_headroom"]
SHIP = M["shipped_page"]
SVD = M["svd_unit_test"]
CREEP = M["shared_dt_creep"]
LAST = RT[-1]
WORST_PILE = PILE[-1]
NAME = {"fluid": "water", "elastic": "rubber", "snow": "snow", "sand": "sand"}


def f(v, n=2):
    return ("%." + str(n) + "f") % v


def e(v):
    return "%.1e" % v


ORDER = ["fluid", "elastic", "snow", "sand"]

# Built in named pieces rather than one long expression, because `%` binds tighter than `+` in Python
# and a formatted paragraph in the middle of a concatenation silently applies to the wrong literal.
_heap_table = (
    "| material | settled slope, browser / canonical | traj_rmse | self-noise band | ratio |\n"
    "| --- | --- | --- | --- | --- |\n"
    + "".join("| %s | %s deg / %s deg | %s | %s | %.1fx |\n"
              % (NAME[m], f(H[m]["repose_angle_webgpu"], 1), f(H[m]["repose_angle_canonical"], 1),
                 e(H[m]["traj_rmse"]), e(H[m]["self_noise"]), H[m]["ratio_to_noise_band"])
              for m in ORDER))

_substep_table = (
    "| materials present | shared dt | substeps per 1/60 s frame |\n| --- | --- | --- |\n"
    + "".join("| %s | %.0e | %d |\n" % (" + ".join(NAME[p] for p in r["present"]),
                                        r["dt"], r["substeps_per_frame"])
              for r in M["substeps_by_materials_present"]))

_pile_table = (
    "| particles | heaviest node | ceiling | headroom |\n| --- | --- | --- | --- |\n"
    + "".join("| %s | %s pm | %d pm | %.1fx |\n"
              % ("{:,}".format(r["n"]), f(r["max_node_mass_pm"], 1),
                 r["mass_saturates_at_pm"], r["mass_headroom"]) for r in PILE))

FULL_REPORT = "\n".join([
    "## What shipped",
    "",
    "`harness/dashboard/src/components/DemoView.jsx` is now a live simulation. It imports nothing from "
    "the harness — no `api.js`, no shared components, no app CSS — only its own bundle in "
    "`src/components/mpm/`, which is generated from this run's `web/` by `web/sync_to_dashboard.py`. "
    "The same four files also run as a plain page with script tags (`web/demo.html`), which is the "
    "transplant target.",
    "",
    "Controls: four material buttons that pour on drag, a grab tool (the existing poke body force), a "
    "remove tool, three render modes (material / grid mass / particles), a target-speed slider, reset "
    "and empty. Every one was clicked on the rendered page, in the dashboard, before shipping.",
    "",
    "## The SVD, and why it was verified in isolation",
    "",
    "The starting engine was elastic-only: P2G computed stress through the closed-form 2D polar rotation "
    "and G2P updated F with no plastic projection, no Jp and no material id. Snow's clamp and sand's "
    "return map both read singular values, so `svd2()` is a line-for-line port of Taichi's `_svd2d` — "
    "polar decomposition including the det<0 reflection branch, then one Jacobi rotation — with the same "
    "descending-order convention and the same `A = U S V^T` factor ordering, because the return maps are "
    "written against those conventions.",
    "",
    "It was proved against `ti.svd` on %s matrices in 11 families before being wired in: random, "
    "near-rotation, near-singular, reflections with negative determinant, anisotropic to a condition "
    "number of 1e4, exact zeros, exact identities, exact reflections, the snow clamp boundary, and 800 "
    "deformation gradients each from a real canonical snow and sand rollout. Worst relative "
    "reconstruction %s, worst orthogonality %s, zero descending-order violations, zero non-finite "
    "results, singular values to %s relative against Taichi."
    % ("{:,}".format(SVD["n_matrices"]), e(SVD["max_rel_reconstruction"]),
       e(SVD["max_orthogonality"]), e(SVD["max_rel_singular_vs_taichi"])),
    "",
    "One deliberate deviation from the reference: the polar step floors `sqrt(|det B|)` at 1e-30. "
    "Taichi's argument that det(B) != 0 for any non-zero input is an exact-arithmetic argument; in "
    "float32 a degenerate F can round it to zero, and an infinity there spreads through the transfer "
    "into every node in that particle's stencil.",
    "",
    "## Nine buffers would have been silent",
    "",
    "The device guarantees 8 storage buffers per shader stage (confirmed on this adapter: "
    "`maxStorageBuffersPerShaderStage = 8`) and the elastic layout already used 7. A per-particle "
    "material id and a per-particle Jp are two more. Exceeding the limit produces an invalid bind group, "
    "every dispatch is dropped, and the simulation runs at the speed of doing nothing — which looks like "
    "a spectacular performance result. Both were packed into the velocity buffer instead, widened from "
    "vec2 to vec4: `vel[p] = (vx, vy, Jp, matId)`. The fluid's scalar J rides in `Fm[p].x`, which the "
    "fluid path does not otherwise use. Still 7 buffers. Every benchmark asserts non-zero mean particle "
    "displacement before its timing is recorded.",
    "",
    "## Agreement, material by material",
    "",
    "The instrument is the angle-of-repose heap: a 60-degree over-steep pile released from rest, so "
    "whatever slope survives is the slope the material genuinely holds. Each material at its OWN "
    "canonical dt, 2000 particles, 1.6 s, identical float32 seed on both sides. Judged against "
    "canonical's own self-noise, measured two ways (canonical against canonical identically, and with "
    "the initial positions nudged by 1e-7 — one float32 rounding unit).",
    "",
    _heap_table,
    "On the mixed four-material scene (canonical `simulate_multi` against the same scene in the browser, "
    "shared dt=5e-5, 3245 particles): whole scene %s against a band of %s, i.e. %.1fx — inside it. "
    "Per material: " % (e(MIX["traj_rmse"]), e(MIX["self_noise"]), MIX["ratio_to_noise_band"])
    + ", ".join("%s %.1fx" % (NAME[k], v["ratio_to_nudge_band"]) for k, v in MIX["per_material"].items())
    + ".",
    "",
    "## Cost, and what sets it",
    "",
    _substep_table,
    "Measured compute per real-time frame with four materials present (sustained wall clock over 24 "
    "frames, motion asserted): "
    + ", ".join("%s particles %s ms" % ("{:,}".format(r["n"]), f(r["sustained_ms"])) for r in RT)
    + ". Adding sand to a scene that already contains snow costs nothing; snow alone doubles everything.",
    "",
    "## Two defects that only a screenshot could catch",
    "",
    "Both were invisible to every numerical check, because in both cases the *state* was correct.",
    "",
    "1. **The no-WebGPU overlay rendered on top of a working simulation.** The overlay carries `hidden`, "
    "but an author `display:grid` rule beats the user-agent stylesheet's `[hidden] {display:none}`. The "
    "page ran perfectly underneath a full-screen scrim reading THIS NEEDS WEBGPU. Fixed with an explicit "
    "`.fallback[hidden] { display:none }`.",
    "2. **A fixed 1/60 s of physics per frame.** On the 133 Hz capture display the simulation advanced "
    "2.22x real time while the readout said 1.00x. Fixed by advancing the MEASURED wall-clock frame "
    "interval, clamped to [1/240, 1/24] so a scheduling hiccup cannot request a thousand substeps at once.",
    "",
    "A third, milder one: the self-throttle estimated substep cost as (frame period − draw) / substeps, "
    "which on a vsync-locked display attributes the GPU's idle wait to the solver — it measured 0.047 "
    "ms/substep against a true 0.010 and throttled the page to 0.83x for no reason. Timing "
    "`queue.onSubmittedWorkDone()` is also wrong, because that promise resolves on a JS task and so "
    "includes up to a frame of scheduling latency (0.039 ms). The cost is now read from `timestamp-query` "
    "on the compute pass, with a decaying-minimum fallback where that feature is absent.",
    "",
    "## Fixed-point headroom under deliberate piling",
    "",
    "P2G accumulates mass into an `atomic<u32>` at 2^24 quanta per particle mass, which saturates at 256 "
    "particle masses on one node and then WRAPS SILENTLY. Prior measurements were on scenes with a fixed "
    "particle count; this page hands the density dial to the visitor, so the worst case was measured by "
    "driving all four materials into one corner with the interaction force and holding them there for 90 "
    "frames:",
    "",
    _pile_table,
    "## How it was checked",
    "",
    "`verify/` holds the whole chain and each stage is re-runnable: `prepare_svd.py` + `svd_test.html` + "
    "`score_svd.py` (the SVD unit test), `prepare.py` + `harness.html` + `score.py` (rollouts against "
    "canonical, the piling probe and the real-time sweep), `render.py` (the comparison videos), "
    "`capture_page.py` (drives the shipped page over the Chrome DevTools Protocol with real pointer "
    "events and records it), and `check_dashboard.py` (opens the actual dashboard on :5174, switches to "
    "the Demo tab and clicks every control there).",
    "",
    "Three traps in the harness itself are worth recording. Headless Chromium is NOT a good instrument "
    "for the no-WebGPU path: it exposes `navigator.gpu` and returns an adapter, but has no compositor, so "
    "requestAnimationFrame never fires and the page sits blank — captioning that screenshot 'no WebGPU' "
    "would have been a picture of a completely different failure. The API is hidden from a real "
    "GPU-backed window instead. An off-screen window gets its screencast throttled below 1 fps, which "
    "would have made a 133 fps page look broken, so the capture window is deliberately on screen. And "
    "Vite caches its transform of a module: a browser asking for the un-suffixed URL was handed the "
    "pre-edit build long after the file changed, so the dashboard check now warms the module with a "
    "cache-busting query and asserts the served source is the new one before it believes anything.",
])

manifest = {
    "schema_version": "2",
    "task_id": "the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page",
    "direction": "material-variants",
    "title": "The Demo MVP — four materials on WebGPU, live on the Demo page",
    "tldr": (
        "All four canonical materials now run together on one MLS-MPM grid in a browser at 1.00x real "
        "time with %s particles, and the Demo tab is that simulation; each material matches canonical "
        "physics on a settled-heap test to within %s degrees, but a mixed scene is NOT quantitatively "
        "canonical because one grid forces one timestep, and two bugs found only by screenshotting the "
        "whole page (a CSS rule that drew the no-WebGPU overlay on top of a working sim, and a loop that "
        "ran 2.2x too fast on a 133 Hz display while reporting 1.00x) would have shipped otherwise."
        % ("{:,}".format(SHIP["particles"]),
           f(max(abs(H[m]["repose_angle_webgpu"] - H[m]["repose_angle_canonical"]) for m in H), 2))),
    "status": "active",
    "created": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "physics_version": M["physics_version"],

    "objective": (
        "Put the first real, running simulation on the Demo tab: all four canonical materials — fluid, "
        "elastic, snow and sand — on ONE shared MLS-MPM grid, in real time, on WebGPU, that a visitor can "
        "pour into, drag around, switch render modes on and reset. The engine to start from was "
        "elastic-only: its P2G used the closed-form 2D polar rotation and its G2P had no plastic "
        "projection at all, so snow's Stomakhin clamp and sand's Drucker-Prager return map both needed a "
        "real 2x2 SVD written in WGSL. The success criteria, in priority order, were (1) a working page, "
        "(2) each material recognisably itself, (3) quantitative agreement with canonical physics — with "
        "explicit permission to sacrifice from the bottom, and one unacceptable outcome: shipping a "
        "material that looks wrong without saying so."),

    # Rendered as a single plain-text paragraph by the dashboard, so no markdown emphasis and no
    # reliance on a blank line surviving. Depth belongs in full_report; the picture is the bespoke page.
    "summary": (
        "The Demo tab is now a live four-material MLS-MPM simulation running on WebGPU at 1.00x real "
        "time: %s particles of water, rubber, snow and sand on one shared grid, %s solver substeps per "
        "simulated second, %d fps, with pouring, dragging, erasing, three render modes and a reset. Each "
        "material is recognisably itself and lands on canonical physics where it should — on the "
        "angle-of-repose heap the settled slopes agree to %s degrees or better (browser / canonical: "
        "water %s / %s, rubber %s / %s, snow %s / %s, sand %s / %s), and on the mixed four-material scene "
        "the whole-scene trajectory error of %s sits inside canonical's own self-noise band of %s. The "
        "genuinely new engineering was a 2x2 SVD in WGSL, without which neither snow's clamp nor sand's "
        "return map can be evaluated at all; it was checked against ti.svd on %s adversarial matrices "
        "before being wired in, reconstructing to %s relative with singular values agreeing to %s. "
        "Three honest caveats. A mixed scene is NOT quantitatively canonical: one grid forces one "
        "timestep, so putting snow in the box runs sand at half its own step and costs it %s degrees of "
        "settled slope, which means a material's behaviour depends on what else is in the scene. The "
        "fixed-point mass accumulator wraps silently at 256 particle masses on a node and was re-measured "
        "under deliberate piling rather than on a fixed scene: %sx of headroom left at %s particles, "
        "which is comfortable but is a function of how much a visitor chooses to pile up. And two defects "
        "passed every numerical check and were caught only by screenshotting the whole rendered page — a "
        "CSS rule that painted the \"no WebGPU\" overlay on top of a perfectly working simulation, and a "
        "loop advancing a fixed 1/60 s of physics per frame, which runs 2.2x too fast on a 133 Hz display "
        "while the readout cheerfully reports 1.00x."
        % ("{:,}".format(SHIP["particles"]), "{:,}".format(SHIP["substeps_per_simulated_second"]),
           round(SHIP["fps"]),
           f(max(abs(H[m]["repose_angle_webgpu"] - H[m]["repose_angle_canonical"]) for m in H), 2),
           f(H["fluid"]["repose_angle_webgpu"], 1), f(H["fluid"]["repose_angle_canonical"], 1),
           f(H["elastic"]["repose_angle_webgpu"], 1), f(H["elastic"]["repose_angle_canonical"], 1),
           f(H["snow"]["repose_angle_webgpu"], 1), f(H["snow"]["repose_angle_canonical"], 1),
           f(H["sand"]["repose_angle_webgpu"], 1), f(H["sand"]["repose_angle_canonical"], 1),
           e(MIX["traj_rmse"]), e(MIX["self_noise"]), "{:,}".format(SVD["n_matrices"]),
           e(SVD["max_rel_reconstruction"]), e(SVD["max_rel_singular_vs_taichi"]),
           f(abs(CREEP["sand"]["delta_deg"]), 1),
           f(WORST_PILE["mass_headroom"], 1), "{:,}".format(WORST_PILE["n"]))),

    "findings": (
        "Tested: four materials, two scene families (a 60-degree over-steep heap of 2000 particles per "
        "material at that material's own canonical dt, and one mixed drop of 3245 particles at the shared "
        "dt=5e-5), 1.6 s of physics each, on one device (RTX 4090 / Chromium 140 / WebGPU), at n_grid=128. "
        "OBSERVED. (1) The WGSL 2x2 SVD agrees with `ti.svd` on %s matrices across 11 adversarial families "
        "including reflections, near-singular and near-rotation cases and 1600 deformation gradients "
        "sampled from real snow and sand rollouts: worst relative reconstruction %s, worst orthogonality "
        "%s, zero ordering violations, singular values to %s relative. (2) On the heap, each material's "
        "settled slope in the browser matches canonical to <= %s degrees, and `traj_rmse` against canonical "
        "sits at %.1fx / %.1fx / %.1fx / %.1fx canonical's own 1e-7-perturbation self-noise band for "
        "water / rubber / snow / sand. (3) On the mixed four-material scene the whole-scene ratio is %.1fx "
        "(inside the band) and no single material exceeds %.1fx. (4) Cost: four materials present forces "
        "dt = min(dt) = 5e-5, i.e. %d substeps per 1/60 s frame; measured %s ms of compute at %s particles, "
        "%s ms at %s. (5) Fixed-point headroom with all four materials deliberately crushed into one corner "
        "by the interaction force: heaviest node %s particle masses against a silent-wrap ceiling of %d, "
        "i.e. %.1fx, at %s particles. (6) The shipped page sustains %.2fx real time at %s particles and "
        "%d fps, in the dashboard's own Demo tab, with zero WebGPU errors, and holds 1.00x at its full "
        "capacity of 16,384 particles during the recorded session -- consistent with the sweep, which "
        "puts four-material compute at 5.25 ms of a 16.67 ms frame there. "
        "NOT OBSERVED / NOT CLAIMED: nothing about other devices, other scenes, other grid resolutions, or "
        "rollouts longer than 1.6 s."
        % ("{:,}".format(SVD["n_matrices"]), e(SVD["max_rel_reconstruction"]),
           e(SVD["max_orthogonality"]), e(SVD["max_rel_singular_vs_taichi"]),
           f(max(abs(H[m]["repose_angle_webgpu"] - H[m]["repose_angle_canonical"]) for m in H), 2),
           H["fluid"]["ratio_to_noise_band"], H["elastic"]["ratio_to_noise_band"],
           H["snow"]["ratio_to_noise_band"], H["sand"]["ratio_to_noise_band"],
           MIX["ratio_to_noise_band"],
           max(v["ratio_to_nudge_band"] for v in MIX["per_material"].values()),
           M["substeps_per_frame_at_60fps_shared"],
           f(RT[0]["sustained_ms"]), "{:,}".format(RT[0]["n"]),
           f(LAST["sustained_ms"]), "{:,}".format(LAST["n"]),
           f(WORST_PILE["max_node_mass_pm"], 1), WORST_PILE["mass_saturates_at_pm"],
           WORST_PILE["mass_headroom"], "{:,}".format(WORST_PILE["n"]),
           SHIP["achieved_x_real_time"], "{:,}".format(SHIP["particles"]), round(SHIP["fps"]))),

    "hypothesis": (
        "HYPOTHESIS (mechanism, not observation). The reason a 2x2 SVD ports cleanly to a GPU shading "
        "language at all is that the plastic laws consume only its *gauge-invariant* part. Right-"
        "multiplying a deformation gradient by any rotation leaves the singular values, the corotated "
        "stress (F-R)F^T and the Hencky stress U(...)U^T exactly unchanged, because an isotropic material "
        "cannot see a relabelling of its reference configuration. So two implementations may return "
        "different U and V and be doing identical physics, and the only outputs that must agree are Sigma "
        "and the observables. That is what makes a static, few-thousand-matrix unit test a sufficient "
        "proof of a routine that will then be called ~10^8 times.\n\n"
        "HYPOTHESIS for the residual disagreement pattern. Water, snow and sand land at or near the "
        "self-noise band while rubber sits at %.0fx it — yet rubber's absolute error (%s domain lengths) "
        "is the smallest of the four. The proposed mechanism is that the ratio measures the wrong thing "
        "on a scene that barely moves: an elastic heap at 56 degrees is nearly static, so canonical "
        "reproduces itself almost exactly (band %s) while fixed-point quantisation in P2G contributes a "
        "small *deterministic* bias rather than chaos. WOULD TEST: re-run the same comparison with the "
        "exact-f32 compare-and-swap atomic path instead of fixed point; if the mechanism is right, "
        "rubber's ratio should collapse toward 1 while the other three barely move.\n\n"
        "HYPOTHESIS about the class of bug this task hit. Both defects that survived every numerical "
        "check — the overlay painted over a working simulation, and the fixed-1/60-s loop — are invisible "
        "to any test that inspects state rather than pixels, because the state was correct in both cases. "
        "The conjecture is that a rendered artifact has a category of failure that only a screenshot of "
        "the whole surface can catch, and that this is why 'open the page and look at it' is a distinct "
        "obligation from 'check the numbers'. WOULD TEST: whether the same two classes recur on the next "
        "browser-facing task when only numerical checks are run."
        % (H["elastic"]["ratio_to_noise_band"], e(H["elastic"]["traj_rmse"]),
           e(H["elastic"]["self_noise"]))),

    "limitations": (
        "1. SHARED TIMESTEP, AND IT IS A CORRECTNESS ISSUE, NOT A COST ONE. One grid means one dt = "
        "min(dt) over the materials present, so a scene containing snow runs sand at half sand's own "
        "canonical step. A plastic material's settled slope decays with SUBSTEP COUNT rather than physical "
        "time, so this gives it MORE creep than canonical shows: measured canonical-against-canonical on "
        "the same heap, sand goes %s -> %s degrees and water %s -> %s; rubber, which has no plastic "
        "projection, moves %s. A MATERIAL'S BEHAVIOUR THEREFORE DEPENDS ON WHAT ELSE IS IN THE SCENE, and "
        "no mixed scene here is claimed to be quantitatively canonical.\n"
        "2. ONE DEVICE. Every number is from one RTX 4090 in Chromium 140. Nothing was measured on the "
        "iPad or the MacBook, so the real-time claim is a claim about this machine.\n"
        "3. FIXED-POINT HEADROOM IS A FUNCTION OF WHAT THE VISITOR DOES. %.1fx at %s particles under "
        "deliberate piling is comfortable, but the accumulator wraps SILENTLY, and an interactive page "
        "hands the particles-per-cell dial to the user. The demo caps capacity at %s partly for this.\n"
        "4. RUBBER'S AGREEMENT IS THE WEAKEST BY RATIO (%.0fx the band) even though it is the strongest by "
        "absolute error; see the hypothesis for why, and note that the explanation is untested.\n"
        "5. PARTICLE DENSITY. The demo runs at ~1.7 particles per grid cell — taken from the mixed-material "
        "verification scene, which is the only scene at that density that was checked against canonical. "
        "The per-material heaps ran at ~4.2 particles per cell. Agreement was not measured as a function "
        "of density.\n"
        "6. NOT DONE: no JS fallback for browsers without WebGPU (the page explains itself instead and "
        "degrades to a message); no per-material colour in the grid-mass view (it shows total node mass); "
        "erased particles are compacted on the host at pointer-up rather than on the GPU, so a very long "
        "erase drag holds inert slots until the pointer is released; and no measurement of whether "
        "per-call SVD agreement predicts drift over rollouts longer than 1.6 s."
        % (f(CREEP["sand"]["repose_own"], 1), f(CREEP["sand"]["repose_shared"], 1),
           f(CREEP["fluid"]["repose_own"], 1), f(CREEP["fluid"]["repose_shared"], 1),
           f(abs(CREEP["elastic"]["delta_deg"]), 2) + " degrees",
           WORST_PILE["mass_headroom"], "{:,}".format(WORST_PILE["n"]), "{:,}".format(16384),
           H["elastic"]["ratio_to_noise_band"])),

    "full_report": FULL_REPORT,

    "results": [
        {"type": "video", "src": REL + "material_vs_canonical.mp4",
         "caption": "Each material against canonical ground truth, as motion. An over-steep 60-degree heap "
                    "released from rest: canonical `sim.physics` on top, the same step running in WebGPU "
                    "underneath, identical seed, dt and substep count. The fitted flank whose slope IS the "
                    "reported angle of repose is drawn on the final frame of every panel."},
        {"type": "video", "src": REL + "mixed4_vs_canonical.mp4",
         "caption": "All four materials on ONE grid at the shared dt=5e-5, canonical `simulate_multi` "
                    "beside the browser. Whole-scene traj_rmse %s against a self-noise band of %s."
                    % (e(MIX["traj_rmse"]), e(MIX["self_noise"]))},
        {"type": "video", "src": REL + "demo_capture.mp4",
         "caption": "The shipped Demo page, captured from a GPU-backed window driven by real pointer "
                    "events: the opening scene lands, sand is poured in, the view switches to grid mass "
                    "and to raw particles, the material is dragged, a channel is carved out, reset. Real "
                    "time — the HUD's speed readout is measured, not assumed."},
        {"type": "image", "src": REL + "verify/shots/dashboard_demo_tab.png",
         "caption": "The Demo tab in the dashboard itself, through Vite, React and a generated ES-module "
                    "copy of the same code. Its own HUD, read off that page: "
                    + "; ".join(c.strip() for c in SHIP["dashboard_tab"]["chips"][:4]) + "."},
        {"type": "image", "src": REL + "repose_bars.png",
         "caption": "The material-identity claim as numbers: settled slope of the heap, browser against "
                    "canonical, all four materials at their own canonical timestep."},
        {"type": "image", "src": REL + "cost_and_headroom.png",
         "caption": "Left: compute per real-time frame with four materials present (333 substeps), against "
                    "the 16.67 ms budget. Right: the heaviest grid node under deliberate piling, against "
                    "the 256-particle-mass ceiling where the fixed-point accumulator wraps silently."},
        {"type": "image", "src": REL + "demo_no_webgpu.png",
         "caption": "Graceful degradation, captured by hiding navigator.gpu from a real window. The page "
                    "distinguishes hidden-by-insecure-origin, unsupported, and no-device, because "
                    "'absent' most often means the origin is not secure rather than the device is old."},
        {"type": "table", "columns": ["material", "slope browser / canonical", "traj_rmse", "self-noise band", "ratio"],
         "rows": [[NAME[m], "%s / %s deg" % (f(H[m]["repose_angle_webgpu"], 1), f(H[m]["repose_angle_canonical"], 1)),
                   e(H[m]["traj_rmse"]), e(H[m]["self_noise"]), "%.1fx" % H[m]["ratio_to_noise_band"]]
                  for m in ["fluid", "elastic", "snow", "sand"]],
         "caption": "Angle-of-repose heap, 2000 particles, each material at its own canonical dt. The band "
                    "is canonical against itself with the initial positions nudged by 1e-7."},
        {"type": "table", "columns": ["materials present", "shared dt", "substeps / frame", "vs water alone"],
         "rows": [[" + ".join(NAME[p] for p in r["present"]), "%.0e" % r["dt"],
                   str(r["substeps_per_frame"]),
                   "%.2fx" % (r["substeps_per_frame"] / M["substeps_by_materials_present"][0]["substeps_per_frame"])]
                  for r in M["substeps_by_materials_present"]],
         "caption": "One grid means one timestep: min(dt) over the materials PRESENT. Snow doubles the "
                    "substep count for every particle in the box; adding sand on top of snow is free."},
        {"type": "table", "columns": ["check", "result", "threshold"],
         "rows": [["reconstruction |U diag(s) V^T - A| / |A|", e(SVD["max_rel_reconstruction"]), "< 1e-4"],
                  ["orthogonality |U^T U - I|, |V^T V - I|", e(SVD["max_orthogonality"]), "< 1e-4"],
                  ["singular values vs ti.svd (relative)", e(SVD["max_rel_singular_vs_taichi"]), "< 1e-4"],
                  ["descending-order violations", str(SVD["order_violations"]), "0"],
                  ["non-finite results", str(SVD["non_finite"]), "0"],
                  ["matrices tested", "{:,}".format(SVD["n_matrices"]), "11 adversarial families"]],
         "caption": "The WGSL 2x2 SVD, proved in isolation before it was wired into the solver. A wrong "
                    "SVD does not crash — it produces plausible-looking motion, which is why this is a "
                    "standalone unit test against ti.svd rather than a visual check."},
    ],

    "custom_html": (RUN / "bespoke_page.html").read_text(encoding="utf-8"),
    "training_refs": ["svd-in-practice", "constitutive-models", "svd-polar", "real-time-cost",
                      "fixed-point-atomics"],
    "metrics_used": ["traj_rmse", "self_noise", "repose_angle", "pile_height", "spread_width",
                     "substeps_per_frame", "frame_ms", "node_mass_headroom", "realtime_factor",
                     "physics_version"],
    "device": M["device"],
    "code": {
        "engine": REL + "web/mpm4-webgpu.js",
        "ui": REL + "web/demo4.js",
        "params_generator": REL + "web/gen_params.py",
        "sync_to_dashboard": REL + "web/sync_to_dashboard.py",
        "standalone_page": REL + "web/demo.html",
        "ships_to": "harness/dashboard/src/components/DemoView.jsx (+ src/components/mpm/)",
        "verification": REL + "verify/",
    },
}


def main():
    missing = []
    for r in manifest["results"]:
        src = r.get("src")
        if src and not (ROOT / src).exists():
            missing.append(src)
    if missing:
        raise SystemExit("DANGLING MEDIA: " + ", ".join(missing))
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote manifest.json — every media src resolves")
    print("tldr:", manifest["tldr"][:200], "...")


if __name__ == "__main__":
    main()
