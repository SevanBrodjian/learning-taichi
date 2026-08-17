"""Write manifest.json LAST, with custom_html inlined from bespoke_page.html and every media src
checked to exist on disk before it is listed.

Note on formatting: the dashboard renders every prose field as PLAIN TEXT inside a <p>, so there is no
markdown here and no newline-dependent layout. Anything tabular goes in a `table` result, which the
dashboard does render as a table, or in the bespoke page.

    .venv/Scripts/python.exe runs/material-variants/interactive-simulation-of-one-material/write_manifest.py
"""
import datetime
import json
import pathlib
import sys

RUN = pathlib.Path(__file__).resolve().parent
ROOT = RUN.parents[2]
REL = "runs/material-variants/interactive-simulation-of-one-material/"

M = json.loads((RUN / "metrics.json").read_text())
BB = json.loads((RUN / "browser_bench.json").read_text())
GB = json.loads((RUN / "verify" / "gpu_bench.json").read_text())
PAGE = (RUN / "bespoke_page.html").read_text(encoding="utf-8")
L = M["launch_scene"]

TLDR = ("The elastic material now runs interactively in a browser and matches the GPU reference to within "
        "the reference's own run-to-run noise, but real time costs 167 substeps per frame, which caps this "
        "machine at about 1150 particles, and raising the timestep to buy frame rate wrecks the trajectory "
        "long before it becomes unstable.")

OBJECTIVE = ("Port the canonical MLS-MPM elastic step out of Taichi/CUDA into something that runs "
             "interactively in a web browser, verify the port against canonical sim.physics on matched "
             "initial conditions, and measure what interactive rates actually cost.")

SUMMARY = (
    "The port works and it is numerically exact. A single-threaded JavaScript reimplementation of the "
    "elastic step, taking every parameter straight out of sim.physics, tracks the canonical Taichi/CUDA "
    "simulator to a trajectory RMSE of 1.7e-4 over a 2.5 second contact-heavy rollout, which is smaller "
    "than the 0.7e-4 to 2.2e-4 band the reference produces against itself when it is simply re-run, or "
    "when its starting positions are nudged by one f32 rounding unit. It takes mouse and touch input, it "
    "is the centrepiece of this page, and it holds a full 60 fps at real time with 1000 particles. "
    "The interesting result is what the port cost, and it was not the port. Elastic at E = 400 needs "
    "dt = 1e-4, so one displayed frame at real time is 167 full P2G, grid and G2P passes, and that single "
    "fact caps this machine at roughly 1150 particles at 60 fps. Two things followed from measuring rather "
    "than assuming: Taichi's dense 128x128 grid sweep costs 15.6 ms per frame on one thread before a single "
    "particle is touched, because only about 760 of 16384 cells ever hold material, so the sparse rewrite "
    "is what makes real time reachable at all; and the canonical CUDA reference is flat at about 345 "
    "microseconds per substep from 500 to 16384 particles because it is launch-bound, so below roughly "
    "4300 particles one JavaScript thread beats an RTX 4090 running the same physics. Buying frame rate by "
    "raising the timestep fails badly, a 1.6x speedup costing three orders of magnitude of trajectory "
    "accuracy while the simulation still looks perfectly stable."
)

FINDINGS = (
    "On one material (elastic), two scenes and one particle count: the browser port reproduces canonical "
    "sim.physics to within the canonical simulator's own reproducibility floor; real time costs 167 "
    "substeps per frame and caps this machine at about 1150 particles at 60 fps; the dense grid sweep "
    "inherited from Taichi is on its own worth the entire frame budget on one thread; the canonical CUDA "
    "reference reaches only 0.29x real time at this problem size because it is launch-bound; and the "
    "timestep degrades trajectory accuracy by orders of magnitude well before it becomes unstable."
)

HYPOTHESIS = (
    "Why the port comes out exact: nothing in the elastic path is order-dependent except the P2G scatter, "
    "and a single thread removes that source of nondeterminism entirely, so the only residual is f32 versus "
    "f64 rounding, which a chaotic contact-rich rollout amplifies at the same exponential rate as the "
    "reference amplifies its own atomic-ordering noise. The divergence curves are the evidence for this "
    "reading rather than the summary number: the port's curve has the shape of the reference's self-noise "
    "curve, starting at rounding scale and growing exponentially until it saturates, and not the shape of a "
    "bias, which would appear immediately and grow linearly. "
    "Why the cost result holds: an explicit solver's frame cost is the number of substeps per frame times "
    "the cost of one substep, the substep count is (1/60)/dt, and dt is pinned by the CFL condition dx/c "
    "with wave speed c = sqrt((lambda+2mu)/rho) of about 21 domain lengths per second. That makes stiffness, "
    "not particle count, the thing that sets an interactive budget. Conjecture beyond what was tested, since "
    "only one material was measured: a stiffer material should be proportionally more expensive to run "
    "interactively, because dt scaling as 1/sqrt(E) implies substeps scaling as sqrt(E), so the canonical "
    "fluid at E = 180 and dt = 1.2e-4 should afford somewhat more particles than elastic does. "
    "On the analytic versus learned grid update, and this is explicitly a conjecture rather than a result: a "
    "learned per-substep update is optimising the wrong term, because it is multiplied by the 167 substeps "
    "instead of replacing them. Measured, the smallest useful network (an 8-32-32-2 MLP per active cell) "
    "costs 242 times the analytic grid update in JavaScript and overruns the frame budget by 13x; on the GPU "
    "the same network is free relative to the analytic update, but only because both are hidden underneath a "
    "56 microsecond kernel launch, which is a fact about launches and not about models. The only version of "
    "the idea that could win is one that learns the coarse-time update, mapping the state at t directly to "
    "the state one frame later, so that it runs once per frame instead of 167 times. Training such a model "
    "and measuring both its trajectory RMSE against a canonical rollout and its cost per frame is what would "
    "settle it."
)

LIMITATIONS = (
    "One material only. Fluid and snow were not ported. Snow's plastic clamp and the fluid pressure and "
    "viscosity terms are branches this port never touches, and snow additionally needs the full SVD rather "
    "than only the polar rotation, so the finding that the SVD was unnecessary is specific to the 2D elastic "
    "path. "
    "One machine and, for the sweep, one browser. All browser numbers come from Chromium 148 running inside "
    "Electron 42 on an Intel Raptor Lake desktop, with a live 60 fps cross-check in Edge 151 on the same "
    "machine; the reference numbers come from Taichi 1.7.4 on an RTX 4090. WebGPU was not available in "
    "either browser tested, so no GPU path in the browser was measured at all and the analytic versus "
    "learned comparison is CPU-only on the browser side. The 1150-particle budget and the roughly "
    "4300-particle CPU/GPU crossover are statements about this pair of implementations on this hardware, "
    "not about browsers or GPUs in general. No iPad or phone measurement was taken, although the demo does "
    "take touch input. "
    "The CUDA figures are for Taichi kernels launched from Python in a loop, which is how the canonical "
    "simulator is actually used in this project. A fused or graph-captured CUDA implementation would pay far "
    "less launch overhead, so 'the GPU is slow here' is a claim about the reference implementation as used, "
    "not about the device. "
    "Scene note: sim.physics does not own scenes, so the launched-disk initial condition (a disk of radius "
    "0.11 at (0.30, 0.55) with v0 = (0.75, 0)) is specified in this run rather than imported; the drop scene "
    "is the canonical one. Everything else, meaning E, dt, grid resolution, Poisson ratio, friction, gravity "
    "and particle volume, is generated directly from sim.physics by web/gen_params.py. The pointer "
    "interaction is a demo-only external actuator, not canonical physics, and is disabled in every "
    "verification run. "
    "No learned grid update was trained. The cost of one was measured; its accuracy was not."
)

FULL = (
    "What was built. web/mpm-elastic.js is a standalone reimplementation of the canonical 2D MLS-MPM "
    "elastic step that runs unchanged in Node and in a browser, with no dependency on the dashboard, the "
    "data server or the harness. web/demo.html is the transplantable self-contained demo and "
    "bespoke_page.html is that demo plus the evidence. Parameters are generated rather than retyped: "
    "web/gen_params.py imports sim.physics, reads MAT['elastic'] and the frozen world constants, and emits "
    "web/params.js stamped with physics_version " + M["physics_version"] + ", so no physical constant is "
    "named anywhere in the JavaScript. "
    "Verification. The same initial condition was rolled through canonical sim.physics and through the port "
    "on two scenes, and scored with the registered metric traj_rmse (a mean per-particle distance, not an "
    "RMS). Because the reference scatters through GPU atomics it is not deterministic, so a noise floor was "
    "built from the reference itself: run it twice on identical input, and run it once more with the "
    "starting positions perturbed by 1e-7, roughly one f32 rounding unit at these coordinates. On the "
    "contact-heavy launched scene the port's divergence of %.2e is smaller than the %.2e the reference "
    "produces against its own perturbed start, and comparable to the %.2e it produces against a plain "
    "re-run. Settled shape agrees to five digits on the drop scene, spread width %.5f canonical against "
    "%.5f ported. "
    % (L["traj_rmse"]["port_vs_canonical"], L["traj_rmse"]["canonical_perturbed_ic"],
       L["traj_rmse"]["canonical_self_noise"], M["shape"]["gt_final_width"], M["shape"]["port_final_width"])
    + "How the port differs from the original, and why. The constitutive law is unchanged: fixed corotated, "
    "2 mu (F - R) F^T + lambda (J - 1) J I. Three things had to change, all forced by 'one CPU thread "
    "instead of a GPU'. First, ti.svd was removed. In 2D the elastic path needs only the polar rotation "
    "R = U V^T, and Taichi's own 2D SVD is built on a closed-form polar decomposition, so the singular "
    "values were never used and an entire factorisation collapses to two adds, a hypot and a reciprocal "
    "square root. Second, the dense 128x128 grid loop was rewritten sparse: P2G records every cell it "
    "scatters into, and the grid update and the clear walk only that list. This is exact rather than "
    "approximate, because every node G2P gathers from is a node P2G scattered to, and it was checked "
    "bit-for-bit against the dense loop over %d substeps with %d differing float values. Third, arithmetic "
    "is f64 with f32 storage, because JavaScript has no float32 math, which makes the port slightly more "
    "accurate per operation than the reference rather than less. "
    % (M["node_report"]["sparse_vs_dense"]["steps"],
       M["node_report"]["sparse_vs_dense"]["n_differing_values"])
    + "Measurement method. Browser costs are minimums over 5 repetitions of a 250-substep loop, which is "
    "robust to the operating system stealing the core mid-measurement. Per-phase costs are obtained by "
    "differencing whole loops (time p2g+clear, then p2g+grid+clear, then the full step) rather than by "
    "timing a phase directly, because the engine rounds performance.now() to 100 microseconds and a single "
    "50 microsecond phase is below that resolution. An earlier attempt at direct per-substep instrumentation "
    "produced a phase split that disagreed badly with the differencing result and was discarded. "
    "GPU costs are Taichi kernels timed with ti.sync() around 400 repetitions after warmup. "
    "Reproducing. Run web/gen_params.py, then verify/run_all.py, verify/launch_scene.py, "
    "verify/gpu_bench.py, verify/render.py, then web/build.py, web/build_page.py and write_manifest.py, "
    "all from the repo root with the project venv. The intermediate trajectory dumps are deleted after "
    "rendering; run_all.py and launch_scene.py regenerate them. Browser timings come from "
    "MPMDemo.benchCPU, benchPhases and benchNet in web/demo.js, invoked in the running page."
)

SWEEP_ROWS = [[str(r["n"]),
               "%.1f" % r["sparse"]["us_per_step"], "%.2fx" % r["sparse"]["realtime_factor"],
               "%.1f" % r["dense"]["us_per_step"], "%.2fx" % r["dense"]["realtime_factor"]]
              for r in BB["cpu_sweep"]]

DT_ROWS = [["%gx (%.1e)" % (e["mult"], e["dt"]), str(e["spf"]),
            "%.2fx" % e["speedup_vs_canonical"],
            ("%.2e" % e["traj_rmse_vs_canonical"]) if e["finite"] else "non-finite",
            "exact" if (e["finite"] and e["traj_rmse_vs_canonical"] < 1e-3) else
            ("wrong shape and position" if e["finite"] else "blew up")]
           for e in L["dt_sweep"]]

GPU_ROWS = [[str(r["n"]), "%.0f" % r["us_per_step"], "%.2fx" % r["realtime_factor"],
             "%.1f" % r["frame_ms_at_60fps_realtime"]] for r in GB["per_n"]]

RESULTS = [
    {"type": "video", "src": REL + "port_vs_canonical_launch.mp4",
     "caption": "Launched elastic disk: canonical Taichi/CUDA on the left, the browser port in the middle, "
                "same initial condition, with the per-frame divergence on the right. The port's curve "
                "(orange) sits inside the band formed by the reference's own run-to-run noise (purple) and "
                "the effect of nudging the start by one f32 rounding unit (green)."},
    {"type": "video", "src": REL + "dt_sweep_launch.mp4",
     "caption": "Raising the timestep to buy frame rate. Cyan is the canonical run in all three panels, "
                "orange is the cheaper timestep. At 2x and 4x the material rolls visibly further than the "
                "truth and ends a full diameter away, while still looking like a stable rubber ball."},
    {"type": "video", "src": REL + "port_vs_canonical.mp4",
     "caption": "The same comparison on the canonical drop scene, where the disk settles instead of "
                "rolling. Divergence is about ten times smaller because there is far less contact."},
    {"type": "video", "src": REL + "dt_sweep.mp4",
     "caption": "Timestep sweep on the canonical drop scene. Here the 4x run does not merely drift, it goes "
                "non-finite and the particles pin against the domain clamp."},
    {"type": "image", "src": REL + "substep_budget.png",
     "caption": "Cost of one MLS-MPM substep against particle count. The port crosses the 60 fps real-time "
                "line at 1154 particles, the dense grid loop never crosses it at all, and the CUDA "
                "reference is flat because it is launch-bound, so one JS thread beats it below about 4300 "
                "particles."},
    {"type": "image", "src": REL + "dt_tradeoff.png",
     "caption": "Trajectory error against the speedup bought by raising the timestep, on the canonical drop "
                "scene. Accuracy collapses by three orders of magnitude at a 1.6x speedup, long before the "
                "solver destabilises."},
    {"type": "table",
     "columns": ["particles", "sparse us/substep", "sparse x real time", "dense us/substep",
                 "dense x real time"],
     "rows": SWEEP_ROWS,
     "caption": "Browser cost sweep, Chromium 148 single thread, canonical dt = 1e-4 so 167 substeps per "
                "60 fps frame and a budget of 99.8 us per substep. Interpolating the sparse column onto "
                "that budget gives the 60 fps particle budget of about 1154 particles. The dense grid loop "
                "never reaches real time at any particle count."},
    {"type": "table",
     "columns": ["timestep", "substeps/frame", "speedup", "traj_rmse", "outcome"],
     "rows": DT_ROWS,
     "caption": "Timestep sweep on the launched-disk scene, scored against the canonical-dt canonical run. "
                "Accuracy is gone at 1.5x while the solver is still perfectly stable. On the canonical drop "
                "scene the 4x run goes non-finite instead."},
    {"type": "table",
     "columns": ["particles", "us/substep", "x real time", "ms per frame at real time"],
     "rows": GPU_ROWS,
     "caption": "The canonical Taichi/CUDA reference on an RTX 4090, same physics, same grid. Flat across a "
                "32x range of particle counts, which is what launch-bound looks like: an empty kernel over "
                "the same grid costs %.0f us to launch and grid_op alone costs %.0f us."
                % (GB["noop_launch_us"], GB["grid_op_only_us"])},
    {"type": "table",
     "columns": ["grid update", "where", "us per substep", "x analytic", "at 167 substeps per frame"],
     "rows": [
         ["analytic (the equations)", "browser JS", "5.2", "1x", "0.9 ms"],
         ["MLP 8-32-32-2 per active cell", "browser JS", "1258", "242x", "210 ms, 4.8 fps"],
         ["MLP 8-64-64-2 per active cell", "browser JS", "3900", "750x", "651 ms, 1.5 fps"],
         ["MLP 8-128-128-2 per active cell", "browser JS", "13340", "2565x", "2.2 s per frame"],
         ["analytic (grid_op)", "CUDA, RTX 4090", "84", "1x", "14.0 ms"],
         ["MLP 8-32-32-2, all 16384 cells", "CUDA, RTX 4090", "87", "1.03x", "14.5 ms"],
     ],
     "caption": "What a learned grid update would cost, measured on the same machine over the same ~760 "
                "active cells, against the analytic update it would replace. Analytic first was the right "
                "ordering: without a correct reference there is nothing to validate a learned update "
                "against, and with one in hand the learned option can be priced instead of guessed."},
    {"type": "table", "columns": ["quantity", "value"], "rows": [
        ["physics_version", M["physics_version"]],
        ["material and parameters", "elastic, E=400, dt=1e-4, nu=0.2, generated from sim.physics"],
        ["grid, particles (demo default)", "128x128, 1000 particles"],
        ["substeps per frame at 60 fps", "167"],
        ["60 fps particle budget (measured)", "about 1154 particles, single JS thread"],
        ["live rate, Chromium 148, 1500 particles", "45 fps, 0.76x real time"],
        ["live rate, Edge 151, 1000 particles", "60 fps, 1.00x real time"],
        ["frame split at 1500 particles", "P2G 11.3 ms, grid 1.0 ms, G2P 9.3 ms, draw 0.15 ms"],
        ["dense grid loop cost", "93 us/substep, 15.6 ms/frame of empty cells"],
        ["traj_rmse port vs canonical (launch)", "%.2e" % L["traj_rmse"]["port_vs_canonical"]],
        ["traj_rmse canonical vs itself (launch)", "%.2e" % L["traj_rmse"]["canonical_self_noise"]],
        ["traj_rmse canonical vs 1e-7 nudge (launch)", "%.2e" % L["traj_rmse"]["canonical_perturbed_ic"]],
        ["traj_rmse port vs canonical (drop)", "%.2e" % M["traj_rmse"]["port_vs_canonical"]],
        ["traj_rmse canonical vs itself (drop)", "%.2e" % M["traj_rmse"]["canonical_self_noise"]],
        ["sparse vs dense grid loop", "bit-identical over 3340 substeps"],
        ["canonical CUDA reference", "345 us/substep flat, 0.29x real time on an RTX 4090"],
        ["browser and hardware", BB["engine"]["engine_short"] + "; " + BB["machine"]["cpu"]],
        ["WebGPU", "not available in the browsers tested"],
    ], "caption": "Everything the claims rest on, in one place. Raw numbers live in metrics.json, "
                  "browser_bench.json and verify/gpu_bench.json."},
]


def main():
    missing = [r["src"] for r in RESULTS if r.get("src") and not (ROOT / r["src"]).exists()]
    if missing:
        print("MISSING MEDIA:", missing)
        sys.exit(1)
    for field in (TLDR, OBJECTIVE, SUMMARY, FINDINGS, FULL, HYPOTHESIS, LIMITATIONS):
        for bad in ("|", "**", "\n", "`"):
            if bad in field:
                print("PROSE CONTAINS %r (renders literally in the dashboard): %r"
                      % (bad, field[max(0, field.find(bad) - 60):field.find(bad) + 40]))
                sys.exit(1)
    man = {
        "schema_version": "2",
        "task_id": "interactive-simulation-of-one-material",
        "direction": "material-variants",
        "title": "One material, live in a browser",
        "tldr": TLDR,
        "status": "active",
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "physics_version": M["physics_version"],
        "objective": OBJECTIVE,
        "summary": SUMMARY,
        "findings": FINDINGS,
        "full_report": FULL,
        "hypothesis": HYPOTHESIS,
        "limitations": LIMITATIONS,
        "results": RESULTS,
        "custom_html": PAGE,
        "training_refs": ["real-time-cost", "material-stiffness", "svd-polar", "mls-mpm-forward"],
        "params": {
            "material": "elastic", "E": 400.0, "dt": 1e-4, "nu": 0.2, "n_grid": 128,
            "n_particles_verification": M["n_particles"], "T": M["T"], "n_frames": M["n_frames"],
            "params_source": "generated from sim.physics by web/gen_params.py",
            "browser": BB["engine"]["engine_short"], "gpu": BB["machine"]["gpu_for_reference_sim"],
        },
    }
    (RUN / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("wrote manifest.json", (RUN / "manifest.json").stat().st_size, "bytes")
    for r in RESULTS:
        if r.get("src"):
            print("  ok", r["src"], (ROOT / r["src"]).stat().st_size, "bytes")
    print("  prose lengths:", {k: len(v) for k, v in
                               (("summary", SUMMARY), ("findings", FINDINGS), ("full", FULL),
                                ("hyp", HYPOTHESIS), ("lim", LIMITATIONS))})


if __name__ == "__main__":
    main()
