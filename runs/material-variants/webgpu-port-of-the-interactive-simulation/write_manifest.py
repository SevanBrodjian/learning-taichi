"""Write manifest.json LAST, referencing only files that already exist on disk.

    .venv/Scripts/python.exe runs/material-variants/webgpu-port-of-the-interactive-simulation/write_manifest.py
"""
import datetime
import json
import pathlib

RUN = pathlib.Path(__file__).resolve().parent
REL = "runs/material-variants/webgpu-port-of-the-interactive-simulation/"
M = json.loads((RUN / "metrics.json").read_text())

b = M["particle_budget_60fps"]
fl = M["launch_floor_us"]
wg2k = next(r for r in M["webgpu_scaling"] if r["n"] == 2048)
tai = M["taichi_cuda"]["rows"]
tai2k = next(r for r in tai if r["n"] == 2048)
js2k = next(r for r in M["javascript"]["rows"] if r["n"] == 2048)
L = M["accuracy"]["launch"]
D = M["accuracy"]["drop"]


def var(sc, name):
    return next(r for r in M["accuracy"][sc]["variants"] if r["variant"] == name)


k20L, k22L, k24L, casL = (var("launch", v) for v in ("fixed_k20", "fixed_k22", "fixed_k24", "casf32"))
k20D, k24D, casD = (var("drop", v) for v in ("fixed_k20", "fixed_k24", "casf32"))
h2h = M["atomics_head_to_head"]
occ = M["density_probe"]

TLDR = ("Recording all 167 substeps into one WebGPU command buffer took the elastic step from "
        "345 to 7 microseconds and the 60 fps particle budget from ~1,200 to ~173,000, but the "
        "fixed-point atomics WGSL forces are only accurate at a scale four bits finer than the "
        "obvious choice -- and the obvious choice fails visibly only on the contact-heavy scene.")

SUMMARY = f"""
The canonical elastic MLS-MPM step now runs interactively in a browser on a WebGPU compute path, and
the measurement that motivated the task is confirmed and then removed. Canonical Taichi/CUDA costs
~345 us per substep **flat in particle count** because Python pays a {fl['taichi_cuda_empty_kernel_from_python']:.0f} us
kernel launch four times per substep, 167 times a frame. Recording every dispatch of every substep
into **one command buffer, submitted once per frame**, drops the empty-dispatch floor to
{fl['webgpu_empty_dispatch_in_recorded_buffer']:.2f} us -- a {fl['ratio']:.0f}x reduction on the same GPU -- and the
substep to {wg2k['us_per_substep_sustained']:.1f} us at 2048 particles. The cost curve stops being flat and starts
tracking the P2G scatter, which is what should dominate. Measured on one RTX 4090, compute only:
**~{b['webgpu']:,.0f} particles at 60 fps** against **{b['javascript']:,.0f}** for the single-threaded JS port and
**none at any particle count** for Taichi/CUDA driven from Python, which never gets under ~55 ms/frame.

The part that did not come free is the crux the brief predicted. WGSL has no atomic float add, so P2G
accumulates in fixed point, and the scale is not a free parameter. At 2^20 quanta per particle mass --
the obvious first guess -- the port sits **{k20L['vs_perturbed_ic']:.0f}x outside canonical's own noise band** on a
disk that bounces and rolls, with the body visibly displaced from ground truth. On the gentler drop
scene the same scale is only {k20D['vs_perturbed_ic']:.1f}x out, so a single-scene check would have passed it. 2^24
lands inside the band on both scenes and matches an exact-f32 compare-and-swap accumulator. That
resolution costs range: a u32 then saturates at 256 particle masses per node, the heaviest node
carries about twice the particles-per-cell, and an overrun **wraps silently** -- driven over the
ceiling on purpose, the block detonates with no NaN and no error.
""".strip()

FULL = f"""
## What was built

`web/mpm-webgpu.js` is a self-contained WebGPU compute port of the canonical elastic MLS-MPM step: no
dependency on the dashboard, the data server or the harness, no CDN and no fetch. Parameters are
generated from `sim.physics` by `web/gen_params.py` and stamped ({M['physics_version']}); **every elastic
constant was verified byte-identical to the JS port's** after the sand promotion bumped the version
(`verify/baselines.js` asserts this at run time and exits non-zero if any constant moves).

**The structure.** One `GPUCommandEncoder` per frame, one compute pass, and inside it
{M['substeps_per_frame']} x 3 = {3 * M['substeps_per_frame']} `dispatchWorkgroups` calls. WebGPU orders dispatches within a pass and
makes each one's writes visible to the next, which is exactly the P2G -> grid -> G2P dependency, so
the chain needs no explicit barriers and never returns to the CPU. Recording all of that costs
{min(r['encode_ms'] for r in M['webgpu_scaling']):.3f}-{max(r['encode_ms'] for r in M['webgpu_scaling'] if r['encode_ms'] < 0.2):.3f} ms of JS per frame, i.e. the CPU is nowhere near the critical path.

**What WebGPU forced, beyond the atomics.**
1. *Eight storage buffers per shader stage* is the baseline limit, and the natural layout (pos, vel, C,
   F, mass, momX, momY, gridV, display) is nine. Exceeding it produces an **invalid bind group**, which
   makes every dispatch a silent no-op: the first run of this harness "achieved" 0.44 ms/frame flat from
   500 to 262144 particles and wrote out trajectories of pure zeros. The fix was to interleave momentum
   x/y into one buffer and let the grid-velocity buffer double as the display buffer (seven), and to
   attach an `uncapturederror` listener plus validation error scopes so this can never be silent again.
2. *No `ti.svd`*: R comes from the closed-form 2D polar rotation, exactly as the JS port does. Not an
   approximation -- Taichi's own svd2d is built on the same rotation and the elastic path never uses the
   singular values.
3. *Dense grid sweep*, unlike the JS port's sparse active-cell list: on a GPU the dense sweep is free and
   a sparse list would need a compaction pass.
4. *The clear is fused into the grid update*, which is the only reader of the accumulators, removing a
   whole 16384-cell dispatch per substep (4 per substep -> 3).

## Performance, measured

Timing came from `timestamp-query` (quantum measured at {M['timestamp_quantum_ns']} ns on this adapter, so
per-pass resolution is real) and from wall-clock totals over >= 30 frames. `performance.now()` was never
used for a single interval, and **nothing load-bearing depends on `requestAnimationFrame`** -- sustained
throughput is F frames submitted back to back with one `onSubmittedWorkDone` await.

| | empty launch/dispatch | substep @2048 | frame @2048 | 60 fps budget |
|---|---|---|---|---|
| Taichi / CUDA, per-kernel from Python | {fl['taichi_cuda_empty_kernel_from_python']:.1f} us | {tai2k['us_per_substep']:.0f} us | {tai2k['frame_ms']:.1f} ms | none at any n |
| JavaScript, one thread | n/a | {js2k['us_per_substep']:.0f} us | {js2k['frame_ms']:.1f} ms | {b['javascript']:,.0f} |
| WebGPU, one command buffer/frame | {fl['webgpu_empty_dispatch_in_recorded_buffer']:.2f} us | {wg2k['us_per_substep_sustained']:.1f} us | {wg2k['sustained_ms']:.2f} ms | ~{b['webgpu']:,.0f} |

The Taichi row is flat from 500 to 16384 particles ({min(r['us_per_substep'] for r in tai if r['us_per_substep']):.0f}-{max(r['us_per_substep'] for r in tai if r['us_per_substep']):.0f} us) and stops
there because that is `sim.physics.MAX_P`. JavaScript crosses above Taichi at roughly 3600 particles.

**Where the frame goes.** The grid sweep is flat at ~0.27 ms for every particle count -- {M['substeps_per_frame']}
dispatches over 16384 cells, almost entirely dispatch overhead at 1.6 us each against the 1.11 us floor.
P2G is the only phase that grows and is what ends the budget (0.37 ms at n=500, 19.6 ms at n=262144).
G2P grows more slowly because a gather has no contention.

**Substep count is still the wall.** At 16384 particles a 167-substep frame costs 2.58 ms and a
333-substep frame (what any scene containing snow is forced to) costs 5.20 ms. Batching did not repeal
the budget equation; it changed what one substep costs.

## Accuracy, measured against canonical on identical initial conditions

Protocol as the JS port used it: `traj_rmse` against `sim.physics.simulate`, read against the
simulator's own noise band -- lower edge = canonical re-run against itself (GPU atomics reorder),
upper edge = canonical with the initial positions nudged by 1e-7, about one f32 ULP at these
coordinates. Two scenes, 2048 particles, 150 frames to t = 2.5 s.

| accumulator | drop: traj_rmse (x band) | launch: traj_rmse (x band) | node ceiling |
|---|---|---|---|
| fixed 2^12 | {var('drop','fixed_k12')['traj_rmse']:.3g} ({var('drop','fixed_k12')['vs_perturbed_ic']:.0f}x) | {var('launch','fixed_k12')['traj_rmse']:.3g} ({var('launch','fixed_k12')['vs_perturbed_ic']:.0f}x) | 1048576 pm |
| fixed 2^16 | {var('drop','fixed_k16')['traj_rmse']:.3g} ({var('drop','fixed_k16')['vs_perturbed_ic']:.0f}x) | {var('launch','fixed_k16')['traj_rmse']:.3g} ({var('launch','fixed_k16')['vs_perturbed_ic']:.0f}x) | 65536 pm |
| fixed 2^20 | {k20D['traj_rmse']:.3g} ({k20D['vs_perturbed_ic']:.1f}x) | {k20L['traj_rmse']:.3g} ({k20L['vs_perturbed_ic']:.0f}x) | 4096 pm |
| fixed 2^22 | {var('drop','fixed_k22')['traj_rmse']:.3g} ({var('drop','fixed_k22')['vs_perturbed_ic']:.1f}x) | {k22L['traj_rmse']:.3g} ({k22L['vs_perturbed_ic']:.1f}x) | 1024 pm |
| **fixed 2^24** | {k24D['traj_rmse']:.3g} ({k24D['vs_perturbed_ic']:.1f}x) | {k24L['traj_rmse']:.3g} ({k24L['vs_perturbed_ic']:.1f}x) | 256 pm |
| fixed 2^26 | {var('drop','fixed_k26')['traj_rmse']:.3g} ({var('drop','fixed_k26')['vs_perturbed_ic']:.1f}x) | {var('launch','fixed_k26')['traj_rmse']:.3g} ({var('launch','fixed_k26')['vs_perturbed_ic']:.1f}x) | 64 pm |
| exact f32 (CAS loop) | {casD['traj_rmse']:.3g} ({casD['vs_perturbed_ic']:.1f}x) | {casL['traj_rmse']:.3g} ({casL['vs_perturbed_ic']:.1f}x) | none |

Bands: drop self-noise {D['band']['self_noise']:.3g}, one-ULP nudge {D['band']['perturbed_ic_1e-7']:.3g};
launch self-noise {L['band']['self_noise']:.3g}, one-ULP nudge {L['band']['perturbed_ic_1e-7']:.3g}.

Three things to read off this, kept separate from their interpretation:

* **The exact-f32 control does not score zero.** It sits at {casL['vs_perturbed_ic']:.1f}x the band on the launch scene
  and {casD['vs_perturbed_ic']:.1f}x on the drop. A port that quantises nothing still diverges, because the dispatch
  order differs from Taichi's and the polar rotation replaces `ti.svd`. Everything at or below that
  level is a statement about chaos, not about the accumulator.
* **Once inside the band the ordering is meaningless.** 2^24 scores better than 2^26 and than exact f32
  on the launch scene. That is which side of the butterfly each landed on, not a precision ranking, and
  it should not be read as one.
* **The scene selects the answer.** 2^20 is {k20D['vs_perturbed_ic']:.1f}x out on the drop and {k20L['vs_perturbed_ic']:.0f}x out on the
  launch. Validating on the settling scene alone would have shipped it.

The default in `params.js` was therefore changed from 2^20/2^18 (the pre-registered guess) to
**2^24/2^22** on the strength of this sweep.

## The range half of the fixed-point budget

Resolution is bought from the same 32 bits as range, so the sweep above has a second axis. Node
occupancy was measured with the exact-f32 path (so the measurement cannot itself saturate) across a
sixty-fold density range: the heaviest node carries **{occ[0]['max_node_mass_pm'] / occ[0]['particles_per_cell']:.1f}x to 2x the particles-per-cell**, in
particle masses, and the relation is close to linear. At 2^24 the u32 ceiling is 256 particle masses,
i.e. good to roughly 120 particles per cell.

Driven past it deliberately (32000 particles at 22 per cell, heaviest node 43 pm), 2^30 -- ceiling 4 pm
-- wraps and the block explodes to a mean radius of 0.47 against 0.11 for every other variant, with
**zero non-finite values and no error raised**. 2^26, 2^24 and 2^22 all land on the exact-f32 result to
within 2e-6. The failure signature is that it depends on density rather than timestep, which is what
distinguishes it from ordinary instability.

## Fixed point vs exact f32, cost

| n | particles/cell | fixed (GPU ms/frame) | CAS f32 | slowdown |
|---|---|---|---|---|
""" + "\n".join(
    f"| {r['n']} | {r['particles_per_cell']:.1f} | {r['fixed_gpu_ms']:.2f} | {r['casf32_gpu_ms']:.2f} | {r['slowdown_total']:.1f}x |"
    for r in h2h) + f"""

On the P2G+grid phases alone the CAS loop is {min(r['slowdown_p2g_grid'] for r in h2h):.1f}-{max(r['slowdown_p2g_grid'] for r in h2h):.1f}x slower. That is a real cost
but not a prohibitive one, and it is worth stating plainly: on this hardware, at demo particle counts,
**taking the exact path and paying 2-4x would have been a defensible engineering choice**, and it
removes the resolution/range problem and the silent-wrap failure mode entirely.

## Method notes and traps hit

* **The first harness run was a complete fiction** and looked like a triumph: 0.44 ms/frame, flat from
  500 to 262144 particles. WebGPU errors are asynchronous and silent by default; a bind group that
  exceeded `maxStorageBuffersPerShaderStage` (8) invalidated every command buffer, so nothing ran. The
  giveaway was the shape of the curve, not any error. Error scopes and an `uncapturederror` listener are
  now permanent.
* **Seeding decides what a scaling curve measures.** Growing a disk's radius as sqrt(n) to hold density
  constant leaves the unit domain past ~35k particles; everything outside is clamped onto the boundary
  cells and P2G then measures a pathological pile-up (121 ms at 262144, vs 19.6 ms once seeded as a
  capped box). The published curve uses a constant-density box and reports particles-per-cell.
* **`timestamp-query` was not quantised** on this adapter/browser (32 ns granularity), contrary to the
  100 us clamp expected from Chromium. It was measured rather than assumed.
* Fixed-point conversion uses `round()`, not truncation: truncation biases a signed momentum toward
  zero, which is a systematic drag rather than noise.
""".strip()

HYPOTHESIS = """
(Hypothesis, not observation.) The mechanism behind the speedup is that the previously measured cost was
per-SUBMISSION, not per unit of work: a Python->CUDA launch crosses a driver boundary and synchronises
bookkeeping, whereas a dispatch already recorded into a command buffer is a few words the GPU's front end
reads. That predicts the same ~50x floor reduction would appear for the *same* CUDA kernels driven through
CUDA graphs or a compiled host, i.e. **this is not a WebGPU result, it is a batched-submission result** --
which this task did not test and which would be the cheapest way to confirm the mechanism.

For the atomics, the hypothesised mechanism is that fixed-point error enters as a rounding of each of the
27 per-particle contributions, so the induced velocity error is ~quantum x sqrt(contributions) / node mass.
That predicts (a) the requirement is one quantum <~ one f32 ULP of a typical node value, which is what
2^-24 particle masses works out to, and (b) accuracy should *improve* with density, because node values
grow while the quantum does not -- consistent with the observation that at 22 particles/cell all of
2^22/2^24/2^26 agree with exact f32 to 2e-6, where at 3.3 particles/cell they differ. The reason the
launch scene is so much less forgiving than the drop scene is hypothesised to be Coulomb friction: it is a
branch on the sign of the tangential velocity, so a quantised velocity can take the other side of the
branch and the error becomes a discrete change rather than a small perturbation. Testing that would mean
re-running the sweep with friction disabled and with a frictionless separating floor.
""".strip()

LIMITATIONS = f"""
One material (elastic), one grid resolution ({M['n_grid']}x{M['n_grid']}), one adapter
({M['device']['vendor']}/{M['device']['architecture']}), one browser, two scenes, 2048 particles for
every accuracy number. Specifically NOT tested:

* **Any other material.** Fluid, snow and sand are untouched. Snow's plastic clamp and sand's return
  mapping both involve an SVD in G2P, which this port does not implement at all -- only the closed-form
  polar rotation the elastic path needs. Extending to the four-material demo is real work, not a
  parameter change.
* **Any other device.** The brief states WebGPU is confirmed present on the iPad and the M4; nothing here
  ran on either. Mobile GPUs have very different atomic throughput, and the fixed-point/CAS cost ratio in
  particular should not be assumed to transfer.
* **Rendering.** Every frame-time and particle-budget number is compute only. The demo draws from the
  same buffers with no readback, but the draw was not included in the budget, so the real 60 fps particle
  count is lower than ~{b['webgpu']:,.0f} by an unmeasured amount.
* **The 60 fps budget is interpolated**, not measured at the crossing, and it is measured at rising
  density (about 10-15 particles per cell at the crossing, against 3.3 for the reference scene).
* **The claim that Taichi/CUDA would match WebGPU if batched** is a conjecture; CUDA graphs were not tried.
* The accuracy conclusion is bounded to these two scenes. "2^24 is enough" is a hypothesis about elastic
  MLS-MPM at this density, supported by two scenes, one of which was chosen to be hard. It is not a
  general statement about fixed-point P2G.
* Scenes are still specified per task rather than centrally (`spec/registry/README.md` names this as a
  known gap), so the drop scene matches `phys.scene("drop")` but the launch and box scenes are defined in
  `verify/prepare.py` and `verify/harness.html`.
""".strip()

results = [
    {"type": "video", "src": REL + "launch_compare.mp4",
     "caption": "Launch scene as motion: canonical sim.physics, then WebGPU at 2^20, then at 2^24, all "
                "on the identical initial condition. Canonical is ghosted in blue under each WebGPU "
                "panel and the divergence is plotted underneath against canonical's own noise band. "
                "2^20 separates visibly from the ghost; 2^24 rides the band."},
    {"type": "video", "src": REL + "drop_compare.mp4",
     "caption": "Drop scene, same construction with 2^16 as the coarse case. On this gentler scene the "
                "coarse scale is far less obviously wrong -- which is why the launch scene exists."},
    {"type": "image", "src": REL + "launch_final_frames.png",
     "caption": "Final frame of the launch scene at every scale tested, canonical ghosted in blue. The "
                "displacement is a whole disk diameter at 2^12 and still plainly visible at 2^20."},
    {"type": "image", "src": REL + "fixed_point_overflow.png",
     "caption": "The range half of the trade, driven over the ceiling on purpose. At 2^30 the u32 holds "
                "only 4 particle masses per node while the scene needs 43; it wraps and the block "
                "detonates -- no NaN, no error, no warning. 2^26/2^24/2^22 land on exact f32."},
    {"type": "table",
     "columns": ["implementation", "empty launch (us)", "us/substep @2048", "ms/frame @2048",
                 "particles at 60 fps"],
     "rows": [
         ["Taichi / CUDA, per-kernel from Python",
          f"{fl['taichi_cuda_empty_kernel_from_python']:.1f}", f"{tai2k['us_per_substep']:.0f}",
          f"{tai2k['frame_ms']:.1f}", "none at any n"],
         ["JavaScript, single thread", "n/a", f"{js2k['us_per_substep']:.0f}",
          f"{js2k['frame_ms']:.1f}", f"{b['javascript']:,.0f}"],
         ["WebGPU, one command buffer per frame",
          f"{fl['webgpu_empty_dispatch_in_recorded_buffer']:.2f}",
          f"{wg2k['us_per_substep_sustained']:.1f}", f"{wg2k['sustained_ms']:.2f}",
          f"~{b['webgpu']:,.0f}"],
     ],
     "caption": "Three implementations, one RTX 4090, one session, matched constant-density scenes. "
                "Compute only, rendering excluded. Canonical elastic, 167 substeps per frame."},
    {"type": "table",
     "columns": ["accumulator", "drop traj_rmse", "x band", "launch traj_rmse", "x band",
                 "node ceiling (particle masses)"],
     "rows": [[("fixed 2^%d" % v["kM"]) if v["atomics"] == "fixed" else "exact f32 (CAS)",
               f"{var('drop', v['variant'])['traj_rmse']:.3g}",
               f"{var('drop', v['variant'])['vs_perturbed_ic']:.1f}",
               f"{var('launch', v['variant'])['traj_rmse']:.3g}",
               f"{var('launch', v['variant'])['vs_perturbed_ic']:.1f}",
               f"{v['mass_saturates_at_pm']:.0f}" if v["atomics"] == "fixed" else "-"]
              for v in M["accuracy"]["launch"]["variants"]],
     "caption": "traj_rmse against canonical, and how many times canonical's own one-ULP band that is. "
                "Below ~1 is indistinguishable from chaos. Note the exact-f32 control is not at zero, "
                "and that the two scenes disagree about whether 2^20 is acceptable."},
    {"type": "plot", "kind": "loss", "series": REL + "metrics.json", "log": True,
     "caption": "Full metrics: per-frame divergence curves for every variant and scene, the scaling and "
                "phase-split benchmarks, the node-occupancy sweep, and both baselines."},
]

manifest = {
    "schema_version": "2",
    "task_id": "webgpu-port-of-the-interactive-simulation",
    "direction": "material-variants",
    "title": "WebGPU port of the interactive simulation",
    "tldr": TLDR,
    "status": "active",
    "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "physics_version": M["physics_version"],
    "objective": ("Port the canonical elastic MLS-MPM step to a WebGPU compute path that records every "
                  "substep into one command buffer, and measure it against the single-threaded JS port "
                  "and the Taichi/CUDA reference on the same machine and the same physics -- including "
                  "what the fixed-point atomics WGSL forces cost numerically."),
    "summary": SUMMARY,
    "findings": SUMMARY,
    "full_report": FULL,
    "hypothesis": HYPOTHESIS,
    "limitations": LIMITATIONS,
    "results": results,
    "custom_html": (RUN / "bespoke_page.html").read_text(encoding="utf-8"),
    "training_refs": ["real-time-cost", "fixed-point-atomics", "math-toolkit"],
    "metrics_used": ["traj_rmse", "self_noise", "substeps_per_frame", "us_per_substep", "frame_ms",
                     "particle_budget_60fps", "dispatch_floor_us"],
    "device": M["device"],
    "code": {
        "engine": REL + "web/mpm-webgpu.js",
        "demo": REL + "web/demo.js",
        "standalone_demo": REL + "web/demo.html",
        "params_generator": REL + "web/gen_params.py",
        "verification": REL + "verify/",
    },
}

missing = [r["src"] for r in results if r.get("src")
           and not (RUN.parents[2] / r["src"]).exists()]
missing += [r["series"] for r in results if r.get("series")
            and not (RUN.parents[2] / r["series"]).exists()]
if missing:
    raise SystemExit("DANGLING MEDIA: " + ", ".join(missing))

(RUN / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("wrote manifest.json;", len(results), "results, all srcs verified on disk")
