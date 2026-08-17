"""Build the bespoke task page: a single self-contained HTML document with the live WebGPU sim in it.

Constraints from spec/style_task_page.md: sandboxed iframe (no CDN, no fetch, no same-origin), media
by absolute /api/data path, height reported automatically, dark theme, charts drawn from the numbers
in metrics.json rather than shipped as PNGs.

    .venv/Scripts/python.exe runs/material-variants/webgpu-port-of-the-interactive-simulation/web/build_page.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE.parent
MEDIA = "/api/data/learning-taichi/runs/material-variants/webgpu-port-of-the-interactive-simulation/"

M = json.loads((RUN / "metrics.json").read_text())

# ------------------------------------------------------------------ the data the page draws from
budget = M["particle_budget_60fps"]
floor = M["launch_floor_us"]
page_data = {
    "media": MEDIA,
    "device": M["device"],
    "spf": M["substeps_per_frame"],
    "nGrid": M["n_grid"],
    "physics": M["physics_version"],
    "tsQuantumNs": M["timestamp_quantum_ns"],
    "threeWay": M["three_way"],
    "budget": {k: budget[k] for k in ("webgpu", "javascript", "taichi_cuda")},
    "floor": floor,
    "phases": [{"n": r["n"], "grid": r["phase_ms"]["grid"], "p2g": r["phase_ms"]["p2g"],
                "g2p": r["phase_ms"]["g2p"], "encode": r["encode_ms"],
                "sustained": r["sustained_ms"], "ppc": r["particles_per_cell"]}
               for r in M["webgpu_scaling"]],
    "substepSweep": M["substep_sweep"],
    "atomics": M["atomics_head_to_head"],
    "accuracy": {
        k: {
            "desc": v["desc"], "band": v["band"],
            "variants": [{"variant": r["variant"], "atomics": r["atomics"], "kM": r["kM"],
                          "kV": r["kV"], "traj_rmse": r["traj_rmse"],
                          "vs_self_noise": r["vs_self_noise"],
                          "vs_perturbed_ic": r["vs_perturbed_ic"],
                          "final_frame_dist": r["final_frame_dist"],
                          "mass_ceiling": r["mass_saturates_at_pm"]}
                         for r in v["variants"]],
            "times": v["times"][::3],
            "curves": {r["variant"]: r["per_frame"][::3] for r in v["variants"]},
            "self_noise_curve": v["per_frame_self_noise"][::3],
        } for k, v in M["accuracy"].items()
    },
    "occupancy": json.loads((RUN / "verify" / "out" / "range.json").read_text())["occupancy"],
    "overflow": json.loads((RUN / "verify" / "out" / "range.json").read_text())["overflow"],
}

DEFS = {
    "traj_rmse": "Mean over frames and particles of the per-particle distance to ground truth. "
                 "THE NAME IS WRONG: it is a mean absolute distance, not an RMS, and not a "
                 "centre-of-mass distance. Units: domain lengths. Use it to rank, never to certify.",
    "self_noise": "The canonical simulator run against itself at the same configuration. Nonzero "
                  "only because GPU atomic scatter order varies between runs. It is the floor "
                  "nothing can beat.",
    "substeps_per_frame": "round((1/60)/dt) -- how many solver steps one real-time frame costs. "
                          "167 for canonical elastic; 333 for any scene containing snow.",
    "particles_per_cell": "Particle count divided by (seeded area x number of grid cells). Sets how "
                          "heavily loaded a grid node gets, which is what the fixed-point range has "
                          "to cover.",
}

CSS = """
:root{--bg:#0a0e14;--panel:#0e141d;--line:#1e2733;--fg:#dfe6ee;--muted:#7f8ea3;
      --accent:#6fd3ee;--warm:#ff9d5c;--bad:#ff7a7a;--good:#8fd9b6;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:960px;margin:0 auto;padding:4px 2px 30px}
h2{font-size:17px;margin:34px 0 4px;letter-spacing:-.01em}
h3{font-size:13px;margin:20px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;font-weight:600}
p{margin:8px 0}
.sub{color:var(--muted);font-size:13px;margin:2px 0 14px}
.verdict{background:linear-gradient(180deg,#101a24,#0d141c);border:1px solid #23343f;
         border-left:3px solid var(--accent);border-radius:10px;padding:14px 16px;margin:6px 0 4px}
.verdict b{color:#fff}
.warnline{color:var(--warm)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:10px;margin:16px 0 4px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:11px 13px}
.kpi .v{font:600 21px/1.15 ui-monospace,Menlo,Consolas,monospace;color:var(--accent);
        font-variant-numeric:tabular-nums}
.kpi .v.warm{color:var(--warm)} .kpi .v.bad{color:var(--bad)}
.kpi .k{color:var(--muted);font-size:11px;margin-top:4px;line-height:1.35}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 16px;margin:12px 0}
.toggle{display:inline-flex;gap:5px;background:#0b1119;border:1px solid var(--line);
        border-radius:8px;padding:4px;margin:4px 0 10px;flex-wrap:wrap}
.toggle button{background:transparent;color:var(--muted);border:0;border-radius:6px;
               padding:6px 12px;font:12.5px inherit;cursor:pointer}
.toggle button.on{background:#14313f;color:var(--accent)}
.toggle button:hover{color:var(--fg)}
svg{display:block;width:100%;height:auto;overflow:visible}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;
          vertical-align:middle}
table{border-collapse:collapse;width:100%;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace}
th,td{padding:6px 9px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
th:first-child,td:first-child{text-align:left}
td.bad{color:var(--bad)} td.good{color:var(--good)} td.warm{color:var(--warm)}
.def{border-bottom:1px dotted #3a4a5a;cursor:help}
video{width:100%;border-radius:9px;border:1px solid var(--line);background:#070a0f;display:block}
img.fig{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
.cap{color:var(--muted);font-size:12px;margin-top:7px;line-height:1.5}
.scope{border:1px dashed #3a4a5a;border-radius:9px;padding:12px 14px;color:var(--muted);
       font-size:12.5px;margin:14px 0}
.scope b{color:var(--fg)}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;color:var(--accent)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){.two{grid-template-columns:1fr}}
"""


def html():
    demo_css = (HERE / "demo.css").read_text(encoding="utf-8")
    params_js = (HERE / "params.js").read_text(encoding="utf-8")
    engine_js = (HERE / "mpm-webgpu.js").read_text(encoding="utf-8")
    demo_js = (HERE / "demo.js").read_text(encoding="utf-8")
    page_js = (HERE / "page.js").read_text(encoding="utf-8")

    wg = next(r for r in M["webgpu_scaling"] if r["n"] == 2048)
    a_launch = M["accuracy"]["launch"]
    k20 = next(r for r in a_launch["variants"] if r["variant"] == "fixed_k20")
    k24 = next(r for r in a_launch["variants"] if r["variant"] == "fixed_k24")

    return f"""<style>{CSS}
{demo_css}</style>
<div class="wrap">

<div class="verdict">
<b>Recording every substep into one command buffer is the whole difference.</b>
The same 128&times;128 elastic step that costs <b>345&nbsp;&micro;s per substep</b> in canonical
Taichi/CUDA &mdash; flat in particle count, because Python pays a ~56&nbsp;&micro;s kernel launch
167 times a frame &mdash; costs <b>{wg['us_per_substep_sustained']:.1f}&nbsp;&micro;s</b> in WebGPU,
and the 60&nbsp;fps particle budget goes from <b>{budget['javascript']:,.0f}</b> (one JavaScript
thread) and <b>zero</b> (Taichi/CUDA from Python never reaches 60&nbsp;fps at any particle count) to
<b>~{budget['webgpu']:,.0f}</b>.
<span class="warnline">What did not come free: WGSL has no atomic float add, and the obvious
fixed-point scale is wrong. At 2<sup>20</sup> quanta per particle mass the port lands
{k20['vs_perturbed_ic']:.0f}&times; outside canonical's own noise band on a bouncing disk &mdash;
visibly displaced, with nothing to warn you. 2<sup>24</sup> lands inside it.</span>
</div>

<div class="kpis">
  <div class="kpi"><div class="v">{wg['us_per_substep_sustained']:.1f} &micro;s</div>
    <div class="k">per substep, WebGPU, 2048 particles<br>(Taichi/CUDA: 378 &micro;s)</div></div>
  <div class="kpi"><div class="v">{floor['webgpu_empty_dispatch_in_recorded_buffer']:.2f} &micro;s</div>
    <div class="k">empty dispatch inside a recorded buffer<br>(empty CUDA launch from Python: {floor['taichi_cuda_empty_kernel_from_python']:.1f} &micro;s)</div></div>
  <div class="kpi"><div class="v">~{budget['webgpu']:,.0f}</div>
    <div class="k">particles at 60&nbsp;fps, real time<br>(JS port: {budget['javascript']:,.0f} &middot; Taichi/CUDA: none)</div></div>
  <div class="kpi"><div class="v warm">{k20['vs_perturbed_ic']:.0f}&times;</div>
    <div class="k">how far <span class="def" data-def="traj_rmse">traj_rmse</span> at
      2<sup>20</sup> sits outside the noise band<br>(at 2<sup>24</sup>: {k24['vs_perturbed_ic']:.1f}&times;)</div></div>
</div>

<h2>The thing itself</h2>
<p class="sub">Canonical elastic (<code>E=400</code>, <code>dt=1e-4</code>) on a
{M['n_grid']}&times;{M['n_grid']} grid, {M['substeps_per_frame']} substeps per frame,
{3 * M['substeps_per_frame']} compute dispatches recorded into <b>one</b> command buffer and
submitted once. Drag inside the box. The <i>grid mass</i> and <i>grid speed</i> views show the
background grid the solver actually solves on; the accumulator buttons switch between the two
fixed-point scales and the exact-f32 compare-and-swap path.</p>
<div class="card"><div id="demo"></div></div>

<h2>Where the 345 &micro;s went</h2>
<p class="sub">Nothing about the arithmetic changed. The 4090 was idle, waiting to be told what to
do, 668 times a frame. Both bars are an <b>empty</b> kernel &mdash; no work at all &mdash; measured
on this machine in this session.</p>
<div class="card"><div id="floorChart"></div></div>

<h2>Three implementations, one machine, one scene family</h2>
<p class="sub">Frame time for a real-time 60&nbsp;fps frame
(<span class="def" data-def="substeps_per_frame">{M['substeps_per_frame']} substeps</span>) against
particle count, all seeded into the same constant-density box. Press the toggle: the second view is
the same data read as "how many particles fits in the budget".</p>
<div class="card">
  <div class="toggle" id="costToggle">
    <button data-v="time" class="on">frame time vs particles</button>
    <button data-v="budget">particle budget at 60 fps</button>
  </div>
  <div id="costChart"></div>
  <div class="legend">
    <span><i style="background:#6fd3ee"></i>WebGPU compute</span>
    <span><i style="background:#ff9d5c"></i>JavaScript, one thread</span>
    <span><i style="background:#b58cf0"></i>Taichi / CUDA from Python</span>
    <span><i style="background:#4a5c6b"></i>60 fps budget (16.7 ms)</span>
  </div>
  <p class="cap">Taichi/CUDA is a horizontal line because its cost is launch overhead, not
  arithmetic &mdash; it is measuring an API usage pattern, not the device. Driven from a compiled
  host or through CUDA graphs the same kernels would look like the WebGPU curve; that is a
  <b>conjecture this task did not test</b>. The Taichi curve also stops at 16384 particles because
  that is <code>sim.physics.MAX_P</code>.</p>
</div>

<h2>What WGSL forced: no atomic float add</h2>
<p class="sub"><code>atomic&lt;f32&gt;</code> does not compile
(<i>"'atomic' only supports 'i32', 'u32' or 'vec2u'"</i>), and P2G scatters mass and momentum as
floats. Two routes were implemented and both were run against canonical ground truth on identical
initial conditions. <b>Flip the scene toggle</b> &mdash; the gentle drop scene forgives a coarse
scale and the bouncing disk does not.</p>
<div class="card">
  <div class="toggle" id="sceneToggle">
    <button data-v="drop" class="on">drop &mdash; released from rest</button>
    <button data-v="launch">launch &mdash; bounces and rolls</button>
  </div>
  <div id="accChart"></div>
  <div class="legend">
    <span><i style="background:#6fd3ee"></i>fixed point, 2<sup>k</sup> quanta per particle mass</span>
    <span><i style="background:#ff9d5c"></i>exact f32 via compare-and-swap</span>
    <span><i style="background:#2a3a4a"></i>canonical's own noise band</span>
  </div>
  <p class="cap">The band is what the canonical simulator does to <i>itself</i>: its lower edge is
  the same code re-run (GPU atomics reorder between runs) and its upper edge is the same code with
  the initial positions nudged by one f32 rounding unit. Anything inside is indistinguishable from
  chaos. <b>Note the exact-f32 point sits inside the band on both scenes but is not at zero</b> --
  a port that quantises nothing still diverges from canonical, because it dispatches in a different
  order and uses a closed-form polar decomposition instead of <code>ti.svd</code>.</p>
  <div id="accTable"></div>
</div>

<h3>The same thing as motion, with ground truth beside it</h3>
<div class="two">
  <div>
    <video controls loop muted playsinline preload="metadata" src="{MEDIA}launch_compare.mp4"></video>
    <p class="cap"><b>Launch scene.</b> Canonical, then WebGPU at 2<sup>20</sup>, then at
    2<sup>24</sup>. Canonical is ghosted in blue under each WebGPU panel and the divergence is
    plotted underneath as it happens. 2<sup>20</sup> separates from the ghost; 2<sup>24</sup> rides
    the noise band.</p>
  </div>
  <div>
    <video controls loop muted playsinline preload="metadata" src="{MEDIA}drop_compare.mp4"></video>
    <p class="cap"><b>Drop scene.</b> Same construction, with 2<sup>16</sup> as the coarse case
    &mdash; on this gentler scene 2<sup>20</sup> is already almost inside the band, which is exactly
    why testing one scene would have given the wrong answer.</p>
  </div>
</div>
<img class="fig" style="margin-top:14px" src="{MEDIA}launch_final_frames.png"
     alt="final frame of the launch scene at five fixed-point scales, canonical ghosted">
<p class="cap">Final frame of the launch scene at every scale tested, canonical ghosted in blue.
The displacement is a whole disk diameter at 2<sup>12</sup> and still plainly visible at
2<sup>20</sup>.</p>

<h2>The other half of the trade: range</h2>
<p class="sub">32 bits have to cover both the resolution the physics needs and the largest value a
node ever holds. Buying resolution spends range, and a saturating <code>u32</code> does not raise
an error &mdash; it wraps.</p>
<div class="card">
  <div id="occChart"></div>
  <p class="cap">Measured with the exact-f32 path so the measurement cannot itself saturate. The
  heaviest node carries roughly <b>2&times; the
  <span class="def" data-def="particles_per_cell">particles per cell</span></b>, in units of one
  particle mass. Read together with the accuracy result, that is the whole design rule:
  <b>k &ge; 22 for accuracy, 2<sup>32-k</sup> &gt; 2 &times; particles-per-cell for range.</b> The
  default shipped here is k=24, good to about 120 particles per cell.</p>
</div>
<img class="fig" src="{MEDIA}fixed_point_overflow.png"
     alt="deliberate fixed-point overflow: 2^30 wraps and the block explodes">
<p class="cap">Driving it over the ceiling on purpose. At 2<sup>30</sup> the accumulator holds only
4 particle masses per node while the scene needs 43; it wraps, and the block detonates. No NaN, no
error, no warning &mdash; just wrong physics. 2<sup>26</sup>, 2<sup>24</sup> and 2<sup>22</sup> all
land on the exact-f32 result to within 2e-6.</p>

<h2>Where the frame time actually goes</h2>
<div class="card">
  <div class="toggle" id="phaseToggle">
    <button data-v="abs" class="on">milliseconds</button>
    <button data-v="rel">share of the frame</button>
  </div>
  <div id="phaseChart"></div>
  <p class="cap">The grid sweep is <b>flat at ~0.27&nbsp;ms</b> for every particle count: 167
  dispatches over 16384 cells, almost entirely dispatch overhead. P2G &mdash; the atomic scatter
  &mdash; is the only part that grows, and it is what eventually ends the frame budget. CPU-side
  recording of all {3 * M['substeps_per_frame']} dispatches costs
  {min(r['encode_ms'] for r in M['webgpu_scaling']):.3f}&ndash;{max(r['encode_ms'] for r in M['webgpu_scaling'] if r['encode_ms'] < 0.2):.3f}&nbsp;ms,
  i.e. the CPU is no longer anywhere near the critical path.</p>
</div>

<div class="scope">
<b>Scope.</b> One material (elastic), one grid resolution ({M['n_grid']}&times;{M['n_grid']}), one
adapter ({M['device']['vendor']} / {M['device']['architecture']}), one browser, two scenes.
The accuracy conclusion is a statement about <i>these two scenes</i>: the coarse scale is fine on
one and clearly wrong on the other, which is itself the reason not to trust a single-scene check.
Nothing here was tested on the iPad or the M4, on snow/sand/fluid, or at any grid size but 128.
The 60&nbsp;fps particle budget is interpolated between two measured points and is measured with
<b>rendering excluded</b> &mdash; it is a compute budget, not a full frame budget. Timing came from
<code>timestamp-query</code> (quantum measured at {M['timestamp_quantum_ns']}&nbsp;ns on this
adapter) and from wall-clock totals over &ge;30 frames, never from a single
<code>performance.now()</code> interval.
</div>

</div>
<script>window.PAGE = {json.dumps(page_data)};
window.DEFS = {json.dumps(DEFS)};</script>
<script>{params_js}</script>
<script>{engine_js}</script>
<script>{demo_js}</script>
<script>{page_js}</script>
"""


def main():
    out = html()
    (RUN / "bespoke_page.html").write_text(out, encoding="utf-8")
    print("wrote", RUN / "bespoke_page.html", len(out), "bytes")


if __name__ == "__main__":
    main()
