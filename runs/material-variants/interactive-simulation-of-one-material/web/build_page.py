"""Assemble bespoke_page.html: the live demo plus the evidence, fully self-contained.

Everything is inlined -- CSS, the port, the demo, the page script, and the numbers pulled straight
out of metrics.json / browser_bench.json / verify/gpu_bench.json. Only the videos are referenced by
absolute /api/data path, which is what the dashboard serves.

    .venv/Scripts/python.exe runs/material-variants/interactive-simulation-of-one-material/web/build_page.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[3]

M = json.loads((RUN / "metrics.json").read_text())
BB = json.loads((RUN / "browser_bench.json").read_text())
GB = json.loads((RUN / "verify" / "gpu_bench.json").read_text())
REG = json.loads((ROOT / "spec" / "registry" / "metrics.json").read_text())

MEDIA = "/api/data/learning-taichi/runs/material-variants/interactive-simulation-of-one-material/"
SPF = 167
BUDGET_US = (1 / 60) / SPF * 1e6


def r3(xs):
    return [float("%.4g" % v) for v in xs]


def scene_blob(src, caption):
    pf = src["per_frame"]
    return {
        "times": r3(pf["times"]),
        "port_vs_canonical": r3(pf["port_vs_canonical"]),
        "canonical_self_noise": r3(pf["canonical_self_noise"]),
        "canonical_perturbed_ic": r3(pf["canonical_perturbed_ic"]),
        "traj_rmse": {k: float("%.4g" % v) for k, v in src["traj_rmse"].items()},
        "dt_sweep": [{"mult": e["mult"], "dt": e["dt"], "spf": e["spf"],
                      "speedup": e["speedup_vs_canonical"], "finite": e["finite"],
                      "rmse": (e["traj_rmse_vs_canonical"] if e["finite"] else None)}
                     for e in src["dt_sweep"]],
        "caption": caption,
    }


bench = [{"n": r["n"], "sparse": r["sparse"]["us_per_step"], "dense": r["dense"]["us_per_step"]}
         for r in BB["cpu_sweep"]]
gpu = [{"n": r["n"], "us": r["us_per_step"]} for r in GB["per_n"]]
budget_particles = 1154

DATA = {
    "media": MEDIA,
    "spf": SPF,
    "budget_us": BUDGET_US,
    "budget_particles": budget_particles,
    "bench": bench,
    "gpu": gpu,
    "defs": {k: REG[k] for k in ("traj_rmse", "physics_version") if k in REG},
    "scenes": {
        "drop": scene_blob(M, "Canonical drop scene: a 2048-particle elastic disk released at rest. "
                              "The two panels are the same initial condition run by the two "
                              "implementations; the chart is how far apart their particles are."),
        "launch": scene_blob(M["launch_scene"],
                             "Launched disk: the same material given a sideways kick so it stays in "
                             "contact with the floor for the whole rollout, which is the branch a "
                             "port is most likely to get subtly wrong."),
    },
}

BODY = """
<div class="pg">

<p class="verdict">
The elastic material runs <b>in the browser</b>, interactively, from a single-threaded JavaScript port
of the canonical MLS-MPM step, and it is <span class="win">numerically exact</span>: over a 2.5&nbsp;s
rollout the port sits <dfn data-def="traj_rmse">1.7&times;10<sup>-4</sup></dfn> from the reference,
which is <em>less</em> than the reference moves when its own starting positions are nudged by one f32
rounding unit. The port did not turn out to be the hard part. <b>The timestep is.</b>
Elastic at <span class="mth">E&nbsp;=&nbsp;400</span> needs &Delta;t&nbsp;=&nbsp;1e-4, so real time costs
<b>167&nbsp;substeps every frame</b>, and that single fact caps this machine at
<b>~1150&nbsp;particles at 60&nbsp;fps</b>. Raising &Delta;t to buy frame rate
<span class="bad">destroys the physics</span> long before it becomes unstable.
</p>

<div class="sec first">
<h2><span class="n">01</span> The thing itself &mdash; drag it</h2>
<div id="demo"></div>
<p class="cap">Elastic disk, canonical parameters, 128&times;128 grid, 167 substeps per frame. Sim speed
is the honest number: frames per second &times; substeps per frame &times; &Delta;t. Push the particle
slider until it reads 1.00&times; and that is this machine's 60&nbsp;fps budget. The other two switches are
the finding: <b>grid loop</b> swaps the sparse active-cell sweep for Taichi's dense one (identical physics,
bit for bit) and <b>&Delta;t</b> makes it cheaper and then wrecks it.</p>
</div>

<div class="sec">
<h2><span class="n">02</span> Does the port actually match the reference?</h2>
<div class="tabs" data-k="scene">
  <button data-v="launch" class="on">launched disk (contact-heavy)</button>
  <button data-v="drop">canonical drop</button>
</div>
<div class="cols">
  <div class="a">
    <video id="divvid" autoplay loop muted playsinline></video>
    <p class="cap" id="divcap"></p>
  </div>
  <div class="b">
    <div id="divchart" class="panel" style="padding:8px"></div>
    <div class="legend">
      <span><s style="background:#ff9d5c"></s>port vs canonical</span>
      <span><s style="background:#c58cf0"></s>canonical vs itself</span>
      <span><s style="background:#5fd39a"></s>canonical, start nudged 1e-7</span>
    </div>
    <table style="margin-top:12px"><thead><tr><th>whole-rollout <dfn data-def="traj_rmse">traj_rmse</dfn></th><th>value</th></tr></thead>
    <tbody id="divnums"></tbody></table>
    <p class="cap">Two runs of the reference simulator on identical input already disagree, because P2G
    scatters through GPU atomics in a nondeterministic order. That disagreement is the floor. The port's
    line lives in the same band, so there is no room left to claim it is wrong.</p>
  </div>
</div>
<div class="scope"><b>What this does and does not show.</b> One material (elastic), two scenes, one
particle count, 2.5&nbsp;s. It shows the port reproduces the reference's dynamics to within the
reference's own reproducibility. It does <b>not</b> show that fluid or snow would port as cleanly &mdash;
both have branches (the plastic clamp, the pressure term) this port never touches.</div>
</div>

<div class="sec">
<h2><span class="n">03</span> Where the frame budget actually goes</h2>
<div class="tabs" data-k="bmode">
  <button data-v="us" class="on">microseconds per substep</button>
  <button data-v="fps">sustained fps at real time</button>
</div>
<div class="cols">
  <div class="a"><div id="budchart" class="panel" style="padding:8px"></div></div>
  <div class="b">
  <div class="kpi">
    <div class="k"><div class="lab">60 fps budget</div><div class="v">1154 <small>particles</small></div>
      <div class="sub">where the port crosses real time</div></div>
    <div class="k"><div class="lab">substeps / frame</div><div class="v">167</div>
      <div class="sub">forced by &Delta;t = 1e-4</div></div>
  </div>
  <p class="note">Flip to <b>fps</b> and only one curve gets above the line. Three things fall out of the
  same chart:<br><br>
  <b>1.</b> Taichi's dense grid sweep costs about <b>93&nbsp;&micro;s</b> per substep on its own &mdash;
  <b>15.6&nbsp;ms</b> per frame, essentially the whole 60&nbsp;fps budget, <i>before any particle is
  touched</i>. Only ~760 of the 16&nbsp;384 cells ever hold material.<br><br>
  <b>2.</b> Below roughly 4300 particles, one JavaScript thread is <b>faster than the project's own CUDA
  reference on an RTX&nbsp;4090</b>. The reference is flat at ~345&nbsp;&micro;s from 500 to 16&nbsp;384
  particles, which is what launch-bound looks like.<br><br>
  <b>3.</b> P2G and G2P split the rest almost evenly (86 and 73&nbsp;&micro;s at 2000 particles); the grid
  update is 5&nbsp;&micro;s and drawing is 0.15&nbsp;ms per frame. Rendering is not the problem.</p>
  </div>
</div>
</div>

<div class="sec">
<h2><span class="n">04</span> The trap: &Delta;t is not a performance knob</h2>
<div class="cols">
  <div class="a">
    <video id="dtvid" autoplay loop muted playsinline></video>
    <p class="cap">Cyan is the canonical run in all three panels; orange is the cheaper timestep. At
    &times;2 and &times;4 the ball rolls visibly further than the truth and ends a full diameter away.
    On the drop scene the &times;4 run does not merely drift, it goes non-finite.</p>
  </div>
  <div class="b">
    <table id="dttable"><thead><tr><th>&Delta;t</th><th>substeps</th><th>speedup</th>
      <th><dfn data-def="traj_rmse">traj_rmse</dfn></th><th></th></tr></thead><tbody></tbody></table>
    <p class="note" style="margin-top:12px">Buying a 1.6&times; speedup costs three to four orders of
    magnitude of accuracy. The elastic wave speed is
    <span class="mth">c = &radic;((&lambda;+2&mu;)/&rho;) &asymp; 21</span> domain lengths per second, and a
    grid cell is <span class="mth">1/128</span>, so the CFL limit is
    <span class="mth">&Delta;x/c &asymp; 3.7&times;10<sup>-4</sup></span>. The canonical
    <span class="mth">&Delta;t = 10<sup>-4</sup></span> sits at a CFL number of 0.27, and the sweep
    runs straight into the wall.
    <b>Stiffness sets the timestep, the timestep sets the substep count, and the substep count is the
    whole cost.</b></p>
  </div>
</div>
</div>

<div class="sec">
<h2><span class="n">05</span> Analytic equations, or a network that learns the grid update?</h2>
<div class="panel">
<p class="note" style="max-width:none">Analytic had to come first &mdash; without a correct reference
there is nothing to validate a learned update against. With the reference in hand the learned option can
be <b>priced</b> rather than guessed. Both numbers below are measured on the same machine, on the same
~760 active cells, against the analytic grid update they would replace.</p>
<table style="margin-top:14px"><thead><tr><th>grid update</th><th>where</th><th>per substep</th>
<th>&times; analytic</th><th>at 167 substeps/frame</th></tr></thead><tbody>
<tr><td>analytic (the equations)</td><td class="mut">browser JS</td><td class="num">5.2 &micro;s</td><td class="num good">1&times;</td><td class="num good">0.9 ms</td></tr>
<tr><td>MLP 8-32-32-2</td><td class="mut">browser JS</td><td class="num">1258 &micro;s</td><td class="num bad">242&times;</td><td class="num bad">210 ms &rarr; 4.8 fps</td></tr>
<tr><td>MLP 8-64-64-2</td><td class="mut">browser JS</td><td class="num">3900 &micro;s</td><td class="num bad">750&times;</td><td class="num bad">651 ms &rarr; 1.5 fps</td></tr>
<tr><td>analytic (the equations)</td><td class="mut">CUDA, RTX 4090</td><td class="num">84 &micro;s</td><td class="num">1&times;</td><td class="num warn">14.0 ms</td></tr>
<tr><td>MLP 8-32-32-2, all 16384 cells</td><td class="mut">CUDA, RTX 4090</td><td class="num">87 &micro;s</td><td class="num good">1.03&times;</td><td class="num warn">14.5 ms</td></tr>
</tbody></table>
<div class="reg">
  <div class="r obs"><div class="h">observed</div>In JavaScript the smallest useful network is 242&times;
  the analytic grid update and blows the frame budget by 13&times;. On the GPU the identical network is
  free relative to the analytic update (87 vs 84&nbsp;&micro;s) because at 16&nbsp;384 cells both are
  dominated by kernel launch, not arithmetic &mdash; an empty kernel already costs 56&nbsp;&micro;s.</div>
  <div class="r hyp"><div class="h">hypothesised</div>A learned grid update can only pay for itself if it
  buys a <b>larger &Delta;t</b>, because &Delta;t is what sets the 167 substeps and the substeps are the
  whole cost. Making the per-substep update smarter is optimising the term that is already cheap.
  Labelled a conjecture: nothing here trained such a model.</div>
  <div class="r tst"><div class="h">would test it</div>Train a network on <i>coarse-time</i> transitions
  &mdash; map the state at <span class="mth">t</span> to the state at
  <span class="mth">t&nbsp;+&nbsp;16.7&nbsp;ms</span> directly, not one substep &mdash; and
  measure both its <dfn data-def="traj_rmse">traj_rmse</dfn> against a canonical rollout and its cost per
  frame. If it holds accuracy at one evaluation per frame instead of 167, it wins by two orders of
  magnitude. If it needs substeps, it cannot win at all.</div>
</div>
</div>
<div class="scope"><b>Scope.</b> WebGPU was not available in the browser this was measured in, so the
browser numbers are all CPU. The CUDA figures come from Taichi kernels launched from Python, which is
how the canonical simulator is actually run in this project; a fused or graph-captured implementation
would pay far less launch overhead. The 4300-particle crossover is a statement about
<b>this</b> reference implementation on <b>this</b> machine, not about GPUs.</div>
</div>

<div class="sec">
<h2><span class="n">06</span> How the port was done</h2>
<div class="panel">
<table><thead><tr><th>piece</th><th>what happened to it</th></tr></thead><tbody>
<tr><td>parameters (<span class="mth">E</span>, &Delta;t, grid, &nu;, friction, gravity)</td><td style="text-align:left">
  Not retyped. <code>gen_params.py</code> imports <code>sim.physics</code> and emits
  <code>params.js</code>, stamped with the physics version.</td></tr>
<tr><td>constitutive law</td><td style="text-align:left">Unchanged. Fixed corotated,
  <span class="mth">2&mu;(<b>F</b>&minus;<b>R</b>)<b>F</b><sup>T</sup> + &lambda;(J&minus;1)J<b>I</b></span>.</td></tr>
<tr><td><code>ti.svd</code></td><td style="text-align:left">Deleted. In 2D the elastic path only needs
  the polar rotation <span class="mth"><b>R</b> = <b>UV</b><sup>T</sup></span>, and Taichi's own 2D SVD is built on a
  closed-form polar decomposition, so the singular values were never used. Two adds, a hypot and a
  reciprocal square root replace a whole factorisation.</td></tr>
<tr><td>the grid loop</td><td style="text-align:left"><b>Rewritten sparse.</b> P2G records every cell it
  scatters into; the grid update and the clear walk only that list. Exact rather than approximate,
  because every node a particle gathers from is a node it scattered to. Verified bit-for-bit against the
  dense loop over 3340 substeps: zero differing values.</td></tr>
<tr><td>precision</td><td style="text-align:left">f32 storage, f64 arithmetic (JS has no float32 math).
  The port is slightly <i>more</i> accurate per operation than the f32 reference, which is why its
  divergence looks like chaos rather than bias.</td></tr>
<tr><td>parallelism</td><td style="text-align:left">Gone. One thread, no atomics, so the port is
  deterministic where the reference is not.</td></tr>
<tr><td>interaction</td><td style="text-align:left">Added, and it is <b>not</b> canonical physics: an
  external actuator that relaxes grid velocity inside a Gaussian window toward the pointer. Off in every
  verification run.</td></tr>
</tbody></table>
</div>
</div>

</div>
"""


def main():
    css = (HERE / "demo.css").read_text(encoding="utf-8") + "\n" + (HERE / "page.css").read_text(encoding="utf-8")
    js = "\n".join((HERE / f).read_text(encoding="utf-8") for f in ("params.js", "mpm-elastic.js", "demo.js"))
    page_js = (HERE / "page.js").read_text(encoding="utf-8")
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<style>:root{color-scheme:dark}\n" + css + "</style></head><body>"
        + BODY
        + "<script>window.PAGE_DATA=" + json.dumps(DATA, separators=(",", ":")) + ";</script>"
        + "<script>" + js + "</script>"
        + "<script>" + page_js + "</script>"
        + "</body></html>"
    )
    (RUN / "bespoke_page.html").write_text(html, encoding="utf-8")
    print("wrote bespoke_page.html", len(html), "bytes")


if __name__ == "__main__":
    main()
