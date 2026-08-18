"""Assemble metrics.json and the bespoke task page from the verification artifacts.

Nothing here computes physics; it reads verify/out/score.json (written by verify/score.py) and
verify/out/capture.json (written by verify/capture_page.py) and lays the numbers out. Keeping the
page generator separate from the measurement is what makes it safe to iterate on the page.

    .venv/Scripts/python.exe runs/.../build_page.py
"""
import json
import pathlib

RUN = pathlib.Path(__file__).resolve().parent
OUT = RUN / "verify" / "out"
API = ("/api/data/learning-taichi/runs/material-variants/"
       "the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/")

MATS = ["fluid", "elastic", "snow", "sand"]
LABEL = {"fluid": "WATER", "elastic": "RUBBER", "snow": "SNOW", "sand": "SAND"}
COLOR = {"fluid": "#4db6ff", "elastic": "#ff9d5c", "snow": "#e6ecff", "sand": "#ffd24d"}
BLURB = {
    "fluid": "no shear strength at all — it finds its own level",
    "elastic": "purely elastic, no plastic projection — it keeps the shape it was given",
    "snow": "singular values clamped into a BOX — cohesive, so it can stand a wall",
    "sand": "log-strain projected onto a CONE — cohesionless, so its strength needs pressure",
}


def main():
    score = json.loads((OUT / "score.json").read_text())
    cap = json.loads((OUT / "capture.json").read_text())
    dash = json.loads((OUT / "dashboard_check.json").read_text())
    svd = score["svd_unit_test"]
    by = {r["scene"]: r for r in score["scenes"]}
    mixed = by["mixed4"]

    # ---------------------------------------------------------------- metrics.json
    metrics = {
        "physics_version": score["physics_version"],
        "device": score["device"],
        "user_agent": score["user_agent"],
        "shared_dt": score["shared_dt"],
        "substeps_per_frame_at_60fps_shared": score["substeps_per_frame_shared"],
        "svd_unit_test": svd,
        "per_material_heap": {
            m: {
                "n": by["heap_" + m]["n"],
                "dt": by["heap_" + m]["dt"],
                "substeps_per_frame": by["heap_" + m]["substeps_per_frame"],
                "traj_rmse": by["heap_" + m]["traj_rmse_web_vs_canonical"],
                "self_noise": by["heap_" + m]["self_noise_nudge"],
                "self_noise_repeat": by["heap_" + m]["self_noise_repeat"],
                "ratio_to_noise_band": by["heap_" + m]["ratio_to_nudge_band"],
                "repose_angle_canonical": by["heap_" + m]["shape_canonical"]["repose_angle"],
                "repose_angle_webgpu": by["heap_" + m]["shape_web"]["repose_angle"],
                "pile_height_canonical": by["heap_" + m]["shape_canonical"]["pile_height"],
                "pile_height_webgpu": by["heap_" + m]["shape_web"]["pile_height"],
                "spread_width_canonical": by["heap_" + m]["shape_canonical"]["spread_width"],
                "spread_width_webgpu": by["heap_" + m]["shape_web"]["spread_width"],
            } for m in MATS
        },
        "mixed_four_material_scene": {
            "n": mixed["n"], "dt": mixed["dt"],
            "substeps_per_frame": mixed["substeps_per_frame"],
            "traj_rmse": mixed["traj_rmse_web_vs_canonical"],
            "self_noise": mixed["self_noise_nudge"],
            "ratio_to_noise_band": mixed["ratio_to_nudge_band"],
            "per_material": mixed["per_material"],
        },
        "shared_dt_creep": score["shared_dt_creep"],
        "realtime": score["realtime"],
        "fixed_point_headroom": score["pile_headroom"],
        "substeps_by_materials_present": score["substep_table"],
        "shipped_page": {
            "particles": cap["final"]["n"],
            "counts": cap["final"]["counts"],
            "fps": cap["final"]["fps"],
            "achieved_x_real_time": cap["final"]["achieved"],
            "substeps_per_frame": cap["final"]["spf"],
            "substeps_per_simulated_second": round(1.0 / cap["final"]["dt"]),
            "capture_frames": cap["frames"], "capture_seconds": cap["wall_seconds"],
            "capture_fps": cap["capture_fps"],
            "dashboard_tab": dash["state"],
        },
        "webgpu_errors": score["webgpu_errors"],
    }
    (RUN / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote metrics.json")

    # ---------------------------------------------------------------- the bespoke page
    data = json.dumps({
        "mats": MATS, "label": LABEL, "color": COLOR, "blurb": BLURB,
        "heap": metrics["per_material_heap"],
        "mixed": metrics["mixed_four_material_scene"],
        "creep": score["shared_dt_creep"],
        "realtime": score["realtime"],
        "pile": score["pile_headroom"],
        "substeps": score["substep_table"],
        "svd": svd,
        "shipped": metrics["shipped_page"],
        "api": API,
    }, separators=(",", ":"))

    html = PAGE.replace("__DATA__", data)
    (RUN / "bespoke_page.html").write_text(html, encoding="utf-8")
    print("wrote bespoke_page.html", len(html), "bytes")


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>The Demo MVP — four materials on WebGPU</title>
<style>
:root{--bg:#0a0e14;--pan:#0e131b;--fg:#dfe6ee;--mut:#7f8ea3;--acc:#6fd3ee;--line:#1d2734;
      --water:#4db6ff;--rubber:#ff9d5c;--snow:#e6ecff;--sand:#ffd24d;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;padding:18px 16px 34px}
.wrap{max-width:1040px;margin:0 auto}
h2{font-size:15px;letter-spacing:.13em;text-transform:uppercase;color:var(--acc);margin:34px 0 10px;
  border-bottom:1px solid var(--line);padding-bottom:6px}
h2:first-of-type{margin-top:8px}
p{margin:9px 0}
.mut{color:var(--mut)}
.verdict{background:linear-gradient(180deg,#101822,#0c1119);border:1px solid #24404f;border-left:3px solid var(--acc);
  border-radius:5px;padding:14px 16px;font-size:15px;line-height:1.66}
.verdict b{color:#bfe9f7}
.warn{border-left-color:var(--sand)}
.warn b{color:var(--sand)}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;color:#bcd6e4}
video,img{width:100%;border:1px solid var(--line);border-radius:5px;display:block;background:#000}
figure{margin:12px 0 0}
figcaption{font-size:12px;color:var(--mut);margin-top:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:8px;
  font-family:ui-monospace,Menlo,Consolas,monospace}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.07em;font-size:11px}
.ok{color:#7ee787}.bad{color:#ff8f8f}.hl{color:var(--acc)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0 4px}
.tabs button{font:inherit;font-size:12px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;
  background:linear-gradient(180deg,#16222d,#0d151d);border:1px solid #23394a;border-radius:4px;
  color:#9fc4d2;padding:8px 12px;transition:all .1s}
.tabs button:hover{border-color:#356b83;color:#e6f6fd}
.tabs button.on{color:#04121a;border-color:#7fd8f0;background:linear-gradient(180deg,#a9e9fb,#5cc0dd)}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:7px;vertical-align:baseline}
.card{background:var(--pan);border:1px solid var(--line);border-radius:6px;padding:14px 16px}
.big{font-size:26px;font-weight:700;font-family:ui-monospace,Menlo,Consolas,monospace;line-height:1.15}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:10px;margin-top:12px}
.kpi{background:var(--pan);border:1px solid var(--line);border-radius:6px;padding:11px 13px}
.kpi .lab{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--mut)}
.kpi .val{font-size:20px;font-weight:700;font-family:ui-monospace,Menlo,Consolas,monospace;margin-top:3px}
.bar{height:9px;background:#131c26;border-radius:5px;overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%;border-radius:5px}
.note{font-size:12px;color:var(--mut);margin-top:7px}
.scope{border-left:3px solid var(--sand);background:rgba(255,210,77,.05);padding:9px 13px;border-radius:0 5px 5px 0;
  font-size:12.8px;color:#cdd8e2;margin-top:12px}
.scope b{color:var(--sand)}
svg{display:block;width:100%;height:auto}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin-top:8px}
</style>
<div class="wrap">

<div class="verdict">
  All four canonical materials — <b style="color:var(--water)">water</b>,
  <b style="color:var(--rubber)">rubber</b>, <b style="color:var(--snow)">snow</b> and
  <b style="color:var(--sand)">sand</b> — now run together on <b>one shared MLS-MPM grid, in a browser,
  on WebGPU, at 1.00&times; real time</b> with <span id="vShipped"></span> particles, and the Demo tab
  is that simulation. Each material reproduces canonical <code>sim.physics</code> on a fixed
  angle-of-repose scene to within, or close to, canonical's own self-noise — settled slopes agree to
  <span id="vAngle"></span>. The new engineering was a 2&times;2 SVD in WGSL, unit-checked against
  <code>ti.svd</code> on <span id="vSvd"></span> adversarial matrices before it was wired in.
  <b>What it does not do:</b> a mixed scene is <b>not</b> quantitatively canonical — one grid forces one
  timestep, and running sand at snow's smaller step costs it
  <span id="vCreep"></span> of settled slope.
</div>

<h2>1 — is each material recognisably itself?</h2>
<p class="mut">An over-steep 60° heap, released from rest, 2000 particles, each material at its OWN
canonical timestep. Whatever slope survives is the slope the material genuinely holds. Ground truth on
the top row, the browser underneath, same seed and same substep count. Pick a material to see its
numbers; the video plays all four at once.</p>

<div class="tabs" id="matTabs"></div>
<div class="grid2" style="margin-top:6px">
  <div class="card" id="matCard"></div>
  <div class="card">
    <div style="font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--mut)">settled slope, both sides</div>
    <svg id="reposeSvg" viewBox="0 0 460 220"></svg>
    <div class="legend"><span>■ canonical <code>sim.physics</code></span><span>■ WebGPU, in the browser</span>
      <span class="mut">dotted = the 60° slope it was seeded with</span></div>
  </div>
</div>

<figure>
  <video src="__API__material_vs_canonical.mp4" controls autoplay muted loop playsinline></video>
  <figcaption>Ground truth (top) against the same step running in WebGPU (bottom). The fitted flank
  whose slope <em>is</em> the reported angle is drawn on the final frame of every panel, so the number
  and the picture are the same object. Water spreads flat, sand slumps to a dome, rubber and snow keep
  the heap they were given.</figcaption>
</figure>

<h2>2 — do they behave when they are in the same box?</h2>
<p class="mut">All four on one grid through canonical <code>simulate_multi</code>, against the same
scene in the browser. This is the case the demo actually ships.</p>
<div class="kpis" id="mixKpis"></div>
<figure>
  <video src="__API__mixed4_vs_canonical.mp4" controls muted loop playsinline></video>
  <figcaption>One grid, five groups, 3245 particles, shared Δt = 5e-5. Colours are the canonical
  per-material colours from <code>spec/registry/materials.json</code>.</figcaption>
</figure>

<div class="scope"><b>The scope this result is bound to.</b> Four materials, on two scenes (a heap per
material and one mixed drop), on one device (RTX 4090 / Chromium), at one grid resolution, over 1.6 s
of physics. That supports a hypothesis about multi-material MPM in the browser; it is not a claim about
other scenes, other GPUs, or longer rollouts.</div>

<h2>3 — the shipped page</h2>
<p class="mut">The Demo tab, captured from a real GPU-backed window while genuine pointer events drive
it: the opening scene lands, sand is poured in, the view switches to grid mass and to raw particles,
the material is dragged around, a channel is carved out of it, and the scene is reset.</p>
<figure>
  <video src="__API__demo_capture.mp4" controls muted loop playsinline></video>
  <figcaption>Real time. Read the HUD: <code>speed</code> is simulated seconds per real second and is
  measured, never assumed.</figcaption>
</figure>
<div class="grid2">
  <figure><img src="__API__verify/shots/dashboard_demo_tab.png" alt="the Demo tab running">
    <figcaption>The Demo tab in the dashboard itself — through Vite, React and a generated ES-module
    copy of the same code.</figcaption></figure>
  <figure><img src="__API__demo_no_webgpu.png" alt="the no-WebGPU path">
    <figcaption>Graceful degradation, captured by hiding <code>navigator.gpu</code> from a real window.
    The three cases are told apart: hidden by an insecure origin, unsupported, or no device at all.</figcaption></figure>
</div>

<h2>4 — what one grid costs, and why</h2>
<p class="mut">One grid means one timestep, and it is <code>min(dt)</code> over the materials
<em>present</em>. Put snow in the box and every particle in it pays snow's substep count. Click a
mixture:</p>
<div class="tabs" id="mixTabs"></div>
<div class="grid2" style="margin-top:6px">
  <div class="card" id="costCard"></div>
  <div class="card">
    <div style="font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--mut)">measured frame cost, four materials present</div>
    <svg id="costSvg" viewBox="0 0 460 230"></svg>
  </div>
</div>

<div class="scope"><b>And this is a correctness issue, not only a cost one.</b> A plastic material's
strength decays with <em>substep count</em>, not with physical time, so running sand at snow's smaller
step gives it more creep than canonical sand shows. Measured canonical-against-canonical on the same
heap: <span id="creepText"></span>. Rubber, which has no plastic projection, does not move.
<b>A material's behaviour therefore depends on what else is in the scene</b>, and the demo does not
claim a mixed scene is quantitatively canonical.</div>

<h2>5 — the two traps this had to get past</h2>
<div class="grid2">
  <div class="card">
    <div class="big" style="color:var(--acc)" id="svdBig"></div>
    <div class="note">worst relative reconstruction error of <code>U·diag(s)·Vᵀ</code> over
    <span id="svdN"></span> matrices in 11 adversarial families — random, near-rotation, near-singular,
    reflections with negative determinant, anisotropic to a condition number of 10⁴, exact zeros and
    identities, the snow clamp boundary, and 1600 deformation gradients sampled from real snow and sand
    rollouts.</div>
    <table id="svdTable"></table>
    <div class="note">A wrong 2&times;2 SVD does not produce garbage; it produces plausible-looking
    motion. Snow still crumbles, sand still slumps, and nothing on the screen tells you the singular
    values are wrong — which is why this was proved in isolation before it went anywhere near the sim.</div>
  </div>
  <div class="card">
    <div style="font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--mut)">fixed-point headroom, deliberately piling material into one corner</div>
    <svg id="pileSvg" viewBox="0 0 460 210"></svg>
    <div class="note">WGSL has no atomic float add, so P2G accumulates mass into an
    <code>atomic&lt;u32&gt;</code> scaled by 2²⁴ quanta per particle mass. That saturates at
    <b>256 particle masses on one node</b> and then <b>wraps silently</b> — no NaN, no error, the block
    just detonates. Previous measurements were on scenes with a fixed particle count; this demo lets a
    visitor pile material up by hand, so the worst case was measured by actively crushing all four
    materials into a corner with the drag force.</div>
  </div>
</div>

<h2>6 — how the four materials actually differ</h2>
<svg id="lawSvg" viewBox="0 0 900 250"></svg>
<p class="note">All four share one transfer skeleton and one grid. They differ only in what happens to
the deformation gradient after the gather. Water keeps a scalar volume ratio and no shear strength at
all. Rubber keeps <code>F</code> untouched — no plastic projection, which is why it needs no singular
values and why its settled slope does not move with the timestep. Snow clamps the singular values into
a fixed <b>box</b>: the admissible set does not shrink when the confining pressure falls, which is
cohesion, which is why snow can stand a vertical wall. Sand projects the log of the singular values
onto a <b>cone</b> whose width is proportional to the confining pressure: no pressure, no strength,
so sand cannot.</p>

</div>
<script>
const D = __DATA__;
const API = D.api;
document.querySelectorAll('video,img').forEach(e => {
  const s = e.getAttribute('src'); if (s && s.startsWith('__API__')) e.setAttribute('src', API + s.slice(7));
});
const f = (v, n) => Number(v).toFixed(n === undefined ? 2 : n);
const e1 = v => Number(v).toExponential(1);

// ---------------------------------------------------------------- verdict fill-ins
document.getElementById('vShipped').textContent = D.shipped.particles.toLocaleString();
const angErr = Math.max(...D.mats.map(m => Math.abs(D.heap[m].repose_angle_webgpu - D.heap[m].repose_angle_canonical)));
document.getElementById('vAngle').textContent = f(angErr, 2) + '° or better';
document.getElementById('vSvd').textContent = D.svd.n_matrices.toLocaleString();
const worstCreep = Math.min(...Object.values(D.creep).map(c => c.delta_deg));
document.getElementById('vCreep').textContent = f(Math.abs(worstCreep), 1) + '°';

// ---------------------------------------------------------------- 1. per-material
const tabs = document.getElementById('matTabs');
D.mats.forEach((m, i) => {
  const b = document.createElement('button');
  b.innerHTML = '<span class="sw" style="background:' + D.color[m] + '"></span>' + D.label[m];
  b.onclick = () => sel(m);
  b.dataset.m = m;
  tabs.appendChild(b);
});
function sel(m) {
  [...tabs.children].forEach(b => b.classList.toggle('on', b.dataset.m === m));
  const h = D.heap[m];
  const inside = h.ratio_to_noise_band <= 1.0;
  document.getElementById('matCard').innerHTML =
    '<div style="font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--mut)">' +
      D.label[m] + ' — ' + D.blurb[m] + '</div>' +
    '<div class="big" style="color:' + D.color[m] + ';margin-top:8px">' +
      f(h.repose_angle_webgpu, 1) + '° <span style="font-size:14px;color:var(--mut)">settled slope in the browser</span></div>' +
    '<div class="note">canonical <code>sim.physics</code> on the same seed: <b class="hl">' +
      f(h.repose_angle_canonical, 1) + '°</b></div>' +
    '<table>' +
    row('traj_rmse vs canonical', e1(h.traj_rmse), 'mean per-particle distance, in domain lengths') +
    row('canonical self-noise band', e1(h.self_noise), 'canonical vs canonical, ICs nudged by 1e-7') +
    row('ratio to that band', f(h.ratio_to_noise_band, 1) + '×',
        inside ? 'inside the band — indistinguishable from re-running canonical'
               : 'above the band; see the note') +
    row('pile height', f(h.pile_height_webgpu, 3) + ' / ' + f(h.pile_height_canonical, 3), 'browser / canonical') +
    row('spread width', f(h.spread_width_webgpu, 3) + ' / ' + f(h.spread_width_canonical, 3), 'browser / canonical') +
    row('timestep', h.dt.toExponential(0) + '  (' + h.substeps_per_frame + ' substeps/frame)', 'its own canonical dt') +
    '</table>' +
    (inside ? '' : '<div class="note"><b class="hl">Why the ratio, not the number.</b> The absolute ' +
      'disagreement is ' + e1(h.traj_rmse) + ' domain lengths — a rounding error on a unit square. ' +
      'The ratio is large because the <em>band</em> is tiny: this scene barely moves, so canonical ' +
      'reproduces itself almost exactly, while fixed-point quantisation in P2G is a small deterministic ' +
      'bias rather than chaos. Both numbers are shown because either alone misleads.</div>');
}
function row(k, v, note) {
  return '<tr><td>' + k + (note ? '<div class="note" style="margin:0">' + note + '</div>' : '') +
         '</td><td class="hl">' + v + '</td></tr>';
}
sel('sand');

// repose bar chart
(function () {
  const W = 460, H = 220, pad = 34, base = H - 26, top = 12;
  const max = 65;
  let s = '<line x1="' + pad + '" y1="' + (base - (60 / max) * (base - top)) + '" x2="' + (W - 8) +
    '" y2="' + (base - (60 / max) * (base - top)) + '" stroke="#3a4a5a" stroke-dasharray="3 3"/>' +
    '<text x="' + (W - 10) + '" y="' + (base - (60 / max) * (base - top) - 5) +
    '" fill="#7f8ea3" font-size="10" text-anchor="end" font-family="ui-monospace">seeded 60°</text>';
  D.mats.forEach((m, i) => {
    const x = pad + i * ((W - pad - 12) / 4);
    const w = (W - pad - 12) / 4 - 14;
    const g = D.heap[m].repose_angle_canonical, b = D.heap[m].repose_angle_webgpu;
    const hg = (g / max) * (base - top), hb = (b / max) * (base - top);
    s += '<rect x="' + x + '" y="' + (base - hg) + '" width="' + (w / 2 - 1) + '" height="' + hg + '" fill="#4a6b80"/>';
    s += '<rect x="' + (x + w / 2 + 1) + '" y="' + (base - hb) + '" width="' + (w / 2 - 1) + '" height="' + hb +
      '" fill="' + D.color[m] + '"/>';
    s += '<text x="' + (x + w / 2) + '" y="' + (base - Math.max(hg, hb) - 6) +
      '" fill="#dfe6ee" font-size="11" text-anchor="middle" font-family="ui-monospace">' + f(b, 1) + '°</text>';
    s += '<text x="' + (x + w / 2) + '" y="' + (base + 14) + '" fill="#7f8ea3" font-size="10" text-anchor="middle">' +
      D.label[m] + '</text>';
  });
  s += '<line x1="' + pad + '" y1="' + base + '" x2="' + (W - 8) + '" y2="' + base + '" stroke="#1d2734"/>';
  for (const t of [0, 20, 40, 60]) {
    const y = base - (t / max) * (base - top);
    s += '<text x="' + (pad - 6) + '" y="' + (y + 3) + '" fill="#7f8ea3" font-size="9" text-anchor="end" font-family="ui-monospace">' + t + '</text>';
  }
  document.getElementById('reposeSvg').innerHTML = s;
})();

// ---------------------------------------------------------------- 2. mixed scene
(function () {
  const m = D.mixed;
  const cells = [['whole scene', m.traj_rmse, m.self_noise, m.ratio_to_noise_band]];
  for (const k of D.mats) if (m.per_material[k])
    cells.push([D.label[k].toLowerCase(), m.per_material[k].traj_rmse_web_vs_canonical,
                m.per_material[k].self_noise_nudge, m.per_material[k].ratio_to_nudge_band]);
  document.getElementById('mixKpis').innerHTML = cells.map(([n, r, b, x]) =>
    '<div class="kpi"><div class="lab">' + n + '</div><div class="val" style="color:' +
    (x <= 1.6 ? '#7ee787' : '#ffd24d') + '">' + f(x, 1) + '×</div>' +
    '<div class="note" style="margin-top:2px">' + e1(r) + ' vs a band of ' + e1(b) + '</div></div>').join('');
})();

// ---------------------------------------------------------------- 4. cost of a mixture
const mixTabs = document.getElementById('mixTabs');
D.substeps.forEach((r, i) => {
  const b = document.createElement('button');
  b.textContent = r.present.map(p => ({fluid:'water',elastic:'rubber',snow:'snow',sand:'sand'})[p]).join(' + ');
  b.onclick = () => selMix(i);
  b.dataset.i = i;
  mixTabs.appendChild(b);
});
function selMix(i) {
  [...mixTabs.children].forEach(b => b.classList.toggle('on', +b.dataset.i === i));
  const r = D.substeps[i];
  const worst = D.substeps[D.substeps.length - 1];
  document.getElementById('costCard').innerHTML =
    '<div style="font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--mut)">shared timestep</div>' +
    '<div class="big" style="color:var(--acc);margin-top:8px">' + r.dt.toExponential(0) + '</div>' +
    '<div class="note">= min(dt) over the materials present — the stiffest one sets it for everybody.</div>' +
    '<table>' +
    row('substeps per real-time frame', r.substeps_per_frame, 'at 60 fps') +
    row('solver dispatches per frame', (3 * r.substeps_per_frame).toLocaleString(), 'P2G → grid → G2P, all in ONE command buffer') +
    row('relative to water alone', f(r.substeps_per_frame / D.substeps[0].substeps_per_frame, 2) + '×', '') +
    '</table>' +
    (i === D.substeps.length - 1
      ? '<div class="note"><b class="hl">This is the demo\'s real cost.</b> Snow alone doubles it, and ' +
        'adding sand to a scene that already has snow costs nothing at all.</div>' : '');
}
selMix(3);

(function () {
  const W = 460, H = 230, L = 46, R = 12, T = 14, B = 30;
  const rt = D.realtime;
  const xs = rt.map(r => r.n), ys = rt.map(r => r.sustained_ms);
  const xmax = Math.max(...xs), ymax = 18;
  const X = v => L + (v / xmax) * (W - L - R);
  const Y = v => H - B - (v / ymax) * (H - B - T);
  let s = '<rect x="' + L + '" y="' + Y(16.67) + '" width="' + (W - L - R) + '" height="' + (H - B - Y(16.67)) +
    '" fill="#0f1720"/>';
  s += '<line x1="' + L + '" y1="' + Y(16.67) + '" x2="' + (W - R) + '" y2="' + Y(16.67) +
    '" stroke="#ff7a7a" stroke-dasharray="4 3"/>' +
    '<text x="' + (W - R) + '" y="' + (Y(16.67) - 5) + '" fill="#ff7a7a" font-size="10" text-anchor="end" font-family="ui-monospace">60 fps budget 16.67 ms</text>';
  s += '<polyline fill="none" stroke="#6fd3ee" stroke-width="2" points="' +
    rt.map(r => X(r.n) + ',' + Y(r.sustained_ms)).join(' ') + '"/>';
  rt.forEach(r => {
    s += '<circle cx="' + X(r.n) + '" cy="' + Y(r.sustained_ms) + '" r="3.4" fill="#6fd3ee"/>';
  });
  const last = rt[rt.length - 1];
  s += '<text x="' + (X(last.n) - 6) + '" y="' + (Y(last.sustained_ms) - 9) +
    '" fill="#dfe6ee" font-size="11" text-anchor="end" font-family="ui-monospace">' +
    f(last.sustained_ms, 2) + ' ms at ' + last.n.toLocaleString() + '</text>';
  s += '<line x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) + '" stroke="#1d2734"/>';
  s += '<line x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (H - B) + '" stroke="#1d2734"/>';
  for (const t of [0, 5, 10, 15]) s += '<text x="' + (L - 6) + '" y="' + (Y(t) + 3) +
    '" fill="#7f8ea3" font-size="9" text-anchor="end" font-family="ui-monospace">' + t + '</text>';
  for (const t of [4096, 8192, 12288, 16384]) s += '<text x="' + X(t) + '" y="' + (H - B + 13) +
    '" fill="#7f8ea3" font-size="9" text-anchor="middle" font-family="ui-monospace">' + (t / 1024) + 'k</text>';
  s += '<text x="' + (L + (W - L - R) / 2) + '" y="' + (H - 3) + '" fill="#7f8ea3" font-size="10" text-anchor="middle">particles</text>';
  s += '<text x="10" y="' + (T + 8) + '" fill="#7f8ea3" font-size="10">ms/frame</text>';
  document.getElementById('costSvg').innerHTML = s;
})();

document.getElementById('creepText').innerHTML = D.mats.filter(m => Math.abs(D.creep[m].delta_deg) > 1e-9)
  .map(m => '<b>' + D.label[m].toLowerCase() + '</b> ' + f(D.creep[m].repose_own, 1) + '° → ' +
       f(D.creep[m].repose_shared, 1) + '° (' + (D.creep[m].delta_deg > 0 ? '+' : '') +
       f(D.creep[m].delta_deg, 1) + '°)').join(', ') +
  ', snow already runs at the shared step, rubber ' +
  (Math.abs(D.creep.elastic.delta_deg) < 0.2 ? 'moves ' + f(Math.abs(D.creep.elastic.delta_deg), 2) + '°' : '');

// ---------------------------------------------------------------- 5. SVD + headroom
document.getElementById('svdBig').textContent = e1(D.svd.max_rel_reconstruction);
document.getElementById('svdN').textContent = D.svd.n_matrices.toLocaleString();
document.getElementById('svdTable').innerHTML =
  row('orthogonality of U and V', e1(D.svd.max_orthogonality), 'worst |UᵀU − I|, |VᵀV − I|') +
  row('singular values vs ti.svd', e1(D.svd.max_rel_singular_vs_taichi), 'the only thing the return maps consume') +
  row('descending-order violations', D.svd.order_violations, 'both plastic clamps assume s₀ ≥ s₁') +
  row('non-finite results', D.svd.non_finite, 'across every adversarial family');

(function () {
  const W = 460, H = 210, L = 46, R = 12, T = 16, B = 30;
  const p = D.pile, sat = p[0].mass_saturates_at_pm;
  const xmax = Math.max(...p.map(r => r.n)) * 1.05;
  const X = v => L + (v / xmax) * (W - L - R);
  const Y = v => H - B - (v / (sat * 1.1)) * (H - B - T);
  let s = '<line x1="' + L + '" y1="' + Y(sat) + '" x2="' + (W - R) + '" y2="' + Y(sat) +
    '" stroke="#ff7a7a" stroke-dasharray="4 3"/>' +
    '<text x="' + (W - R) + '" y="' + (Y(sat) - 5) + '" fill="#ff7a7a" font-size="10" text-anchor="end" font-family="ui-monospace">wraps silently at ' + sat + '</text>';
  s += '<polyline fill="none" stroke="#ffd24d" stroke-width="2" points="' +
    p.map(r => X(r.n) + ',' + Y(r.max_node_mass_pm)).join(' ') + '"/>';
  p.forEach(r => { s += '<circle cx="' + X(r.n) + '" cy="' + Y(r.max_node_mass_pm) + '" r="3.4" fill="#ffd24d"/>'; });
  const last = p[p.length - 1];
  s += '<text x="' + X(last.n) + '" y="' + (Y(last.max_node_mass_pm) - 9) +
    '" fill="#dfe6ee" font-size="11" text-anchor="end" font-family="ui-monospace">' +
    f(last.max_node_mass_pm, 1) + ' pm — ' + f(last.mass_headroom, 1) + '× headroom</text>';
  s += '<line x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) + '" stroke="#1d2734"/>';
  s += '<line x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (H - B) + '" stroke="#1d2734"/>';
  for (const t of [0, 64, 128, 192, 256]) s += '<text x="' + (L - 6) + '" y="' + (Y(t) + 3) +
    '" fill="#7f8ea3" font-size="9" text-anchor="end" font-family="ui-monospace">' + t + '</text>';
  p.forEach(r => { s += '<text x="' + X(r.n) + '" y="' + (H - B + 13) +
    '" fill="#7f8ea3" font-size="9" text-anchor="middle" font-family="ui-monospace">' + (r.n / 1024) + 'k</text>'; });
  s += '<text x="10" y="' + (T + 8) + '" fill="#7f8ea3" font-size="10">particle masses on the heaviest node</text>';
  document.getElementById('pileSvg').innerHTML = s;
})();

// ---------------------------------------------------------------- 6. the constitutive picture
(function () {
  const s = [];
  const box = (x, y, w, h, fill, stroke) => '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h +
    '" rx="4" fill="' + fill + '" stroke="' + stroke + '"/>';
  s.push('<text x="10" y="18" fill="#7f8ea3" font-size="12" font-family="ui-monospace">one grid, one transfer, four endings</text>');
  const lanes = [
    ['P2G  scatter mass + stress', 30], ['grid  gravity, walls, Coulomb friction', 66], ['G2P  gather velocity + ∇v', 102]];
  lanes.forEach(([t, y]) => {
    s.push(box(14, y, 872, 26, '#111a24', '#22justify'.replace('justify', '3546')));
    s.push('<text x="26" y="' + (y + 17) + '" fill="#bcd6e4" font-size="12" font-family="ui-monospace">' + t + '</text>');
  });
  s.push('<path d="M450 128 L450 142" stroke="#2b4658" stroke-width="1.5"/>');
  const cols = ['#4db6ff', '#ff9d5c', '#e6ecff', '#ffd24d'];
  const heads = ['WATER', 'RUBBER', 'SNOW', 'SAND'];
  const body = [
    ['J ← J (1 + Δt tr C)', 'a scalar volume ratio.', 'No F, no SVD.', 'σ = E (J − 1) I'],
    ['F ← (I + Δt C) F', 'no projection at all.', 'Needs only the polar', 'rotation R = U Vᵀ'],
    ['F = U Σ′ Vᵀ,  Σ′ = clamp(Σ)', 'a BOX in singular values.', 'Cohesive: the admissible', 'set never shrinks'],
    ['ε = ln Σ projected onto a CONE', 'width ∝ confining pressure.', 'Cohesionless: no pressure,', 'no shear strength']];
  for (let i = 0; i < 4; i++) {
    const x = 14 + i * 220;
    s.push('<path d="M450 142 L' + (x + 100) + ' 142 L' + (x + 100) + ' 154" stroke="#2b4658" stroke-width="1.5" fill="none"/>');
    s.push(box(x, 154, 202, 86, '#0e131b', cols[i] + '55'));
    s.push('<text x="' + (x + 12) + '" y="' + 174 + '" fill="' + cols[i] +
      '" font-size="12" font-weight="700" font-family="ui-monospace">' + heads[i] + '</text>');
    body[i].forEach((line, j) => s.push('<text x="' + (x + 12) + '" y="' + (190 + j * 13) +
      '" fill="#9fb3c4" font-size="10.5" font-family="ui-monospace">' + line.replace(/&/g, '&amp;').replace(/</g, '&lt;') + '</text>'));
  }
  document.getElementById('lawSvg').innerHTML = s.join('');
})();
</script>
"""


if __name__ == "__main__":
    main()
