"""Build the bespoke task page from the run's own JSON. Writes bespoke_page.html.

Designed around one thing: a shared grid runs at ONE timestep, so the material mix is a budget
decision. Everything else on the page is there to keep that honest -- including the finding that
"stable" is not the criterion that should have set those timesteps.
"""
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
BASE = "/api/data/learning-taichi/runs/material-variants/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid"

M = json.load(open(os.path.join(D, "metrics.json")))
SW = json.load(open(os.path.join(D, "dt_sweep2.json")))
CR = json.load(open(os.path.join(D, "creep.json")))
EQ = json.load(open(os.path.join(D, "equivalence.json")))

MATS = ["fluid", "elastic", "snow", "sand"]
LABEL = {"fluid": "water", "elastic": "elastic", "snow": "snow", "sand": "sand"}
COL = {"fluid": "#4db6ff", "elastic": "#ff4d4d", "snow": "#f2f6fc", "sand": "#ffd24d"}

DATA = {
    "base": BASE,
    "label": LABEL,
    "col": COL,
    "mats": MATS,
    "materials": {m: {"E": M["materials"][m]["E"], "dt": M["materials"][m]["dt"],
                      "phi": M["materials"][m].get("phi", 0.0)} for m in MATS},
    "heap": M["heap"],
    "drop": M["drop"],
    "multi": M["multi_columns"],
    "multi_heaps": M["multi_heaps"],
    "equivalence": {m: {k: v for k, v in EQ["rows"][m].items()
                        if not isinstance(v, list)} for m in MATS},
    "eq_reps": EQ["reps_per_path"],
    "dtsweep": {sc: {m: {"dt_canonical": SW["scenes"][sc][m]["dt_canonical"],
                         "ref": {k: SW["scenes"][sc][m]["ref"][k]
                                 for k in ("width", "height", "repose")},
                         "rows": [{"mult": r["mult"], "dt": r["dt"], "ok": r["ok"],
                                   "drift": r["shape_drift"], "repose": r["repose"],
                                   "width": r["width"]}
                                  for r in SW["scenes"][sc][m]["sweep"]]}
                     for m in MATS} for sc in ("drop", "heap")},
    "creep": {m: {k: {"dt": v["dt"], "t": v["t"], "steps": v["substeps_cumulative"],
                      "repose": v["repose_deg"]}
                  for k, v in CR["series"][m].items()} for m in MATS},
    "creep_diag": CR["collapse_diagnostic"],
    "seeded": CR["seeded_slope_deg"],
    "phi_cal": M["phi_calibration"],
    "version": M["physics_version"],
}

TLDR = ("Sand is now canonical physics and it is cheap &mdash; snow, not sand, still sets the demo's "
        "frame budget &mdash; but chasing the angle of repose exposed that the settled slope of every "
        "plastic material here decays with substep count, and snow's decays almost entirely.")

VERDICT = ("<b>What happened.</b> Sand entered <code>sim/physics/</code> as a Drucker-Prager granular "
           "material with its own golden signature, all pre-existing signatures still green, and the "
           "frozen materials provably untouched. It runs at <b>dt = 1e-4</b>, i.e. <b>167 "
           "substeps/frame</b> at 60 fps, the same as elastic, so <b>snow remains the binding "
           "constraint at 333</b> and adding sand to a scene that already contains snow costs exactly "
           "nothing. (Sand alone would raise a water-only scene from 139 to 167.) Four materials now "
           "share one grid through a per-particle material id, and a single material pushed through "
           "that path matches the canonical one to below its own run-to-run noise. "
           "<b>What did not work.</b> The angle of repose is not a converged quantity. Held for 4 s and "
           "then refined fourfold in the timestep, sand's settled slope goes from 24&deg; to 19&deg;, "
           "and canonical snow's collapses from 56&deg; to 19&deg;. Both track the number of substeps "
           "taken rather than the physical time elapsed, so refining the timestep makes them worse, not "
           "better.")

CREEP_VERDICT = ("<b>Observed.</b> Snow's settled slope spreads <b>37&deg;</b> across timesteps at "
                 "equal physical time and only <b>1.6&deg;</b> at equal substep count. Sand spreads "
                 "6.5&deg; and 3.1&deg;. Elastic, which has no plastic projection, spreads 0.0&deg; "
                 "on both. <b>So the small-dt run is the corrupted one, not the converged one</b> "
                 "&mdash; which inverts the reading of the chart above it. "
                 "<b>Hypothesised mechanism:</b> the particle-grid round trip leaves a little noise in "
                 "the trial state each substep, and a plastic return mapping is one-sided &mdash; it "
                 "can remove elastic strain and never restore it &mdash; so that noise is rectified "
                 "into permanent plastic strain and ratchets once per substep. Elastic stores and "
                 "returns the same noise instead of rectifying it, which is why it is the flat control. "
                 "<b>What would test it:</b> the same sweep at several grid resolutions (transfer noise "
                 "should scale with dx), and a rate-independent implicit or sub-stepped return mapping, "
                 "which should remove the substep dependence if the mechanism is right.")

HTML = """<style>
*{box-sizing:border-box}
body{margin:0}
.wrap{font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  background:#0a0e14;color:#dfe6ee;padding:20px 18px 34px;max-width:1080px;margin:0 auto}
h2{font-size:19px;margin:34px 0 6px;color:#fff;letter-spacing:.1px}
h2 .n{color:#6fd3ee;font-size:13px;font-weight:600;margin-right:8px;vertical-align:2px}
h3{font-size:14px;margin:18px 0 6px;color:#cfd8e3;font-weight:650}
p{margin:8px 0;color:#c4cedb}
.muted{color:#7f8ea3}
.sm{font-size:13px}
.verdict{background:linear-gradient(180deg,#131a24,#0f1620);border:1px solid #22303f;
  border-left:3px solid #6fd3ee;border-radius:10px;padding:15px 17px;margin:6px 0 4px}
.verdict b{color:#fff}
.tldr{font-size:16px;color:#fff;margin:0 0 10px;font-weight:600}
.grid{display:grid;gap:12px}
.card{background:#0f151d;border:1px solid #1e2936;border-radius:10px;padding:13px 14px}
.chip{display:inline-flex;align-items:center;gap:6px;background:#131b25;border:1px solid #22303f;
  border-radius:999px;padding:3px 11px;font-size:12.5px;margin:3px 5px 3px 0}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
video{width:100%;border-radius:9px;background:#000;display:block;border:1px solid #1e2936}
img{max-width:100%;border-radius:9px;display:block;border:1px solid #1e2936}
.cap{font-size:12.5px;color:#7f8ea3;margin:7px 2px 0}
button{font:inherit;font-size:13px;background:#141d28;color:#c4cedb;border:1px solid #26333f;
  border-radius:7px;padding:6px 13px;cursor:pointer}
button.on{background:#6fd3ee;color:#06121a;border-color:#6fd3ee;font-weight:650}
button:hover:not(.on){background:#1b2733}
.tabs{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13.4px;margin-top:8px}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid #1b2531}
th:first-child,td:first-child{text-align:left}
th{color:#8fa0b3;font-weight:600;font-size:12.4px}
.big{font-size:32px;font-weight:700;color:#fff;line-height:1.1}
.unit{font-size:13px;color:#7f8ea3;font-weight:400}
.sel{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px}
.msel{display:flex;align-items:center;gap:7px;background:#131b25;border:1px solid #26333f;
  border-radius:8px;padding:7px 12px;cursor:pointer;user-select:none;font-size:13.5px}
.msel.on{border-color:#6fd3ee;background:#16232d}
.msel .box{width:13px;height:13px;border-radius:3px;border:1.5px solid #3d4c5c}
.msel.on .box{background:#6fd3ee;border-color:#6fd3ee}
.bar{height:12px;border-radius:6px;background:#1a2431;overflow:hidden;margin-top:5px}
.bar > i{display:block;height:100%;border-radius:6px}
.warn{border-left:3px solid #ffb454;background:#181509;border:1px solid #3a2f14;
  border-left:3px solid #ffb454;border-radius:8px;padding:11px 13px;font-size:13.4px;color:#e6d9b8;
  margin:12px 0}
.scope{border-left:3px solid #7f8ea3;background:#10151c;border:1px solid #1e2936;
  border-left:3px solid #7f8ea3;border-radius:8px;padding:11px 13px;font-size:13px;color:#9fb0c0;
  margin:12px 0}
svg{display:block;width:100%;height:auto}
.leg{display:flex;gap:13px;flex-wrap:wrap;font-size:12.5px;color:#9fb0c0;margin-top:6px}
code{background:#141d28;padding:1px 5px;border-radius:4px;font-size:12.6px;color:#a8d8e8}
@media(min-width:760px){.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:1fr 1fr 1fr}}
</style>
<div class="wrap">
<div class="verdict">
  <div class="tldr">__TLDR__</div>
  <div class="sm">__VERDICT__</div>
</div>

<h2><span class="n">01</span>One grid means one timestep</h2>
<p class="sm">An explicit solver's frame cost is <code>substeps x cost-per-substep</code>, and
<code>substeps = (1/60)/dt</code>. A shared grid carries a single velocity field advanced once per
substep, so every material in it runs at the same <code>dt</code> &mdash; the smallest one present.
Pick a material mix and watch what it bills.</p>
<div class="card">
  <div class="sel" id="mixsel"></div>
  <div class="grid g3" style="gap:16px">
    <div>
      <div class="muted sm">shared timestep</div>
      <div class="big" id="o-dt">&mdash;</div>
      <div class="muted sm" id="o-bind"></div>
    </div>
    <div>
      <div class="muted sm">substeps per frame at 60 fps</div>
      <div class="big" id="o-spf">&mdash;</div>
      <div class="bar"><i id="o-bar"></i></div>
    </div>
    <div>
      <div class="muted sm">cost vs water alone</div>
      <div class="big" id="o-rel">&mdash;</div>
      <div class="muted sm">139 substeps is the cheapest scene available</div>
    </div>
  </div>
  <p class="sm muted" id="o-note" style="margin-top:12px"></p>
</div>
<p class="sm">Sand lands on <b>167</b> substeps per frame, the same as elastic and 20% above water.
It is <b>not</b> the expensive material. Snow is, at <b>333</b>, and one snowball in a water scene
more than doubles the bill for every particle of water in it. Untick snow above and the whole scene
gets 2&times; cheaper; untick sand and nothing happens at all, because snow was already paying.</p>

<h2><span class="n">02</span>Is it sand? The over-steep heap</h2>
<p class="sm">The same 60&deg; triangular pile, released from rest, under each of the four materials.
A drop test cannot tell these apart &mdash; everything looks like a splat at impact. This can.</p>
<video src="__BASE__/heap_four_alone.mp4" autoplay loop muted playsinline></video>
<div class="cap">Each panel reports its own free-surface slope as it evolves. Water runs flat. Elastic
and snow keep essentially the whole seeded slope. Sand slumps partway and stops.</div>
<div class="sel" id="heapchips" style="margin-top:12px"></div>
<p class="sm">Elastic and snow keeping the slope is the informative part. Snow's yield criterion is a
<i>fixed</i> box on elastic stretch, so it does not weaken as confining pressure drops &mdash; snow has
cohesion and can stand a wall. Sand's admissible set is a <i>cone</i> whose width is proportional to
pressure, so near a free surface it has no strength at all. The slope where its own weight supplies
just enough confinement is the angle of repose, and nothing in the model prescribes it.</p>
<img src="__BASE__/repose_profile.png" alt="settled free surface per material with fitted flank slopes">
<div class="cap">The picture behind the number: binned free surface (white) and the least-squares
flank fits (dashed) the reported angle comes from.</div>

<h3>The friction angle is not the angle of repose</h3>
<img src="__BASE__/phi_calibration.png" alt="measured repose angle vs friction angle parameter">
<div class="cap">Measured angle against the model's friction-angle parameter, over 4 seeds x 3 pile
sizes. Tight and size-independent up to about 50&deg;; past that the error bars explode and the angle
starts tracking how many grid cells the pile spans, which is a discretisation artifact rather than a
material property. Canonical sand is frozen at the top of the reproducible band.</div>

<h2><span class="n">03</span>The trap: &ldquo;stable&rdquo; is not the criterion</h2>
<p class="sm">Every canonical material tolerates a far larger timestep than it uses without ever going
non-finite. That fact is useless, and acting on it is how a simulation quietly becomes a different
physical system. Flip between the two readings of the same sweep.</p>
<div class="tabs">
  <button id="t-stab" class="on">does it stay finite?</button>
  <button id="t-shape">is it still the same material?</button>
</div>
<div class="card"><svg id="chart-dt" viewBox="0 0 900 380"></svg>
  <div class="leg" id="leg-dt"></div></div>
<div class="cap" id="cap-dt"></div>

<h2><span class="n">04</span>Why: the plastic materials creep per substep, not per second</h2>
<p class="sm">Refining the timestep does not converge the settled pile, it <i>changes</i> it, and always
in the same direction. Which axis the curves collapse on says whether that is physics or bookkeeping.
Flip the horizontal axis.</p>
<div class="tabs">
  <button id="x-time" class="on">vs physical time</button>
  <button id="x-step">vs substeps taken</button>
  <span style="flex:1"></span>
  <div id="matsel" class="sel" style="margin:0"></div>
</div>
<div class="card"><svg id="chart-creep" viewBox="0 0 900 360"></svg>
  <div class="leg" id="leg-creep"></div></div>
<div class="cap" id="cap-creep"></div>
<div class="warn" id="creep-verdict"></div>

<h2><span class="n">05</span>Four materials, one grid</h2>
<p class="sm">A per-particle material id and a runtime branch, so a single grid can hold all four at
once. Every material still takes exactly its canonical path.</p>
<div class="grid g2">
  <div>
    <video src="__BASE__/four_in_one_grid.mp4" autoplay loop muted playsinline></video>
    <div class="cap">Four blocks released from rest in one shared grid, at the shared
    <span id="mc-dt"></span> that snow forces on all of them.</div>
  </div>
  <div>
    <video src="__BASE__/four_heaps_one_grid.mp4" autoplay loop muted playsinline></video>
    <div class="cap">The heap test again, all four materials simultaneously in one simulation rather
    than four.</div>
  </div>
</div>
<div class="scope">Contact between different materials is whatever a shared node velocity produces:
two materials meeting at a node exchange momentum as if the node held one blended material. That is
coexistence, not a calibrated multi-phase contact model, and nothing on this page is a claim about
what happens at a sand-water interface.</div>

<h2><span class="n">06</span>The refactor changed nothing</h2>
<p class="sm">A single material pushed through the new branching path has to land where the canonical
compile-time path lands. Equality is the wrong bar: GPU atomic scatter order varies run to run, and the
two paths compile different kernels so they can order the same arithmetic differently at the last bit,
which a chaotic rollout then amplifies. The bar is therefore a <b>bracket of disagreements that
provably carry no information</b> &mdash; run-to-run nondeterminism at the bottom, and re-running
canonical with the initial positions nudged by one float32 rounding unit at the top.</p>
<table id="eqtab"></table>
<div class="cap">All figures are <b>traj_rmse</b>: mean per-particle distance over the whole rollout, in
domain lengths, over <span id="eq-reps"></span> repeats of each path. The refactor's disagreement is at
or below the one-ulp nudge for every material, i.e. smaller than the effect of perturbing the initial
condition by a rounding error.</div>
<img src="__BASE__/equivalence.png" style="margin-top:12px" alt="multi vs canonical inside the no-information band">
<div class="cap">The shape is the evidence, not the endpoint. A real bias would appear immediately and
grow linearly. These start at rounding scale and grow exponentially into the same plateau as the grey
band, which is chaos amplifying a last-bit difference and nothing else.</div>
<p class="sm" style="margin-top:12px">The frozen materials also had to be untouched. A single
before/after comparison would prove nothing here, because two runs of the <i>same</i> code already
differ by up to 1e-2 on the chaotic fluid column. So the test is distributional: <b id="fz-floor"></b>
repeats of each configuration under the pre-promotion physics (restored from git) and under the new
one, then every pairwise distance <i>within</i> a version against every distance <i>across</i>
versions. Across-version distances come out at <b id="fz-worst"></b> the within-version mean at worst,
over fluid, elastic and snow on two scenes &mdash; indistinguishable. Physics version went
<code id="v-before"></code> &rarr; <code id="v-after"></code>.</p>
<table id="fztab" style="margin-top:8px"></table>
<div class="cap">Within-code distances are the simulator disagreeing with itself; across-code are
before-the-promotion against after. If the promotion had moved a frozen material, the right-hand
column would sit outside the left-hand range.</div>

<div class="scope" style="margin-top:26px"><b>Scope.</b> 2D, one grid resolution (128&sup2;), one floor
friction, particle counts in the low thousands, forward simulation only. The angle of repose is
measured on one scene family. The creep result is measured on that same scene family at four timesteps
per material. Sand's constitutive model is one choice (Drucker-Prager with a non-dilatant flow rule)
and no alternative granular model was run against it.</div>
</div>
<script>
const D = __DATA__;
const FZ = __FROZEN__;
const $ = s => document.querySelector(s);
const fmt = (x,n=2) => Number(x).toFixed(n);
const sci = x => { const e = Math.floor(Math.log10(Math.abs(x))); const m = x/Math.pow(10,e);
  return m.toFixed(1)+'e'+(e<0?'-':'')+String(Math.abs(e)).padStart(1,'0'); };

/* ---------- 01 budget mixer ---------- */
let mix = {fluid:true, elastic:true, snow:true, sand:true};
const mixsel = $('#mixsel');
D.mats.forEach(m => {
  const b = document.createElement('div');
  b.className = 'msel on'; b.dataset.m = m;
  b.innerHTML = `<span class="box"></span><span class="dot" style="background:${D.col[m]}"></span>${D.label[m]}`;
  b.onclick = () => { mix[m] = !mix[m]; b.classList.toggle('on', mix[m]); budget(); };
  mixsel.appendChild(b);
});
function budget(){
  const on = D.mats.filter(m => mix[m]);
  if(!on.length){ $('#o-dt').textContent='—'; $('#o-spf').textContent='—'; $('#o-rel').textContent='—';
    $('#o-bind').textContent='pick at least one material'; $('#o-bar').style.width='0';
    $('#o-note').textContent=''; return; }
  const bind = on.reduce((a,b)=> D.materials[a].dt <= D.materials[b].dt ? a : b);
  const dt = D.materials[bind].dt;
  const spf = Math.round((1/60)/dt);
  const base = Math.round((1/60)/D.materials.fluid.dt);
  $('#o-dt').innerHTML = sci(dt)+' <span class="unit">s</span>';
  $('#o-spf').textContent = spf;
  $('#o-rel').innerHTML = fmt(spf/base,2)+'<span class="unit">&times;</span>';
  $('#o-bind').innerHTML = `set by <b style="color:${D.col[bind]}">${D.label[bind]}</b>`;
  const mx = Math.round((1/60)/Math.min(...D.mats.map(m=>D.materials[m].dt)));
  $('#o-bar').style.width = (100*spf/mx)+'%';
  $('#o-bar').style.background = D.col[bind];
  const others = on.filter(m=>m!==bind).map(m=>D.label[m]);
  $('#o-note').textContent = others.length
    ? `${others.join(', ')} would each be happy at a larger timestep and pay ${LABELCAP(bind)}'s bill anyway.`
    : `${LABELCAP(bind)} alone.`;
}
const LABELCAP = m => D.label[m];
budget();

/* ---------- 02 heap chips ---------- */
const hc = $('#heapchips');
D.mats.forEach(m => {
  const f = D.heap.final[m];
  const c = document.createElement('span');
  c.className = 'chip';
  c.innerHTML = `<span class="dot" style="background:${D.col[m]}"></span><b>${D.label[m]}</b>
    &nbsp;slope <b style="color:#fff">${fmt(f.repose_angle,1)}&deg;</b>
    <span class="muted">&nbsp;width ${fmt(f.spread_width,2)}</span>`;
  hc.appendChild(c);
});
const seedChip = document.createElement('span');
seedChip.className='chip';
seedChip.innerHTML = `<span class="muted">seeded</span> <b style="color:#fff">${fmt(D.seeded,0)}&deg;</b>`;
hc.insertBefore(seedChip, hc.firstChild);

/* ---------- generic chart ---------- */
function chart(svg, series, opt){
  const W=900, H=opt.H||380, L=68, R=16, T=18, B=46;
  const xs = series.flatMap(s=>s.x), ys = series.flatMap(s=>s.y).filter(v=>isFinite(v)&&v>0);
  const lg = v => Math.log10(v);
  const x0=Math.min(...xs), x1=Math.max(...xs);
  // a fixed y range can be forced, so that switching between series does not silently rescale the
  // axis and turn a flat line into a dramatic wiggle
  const y0 = opt.yrange ? opt.yrange[0] : Math.min(...ys);
  const y1 = opt.yrange ? opt.yrange[1] : Math.max(...ys);
  const LX = opt.logx!==false, LY = opt.logy!==false;
  const px = v => L + (W-L-R)*((LX?lg(v)-lg(x0):v-x0)/((LX?lg(x1)-lg(x0):x1-x0)||1));
  const py = v => T + (H-T-B)*(1-((LY?lg(v)-lg(y0):v-y0)/((LY?lg(y1)-lg(y0):y1-y0)||1)));
  let g = `<rect x="0" y="0" width="${W}" height="${H}" fill="#0f151d"/>`;
  const ticks = (a,b,log) => { const o=[]; if(log){ for(let e=Math.floor(lg(a));e<=Math.ceil(lg(b));e++)
      for(const m of [1,3]){ const v=m*Math.pow(10,e); if(v>=a*0.999&&v<=b*1.001) o.push(v);} }
    else { const n=5; for(let i=0;i<=n;i++) o.push(a+(b-a)*i/n); } return o; };
  ticks(y0,y1,LY).forEach(v => { g += `<line x1="${L}" x2="${W-R}" y1="${py(v)}" y2="${py(v)}"
    stroke="#22303f" stroke-width="1"/><text x="${L-9}" y="${py(v)+4}" fill="#7f8ea3" font-size="11"
    text-anchor="end">${opt.fy?opt.fy(v):v}</text>`; });
  ticks(x0,x1,LX).forEach(v => { g += `<line x1="${px(v)}" x2="${px(v)}" y1="${T}" y2="${H-B}"
    stroke="#1a2431" stroke-width="1"/><text x="${px(v)}" y="${H-B+18}" fill="#7f8ea3" font-size="11"
    text-anchor="middle">${opt.fx?opt.fx(v):v}</text>`; });
  (opt.hlines||[]).forEach(h => { g += `<line x1="${L}" x2="${W-R}" y1="${py(h.v)}" y2="${py(h.v)}"
    stroke="${h.c}" stroke-width="1.5" stroke-dasharray="6 4"/>
    <text x="${W-R-4}" y="${py(h.v)-6}" fill="${h.c}" font-size="11.5" text-anchor="end">${h.t}</text>`; });
  (opt.vlines||[]).forEach(h => { g += `<line x1="${px(h.v)}" x2="${px(h.v)}" y1="${T}" y2="${H-B}"
    stroke="${h.c}" stroke-width="1.3" stroke-dasharray="4 4"/>
    <text x="${px(h.v)+5}" y="${T+13}" fill="${h.c}" font-size="11.5">${h.t}</text>`; });
  series.forEach(s => {
    const pts = s.x.map((v,i)=> (isFinite(s.y[i])&&s.y[i]>0)?`${px(v)},${py(s.y[i])}`:null).filter(Boolean);
    if(pts.length>1) g += `<polyline points="${pts.join(' ')}" fill="none" stroke="${s.c}"
      stroke-width="${s.w||2.2}" stroke-dasharray="${s.dash||''}" opacity="${s.o||1}"/>`;
    s.x.forEach((v,i)=>{ if(!isFinite(s.y[i])||s.y[i]<=0) return;
      if(s.mark && s.mark[i]==='x'){ const X=px(v),Y=py(s.y[i]);
        g += `<path d="M${X-6},${Y-6}L${X+6},${Y+6}M${X+6},${Y-6}L${X-6},${Y+6}" stroke="#ff6e6e"
          stroke-width="2.4"/>`; }
      else g += `<circle cx="${px(v)}" cy="${py(s.y[i])}" r="${s.r||3}" fill="${s.c}" opacity="${s.o||1}"/>`; });
  });
  g += `<text x="${(L+W-R)/2}" y="${H-8}" fill="#9fb0c0" font-size="12" text-anchor="middle">${opt.xl||''}</text>`;
  g += `<text x="14" y="${(T+H-B)/2}" fill="#9fb0c0" font-size="12" text-anchor="middle"
    transform="rotate(-90 14 ${(T+H-B)/2})">${opt.yl||''}</text>`;
  svg.innerHTML = g;
}

/* ---------- 03 stable vs faithful ---------- */
function rangeChart(svg, sc){
  const W=900, H=380, L=88, R=90, T=26, B=50;
  const rows = D.mats.map(m => D.dtsweep[sc][m]);
  const mults = rows[0].rows.map(r=>r.mult);
  const x0=Math.min(...mults), x1=Math.max(...mults);
  const lg=Math.log10, px=v=> L+(W-L-R)*((lg(v)-lg(x0))/(lg(x1)-lg(x0)));
  const rowH=(H-T-B)/D.mats.length;
  let g=`<rect width="${W}" height="${H}" fill="#0f151d"/>`;
  for(let e=Math.floor(lg(x0)); e<=Math.ceil(lg(x1)); e++)
    for(const mm of [1,2,5]){ const v=mm*Math.pow(10,e); if(v<x0||v>x1) continue;
      g+=`<line x1="${px(v)}" x2="${px(v)}" y1="${T}" y2="${H-B}" stroke="#1a2431"/>
      <text x="${px(v)}" y="${H-B+18}" fill="#7f8ea3" font-size="11" text-anchor="middle">${v>=1?v+'\\u00d7':'\\u00f7'+Math.round(1/v)}</text>`; }
  g+=`<line x1="${px(1)}" x2="${px(1)}" y1="${T-8}" y2="${H-B}" stroke="#dfe6ee" stroke-width="1.6"
      stroke-dasharray="4 3"/><text x="${px(1)}" y="${T-13}" fill="#dfe6ee" font-size="11.5"
      text-anchor="middle">canonical dt</text>`;
  D.mats.forEach((m,i)=>{
    const r=D.dtsweep[sc][m];
    const okMults=r.rows.filter(v=>v.ok).map(v=>v.mult);
    const top=Math.max(...okMults), y=T+i*rowH+rowH*0.28, hh=rowH*0.42;
    g+=`<rect x="${px(x0)}" y="${y}" width="${px(top)-px(x0)}" height="${hh}" rx="4"
        fill="${D.col[m]}" opacity="0.9"/>`;
    if(top<x1) g+=`<rect x="${px(top)}" y="${y}" width="${px(x1)-px(top)}" height="${hh}" rx="4"
        fill="#3d1414" stroke="#ff6e6e" stroke-width="1"/>`;
    g+=`<text x="${L-10}" y="${y+hh*0.72}" fill="#dfe6ee" font-size="12.5" text-anchor="end">${D.label[m]}</text>`;
    g+=`<text x="${W-R+8}" y="${y+hh*0.72}" fill="#9fb0c0" font-size="11.5">stable to ${top}\\u00d7</text>`;
  });
  g+=`<text x="${(L+W-R)/2}" y="${H-10}" fill="#9fb0c0" font-size="12" text-anchor="middle">timestep, as a multiple of each material\\u2019s canonical dt</text>`;
  svg.innerHTML=g;
}
let dtMode = 'stab';
function drawDt(){
  const sc = 'heap';
  // a diverged run has no settled shape to score, so it is dropped from the line rather than drawn
  // as if its nan-scrubbed positions were a drift value
  const series = D.mats.map(m => {
    const r = D.dtsweep[sc][m].rows.filter(v => v.ok);
    return {x: r.map(v => v.dt/D.dtsweep[sc][m].dt_canonical),
            y: r.map(v => Math.max(v.drift,1e-6)), c: D.col[m]};
  });
  const opt = {logx:true, logy:true, xl:'timestep, as a multiple of each material\\u2019s canonical dt',
    fx:v=> (v>=1?`${v}\\u00d7`:`\\u00f7${Math.round(1/v)}`),
    vlines:[{v:1,c:'#7f8ea3',t:'canonical'}]};
  if(dtMode==='stab'){
    rangeChart($('#chart-dt'), sc);
    $('#leg-dt').innerHTML = '<span><span class="dot" style="background:#3d1414;'+
      'box-shadow:0 0 0 1px #ff6e6e"></span> diverged</span>';
    $('#cap-dt').innerHTML = 'Stability alone says every material is fine at <b>2&ndash;16&times;</b> '+
      'the timestep it actually uses, on the harder of the two scenes. Read this chart on its own and '+
      'you would happily quadruple the timestep and quarter the cost of the whole demo.';
    return;
  } else {
    opt.yl = 'settled-shape drift vs a dt/8 run';
    opt.fy = v => v>=1 ? String(v) : (v>=0.01 ? v.toFixed(2) : v.toExponential(0));
    opt.hlines = [{v:0.15,c:'#6fd3ee',t:'15% drift'}];
    $('#cap-dt').innerHTML = 'The same runs, scored on whether the settled pile still matches the '+
      'same run at dt/8. <b style="color:#ff4d4d">Elastic</b> sits near zero everywhere. '+
      '<b style="color:#f2f6fc">Snow</b> and <b style="color:#ffd24d">sand</b> disagree with the finer '+
      'run badly, <i>at their own canonical timesteps</i>. The obvious conclusion is that the canonical '+
      'timesteps are too coarse. <b>Section 04 shows that conclusion is backwards</b> &mdash; for a '+
      'plastic material the dt/8 run is not a fixed point, so this chart is measuring drift against a '+
      'moving reference.';
  }
  $('#leg-dt').innerHTML = D.mats.map(m =>
    `<span><span class="dot" style="background:${D.col[m]}"></span> ${D.label[m]}</span>`).join('')
    + '<span class="muted">diverged runs are omitted &mdash; they have no settled shape to score</span>';
  chart($('#chart-dt'), series, opt);
}
$('#t-stab').onclick = () => { dtMode='stab'; $('#t-stab').classList.add('on');
  $('#t-shape').classList.remove('on'); drawDt(); };
$('#t-shape').onclick = () => { dtMode='shape'; $('#t-shape').classList.add('on');
  $('#t-stab').classList.remove('on'); drawDt(); };
drawDt();

/* ---------- 04 creep ---------- */
let creepX = 'time', creepM = 'snow';
const ms = $('#matsel');
D.mats.forEach(m => {
  const b = document.createElement('div');
  b.className = 'msel' + (m===creepM?' on':''); b.dataset.m=m;
  b.innerHTML = `<span class="dot" style="background:${D.col[m]}"></span>${D.label[m]}`;
  b.onclick = () => { creepM=m; [...ms.children].forEach(c=>c.classList.toggle('on',c.dataset.m===m));
    drawCreep(); };
  ms.appendChild(b);
});
const SHADES = ['#ffffff','#a8d8e8','#6fd3ee','#3a8fa8'];
function drawCreep(){
  const runs = D.creep[creepM];
  const keys = Object.keys(runs).sort((a,b)=>parseFloat(b)-parseFloat(a));
  const series = keys.map((k,i) => ({
    x: (creepX==='time' ? runs[k].t : runs[k].steps).slice(1),
    y: runs[k].repose.slice(1).map(v=>Math.max(v,0.3)),
    c: SHADES[i%4], r:2.4, w:2.0}));
  chart($('#chart-creep'), series, {H:360, logx: creepX==='step', logy:false,
    yrange:[0, Math.ceil(D.seeded)+3],
    xl: creepX==='time' ? 'physical time (s)' : 'substeps taken (cumulative)',
    yl: 'free-surface slope (degrees)',
    fx: v => creepX==='time' ? v.toFixed(1) : (v>=1000? (v/1000).toFixed(0)+'k' : v.toFixed(0)),
    fy: v => v.toFixed(0),
    hlines:[{v:D.seeded,c:'#7f8ea3',t:'seeded '+fmt(D.seeded,0)+'\\u00b0'}]});
  $('#leg-creep').innerHTML = keys.map((k,i)=>
    `<span><span class="dot" style="background:${SHADES[i%4]}"></span> dt = ${sci(runs[k].dt)} s
     (&times;${k})</span>`).join('');
  const dg = D.creep_diag[creepM];
  $('#cap-creep').innerHTML = creepX==='time'
    ? `Four timesteps, same physical time. If the slump were physical these would lie on top of each
       other. For <b>${D.label[creepM]}</b> they spread by
       <b>${fmt(dg.spread_at_equal_time,1)}&deg;</b> at the end.`
    : `The same four runs against cumulative substep count. For <b>${D.label[creepM]}</b> the spread at
       a common substep count is <b>${fmt(dg.spread_at_equal_substeps,1)}&deg;</b>, against
       <b>${fmt(dg.spread_at_equal_time,1)}&deg;</b> at common physical time. Whichever number is
       smaller is the axis the process is really running on.`;
}
$('#x-time').onclick = () => { creepX='time'; $('#x-time').classList.add('on');
  $('#x-step').classList.remove('on'); drawCreep(); };
$('#x-step').onclick = () => { creepX='step'; $('#x-step').classList.add('on');
  $('#x-time').classList.remove('on'); drawCreep(); };
drawCreep();
$('#creep-verdict').innerHTML = __CREEPVERDICT__;

/* ---------- 05/06 ---------- */
$('#mc-dt').textContent = sci(D.multi.dt_shared)+' s';
$('#eq-reps').textContent = D.eq_reps;
$('#eqtab').innerHTML = '<tr><th>material</th><th>run-to-run self-noise</th>'
  + '<th>one-ulp nudge</th><th>multi vs canonical</th><th>vs the nudge</th></tr>'
  + D.mats.map(m => { const e = D.equivalence[m];
  return `<tr><td><span class="dot" style="background:${D.col[m]}"></span> ${D.label[m]}</td>
    <td class="muted">${e.within_path_mean.toExponential(2)}</td>
    <td class="muted">${e.rounding_nudge_mean.toExponential(2)}</td>
    <td>${e.across_path_mean.toExponential(2)}</td>
    <td style="color:${e.ratio_to_nudge<=1.5?'#8fe3b0':'#ffb454'}">${fmt(e.ratio_to_nudge,2)}×</td></tr>`;
  }).join('');
$('#fz-worst').textContent = FZ.worst_ratio_of_means.toFixed(2)+'×';
$('#fz-floor').textContent = FZ.reps_per_side;
$('#v-before').textContent = FZ.version_before;
$('#v-after').textContent = FZ.version_after;
$('#fztab').innerHTML = '<tr><th>scene / material</th><th>within-code range</th>'
  + '<th>across-code range</th><th>ratio of means</th></tr>' + FZ.rows.map(r =>
  `<tr><td>${r.scene} &middot; ${D.label[r.material]}</td>
   <td>${r.within_code_min.toExponential(1)} &ndash; ${r.within_code_max.toExponential(1)}</td>
   <td>${r.across_code_min.toExponential(1)} &ndash; ${r.across_code_max.toExponential(1)}</td>
   <td style="color:${r.passed?'#8fe3b0':'#ffb454'}">${r.ratio_of_means.toFixed(2)}</td></tr>`).join('');
</script>
"""


def main():
    frozen = json.load(open(os.path.join(D, "frozen_materials_check.json")))
    html = (HTML.replace("__DATA__", json.dumps(DATA))
                .replace("__FROZEN__", json.dumps(frozen))
                .replace("__BASE__", BASE)
                .replace("__TLDR__", TLDR)
                .replace("__VERDICT__", VERDICT)
                .replace("__CREEPVERDICT__", json.dumps(CREEP_VERDICT)))
    open(os.path.join(D, "bespoke_page.html"), "w", encoding="utf-8").write(html)
    print("wrote bespoke_page.html", len(html), "bytes")
    return html


if __name__ == "__main__":
    main()
