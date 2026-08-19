"""Build the bespoke task page.

Design brief for this page: the reader must leave knowing WHAT was broken (nameable, measured), that
the fix was two parameters plus a boundary condition, that snow and sand provably did not move, and that
sinking/floating now comes out of density alone. The strong form is a FLIP -- one frame, one toggle,
old physics against new on the same scene and seed -- so that the reader performs the comparison.
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/api/data/learning-taichi/runs/material-variants/improve-material-realism-in-behavior"

db = json.load(open(os.path.join(HERE, "diag_before.json")))
da = json.load(open(os.path.join(HERE, "diag_after.json")))
vb = json.load(open(os.path.join(HERE, "volcurve_before.json")))
va = json.load(open(os.path.join(HERE, "volcurve_after.json")))
ab = json.load(open(os.path.join(HERE, "ablation.json")))
bu = json.load(open(os.path.join(HERE, "buoyancy_after.json")))
wf = json.load(open(os.path.join(HERE, "wall_film.json")))
sig = open(os.path.join(HERE, "signatures_output.txt"), encoding="utf-8").read().splitlines()

sig_rows = []
for ln in sig:
    ln = ln.strip()
    if ln.startswith("[PASS]") or ln.startswith("[FAIL]"):
        ok = ln.startswith("[PASS]")
        rest = ln[6:].strip()
        name, _, detail = rest.partition("   (")
        sig_rows.append({"ok": ok, "name": name.strip(), "detail": detail.rstrip(")").strip()})

NEWSIG = ("pool:", "density rescaling", "slam: rubber holds", "drop: water is nearly",
          "dam break: frictionless")
for r in sig_rows:
    r["new"] = any(r["name"].startswith(p) for p in NEWSIG)

DATA = {
    "flip": [
        {"id": "water", "label": "Water", "scene": "dam", "mat": "fluid",
         "old": f"{BASE}/flip_dam_fluid_old.mp4", "new": f"{BASE}/flip_dam_fluid_new.mp4",
         "complaint": "“too mushy and sticky”",
         "metric": "worst-squashed water particle, J",
         "vold": f"{min(vb['curves']['fluid/drop']['min']):.2f}",
         "vnew": f"{min(va['curves']['fluid/drop']['min']):.2f}",
         "note": "A dam break released against the left wall. The front runs free across the floor "
                 "before it reaches the far wall."},
        {"id": "rubber", "label": "Rubber", "scene": "slam", "mat": "elastic",
         "old": f"{BASE}/flip_slam_elastic_old.mp4", "new": f"{BASE}/flip_slam_elastic_new.mp4",
         "complaint": "“compresses far too much”",
         "metric": "body area at peak compression",
         "vold": f"{100 * min(vb['curves']['elastic/slam']['mean']):.1f}%",
         "vnew": f"{100 * min(va['curves']['elastic/slam']['mean']):.1f}%",
         "note": "A hard floor impact. Watch the readout: the old blob genuinely occupies less area "
                 "while it is squashed, and that is the state the eye sees."},
        {"id": "snow", "label": "Snow", "scene": "heap", "mat": "snow",
         "old": f"{BASE}/flip_heap_snow_old.mp4", "new": f"{BASE}/flip_heap_snow_new.mp4",
         "complaint": "must NOT change", "metric": "settled slope",
         "vold": f"{db['runs']['heap/snow']['repose_angle']:.1f}°",
         "vnew": f"{da['runs']['heap/snow']['repose_angle']:.1f}°",
         "note": "Snow's density went from 1.0 to 0.3 and its stiffness fell with it, which is a "
                 "change the material cannot feel. Flip it and nothing happens."},
        {"id": "sand", "label": "Sand", "scene": "heap", "mat": "sand",
         "old": f"{BASE}/flip_heap_sand_old.mp4", "new": f"{BASE}/flip_heap_sand_new.mp4",
         "complaint": "must NOT change", "metric": "angle of repose",
         "vold": f"{db['runs']['heap/sand']['repose_angle']:.1f}°",
         "vnew": f"{da['runs']['heap/sand']['repose_angle']:.1f}°",
         "note": "Same story at density 1.6. The heap collapses to the same angle it always did."},
    ],
    "vol": {"before": vb["curves"], "after": va["curves"]},
    "ablate": ab,
    "buoy": bu,
    "sig": sig_rows,
    "regress": {
        "scenes": ["drop", "column", "heap", "slam", "dam"],
        "mats": ["snow", "sand", "fluid", "elastic"],
        "before": {k: {m: db["runs"][k][m] for m in ("spread_width", "pile_height", "repose_angle")}
                   for k in db["runs"]},
        "after": {k: {m: da["runs"][k][m] for m in ("spread_width", "pile_height", "repose_angle")}
                  for k in da["runs"]},
    },
    "mat": {k: {kk: vv for kk, vv in v.items() if kk != "color"} for k, v in da["MAT"].items()},
    "mat_old": {k: {kk: vv for kk, vv in v.items() if kk != "color"} for k, v in db["MAT"].items()},
    "ver": {"before": db["physics_version"], "after": da["physics_version"]},
}

HTML = """<style>
:root{--bg:#0a0e14;--ink:#dfe6ee;--mut:#7f8ea3;--acc:#6fd3ee;--old:#e0736a;--new:#6fd3ee;
      --pan:#111823;--line:#222c3a;}
*{box-sizing:border-box}
.wrap{background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,
      "Segoe UI",Roboto,sans-serif;padding:22px 20px 40px;max-width:1180px;margin:0 auto}
h2{font-size:19px;margin:34px 0 6px;letter-spacing:.2px}
h3{font-size:15px;margin:20px 0 6px;color:var(--acc);font-weight:600}
p{margin:8px 0}
.sub{color:var(--mut);font-size:13.5px;margin:2px 0 12px}
.verdict{background:var(--pan);border:1px solid var(--line);border-left:3px solid var(--acc);
      border-radius:8px;padding:14px 16px;margin-bottom:8px}
.verdict b{color:#fff}
.scope{background:#1a1410;border:1px solid #4a3a28;border-radius:8px;padding:12px 15px;
      color:#e8d5bd;font-size:13.5px;margin:14px 0}
.flipbar{display:flex;align-items:center;gap:12px;margin:14px 0 4px;flex-wrap:wrap}
.toggle{display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden;
      background:var(--pan)}
.toggle button{background:none;border:0;color:var(--mut);padding:8px 22px;font:600 14px/1 inherit;
      cursor:pointer}
.toggle button.on{background:var(--acc);color:#04222b}
.toggle button.on.oldstate{background:var(--old);color:#2a0f0c}
.hint{color:var(--mut);font-size:13px}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:12px}
.card{background:var(--pan);border:1px solid var(--line);border-radius:9px;padding:12px}
.card h4{margin:0 0 2px;font-size:14.5px}
.card .cmp{color:var(--mut);font-size:12.5px;margin-bottom:8px}
.card video{width:100%;border-radius:6px;display:block;background:#000}
.readout{display:flex;justify-content:space-between;align-items:baseline;margin-top:9px;
      border-top:1px solid var(--line);padding-top:8px}
.readout .lab{color:var(--mut);font-size:12.5px}
.readout .val{font:700 20px/1 ui-monospace,SFMono-Regular,Menlo,monospace}
.note{color:var(--mut);font-size:12.5px;margin-top:7px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:10px}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left}
th{color:var(--mut);font-weight:600;font-size:12.5px}
td.num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;text-align:right}
.tag{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:700}
.tag.pass{background:#123326;color:#6ee7a8}.tag.fail{background:#3a1616;color:#ff8f8f}
.tag.newsig{background:#12303a;color:var(--acc);margin-left:6px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.bar{height:9px;border-radius:5px;background:#1b2432;overflow:hidden}
.bar i{display:block;height:100%}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--mut);font-size:12.5px;margin:8px 0}
.legend span{display:inline-flex;align-items:center;gap:5px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.mbtn{background:var(--pan);border:1px solid var(--line);color:var(--mut);border-radius:6px;
      padding:5px 11px;font:13px/1 inherit;cursor:pointer;margin-right:6px}
.mbtn.on{border-color:var(--acc);color:var(--acc)}
</style>
<div class="wrap">

<div class="verdict">
<b>Verdict.</b> Two named defects, both measured before anything was changed: a water particle could be
squashed to <span class="mono">J = 0.52</span> of its volume on impact (that is the &ldquo;mushy&rdquo;),
and a rubber blob genuinely lost <span class="mono">11%</span> of its area at the moment it hit the floor,
its worst particle crushed to <span class="mono">18%</span> of rest volume. One Poisson ratio and one
stiffness fix both. Per&#8209;material density was added and sinking and floating fall out of the mass
ratio with no buoyancy force anywhere. Snow and sand are unchanged, provably rather than approximately.
<b>What did not work.</b> The &ldquo;sticky&rdquo; complaint splits in two and only one half is fixed.
Against the <i>floor</i>, water is measurably less draggy: its dam&#8209;break front runs
__FRONTPCT__% faster, and the ablation puts essentially all of that on the friction change. Against the
<i>walls</i> it got worse, and by the same knob. And &ldquo;rubber breaks too easily&rdquo; was never
reproduced on any scene tried, so nothing here can claim to have fixed it.
</div>

<div class="scope"><b>The measurement that went the wrong way, and the mechanism.</b> Fraction of water
sitting within five cells of a side wall <i>and</i> above the bulk surface &mdash; thin sheets riding up
the wall. Peak during the splash: __WALLPEAK__. Still hanging there at the end: __WALLRES__. The
ablation says which knob did it, and it is not the one that looks guilty: reverting the <i>stiffness</i>
changes the peak by about one percent, while reverting the <i>friction</i> alone removes most of the
increase. Frictionless water slides <i>up</i> a wall as freely as it slides along a floor, and a closed
two&#8209;dimensional box with no way to break a sheet into droplets is the worst possible place to put
it. Less mechanical drag and more visual clinging are the same change.</div>

<h2>The flip</h2>
<p class="sub">Same scene, same seed, same random numbers. Only the physics differs. The readout under
each clip is the quantity the claim rests on.</p>
<div class="flipbar">
  <div class="toggle" id="tg">
    <button id="bo" class="on oldstate">OLD physics</button>
    <button id="bn">NEW physics</button>
  </div>
  <span class="hint" id="verline"></span>
</div>
<div class="grid4" id="flipgrid"></div>

<h2>Where the volume went</h2>
<p class="sub">det(F) is the model&rsquo;s own volume ratio, so a body&rsquo;s true area is its initial
area times the mean of det(F). This is the quantity the rubber complaint is about, and it is a
<i>transient</i>: the settled blob looks fine long after it was crushed.</p>
<div class="two">
  <img src="__BASE__/rubber_volume.png" style="width:100%;border-radius:8px">
  <img src="__BASE__/water_volume.png" style="width:100%;border-radius:8px">
</div>

<h2>Which knob did which job</h2>
<p class="sub">Every row runs on the new physics with exactly one knob reverted, so the contributions
separate. Read it as: friction owns the runout, stiffness owns the squashiness, and they barely
interact.</p>
<div id="ablate"></div>

<h2>Density: nobody added a buoyancy force</h2>
<p class="sub">Each solid is released <i>at rest</i>, fully submerged, in the same pool. The only
difference between these runs is particle mass.</p>
<video src="__BASE__/buoyancy_three.mp4" autoplay loop muted playsinline
       style="width:100%;border-radius:8px;background:#000"></video>
<div class="two" style="margin-top:14px">
  <div>
    <h3>The control</h3>
    <p class="sub">One material, one stiffness, one scene, four densities. If the outcome follows
    &rho; and nothing else, it is buoyancy rather than a quirk of the constitutive model.</p>
    <div id="ladder"></div>
    <video src="__BASE__/density_ladder.mp4" autoplay loop muted playsinline
           style="width:100%;border-radius:8px;background:#000;margin-top:10px"></video>
  </div>
  <div>
    <h3>Settled state</h3>
    <img src="__BASE__/buoyancy_three.png" style="width:100%;border-radius:8px">
    <img src="__BASE__/density_ladder.png" style="width:100%;border-radius:8px;margin-top:10px">
  </div>
</div>
<div class="scope"><b>Scope, next to the claim it bounds.</b> A shared grid gives every material at a
node one velocity, so two materials meeting at an interface exchange momentum as if the node held one
blended substance. That is enough for bulk buoyancy, whose dominant term is the interior of a blob many
cells wide. It is <i>not</i> a calibrated multi&#8209;phase contact model, and the measured submerged
fraction is only qualitatively Archimedean: a deformable body spreads and pushes a bump of water up
around itself, so the reading runs above the density ratio and drifts with how much the body has
deformed. A second caveat sits in the water rather than the solid. A pool still loses about a tenth of
its height over a couple of seconds, because the weakly&#8209;compressible fluid takes its pressure from
an advected volume ratio rather than from the actual particle packing, so nothing resists a settling
pack tightening up. The visible consequence is that a <i>neutrally</i> buoyant blob does not hang
perfectly still, it creeps upward, because the water around it is compacting and it is not.
<b>The ordering across densities is the result. The individual numbers are not.</b></div>

<h2>Snow and sand did not move</h2>
<p class="sub">Every material on every scene, old against new. On the dashed line means unchanged. Water
and rubber are on the chart too, and they are the points that leave it.</p>
<div style="margin:10px 0"><span id="mbtns"></span></div>
<div id="regress"></div>

<h2>The golden signatures</h2>
<p class="sub">Every pre-existing signature stayed green through a change to frozen ground truth, and the
new behaviour brought its own.</p>
<div id="sigtable"></div>

</div>
<script>
const D = __DATA__;
const BASE = "__BASE__";
let side = "old";

function flipGrid(){
  const g = document.getElementById("flipgrid");
  g.innerHTML = D.flip.map(f => `
    <div class="card">
      <h4>${f.label}</h4>
      <div class="cmp">${f.complaint} &nbsp;&middot;&nbsp; ${f.scene} scene</div>
      <video src="${side==='old'?f.old:f.new}" autoplay loop muted playsinline></video>
      <div class="readout"><span class="lab">${f.metric}</span>
        <span class="val" style="color:${side==='old'?'var(--old)':'var(--new)'}">${side==='old'?f.vold:f.vnew}</span></div>
      <div class="note">${f.note}</div>
    </div>`).join("");
  document.getElementById("verline").textContent =
    side==='old' ? "physics " + D.ver.before : "physics " + D.ver.after;
}
document.getElementById("bo").onclick = () => { side="old";
  document.getElementById("bo").classList.add("on","oldstate");
  document.getElementById("bn").classList.remove("on"); flipGrid(); };
document.getElementById("bn").onclick = () => { side="new";
  document.getElementById("bn").classList.add("on");
  document.getElementById("bo").classList.remove("on","oldstate"); flipGrid(); };
flipGrid();

// ---- attribution table -------------------------------------------------------------------------
(function(){
  const w = D.ablate.water, r = D.ablate.rubber;
  const wk = Object.keys(w), rk = Object.keys(r);
  const base = w[wk[0]].front_speed;
  let h = `<h3>Water &mdash; dam-break front speed while the front runs free</h3><table>
    <tr><th>configuration</th><th style="text-align:right">front speed<br>(domain lengths / s)</th>
    <th style="text-align:right">spread of J<br>(99th &minus; 1st pct)</th><th style="width:34%"></th></tr>`;
  wk.forEach(k => {
    const v = w[k];
    h += `<tr><td>${k}</td><td class="num">${v.front_speed.toFixed(3)}</td>
      <td class="num">${v.J_spread.toFixed(3)}</td>
      <td><div class="bar"><i style="width:${(100*v.front_speed/D.ablate.ritter_front_speed).toFixed(1)}%;
        background:${k.indexOf('new')===0?'var(--new)':'var(--old)'}"></i></div></td></tr>`;
  });
  h += `<tr><td style="color:var(--mut)">ideal-fluid limit 2&radic;(g h&#8320;)</td>
        <td class="num" style="color:var(--mut)">${D.ablate.ritter_front_speed.toFixed(3)}</td>
        <td class="num" style="color:var(--mut)">0</td>
        <td><div class="bar"><i style="width:100%;background:#2a3444"></i></div></td></tr></table>`;
  h += `<h3>Rubber &mdash; volume held through a hard floor impact</h3><table>
    <tr><th>configuration</th><th style="text-align:right">body area at peak<br>(mean det F)</th>
    <th style="text-align:right">1st-pct particle</th>
    <th style="text-align:right">visible footprint</th></tr>`;
  rk.forEach(k => {
    const v = r[k];
    h += `<tr><td>${k}</td><td class="num">${(100*v.worst_mean).toFixed(1)}%</td>
      <td class="num">${v.worst_p01.toFixed(3)}</td>
      <td class="num">${(100*v.retained_area_min).toFixed(1)}%</td></tr>`;
  });
  h += `</table>`;
  document.getElementById("ablate").innerHTML = h;
})();

// ---- density ladder ----------------------------------------------------------------------------
(function(){
  const keys = Object.keys(D.buoy.runs).filter(k => k.startsWith("rho_"));
  const W=440,H=200,L=52,R=14,Tp=14,B=34;
  const rows = keys.map(k => D.buoy.runs[k]);
  const xs = rows.map(r => r.rho), ys = rows.map(r => r.rest_depth_final);
  const xmin=0, xmax=1.8, ymin=Math.min(-0.02,Math.min(...ys)), ymax=Math.max(...ys)*1.15;
  const X = v => L + (v-xmin)/(xmax-xmin)*(W-L-R);
  const Y = v => Tp + (v-ymin)/(ymax-ymin)*(H-Tp-B);
  let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;background:var(--pan);border:1px solid var(--line);border-radius:8px">`;
  s += `<line x1="${L}" y1="${Y(0)}" x2="${W-R}" y2="${Y(0)}" stroke="#3d5566" stroke-dasharray="4 4"/>`;
  s += `<text x="${L+4}" y="${Y(0)-6}" fill="#7f8ea3" font-size="10">waterline</text>`;
  s += `<polyline fill="none" stroke="#6fd3ee" stroke-width="2.2" points="${
        rows.map(r=>`${X(r.rho)},${Y(r.rest_depth_final)}`).join(" ")}"/>`;
  rows.forEach(r => {
    s += `<circle cx="${X(r.rho)}" cy="${Y(r.rest_depth_final)}" r="5" fill="#6fd3ee"/>`;
    s += `<text x="${X(r.rho)}" y="${Y(r.rest_depth_final)-11}" fill="#dfe6ee" font-size="10"
           text-anchor="middle">${(100*r.submerged_final).toFixed(0)}% under</text>`;
    s += `<text x="${X(r.rho)}" y="${H-14}" fill="#7f8ea3" font-size="10.5"
           text-anchor="middle">&rho;=${r.rho}</text>`;
  });
  s += `<text x="6" y="${Tp+10}" fill="#7f8ea3" font-size="10">at surface</text>`;
  s += `<text x="6" y="${H-B+4}" fill="#7f8ea3" font-size="10">deep</text>`;
  s += `</svg>`;
  document.getElementById("ladder").innerHTML = s;
})();

// ---- regression scatter ------------------------------------------------------------------------
(function(){
  const METRICS = [["spread_width","spread width"],["pile_height","pile height"],
                   ["repose_angle","angle of repose (deg)"]];
  const COL = {snow:"#f2f6fc",sand:"#ffd24d",fluid:"#4db6ff",elastic:"#ff4d4d"};
  let cur = 0;
  const bt = document.getElementById("mbtns");
  bt.innerHTML = METRICS.map((m,i)=>`<button class="mbtn${i===0?' on':''}" data-i="${i}">${m[1]}</button>`).join("");
  function draw(){
    const key = METRICS[cur][0];
    const W=760,H=340,L=58,R=16,Tp=14,B=44;
    const pts=[];
    D.regress.mats.forEach(m => D.regress.scenes.forEach(sc => {
      const k = sc+"/"+m;
      pts.push({m,sc,a:D.regress.before[k][key],b:D.regress.after[k][key]});
    }));
    const lim = Math.max(...pts.map(p=>Math.max(p.a,p.b)))*1.09 + 1e-6;
    const X = v => L + v/lim*(W-L-R), Y = v => H-B - v/lim*(H-Tp-B);
    let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;background:var(--pan);border:1px solid var(--line);border-radius:8px">`;
    s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(lim)}" y2="${Y(lim)}" stroke="#3d5566" stroke-dasharray="5 5"/>`;
    s += `<line x1="${L}" y1="${Y(0)}" x2="${W-R}" y2="${Y(0)}" stroke="#2a3444"/>`;
    s += `<line x1="${L}" y1="${Tp}" x2="${L}" y2="${Y(0)}" stroke="#2a3444"/>`;
    for(let i=0;i<=4;i++){const v=lim*i/4;
      s += `<text x="${X(v)}" y="${H-B+16}" fill="#7f8ea3" font-size="10" text-anchor="middle">${v.toFixed(2)}</text>`;
      s += `<text x="${L-7}" y="${Y(v)+3}" fill="#7f8ea3" font-size="10" text-anchor="end">${v.toFixed(2)}</text>`;}
    s += `<text x="${(L+W-R)/2}" y="${H-8}" fill="#dfe6ee" font-size="11.5" text-anchor="middle">OLD physics</text>`;
    s += `<text x="14" y="${(Tp+Y(0))/2}" fill="#dfe6ee" font-size="11.5" text-anchor="middle"
            transform="rotate(-90 14 ${(Tp+Y(0))/2})">NEW physics</text>`;
    pts.forEach(p => {
      const off = Math.hypot(p.a-p.b);
      s += `<circle cx="${X(p.a)}" cy="${Y(p.b)}" r="${off>0.03*lim?7:5.2}" fill="${COL[p.m]}"
             stroke="#0a0e14" stroke-width="1"><title>${p.sc} / ${p.m}
old ${p.a.toFixed(3)} -> new ${p.b.toFixed(3)}</title></circle>`;
      if(off > 0.06*lim)
        s += `<text x="${X(p.a)+10}" y="${Y(p.b)+4}" fill="#7f8ea3" font-size="10">${p.sc}/${p.m}</text>`;
    });
    s += `</svg>`;
    s += `<div class="legend">` + Object.keys(COL).map(m =>
      `<span><i class="dot" style="background:${COL[m]}"></i>${m}</span>`).join("") +
      `<span style="color:#7f8ea3">hover a point for its numbers</span></div>`;
    document.getElementById("regress").innerHTML = s;
  }
  bt.onclick = e => { if(!e.target.dataset.i) return;
    cur = +e.target.dataset.i;
    [...bt.children].forEach((c,i)=>c.classList.toggle("on", i===cur)); draw(); };
  draw();
})();

// ---- signature table ---------------------------------------------------------------------------
(function(){
  let h = `<table><tr><th style="width:62px"></th><th>signature</th><th>measurement</th></tr>`;
  D.sig.forEach(r => {
    h += `<tr><td><span class="tag ${r.ok?'pass':'fail'}">${r.ok?'PASS':'FAIL'}</span></td>
      <td>${r.name}${r.new?'<span class="tag newsig">NEW</span>':''}</td>
      <td class="mono" style="color:var(--mut);font-size:12px">${r.detail}</td></tr>`;
  });
  h += `</table>`;
  document.getElementById("sigtable").innerHTML = h;
})();
</script>
"""

frontpct = 100 * (ab["water"]["new (canonical)"]["front_speed"]
                  / ab["water"]["revert both"]["front_speed"] - 1)
wall_peak = " &middot; ".join(
    "%s %.2f &rarr; %.2f" % (s, wf[s]["before"]["peak"], wf[s]["after"]["peak"])
    for s in ("drop", "dam", "slam"))
wall_res = " &middot; ".join(
    "%s %.4f &rarr; %.4f" % (s, wf[s]["before"]["residual"], wf[s]["after"]["residual"])
    for s in ("drop", "dam", "slam"))

html = (HTML.replace("__DATA__", json.dumps(DATA)).replace("__BASE__", BASE)
        .replace("__FRONTPCT__", "%.0f" % frontpct)
        .replace("__WALLPEAK__", wall_peak).replace("__WALLRES__", wall_res))
open(os.path.join(HERE, "bespoke_page.html"), "w", encoding="utf-8").write(html)
print("wrote bespoke_page.html", len(html) // 1024, "KB")
