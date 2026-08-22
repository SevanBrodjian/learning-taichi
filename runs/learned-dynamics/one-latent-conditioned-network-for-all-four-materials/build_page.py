"""Build the bespoke task page.

The page is designed around ONE thing a reader must walk away knowing: capacity and cost were both
measured, and the widths where the network is affordable are not the widths where it is accurate.
Everything on the page exists to make that collision visible rather than to display outputs.

The central control is a single WIDTH slider that drives both panels at once -- the cost bar against
the two real-time lines, and the per-material accuracy bars. Dragging it is the finding: the cost bar
crosses the budget long before the accuracy bars come down. Two separate figures would have made the
reader do that comparison in their head.

    .venv/Scripts/python.exe runs/.../build_page.py
"""
import json
import pathlib
import subprocess

RUN = pathlib.Path(__file__).resolve().parent
BASE = ("/api/data/learning-taichi/runs/learned-dynamics/"
        "one-latent-conditioned-network-for-all-four-materials/")
MATS = ["fluid", "elastic", "snow", "sand"]
COLS = {"fluid": "#4db6ff", "elastic": "#ff9d5c", "snow": "#e6ecff", "sand": "#ffd24d"}


def load(p, default=None):
    q = RUN / p
    return json.loads(q.read_text()) if q.exists() else default


def main():
    bench = load("verify/out/bench_cost.json")
    train = load("train/train_stats.json", {"results": {}})
    # keyed by a LABEL, not by width: two different networks here happen to share width 64, and a
    # tab reading "width 64" beside another reading "width 32" would quietly imply they are the same
    # family. Only one of the two had the (slow) signature suite run against it, and the tab says so.
    evals = {}
    for p in sorted((RUN / "eval").glob("eval_h*.json")):
        e = json.loads(p.read_text())
        lab = f"width {e['hidden']}, {e.get('layers', 1)} hidden layer" + \
              ("s" if e.get("layers", 1) > 1 else "")
        evals[lab] = e
    parity = load("verify/out/parity.json", {})
    metrics = load("metrics.json", {})

    ws = bench["width_sweep"]
    def pick(pred):
        return sorted([x for x in ws if pred(x)], key=lambda x: x["hidden"])
    f32 = pick(lambda x: x["mode"] == "nn" and not x["f16"] and x["weights"] == "uniform"
               and not x["variant"].startswith("dyn"))
    dyn = pick(lambda x: x["variant"].startswith("dyn"))
    f16 = pick(lambda x: x["f16"])
    sto = pick(lambda x: x["weights"] == "storage")
    an = [x for x in ws if x["mode"] == "analytic"][0]
    budget = bench["budget"]

    data = {
        "budget": budget,
        "analytic": {"full": an["us_per_substep_full"], "g2p": an["us_per_substep_g2p"]},
        "cost": {"f32": [[x["hidden"], x["us_per_substep_full"], x["us_per_substep_g2p"]] for x in f32],
                 "dyn": [[x["hidden"], x["us_per_substep_g2p"]] for x in dyn],
                 "f16": [[x["hidden"], x["us_per_substep_full"], x["us_per_substep_g2p"]] for x in f16],
                 "sto": [[x["hidden"], x["us_per_substep_full"], x["us_per_substep_g2p"]] for x in sto]},
        "acc": {w: {m: {"stress": sum(train["results"][w]["held_out"][m][k]
                                      for k in ("tau00", "tau01", "tau11")) / 3.0,
                        "plastic": sum(train["results"][w]["held_out"][m][k]
                                       for k in ("dS00", "dS01", "dS11", "dJp")) / 4.0}
                    for m in MATS}
                for w in train["results"]},
        "params": {w: train["results"][w]["n_params"] for w in train["results"]},
        "evals": {w: {"sig": evals[w]["signatures"], "summary": evals[w]["signature_summary"],
                      "traj": evals[w]["trajectory"], "hidden": evals[w]["hidden"],
                      "layers": evals[w].get("layers", 1)} for w in evals},
        "parity": parity,
        "port": bench["analytic_port"],
        "device": bench["device"],
        "dispatch_floor_us": bench["dispatch_floor"][-1]["ns_per_dispatch"] / 1000.0,
        "batching": bench["batching"],
        "zcodes": bench["net_shape"]["z_codes"],
        "zsep": bench["net_shape"]["z_sep"], "zjit": bench["net_shape"]["z_jitter"],
        "verdict": metrics.get("verdict", {}),
        "base": BASE,
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    (RUN / "bespoke_page.html").write_text(html, encoding="utf-8")
    # Verify the SCRIPT parses after the entity decoding an iframe srcdoc performs. A page can render
    # perfectly from disk and be blank in the dashboard because `&&` or `<` came back different.
    body = html.split("<script>")[-1].split("</script>")[0]
    dec = (body.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
           .replace("&quot;", '"').replace("&#39;", "'"))
    tmp = RUN / ".decoded_check.js"
    tmp.write_text(dec, encoding="utf-8")
    try:
        r = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
        print("node --check on the DECODED script:", "OK" if r.returncode == 0 else r.stderr[:400])
    except FileNotFoundError:
        print("node not available -- decoded-script check skipped")
    finally:
        tmp.unlink(missing_ok=True)
    print("wrote", RUN / "bespoke_page.html", len(html), "bytes")
    return html


TEMPLATE = r"""<!doctype html><meta charset="utf-8">
<style>
:root{--bg:#0a0e14;--fg:#dfe6ee;--mut:#7f8ea3;--acc:#6fd3ee;--pan:#0f141c;--line:#222c3c;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:4px 2px 28px}
h2{font-size:19px;margin:26px 0 8px;color:var(--fg);font-weight:650}
h3{font-size:14.5px;margin:16px 0 6px;color:var(--acc);font-weight:600;letter-spacing:.02em}
p{margin:8px 0}
.mut{color:var(--mut)}
.verdict{background:linear-gradient(180deg,#131b26,#0f141c);border:1px solid #2b3a4e;border-left:3px solid var(--acc);border-radius:8px;padding:14px 16px;margin:6px 0 4px}
.verdict b{color:#fff}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.pan{background:var(--pan);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.ctl{display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:#101823;border:1px solid #2b3a4e;border-radius:8px;padding:10px 14px;margin:14px 0}
input[type=range]{flex:1;min-width:180px;accent-color:var(--acc)}
.big{font-variant-numeric:tabular-nums;font-weight:700;font-size:21px}
table{border-collapse:collapse;width:100%;font-size:12.6px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:700;letter-spacing:.03em}
.pass{background:#12331f;color:#7ee787;border:1px solid #1f5c33}
.fail{background:#3a1a1a;color:#ff9d9d;border:1px solid #6b2b2b}
.na{background:#26252f;color:#b9a6e0;border:1px solid #453d5c}
video{width:100%;border-radius:6px;border:1px solid var(--line);background:#000}
.tabs{display:flex;gap:6px;margin:10px 0 6px;flex-wrap:wrap}
.tab{padding:4px 11px;border-radius:6px;border:1px solid #2b3a4e;background:#131b26;color:var(--mut);cursor:pointer;font-size:12.5px}
.tab.on{background:#17364a;color:#cfefff;border-color:#3b6f8c}
.note{border-left:2px solid #55617a;padding-left:11px;color:var(--mut);font-size:13px;margin:10px 0}
svg{display:block;max-width:100%}
code{background:#131b26;padding:1px 5px;border-radius:4px;font-size:12.4px;color:#cfe3f2}
.legend{font-size:11.5px;color:var(--mut);display:flex;gap:14px;flex-wrap:wrap;margin-top:6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
</style>
<div id="app"></div>
<script>
const D = __DATA__;
const MATS=["fluid","elastic","snow","sand"];
const COL={fluid:"#4db6ff",elastic:"#ff9d5c",snow:"#e6ecff",sand:"#ffd24d"};
const WIDTHS = D.cost.f32.map(r=>r[0]);
const ACCW = Object.keys(D.acc).map(Number).sort((a,b)=>a-b);
const nearestAcc = w => ACCW.reduce((a,b)=>Math.abs(b-w)<Math.abs(a-w)?b:a, ACCW[0]);
const fmt=(x,n)=>x.toFixed(n===undefined?2:n);
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

// widest width whose WHOLE-SOLVER cost still fits each budget
function maxFit(b){let m=0;for(const [h,f] of D.cost.f32) if(f<=b) m=Math.max(m,h);return m;}
const FITQ = maxFit(D.budget.us_per_substep_quarter_gpu);
const FITF = maxFit(D.budget.us_per_substep_full_gpu);

function svgCost(sel){
  const W=470,H=250,L=44,R=14,T=14,B=34;
  const xs=v=>L+(Math.log2(v)-3)/(Math.log2(256)-3)*(W-L-R);
  const ymax=62, ys=v=>H-B-(v/ymax)*(H-T-B);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  s+=`<rect x="${L}" y="${ys(D.budget.us_per_substep_quarter_gpu)}" width="${W-L-R}" height="${H-B-ys(D.budget.us_per_substep_quarter_gpu)}" fill="#7ee787" opacity="0.07"/>`;
  for(const v of [0,10,20,30,40,50,60]) s+=`<line x1="${L}" y1="${ys(v)}" x2="${W-R}" y2="${ys(v)}" stroke="#222c3c"/><text x="${L-6}" y="${ys(v)+4}" fill="#7f8ea3" font-size="9" text-anchor="end">${v}</text>`;
  for(const h of [8,16,32,64,128,256]) s+=`<text x="${xs(h)}" y="${H-14}" fill="#7f8ea3" font-size="9" text-anchor="middle">${h}</text>`;
  const line=(pts,c,dash)=>{let d="";pts.forEach((p,i)=>{d+=(i?"L":"M")+xs(p[0])+" "+ys(Math.min(p[1],ymax));});return `<path d="${d}" fill="none" stroke="${c}" stroke-width="2" ${dash?'stroke-dasharray="5 4"':''}/>`;};
  s+=`<line x1="${L}" y1="${ys(D.budget.us_per_substep_full_gpu)}" x2="${W-R}" y2="${ys(D.budget.us_per_substep_full_gpu)}" stroke="#ffd24d" stroke-width="1.4" stroke-dasharray="6 4"/>`;
  s+=`<text x="${L+5}" y="${ys(D.budget.us_per_substep_full_gpu)-5}" fill="#ffd24d" font-size="9.5">60 fps, whole GPU (${D.budget.us_per_substep_full_gpu} us)</text>`;
  s+=`<line x1="${L}" y1="${ys(D.budget.us_per_substep_quarter_gpu)}" x2="${W-R}" y2="${ys(D.budget.us_per_substep_quarter_gpu)}" stroke="#7ee787" stroke-width="1.4" stroke-dasharray="6 4"/>`;
  s+=`<text x="${L+5}" y="${ys(D.budget.us_per_substep_quarter_gpu)-5}" fill="#7ee787" font-size="9.5">60 fps, quarter GPU (${D.budget.us_per_substep_quarter_gpu} us)</text>`;
  s+=`<line x1="${L}" y1="${ys(D.analytic.full)}" x2="${W-R}" y2="${ys(D.analytic.full)}" stroke="#ff8f8f" stroke-width="1.6"/>`;
  s+=`<text x="${W-R-4}" y="${ys(D.analytic.full)+12}" fill="#ff8f8f" font-size="9.5" text-anchor="end">the analytic solver it replaces</text>`;
  s+=line(D.cost.sto.map(r=>[r[0],r[1]]),"#ff9d5c",false);
  s+=line(D.cost.f16.map(r=>[r[0],r[1]]),"#c792ea",false);
  s+=line(D.cost.f32.map(r=>[r[0],r[1]]),"#6fd3ee",false);
  const cur=D.cost.f32.find(r=>r[0]===sel)||D.cost.f32[0];
  s+=`<line x1="${xs(sel)}" y1="${T}" x2="${xs(sel)}" y2="${H-B}" stroke="#dfe6ee" stroke-width="1" opacity="0.45"/>`;
  s+=`<circle cx="${xs(sel)}" cy="${ys(Math.min(cur[1],ymax))}" r="5" fill="#fff"/>`;
  s+=`<text x="${L}" y="${H-2}" fill="#7f8ea3" font-size="9.5">hidden width</text>`;
  s+=`<text x="6" y="${T+4}" fill="#7f8ea3" font-size="9.5">us/substep</text></svg>`;
  return s;
}

function svgAcc(sel){
  const w=nearestAcc(sel);
  const W=470,H=250,L=64,R=90,T=18,B=30;
  const bw=(H-T-B)/4;
  const xs=v=>L+Math.min(v,1.25)/1.25*(W-L-R);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(const v of [0,0.25,0.5,0.75,1.0,1.25]) s+=`<line x1="${xs(v)}" y1="${T}" x2="${xs(v)}" y2="${H-B}" stroke="#222c3c"/><text x="${xs(v)}" y="${H-14}" fill="#7f8ea3" font-size="9" text-anchor="middle">${v}</text>`;
  s+=`<line x1="${xs(1)}" y1="${T}" x2="${xs(1)}" y2="${H-B}" stroke="#ff8f8f" stroke-width="1.3" stroke-dasharray="4 3"/>`;
  s+=`<text x="${xs(1)+4}" y="${T+10}" fill="#ff8f8f" font-size="9">no better than predicting the mean</text>`;
  MATS.forEach((m,i)=>{
    const y=T+i*bw, a=D.acc[w][m];
    s+=`<text x="${L-8}" y="${y+bw*0.42}" fill="${COL[m]}" font-size="10.5" text-anchor="end">${m}</text>`;
    s+=`<rect x="${L}" y="${y+bw*0.16}" width="${xs(a.stress)-L}" height="${bw*0.30}" fill="${COL[m]}" opacity="0.95"/>`;
    s+=`<rect x="${L}" y="${y+bw*0.52}" width="${xs(a.plastic)-L}" height="${bw*0.30}" fill="${COL[m]}" opacity="0.42"/>`;
    s+=`<text x="${xs(Math.min(a.stress,1.25))+5}" y="${y+bw*0.36}" fill="#cfe3f2" font-size="9.5">${fmt(a.stress)} stress</text>`;
    s+=`<text x="${xs(Math.min(a.plastic,1.25))+5}" y="${y+bw*0.72}" fill="#8fa3bf" font-size="9.5">${fmt(a.plastic)} plastic</text>`;
  });
  s+=`<text x="${L}" y="${H-2}" fill="#7f8ea3" font-size="9.5">held-out one-step error / that material's own spread &nbsp; (trained width ${w})</text></svg>`;
  return s;
}

function svgCliff(mode){
  const W=470,H=230,L=44,R=14,T=16,B=32;
  const xs=v=>L+(Math.log2(v)-3)/(Math.log2(256)-3)*(W-L-R);
  const ymax=55, ys=v=>H-B-(v/ymax)*(H-T-B);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(const v of [0,10,20,30,40,50]) s+=`<line x1="${L}" y1="${ys(v)}" x2="${W-R}" y2="${ys(v)}" stroke="#222c3c"/><text x="${L-6}" y="${ys(v)+4}" fill="#7f8ea3" font-size="9" text-anchor="end">${v}</text>`;
  for(const h of [8,16,32,64,128,256]) s+=`<text x="${xs(h)}" y="${H-12}" fill="#7f8ea3" font-size="9" text-anchor="middle">${h}</text>`;
  const line=(pts,c,dash)=>{let d="";pts.forEach((p,i)=>{d+=(i?"L":"M")+xs(p[0])+" "+ys(Math.min(p[1],ymax));});return `<path d="${d}" fill="none" stroke="${c}" stroke-width="2.1" ${dash?'stroke-dasharray="5 4"':''}/>`;};
  s+=`<line x1="${L}" y1="${ys(D.analytic.g2p)}" x2="${W-R}" y2="${ys(D.analytic.g2p)}" stroke="#ff8f8f" stroke-width="1.4"/>`;
  if(mode!=="unrolled") s+=line(D.cost.dyn,"#ffd24d",true);
  s+=line(D.cost.f32.map(r=>[r[0],r[2]]),"#6fd3ee",false);
  s+=`<text x="6" y="${T+4}" fill="#7f8ea3" font-size="9.5">us/substep, G2P only</text>`;
  s+=`<text x="${L}" y="${H-1}" fill="#7f8ea3" font-size="9.5">hidden width</text></svg>`;
  return s;
}

function sigTable(w){
  const e=D.evals[w];
  if(!e||!e.sig||!e.sig.length) return `<p class="note">The golden signature suite was not run against
  this network. It is ~35 full rollouts of the learned simulator and takes far longer than the
  trajectory table below; it was run against the width-32 single-layer net only. The trajectory
  numbers below ARE this network's.</p>`;
  let rows="";
  for(const r of e.sig){
    const cls=r.na?"na":(r.pass?"pass":"fail"), lab=r.na?"N/A":(r.pass?"PASS":"FAIL");
    rows+=`<tr><td><span class="pill ${cls}">${lab}</span></td><td>${esc(r.name)}</td><td class="mut" style="font-size:11.6px">${esc(r.detail)}</td></tr>`;
  }
  return `<table><thead><tr><th></th><th>golden signature (canonical thresholds, unmodified)</th><th>detail</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function trajTable(w){
  const e=D.evals[w]; if(!e) return "";
  let rows="";
  for(const r of e.traj){
    const bad=r.traj_rmse_learned>0.05;
    rows+=`<tr><td>${r.scene}</td><td style="color:${COL[r.material]}">${r.material}</td>`
      +`<td class="n" style="color:${bad?'#ff9d9d':'#dfe6ee'}">${r.traj_rmse_learned.toFixed(4)}</td>`
      +`<td class="n mut">${r.traj_rmse_oracle.toExponential(1)}</td>`
      +`<td class="n mut">${r.ic_nudge_band.toExponential(1)}</td>`
      +`<td class="n">${r.spread_learned.toFixed(3)} <span class="mut">vs ${r.spread_canonical.toFixed(3)}</span></td>`
      +`<td class="n">${r.repose_learned.toFixed(0)}&deg; <span class="mut">vs ${r.repose_canonical.toFixed(0)}&deg;</span></td>`
      +`<td>${r.learned_stable?'<span class="pill pass">stable</span>':'<span class="pill fail">blew up</span>'}</td></tr>`;
  }
  return `<table><thead><tr><th>scene</th><th>material</th><th>traj_rmse learned</th><th>oracle floor</th><th>1e-7 nudge band</th><th>spread width</th><th>repose angle</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
}

let SEL = 64, CLIFF="both";
// default to whichever evaluated network actually has signature rows
let SIGW = Object.keys(D.evals).find(k=>D.evals[k].sig && D.evals[k].sig.length) || Object.keys(D.evals)[0];
function render(){
  const cur=D.cost.f32.find(r=>r[0]===SEL);
  const accw=nearestAcc(SEL);
  const fitsQ=cur[1]<=D.budget.us_per_substep_quarter_gpu, fitsF=cur[1]<=D.budget.us_per_substep_full_gpu;
  const v=D.verdict||{};
  document.getElementById("app").innerHTML = `
<div class="verdict">
  <p style="margin:0 0 6px"><b>Both halves answered, and they do not meet.</b> One network with one shared
  weight set, told which material it is only by a 4-dimensional code, runs the constitutive model for
  fluid, rubber, snow and sand on WebGPU. <b>Cost is not the obstacle</b>: the analytic four-material
  law it replaces already costs ${fmt(D.analytic.full)} us per substep, and a width-16 network costs
  ${fmt(D.cost.f32.find(r=>r[0]===16)[1])} us, so the seam is essentially <b>free up to width
  ${FITQ}</b> at a quarter of this GPU and affordable to width ${FITF} on the whole device.
  <b>Capacity is the obstacle.</b> ${esc(v.capacity_line||"See the accuracy panel and the signature table below.")}</p>
  <p class="mut" style="margin:0;font-size:12.8px">One RTX 4090, Chromium/WebGPU, 128&times;128 grid,
  8192 particles, four materials on one shared grid at the timestep they force
  (dt = ${D.budget.dt}). Every number is that setup and no other.</p>
</div>

<div class="ctl">
  <span class="mut">hidden width</span>
  <span class="big" style="color:var(--acc);min-width:52px">${SEL}</span>
  <input type="range" min="0" max="${WIDTHS.length-1}" value="${WIDTHS.indexOf(SEL)}" id="wslider">
  <span class="big" style="color:${fitsQ?'#7ee787':(fitsF?'#ffd24d':'#ff8f8f')}">${fmt(cur[1])} us</span>
  <span class="mut">${fitsQ?'fits 60 fps at a quarter GPU':(fitsF?'needs the WHOLE GPU':'misses real time')}</span>
  <span class="mut">&nbsp;|&nbsp; ${D.params[String(accw)]||"?"} parameters</span>
</div>

<div class="two">
  <div class="pan"><h3>What it costs</h3>${svgCost(SEL)}
    <div class="legend"><span><span class="sw" style="background:#6fd3ee"></span>f32, weights in a uniform buffer</span>
    <span><span class="sw" style="background:#c792ea"></span>f16 weights</span>
    <span><span class="sw" style="background:#ff9d5c"></span>f32, weights in a storage buffer</span></div>
  </div>
  <div class="pan"><h3>What it learns</h3>${svgAcc(SEL)}
    <p class="mut" style="font-size:11.8px;margin:6px 0 0">Solid bar: the stress the network predicts.
    Faded bar: the plastic state update (snow's clamp, sand's return map). Both in units of that
    material's own spread, so 1.0 means the network has learned nothing about it.</p>
  </div>
</div>
<p class="note">Drag the slider. The cost bar crosses the quarter-GPU budget at width ${FITQ}; the
accuracy bars are still well above where they need to be there, and they are still coming down at the
largest width measured. That gap is the result.</p>

<h2>Does it still behave like the material? The golden signatures</h2>
<p class="mut" style="margin-top:0">These are <code>sim/physics/signatures.py</code>, unmodified, with
canonical thresholds, run against the learned simulator instead of the canonical one. Nothing was tuned
to them.</p>
<div class="tabs">${Object.keys(D.evals).map(w=>`<span class="tab ${w===SIGW?'on':''}" data-sig="${w}">${w}</span>`).join("")}</div>
${sigTable(SIGW)}

<h2>Learned vs canonical, same scene, same seed</h2>
<div class="two">
  <div><p class="mut" style="margin-top:0">The angle-of-repose scene, which separates all four materials:
  fluid runs flat, rubber and snow keep the seeded slope, sand yields to a finite one.</p>
  <video src="${D.base}learned_vs_canonical_heap.mp4" controls muted loop playsinline></video></div>
  <div><p class="mut" style="margin-top:0">A dropped disk. Top row canonical, bottom row the one shared
  network.</p>
  <video src="${D.base}learned_vs_canonical_drop.mp4" controls muted loop playsinline></video></div>
</div>

<h2>Trajectory error, against references that make it mean something &nbsp;<span class="mut"
  style="font-size:13px;font-weight:400">&mdash; ${esc(SIGW)}</span></h2>
<p class="mut" style="margin-top:0"><code>traj_rmse</code> is the mean per-particle Euclidean distance
(not an RMS). It is quoted beside the ORACLE floor -- the same scaffolding running the exact analytic
law, so the best any network could do -- and the 1e-7 initial-condition nudge band, which is how far
canonical moves from a perturbation with no physical meaning. Against zero it would be meaningless.</p>
${trajTable(SIGW)}

<h2>The cost cliff is the compiler, not the hardware</h2>
<div class="two">
  <div class="pan">${svgCliff(CLIFF)}
  <div class="tabs" style="margin-top:8px">
    <span class="tab ${CLIFF==='unrolled'?'on':''}" data-cliff="unrolled">just the shipped shader</span>
    <span class="tab ${CLIFF==='both'?'on':''}" data-cliff="both">with the un-unrollable control</span>
  </div></div>
  <div>
  <p style="margin-top:0">The G2P cost is a clean straight line in width up to 88, then jumps 2.2&times;
  between width 88 and 92 for 1.05&times; the arithmetic, then is a straight line again on a steeper
  slope.</p>
  <p>Flip to the control: the identical shader with the hidden-loop bound read from a uniform, so the
  compiler cannot unroll it. Below the cliff the unrolled version is 2-3&times; faster. Above it, the two
  curves lie on top of each other. The compiler was fully unrolling the loop, that was worth a factor of
  two to three, and somewhere between 88 and 92 hidden units it stopped.</p>
  <p class="mut" style="font-size:12.8px">This refines rather than confirms the earlier guess that the
  width cliff was register spilling. A spill would degrade past the un-unrolled cost, not land exactly on
  it.</p>
  </div>
</div>

<h2>Is the WebGPU solver the canonical physics?</h2>
<p class="mut" style="margin-top:0">A cost measurement against a baseline that is not the real physics is
a cost measurement of something else. The analytic WGSL path was scored against canonical Taichi on the
same initial condition, per material.</p>
<table><thead><tr><th>material</th><th>traj_rmse, WGSL vs canonical Taichi</th><th>1e-7 IC-nudge band</th><th>final spread width</th><th></th></tr></thead><tbody>
${D.port.map(r=>`<tr><td style="color:${COL[r.material]}">${r.material}</td><td class="n">${r.traj_rmse_vs_canonical.toExponential(2)}</td><td class="n mut">${r.ic_nudge_band.toExponential(2)}</td><td class="n">${r.final_spread_wgsl.toFixed(4)} <span class="mut">vs ${r.final_spread_canonical.toFixed(4)}</span></td><td>${r.traj_rmse_vs_canonical<=3*r.ic_nudge_band?'<span class="pill pass">within the band</span>':'<span class="pill fail">outside</span>'}</td></tr>`).join("")}
</tbody></table>
${D.parity && D.parity.mlp ? `<p style="margin-top:12px">Host-vs-shader parity for the trained network:
max absolute disagreement <b>${D.parity.mlp.max_abs.toExponential(2)}</b> over
${D.parity.mlp.n} real feature vectors (relative to an output spread of
${D.parity.mlp.out_sd.toExponential(2)}), in f32.
${D.parity.mlp_f16 ? `In f16 the same comparison gives ${D.parity.mlp_f16.max_abs.toExponential(2)}.` : ""}</p>` : ""}

<h2>The seam, and what is learned</h2>
<svg viewBox="0 0 720 128" width="100%" style="margin:6px 0 2px">
  <defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#55617a"/></marker></defs>
  <g font-size="11.5" fill="#dfe6ee" text-anchor="middle">
    <rect x="8" y="34" width="120" height="42" rx="6" fill="#131b26" stroke="#2b3a4e"/><text x="68" y="52">P2G</text><text x="68" y="66" font-size="9.5" fill="#7f8ea3">scatters cached stress</text>
    <rect x="164" y="34" width="120" height="42" rx="6" fill="#131b26" stroke="#2b3a4e"/><text x="224" y="52">grid update</text><text x="224" y="66" font-size="9.5" fill="#7f8ea3">gravity, walls, friction</text>
    <rect x="320" y="34" width="120" height="42" rx="6" fill="#131b26" stroke="#2b3a4e"/><text x="380" y="52">G2P gather</text><text x="380" y="66" font-size="9.5" fill="#7f8ea3">advect, trial F</text>
    <rect x="476" y="20" width="236" height="70" rx="6" fill="#17364a" stroke="#6fd3ee" stroke-width="1.6"/>
    <text x="594" y="38" fill="#cfefff" font-weight="700">THE ONE NETWORK</text>
    <text x="594" y="54" font-size="9.8" fill="#cfe3f2">in: S00 S01 S11, C(4), v(2), Jp, z(4)</text>
    <text x="594" y="68" font-size="9.8" fill="#cfe3f2">out: stress(3), dS(3), dJp(1)</text>
    <text x="594" y="82" font-size="9.5" fill="#8fbdd6">fused into G2P -- no extra dispatch</text>
    <path d="M130,55 L160,55" stroke="#55617a" marker-end="url(#a)"/>
    <path d="M286,55 L316,55" stroke="#55617a" marker-end="url(#a)"/>
    <path d="M442,55 L472,55" stroke="#55617a" marker-end="url(#a)"/>
    <path d="M594,92 L594,110 L20,110 L20,80" stroke="#3b6f8c" fill="none" stroke-dasharray="4 3" marker-end="url(#a)"/>
    <text x="300" y="123" font-size="9.5" fill="#7f8ea3">stress cached for the next substep's P2G</text>
  </g>
</svg>
<div class="two" style="margin-top:8px">
  <div class="pan"><h3>z_m is IDENTITY, and only identity</h3>
  <p class="mut" style="font-size:13px;margin-top:4px">Four fixed codes at the corners of a regular
  simplex in R<sup>4</sup>, separation ${fmt(D.zsep,3)}, jittered by ${fmt(D.zjit,3)} every training batch
  so the network learns a neighbourhood rather than four point lookups. Never updated during a rollout.</p>
  <p class="mut" style="font-size:13px"><b>This is a label, not a physical axis.</b> Four structurally
  unrelated materials have no ground truth anywhere between their codes, so nothing here is a claim about
  interpolating between them, and none was tested.</p></div>
  <div class="pan"><h3>The carried state is HISTORY</h3>
  <p class="mut" style="font-size:13px;margin-top:4px">Per particle, updated every substep, in the known
  parameterisation: the symmetric stretch <code>S</code> of the deformation gradient and the plastic
  record <code>Jp</code>. The network predicts their <i>update</i>, which is what makes snow's clamp and
  sand's return mapping learned rather than applied analytically afterwards.</p>
  <p class="mut" style="font-size:13px">A free learned latent state was deliberately out of scope: it
  needs backprop through a long rollout, which is this project's documented failure mode.</p></div>
</div>

<h2 style="margin-bottom:2px">Scope</h2>
<p class="note" style="margin-top:6px">One GPU, one browser, one grid resolution, one particle count,
one scene family. The training used per-step supervision only, with no data aggregation round, so
rollout drift is not controlled for. Trajectory numbers are single-seed. The four materials are the
canonical four and nothing was tested between them.</p>
`;
  const sl=document.getElementById("wslider");
  if(sl) sl.oninput=e=>{SEL=WIDTHS[+e.target.value];render();};
  document.querySelectorAll("[data-cliff]").forEach(t=>t.onclick=()=>{CLIFF=t.dataset.cliff;render();});
  document.querySelectorAll("[data-sig]").forEach(t=>t.onclick=()=>{SIGW=t.dataset.sig;render();});
}
render();
</script>
"""

if __name__ == "__main__":
    main()
