"""Build the bespoke task page for train-one-nn-to-mimic-viscosity-and-st.

This is the Track A exemplar: a page designed around what THIS result actually needs a reader to see,
rather than a fixed card stack.

The design decision, and the reason this page exists: the honest story of this run is a trap. The
held-out corner's trajectory RMSE (0.245) does not look catastrophic, but its shape is completely wrong
(roundness 0.19 against a ground truth of 0.69) -- a vertical spike and a compact blob share a centre of
mass, so the scalar reads fine while the physics is broken. On the old page the RMSE heatmap and the
roundness heatmap were two separate PNGs several scrolls apart, and nobody would ever put them side by
side and notice.

So the centrepiece is ONE grid with a metric toggle. Flipping it redraws the same 25 cells from "almost
all fine" to "the whole top-right is wrong". The toggle IS the finding.

Data is read from metrics.json and inlined, so the page is fully self-contained (a sandboxed iframe with
no same-origin access and a strict CSP -- no CDNs, no fetch).

Grid orientation, verified against the trained-corner values in metrics['edge']:
    grid[row][col]  ->  row = m_st index (surface tension), col = m_visc index (viscosity)
Rendered with surface tension increasing UPWARD, so row 4 is drawn first.
"""
import json, os

RUN = 'runs/material-variants/train-one-nn-to-mimic-viscosity-and-st'
BASE = '/api/data/learning-taichi/' + RUN

m = json.load(open(os.path.join(RUN, 'metrics.json'), encoding='utf-8'))

payload = {
    'mv': m['m_viscs'],
    'ms': m['m_sts'],
    'rmse': m['rmse_grid'],
    'rnn': m['round_grid'],
    'rgt': m['round_gt_grid'],
    'muLow': m['MU_LOW'], 'muHigh': m['MU_HIGH'], 'sigMax': m['SIGMA_MAX'], 'stP': m['ST_P'],
    'trained': [[0, 0], [4, 0], [0, 4]],          # [row(st), col(visc)] of the three trained corners
    'heldOut': [4, 4],
    'edge': {k: {'m': v['m'], 'rmse': v['net_vs_true'], 'noise': v['true_self_noise']}
             for k, v in m['edge'].items()},
    'ho': m['held_out'],
    'base': BASE,
}

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0a0e14;color:#dfe6ee;
  font-family:-apple-system,BlinkMacSystemFont,system-ui,"Segoe UI",sans-serif;
  font-size:14px;line-height:1.55;padding:22px 20px 30px}
h2{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;text-transform:uppercase;
  letter-spacing:.09em;color:#7f8ea3;margin:0 0 12px;font-weight:600}
.verdict{max-width:78ch;margin:0 0 26px;font-size:15px;color:#c8d3e0}
.verdict b{color:#e8eef6}
.win{color:#5fd39a}.bad{color:#ff8f6b}
.panel{border:1px solid #1c2430;border-radius:6px;background:#0d131b;padding:18px;margin:0 0 22px}
.tabs{display:flex;gap:0;margin:0 0 16px;border:1px solid #22303f;border-radius:5px;overflow:hidden;width:max-content;max-width:100%}
.tab{padding:9px 16px;cursor:pointer;background:#0f1620;color:#8b9aad;border:0;font-size:13px;
  font-family:inherit;border-right:1px solid #22303f;transition:background .12s,color .12s}
.tab:last-child{border-right:0}
.tab:hover{color:#cfd9e6}
.tab.on{background:#17b0d422;color:#6fd3ee;font-weight:600}
.layout{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}
.gridwrap{display:flex;gap:9px;align-items:stretch}
.ylab{writing-mode:vertical-rl;transform:rotate(180deg);font-size:11px;color:#7f8ea3;
  text-align:center;letter-spacing:.05em;padding:4px 0}
.gcol{display:flex;flex-direction:column;gap:7px}
.grid{display:grid;grid-template-columns:repeat(5,58px);grid-auto-rows:58px;gap:4px}
.cell{position:relative;border-radius:4px;cursor:pointer;border:1.5px solid transparent;
  display:flex;align-items:center;justify-content:center;font-size:11px;color:#06090d;font-weight:700;
  transition:transform .1s,border-color .1s}
.cell:hover{transform:scale(1.07);border-color:#dfe6ee;z-index:2}
.cell.sel{border-color:#6fd3ee;z-index:2}
.cell .mark{position:absolute;top:2px;right:4px;font-size:11px;line-height:1;text-shadow:0 0 3px #000}
.xlab{font-size:11px;color:#7f8ea3;text-align:center;letter-spacing:.05em}
.readout{min-width:250px;flex:1;max-width:400px}
.readout .rtitle{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#6fd3ee;margin:0 0 10px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13px;margin:0 0 14px}
.kv .k{color:#8b9aad}
.kv .v{font-family:ui-monospace,Menlo,monospace;text-align:right}
.note{font-size:13px;color:#9fb0c4;border-left:2px solid #22303f;padding:2px 0 2px 12px;margin:14px 0 0}
.legend{display:flex;align-items:center;gap:8px;font-size:11px;color:#7f8ea3;margin:12px 0 0}
.bar{height:9px;width:150px;border-radius:2px}
.callout{border:1px solid #4a2a22;background:#170f0c;border-radius:6px;padding:16px 18px;margin:0 0 22px}
.callout h3{margin:0 0 8px;font-size:14px;color:#ff8f6b}
.cmp{display:flex;gap:26px;flex-wrap:wrap;margin:12px 0 0}
.cmp div{font-size:13px}
.cmp .big{font-family:ui-monospace,Menlo,monospace;font-size:21px;display:block;margin-top:2px}
video{width:100%;max-width:640px;border-radius:5px;border:1px solid #1c2430;margin:14px 0 0;display:block}
table{border-collapse:collapse;font-size:13px;width:100%;max-width:620px}
th,td{text-align:right;padding:7px 12px;border-bottom:1px solid #18202b}
th{color:#7f8ea3;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
td:first-child,th:first-child{text-align:left;font-family:ui-monospace,Menlo,monospace}
.cap{font-size:12px;color:#7f8ea3;margin:9px 0 0;max-width:640px}
</style></head><body>

<div class="verdict">
  One network learned the <b>entire</b> liquid material &mdash; stress, capillary force, and volume
  evolution &mdash; selected by a two-scalar descriptor. Trained on <b>three corners</b>, held out the
  fourth. At the trained corners it is <span class="win">edge-exact</span>. In the interior and at the
  held-out corner it is <span class="bad">stable but not physical</span>.
</div>

<div class="panel">
  <h2>The same 25 cells, two metrics &mdash; flip between them</h2>
  <div class="tabs">
    <button class="tab on" data-k="rmse">Trajectory RMSE</button>
    <button class="tab" data-k="shape">Shape error</button>
  </div>
  <div class="layout">
    <div class="gridwrap">
      <div class="ylab">surface tension &rarr;</div>
      <div class="gcol">
        <div class="grid" id="grid"></div>
        <div class="xlab">viscosity &rarr;</div>
      </div>
    </div>
    <div class="readout">
      <div class="rtitle" id="rtitle">Click any cell</div>
      <div class="kv" id="kv"></div>
      <div class="legend"><span id="lo"></span><div class="bar" id="bar"></div><span id="hi"></span></div>
      <p class="note" id="note"></p>
    </div>
  </div>
</div>

<div class="callout">
  <h3>&#9888; The trap this page exists to show</h3>
  <div>At the held-out corner the trajectory RMSE looks survivable. The shape is completely wrong. A
  vertical spike and a settled blob <b>share a centre of mass</b>, so the distance-to-truth scalar reads
  fine while the physics is broken.</div>
  <div class="cmp">
    <div>Trajectory RMSE<span class="big" id="hoR"></span></div>
    <div>Roundness, learned<span class="big bad" id="hoN"></span></div>
    <div>Roundness, ground truth<span class="big win" id="hoG"></span></div>
  </div>
  <video controls muted loop playsinline id="hov"></video>
  <p class="cap">The held-out corner (high viscosity, high surface tension). Left: the whole learned
  material. Right: the ground-truth liquid. Judge a learned simulator by shape and motion against ground
  truth &mdash; never by one scalar.</p>
</div>

<div class="panel">
  <h2>The three trained corners &mdash; edge-exactness</h2>
  <table id="edge"><thead><tr><th>corner</th><th>descriptor</th><th>RMSE vs truth</th>
    <th>simulator self-noise</th></tr></thead><tbody></tbody></table>
  <p class="cap">Self-noise is the same ground-truth simulator run against itself, i.e. the floor any
  learned model could reach. The learned material sits near it at every trained corner.</p>
</div>

<script>
const D = __DATA__;
let metric = 'rmse', sel = null;

const shape = (r,c) => Math.abs(D.rnn[r][c] - D.rgt[r][c]);
function vals(k){const o=[];for(let r=0;r<5;r++)for(let c=0;c<5;c++)o.push(k==='rmse'?D.rmse[r][c]:shape(r,c));return o;}
function ramp(t){ // deep teal (good) -> amber -> hot orange (bad); readable on dark, not red/green
  t=Math.max(0,Math.min(1,t));
  const stops=[[16,58,74],[26,140,150],[168,190,90],[232,150,70],[232,92,60]];
  const x=t*(stops.length-1), i=Math.min(Math.floor(x),stops.length-2), f=x-i;
  const a=stops[i], b=stops[i+1];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;
}
const isTrained=(r,c)=>D.trained.some(t=>t[0]===r&&t[1]===c);
const isHeld=(r,c)=>D.heldOut[0]===r&&D.heldOut[1]===c;

function draw(){
  const v=vals(metric), lo=Math.min(...v), hi=Math.max(...v);
  const g=document.getElementById('grid'); g.innerHTML='';
  for(let r=4;r>=0;r--){                                  // surface tension increases upward
    for(let c=0;c<5;c++){
      const val = metric==='rmse' ? D.rmse[r][c] : shape(r,c);
      const d=document.createElement('div');
      d.className='cell'+(sel&&sel[0]===r&&sel[1]===c?' sel':'');
      d.style.background=ramp((val-lo)/(hi-lo||1));
      d.textContent=val.toFixed(2);
      if(isTrained(r,c)) d.innerHTML+='<span class="mark">&#9733;</span>';
      if(isHeld(r,c))    d.innerHTML+='<span class="mark">&#10007;</span>';
      d.onclick=()=>{sel=[r,c];draw();};
      g.appendChild(d);
    }
  }
  document.getElementById('lo').textContent=lo.toFixed(3);
  document.getElementById('hi').textContent=hi.toFixed(3);
  document.getElementById('bar').style.background=
    `linear-gradient(90deg,${ramp(0)},${ramp(.25)},${ramp(.5)},${ramp(.75)},${ramp(1)})`;
  document.getElementById('note').innerHTML = metric==='rmse'
    ? 'Centre-of-mass distance to the true liquid. It rises toward the held-out corner, but never screams &mdash; nothing here says "this cell is not a liquid". <b>This is the metric that lies.</b>'
    : 'How far the drop\'s roundness sits from the ground truth\'s. The whole high-viscosity, high-surface-tension region lights up. <b>Same 25 runs &mdash; the shape metric knows they are broken.</b>';
  if(sel){
    const [r,c]=sel;
    document.getElementById('rtitle').textContent=
      `m = (${D.mv[c].toFixed(2)} visc, ${D.ms[r].toFixed(2)} st)`
      + (isTrained(r,c)?'  ★ trained':(isHeld(r,c)?'  ✗ held out':'  interpolated'));
    document.getElementById('kv').innerHTML=
      `<span class="k">viscosity &mu;</span><span class="v">${(D.muLow+(D.muHigh-D.muLow)*D.mv[c]).toFixed(3)}</span>`+
      `<span class="k">surface tension &sigma;</span><span class="v">${(D.sigMax*Math.pow(D.ms[r],D.stP)).toFixed(4)}</span>`+
      `<span class="k">trajectory RMSE</span><span class="v">${D.rmse[r][c].toFixed(4)}</span>`+
      `<span class="k">roundness, learned</span><span class="v">${D.rnn[r][c].toFixed(3)}</span>`+
      `<span class="k">roundness, truth</span><span class="v">${D.rgt[r][c].toFixed(3)}</span>`+
      `<span class="k">shape error</span><span class="v">${shape(r,c).toFixed(3)}</span>`;
  }
}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); metric=b.dataset.k; draw();
});

document.getElementById('hoR').textContent=D.ho.rmse.toFixed(3);
document.getElementById('hoN').textContent=D.ho.round_nn.toFixed(2);
document.getElementById('hoG').textContent=D.ho.round_gt.toFixed(2);
document.getElementById('hov').src=D.base+'/heldout_corner.mp4';

const NAMES={ll:'low visc, low ST',hl:'high visc, low ST',lh:'low visc, high ST'};
document.querySelector('#edge tbody').innerHTML=Object.entries(D.edge).map(([k,v])=>
  `<tr><td>${NAMES[k]||k}</td><td>(${v.m[0].toFixed(1)}, ${v.m[1].toFixed(1)})</td>`+
  `<td>${v.rmse.toFixed(4)}</td><td>${v.noise.toExponential(1)}</td></tr>`).join('');

sel=[4,4]; draw();
</script></body></html>"""

html = HTML.replace('__DATA__', json.dumps(payload))

mp = os.path.join(RUN, 'manifest.json')
man = json.load(open(mp, encoding='utf-8'))
man['custom_html'] = html
open(mp, 'w', encoding='utf-8').write(json.dumps(man, indent=2, ensure_ascii=False) + '\n')
print('wrote custom_html: %d chars -> %s' % (len(html), mp))

# Also emit standalone, so the page can be opened (and reviewed) outside the dashboard iframe.
side = os.path.join(RUN, 'bespoke_page.html')
open(side, 'w', encoding='utf-8').write(html)
print('wrote standalone     -> %s' % side)
