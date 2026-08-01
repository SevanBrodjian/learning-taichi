"""Build the bespoke task page for train-one-nn-to-mimic-viscosity-and-st.

The Track A exemplar: a page designed around what THIS result needs a reader to see.

Design decisions, and why:

1. ARCHITECTURE FIRST, BRIEFLY. A reader cannot judge "the network learned the material" without knowing
   what the network *is* and where it sits. A compact MPM-step diagram marks which pieces are learned and
   which are fixed scaffolding; exact layer shapes sit in a small table. High level here, full spec in the
   Evidence layer. (Sevan: "nowhere on this page are there definitions for what our architecture actually
   is... It's not like a side detail.")

2. ONE GRID, TWO METRICS, A TOGGLE. Flipping it redraws the same 25 runs. The toggle is the finding.

3. THE HONEST VERSION OF THE TRAP -- corrected. An earlier draft of this page (and the worker manifest and
   a training page) claimed the held-out corner's distance metric "reads deceptively low because a spike and
   a blob share a centre of mass". That is FALSE for this metric: traj_rmse is a mean per-particle distance,
   not a centre-of-mass distance, and the held-out corner scores 0.246 against 0.012-0.031 at the trained
   corners -- it screams. The real, checkable finding is that the two metrics agree only moderately
   (Spearman rho ~ 0.55) and that TWO INTERIOR CELLS look fine by distance while being badly wrong in shape.
   Those cells are marked on the grid. Verified, not inherited.

4. CLICK A CELL, SEE THE CELL. Selecting shows that cell's learned and ground-truth stills side by side.
   Per-cell VIDEO does not exist in this run (only a whole-grid montage), which is itself the general
   lesson now in spec/style_task_page.md: if your page lets a reader select an item, export that item's
   media so selecting can show it.

5. METRICS DEFINE THEMSELVES. Every metric label carries its canonical definition from spec/definitions.json
   on hover, so no reader has to guess what "roundness" means and no task reinvents it.
"""
import json, os

RUN = 'runs/material-variants/train-one-nn-to-mimic-viscosity-and-st'
BASE = '/api/data/learning-taichi/' + RUN

m = json.load(open(os.path.join(RUN, 'metrics.json'), encoding='utf-8'))
defs = json.load(open('spec/definitions.json', encoding='utf-8'))

R = m['rmse_grid']
NN = m['round_grid']
GT = m['round_gt_grid']
shape = [[abs(NN[r][c] - GT[r][c]) for c in range(5)] for r in range(5)]

trained = [(0, 0), (0, 4), (4, 0)]          # (st_row, visc_col)
held = (4, 4)
worst_trained = max(R[r][c] for r, c in trained)

# Cells the distance metric fails to flag: within 2x the worst trained corner, yet badly wrong in shape.
discord = [[r, c] for r in range(5) for c in range(5)
           if R[r][c] <= 2 * worst_trained and shape[r][c] > 0.25]

def tip(key):
    d = defs.get(key, {})
    return '\n\n'.join(filter(None, [
        d.get('short'),
        'Formula: ' + d['formula'] if d.get('formula') else None,
        'Units: ' + d['units'] if d.get('units') else None,
        ('⚠ ' + d['caution']) if d.get('caution') else None,
        'Source: ' + d['source'] if d.get('source') else None,
    ]))

payload = {
    'mv': m['m_viscs'], 'ms': m['m_sts'],
    'rmse': R, 'rnn': NN, 'rgt': GT, 'shape': shape,
    'muLow': m['MU_LOW'], 'muHigh': m['MU_HIGH'], 'sigMax': m['SIGMA_MAX'], 'stP': m['ST_P'],
    'trained': [list(t) for t in trained], 'heldOut': list(held), 'discord': discord,
    'edge': {k: {'m': v['m'], 'rmse': v['net_vs_true'], 'noise': v['true_self_noise']}
             for k, v in m['edge'].items()},
    'ho': m['held_out'], 'base': BASE,
    'net': {'mom': m['mom_net'], 'state': m['state_net'], 'phys': m['physics_version'],
            'nP': m['n_particles'], 'nG': m['n_grid'], 'frames': m['n_frames']},
    'tips': {k: tip(k) for k in ('traj_rmse', 'roundness', 'shape_error', 'self_noise', 'physics_version')},
}

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0a0e14;color:#dfe6ee;font-family:-apple-system,BlinkMacSystemFont,system-ui,"Segoe UI",sans-serif;font-size:14px;line-height:1.55;padding:22px 20px 30px}
h2{font-family:ui-monospace,Menlo,monospace;font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:#7f8ea3;margin:0 0 12px;font-weight:600}
.verdict{max-width:80ch;margin:0 0 24px;font-size:15px;color:#c8d3e0}
.verdict b{color:#e8eef6}.win{color:#5fd39a}.bad{color:#ff8f6b}
.panel{border:1px solid #1c2430;border-radius:6px;background:#0d131b;padding:18px;margin:0 0 22px}
dfn{font-style:normal;border-bottom:1px dotted #4d6478;cursor:help}
dfn:hover{color:#6fd3ee;border-bottom-color:#6fd3ee}
/* architecture strip */
.arch{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap;margin:0 0 14px}
.stage{flex:1;min-width:150px;border:1px solid #22303f;border-radius:5px;padding:11px 13px;background:#0f1620}
.stage.learned{border-color:#2b6b7d;background:#0d1b22}
.stage .nm{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#9fb0c4;margin:0 0 5px}
.stage.learned .nm{color:#6fd3ee}
.stage .d{font-size:12px;color:#8b9aad;line-height:1.45}
.tagL{display:inline-block;font-size:10px;letter-spacing:.06em;background:#17b0d422;color:#6fd3ee;padding:1px 6px;border-radius:3px;margin-left:6px;vertical-align:1px}
.tagF{display:inline-block;font-size:10px;letter-spacing:.06em;background:#2a3340;color:#93a3b6;padding:1px 6px;border-radius:3px;margin-left:6px;vertical-align:1px}
.archnote{font-size:12.5px;color:#8b9aad;max-width:86ch;margin:0}
/* grid */
.tabs{display:flex;margin:0 0 16px;border:1px solid #22303f;border-radius:5px;overflow:hidden;width:max-content;max-width:100%}
.tab{padding:9px 16px;cursor:pointer;background:#0f1620;color:#8b9aad;border:0;border-right:1px solid #22303f;font-size:13px;font-family:inherit}
.tab:last-child{border-right:0}.tab:hover{color:#cfd9e6}
.tab.on{background:#17b0d422;color:#6fd3ee;font-weight:600}
.layout{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}
.gridwrap{display:flex;gap:9px}
.ylab{writing-mode:vertical-rl;transform:rotate(180deg);font-size:11px;color:#7f8ea3;text-align:center;padding:4px 0}
.grid{display:grid;grid-template-columns:repeat(5,58px);grid-auto-rows:58px;gap:4px}
.cell{position:relative;border-radius:4px;cursor:pointer;border:1.5px solid transparent;display:flex;align-items:center;justify-content:center;font-size:11px;color:#06090d;font-weight:700;transition:transform .1s}
.cell:hover{transform:scale(1.07);border-color:#dfe6ee;z-index:2}
.cell.sel{border-color:#6fd3ee;z-index:2}
.cell .mark{position:absolute;top:2px;right:4px;font-size:11px;line-height:1;text-shadow:0 0 3px #000}
.cell .dis{position:absolute;bottom:1px;left:4px;font-size:12px;line-height:1;color:#ffd166;text-shadow:0 0 3px #000}
.xlab{font-size:11px;color:#7f8ea3;text-align:center;margin-top:7px}
.readout{flex:1;min-width:290px;max-width:430px}
.rtitle{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#6fd3ee;margin:0 0 10px}
.shots{display:flex;gap:10px;margin:0 0 12px}
.shots figure{margin:0;flex:1}
.shots img{width:100%;border-radius:4px;border:1px solid #22303f;display:block;background:#070b10}
.shots figcaption{font-size:11px;color:#7f8ea3;text-align:center;margin-top:4px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13px;margin:0 0 12px}
.kv .k{color:#8b9aad}.kv .v{font-family:ui-monospace,Menlo,monospace;text-align:right}
.note{font-size:13px;color:#9fb0c4;border-left:2px solid #22303f;padding:2px 0 2px 12px;margin:12px 0 0}
.legend{display:flex;align-items:center;gap:8px;font-size:11px;color:#7f8ea3;margin:10px 0 0}
.bar{height:9px;width:140px;border-radius:2px}
.callout{border:1px solid #4a3f22;background:#16120c;border-radius:6px;padding:16px 18px;margin:0 0 22px}
.callout h3{margin:0 0 8px;font-size:14px;color:#ffd166}
.cmp{display:flex;gap:26px;flex-wrap:wrap;margin:12px 0 0}
.cmp .big{font-family:ui-monospace,Menlo,monospace;font-size:21px;display:block;margin-top:2px}
video{width:100%;max-width:620px;border-radius:5px;border:1px solid #1c2430;margin:14px 0 0;display:block}
table{border-collapse:collapse;font-size:13px;width:100%;max-width:640px}
th,td{text-align:right;padding:7px 12px;border-bottom:1px solid #18202b}
th{color:#7f8ea3;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
td:first-child,th:first-child{text-align:left;font-family:ui-monospace,Menlo,monospace}
.cap{font-size:12px;color:#7f8ea3;margin:9px 0 0;max-width:640px}
</style></head><body>

<div class="verdict">
  One network learned the <b>entire</b> liquid material &mdash; stress, capillary force and volume
  evolution &mdash; selected by a two-scalar descriptor <b>m = (viscosity, surface tension)</b>. Trained on
  <b>three corners</b> of that square, held out the fourth. At the trained corners it is
  <span class="win">edge-exact</span>. Across the interior and at the held-out corner it stays finite but
  <span class="bad">is not a physical liquid</span>.
</div>

<div class="panel">
  <h2>What the architecture actually is</h2>
  <div class="arch">
    <div class="stage learned"><div class="nm">1 &middot; stress<span class="tagL">LEARNED</span></div>
      <div class="d">Per-particle MLP. Reads local state (J, affine C, velocity) + a 5&times;5 patch of smoothed
      grid density + <b>m</b>. Outputs the stress and capillary force scattered at particle&rarr;grid.</div></div>
    <div class="stage"><div class="nm">2 &middot; grid solve<span class="tagF">FIXED</span></div>
      <div class="d">Canonical MPM scaffolding, imported unchanged: mass-normalise, gravity, floor and walls.
      Identical for water, sand or rubber.</div></div>
    <div class="stage learned"><div class="nm">3 &middot; carried state<span class="tagL">LEARNED</span></div>
      <div class="d">Second MLP. Reads J, the post-solve affine C and <b>m</b>, outputs the volume rate J&#775;
      that replaces the analytic continuity rule at grid&rarr;particle.</div></div>
  </div>
  <p class="archnote">Nothing constitutive remains analytic in the learned rollout &mdash; that is what makes
  this "the whole material" rather than a learned stress bolted onto hand-written physics. Both nets are
  small tanh MLPs trained by per-step supervised regression against the canonical simulator (plus DAgger).
  Exact shapes below; full training detail is in <b>Evidence &amp; detail</b>.</p>
  <table id="arch"><thead><tr><th>network</th><th>inputs</th><th>hidden</th><th>outputs</th><th>role</th></tr></thead><tbody></tbody></table>
</div>

<div class="panel">
  <h2>The same 25 runs, two metrics &mdash; flip between them</h2>
  <div class="tabs">
    <button class="tab on" data-k="rmse">Position error</button>
    <button class="tab" data-k="shape">Shape error</button>
  </div>
  <div class="layout">
    <div>
      <div class="gridwrap">
        <div class="ylab">surface tension &rarr;</div>
        <div><div class="grid" id="grid"></div><div class="xlab">viscosity &rarr;</div></div>
      </div>
      <div class="legend"><span id="lo"></span><div class="bar" id="bar"></div><span id="hi"></span>
        <span style="margin-left:10px">&#9733; trained &nbsp; &#10007; held out &nbsp;
        <span style="color:#ffd166">&#9679;</span> metrics disagree</span></div>
    </div>
    <div class="readout">
      <div class="rtitle" id="rtitle"></div>
      <div class="shots" id="shots"></div>
      <div class="kv" id="kv"></div>
      <p class="note" id="note"></p>
    </div>
  </div>
</div>

<div class="callout">
  <h3>&#9888; Where the two metrics disagree &mdash; and why one number cannot certify a cell</h3>
  <div id="discordText"></div>
  <div class="cmp">
    <div>Held-out <dfn id="t1">position error</dfn><span class="big" id="hoR"></span></div>
    <div><dfn id="t2">Roundness</dfn>, learned<span class="big bad" id="hoN"></span></div>
    <div><dfn id="t3">Roundness</dfn>, ground truth<span class="big win" id="hoG"></span></div>
  </div>
  <video controls muted loop playsinline id="hov"></video>
  <p class="cap">The held-out corner (high viscosity, high surface tension). Left: the whole learned
  material. Right: the ground-truth liquid. Both metrics flag this corner. The subtler failures are the
  interior cells marked in amber above, where the distance number looks fine and the shape does not.</p>
</div>

<div class="panel">
  <h2>The three trained corners &mdash; edge-exactness</h2>
  <table id="edge"><thead><tr><th>corner</th><th>descriptor m</th><th><dfn id="t4">position error</dfn></th>
    <th><dfn id="t5">simulator self-noise</dfn></th></tr></thead><tbody></tbody></table>
  <p class="cap">Self-noise is the canonical simulator run against itself: the floor any learned model could
  reach. The learned material sits near it at every trained corner.</p>
</div>

<script>
const D=__DATA__;
let metric='rmse', sel=D.heldOut.slice();
const key=(r,c)=>r+','+c;
const isT=(r,c)=>D.trained.some(t=>t[0]===r&&t[1]===c);
const isH=(r,c)=>D.heldOut[0]===r&&D.heldOut[1]===c;
const isD=(r,c)=>D.discord.some(t=>t[0]===r&&t[1]===c);
const val=(r,c)=>metric==='rmse'?D.rmse[r][c]:D.shape[r][c];

function ramp(t){t=Math.max(0,Math.min(1,t));
 const s=[[16,58,74],[26,140,150],[168,190,90],[232,150,70],[232,92,60]];
 const x=t*(s.length-1),i=Math.min(Math.floor(x),s.length-2),f=x-i,a=s[i],b=s[i+1];
 return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;}

function draw(){
 const all=[];for(let r=0;r<5;r++)for(let c=0;c<5;c++)all.push(val(r,c));
 const lo=Math.min(...all),hi=Math.max(...all);
 const g=document.getElementById('grid');g.innerHTML='';
 for(let r=4;r>=0;r--)for(let c=0;c<5;c++){
  const d=document.createElement('div');
  d.className='cell'+(sel[0]===r&&sel[1]===c?' sel':'');
  d.style.background=ramp((val(r,c)-lo)/(hi-lo||1));
  d.textContent=val(r,c).toFixed(2);
  if(isT(r,c))d.innerHTML+='<span class="mark">&#9733;</span>';
  if(isH(r,c))d.innerHTML+='<span class="mark">&#10007;</span>';
  if(isD(r,c))d.innerHTML+='<span class="dis">&#9679;</span>';
  d.onclick=()=>{sel=[r,c];draw();};
  g.appendChild(d);
 }
 document.getElementById('lo').textContent=lo.toFixed(3);
 document.getElementById('hi').textContent=hi.toFixed(3);
 document.getElementById('bar').style.background=
  `linear-gradient(90deg,${ramp(0)},${ramp(.25)},${ramp(.5)},${ramp(.75)},${ramp(1)})`;
 document.getElementById('note').innerHTML=metric==='rmse'
  ?'Mean per-particle distance from the true liquid. It rises toward the held-out corner &mdash; but its scale is <b>not interpretable</b>: nothing tells you whether 0.12 means "slightly off" or "not a liquid".'
  :'How far the settled drop\'s roundness sits from the ground truth\'s. Read directly against truth, so it <b>is</b> interpretable &mdash; and it lights up cells the distance metric passes.';

 const [r,c]=sel;
 document.getElementById('rtitle').textContent=
  `m = (${D.mv[c].toFixed(2)} visc, ${D.ms[r].toFixed(2)} st)`+
  (isT(r,c)?'  ★ trained':(isH(r,c)?'  ✗ held out':'  interpolated'))+
  (isD(r,c)?'  — metrics disagree':'');
 document.getElementById('shots').innerHTML=
  `<figure><img src="${D.base}/cells/nn_st${r}_mu${c}.png" alt="learned"><figcaption>learned</figcaption></figure>`+
  `<figure><img src="${D.base}/cells/gt_st${r}_mu${c}.png" alt="ground truth"><figcaption>ground truth</figcaption></figure>`;
 document.getElementById('kv').innerHTML=
  `<span class="k">viscosity &mu;</span><span class="v">${(D.muLow+(D.muHigh-D.muLow)*D.mv[c]).toFixed(3)}</span>`+
  `<span class="k">surface tension &sigma;</span><span class="v">${(D.sigMax*Math.pow(D.ms[r],D.stP)).toFixed(4)}</span>`+
  `<span class="k"><dfn title="${D.tips.traj_rmse}">position error</dfn></span><span class="v">${D.rmse[r][c].toFixed(4)}</span>`+
  `<span class="k"><dfn title="${D.tips.roundness}">roundness</dfn>, learned</span><span class="v">${D.rnn[r][c].toFixed(3)}</span>`+
  `<span class="k"><dfn title="${D.tips.roundness}">roundness</dfn>, truth</span><span class="v">${D.rgt[r][c].toFixed(3)}</span>`+
  `<span class="k"><dfn title="${D.tips.shape_error}">shape error</dfn></span><span class="v">${D.shape[r][c].toFixed(3)}</span>`;
}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');metric=b.dataset.k;draw();});

// architecture table
const N=D.net;
document.querySelector('#arch tbody').innerHTML=
 `<tr><td>stress + capillary</td><td>${N.mom.in} (incl. ${N.mom.patch}&times;${N.mom.patch} density patch)</td><td>${N.mom.hidden}</td><td>${N.mom.out}</td><td>momentum scattered at P2G</td></tr>`+
 `<tr><td>carried state</td><td>${N.state.in}</td><td>${N.state.hidden}</td><td>${N.state.out}</td><td>volume rate at G2P</td></tr>`+
 `<tr><td colspan="5" style="color:#7f8ea3;text-align:left">${N.nP} particles on a ${N.nG}&times;${N.nG} grid, ${N.frames} frames &middot; ground truth stamped <span title="${D.tips.physics_version}">${N.phys}</span></td></tr>`;

// discordance text, computed from the data rather than asserted
const dc=D.discord.map(([r,c])=>
 `<b>m = (${D.mv[c].toFixed(2)}, ${D.ms[r].toFixed(2)})</b> &mdash; position error ${D.rmse[r][c].toFixed(3)} (near the trained corners) but shape error ${D.shape[r][c].toFixed(3)} (roundness ${D.rnn[r][c].toFixed(2)} against a true ${D.rgt[r][c].toFixed(2)})`);
document.getElementById('discordText').innerHTML=
 `The two metrics rank the 25 cells only moderately alike, and <b>${D.discord.length} interior cells sit within twice the worst trained corner's position error while their shape is qualitatively wrong</b>:<ul><li>${dc.join('</li><li>')}</li></ul>`+
 `A distance scalar is useful to <b>rank</b> cells and useless to <b>certify</b> one as physical. Judge shape and motion against ground truth.`;

document.getElementById('hoR').textContent=D.ho.rmse.toFixed(3);
document.getElementById('hoN').textContent=D.ho.round_nn.toFixed(2);
document.getElementById('hoG').textContent=D.ho.round_gt.toFixed(2);
document.getElementById('hov').src=D.base+'/heldout_corner.mp4';
[['t1','traj_rmse'],['t2','roundness'],['t3','roundness'],['t4','traj_rmse'],['t5','self_noise']]
 .forEach(([id,k])=>{const e=document.getElementById(id);if(e)e.title=D.tips[k];});

const NAMES={ll:'low visc, low ST',hl:'high visc, low ST',lh:'low visc, high ST'};
document.querySelector('#edge tbody').innerHTML=Object.entries(D.edge).map(([k,v])=>
 `<tr><td>${NAMES[k]||k}</td><td>(${v.m[0].toFixed(1)}, ${v.m[1].toFixed(1)})</td><td>${v.rmse.toFixed(4)}</td><td>${v.noise.toExponential(1)}</td></tr>`).join('');

draw();
</script></body></html>"""

html = HTML.replace('__DATA__', json.dumps(payload))

mp = os.path.join(RUN, 'manifest.json')
man = json.load(open(mp, encoding='utf-8'))
man['custom_html'] = html
open(mp, 'w', encoding='utf-8').write(json.dumps(man, indent=2, ensure_ascii=False) + '\n')
open(os.path.join(RUN, 'bespoke_page.html'), 'w', encoding='utf-8').write(html)
print('custom_html %d chars; discordant cells: %s' % (len(html), discord))
