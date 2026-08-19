"""Assemble the standalone task page: bespoke_page.html + the custom_html the manifest carries.

The page must be fully self-contained (sandboxed iframe, no network, no CDNs), and it hosts a LIVE
WebGPU demo, so the engine, the trained weights and the measured numbers are all inlined.
"""
import json
import pathlib

RUN = pathlib.Path(__file__).resolve().parent
WEB = RUN / "web"

CSS = """
:root{--bg:#0a0e14;--fg:#dfe6ee;--mut:#7f8ea3;--acc:#6fd3ee;--warm:#ff9d5c;--good:#7ee787;
      --amber:#ffd24d;--vio:#b48ead;--line:#212a36;--card:#111823;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:22px 20px 40px}
h2{font-size:20px;margin:34px 0 6px;letter-spacing:.2px}
h3{font-size:15.5px;margin:20px 0 6px;color:var(--acc);font-weight:600}
p{margin:8px 0 12px;color:#c9d4e0}
.sub{color:var(--mut);font-size:13.2px;margin-top:-2px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.verdict{border:1px solid #2b3a4a;border-left:4px solid var(--warm);background:linear-gradient(180deg,#131c27,#0e141d);
  border-radius:10px;padding:16px 18px}
.verdict .big{font-size:18.5px;line-height:1.5;color:#eef4fa;margin:0 0 8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(124px,1fr));gap:8px;margin:13px 0 4px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:8px 12px}
.tile .v{font-size:17.5px;font-weight:650;font-family:ui-monospace,Menlo,Consolas,monospace}
.tile .k{font-size:10.8px;color:var(--mut);text-transform:uppercase;letter-spacing:.55px;margin-top:3px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
button,select{background:#1a2432;color:var(--fg);border:1px solid #2e3b4c;border-radius:7px;
  padding:6px 12px;font:inherit;font-size:13.4px;cursor:pointer}
button:hover,select:hover{background:#22303f;border-color:#3d4d61}
button.on{background:#1d3a46;border-color:var(--acc);color:#cdf1fb}
button:disabled{opacity:.45;cursor:default}
label{font-size:13px;color:var(--mut)}
.demo{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:720px){.demo{grid-template-columns:1fr}}
.pane{background:#070a0f;border:1px solid var(--line);border-radius:9px;overflow:hidden}
.pane h4{margin:0;padding:8px 11px;font-size:13.2px;background:#131b26;border-bottom:1px solid var(--line);
  font-weight:600;display:flex;justify-content:space-between;align-items:center;gap:8px}
.pane canvas{display:block;width:100%;aspect-ratio:1/1;max-height:270px;background:#070a0f}
.pane .ft{padding:6px 11px;font:12.2px ui-monospace,Menlo,Consolas,monospace;color:var(--mut);
  border-top:1px solid var(--line);min-height:28px}
.pill{font:11.4px ui-monospace,monospace;padding:2px 7px;border-radius:99px;border:1px solid #2e3b4c;color:var(--mut)}
.pill.an{border-color:#2d5f70;color:var(--acc)} .pill.nn{border-color:#7a4a2a;color:var(--warm)}
table{border-collapse:collapse;width:100%;font-size:13.2px;font-family:ui-monospace,Menlo,Consolas,monospace}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.4px}
.yes{color:var(--good)} .no{color:#ff8f8f} .warnc{color:var(--amber)}
.note{border-left:3px solid var(--amber);background:#181509;padding:10px 14px;border-radius:0 8px 8px 0;
  font-size:13.6px;color:#e6dcc4;margin:14px 0}
.scope{border-left:3px solid var(--mut);background:#10151c;padding:10px 14px;border-radius:0 8px 8px 0;
  font-size:13.4px;color:#b3c0cf;margin:14px 0}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.6px;color:var(--mut);margin:6px 0 2px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
svg{display:block;max-width:100%;height:auto}
.flowrow{display:flex;align-items:stretch;gap:0;flex-wrap:wrap}
.fbox{flex:1;min-width:150px;border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#0d131c}
.fbox.learned{border-color:var(--warm);background:#1a1209}
.fbox .t{font-weight:650;font-size:13.6px} .fbox .d{font-size:12.2px;color:var(--mut);margin-top:3px}
.arrow{display:flex;align-items:center;color:var(--mut);padding:0 9px;font-size:19px}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:2px 16px 4px;margin:12px 0}
details[open]{padding-bottom:14px}
summary{cursor:pointer;padding:11px 0;font-size:14.6px;color:var(--acc);font-weight:600;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:'B8  ';color:var(--mut)} details[open] summary::before{content:'BE  '}
.err{color:#ff8f8f;padding:10px;font-family:ui-monospace,monospace;font-size:12.6px}
"""


def main():
    m = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    engine = (WEB / "mpm-nn-webgpu.js").read_text(encoding="utf-8")
    params = (WEB / "params.js").read_text(encoding="utf-8")
    weights = (WEB / "nnweights.js").read_text(encoding="utf-8")

    # only what the page draws, so the inlined blob stays small
    slim = {
        "sweep": [{k: r[k] for k in ("n", "hidden", "full_us_analytic", "full_us_nn",
                                     "full_us_nnsparse", "grid_us", "null_pg_us", "flops_per_cell",
                                     "net_floats", "particles_per_cell")} for r in m["sweep"]],
        "width_scan": m.get("width_scan", []),
        "compaction": m.get("compaction", []),
        "occupancy": [{k: r[k] for k in ("n", "occupied_cells", "occupied_frac",
                                         "occupied_workgroups", "occupied_wg_frac")}
                      for r in m["occupancy"]],
        "survival": [{k: r[k] for k in ("net", "hidden", "frames_60fps_tracked", "reason",
                                        "sim_seconds_tracked", "final_dist")} for r in m["survival"]],
        "accuracy": m["accuracy"], "verdict": m["verdict"], "grav": m["gravity_vs_error"],
        "budget_us": m["budget_us_per_substep"],
        "budget_us_q": m["budget_us_per_substep_quarter_gpu"],
        "spf": m["substeps_per_frame"], "floor_us": m["dispatch_floor_us"],
        "device": m["device"], "physics_version": m["physics_version"],
        "analytic_check": m["analytic_port_check"], "self_noise": m["canonical_self_noise"],
        "inference": m["inference_verification"], "traj": m["traj"],
    }

    body = PAGE_JS.replace("__DATA__", json.dumps(slim))
    # Without this the standalone file is sniffed as windows-1252 and every micro sign and em dash in
    # the inlined JS renders as mojibake. Harmless inside the dashboard's srcdoc frame, required here.
    html = (
        '<meta charset="utf-8">\n'
        f"<style>{CSS}</style>\n"
        f"<script>{params}</script>\n<script>{weights}</script>\n<script>{engine}</script>\n"
        f'<div class="wrap" id="root"></div>\n<script>{body}</script>\n'
    )
    (RUN / "bespoke_page.html").write_text(html, encoding="utf-8")
    print("wrote bespoke_page.html", len(html) // 1024, "KB")
    return html


PAGE_JS = r"""
(function(){
'use strict';
var D = __DATA__;
var R = document.getElementById('root');
var W = [8,16,32,64];
var WCOL = {8:'#a3d977',16:'#6fd3ee',32:'#b48ead',64:'#ff9d5c'};
var NS = D.sweep.map(function(r){return r.n;}).filter(function(v,i,a){return a.indexOf(v)===i;}).sort(function(a,b){return a-b;});
function pick(n,h){ for (var i=0;i<D.sweep.length;i++){var r=D.sweep[i]; if(r.n===n&&r.hidden===h) return r;} return null; }
function fmt(x,d){ return (x===null||x===undefined)?'-':Number(x).toFixed(d===undefined?2:d); }
function el(tag, cls, html){ var e=document.createElement(tag); if(cls)e.className=cls; if(html!==undefined)e.innerHTML=html; return e; }

// ---------------------------------------------------------------- tiny SVG chart helper
function chart(opt){
  var w=opt.w||960, h=opt.h||400, ml=opt.ml||74, mr=opt.mr||18, mt=opt.mt||16, mb=opt.mb||52;
  var iw=w-ml-mr, ih=h-mt-mb;
  var lx = opt.logx!==false, ly = opt.logy!==false;
  var X0=opt.x0, X1=opt.x1, Y0=opt.y0, Y1=opt.y1;
  function px(v){ var t = lx ? (Math.log(v)-Math.log(X0))/(Math.log(X1)-Math.log(X0)) : (v-X0)/(X1-X0); return ml+t*iw; }
  function py(v){ v=Math.max(v,ly?Y0:v); var t = ly ? (Math.log(v)-Math.log(Y0))/(Math.log(Y1)-Math.log(Y0)) : (v-Y0)/(Y1-Y0); return mt+ih-t*ih; }
  var s = ['<svg viewBox="0 0 '+w+' '+h+'" width="'+w+'" height="'+h+'" font-family="ui-monospace,Menlo,Consolas,monospace">'];
  s.push('<rect x="'+ml+'" y="'+mt+'" width="'+iw+'" height="'+ih+'" fill="#0b1119" stroke="#212a36"/>');
  (opt.yticks||[]).forEach(function(t){ var y=py(t); s.push('<line x1="'+ml+'" y1="'+y+'" x2="'+(ml+iw)+'" y2="'+y+'" stroke="#1c2430"/>'
    +'<text x="'+(ml-8)+'" y="'+(y+4)+'" fill="#7f8ea3" font-size="11.5" text-anchor="end">'+t+'</text>'); });
  (opt.xticks||[]).forEach(function(t){ var x=px(t); s.push('<line x1="'+x+'" y1="'+mt+'" x2="'+x+'" y2="'+(mt+ih)+'" stroke="#1c2430"/>'
    +'<text x="'+x+'" y="'+(mt+ih+18)+'" fill="#7f8ea3" font-size="11.5" text-anchor="middle">'+t+'</text>'); });
  s.push('<text x="'+(ml+iw/2)+'" y="'+(h-10)+'" fill="#9fb0c2" font-size="12.5" text-anchor="middle">'+(opt.xlabel||'')+'</text>');
  s.push('<text transform="translate(15,'+(mt+ih/2)+') rotate(-90)" fill="#9fb0c2" font-size="12.5" text-anchor="middle">'+(opt.ylabel||'')+'</text>');
  return { s:s, px:px, py:py, ml:ml, mt:mt, iw:iw, ih:ih,
    line:function(pts,col,dash,wd){ var d=pts.map(function(p,i){return (i?'L':'M')+px(p[0]).toFixed(1)+' '+py(p[1]).toFixed(1);}).join(' ');
      s.push('<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="'+(wd||2.2)+'"'+(dash?' stroke-dasharray="'+dash+'"':'')+' stroke-linejoin="round"/>'); },
    dots:function(pts,col,r){ pts.forEach(function(p){ s.push('<circle cx="'+px(p[0]).toFixed(1)+'" cy="'+py(p[1]).toFixed(1)+'" r="'+(r||3.6)+'" fill="'+col+'"/>'); }); },
    hline:function(v,col,lab,dash,anchor){ var y=py(v); s.push('<line x1="'+ml+'" y1="'+y+'" x2="'+(ml+iw)+'" y2="'+y+'" stroke="'+col+'" stroke-width="1.8"'+(dash?' stroke-dasharray="'+dash+'"':'')+'/>');
      if(lab){ var right = anchor==='end';
        s.push('<text x="'+(right?(ml+iw-7):(ml+7))+'" y="'+(y-7)+'" fill="'+col+'" font-size="12" text-anchor="'+(right?'end':'start')+'" paint-order="stroke" stroke="#0b1119" stroke-width="4">'+lab+'</text>'); } },
    text:function(x,y,t,col,size,anchor){ s.push('<text x="'+x+'" y="'+y+'" fill="'+(col||'#dfe6ee')+'" font-size="'+(size||12)+'" text-anchor="'+(anchor||'start')+'">'+t+'</text>'); },
    done:function(){ s.push('</svg>'); return s.join(''); } };
}

// ================================================================ verdict
var an = pick(NS[0],8);
var gAn = 0, gN = {};
D.sweep.forEach(function(r){ gAn += r.grid_us.analytic; });
gAn /= D.sweep.length;
W.forEach(function(h){ var a=D.sweep.filter(function(r){return r.hidden===h;});
  gN[h] = a.reduce(function(s,r){return s+r.grid_us.nn;},0)/a.length; });

// largest width that fits the QUARTER-GPU budget, and at what particle count
var bestQ = null;
D.verdict.rows.forEach(function(r){ if(r.max_width_nn_quarter_gpu){
  if(!bestQ || r.max_width_nn_quarter_gpu > bestQ.w || (r.max_width_nn_quarter_gpu===bestQ.w && r.n>bestQ.n))
    bestQ = {w:r.max_width_nn_quarter_gpu, n:r.n}; } });
var survBest = D.survival.reduce(function(a,b){ return b.frames_60fps_tracked > a.frames_60fps_tracked ? b : a; });
var survAtBest = bestQ ? D.survival.filter(function(s){return s.hidden===bestQ.w;})
  .reduce(function(a,b){return b.frames_60fps_tracked>a.frames_60fps_tracked?b:a;}) : null;

R.appendChild(el('div','verdict',
  '<p class="big">Replacing the whole grid update with a per-cell network costs <b>'+
  fmt(gN[16]/gAn,0)+'&times;</b> the analytic kernel at hidden width 16 and <b>'+fmt(gN[64]/gAn,0)+
  '&times;</b> at width 64. Assuming a quarter of this GPU, the largest width that still fits a 60 fps '+
  'frame is <b>'+(bestQ?bestQ.w:'none')+'</b>, at <b>'+(bestQ?bestQ.n.toLocaleString():'-')+
  ' particles</b>&nbsp;&mdash; and a network that size holds the fluid together for <b>'+
  (survAtBest?fmt(survAtBest.frames_60fps_tracked,0):'-')+' frames</b>.</p>'+
  '<p class="sub">The cost is also <b>latency-bound</b>: dispatching the kernel over 7 workgroups instead '+
  'of 256 changes nothing, so neither fewer particles nor skipping empty cells makes it cheaper. '+
  'One RTX 4090 in Chromium, 128&times;128 grid, canonical water.</p>'));

var tiles = el('div','tiles');
[['analytic grid update', fmt(gAn,3)+' &micro;s', 'per substep, isolated'],
 ['learned, width 16', fmt(gN[16],2)+' &micro;s', fmt(gN[16]/gAn,0)+'&times; the analytic kernel'],
 ['learned, width 64', fmt(gN[64],1)+' &micro;s', fmt(gN[64]/gAn,0)+'&times; the analytic kernel'],
 ['budget per substep', fmt(D.budget_us,1)+' &micro;s', D.spf+' substeps per 60 fps frame'],
 ['&hellip;at a quarter GPU', fmt(D.budget_us_q,1)+' &micro;s', 'the assumption in force here'],
 ['best rollout survival', fmt(survBest.frames_60fps_tracked,0)+' frames', 'width '+survBest.hidden+', '+fmt(survBest.sim_seconds_tracked,2)+' s of fluid']
].forEach(function(t){ tiles.appendChild(el('div','tile','<div class="v">'+t[1]+'</div><div class="k">'+t[0]+'</div><div class="k" style="text-transform:none;letter-spacing:0;color:#61707f">'+t[2]+'</div>')); });
R.appendChild(tiles);

// ================================================================ the seam
R.appendChild(el('h2',null,'What was replaced'));
R.appendChild(el('p',null,
  'P2G and G2P are untouched and analytic. The network replaced the whole grid update &mdash; the division '+
  'by node mass, gravity, the separating-wall test and the Coulomb friction cap. <b>All four are inside '+
  'it</b>: nothing is applied to its output afterwards, and the vector it emits is the node velocity G2P '+
  'gathers.'));
R.appendChild(el('div','card',
  '<div class="flowrow">'+
  '<div class="fbox"><div class="t">P2G &nbsp;<span class="pill">analytic</span></div>'+
  '<div class="d">particles scatter mass and momentum onto the grid</div></div>'+
  '<div class="arrow">&rarr;</div>'+
  '<div class="fbox learned"><div class="t">grid update &nbsp;<span class="pill nn">learned</span></div>'+
  '<div class="d">MLP 8&rarr;h&rarr;h&rarr;2, ReLU, evaluated once per cell per substep</div></div>'+
  '<div class="arrow">&rarr;</div>'+
  '<div class="fbox"><div class="t">G2P &nbsp;<span class="pill">analytic</span></div>'+
  '<div class="d">particles gather the velocity and its gradient back</div></div></div>'+
  '</div>'));
R.appendChild(el('details',null,
  '<summary>The exact inputs and outputs, and the four things the network has to reproduce</summary>'+
  '<div class="row" style="gap:22px">'+
  '<div style="flex:1;min-width:250px"><h3>in, per cell (8)</h3>'+
  '<div class="mono" style="font-size:12.8px;color:#c9d4e0">node mass, node momentum x, node momentum y<br>'+
  'wall flags: left, right, floor, ceiling<br>friction coefficient</div>'+
  '<p class="sub" style="margin-top:6px">Mass and momentum are in units of one particle mass. That scale is a '+
  'single number, uniform over the grid and known before the substep starts, so it is folded into the first '+
  'layer&rsquo;s weights &mdash; it is not the per-cell division the network has to learn.</p></div>'+
  '<div style="flex:1;min-width:250px"><h3>out, per cell (2)</h3>'+
  '<div class="mono" style="font-size:12.8px;color:#c9d4e0">node velocity x, node velocity y</div>'+
  '<h3>what it has to reproduce</h3>'+
  '<div style="font-size:13.2px;color:#c9d4e0">divide momentum by mass &middot; subtract gravity &middot; '+
  'zero the normal component at a wall, but only when moving into it &middot; cap the tangential component '+
  'by Coulomb friction</div></div></div>'));

// ================================================================ THE DEMO
R.appendChild(el('h2',null,'Watch it run'));
R.appendChild(el('p',null,
  'Live WebGPU, same seed, same analytic P2G and G2P, same substeps per drawn frame. The only difference '+
  'is the grid update. Change the width and watch what the learned kernel does to the water, and read the '+
  'microseconds each grid kernel is costing while it does it.'));
var demo = el('div','card');
demo.innerHTML =
  '<div class="row" style="margin-bottom:11px">'+
  '<label>hidden width</label><select id="dw"><option value="8">8</option><option value="16">16</option>'+
  '<option value="32">32</option><option value="64" selected>64</option></select>'+
  '<label>trained with</label><select id="dt"><option value="deriv" selected>derivative loss</option>'+
  '<option value="point">cell-wise loss</option></select>'+
  '<button id="dgo">pause</button><button id="drs">restart</button>'+
  '<span id="dstat" class="mono" style="font-size:12.6px;color:#7f8ea3"></span></div>'+
  '<div class="demo">'+
  '<div class="pane"><h4><span>analytic grid update <span class="pill an">ground truth</span></span>'+
  '<span id="ta" class="mono" style="color:#6fd3ee"></span></h4>'+
  '<canvas id="ca" width="420" height="420"></canvas><div class="ft" id="fa"></div></div>'+
  '<div class="pane"><h4><span>learned grid update <span class="pill nn" id="lp">width 64</span></span>'+
  '<span id="tn" class="mono" style="color:#ff9d5c"></span></h4>'+
  '<canvas id="cn" width="420" height="420"></canvas><div class="ft" id="fn"></div></div></div>'+
  '<div id="derr" class="err" style="display:none"></div>';
R.appendChild(demo);
R.appendChild(el('div','note',
  'What to look for: the learned panel is a recognisable blob for a fraction of a second and then stops '+
  'being water &mdash; it either freezes into a clump or smears across the floor. The measured grid-kernel '+
  'time in each header is the finding this page is about; the divergence is the price of the thing the time '+
  'bought.'));

// ================================================================ THE COST CHART
R.appendChild(el('h2',null,'The cost, against the budget it has to fit'));
R.appendChild(el('p',null,
  'Whole-solver time per substep, measured with a GPU timestamp query over a pass of 200 substeps, minimum '+
  'of 11 repetitions. The two lines are the same 60 fps budget: '+D.spf+' substeps must fit in 16.7&nbsp;ms, '+
  'so a substep may cost '+fmt(D.budget_us,1)+'&nbsp;&micro;s on the whole GPU or '+fmt(D.budget_us_q,1)+
  '&nbsp;&micro;s on a quarter of it.'));
var cc = el('div','card');
cc.innerHTML = '<div class="row" id="cbtn" style="margin-bottom:8px"></div><div id="cplot"></div>'+
  '<div class="legend" id="cleg"></div>';
R.appendChild(cc);
var shown = {8:true,16:true,32:true,64:true,analytic:true,sparse:false};
function drawCost(){
  var vals=[]; NS.forEach(function(n){ vals.push(pick(n,8).full_us_analytic);
    W.forEach(function(h){ vals.push(pick(n,h).full_us_nn); }); });
  var c = chart({w:1020,h:340,x0:NS[0]*0.85,x1:NS[NS.length-1]*1.18,y0:3,y1:120,
    xticks:NS, yticks:[3,5,10,20,50,100],
    xlabel:'particles', ylabel:'whole solver, µs per substep'});
  c.hline(D.budget_us,'#7ee787','60 fps budget',null,'end');
  c.hline(D.budget_us_q,'#ffd24d','the same budget at a quarter of this GPU');
  if(shown.analytic){ var pa=NS.map(function(n){return [n,pick(n,8).full_us_analytic];});
    c.line(pa,'#6fd3ee',null,3); c.dots(pa,'#6fd3ee',4.4); }
  W.forEach(function(h){ if(!shown[h]) return;
    var p=NS.map(function(n){return [n,pick(n,h).full_us_nn];});
    c.line(p,WCOL[h],'6 4',2.1); c.dots(p,WCOL[h],3.6);
    if(shown.sparse){ var q=NS.map(function(n){return [n,pick(n,h).full_us_nnsparse];});
      c.line(q,WCOL[h],'2 3',1.2); } });
  document.getElementById('cplot').innerHTML = c.done();
}
var cb = document.getElementById('cbtn');
[['analytic','analytic baseline','#6fd3ee']].concat(W.map(function(h){return [h,'learned width '+h,WCOL[h]];}))
  .concat([['sparse','skip empty cells','#7f8ea3']]).forEach(function(t){
  var b=el('button',shown[t[0]]?'on':'', t[1]);
  b.onclick=function(){ shown[t[0]]=!shown[t[0]]; b.className=shown[t[0]]?'on':''; drawCost(); };
  cb.appendChild(b); });
drawCost();
document.getElementById('cleg').innerHTML =
  '<span><i style="background:#6fd3ee"></i>solid: analytic</span><span><i style="background:#ff9d5c"></i>dashed: learned</span>'+
  '<span>every learned curve is FLAT in particle count &mdash; the grid has 16,384 cells whether 512 or 32,768 particles sit on it</span>';

// ---- verdict table ----
var vt = el('div','card');
var rows = D.verdict.rows.map(function(r){
  function cell(v){ return v ? '<span class="yes">width '+v+'</span>' : '<span class="no">none</span>'; }
  return '<tr><td>'+r.n.toLocaleString()+'</td><td>'+fmt(r.analytic_full_us,1)+'</td>'+
    '<td>'+cell(r.max_width_nn_full_gpu)+'</td><td>'+cell(r.max_width_nn_quarter_gpu)+'</td></tr>';
}).join('');
vt.innerHTML = '<h3>Largest learned grid update that still holds 60 fps</h3>'+
  '<table><tr><th>particles</th><th>analytic us/substep</th><th>whole GPU</th><th>a quarter of it</th></tr>'+
  rows+'</table>';
R.appendChild(vt);

// ================================================================ WHY IT DOESN'T GET CHEAPER
R.appendChild(el('h2',null,'Why the curve is flat, and why compaction does not fix it'));
R.appendChild(el('p',null,
  'The obvious repair for a flat curve is to run the network only on cells that hold material. That is '+
  'exact, not an approximation: G2P gathers from precisely the cells P2G scattered into, so an empty cell '+
  'is unobservable. Two versions were measured and neither helps, which is the interesting part.'));
var cmp = el('details',null);
(function(){
  var byN = {};
  D.compaction.forEach(function(r){ (byN[r.n]=byN[r.n]||{})[r.hidden]=r; });
  var occ={}; D.occupancy.forEach(function(o){ occ[o.n]=o; });
  var rows = NS.map(function(n){
    var o=occ[n], c=byN[n][32], c64=byN[n][64];
    var sp = pick(n,32).grid_us.nnsparse;
    return '<tr><td>'+n.toLocaleString()+'</td><td>'+o.occupied_cells.toLocaleString()+' ('+
      fmt(100*o.occupied_frac,1)+'%)</td><td>'+c.workgroups_compacted+' / '+c.workgroups_full+'</td>'+
      '<td>'+fmt(c.grid_us_dense,1)+'</td><td>'+fmt(sp,1)+'</td><td>'+fmt(c.grid_us_compacted,1)+'</td>'+
      '<td>'+fmt(c.speedup,2)+'&times;</td></tr>'; }).join('');
  cmp.innerHTML = '<summary>The measurements: width 32, grid-update kernel only, microseconds per substep</summary>'+
    '<table><tr><th>particles</th><th>occupied cells</th><th>workgroups</th><th>dense</th>'+
    '<th>empty cells exit early</th><th>dispatch compacted</th><th>speedup</th></tr>'+rows+'</table>'+
    '<p class="sub" style="margin-top:9px">At 512 particles only 2.4% of the grid holds material, and '+
    'dispatching the kernel over 7 workgroups instead of 256 &mdash; a 36-fold cut in the work issued &mdash; '+
    'changes the time by a few percent. That is the signature of a kernel that is <b>latency-bound</b>: '+
    '16,384 threads is nowhere near enough to occupy this GPU, so the elapsed time is set by how long one '+
    'thread takes to walk its own network, not by how many threads there are.</p>';
})();
R.appendChild(cmp);
R.appendChild(el('div','note',
  'The consequence is blunt. Making the problem <i>smaller</i> does not make a learned grid update cheaper, '+
  'so the usual sparsity lever does nothing here. The only thing that shortens the time is a shorter serial '+
  'chain per cell &mdash; a smaller network.'));

// ================================================================ width cliff
var wc = el('details',null,
  '<summary>Cost is not proportional to arithmetic either &mdash; width 48 is cheaper than width 40</summary>'+
  '<p class="sub">Cost does not depend on what is in the weight buffer, so the width axis can be swept with '+
  'untrained networks. At fine spacing the cost is not a smooth function of the network size at all. Both '+
  'series are independent repeats of the whole scan.</p><div id="wplot"></div>');
R.appendChild(wc);
(function(){
  var p0 = D.width_scan.filter(function(r){return r.pass===0||r.pass===undefined;});
  var p1 = D.width_scan.filter(function(r){return r.pass===1;});
  var hs = p0.map(function(r){return r.hidden;});
  var c = chart({w:1020,h:330,x0:3.4,x1:150,y0:0.4,y1:220,xticks:hs,yticks:[0.5,1,2,5,10,20,50,100,200],
    xlabel:'hidden width', ylabel:'grid-update kernel, µs per substep'});
  var ref = p0.filter(function(r){return r.hidden===16;})[0];
  c.line(p0.map(function(r){return [r.hidden, ref.grid_us*r.flops_per_cell/ref.flops_per_cell];}),
    '#4c5b6c','5 4',1.6);
  c.line(p0.map(function(r){return [r.hidden,r.grid_us];}),'#6fd3ee',null,2.4);
  c.dots(p0.map(function(r){return [r.hidden,r.grid_us];}),'#6fd3ee',3.8);
  if(p1.length){ c.line(p1.map(function(r){return [r.hidden,r.grid_us];}),'#ff9d5c','3 3',1.6);
    c.dots(p1.map(function(r){return [r.hidden,r.grid_us];}),'#ff9d5c',2.6); }
  c.hline(D.budget_us,'#7ee787','60 fps budget for the WHOLE substep');
  c.hline(D.budget_us_q,'#ffd24d','at a quarter GPU');
  c.text(c.px(30), c.py(115), 'widths 24-40 run at about a third of', '#b48ead', 12);
  c.text(c.px(30), c.py(88), 'the throughput of 20 or of 48', '#b48ead', 12);
  document.getElementById('wplot').innerHTML = c.done();
  var t = p0.map(function(r,i){ var q=p1[i]||{};
    return '<tr><td>'+r.hidden+'</td><td>'+r.flops_per_cell+'</td><td>'+(r.net_floats*4).toLocaleString()+
      '</td><td>'+r.hidden_bytes+'</td><td>'+fmt(r.grid_us,2)+'</td><td>'+fmt(q.grid_us,2)+'</td><td>'+
      fmt(r.achieved_gflops,0)+'</td></tr>'; }).join('');
  wc.appendChild(el('div',null,'<div class="legend"><span><i style="background:#6fd3ee"></i>pass 1</span>'+
    '<span><i style="background:#ff9d5c"></i>pass 2 (independent repeat)</span>'+
    '<span><i style="background:#4c5b6c"></i>what proportional-to-arithmetic would look like</span></div>'+
    '<table style="margin-top:8px"><tr><th>width</th><th>FLOP/cell</th><th>weights (B)</th>'+
    '<th>hidden state (B/thread)</th><th>pass 1 us</th><th>pass 2 us</th><th>GFLOP/s</th></tr>'+t+'</table>'));
})();
R.appendChild(el('div','scope',
  '<b>Observed:</b> ~3,500 GFLOP/s at widths 4&ndash;20 and 48&ndash;128, ~1,000 in between, in two independent '+
  'passes. <b>Hypothesised:</b> the hidden vectors are function-scope arrays the compiler keeps in registers '+
  'only up to some size. <b>Would test it:</b> read the generated ISA, or hold the hidden state in workgroup '+
  'memory and see whether the band disappears.'));

// ================================================================ accuracy
R.appendChild(el('h2',null,'What that cost buys'));
R.appendChild(el('p',null,
  'Accuracy is secondary here, but a cost only means something next to what it delivers. The kernel '+
  'reproduces the node velocity to a few percent and still cannot run a fluid, because almost none of what '+
  'the grid update <i>does</i> shows up in the size of its output.'));

// --- the sharpest number on the page ---
var gsum = el('div','card');
(function(){
  var g = D.grav, gs = g.gravity_velocity_per_substep;
  var rows = [8,16,32,64].map(function(h){ var b=g.by_width[h];
    return '<tr><td>'+h+'</td><td>'+b.node_v_mae_massw.toExponential(2)+'</td><td>'+
      gs.toExponential(2)+'</td><td class="warnc">'+fmt(b.times_gravity_step,0)+'&times;</td></tr>'; }).join('');
  gsum.innerHTML =
    '<h3>Gravity is smaller than the fitting error, by two orders of magnitude</h3>'+
    '<p>Gravity is one line, <span class="mono">v.y &minus;= dt &times; g</span>, worth <b>'+
    gs.toExponential(2)+'</b> of velocity per substep &mdash; a thousandth of a typical node speed. It only '+
    'becomes a falling drop because it is applied '+D.spf.toLocaleString()+' times a frame. A network fitted '+
    'to the grid update&rsquo;s <i>output</i> is fitted to a quantity in which gravity is a rounding error.</p>'+
    '<table><tr><th>hidden width</th><th>the network&rsquo;s velocity error</th>'+
    '<th>gravity, per substep</th><th>ratio</th></tr>'+rows+'</table>'+
    '<p class="sub" style="margin-top:9px">This is why the learned panel above hovers instead of falling, '+
    'and it is not a training failure more epochs would fix: at width 64 the network would have to get 56 '+
    'times more accurate before gravity was visible to it at all.</p>';
})();
R.appendChild(gsum);
R.appendChild(el('p',null,
  'The same trap has a second form. G2P never reads the node velocity: it gathers the affine matrix '+
  '<span class="mono">C</span>, whose entries carry a <span class="mono">1/dx&sup2;</span> factor, so what '+
  'reaches the material is a spatial <i>derivative</i> of what the grid update wrote. Fitting cell by cell '+
  'leaves that derivative free, and it is wrong by 40 to 90 percent where the velocity is right to 3.'));
var ac = el('details',null,
  '<summary>The derivative error by width, and how long each rollout stays the same fluid</summary>'+
  '<div id="aplot"></div>');
R.appendChild(ac);
(function(){
  var c = chart({w:1020,h:290,logx:true,logy:false,x0:6.5,x1:80,y0:0,y1:1.0,
    xticks:[8,16,32,64],yticks:[0,0.2,0.4,0.6,0.8,1.0],
    xlabel:'hidden width', ylabel:'relative error vs the analytic kernel'});
  var A=D.accuracy;
  var hs=[8,16,32,64];
  c.line(hs.map(function(h){return [h, A.stage1_pointwise[h].node_v_rel_massw];}),'#6fd3ee',null,2.6);
  c.dots(hs.map(function(h){return [h, A.stage1_pointwise[h].node_v_rel_massw];}),'#6fd3ee',4);
  c.line(hs.map(function(h){return [h, A.stage2_derivative[h].grad_rel_before];}),'#b48ead','6 4',2.2);
  c.dots(hs.map(function(h){return [h, A.stage2_derivative[h].grad_rel_before];}),'#b48ead',3.6);
  c.line(hs.map(function(h){return [h, A.stage2_derivative[h].grad_rel_after];}),'#ff9d5c','2 3',2.2);
  c.dots(hs.map(function(h){return [h, A.stage2_derivative[h].grad_rel_after];}),'#ff9d5c',3.6);
  c.text(c.ml+c.iw-10, c.py(0.06), 'node velocity itself', '#6fd3ee', 12.5, 'end');
  c.text(c.ml+c.iw-10, c.py(0.47), 'its spatial derivative (cell-wise loss)', '#b48ead', 12.5, 'end');
  c.text(c.ml+c.iw-10, c.py(0.30), 'its derivative, trained against it', '#ff9d5c', 12.5, 'end');
  document.getElementById('aplot').innerHTML=c.done();
  var rows = D.survival.map(function(s){
    return '<tr><td>'+(s.net.indexOf('deriv')===0?'derivative loss':'cell-wise loss')+'</td><td>'+s.hidden+
      '</td><td>'+fmt(s.frames_60fps_tracked,0)+'</td><td>'+fmt(s.sim_seconds_tracked,3)+'</td><td>'+
      fmt(s.final_dist,2)+'</td><td style="text-align:left;color:#7f8ea3">'+(s.reason||'-')+'</td></tr>'; }).join('');
  ac.appendChild(el('div',null,'<h3>How long the learned rollout is still the same fluid</h3>'+
    '<table><tr><th>trained with</th><th>width</th><th>60 fps frames tracked</th><th>seconds of fluid</th>'+
    '<th>final distance from truth</th><th>how it ended</th></tr>'+rows+'</table>'+
    '<p class="sub" style="margin-top:8px">&ldquo;Tracked&rdquo; means the mean per-particle distance from the '+
    'canonical rollout stayed under 0.05 of a domain length, about a fifth of the drop&rsquo;s own diameter. '+
    'The domain is the unit square, so a final distance near 1.0 means the particles are nowhere near where '+
    'the water should be.</p>'));
})();

// ================================================================ what was checked
var chk = el('details',null);
chk.innerHTML = '<summary>What was checked before any of this was believed</summary><table><tr><th>check</th><th>result</th></tr>'+
  '<tr><td style="text-align:left">Analytic WGSL baseline vs canonical Taichi (mean per-particle distance over a 1 s rollout)</td>'+
  '<td>'+fmt(D.analytic_check.traj_rmse_vs_canonical,5)+' against the reference&rsquo;s own run-to-run noise of '+
  fmt(D.self_noise,5)+'</td></tr>'+
  '<tr><td style="text-align:left">Particles actually moved by frame 1 (a dropped dispatch would read as a flat, perfect curve)</td>'+
  '<td class="yes">'+fmt(D.analytic_check.moved_by_frame1,5)+' domain lengths</td></tr>'+
  '<tr><td style="text-align:left">WGSL MLP vs the same weights evaluated on the host, largest disagreement over all cells</td>'+
  '<td class="yes">'+D.inference.map(function(r){return r.max_abs_diff.toExponential(1);}).join(', ')+'</td></tr>'+
  '<tr><td style="text-align:left">Dense vs empty-cell-skipping kernels on every cell a particle can gather from</td>'+
  '<td class="yes">identical to the last bit</td></tr>'+
  '<tr><td style="text-align:left">Per-dispatch floor (an empty dispatch inside the recorded command buffer)</td>'+
  '<td>'+fmt(D.floor_us,2)+' µs</td></tr>'+
  '<tr><td style="text-align:left">GPU errors raised during the whole run</td><td class="yes">none</td></tr></table>';
R.appendChild(chk);

R.appendChild(el('div','scope',
  '<b>Scope.</b> One GPU (RTX 4090), one browser, one grid (128&times;128), one material (canonical water, '+
  'physics '+D.physics_version+'), one architecture family (two-hidden-layer ReLU MLP per cell). The '+
  'latency-bound conclusion is specifically about 16,384 cells failing to occupy a very large GPU and would '+
  'not survive a much finer grid. Friction is an input but every grid state came from water, which is '+
  'frictionless at the boundary. Nothing here says a learned <i>frame-to-frame</i> operator is expensive; it '+
  'says a learned <i>per-substep, per-cell</i> one is.'));

// ================================================================ THE LIVE DEMO
(function(){
  var derr = document.getElementById('derr');
  function fail(msg){ derr.style.display='block'; derr.textContent = msg;
    document.getElementById('dgo').disabled=true; document.getElementById('drs').disabled=true; }
  if (typeof MPMNN === 'undefined' || !MPMNN.supported()) {
    fail('WebGPU is not available here (navigator.gpu is hidden outside a secure context). '+
         'The measurements above are unaffected; only the live panels need a GPU. '+
         'Open this page over https or http://localhost to run them.');
    return;
  }
  // Match the SCENE the networks were trained on: the canonical drop disk at the same particle
  // density. Halving the density halves the node mass in particle-mass units and puts every cell off
  // the distribution the network saw, which makes it fail for the wrong reason.
  var N = 6000, SUB = 45;
  var simA=null, simN=null, renA=null, renN=null, running=true, subs=0, busy=false, frames=0;
  var pts = MPMNN.seedDisk(0.5, 0.52, 0.11, N, 7);
  var area = Math.PI*0.11*0.11;
  var FLOORY = MPMNN.PARAMS.floor_y;
  function edgeFrac(p){ var k=0, m=p.length>>1, e=1e-4;
    for(var i=0;i<m;i++){ var x=p[2*i], y=p[2*i+1];
      if(x<=FLOORY+e||x>=1-FLOORY-e||y<=FLOORY+e||y>=1-FLOORY-e) k++; }
    return k/m; }
  function meanDist(a,b){ var s=0, m=a.length>>1;
    for(var i=0;i<m;i++){ var dx=a[2*i]-b[2*i], dy=a[2*i+1]-b[2*i+1]; s+=Math.sqrt(dx*dx+dy*dy); }
    return s/m; }
  var caE=document.getElementById('ca'), cnE=document.getElementById('cn');
  function sty(){ return document.getElementById('dt').value + document.getElementById('dw').value; }

  // Rebuilding on a width change destroys both simulators. The animation loop may be parked on an
  // await holding references to them, so the two have to be serialised or the loop wakes up to a
  // null sim and the demo dies with "cannot read properties of null". `busy` is the lock and every
  // rebuild is queued behind the last one.
  var chain = Promise.resolve();
  function rebuild(){ chain = chain.then(build).catch(function(e){ fail('WebGPU demo failed: ' + (e && e.message || e)); }); }

  async function build(){
    while (busy) { await new Promise(function(r){ setTimeout(r, 16); }); }
    busy = true;
    try{
      if(simA){ await simA.idle(); simA.destroy(); simA=null; }
      if(simN){ await simN.idle(); simN.destroy(); simN=null; }
      simA = await MPMNN.createSim({n:N, area:area, net:'point8'});
      simN = await MPMNN.createSim({n:N, area:area, net:sty()});
      renA = await MPMNN.createRenderer(caE, simA);
      renN = await MPMNN.createRenderer(cnE, simN);
      simA.seed(pts,0,0); simN.seed(pts,0,0);
      subs=0; frames=0; accA=[]; accN=[];
      document.getElementById('lp').textContent='width '+document.getElementById('dw').value;
    }catch(e){ fail('WebGPU demo failed to start: '+(e&&e.message||e)); }
    busy = false;
  }

  var accA=[], accN=[];
  async function tick(){
    if(!busy && running && simA && simN){
      busy = true;
      var A = simA, Nn = simN, rA = renA, rN = renN;
      try{
        frames++;
        // Loop. Past about 1.5 s the analytic panel is a settled puddle and the learned one is off
        // the map, so a viewer arriving late would see two static pictures instead of the comparison.
        if (subs * MPMNN.PARAMS.dt > 1.5) { A.seed(pts,0,0); Nn.seed(pts,0,0); subs = 0; }
        var probe = (frames % 5) === 0 || frames === 2;   // show the readouts early, not after a second
        A.encodeFrame(SUB,{grid:'analytic',timed:true,readback:probe}); accA.push(await A.lastGpuNanos());
        var pA = probe ? await A.readPositions() : null;
        Nn.encodeFrame(SUB,{grid:'nn',timed:true,readback:probe});         accN.push(await Nn.lastGpuNanos());
        var pN = probe ? await Nn.readPositions() : null;
        subs += SUB;
        rA.draw({radius:0.0092, vRef:3.2});
        rN.draw({radius:0.0092, vRef:3.2, tint:1.0});
        if(probe){
          var mA=Math.min.apply(null,accA)/1000/SUB, mN=Math.min.apply(null,accN)/1000/SUB;
          document.getElementById('ta').textContent = mA.toFixed(1)+' µs/substep';
          document.getElementById('tn').textContent = mN.toFixed(1)+' µs/substep';
          document.getElementById('fa').textContent =
            'whole solver · '+(mA<=D.budget_us_q?'inside the quarter-GPU budget'
              :(mA<=D.budget_us?'inside the full-GPU budget only':'over budget'))+
            ' · '+(100*edgeFrac(pA)).toFixed(0)+'% of particles touching a wall or the floor';
          document.getElementById('fn').textContent =
            (mN/mA).toFixed(1)+'× the analytic panel · mean distance from it '+
            meanDist(pA,pN).toFixed(3)+' · '+(100*edgeFrac(pN)).toFixed(0)+'% touching a wall or the floor';
          accA=[]; accN=[];
        }
        document.getElementById('dstat').textContent =
          'sim time '+(subs*MPMNN.PARAMS.dt).toFixed(2)+' s (loops at 1.5 s)   ·   '+SUB+' substeps per drawn frame, '+
          'against the '+D.spf+' real time would need — so this plays at about '+
          (SUB/D.spf*100).toFixed(0)+'% speed';
      }catch(e){ fail('WebGPU demo stopped: '+(e&&e.message||e)); running=false; }
      busy = false;
    }
    requestAnimationFrame(tick);
  }
  document.getElementById('dgo').onclick=function(){ running=!running; this.textContent = running?'pause':'resume'; };
  document.getElementById('drs').onclick=function(){ if(simA&&simN){ simA.seed(pts,0,0); simN.seed(pts,0,0); subs=0; frames=0; } };
  document.getElementById('dw').onchange=rebuild;
  document.getElementById('dt').onchange=rebuild;
  rebuild(); chain.then(function(){ requestAnimationFrame(tick); });
})();
})();
"""

if __name__ == "__main__":
    main()
