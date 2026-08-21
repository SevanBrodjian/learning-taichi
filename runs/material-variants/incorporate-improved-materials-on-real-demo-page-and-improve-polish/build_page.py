"""Build the bespoke task page from metrics.json.

Designed around ONE thing (spec/style_task_page.md): the demo page was running physics that had no
concept of density, and now it is not -- and you can see that happen. So the page leads with a flip
between the two, on the same scene, in the same medium as the claim (video, because the claim is
about motion), and only then shows the two smaller halves (the treatments, and the layout).

Self-contained: no CDN, no fetch, no external font. Media by absolute /api/data/ path.

    .venv/Scripts/python.exe runs/.../build_page.py
"""
import json
import pathlib

RUN = pathlib.Path(__file__).resolve().parent
MEDIA = ("/api/data/learning-taichi/runs/material-variants/"
         "incorporate-improved-materials-on-real-demo-page-and-improve-polish/")

M = json.loads((RUN / "metrics.json").read_text())

VIEWPORTS = [("phone_portrait", "iPhone portrait", "390 x 844"),
             ("phone_landscape", "iPhone landscape", "844 x 390"),
             ("tablet_portrait", "iPad portrait", "820 x 1180"),
             ("laptop", "small laptop", "1280 x 800"),
             ("desktop", "large monitor", "1920 x 1080")]

DATA = {
    "media": MEDIA,
    "physics_version": M["physics_version"],
    "device": M["device"].get("vendor", "?") + " " + M["device"].get("architecture", ""),
    "ordering": M["ordering_pass"],
    "buoy": M["buoyancy"]["pool_three"],
    "agree": M["agreement"],
    "canon": M["canonical_reference"],
    "render": M["render_gpu_ms"],
    "solver": M["solver"],
    "budget": M["frame_budget"],
    "layout": M["layout"],
    "viewports": VIEWPORTS,
    "pile": M["pile"],
    "water": M["water_reconstruction"],
}

HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>T-027 — the demo page, on the real materials</title>
<style>
  :root { --bg:#0a0e14; --fg:#dfe6ee; --mut:#7f8ea3; --acc:#6fd3ee; --line:#1b2735; --warm:#ffb26b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); padding:4px 2px 26px;
         font:14.5px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  h2 { font-size:15px; letter-spacing:.14em; text-transform:uppercase; color:var(--acc);
       margin:34px 0 6px; font-weight:600; }
  h2:first-of-type { margin-top:10px; }
  p { margin:8px 0; max-width:74ch; }
  .sub { color:var(--mut); font-size:13px; }
  .verdict { border-left:3px solid var(--acc); padding:2px 0 2px 14px; margin:6px 0 20px; }
  .verdict b { color:#fff; }
  .miss { color:var(--warm); }
  .card { border:1px solid var(--line); border-radius:8px; background:#0c121a; padding:14px; }
  .row { display:flex; gap:14px; flex-wrap:wrap; }
  video, img { max-width:100%; display:block; border-radius:6px; background:#000; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; margin:12px 0 10px; }
  button { font:inherit; font-size:12.5px; letter-spacing:.06em; color:#9fc4d2; cursor:pointer;
    background:#121c26; border:1px solid #22394a; border-radius:5px; padding:7px 12px; }
  button:hover { color:#dff2fa; border-color:#356b83; }
  button.on { background:var(--acc); border-color:var(--acc); color:#04121a; font-weight:600; }
  table { border-collapse:collapse; font-size:13px; margin:8px 0; }
  th,td { border-bottom:1px solid var(--line); padding:5px 12px 5px 0; text-align:left;
          vertical-align:baseline; }
  th { color:var(--mut); font-weight:500; font-size:11.5px; letter-spacing:.09em;
       text-transform:uppercase; }
  td.n { font-variant-numeric:tabular-nums; }
  .ok { color:#7ee787 } .bad { color:#ff8f8f } .warnc { color:#ffd24d }
  .lab { font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--mut); }
  .bars { margin:10px 0 4px; }
  .bar { display:grid; grid-template-columns:104px 1fr 122px; gap:10px; align-items:center;
         margin:5px 0; font-size:13px; }
  .track { height:20px; background:#111b24; border-radius:3px; position:relative; overflow:hidden; }
  .fill { position:absolute; top:0; bottom:0; }
  .mid { position:absolute; top:-3px; bottom:-3px; width:1px; background:#3a5364; }
  .note { color:var(--mut); font-size:12.5px; max-width:74ch; }
  .kv { display:flex; gap:26px; flex-wrap:wrap; margin:10px 0 2px; }
  .kv div span { display:block; font-size:11px; letter-spacing:.1em; text-transform:uppercase;
                 color:var(--mut); }
  .kv div b { font-size:21px; font-weight:600; font-variant-numeric:tabular-nums; }
  .shot { flex:1 1 320px; min-width:280px; }
  .shot img { border:1px solid var(--line); }
  .shot .lab { margin-bottom:5px; }
  .scope { border:1px solid #3a2c1a; background:#140f07; border-radius:8px; padding:11px 14px;
           margin:16px 0; color:#e3c9a5; font-size:13px; max-width:80ch; }
  code { background:#101a24; padding:1px 5px; border-radius:3px; font-size:12.5px; color:#cfe6ef; }
</style>
<body>
<div id="app"></div>
<script>
var D = __DATA__;
var $ = function (h) { var d = document.createElement('div'); d.innerHTML = h; return d.firstElementChild; };
// add() must append EVERY top-level element in the fragment, not just the first. The first draft
// returned firstElementChild and silently dropped the two tables that carry the agreement bars and
// the render-cost numbers -- the page rendered, looked plausible, and was missing two sections.
// Opening the rendered page and counting the elements is what caught it.
// The entity table is written with \u escapes on purpose. This page is delivered to the dashboard
// through an iframe `srcdoc` ATTRIBUTE, and the HTML parser entity-decodes that attribute before the
// script is parsed -- so a literal '&amp;' in the source arrives as '&' and esc() quietly becomes the
// identity function. \u0026 is invisible to the entity decoder and means the same thing to JS.
function esc(s){ return String(s).replace(/[&<>]/g,function(c){
  return {'&':'\u0026amp;','<':'\u0026lt;','>':'\u0026gt;'}[c]; }); }
function f(x, n) { return (x === null || x === undefined) ? '--' : Number(x).toFixed(n === undefined ? 3 : n); }
var app = document.getElementById('app');
function add(h) {
  var d = document.createElement('div');
  d.innerHTML = h;
  var kids = [].slice.call(d.children), last = null;
  kids.forEach(function (k) { app.appendChild(k); last = k; });
  return kids.length === 1 ? kids[0] : last;
}

// ---------------------------------------------------------------- verdict
add('<div class="verdict"><p><b>The Demo page was running materials with no concept of density.</b> ' +
    'It now runs the current canonical physics (' + esc(D.physics_version) + '), and the ' +
    'sink/float ordering reproduces on the page\'s own WGSL solver, matching canonical to within ' +
    'canonical\'s own run-to-run noise. Snow, sand and water are on their chosen new treatments and ' +
    'rubber is on the old one with two tweaks. The field is now genuinely square on a phone, where ' +
    'it previously stretched the simulation by 42%.</p>' +
    '<p class="miss">Corrected after review: the first version of this task claimed water was on ' +
    'its new treatment, and it was only half on it. The SHADING was ported and the RECONSTRUCTION ' +
    'that shading reads was not, so the water still came out as the speckled "smoothie" the ' +
    'proposal existed to replace. Section 4 is the fix and its evidence.</p>' +
    '<p class="miss">What did not make it in: nothing was done about the shared timestep, so a ' +
    'phone still runs this in labelled slow motion; and every number below is one GPU, one browser, ' +
    'one scene.</p></div>');

// ---------------------------------------------------------------- 1. THE FLIP
add('<h2>1 &nbsp;The physics, before and after</h2>');
add('<p>Three blobs &mdash; snow, rubber, sand &mdash; released <em>at rest</em>, side by side, at ' +
    'the same depth in one pool. Identical initial conditions, identical substep schedule, and ' +
    '<b>both halves drawn with the old shading</b>, so the only thing that differs between them is ' +
    'the material parameters. There is no buoyancy force on either side: the whole effect is that a ' +
    'particle\'s mass is now <code>p_vol &times; rho</code> instead of <code>p_vol</code>.</p>');
add('<div class="card"><video src="' + D.media + 'cmp_buoyancy.mp4" controls loop muted playsinline ' +
    'style="width:100%"></video><p class="note" style="margin-top:10px">Left: the physics the page ' +
    'shipped with. Right: the physics it ships with now. Same scene, same seed, same 667 substeps ' +
    'per frame.</p></div>');

// per-material read-out, canonical vs the page's own solver
(function () {
  var rows = [['snow', 'SNOW', 0.3], ['elastic', 'RUBBER', 1.2], ['sand', 'SAND', 1.6]];
  var h = '<table><tr><th>material</th><th>density &rho;</th>' +
    '<th title="waterline minus the body\'s mean height, as a CHANGE from release. Negative = it rose.">' +
    '&Delta; rest_depth &mdash; canonical</th>' +
    '<th>&Delta; rest_depth &mdash; this page</th>' +
    '<th title="fraction of the body below the waterline at the end">submerged_fraction</th>' +
    '<th>verdict</th></tr>';
  rows.forEach(function (r) {
    var e = D.buoy[r[0]];
    h += '<tr><td>' + r[1] + '</td><td class="n">' + f(e.rho, 2) + '</td>' +
      '<td class="n">' + (e.canonical.rest_depth_change >= 0 ? '+' : '') + f(e.canonical.rest_depth_change, 4) + '</td>' +
      '<td class="n">' + (e.webgpu.rest_depth_change >= 0 ? '+' : '') + f(e.webgpu.rest_depth_change, 4) + '</td>' +
      '<td class="n">' + f(e.webgpu.submerged_fraction, 3) + '</td>' +
      '<td class="' + (e.sign_agrees ? 'ok' : 'bad') + '">' +
      (e.webgpu.rest_depth_change < 0 ? 'rises' : 'sinks') + (e.sign_agrees ? '' : ' &mdash; DISAGREES') +
      '</td></tr>';
  });
  h += '</table>';
  add(h);
  add('<p class="note">Final mean height on the page\'s own solver: snow <b>' +
      f(D.ordering.mean_y_end_webgpu.snow, 3) + '</b> &gt; rubber <b>' +
      f(D.ordering.mean_y_end_webgpu.elastic, 3) + '</b> &gt; sand <b>' +
      f(D.ordering.mean_y_end_webgpu.sand, 3) + '</b> &mdash; ordered by density, which is the ' +
      'pass condition for this half of the task. <code>rest_depth</code> and ' +
      '<code>submerged_fraction</code> are the registered metrics of the same names ' +
      '(spec/registry/metrics.json), computed by <code>sim.physics.core</code> on both sides.</p>');
})();

// agreement against canonical's own noise
(function () {
  var h = '<h2>2 &nbsp;Is the port still the same physics?</h2>' +
    '<p>Every scene was run twice on the WGSL solver\'s side and three times on canonical\'s: ' +
    'once as the reference, once repeated (GPU atomic ordering alone is non-deterministic), and ' +
    'once with the initial positions nudged by 1e-7, a single float32 rounding unit. That spread is ' +
    'the <b>self-noise band</b>, and it &mdash; not zero &mdash; is what the browser is judged ' +
    'against. Bars show the ratio; 1.0 means indistinguishable from re-running canonical.</p>' +
    '<div class="bars">';
  var names = Object.keys(D.agree);
  var mx = Math.max(6, Math.max.apply(null, names.map(function (k) { return D.agree[k].ratio_to_self_noise; })));
  names.forEach(function (k) {
    var r = D.agree[k], u = Math.min(1, r.ratio_to_self_noise / mx);
    var col = r.ratio_to_self_noise <= 3 ? '#6fd3ee' : (r.ratio_to_self_noise <= 8 ? '#ffd24d' : '#ff8f8f');
    h += '<div class="bar"><span class="lab">' + esc(k) + '</span>' +
      '<span class="track"><span class="fill" style="left:0;width:' + (u * 100).toFixed(1) +
      '%;background:' + col + '"></span><span class="mid" style="left:' + (100 / mx).toFixed(1) + '%"></span></span>' +
      '<span class="n">&times;' + f(r.ratio_to_self_noise, 2) + ' &nbsp;<span class="sub">(' +
      f(r.traj_rmse, 5) + ')</span></span></div>';
  });
  h += '</div>';
  add(h);
  add('<p class="note">The thin vertical line is &times;1, canonical\'s own noise floor. The number ' +
      'in brackets is <code>traj_rmse</code> &mdash; which, as the registry warns, is a <em>mean ' +
      'per-particle distance</em>, not a root-mean-square. <code>pool_fluid</code> is the widest ' +
      'ratio at &times;' + f(D.agree.pool_fluid.ratio_to_self_noise, 2) + ', and that is an artefact ' +
      'of a very small denominator: a blob of water inside water barely moves relative to the pool, ' +
      'so canonical repeats itself to ' + f(D.agree.pool_fluid.self_noise, 6) + ' and the absolute ' +
      'disagreement is still ' + f(D.agree.pool_fluid.traj_rmse, 6) + ' domain lengths.</p>');
})();

// ---------------------------------------------------------------- 3. THE TREATMENTS
add('<h2>3 &nbsp;The four treatments</h2>');
add('<p>Same build, same physics, same particle positions in both halves &mdash; the only thing ' +
    'that changes is what the resolve pass does with the reconstructed surface. Water is ' +
    'T-020\'s option B ("film"): the same reconstruction as the glass option but with no background ' +
    'sampling and no chromatic dispersion, chosen because the demo\'s background is a flat dark ' +
    'gradient and there is nothing behind the water worth refracting. Snow is option A ("powder"), ' +
    'sand is option A ("grains over a packed body"), and rubber is <b>not</b> a new option &mdash; ' +
    'it is the shipped treatment with a smaller splat kernel and an added border band.</p>');
add('<div class="card"><video src="' + D.media + 'cmp_render.mp4" controls loop muted playsinline ' +
    'style="width:100%"></video><p class="note" style="margin-top:10px">Left: one treatment for all ' +
    'four materials. Right: four treatments. The particles are in the same place in both halves.</p></div>');

// render cost, at two resolutions -- the resolution check IS the evidence the timing is real
(function () {
  var h = '<p style="margin-top:18px">Cost, from <code>timestamp-query</code> across both render ' +
    'passes, at 16,384 particles with all four materials present. <b>Measured at two resolutions on ' +
    'purpose</b>: a screen-space treatment costs per pixel, so a number that does not move with the ' +
    'pixel count is a clock reading, not a GPU reading &mdash; which is exactly the trap that nearly ' +
    'produced a wrong conclusion upstream.</p>' +
    '<table><tr><th>view</th><th>512&sup2; before</th><th>512&sup2; after</th>' +
    '<th>1024&sup2; before</th><th>1024&sup2; after</th><th>scales with pixels?</th></tr>';
  ['blob', 'pts', 'grid'].forEach(function (v) {
    var b5 = D.render.before['512_' + v], a5 = D.render.after['512_' + v];
    var b10 = D.render.before['1024_' + v], a10 = D.render.after['1024_' + v];
    var ratio = a10 / Math.max(a5, 1e-9);
    h += '<tr><td>' + v + '</td><td class="n">' + f(b5) + '</td><td class="n">' + f(a5) + '</td>' +
      '<td class="n">' + f(b10) + '</td><td class="n">' + f(a10) + '</td>' +
      '<td class="n">&times;' + f(ratio, 2) + '</td></tr>';
  });
  h += '</table>';
  add(h);
  add('<p class="note">The material view went from ' + f(D.render.before['1024_blob']) + ' ms to ' +
      f(D.render.after['1024_blob']) + ' ms at 1024&sup2;. Almost all of that is a fixed +0.05 ms ' +
      'that does <em>not</em> grow with resolution &mdash; it is the sand pass, which draws 6n ' +
      'instances and is bound by geometry, not fill. The per-pixel part of the resolve barely ' +
      'moved (0.063 ms of resolution-dependent cost before, 0.060 ms after) even though it now ' +
      'carries four shading models, because the extra work is arithmetic on values already loaded.</p>');
  add('<p class="note"><b>Read the last column carefully.</b> A flat &times;1.00 is the shape of the ' +
      'trap this project has already been caught by once &mdash; but here it is the right answer, not ' +
      'a broken clock: the <code>pts</code> view draws 16,384 tiny hard squares and is bound by ' +
      'geometry, so quadrupling the pixels genuinely does not cost it anything. The two views that ' +
      'ARE fill-bound (<code>blob</code>, <code>grid</code>) do move with resolution, which is what ' +
      'says the instrument is working.</p>');
})();

// frame budget
(function () {
  var b = D.budget;
  add('<div class="kv">' +
    '<div><span>solver, 16,384 particles</span><b>' + f(b.solver_gpu_ms, 2) + ' ms</b></div>' +
    '<div><span>drawing, 1024&sup2;</span><b>' + f(b.render_gpu_ms_1024, 2) + ' ms</b></div>' +
    '<div><span>total on the GPU</span><b>' + f(b.total_gpu_ms, 2) + ' ms</b></div>' +
    '<div><span>60 fps budget</span><b>' + f(b.budget_60fps_ms, 2) + ' ms</b></div>' +
    '</div>');
  add('<p class="note">Drawing is ' + f(100 * b.render_gpu_ms_1024 / b.budget_60fps_ms, 1) +
      '% of a 60 fps frame. The solver is ' + f(100 * b.solver_gpu_ms / b.budget_60fps_ms, 0) +
      '% of it, and it is the solver, not the rendering, that decides whether this page is real ' +
      'time &mdash; which is why the rendering half of this task was allowed to be the ambitious one.</p>');
})();

// ---------------------------------------------------------------- 4. THE WATER REWORK
add('<h2>4 &nbsp;The water: the shading was never the treatment</h2>');
add('<p>This section is a <b>correction</b>. The version of this task that first shipped ported ' +
    'T-020\'s water <em>shading</em> faithfully &mdash; Beer&ndash;Lambert absorption, a tight ' +
    'specular, a Fresnel rim, motion-gated foam, every line of it present in the resolve pass ' +
    '&mdash; and shipped water that still looked like the old water. Every quantity that shading ' +
    'reads (how thick the water is here, which way the surface faces here, how opaque it is here) ' +
    'came from four neighbour taps of the raw splat accumulation, and a splat accumulation is a ' +
    'sum of a few thousand overlapping bumps. It is lumpy at the particle scale, so the shading ' +
    'faithfully lit a lumpy thing.</p>');
add('<p>What was missing is the <b>reconstruction</b>: blur the density, threshold it to a binary ' +
    'body, and take optical thickness from a <b>distance transform</b> of that body instead of ' +
    'from a local density count. A distance field cannot carry particle-scale noise, because it ' +
    'does not know where the particles are &mdash; only where the surface is.</p>');

(function () {
  var wrap = add('<div></div>');
  var tabs = document.createElement('div'); tabs.className = 'tabs';
  var card = document.createElement('div'); card.className = 'card';
  var CLIPS = [
    ['the demo\'s own scene', 'cmp_water.mp4',
     'The page\'s opening scene. Same build, same physics, same particle positions, same snow, ' +
     'sand and rubber in both halves &mdash; the ONLY difference is where the water reads its ' +
     'thickness and its normal from.'],
    ['water alone, with a splash', 'cmp_water_pool.mp4',
     'Water plus one rubber ball, so the surface and the spray are both in frame and nothing else ' +
     'competes for attention. Watch the surface line, and watch the interior when the ball lands.']
  ];
  var vid = document.createElement('video');
  vid.controls = vid.loop = vid.muted = vid.playsInline = true; vid.style.width = '100%';
  var cap = document.createElement('p'); cap.className = 'note'; cap.style.marginTop = '10px';
  function pick(i) {
    vid.src = D.media + CLIPS[i][1]; cap.innerHTML = CLIPS[i][2];
    [].forEach.call(tabs.children, function (b, k) { b.classList.toggle('on', k === i); });
  }
  CLIPS.forEach(function (c, i) {
    var b = document.createElement('button'); b.textContent = c[0];
    b.onclick = function () { pick(i); }; tabs.appendChild(b);
  });
  card.appendChild(vid); card.appendChild(cap);
  wrap.appendChild(tabs); wrap.appendChild(card);
  pick(0);
})();

add('<p style="margin-top:18px">And the question that actually decides it &mdash; did it land where ' +
    'the proposal was? The left panel is T-020\'s own Taichi render of option B. Different scene, ' +
    'different resolution, different graphics API, so this compares the <em>treatment</em>, never ' +
    'the pixels.</p>');
add('<div class="card"><img src="' + D.media + 'cmp_water_target.png" style="width:100%">' +
    '<p class="note" style="margin-top:10px">Interior colour at mid-depth, sampled: T-020\'s film ' +
    'render reads (50, 90, 112); the page now reads (51, 90, 109). Nothing was tuned against that ' +
    'number &mdash; it falls out of using T-020\'s palette, its absorption coefficient and its tone ' +
    'curve. Read it as a check that the colour pipeline was ported correctly, <em>not</em> as proof ' +
    'the two images are equivalent: T-020\'s pool is deeper (optical depth ~3.2 against ~1.7 here) ' +
    'and the absorption curve happens to be flat across that range.</p></div>');

(function () {
  var STAGES = [
    ['splat', '1, full res', 'Four-channel per-material weight. Untouched &mdash; this is still ' +
      'what the other three treatments shade off.'],
    ['blur', '2, half res', 'Separable Gaussian. The horizontal pass also does the 2x downsample, ' +
      'and its sigma is picked so that the splat kernel\'s own width PLUS this blur equals the ' +
      'smoothing T-020 applied to a point histogram.'],
    ['threshold + seed', '1', 'Binary body at a fixed fraction (0.24) of full packing &mdash; a ' +
      'physical reference rather than a per-frame percentile, which is what keeps a thin sheet of ' +
      'spray reading as thin. Every pixel OUTSIDE the body seeds itself.'],
    ['jump flood', '6-7', 'log2(range) doublings turn those seeds into the distance to the nearest ' +
      'outside pixel, for every interior pixel. This is the step that makes a distance transform ' +
      'affordable in a real-time frame at all.'],
    ['seeds &rarr; distance', '1', 'Plus a 3x3 box: a distance field off a thresholded mask is ' +
      'quantised in whole pixels, and Beer&ndash;Lambert turns quantisation into visible banding.'],
    ['resolve', '1, full res', 'Opacity, normal and optical thickness all read out of the distance ' +
      'field. Premultiplied, so the alpha carries the transmission.']
  ];
  var h = '<p style="margin-top:18px">The chain, in the order it runs. Everything between the splat ' +
    'and the resolve is at <b>half resolution</b>: optical thickness is the one quantity in the ' +
    'frame that is genuinely low-frequency, so halving it costs a quarter of the pixels on eleven ' +
    'passes and shows up nowhere in the image.</p><table>' +
    '<tr><th>stage</th><th>passes</th><th>what it is for</th></tr>';
  STAGES.forEach(function (r) {
    h += '<tr><td><b>' + r[0] + '</b></td><td class="n">' + r[1] + '</td><td>' + r[2] + '</td></tr>';
  });
  h += '</table>';
  add(h);
})();

(function () {
  var W = D.water, b = D.budget;
  var h = '<p style="margin-top:18px">What it costs, from <code>timestamp-query</code> across the ' +
    'whole blob draw at 16,384 particles. <b>before</b> is this task exactly as it originally ' +
    'shipped and <b>after</b> is the same build with the reconstruction switched on &mdash; same ' +
    'run, same instrument, same frame, so the difference is the chain and nothing else.</p>' +
    '<table><tr><th>resolution</th><th>pixels</th><th>before</th><th>after</th>' +
    '<th>chain, direct</th><th>chain, amplified</th><th>passes</th></tr>';
  ['480', '720', '1080'].forEach(function (r) {
    var v = W.by_res[r];
    h += '<tr><td>' + r + '&sup2;</td><td class="n">' + v.pixels.toLocaleString() + '</td>' +
      '<td class="n">' + f(v.before_gpu_ms) + '</td><td class="n">' + f(v.after_gpu_ms) + '</td>' +
      '<td class="n">' + f(v.chain_gpu_ms_direct) + '</td>' +
      '<td class="n">' + f(v.chain_gpu_ms_slope) + '</td>' +
      '<td class="n">' + v.passes_after + '</td></tr>';
  });
  h += '</table>';
  add(h);
  add('<p class="note"><b>Two independent measurements of the same thing, because one of them was ' +
      'not trustworthy on its own.</b> Chromium quantises <code>timestamp-query</code>; with ' +
      'quantisation disabled the residual granularity was still 16&ndash;33 &micro;s, which is the ' +
      'same size as the thing being measured. So the chain is priced twice: once as the difference ' +
      'against a matched control, and once as the slope of running it K times inside one timed ' +
      'region. The two columns agree to within one quantum, and the 1080&sup2; row is the loosest ' +
      'of the three &mdash; exactly one quantum apart, because its K=8 point was contaminated by ' +
      'running hundreds of render passes in a single submission. A first pass at all of this, ' +
      'before the flag went on, reported exact multiples of 32,768 ns for everything &mdash; that ' +
      'number is the quantum, not a cost.</p>');
  add('<p class="note">The cost is <b>sub-linear in pixels</b> (&times;1.9 for &times;5.1 the ' +
      'pixels), which is the honest shape of twelve render passes: a fixed ~0.020 ms of attachment ' +
      'setup dominates at demo resolutions, and ~0.025 ms per megapixel is what a bigger canvas ' +
      'actually buys. It does move with resolution, which is what says the instrument is reading ' +
      'the GPU and not the clock.</p>');
  add('<div class="kv">' +
    '<div><span>solver, 16,384 particles</span><b>' + f(b.solver_gpu_ms, 2) + ' ms</b></div>' +
    '<div><span>drawing, before</span><b>' + f(b.render_gpu_ms_1024, 2) + ' ms</b></div>' +
    '<div><span>drawing, with the reconstruction</span><b>' +
      f(b.render_gpu_ms_1024_water_rework, 2) + ' ms</b></div>' +
    '<div><span>60 fps budget</span><b>' + f(b.budget_60fps_ms, 2) + ' ms</b></div>' +
    '</div>');
  add('<p class="note">The frame goes from ' + f(b.total_gpu_ms, 2) + ' ms to ' +
      f(b.total_gpu_ms_water_rework, 2) + ' ms of GPU work against a ' + f(b.budget_60fps_ms, 2) +
      ' ms budget. The solver is still ' + f(100 * b.solver_gpu_ms / b.budget_60fps_ms, 0) +
      '% of it and drawing is ' +
      f(100 * b.render_gpu_ms_1024_water_rework / b.budget_60fps_ms, 1) + '%. Note that the two ' +
      'drawing figures come from different instruments &mdash; only the DELTA was measured against ' +
      'a matched control, and only the delta is carried across.</p>');
  add('<p class="note">One thing that was checked rather than assumed: the foam term is gated on ' +
      'motion, and a gate that never opens looks exactly like a gate that was never written. At the ' +
      'splash peak the reworked water has <b>' + W.foam_gate.peak_near_white_px_after + '</b> ' +
      'whitewater pixels against <b>' + W.foam_gate.peak_near_white_px_before + '</b> before, so it ' +
      'fires. It is restrained, which is what T-020\'s film option is.</p>');
})();

add('<div class="scope" style="margin-top:18px"><b>The trade this treatment makes, found by driving ' +
    'the real page rather than by reasoning about it.</b> Option B makes <em>shallow</em> water ' +
    'nearly invisible, and that is not a bug in the port &mdash; it is what the treatment is. ' +
    'Transmission goes as <code>exp(-absorb&middot;t)</code>, so a thin sheet transmits almost ' +
    'everything, and the demo\'s background is nearly black. Pouring water into an empty scene ' +
    'gives a sheet reading about (16,&nbsp;36,&nbsp;44) against a (6,&nbsp;9,&nbsp;13) background ' +
    'until it pools. The default scene\'s pool is 0.155 of the domain deep and is well clear of ' +
    'this; a small hand-poured puddle is not. The water that shipped before had the opposite ' +
    'failing &mdash; equally bright at every depth, which is why it had no depth cue at all. If the ' +
    'shallow case matters more than the deep one, the knob is <code>absorb</code> (0.52) or a small ' +
    'ambient floor; neither was touched, because neither is T-020\'s.</div>');
add('<p class="note" style="margin-top:14px"><b>What did not change, and was checked rather than ' +
    'assumed:</b> snow, sand and rubber take the same branch they took before and are visible in ' +
    'both halves of every clip above; <code>sim/physics/</code> was not opened; and the layout ' +
    're-measures byte-for-byte identical at all five viewports &mdash; field 390x390 on a phone, ' +
    '658x658 on a laptop, smallest control 40 px, no horizontal overflow.</p>');

// ---------------------------------------------------------------- 4. LAYOUT
add('<h2>5 &nbsp;The page, at five viewport sizes</h2>');
add('<p>A claim about what fits on a screen cannot be made from a stylesheet, so these are real ' +
    'screenshots of the real page in a real GPU-backed window, with the device metrics overridden ' +
    'per size. The measurements under each pair are read off the live layout, not the CSS.</p>');

(function () {
  var wrap = add('<div id="vpwrap"></div>');
  var tabs = $('<div class="tabs"></div>');
  wrap.appendChild(tabs);
  var body = $('<div></div>');
  wrap.appendChild(body);
  function draw(key) {
    var L = D.layout['before_' + key], A = D.layout['after_' + key];
    var meta = D.viewports.filter(function (v) { return v[0] === key; })[0];
    body.innerHTML =
      '<div class="row">' +
      '<div class="shot"><div class="lab">before &mdash; ' + esc(meta[2]) + '</div>' +
      '<img src="' + D.media + 'shots/before_' + key + '.png" alt="before ' + esc(key) + '"></div>' +
      '<div class="shot"><div class="lab">after &mdash; ' + esc(meta[2]) + '</div>' +
      '<img src="' + D.media + 'shots/after_' + key + '.png" alt="after ' + esc(key) + '"></div>' +
      '</div>' +
      '<table style="margin-top:12px"><tr><th></th><th>before</th><th>after</th></tr>' +
      '<tr><td>field, rendered</td><td class="n">' + L.field_w + ' &times; ' + L.field_h +
        '</td><td class="n">' + A.field_w + ' &times; ' + A.field_h + '</td></tr>' +
      '<tr><td>field is square</td><td class="' + (L.field_square ? 'ok' : 'bad') + '">' +
        (L.field_square ? 'yes' : 'NO &mdash; ' + f(L.aspect_error_pct, 0) + '% stretched') +
        '</td><td class="' + (A.field_square ? 'ok' : 'bad') + '">' +
        (A.field_square ? 'yes' : 'NO &mdash; ' + f(A.aspect_error_pct, 0) + '% stretched') + '</td></tr>' +
      '<tr><td>field area, share of viewport</td><td class="n">' + f(100 * L.field_frac_of_viewport, 0) +
        '%</td><td class="n">' + f(100 * A.field_frac_of_viewport, 0) + '%</td></tr>' +
      '<tr><td>smallest control</td><td class="n">' + L.min_button_h + ' px</td><td class="n">' +
        A.min_button_h + ' px</td></tr>' +
      '<tr><td>horizontal overflow</td><td class="n">' + L.doc_overflow_x + ' px</td><td class="n">' +
        A.doc_overflow_x + ' px</td></tr>' +
      '</table>';
  }
  D.viewports.forEach(function (v, i) {
    var b = $('<button' + (i === 0 ? ' class="on"' : '') + '>' + esc(v[1]) + '</button>');
    b.onclick = function () {
      [].forEach.call(tabs.children, function (c) { c.classList.remove('on'); });
      b.classList.add('on');
      draw(v[0]);
    };
    tabs.appendChild(b);
  });
  draw(D.viewports[0][0]);
})();

add('<p class="note" style="margin-top:14px"><b>The share of the viewport taken by the field goes ' +
    'DOWN on a ' +
    'phone in portrait (81% to 46%), and that is the fix, not a regression.</b> The old 81% was 81% ' +
    'of a rectangle showing a square domain: bigger, and wrong. A correct square at 390 px wide is ' +
    'the largest correct field a 390 px viewport can hold, and the rest of the stage is deliberately ' +
    'left empty so nothing competes with it.</p>');
add('<p class="note">The failure the old layout had is not crowding, it is ' +
    'a <b>distortion</b>: the frame was sized <code>height:100%; max-width:100%</code>, so on any ' +
    'viewport taller than it is wide the height won and the width clamped. The canvas backing store ' +
    'is square and the simulation domain is the unit square, so a non-square frame does not crop ' +
    'the simulation &mdash; it stretches it. On an iPhone in portrait the physics was being shown ' +
    '42% too tall. The landscape phone had the opposite problem and it was worse: a stacked control ' +
    'bar took 251 px of a 390 px viewport and left the field a 141 px square.</p>');

// ---------------------------------------------------------------- 5. scope
add('<h2>6 &nbsp;What this does and does not show</h2>');
add('<div class="scope"><b>Scope.</b> Every timing here is <b>one GPU (' + esc(D.device) + '), one ' +
    'browser (Chromium/WebGPU), one scene</b>, and the solver numbers move by ~15% between repeat ' +
    'runs of the identical build, so treat differences under that as noise. "Looks better" is a ' +
    'judgement and is labelled as one everywhere on this page; what was <em>measured</em> is frame ' +
    'cost, the buoyancy ordering, and the rendered geometry of the layout. The buoyancy result is ' +
    'one scene at one particle density with one blob radius &mdash; it shows that density now ' +
    'reaches the grid and produces the canonical ordering, not that the page has a calibrated ' +
    'multi-phase contact model, which it does not. No physical device was tested: the viewport ' +
    'sizes are device-metric overrides in a desktop browser, so they establish what fits and what ' +
    'the layout does, not how a real phone\'s touch handling or thermal budget behaves.</div>');

add('<h2>7 &nbsp;The whole page, before and after</h2>');
add('<p class="note">Both changes at once, which is what a visitor actually sees. Kept separate ' +
    'from the two single-variable comparisons above on purpose.</p>');
add('<div class="card"><video src="' + D.media + 'cmp_page.mp4" controls loop muted playsinline ' +
    'style="width:100%"></video></div>');
</script>
</body>
"""


def check_as_the_dashboard_will_see_it(html):
    """Parse the page's script the way the DASHBOARD will, not the way a file server will.

    The dashboard hands `custom_html` to an iframe through a `srcdoc` attribute, and the HTML parser
    entity-decodes an attribute value before the inner document is parsed. So every `&mdash;`,
    `&amp;` and `&nbsp;` written inside a JS string literal is decoded first and the script that runs
    is NOT the script in the file. Opening the standalone file therefore proves nothing about the
    embedded copy, which is exactly how a page that rendered perfectly on disk arrived on the
    dashboard as an empty frame.
    """
    import html as _html
    import shutil
    import subprocess
    import tempfile

    dec = _html.unescape(html)
    body = dec[dec.index("<script>") + 8: dec.rindex("</script>")]
    node = shutil.which("node")
    if not node:
        print("WARNING: node not found, could not syntax-check the decoded script")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(body)
        tmp = f.name
    r = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
    pathlib.Path(tmp).unlink(missing_ok=True)
    if r.returncode:
        raise SystemExit("the page is broken AFTER srcdoc entity decoding:\n" + r.stderr[:1200])
    print("decoded script parses clean")


def main():
    html = HTML.replace("__DATA__", json.dumps(DATA))
    check_as_the_dashboard_will_see_it(html)
    (RUN / "bespoke_page.html").write_text(html, encoding="utf-8")
    print("wrote bespoke_page.html", len(html), "bytes")
    return html


if __name__ == "__main__":
    main()
