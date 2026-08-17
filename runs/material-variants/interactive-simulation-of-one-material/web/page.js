// page.js -- the bespoke task page's own charts and toggles. DATA is inlined by build_page.py.
(function () {
  'use strict';
  var D = window.PAGE_DATA;
  var $ = function (s) { return document.querySelector(s); };
  var $$ = function (s) { return Array.prototype.slice.call(document.querySelectorAll(s)); };

  // ---------------------------------------------------------------- metric definitions on hover
  var pop = document.createElement('div'); pop.id = 'pop'; document.body.appendChild(pop);
  function showPop(el) {
    var k = el.getAttribute('data-def');
    var d = D.defs[k]; if (!d) return;
    pop.textContent = d.label + '\n\n' + d.short + '\n\nformula: ' + d.formula +
      '\nunits: ' + d.units + (d.caution ? '\n\ncaution: ' + d.caution : '');
    var r = el.getBoundingClientRect();
    pop.classList.add('show');
    var w = pop.offsetWidth, h = pop.offsetHeight;
    pop.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left)) + 'px';
    pop.style.top = (r.bottom + h + 10 > window.innerHeight ? r.top - h - 8 : r.bottom + 8) + 'px';
  }
  document.addEventListener('pointerover', function (e) {
    var t = e.target.closest ? e.target.closest('dfn[data-def]') : null;
    if (t) showPop(t);
  });
  document.addEventListener('pointerout', function (e) {
    if (e.target.closest && e.target.closest('dfn[data-def]')) pop.classList.remove('show');
  });
  document.addEventListener('click', function (e) {
    var t = e.target.closest ? e.target.closest('dfn[data-def]') : null;
    if (t) showPop(t); else pop.classList.remove('show');
  });

  // ---------------------------------------------------------------- tiny SVG chart helpers
  function svg(w, h) {
    var s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    s.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    s.setAttribute('width', '100%');
    s.style.display = 'block';
    return s;
  }
  function node(tag, attrs, text) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // reloading an identical video src on every resize would restart playback for no reason
  function setSrc(v, url) { if (v.getAttribute('src') !== url) v.setAttribute('src', url); }

  // =============================================================== 1. divergence, per scene
  var SCENE = 'launch';
  function drawDiv() {
    var s = D.scenes[SCENE];
    var W = 620, H = 270, L = 62, R = 12, T = 28, B = 34;
    var el = $('#divchart'); el.innerHTML = '';
    var g = svg(W, H); el.appendChild(g);
    var lo = -9, hi = -2;                                   // log10 y range
    var t0 = s.times[0], t1 = s.times[s.times.length - 1];
    var X = function (t) { return L + (t - t0) / (t1 - t0) * (W - L - R); };
    var Y = function (v) { var l = Math.log10(Math.max(v, 1e-12)); return T + (hi - l) / (hi - lo) * (H - T - B); };
    for (var e = lo; e <= hi; e++) {
      g.appendChild(node('line', { x1: L, x2: W - R, y1: Y(Math.pow(10, e)), y2: Y(Math.pow(10, e)), stroke: '#1c2430', 'stroke-width': 1 }));
      g.appendChild(node('text', { x: L - 7, y: Y(Math.pow(10, e)) + 4, fill: '#5c6b7d', 'font-size': 10, 'text-anchor': 'end' }, '1e' + e));
    }
    [['canonical_self_noise', '#c58cf0', 1.6], ['canonical_perturbed_ic', '#5fd39a', 1.6],
    ['port_vs_canonical', '#ff9d5c', 2.4]].forEach(function (c) {
      var d = '';
      for (var i = 0; i < s.times.length; i++) d += (i ? 'L' : 'M') + X(s.times[i]).toFixed(1) + ',' + Y(s[c[0]][i]).toFixed(1);
      g.appendChild(node('path', { d: d, fill: 'none', stroke: c[1], 'stroke-width': c[2], 'stroke-linejoin': 'round' }));
    });
    g.appendChild(node('text', { x: (L + W - R) / 2, y: H - 8, fill: '#7f8ea3', 'font-size': 10.5, 'text-anchor': 'middle' }, 'simulated time (s)'));
    g.appendChild(node('text', { x: 8, y: 12, fill: '#7f8ea3', 'font-size': 10.5 },
      'mean particle distance from canonical run A (domain lengths)'));
    $('#divnums').innerHTML = [
      ['port vs canonical', s.traj_rmse.port_vs_canonical, '#ff9d5c'],
      ['canonical vs itself (GPU atomics reorder)', s.traj_rmse.canonical_self_noise, '#c58cf0'],
      ['canonical vs the same run with the start nudged by 1e-7', s.traj_rmse.canonical_perturbed_ic, '#5fd39a']
    ].map(function (r) {
      return '<tr><td><s style="background:' + r[2] + ';width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:7px"></s>' +
        r[0] + '</td><td class="num">' + r[1].toExponential(2) + '</td></tr>';
    }).join('');
    setSrc($('#divvid'), D.media + (SCENE === 'launch' ? 'port_vs_canonical_launch.mp4' : 'port_vs_canonical.mp4'));
    setSrc($('#dtvid'), D.media + (SCENE === 'launch' ? 'dt_sweep_launch.mp4' : 'dt_sweep.mp4'));
    $('#divcap').textContent = s.caption;
    drawDt();
  }

  // =============================================================== 2. the budget chart
  var BMODE = 'us';
  function drawBudget() {
    var W = 640, H = 300, L = 56, R = 118, T = 18, B = 40;
    var el = $('#budchart'); el.innerHTML = '';
    var g = svg(W, H); el.appendChild(g);
    var series = [
      { name: 'browser JS, sparse grid (the port)', c: '#ff9d5c', pts: D.bench.map(function (r) { return [r.n, r.sparse]; }) },
      { name: 'browser JS, dense grid (Taichi loop)', c: '#c58cf0', pts: D.bench.map(function (r) { return [r.n, r.dense]; }) },
      { name: 'canonical Taichi / CUDA on an RTX 4090', c: '#6fd3ee', pts: D.gpu.map(function (r) { return [r.n, r.us]; }) }
    ];
    var conv = function (us) { return BMODE === 'us' ? us : Math.min(240, 1000 / (us * D.spf / 1000)); };
    var lo, hi, ticks;
    if (BMODE === 'us') { lo = Math.log10(40); hi = Math.log10(800); }
    else { lo = Math.log10(2); hi = Math.log10(240); }
    var nx0 = Math.log10(450), nx1 = Math.log10(18000);
    var X = function (n) { return L + (Math.log10(n) - nx0) / (nx1 - nx0) * (W - L - R); };
    var Y = function (v) { return T + (hi - Math.log10(Math.max(v, 1e-6))) / (hi - lo) * (H - T - B); };
    ticks = BMODE === 'us' ? [50, 100, 200, 400, 800] : [5, 10, 30, 60, 120, 240];
    ticks.forEach(function (v) {
      g.appendChild(node('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: '#1c2430' }));
      g.appendChild(node('text', { x: L - 7, y: Y(v) + 4, fill: '#5c6b7d', 'font-size': 10, 'text-anchor': 'end' }, v));
    });
    [500, 1000, 2000, 4000, 8000, 16000].forEach(function (n) {
      g.appendChild(node('line', { x1: X(n), x2: X(n), y1: T, y2: H - B, stroke: '#151d27' }));
      g.appendChild(node('text', { x: X(n), y: H - B + 15, fill: '#5c6b7d', 'font-size': 10, 'text-anchor': 'middle' }, n >= 1000 ? (n / 1000) + 'k' : n));
    });
    // the 60 fps real-time line
    var target = BMODE === 'us' ? D.budget_us : 60;
    g.appendChild(node('line', { x1: L, x2: W - R, y1: Y(target), y2: Y(target), stroke: '#5fd39a', 'stroke-width': 1.6, 'stroke-dasharray': '6 4' }));
    g.appendChild(node('text', { x: W - R - 4, y: Y(target) - 6, fill: '#5fd39a', 'font-size': 10.5, 'text-anchor': 'end' },
      BMODE === 'us' ? '60 fps at real time (' + D.budget_us.toFixed(0) + ' us/substep)' : '60 fps'));
    series.forEach(function (s, si) {
      var d = '';
      s.pts.forEach(function (p, i) { d += (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(conv(p[1])).toFixed(1); });
      g.appendChild(node('path', { d: d, fill: 'none', stroke: s.c, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }));
      s.pts.forEach(function (p) { g.appendChild(node('circle', { cx: X(p[0]), cy: Y(conv(p[1])), r: 2.6, fill: s.c })); });
      // the last points of the two JS curves nearly coincide on a log axis, so nudge their labels apart
      var last = s.pts[s.pts.length - 1];
      var dy = si === 0 ? 13 : (si === 1 ? -7 : 4);
      g.appendChild(node('text', { x: W - R + 6, y: Y(conv(last[1])) + dy, fill: s.c, 'font-size': 10 },
        si === 0 ? 'JS sparse' : (si === 1 ? 'JS dense' : 'CUDA')));
    });
    // the crossing that matters
    var cy = Y(target), cx = X(D.budget_particles);
    g.appendChild(node('circle', { cx: cx, cy: cy, r: 5, fill: '#dfe6ee', stroke: '#0a0e14', 'stroke-width': 2 }));
    g.appendChild(node('text', { x: cx + 9, y: cy + 17, fill: '#dfe6ee', 'font-size': 11.5, 'font-weight': 700 }, D.budget_particles + ' particles'));
    g.appendChild(node('text', { x: (L + W - R) / 2, y: H - 6, fill: '#7f8ea3', 'font-size': 10.5, 'text-anchor': 'middle' }, 'particles'));
    g.appendChild(node('text', { x: 8, y: 13, fill: '#7f8ea3', 'font-size': 10.5 },
      BMODE === 'us' ? 'microseconds per substep' : 'sustained fps at real time'));
  }

  // =============================================================== 3. the timestep table
  function drawDt() {
    var rows = D.scenes[SCENE].dt_sweep;
    var floor = D.scenes[SCENE].traj_rmse.canonical_self_noise;
    $('#dttable tbody').innerHTML = rows.map(function (r) {
      var bad = !r.finite || r.rmse > 1e-2;
      var cls = !r.finite ? 'bad' : (r.rmse > 1e-3 ? 'bad' : (r.rmse > 10 * floor ? 'warn' : 'good'));
      return '<tr><td class="num">' + r.mult + '&times;  (' + r.dt.toExponential(1) + ')</td>' +
        '<td class="num">' + r.spf + '</td>' +
        '<td class="num">' + r.speedup.toFixed(2) + '&times;</td>' +
        '<td class="num ' + cls + '">' + (r.finite ? r.rmse.toExponential(2) : 'non-finite') + '</td>' +
        '<td class="num ' + cls + '">' + (!r.finite ? 'blew up' : (bad ? 'wrong' : (r.rmse > 10 * floor ? 'drifting' : 'exact'))) + '</td></tr>';
    }).join('');
  }

  // =============================================================== wire up
  $$('.tabs[data-k=scene] button').forEach(function (b) {
    b.addEventListener('click', function () {
      $$('.tabs[data-k=scene] button').forEach(function (o) { o.classList.remove('on'); });
      b.classList.add('on'); SCENE = b.getAttribute('data-v'); drawDiv();
    });
  });
  $$('.tabs[data-k=bmode] button').forEach(function (b) {
    b.addEventListener('click', function () {
      $$('.tabs[data-k=bmode] button').forEach(function (o) { o.classList.remove('on'); });
      b.classList.add('on'); BMODE = b.getAttribute('data-v'); drawBudget();
    });
  });
  window.addEventListener('resize', function () { drawDiv(); drawBudget(); });
  drawDiv(); drawBudget();
  if (window.MPMDemo) window.__demo = MPMDemo.mount(document.getElementById('demo'), { n: 1000 });
})();
