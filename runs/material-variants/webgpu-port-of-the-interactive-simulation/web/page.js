// page.js -- the charts on the bespoke task page. Everything is drawn as inline SVG from the
// numbers in window.PAGE (extracted from metrics.json at build time); no images, no libraries.
(function () {
  'use strict';
  var D = window.PAGE;
  var NS = 'http://www.w3.org/2000/svg';
  var COL = { webgpu: '#6fd3ee', javascript: '#ff9d5c', taichi_cuda: '#b58cf0',
              grid: '#8fd9b6', p2g: '#6fd3ee', g2p: '#ff9d5c', encode: '#b58cf0',
              muted: '#7f8ea3', line: '#1e2733', bad: '#ff7a7a', good: '#8fd9b6' };
  var NAME = { webgpu: 'WebGPU', javascript: 'JavaScript', taichi_cuda: 'Taichi / CUDA' };

  function E(t, a, kids) {
    var e = document.createElementNS(NS, t);
    for (var k in a) if (a[k] !== null && a[k] !== undefined) e.setAttribute(k, a[k]);
    (kids || []).forEach(function (c) { e.appendChild(c); });
    return e;
  }
  function T(x, y, s, o) {
    o = o || {};
    var e = E('text', { x: x, y: y, fill: o.fill || COL.muted, 'font-size': o.size || 11,
      'text-anchor': o.anchor || 'start', 'font-family': o.mono ? 'ui-monospace,Menlo,Consolas,monospace' : 'inherit',
      'font-weight': o.weight || 400, transform: o.transform });
    e.textContent = s;
    return e;
  }
  function svg(w, h) {
    return E('svg', { viewBox: '0 0 ' + w + ' ' + h, preserveAspectRatio: 'xMidYMid meet' });
  }
  function path(d, stroke, width, extra) {
    return E('path', Object.assign({ d: d, fill: 'none', stroke: stroke, 'stroke-width': width || 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, extra || {}));
  }
  function fmt(v) {
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1000) return Math.round(v / 1000) + 'k';
    return String(Math.round(v));
  }

  // ---------------------------------------------------------------- launch-floor bars
  (function () {
    var host = document.getElementById('floorChart');
    if (!host) return;
    var W = 900, H = 190, L = 260, R = 40;
    var s = svg(W, H);
    var rows = [
      { lab: 'empty CUDA kernel, launched from Python', v: D.floor.taichi_cuda_empty_kernel_from_python, c: COL.taichi_cuda },
      { lab: 'empty WGSL dispatch, inside one recorded command buffer', v: D.floor.webgpu_empty_dispatch_in_recorded_buffer, c: COL.webgpu }
    ];
    var max = rows[0].v * 1.18;
    rows.forEach(function (r, i) {
      var y = 42 + i * 62;
      var w = (W - L - R) * r.v / max;
      s.appendChild(T(L - 12, y + 17, r.lab, { anchor: 'end', size: 12.5, fill: '#dfe6ee' }));
      s.appendChild(E('rect', { x: L, y: y, width: Math.max(w, 2), height: 26, rx: 4, fill: r.c, opacity: 0.85 }));
      s.appendChild(T(L + Math.max(w, 2) + 10, y + 18, r.v.toFixed(2) + ' \u00b5s',
        { size: 13, mono: true, fill: r.c, weight: 600 }));
    });
    s.appendChild(T(L, H - 26, 'The same substep issues 3 (WebGPU) or 4 (Taichi) of these, ' +
      D.spf + ' times a frame.', { size: 12 }));
    s.appendChild(T(L, H - 8, 'Doing nothing therefore costs ' +
      (D.floor.taichi_cuda_empty_kernel_from_python * 4 * D.spf / 1000).toFixed(0) + ' ms/frame from Python and ' +
      (D.floor.webgpu_empty_dispatch_in_recorded_buffer * 3 * D.spf / 1000).toFixed(2) +
      ' ms/frame in WebGPU \u2014 a ' + D.floor.ratio.toFixed(0) + '\u00d7 difference in the floor.',
      { size: 12, fill: '#dfe6ee' }));
    host.appendChild(s);
  })();

  // ---------------------------------------------------------------- three-way cost curve
  (function () {
    var host = document.getElementById('costChart');
    if (!host) return;
    var mode = 'time';
    var impls = ['webgpu', 'javascript', 'taichi_cuda'];
    function draw() {
      host.innerHTML = '';
      var W = 900, H = 420, L = 66, R = 22, Tp = 18, B = 52;
      var s = svg(W, H);
      if (mode === 'time') {
        var pts = {};
        impls.forEach(function (k) {
          pts[k] = D.threeWay.filter(function (r) { return r.impl === k; })
            .sort(function (a, b) { return a.n - b.n; });
        });
        var allN = D.threeWay.map(function (r) { return r.n; });
        var allY = D.threeWay.map(function (r) { return r.frame_ms; });
        var x0 = Math.log10(Math.min.apply(null, allN)), x1 = Math.log10(Math.max.apply(null, allN));
        var y0 = Math.log10(Math.min.apply(null, allY) * 0.75), y1 = Math.log10(Math.max.apply(null, allY) * 1.3);
        var X = function (n) { return L + (Math.log10(n) - x0) / (x1 - x0) * (W - L - R); };
        var Y = function (v) { return H - B - (Math.log10(v) - y0) / (y1 - y0) * (H - Tp - B); };
        [0.5, 1, 2, 5, 10, 16.67, 50, 100, 250, 500, 900].forEach(function (v) {
          if (Math.log10(v) < y0 || Math.log10(v) > y1) return;
          var is60 = v === 16.67;
          s.appendChild(E('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v),
            stroke: is60 ? '#4a5c6b' : COL.line, 'stroke-width': is60 ? 1.6 : 1,
            'stroke-dasharray': is60 ? '7 5' : null }));
          s.appendChild(T(L - 8, Y(v) + 4, is60 ? '16.7' : String(v), { anchor: 'end', size: 10.5, mono: true }));
        });
        s.appendChild(T(L + 8, Y(16.67) - 7, '60 fps budget', { anchor: 'start', size: 11, fill: '#8ea3b5' }));
        [500, 2048, 8192, 32768, 131072, 262144].forEach(function (n) {
          if (Math.log10(n) < x0 - 1e-9 || Math.log10(n) > x1 + 1e-9) return;
          s.appendChild(E('line', { x1: X(n), x2: X(n), y1: Tp, y2: H - B, stroke: COL.line, 'stroke-width': 1 }));
          s.appendChild(T(X(n), H - B + 17, fmt(n), { anchor: 'middle', size: 10.5, mono: true }));
        });
        impls.forEach(function (k) {
          var p = pts[k];
          if (!p.length) return;
          var d = p.map(function (r, i) { return (i ? 'L' : 'M') + X(r.n).toFixed(1) + ' ' + Y(r.frame_ms).toFixed(1); }).join(' ');
          s.appendChild(path(d, COL[k], 2.4));
          p.forEach(function (r) {
            s.appendChild(E('circle', { cx: X(r.n), cy: Y(r.frame_ms), r: 3.1, fill: COL[k] }));
          });
          // label at the series' own midpoint, not its last point, so nothing collides with the
          // right edge or with the 60 fps rule
          var mid = p[Math.max(0, Math.floor(p.length / 2) - 1)];
          s.appendChild(T(X(mid.n), Y(mid.frame_ms) - 13, NAME[k],
            { anchor: 'middle', size: 12, fill: COL[k], weight: 600 }));
        });
        s.appendChild(T(L, H - 10, 'particles  (log)', { size: 11.5 }));
        var ym = (Tp + H - B) / 2;
        s.appendChild(T(18, ym, 'frame time, ms (log)',
          { size: 11.5, anchor: 'middle', transform: 'rotate(-90 18 ' + ym + ')' }));
      } else {
        var rows = impls.map(function (k) { return { k: k, v: D.budget[k] }; });
        var max = Math.max.apply(null, rows.map(function (r) { return r.v || 1; })) * 1.15;
        var LB = 150;
        rows.forEach(function (r, i) {
          var y = 44 + i * 92;
          s.appendChild(T(LB - 12, y + 26, NAME[r.k], { anchor: 'end', size: 13.5, fill: '#dfe6ee', weight: 600 }));
          if (r.v === null) {
            s.appendChild(E('rect', { x: LB, y: y, width: 4, height: 40, rx: 2, fill: COL.bad }));
            s.appendChild(T(LB + 16, y + 20, 'never reaches 60 fps',
              { size: 14, fill: COL.bad, weight: 600 }));
            s.appendChild(T(LB + 16, y + 40, 'flat ~55 ms/frame from 500 to 16384 particles \u2014 ' +
              'the cost is launch overhead, so no particle count helps', { size: 11.5 }));
          } else {
            var w = (W - LB - 40) * r.v / max;
            s.appendChild(E('rect', { x: LB, y: y, width: Math.max(w, 3), height: 40, rx: 5, fill: COL[r.k], opacity: 0.85 }));
            s.appendChild(T(LB + Math.max(w, 3) + 12, y + 26, Math.round(r.v).toLocaleString() + ' particles',
              { size: 15, mono: true, fill: COL[r.k], weight: 600 }));
          }
        });
        s.appendChild(T(LB, H - 16, 'Largest particle count whose measured frame still fits 16.67 ms, ' +
          'interpolated between the two bracketing measurements. Compute only \u2014 rendering excluded.',
          { size: 11.5 }));
      }
      host.appendChild(s);
    }
    var tg = document.getElementById('costToggle');
    tg.addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      [].forEach.call(tg.children, function (c) { c.classList.remove('on'); });
      b.classList.add('on'); mode = b.dataset.v; draw();
    });
    draw();
  })();

  // ---------------------------------------------------------------- accuracy vs scale
  (function () {
    var host = document.getElementById('accChart');
    var tbl = document.getElementById('accTable');
    if (!host) return;
    var scene = 'drop';
    function draw() {
      host.innerHTML = ''; tbl.innerHTML = '';
      var A = D.accuracy[scene];
      var W = 900, H = 380, L = 74, R = 30, Tp = 20, B = 58;
      var s = svg(W, H);
      var fixed = A.variants.filter(function (v) { return v.atomics === 'fixed'; })
        .sort(function (a, b) { return a.kM - b.kM; });
      var cas = A.variants.find(function (v) { return v.atomics === 'casf32'; });
      var ks = fixed.map(function (v) { return v.kM; });
      var kmin = Math.min.apply(null, ks) - 1, kmax = Math.max.apply(null, ks) + 1;
      var vals = A.variants.map(function (v) { return v.traj_rmse; })
        .concat([A.band.self_noise, A.band['perturbed_ic_1e-7']]);
      var y0 = Math.log10(Math.min.apply(null, vals) * 0.45);
      var y1 = Math.log10(Math.max.apply(null, vals) * 3.2);
      var X = function (k) { return L + (k - kmin) / (kmax - kmin) * (W - L - R); };
      var Y = function (v) { return H - B - (Math.log10(v) - y0) / (y1 - y0) * (H - Tp - B); };
      for (var e = Math.floor(y0); e <= Math.ceil(y1); e++) {
        if (e < y0 || e > y1) continue;
        s.appendChild(E('line', { x1: L, x2: W - R, y1: Y(Math.pow(10, e)), y2: Y(Math.pow(10, e)),
          stroke: COL.line, 'stroke-width': 1 }));
        s.appendChild(T(L - 8, Y(Math.pow(10, e)) + 4, '1e' + e, { anchor: 'end', size: 10.5, mono: true }));
      }
      // the band: everything below the top edge is indistinguishable from chaos
      s.appendChild(E('rect', { x: L, y: Y(A.band['perturbed_ic_1e-7']), width: W - L - R,
        height: Math.max(1, (H - B) - Y(A.band['perturbed_ic_1e-7'])), fill: '#22303d', opacity: 0.75 }));
      // band labels go on the LEFT: at coarse k the curve is high up, so the lower-left is empty
      [['perturbed_ic_1e-7', 'canonical with a one-ULP nudge to the initial positions', -6],
       ['self_noise', 'canonical re-run against itself', 14]]
        .forEach(function (p) {
          var v = A.band[p[0]];
          s.appendChild(E('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: COL.webgpu,
            'stroke-width': 1.2, 'stroke-dasharray': p[0] === 'self_noise' ? '2 4' : '7 4', opacity: 0.85 }));
          s.appendChild(T(L + 8, Y(v) + p[2], p[1], { anchor: 'start', size: 10.5, fill: '#7fb9cf' }));
        });
      var d = fixed.map(function (v, i) { return (i ? 'L' : 'M') + X(v.kM).toFixed(1) + ' ' + Y(v.traj_rmse).toFixed(1); }).join(' ');
      s.appendChild(path(d, COL.webgpu, 2.4));
      fixed.forEach(function (v) {
        var inside = v.traj_rmse <= A.band['perturbed_ic_1e-7'];
        s.appendChild(E('circle', { cx: X(v.kM), cy: Y(v.traj_rmse), r: 4.6,
          fill: inside ? COL.good : COL.webgpu }));
        s.appendChild(T(X(v.kM), H - B + 18, '2^' + v.kM, { anchor: 'middle', size: 11, mono: true }));
        s.appendChild(T(X(v.kM), Y(v.traj_rmse) - 11, (v.vs_perturbed_ic < 10 ? v.vs_perturbed_ic.toFixed(1) : Math.round(v.vs_perturbed_ic)) + '\u00d7',
          { anchor: 'middle', size: 10.5, mono: true, fill: inside ? COL.good : '#a9c3d2' }));
      });
      if (cas) {
        s.appendChild(E('line', { x1: L, x2: W - R, y1: Y(cas.traj_rmse), y2: Y(cas.traj_rmse),
          stroke: COL.javascript, 'stroke-width': 1.8, 'stroke-dasharray': '5 4' }));
        s.appendChild(T(W - R, Y(cas.traj_rmse) - 8, 'exact f32 (compare-and-swap): ' + cas.traj_rmse.toExponential(2),
          { size: 11, anchor: 'end', fill: COL.javascript, weight: 600 }));
      }
      s.appendChild(T(L, H - 12, 'fixed-point resolution: quanta per particle mass', { size: 11.5 }));
      var ym2 = (Tp + H - B) / 2;
      s.appendChild(T(18, ym2, 'traj_rmse vs canonical (domain lengths)',
        { size: 11.5, anchor: 'middle', transform: 'rotate(-90 18 ' + ym2 + ')' }));
      s.appendChild(T(W - R, Tp + 6, A.desc, { anchor: 'end', size: 11.5, fill: '#dfe6ee' }));
      host.appendChild(s);

      var h = '<table><tr><th>accumulator</th><th>traj_rmse</th><th>&times; self-noise</th>' +
        '<th>&times; IC nudge</th><th>final-frame gap</th><th>node ceiling</th></tr>';
      A.variants.forEach(function (v) {
        var inside = v.traj_rmse <= A.band['perturbed_ic_1e-7'];
        var cls = inside ? 'good' : (v.vs_perturbed_ic > 20 ? 'bad' : 'warm');
        h += '<tr><td>' + (v.atomics === 'fixed' ? 'fixed 2<sup>' + v.kM + '</sup>' : 'exact f32 (CAS)') +
          '</td><td class="' + cls + '">' + v.traj_rmse.toExponential(2) + '</td><td>' +
          v.vs_self_noise.toFixed(1) + '</td><td class="' + cls + '">' + v.vs_perturbed_ic.toFixed(1) +
          '</td><td>' + v.final_frame_dist.toExponential(2) + '</td><td>' +
          (v.atomics === 'fixed' ? Math.round(v.mass_ceiling) + ' pm' : '&mdash;') + '</td></tr>';
      });
      tbl.innerHTML = h + '</table>';
    }
    var tg = document.getElementById('sceneToggle');
    tg.addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      [].forEach.call(tg.children, function (c) { c.classList.remove('on'); });
      b.classList.add('on'); scene = b.dataset.v; draw();
    });
    draw();
  })();

  // ---------------------------------------------------------------- node occupancy vs density
  (function () {
    var host = document.getElementById('occChart');
    if (!host) return;
    var W = 900, H = 320, L = 70, R = 196, Tp = 20, B = 52;
    var s = svg(W, H);
    var o = D.occupancy.slice().sort(function (a, b) { return a.particles_per_cell - b.particles_per_cell; });
    var x0 = Math.log10(o[0].particles_per_cell * 0.8), x1 = Math.log10(o[o.length - 1].particles_per_cell * 1.25);
    var y1 = Math.log10(1400), y0 = Math.log10(1.5);
    var X = function (v) { return L + (Math.log10(v) - x0) / (x1 - x0) * (W - L - R); };
    var Y = function (v) { return H - B - (Math.log10(v) - y0) / (y1 - y0) * (H - Tp - B); };
    [2, 5, 10, 20, 50, 100, 250, 500, 1000].forEach(function (v) {
      if (Math.log10(v) < y0 || Math.log10(v) > y1) return;
      s.appendChild(E('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: COL.line, 'stroke-width': 1 }));
      s.appendChild(T(L - 8, Y(v) + 4, String(v), { anchor: 'end', size: 10.5, mono: true }));
    });
    [{ k: 20, c: '#3d5566' }, { k: 22, c: '#4d7fa3' }, { k: 24, c: COL.webgpu }, { k: 26, c: '#8fd9b6' }, { k: 30, c: COL.bad }]
      .forEach(function (r) {
        var cap = Math.pow(2, 32 - r.k);
        if (cap < Math.pow(10, y0) || cap > Math.pow(10, y1)) return;
        s.appendChild(E('line', { x1: L, x2: W - R, y1: Y(cap), y2: Y(cap), stroke: r.c,
          'stroke-width': 1.4, 'stroke-dasharray': '6 4' }));
        s.appendChild(T(W - R + 8, Y(cap) + 4, 'ceiling at 2^' + r.k + ' \u2192 ' + cap + ' pm',
          { size: 10.5, mono: true, fill: r.c }));
      });
    var d = o.map(function (r, i) { return (i ? 'L' : 'M') + X(r.particles_per_cell).toFixed(1) + ' ' + Y(r.max_node_mass_pm).toFixed(1); }).join(' ');
    s.appendChild(path(d, '#ffffff', 2.4));
    o.forEach(function (r) {
      s.appendChild(E('circle', { cx: X(r.particles_per_cell), cy: Y(r.max_node_mass_pm), r: 3.4, fill: '#fff' }));
      s.appendChild(T(X(r.particles_per_cell), H - B + 17, r.particles_per_cell.toFixed(1),
        { anchor: 'middle', size: 10.5, mono: true }));
    });
    s.appendChild(T(L, H - 10, 'particles per cell', { size: 11.5 }));
    var ym3 = (Tp + H - B) / 2;
    s.appendChild(T(18, ym3, 'heaviest node (particle masses)',
      { size: 11.5, anchor: 'middle', transform: 'rotate(-90 18 ' + ym3 + ')' }));
    host.appendChild(s);
  })();

  // ---------------------------------------------------------------- phase split
  (function () {
    var host = document.getElementById('phaseChart');
    if (!host) return;
    var mode = 'abs';
    function draw() {
      host.innerHTML = '';
      var rows = D.phases;
      var W = 900, H = 330, L = 62, R = 20, Tp = 18, B = 56;
      var s = svg(W, H);
      var keys = ['grid', 'p2g', 'g2p'];
      var totals = rows.map(function (r) { return keys.reduce(function (a, k) { return a + r[k]; }, 0); });
      var max = mode === 'abs' ? Math.max.apply(null, totals) * 1.08 : 1;
      var bw = (W - L - R) / rows.length * 0.66;
      rows.forEach(function (r, i) {
        var cx = L + (i + 0.5) * (W - L - R) / rows.length;
        var tot = totals[i], acc = 0;
        keys.forEach(function (k) {
          var v = mode === 'abs' ? r[k] : r[k] / tot;
          var h = v / max * (H - Tp - B);
          var y = H - B - acc - h;
          s.appendChild(E('rect', { x: cx - bw / 2, y: y, width: bw, height: Math.max(h, 0.6),
            fill: COL[k], opacity: 0.88 }));
          acc += h;
        });
        s.appendChild(T(cx, H - B + 16, fmt(r.n), { anchor: 'middle', size: 10, mono: true }));
        if (mode === 'abs') {
          s.appendChild(T(cx, H - B - acc - 6, tot.toFixed(1), { anchor: 'middle', size: 9.5, mono: true }));
        }
      });
      var ymp = (Tp + H - B) / 2;
      s.appendChild(T(16, ymp, mode === 'abs' ? 'GPU time per frame' : 'share of the frame',
        { size: 11.5, anchor: 'middle', transform: 'rotate(-90 16 ' + ymp + ')' }));
      var ticks = mode === 'abs' ? [0, 5, 10, 15, 20] : [0, 0.25, 0.5, 0.75, 1];
      ticks.forEach(function (v) {
        if (v > max) return;
        var y = H - B - v / max * (H - Tp - B);
        s.appendChild(E('line', { x1: L, x2: W - R, y1: y, y2: y, stroke: COL.line, 'stroke-width': 1 }));
        s.appendChild(T(L - 8, y + 4, mode === 'abs' ? v + ' ms' : (v * 100) + '%',
          { anchor: 'end', size: 10.5, mono: true }));
      });
      s.appendChild(T(L, H - 10, 'particles', { size: 11.5 }));
      host.appendChild(s);
      var lg = document.createElement('div');
      lg.className = 'legend';
      lg.innerHTML = '<span><i style="background:' + COL.grid + '"></i>grid update (+ fused clear), ' +
        D.spf + ' dispatches</span><span><i style="background:' + COL.p2g +
        '"></i>P2G, the atomic scatter</span><span><i style="background:' + COL.g2p +
        '"></i>G2P, the gather</span>';
      host.appendChild(lg);
    }
    var tg = document.getElementById('phaseToggle');
    tg.addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      [].forEach.call(tg.children, function (c) { c.classList.remove('on'); });
      b.classList.add('on'); mode = b.dataset.v; draw();
    });
    draw();
  })();

  // ---------------------------------------------------------------- metric definitions on hover
  [].forEach.call(document.querySelectorAll('.def'), function (e) {
    var d = window.DEFS[e.dataset.def];
    if (d) e.setAttribute('title', d);
  });

  // ---------------------------------------------------------------- the live sim
  if (window.MPMDemo) {
    var host = document.getElementById('demo');
    if (host) MPMDemo(host, { n: 8000 });
  }
})();
