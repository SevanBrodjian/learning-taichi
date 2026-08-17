// demo.js -- the interactive shell around mpm-elastic.js.
// Transplant contract: needs only params.js + mpm-elastic.js + demo.css. No dashboard, no server,
// no network. Mount with MPMDemo.mount(element).
(function (root) {
  'use strict';
  var M = root.MPMElastic;
  var P = M.PARAMS;
  var NG = P.n_grid;

  var CANVAS = 720;          // fixed backing resolution: keeps draw cost comparable across devices
  var RES = 128;             // density field resolution (matches the MPM grid, upscaled smoothly)

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }

  var MARKUP =
    '<div class="stage">' +
    '<canvas></canvas>' +
    '<div class="hud">' +
    '<div class="chip big" data-k="speed">sim speed <b>--</b></div>' +
    '<div class="chip" data-k="fps">fps <b>--</b></div>' +
    '<div class="chip" data-k="np">particles <b>--</b></div>' +
    '<div class="chip" data-k="spf">substeps/frame <b>--</b></div>' +
    '<div class="chip" data-k="cells">active cells <b>--</b> / ' + (NG * NG) + '</div>' +
    '</div>' +
    '<div class="hint">drag to grab &middot; touch works</div>' +
    '</div>' +
    '<div class="warnbar"></div>' +
    '<div class="ctl">' +
    '<div class="grp"><div class="lbl">particles</div>' +
    '<div class="row"><input type="range" min="250" max="8000" step="250" data-k="np"><span class="val" data-k="npv"></span></div>' +
    '<div class="note">Drag it down until <b>sim speed</b> reads 1.0&times;. That number is the 60&nbsp;fps budget on this machine.</div>' +
    '</div>' +
    '<div class="grp"><div class="lbl">timestep &Delta;t (canonical = 1e-4)</div>' +
    '<div class="seg" data-k="dt">' +
    '<button data-v="1">&times;1</button><button data-v="2">&times;2</button><button data-v="3">&times;3</button><button data-v="4">&times;4</button>' +
    '</div>' +
    '<div class="note">Bigger &Delta;t means fewer substeps per frame, so it looks like free speed. It is not.</div>' +
    '</div>' +
    '<div class="grp"><div class="lbl">grid loop</div>' +
    '<div class="seg" data-k="dense">' +
    '<button data-v="0">sparse (ported)</button><button data-v="1">dense (Taichi)</button>' +
    '</div>' +
    '<div class="note">Same physics bit-for-bit. The dense sweep visits all ' + (NG * NG) + ' cells because a GPU does not care.</div>' +
    '</div>' +
    '<div class="grp"><div class="lbl">view / actions</div>' +
    '<div class="seg" data-k="view"><button data-v="blob">material</button><button data-v="grid">grid mass</button><button data-v="pts">particles</button></div>' +
    '<div class="row" style="margin-top:8px"><button class="act" data-k="reset">reset</button>' +
    '<button class="act" data-k="drop">drop again</button></div>' +
    '</div>' +
    '<div class="grp" style="min-width:230px"><div class="lbl">where the frame goes</div>' +
    '<div class="bars"><i data-k="bp2g"></i><i data-k="bgrid"></i><i data-k="bg2p"></i><i data-k="bdraw"></i><i data-k="bidle"></i></div>' +
    '<div class="lg">' +
    '<span><s style="background:#6fd3ee"></s>P2G <b data-k="tp2g">--</b></span>' +
    '<span><s style="background:#c58cf0"></s>grid <b data-k="tgrid">--</b></span>' +
    '<span><s style="background:#ff9d5c"></s>G2P <b data-k="tg2p">--</b></span>' +
    '<span><s style="background:#5fd39a"></s>draw <b data-k="tdraw">--</b></span>' +
    '</div>' +
    '<div class="note">Per phase, per frame. Measured by differencing whole loops, because a browser ' +
    'rounds its clock to 100&nbsp;&micro;s and a single 50&nbsp;&micro;s phase is below that.</div>' +
    '</div>' +
    '</div>';

  function mount(host, opts) {
    opts = opts || {};
    host.classList.add('mpmd');
    host.innerHTML = MARKUP;
    var q = function (sel) { return host.querySelector(sel); };
    var canvas = q('canvas');
    canvas.width = CANVAS; canvas.height = CANVAS;
    var ctx = canvas.getContext('2d');
    var off = document.createElement('canvas');
    off.width = RES; off.height = RES;
    var octx = off.getContext('2d');
    var img = octx.createImageData(RES, RES);
    var field = new Float32Array(RES * RES);
    var tmp = new Float32Array(RES * RES);

    var state = { n: opts.n || 1500, dtMult: 1, dense: 0, view: 'blob', diverged: false };
    var sim = null;
    var spf = 167;
    var split = { p2g_us: 0, grid_us: 0, g2p_us: 0, draw_ms: 0 };
    var splitTimer = 0;

    function rebuild() {
      var dt = P.dt * state.dtMult;
      spf = Math.max(1, Math.round((1 / 60) / dt));
      sim = M.createSim({ n: state.n, area: AREA, dt: dt, dense: !!state.dense });
      sim.seed(M.seedDisk(0.5, 0.62, 0.11, state.n, 20260816), 0, 0);
      state.diverged = false;
      q('.warnbar').classList.remove('show');
      clearTimeout(splitTimer);
      splitTimer = setTimeout(measureSplit, 260);       // debounced: slider drags rebuild often
    }

    // Phase costs, measured off to the side on a scratch simulation, because the live per-substep
    // route is not measurable in a browser (100 us clock granularity vs a 50 us phase).
    function measureSplit() {
      var r = benchPhases(state.n, 120, 3, !!state.dense);
      split.p2g_us = r.p2g_us; split.grid_us = r.grid_us; split.g2p_us = r.g2p_us;
      var d0 = performance.now();
      for (var i = 0; i < 20; i++) draw();
      split.draw_ms = (performance.now() - d0) / 20;
      readout();
    }

    // ------------------------------------------------------------------ rendering
    function splat() {
      field.fill(0);
      var x = sim.x, n = sim.n;
      for (var p = 0; p < n; p++) {
        var fx = x[2 * p] * RES, fy = (1 - x[2 * p + 1]) * RES;
        var bx = Math.floor(fx - 0.5), by = Math.floor(fy - 0.5);
        var ax = fx - bx, ay = fy - by;
        var wx0 = 0.5 * (1.5 - ax) * (1.5 - ax), wx1 = 0.75 - (ax - 1) * (ax - 1), wx2 = 0.5 * (ax - 0.5) * (ax - 0.5);
        var wy0 = 0.5 * (1.5 - ay) * (1.5 - ay), wy1 = 0.75 - (ay - 1) * (ay - 1), wy2 = 0.5 * (ay - 0.5) * (ay - 0.5);
        if (bx < 0 || by < 0 || bx > RES - 3 || by > RES - 3) continue;
        for (var i = 0; i < 3; i++) {
          var wxi = i === 0 ? wx0 : (i === 1 ? wx1 : wx2);
          var row = (by) * RES + bx + i;
          field[row] += wxi * wy0;
          field[row + RES] += wxi * wy1;
          field[row + 2 * RES] += wxi * wy2;
        }
      }
      // one separable [1 2 1]/4 pass: particle placement is random, so without it the interior
      // looks like gravel rather than a solid
      var r, c, i0;
      for (r = 0; r < RES; r++) {
        i0 = r * RES;
        for (c = 1; c < RES - 1; c++) tmp[i0 + c] = 0.25 * field[i0 + c - 1] + 0.5 * field[i0 + c] + 0.25 * field[i0 + c + 1];
        tmp[i0] = field[i0]; tmp[i0 + RES - 1] = field[i0 + RES - 1];
      }
      for (r = 1; r < RES - 1; r++) {
        i0 = r * RES;
        for (c = 0; c < RES; c++) field[i0 + c] = 0.25 * tmp[i0 + c - RES] + 0.5 * tmp[i0 + c] + 0.25 * tmp[i0 + c + RES];
      }
      for (c = 0; c < RES; c++) { field[c] = tmp[c]; field[(RES - 1) * RES + c] = tmp[(RES - 1) * RES + c]; }
    }

    function colorize() {
      // normalise so a solid interior sits near 1 regardless of particle count
      var diskArea = Math.PI * 0.11 * 0.11;
      var ref = sim.n / (diskArea * RES * RES);
      var inv = 1 / Math.max(ref, 1e-6);
      var d = img.data;
      for (var i = 0, k = 0; i < field.length; i++, k += 4) {
        var v = field[i] * inv;
        if (v < 0.12) { d[k + 3] = 0; continue; }
        var a = v < 0.42 ? (v - 0.12) / 0.30 : 1;
        var t = v < 0.30 ? 0 : Math.min(1, (v - 0.30) / 0.85);
        // rim -> core ramp on the canonical elastic colour #ff9d5c
        var r = 255 * (1 - t) + 196 * t;
        var g = 224 * (1 - t) + 108 * t;
        var b = 176 * (1 - t) + 62 * t;
        d[k] = r; d[k + 1] = g; d[k + 2] = b; d[k + 3] = 255 * a;
      }
      octx.putImageData(img, 0, 0);
    }

    function draw() {
      ctx.clearRect(0, 0, CANVAS, CANVAS);
      // boundaries from the canonical solver: floor + sticky side walls at `bound` cells
      var bpx = P.bound / NG * CANVAS;
      ctx.fillStyle = 'rgba(111,211,238,.05)';
      ctx.fillRect(0, CANVAS - bpx, CANVAS, bpx);
      ctx.fillRect(0, 0, bpx, CANVAS); ctx.fillRect(CANVAS - bpx, 0, bpx, CANVAS);
      ctx.strokeStyle = 'rgba(111,211,238,.30)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, CANVAS - bpx); ctx.lineTo(CANVAS, CANVAS - bpx); ctx.stroke();

      if (state.view === 'grid') {
        // a substep ends with the grid cleared, so re-scatter once purely to have something to show
        sim.phases.p2g();
        var cs = CANVAS / NG, gm = sim.grid.m, al = sim.activeList, na = sim.activeCells();
        var mmax = 1e-12;
        for (var a = 0; a < na; a++) { var m = gm[al[a]]; if (m > mmax) mmax = m; }
        for (var a2 = 0; a2 < na; a2++) {
          var idx = al[a2], gi = (idx / NG) | 0, gj = idx - gi * NG;
          var f = Math.min(1, Math.pow(gm[idx] / mmax, 0.4));
          ctx.fillStyle = 'rgba(' + Math.round(70 + 185 * f) + ',' + Math.round(205 - 48 * f) + ',' +
            Math.round(245 - 153 * f) + ',' + (0.22 + 0.72 * f) + ')';
          ctx.fillRect(gi * cs, CANVAS - (gj + 1) * cs, cs - 0.4, cs - 0.4);
        }
        sim.phases.clear();
        ctx.fillStyle = 'rgba(223,230,238,.55)';
        ctx.font = '13px ui-monospace,Menlo,Consolas,monospace';
        ctx.fillText('grid node mass  (cyan = thin halo, orange = full)', 14, CANVAS - 46);
      } else if (state.view === 'pts') {
        ctx.fillStyle = '#ff9d5c';
        var xs = sim.x;
        for (var p = 0; p < sim.n; p++) {
          ctx.fillRect(xs[2 * p] * CANVAS - 1.4, (1 - xs[2 * p + 1]) * CANVAS - 1.4, 2.8, 2.8);
        }
      } else {
        splat(); colorize();
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(off, 0, 0, CANVAS, CANVAS);
      }

      if (pointer.down) {
        var px = pointer.x * CANVAS, py = (1 - pointer.y) * CANVAS, rr = 0.09 * CANVAS;
        var g = ctx.createRadialGradient(px, py, 0, px, py, rr);
        g.addColorStop(0, 'rgba(111,211,238,.20)'); g.addColorStop(1, 'rgba(111,211,238,0)');
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, py, rr, 0, 6.2832); ctx.fill();
        ctx.strokeStyle = 'rgba(111,211,238,.55)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(px, py, rr * 0.55, 0, 6.2832); ctx.stroke();
      }
    }

    // ------------------------------------------------------------------ pointer (mouse + touch)
    var pointer = { down: false, x: 0.5, y: 0.5, px: 0.5, py: 0.5, t: 0 };
    function toSim(ev) {
      var r = canvas.getBoundingClientRect();
      return { x: (ev.clientX - r.left) / r.width, y: 1 - (ev.clientY - r.top) / r.height };
    }
    canvas.addEventListener('pointerdown', function (ev) {
      var s = toSim(ev);
      pointer.down = true; pointer.x = pointer.px = s.x; pointer.y = pointer.py = s.y;
      pointer.t = performance.now();
      sim.poke.on = true; sim.poke.x = s.x; sim.poke.y = s.y; sim.poke.vx = 0; sim.poke.vy = 0;
      if (canvas.setPointerCapture) { try { canvas.setPointerCapture(ev.pointerId); } catch (e) {} }
      ev.preventDefault();
    });
    canvas.addEventListener('pointermove', function (ev) {
      if (!pointer.down) return;
      var s = toSim(ev), now = performance.now();
      var dtms = Math.max(8, now - pointer.t);
      // smoothed pointer velocity in domain-lengths per second
      var vx = (s.x - pointer.x) / (dtms / 1000), vy = (s.y - pointer.y) / (dtms / 1000);
      sim.poke.vx = 0.6 * sim.poke.vx + 0.4 * Math.max(-6, Math.min(6, vx));
      sim.poke.vy = 0.6 * sim.poke.vy + 0.4 * Math.max(-6, Math.min(6, vy));
      pointer.x = s.x; pointer.y = s.y; pointer.t = now;
      sim.poke.x = s.x; sim.poke.y = s.y;
      ev.preventDefault();
    });
    function up(ev) {
      pointer.down = false; sim.poke.on = false;
      if (ev && ev.preventDefault) ev.preventDefault();
    }
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('pointerleave', up);

    // ------------------------------------------------------------------ loop
    var raf = 0, lastBeat = performance.now(), frames = 0, tWin = performance.now();
    var fps = 0, simMs = 0, checkCounter = 0;

    function tick() {
      var t0 = performance.now();
      lastBeat = t0;
      // A hidden tab gets throttled rAF. Stepping anyway would burn a phone battery and would make
      // the sim-speed readout report the throttling rather than the solver.
      if (document.hidden) { tWin = t0; frames = 0; raf = requestAnimationFrame(tick); return; }
      if (!state.diverged) {
        var s0 = performance.now();
        sim.substeps(spf);
        simMs = 0.85 * simMs + 0.15 * (performance.now() - s0);
        if (++checkCounter % 20 === 0 && !sim.finite()) {
          state.diverged = true;
          var w = q('.warnbar');
          w.innerHTML = '<b>The solver diverged.</b> At &Delta;t = ' + (P.dt * state.dtMult).toExponential(0) +
            ' the elastic wave crosses more than one grid cell per step, so MLS-MPM goes unstable. ' +
            'This is the CFL limit of the material, not a bug in the port. Press reset.';
          w.classList.add('show');
        }
      }
      draw();
      frames++;
      var now = performance.now();
      if (now - tWin > 400) {
        fps = frames * 1000 / (now - tWin); frames = 0; tWin = now;
        readout();
      }
      raf = requestAnimationFrame(tick);
    }

    // A frame that never arrives must not be able to wedge this permanently.
    var watchdog = setInterval(function () {
      if (document.hidden) return;
      if (performance.now() - lastBeat > 600) {
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(tick);
      }
    }, 800);

    function fmt(v, d) { return v.toFixed(d === undefined ? 1 : d); }

    function readout() {
      var speed = fps * spf * (P.dt * state.dtMult);
      var cls = speed >= 0.95 ? 'ok' : (speed >= 0.6 ? 'warn' : 'bad');
      q('[data-k=speed] b').innerHTML = state.diverged
        ? '<span class="bad">diverged</span>'
        : '<span class="' + cls + '">' + fmt(speed, 2) + '&times; real time</span>';
      q('[data-k=fps] b').textContent = fmt(fps, 0);
      q('[data-k=np] b').textContent = sim.n;
      q('[data-k=spf] b').textContent = spf;
      q('[data-k=cells] b').textContent = sim.activeCells();

      var p2gMs = split.p2g_us * spf / 1000, gridMs = split.grid_us * spf / 1000,
        g2pMs = split.g2p_us * spf / 1000, drawMs = split.draw_ms;
      var tot = Math.max(1e-6, p2gMs + gridMs + g2pMs + drawMs);
      var frameMs = fps > 0 ? 1000 / fps : tot;
      var idle = Math.max(0, frameMs - tot);
      var total = tot + idle;
      var seg = [['bp2g', p2gMs, '#6fd3ee'], ['bgrid', gridMs, '#c58cf0'], ['bg2p', g2pMs, '#ff9d5c'],
      ['bdraw', drawMs, '#5fd39a'], ['bidle', idle, '#1c2430']];
      for (var i = 0; i < seg.length; i++) {
        var e = q('[data-k=' + seg[i][0] + ']');
        e.style.width = (100 * seg[i][1] / total) + '%';
        e.style.background = seg[i][2];
      }
      q('[data-k=tp2g]').textContent = fmt(p2gMs, 1) + 'ms';
      q('[data-k=tgrid]').textContent = fmt(gridMs, 1) + 'ms';
      q('[data-k=tg2p]').textContent = fmt(g2pMs, 1) + 'ms';
      q('[data-k=tdraw]').textContent = fmt(drawMs, 2) + 'ms';
    }

    // ------------------------------------------------------------------ controls
    function segSet(key, val) {
      var btns = host.querySelectorAll('.seg[data-k=' + key + '] button');
      for (var i = 0; i < btns.length; i++) {
        var on = btns[i].getAttribute('data-v') === String(val);
        btns[i].classList.toggle('on', on);
        btns[i].classList.toggle('warnstate', on && key === 'dt' && Number(val) >= 3);
      }
    }
    host.addEventListener('click', function (ev) {
      var b = ev.target.closest ? ev.target.closest('button') : null;
      if (!b) return;
      var seg = b.parentElement.classList.contains('seg') ? b.parentElement.getAttribute('data-k') : null;
      if (seg === 'dt') { state.dtMult = Number(b.getAttribute('data-v')); segSet('dt', state.dtMult); rebuild(); }
      else if (seg === 'dense') { state.dense = Number(b.getAttribute('data-v')); segSet('dense', state.dense); rebuild(); }
      else if (seg === 'view') { state.view = b.getAttribute('data-v'); segSet('view', state.view); }
      else if (b.getAttribute('data-k') === 'reset') { rebuild(); }
      else if (b.getAttribute('data-k') === 'drop') {
        sim.seed(M.seedDisk(0.5, 0.62, 0.11, state.n, (Math.random() * 1e9) | 0), 0, 0);
        state.diverged = false; q('.warnbar').classList.remove('show');
      }
    });
    var slider = q('input[data-k=np]');
    slider.value = state.n;
    q('[data-k=npv]').textContent = state.n;
    slider.addEventListener('input', function () {
      state.n = Number(slider.value);
      q('[data-k=npv]').textContent = state.n;
      rebuild();
    });

    segSet('dt', 1); segSet('dense', 0); segSet('view', 'blob');
    rebuild();
    readout();
    raf = requestAnimationFrame(tick);

    return {
      state: state, get sim() { return sim; },
      stop: function () { cancelAnimationFrame(raf); clearInterval(watchdog); },
      setN: function (n) { state.n = n; slider.value = n; q('[data-k=npv]').textContent = n; rebuild(); },
      fps: function () { return fps; }
    };
  }

  // ---------------------------------------------------------------------- benchmarks
  // Pure measurement, no rendering: microseconds per canonical substep as a function of particle
  // count, with the phase split, for both the sparse and the dense grid loop.
  var AREA = Math.PI * 0.11 * 0.11;

  function fresh(n, dense, warm) {
    var s = M.createSim({ n: n, area: AREA, dt: P.dt, dense: !!dense });
    s.seed(M.seedDisk(0.5, 0.52, 0.11, n, 12345), 0, 0);
    s.substeps(warm === undefined ? 250 : warm);      // let the JIT settle and the blob land
    return s;
  }
  // min over repetitions: robust to the OS stealing the core mid-measurement
  function timeMin(fn, K, reps) {
    var best = Infinity;
    for (var r = 0; r < (reps || 5); r++) {
      var t0 = performance.now();
      fn(K);
      var ms = performance.now() - t0;
      if (ms < best) best = ms;
    }
    return best * 1000 / K;                            // microseconds per iteration
  }

  function benchCPU(cfg) {
    cfg = cfg || {};
    var counts = cfg.counts || [500, 1000, 1500, 2000, 3000, 4000, 6000, 8000];
    var K = cfg.steps || 300, reps = cfg.reps || 5;
    var rows = [];
    for (var c = 0; c < counts.length; c++) {
      var n = counts[c], row = { n: n };
      ['sparse', 'dense'].forEach(function (mode) {
        var s = fresh(n, mode === 'dense');
        var us = timeMin(function (k) { s.substeps(k); }, K, reps);
        row[mode] = {
          us_per_step: us, active_cells: s.activeCells(),
          realtime_factor: 1 / (us * 1e-6 / P.dt),
          substeps_per_frame_at_60: Math.round((1 / 60) / P.dt),
          frame_ms_at_60fps_realtime: us * Math.round((1 / 60) / P.dt) / 1000
        };
      });
      rows.push(row);
    }
    return rows;
  }

  // Price one phase by differencing whole loops. Timing a 50 us phase directly is impossible in a
  // browser (performance.now() is clamped to ~100 us unless the page is cross-origin isolated), but
  // a 50 ms loop is measured accurately, so each phase is (loop with it) - (loop without it).
  function benchPhases(n, K, reps, dense) {
    K = K || 300; reps = reps || 7;
    var s = fresh(n, !!dense);
    var ph = s.phases;
    s.phases.p2g();                                     // populate an active-cell list for clear()
    var tClear = timeMin(function (k) { for (var i = 0; i < k; i++) ph.clear(); }, K, reps);
    var tP = timeMin(function (k) { for (var i = 0; i < k; i++) { ph.p2g(); ph.clear(); } }, K, reps);
    var tPG = timeMin(function (k) { for (var i = 0; i < k; i++) { ph.p2g(); ph.grid(); ph.clear(); } }, K, reps);
    var tAll = timeMin(function (k) { s.substeps(k); }, K, reps);
    return {
      n: n, active_cells: s.activeCells(),
      clear_us: tClear, p2g_us: tP - tClear, grid_us: tPG - tP, g2p_us: tAll - tPG,
      step_us: tAll
    };
  }

  // Cost of replacing the analytic grid update with a small MLP evaluated per active cell.
  function benchNet(nCells, hidden, K) {
    var IN = 8, H = hidden || 64, OUT = 2;
    function rnd(k) { var a = new Float32Array(k); for (var i = 0; i < k; i++) a[i] = Math.random() - 0.5; return a; }
    var W1 = rnd(IN * H), b1 = rnd(H), W2 = rnd(H * H), b2 = rnd(H), W3 = rnd(H * OUT), b3 = rnd(OUT);
    var inp = rnd(nCells * IN), out = new Float32Array(nCells * OUT);
    var h1 = new Float32Array(H), h2 = new Float32Array(H);
    function pass() {
      for (var c = 0; c < nCells; c++) {
        var o = c * IN, j, i, s;
        for (j = 0; j < H; j++) {
          s = b1[j];
          for (i = 0; i < IN; i++) s += inp[o + i] * W1[i * H + j];
          h1[j] = s > 0 ? s : 0;
        }
        for (j = 0; j < H; j++) {
          s = b2[j];
          for (i = 0; i < H; i++) s += h1[i] * W2[i * H + j];
          h2[j] = s > 0 ? s : 0;
        }
        for (j = 0; j < OUT; j++) {
          s = b3[j];
          for (i = 0; i < H; i++) s += h2[i] * W3[i * OUT + j];
          out[c * OUT + j] = s;
        }
      }
    }
    pass();
    var t0 = performance.now();
    for (var k = 0; k < K; k++) pass();
    var ms = (performance.now() - t0) / K;
    var flops = nCells * 2 * (IN * H + H * H + H * OUT);
    return {
      cells: nCells, hidden: H, layers: IN + '-' + H + '-' + H + '-' + OUT,
      ms_per_grid_update: ms, flops_per_grid_update: flops,
      gflops_achieved: flops / (ms * 1e6)
    };
  }

  // How much does WebGPU charge just to LAUNCH the kernels a canonical frame needs? The substeps
  // are strictly sequential, so 167 of them means >=501 dispatches per frame no matter how fast
  // the arithmetic inside each one is.
  function benchGPU(dispatches) {
    dispatches = dispatches || 501;
    if (!navigator.gpu) return Promise.resolve({ available: false, reason: 'navigator.gpu missing' });
    return navigator.gpu.requestAdapter().then(function (ad) {
      if (!ad) return { available: false, reason: 'no adapter' };
      return ad.requestDevice().then(function (dev) {
        var NCELL = NG * NG;
        var buf = dev.createBuffer({ size: NCELL * 4, usage: GPUBufferUsage.STORAGE });
        var mod = dev.createShaderModule({
          code: '@group(0) @binding(0) var<storage, read_write> d: array<f32>;\n' +
            '@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n' +
            '  let i = g.x; if (i >= arrayLength(&d)) { return; }\n' +
            '  d[i] = d[i] * 0.999 + 0.001;\n}'
        });
        var pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
        var bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
        var wg = Math.ceil(NCELL / 64);
        function oneFrame() {
          var enc = dev.createCommandEncoder();
          for (var i = 0; i < dispatches; i++) {
            var pass = enc.beginComputePass();
            pass.setPipeline(pipe); pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(wg); pass.end();
          }
          dev.queue.submit([enc.finish()]);
          return dev.queue.onSubmittedWorkDone();
        }
        return oneFrame().then(function () {
          var t0 = performance.now(), reps = 10, chain = Promise.resolve();
          for (var r = 0; r < reps; r++) chain = chain.then(oneFrame);
          return chain.then(function () {
            var ms = (performance.now() - t0) / reps;
            return {
              available: true, adapter: (ad.info && ad.info.description) || 'unknown',
              dispatches_per_frame: dispatches, cells: NCELL,
              ms_per_frame: ms, us_per_dispatch: ms * 1000 / dispatches,
              max_fps: 1000 / ms
            };
          });
        });
      });
    }).catch(function (e) { return { available: false, reason: String(e) }; });
  }

  root.MPMDemo = {
    mount: mount, benchCPU: benchCPU, benchPhases: benchPhases,
    benchNet: benchNet, benchGPU: benchGPU, PARAMS: P
  };
})(typeof window !== 'undefined' ? window : this);
