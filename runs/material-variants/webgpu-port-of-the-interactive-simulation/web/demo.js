// demo.js -- the interactive WebGPU elastic sim UI.
//
// PORTABILITY: needs only params.js + mpm-webgpu.js and a <div id=...> to mount into. No CDN, no
// fetch, no harness. It runs inside a sandboxed `allow-scripts` iframe (the dashboard task page),
// as a standalone file, or on a portfolio page.
//
// NOTHING LOAD-BEARING DEPENDS ON requestAnimationFrame. rAF only decides WHEN to draw; the sim
// advances by exactly one frame's worth of substeps per drawn frame and every reported number comes
// from either the GPU timestamp query or a wall-clock total over >= 30 frames. If rAF stops (tab
// hidden, browser throttling), the sim simply stops advancing and resumes cleanly -- no latched
// state, no accumulator that runs away, no measurement that silently becomes wrong.
(function (root) {
  'use strict';

  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt !== undefined) e.textContent = txt;
    return e;
  }

  root.MPMDemo = function mount(host, opts) {
    opts = opts || {};
    var M = root.MPMWebGPU;
    var P = M.PARAMS;

    host.innerHTML = '';
    var wrap = el('div', 'mpmd');
    var stage = el('div', 'mpmd-stage');
    var canvas = document.createElement('canvas');
    canvas.className = 'mpmd-canvas';
    stage.appendChild(canvas);
    var hud = el('div', 'mpmd-hud');
    stage.appendChild(hud);
    var badge = el('div', 'mpmd-badge', 'drag to poke the block');
    stage.appendChild(badge);
    wrap.appendChild(stage);

    var controls = el('div', 'mpmd-controls');
    wrap.appendChild(controls);
    var note = el('div', 'mpmd-note');
    wrap.appendChild(note);
    host.appendChild(wrap);

    if (!M.supported()) {
      stage.innerHTML = '<div class="mpmd-fail">This browser exposes no <code>navigator.gpu</code>.' +
        ' WebGPU is only visible in a <b>secure context</b> &mdash; https, or http://localhost.' +
        ' On a plain-HTTP LAN address the API is hidden even on hardware that fully supports it.</div>';
      return;
    }

    // ---------------------------------------------------------------- controls
    function group(label) {
      var g = el('div', 'mpmd-g');
      g.appendChild(el('label', 'mpmd-lab', label));
      controls.appendChild(g);
      return g;
    }
    function slider(label, min, max, step, val, fmt, onchange) {
      var g = group(label);
      var row = el('div', 'mpmd-row');
      var inp = document.createElement('input');
      inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = val;
      var read = el('span', 'mpmd-read', fmt(val));
      inp.addEventListener('input', function () {
        read.textContent = fmt(+inp.value); onchange(+inp.value);
      });
      row.appendChild(inp); row.appendChild(read); g.appendChild(row);
      return { input: inp, read: read, set: function (v) { inp.value = v; read.textContent = fmt(v); } };
    }
    function seg(label, items, initial, onchange) {
      var g = group(label);
      var row = el('div', 'mpmd-seg');
      var btns = items.map(function (it) {
        var b = el('button', 'mpmd-sb' + (it.value === initial ? ' on' : ''), it.label);
        b.type = 'button';
        if (it.title) b.title = it.title;
        b.addEventListener('click', function () {
          btns.forEach(function (x) { x.classList.remove('on'); });
          b.classList.add('on');
          onchange(it.value);
        });
        row.appendChild(b);
        return b;
      });
      g.appendChild(row);
      return btns;
    }

    var state = {
      n: opts.n || 8000,
      speed: 1.0,
      view: 'particles',
      overlay: true,
      kM: 24, kV: 22,
      atomics: 'fixed',
      running: true
    };

    var sim = null, renderer = null, building = false, generation = 0;

    function sceneSeed(n) {
      // a slab of rubber resting on the floor, wide enough to grab and deform
      var half = 0.20, h = 0.24, y0 = P.floor_y + 0.004;
      var s = 20250816 >>> 0;
      function rnd() { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }
      var pts = new Float32Array(2 * n);
      for (var i = 0; i < n; i++) {
        pts[2 * i] = 0.5 - half + rnd() * 2 * half;
        pts[2 * i + 1] = y0 + rnd() * h;
      }
      return { pts: pts, area: 2 * half * h };
    }

    async function build() {
      if (building) return;
      building = true;
      var myGen = ++generation;
      if (sim) { try { sim.destroy(); } catch (e) {} sim = null; }
      var sc = sceneSeed(state.n);
      try {
        sim = await M.createSim({ n: state.n, area: sc.area, atomics: state.atomics,
                                  kM: state.kM, kV: state.kV });
      } catch (e) {
        stage.innerHTML = '<div class="mpmd-fail">WebGPU init failed: ' + e.message + '</div>';
        building = false; return;
      }
      if (myGen !== generation) { building = false; return; }
      sim.seed(sc.pts, 0, 0);
      renderer = await M.createRenderer(canvas, sim);
      building = false;
      fpsWin.length = 0; lastGpuMs = null;
    }

    // ---------------------------------------------------------------- pointer interaction
    var pointer = { active: false, x: 0, y: 0, px: 0, py: 0, t: 0 };
    function toSim(ev) {
      var r = canvas.getBoundingClientRect();
      return { x: (ev.clientX - r.left) / r.width, y: 1.0 - (ev.clientY - r.top) / r.height };
    }
    stage.addEventListener('pointerdown', function (ev) {
      var p = toSim(ev);
      pointer.active = true; pointer.x = p.x; pointer.y = p.y;
      pointer.px = p.x; pointer.py = p.y; pointer.t = performance.now();
      stage.setPointerCapture(ev.pointerId);
      badge.style.opacity = 0;
      ev.preventDefault();
    });
    stage.addEventListener('pointermove', function (ev) {
      if (!pointer.active) return;
      var p = toSim(ev);
      pointer.x = p.x; pointer.y = p.y;
      ev.preventDefault();
    });
    function release(ev) {
      pointer.active = false;
      if (sim) { sim.poke.on = false; sim.syncUniform(); }
      if (ev && ev.pointerId !== undefined && stage.hasPointerCapture &&
        stage.hasPointerCapture(ev.pointerId)) stage.releasePointerCapture(ev.pointerId);
    }
    stage.addEventListener('pointerup', release);
    stage.addEventListener('pointercancel', release);
    stage.addEventListener('pointerleave', release);
    stage.style.touchAction = 'none';

    // ---------------------------------------------------------------- HUD
    var rows = {};
    ['fps', 'gpu', 'substeps', 'particles', 'speed'].forEach(function (k) {
      var r = el('div', 'mpmd-hrow');
      r.appendChild(el('span', 'mpmd-hk', k));
      var v = el('span', 'mpmd-hv', '--');
      r.appendChild(v); hud.appendChild(r); rows[k] = v;
    });

    var fpsWin = [];
    var lastGpuMs = null;
    var frameCount = 0;
    var inflight = false;

    function resize() {
      var r = stage.getBoundingClientRect();
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = Math.max(64, Math.round(r.width * dpr));
      var h = Math.max(64, Math.round(r.height * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    }

    async function tick() {
      if (!sim || !renderer || inflight) return;
      resize();
      var spf = Math.max(1, Math.round(state.speed * (1 / 60) / P.dt));
      if (pointer.active) {
        var t = performance.now();
        var dtp = Math.max(1e-3, (t - pointer.t) / 1000);
        sim.poke.on = true;
        sim.poke.x = pointer.x; sim.poke.y = pointer.y;
        sim.poke.vx = (pointer.x - pointer.px) / dtp;
        sim.poke.vy = (pointer.y - pointer.py) / dtp;
        pointer.px = pointer.x; pointer.py = pointer.y; pointer.t = t;
        sim.syncUniform();
      }
      var timed = (frameCount % 30) === 0;
      var t0 = performance.now();
      inflight = true;
      sim.encodeFrame(spf, { timed: timed });
      renderer.draw({
        view: state.view, overlay: state.overlay,
        radius: Math.max(0.004, 0.030 / Math.sqrt(state.n / 900)),
        // reference values from the measured range: a loaded node carries ~2x the
        // particles-per-cell in particle masses, and node speeds run to a couple of m/s
        vRef: state.view === 'particles' ? 2.5 : 4.0,
        massRef: Math.max(4, 2.2 * state.n / (0.4 * 0.24 * P.n_grid * P.n_grid))
      });
      if (timed) { try { lastGpuMs = (await sim.lastGpuNanos()) / 1e6; } catch (e) { lastGpuMs = null; } }
      await sim.idle();
      inflight = false;
      var dtms = performance.now() - t0;
      fpsWin.push(dtms);
      if (fpsWin.length > 40) fpsWin.shift();
      frameCount++;

      if (fpsWin.length >= 30) {
        // averaged over >= 30 frames: performance.now()'s ~100 us clamp is 1000x below this total
        var mean = fpsWin.reduce(function (a, b) { return a + b; }, 0) / fpsWin.length;
        rows.fps.textContent = (1000 / mean).toFixed(0) + ' fps  (' + mean.toFixed(1) + ' ms)';
      }
      rows.gpu.textContent = lastGpuMs === null ? 'n/a' : lastGpuMs.toFixed(2) + ' ms compute';
      rows.substeps.textContent = spf + ' / frame  (dt = ' + P.dt.toExponential(1) + ' s)';
      rows.particles.textContent = state.n.toLocaleString();
      rows.speed.textContent = state.speed.toFixed(2) + ' x real time';
    }

    var raf = 0;
    function loop() {
      raf = requestAnimationFrame(loop);
      if (!state.running) return;
      tick();
    }

    // ---------------------------------------------------------------- build the controls
    slider('particles', 500, 40000, 500, state.n, function (v) { return v.toLocaleString(); },
      function (v) { state.n = v; build(); });
    slider('sim speed (x real time)', 0.1, 2, 0.05, 1.0, function (v) { return v.toFixed(2) + ' x'; },
      function (v) { state.speed = v; });
    seg('view', [
      { value: 'particles', label: 'particles', title: 'the material itself, shaded by speed' },
      { value: 'mass', label: 'grid mass', title: 'the background grid the solver actually solves on' },
      { value: 'speed', label: 'grid speed', title: 'the node velocity field after the grid update' }
    ], 'particles', function (v) { state.view = v; });
    seg('atomic accumulator', [
      { value: 'f24', label: 'fixed 2^24', title: 'fixed point, 24 bits per particle mass -- matches exact f32 to within the simulator\u2019s own noise' },
      { value: 'f20', label: 'fixed 2^20', title: 'fixed point, 20 bits per particle mass -- coarser; measurably off on contact-heavy scenes' },
      { value: 'cas', label: 'exact f32', title: 'compare-and-swap loop: exact f32 addition, but slower under contention' }
    ], 'f24', function (v) {
      if (v === 'cas') { state.atomics = 'casf32'; }
      else { state.atomics = 'fixed'; state.kM = v === 'f24' ? 24 : 20; state.kV = v === 'f24' ? 22 : 18; }
      build();
    });
    var g = group('');
    var btnRow = el('div', 'mpmd-seg');
    var bReset = el('button', 'mpmd-sb', 'reset');
    bReset.type = 'button';
    bReset.addEventListener('click', function () { build(); });
    var bPause = el('button', 'mpmd-sb', 'pause');
    bPause.type = 'button';
    bPause.addEventListener('click', function () {
      state.running = !state.running;
      bPause.textContent = state.running ? 'pause' : 'resume';
    });
    btnRow.appendChild(bReset); btnRow.appendChild(bPause);
    g.appendChild(btnRow);

    note.innerHTML = 'One command buffer per frame: <b>' +
      (3 * Math.round((1 / 60) / P.dt)) + ' compute dispatches</b> recorded and submitted once. ' +
      'Physics parameters generated from <code>sim.physics</code> (<code>' + P.physics_version +
      '</code>), elastic <code>E=' + P.E + '</code>, <code>dt=' + P.dt.toExponential(0) +
      '</code> on a ' + P.n_grid + '\u00d7' + P.n_grid + ' grid.';

    build().then(function () { loop(); });
    window.addEventListener('resize', resize);
    return { state: state, stop: function () { cancelAnimationFrame(raf); } };
  };
})(typeof self !== 'undefined' ? self : this);
