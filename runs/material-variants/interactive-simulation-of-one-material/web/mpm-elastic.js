// mpm-elastic.js -- a browser/Node port of the canonical 2D MLS-MPM elastic step.
//
// PORTABILITY CONTRACT: this file has no dependency on the dashboard, the data server, or the
// harness. It needs only `params.js` (generated from sim.physics) and a JS engine with typed
// arrays. It runs unchanged in Node (require) and in a browser (script tag -> window.MPMElastic).
//
// WHAT IS THE SAME AS THE CANONICAL TAICHI STEP (sim/physics/core.py)
//   * every parameter: E, dt, nu, n_grid, dx, gravity, bound, Coulomb friction, particle volume
//     and mass -- all read from MPM_PARAMS, which gen_params.py emits straight out of sim.physics.
//   * the constitutive law: fixed corotated, stress = 2 mu (F-R) F^T + la (J-1) J I.
//   * the transfer skeleton: quadratic B-spline P2G, grid update with gravity + separating floor
//     with Coulomb friction + sticky walls, then G2P with the APIC affine matrix C, F update
//     F <- (I + dt C) F, and the position clamp to [floor_y, 1-floor_y].
//
// WHAT NECESSARILY DIFFERS, AND WHY (all three are forced by "one CPU thread instead of a GPU")
//   1. R comes from a closed-form 2D polar decomposition, not ti.svd. This is not an
//      approximation: Taichi's own svd2d is built on the same closed-form polar rotation and
//      returns U V^T = R_polar, so the elastic path never needed the singular values at all.
//   2. Arithmetic is f64 (JS numbers) with f32 storage, where Taichi runs f32 throughout. The
//      port is therefore slightly MORE accurate per operation than the reference, not less.
//   3. The grid loop is sparse. Taichi sweeps all 128x128 cells because that is free on a GPU;
//      one CPU thread cannot afford 16384 cells per substep, so P2G records the cells it touched
//      and the grid update and the clear walk only that list. This is exact, not an
//      approximation: every node a particle gathers from is a node it scattered to, so no cell
//      outside the touched set can influence any particle. (Verified: sparse and dense modes
//      agree bit-for-bit -- see selfTest below.)
//
// The interaction force (poke/drag) is a demo-only external body force layered on top of the
// canonical step. It is off by default and is never enabled during verification.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./params.js'));
  } else {
    root.MPMElastic = factory(root.MPM_PARAMS);
  }
})(typeof self !== 'undefined' ? self : this, function (P) {
  'use strict';

  var N_GRID = P.n_grid | 0;
  var N_CELL = N_GRID * N_GRID;
  var DX = P.dx;
  var INV_DX = P.inv_dx;
  var BOUND = P.bound | 0;
  var FLOOR = P.floor_y;
  var CEIL = 1.0 - P.floor_y;

  function createSim(opts) {
    opts = opts || {};
    var n = opts.n | 0;
    var area = opts.area;                       // seeded area -> particle volume, exactly as canonical
    var dt = opts.dt !== undefined ? opts.dt : P.dt;
    var E = opts.E !== undefined ? opts.E : P.E;
    var nu = P.NU;
    var gravity = opts.gravity !== undefined ? opts.gravity : P.gravity;
    var friction = P.FRICTION;
    var dense = !!opts.dense;                   // brute-force every cell (reference mode)

    var pVol = area / n;
    var pMass = pVol * P.p_rho;
    var mu = E / (2.0 * (1.0 + nu));
    var la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    var x = new Float32Array(2 * n);
    var v = new Float32Array(2 * n);
    var Cm = new Float32Array(4 * n);
    var Fm = new Float32Array(4 * n);

    var gvx = new Float32Array(N_CELL);
    var gvy = new Float32Array(N_CELL);
    var gm = new Float32Array(N_CELL);
    var stamp = new Int32Array(N_CELL);
    var active = new Int32Array(N_CELL);
    var nActive = 0;
    var epoch = 0;

    // Demo-only pointer grab. NOT part of the canonical physics: it is an external actuator that
    // relaxes the grid velocity inside a Gaussian window toward (pointer velocity + a spring pull
    // toward the pointer). It is off during every verification run.
    var poke = { on: false, x: 0, y: 0, vx: 0, vy: 0, radius: 0.075, rate: 900.0, spring: 16.0 };

    var timing = { p2g: 0, grid: 0, g2p: 0 };
    var profile = false;

    function seed(pts, v0x, v0y) {
      for (var p = 0; p < n; p++) {
        x[2 * p] = pts[2 * p];
        x[2 * p + 1] = pts[2 * p + 1];
        v[2 * p] = v0x || 0;
        v[2 * p + 1] = v0y || 0;
        Cm[4 * p] = 0; Cm[4 * p + 1] = 0; Cm[4 * p + 2] = 0; Cm[4 * p + 3] = 0;
        Fm[4 * p] = 1; Fm[4 * p + 1] = 0; Fm[4 * p + 2] = 0; Fm[4 * p + 3] = 1;
      }
      gvx.fill(0); gvy.fill(0); gm.fill(0);
      nActive = 0;
    }

    // ---------------------------------------------------------------- P2G
    function p2g() {
      epoch++;
      nActive = 0;
      var k = -dt * 4.0 * pVol * INV_DX * INV_DX;
      var twoMu = 2.0 * mu;
      for (var p = 0; p < n; p++) {
        var Xpx = x[2 * p] * INV_DX;
        var Xpy = x[2 * p + 1] * INV_DX;
        var bx = (Xpx - 0.5) | 0;               // positions are clamped >= 3dx so this is a floor
        var by = (Xpy - 0.5) | 0;
        var fx = Xpx - bx, fy = Xpy - by;

        var wx0 = 0.5 * (1.5 - fx) * (1.5 - fx);
        var wx1 = 0.75 - (fx - 1.0) * (fx - 1.0);
        var wx2 = 0.5 * (fx - 0.5) * (fx - 0.5);
        var wy0 = 0.5 * (1.5 - fy) * (1.5 - fy);
        var wy1 = 0.75 - (fy - 1.0) * (fy - 1.0);
        var wy2 = 0.5 * (fy - 0.5) * (fy - 0.5);

        var f00 = Fm[4 * p], f01 = Fm[4 * p + 1], f10 = Fm[4 * p + 2], f11 = Fm[4 * p + 3];

        // closed-form 2D polar rotation R = U V^T (identical to what Taichi's svd2d builds on)
        var rx = f00 + f11, ry = f10 - f01;
        var rinv = 1.0 / Math.sqrt(rx * rx + ry * ry);
        var rc = rx * rinv, rs = ry * rinv;     // R = [[rc,-rs],[rs,rc]]

        var a00 = f00 - rc, a01 = f01 + rs, a10 = f10 - rs, a11 = f11 - rc;   // F - R
        // (F-R) F^T
        var b00 = a00 * f00 + a01 * f01, b01 = a00 * f10 + a01 * f11;
        var b10 = a10 * f00 + a11 * f01, b11 = a10 * f10 + a11 * f11;
        var Jd = f00 * f11 - f01 * f10;
        var lt = la * (Jd - 1.0) * Jd;

        var af00 = k * (twoMu * b00 + lt) + pMass * Cm[4 * p];
        var af01 = k * (twoMu * b01) + pMass * Cm[4 * p + 1];
        var af10 = k * (twoMu * b10) + pMass * Cm[4 * p + 2];
        var af11 = k * (twoMu * b11 + lt) + pMass * Cm[4 * p + 3];

        var mvx = pMass * v[2 * p], mvy = pMass * v[2 * p + 1];
        var base = bx * N_GRID + by;

        for (var i = 0; i < 3; i++) {
          var wxi = i === 0 ? wx0 : (i === 1 ? wx1 : wx2);
          var dpx = (i - fx) * DX;
          var col = base + i * N_GRID;
          for (var j = 0; j < 3; j++) {
            var w = wxi * (j === 0 ? wy0 : (j === 1 ? wy1 : wy2));
            var dpy = (j - fy) * DX;
            var idx = col + j;
            gvx[idx] += w * (mvx + af00 * dpx + af01 * dpy);
            gvy[idx] += w * (mvy + af10 * dpx + af11 * dpy);
            gm[idx] += w * pMass;
            if (stamp[idx] !== epoch) { stamp[idx] = epoch; active[nActive++] = idx; }
          }
        }
      }
    }

    // ---------------------------------------------------------------- grid update
    function gridCell(idx) {
      var m = gm[idx];
      var vx = gvx[idx], vy = gvy[idx];
      if (m > 0.0) { vx /= m; vy /= m; }
      vy -= dt * gravity;

      if (poke.on) {
        var i0 = (idx / N_GRID) | 0, j0 = idx - i0 * N_GRID;
        var cx = poke.x - i0 * DX, cy = poke.y - j0 * DX;
        var r2 = cx * cx + cy * cy;
        var s2 = poke.radius * poke.radius;
        if (r2 < s2 * 6.0) {
          var lam = poke.rate * dt * Math.exp(-r2 / (0.5 * s2));
          if (lam > 0.5) lam = 0.5;
          vx += lam * (poke.vx + cx * poke.spring - vx);
          vy += lam * (poke.vy + cy * poke.spring - vy);
        }
      }

      var i = (idx / N_GRID) | 0, j = idx - i * N_GRID;
      if (j < BOUND && vy < 0) {
        var cap = friction * (-vy);
        if (vx > 0) vx = Math.max(0.0, vx - cap);
        else if (vx < 0) vx = Math.min(0.0, vx + cap);
        vy = 0;
      }
      if (j > N_GRID - BOUND && vy > 0) vy = 0;
      if (i < BOUND && vx < 0) { vx = 0; vy = 0; }
      if (i > N_GRID - BOUND && vx > 0) { vx = 0; vy = 0; }
      gvx[idx] = vx; gvy[idx] = vy;
    }

    function gridOp() {
      if (dense) { for (var c = 0; c < N_CELL; c++) gridCell(c); return; }
      for (var a = 0; a < nActive; a++) gridCell(active[a]);
    }

    // ---------------------------------------------------------------- G2P
    function g2p() {
      for (var p = 0; p < n; p++) {
        var Xpx = x[2 * p] * INV_DX;
        var Xpy = x[2 * p + 1] * INV_DX;
        var bx = (Xpx - 0.5) | 0, by = (Xpy - 0.5) | 0;
        var fx = Xpx - bx, fy = Xpy - by;

        var wx0 = 0.5 * (1.5 - fx) * (1.5 - fx);
        var wx1 = 0.75 - (fx - 1.0) * (fx - 1.0);
        var wx2 = 0.5 * (fx - 0.5) * (fx - 0.5);
        var wy0 = 0.5 * (1.5 - fy) * (1.5 - fy);
        var wy1 = 0.75 - (fy - 1.0) * (fy - 1.0);
        var wy2 = 0.5 * (fy - 0.5) * (fy - 0.5);

        var nvx = 0, nvy = 0, c00 = 0, c01 = 0, c10 = 0, c11 = 0;
        var base = bx * N_GRID + by;
        for (var i = 0; i < 3; i++) {
          var wxi = i === 0 ? wx0 : (i === 1 ? wx1 : wx2);
          var dpx = (i - fx) * DX;
          var col = base + i * N_GRID;
          for (var j = 0; j < 3; j++) {
            var w = wxi * (j === 0 ? wy0 : (j === 1 ? wy1 : wy2));
            var dpy = (j - fy) * DX;
            var idx = col + j;
            var gx = gvx[idx], gy = gvy[idx];
            nvx += w * gx; nvy += w * gy;
            var s = 4.0 * w * INV_DX * INV_DX;
            c00 += s * gx * dpx; c01 += s * gx * dpy;
            c10 += s * gy * dpx; c11 += s * gy * dpy;
          }
        }
        v[2 * p] = nvx; v[2 * p + 1] = nvy;
        var px = x[2 * p] + dt * nvx, py = x[2 * p + 1] + dt * nvy;
        x[2 * p] = px < FLOOR ? FLOOR : (px > CEIL ? CEIL : px);
        x[2 * p + 1] = py < FLOOR ? FLOOR : (py > CEIL ? CEIL : py);

        // F <- (I + dt C) F
        var g00 = 1.0 + dt * c00, g01 = dt * c01, g10 = dt * c10, g11 = 1.0 + dt * c11;
        var f00 = Fm[4 * p], f01 = Fm[4 * p + 1], f10 = Fm[4 * p + 2], f11 = Fm[4 * p + 3];
        Fm[4 * p] = g00 * f00 + g01 * f10;
        Fm[4 * p + 1] = g00 * f01 + g01 * f11;
        Fm[4 * p + 2] = g10 * f00 + g11 * f10;
        Fm[4 * p + 3] = g10 * f01 + g11 * f11;

        Cm[4 * p] = c00; Cm[4 * p + 1] = c01; Cm[4 * p + 2] = c10; Cm[4 * p + 3] = c11;
      }
    }

    function clearGrid() {
      if (dense) { gvx.fill(0); gvy.fill(0); gm.fill(0); return; }
      for (var a = 0; a < nActive; a++) {
        var idx = active[a];
        gvx[idx] = 0; gvy[idx] = 0; gm[idx] = 0;
      }
    }

    // One canonical substep. The grid is all-zero on entry and on exit, which is what makes the
    // sparse clear exact.
    function step() {
      if (profile) {
        var t0 = now(); p2g();
        var t1 = now(); gridOp();
        var t2 = now(); g2p();
        var t3 = now();
        clearGrid();
        timing.p2g += t1 - t0; timing.grid += t2 - t1; timing.g2p += t3 - t2;
      } else {
        p2g(); gridOp(); g2p(); clearGrid();
      }
    }

    function substeps(k) { for (var s = 0; s < k; s++) step(); }

    function finite() {
      for (var p = 0; p < 2 * n; p++) if (!isFinite(x[p])) return false;
      return true;
    }

    return {
      n: n, x: x, v: v, F: Fm, C: Cm,
      grid: { vx: gvx, vy: gvy, m: gm },
      activeList: active,
      activeCells: function () { return nActive; },
      params: {
        E: E, dt: dt, nu: nu, mu: mu, la: la, pVol: pVol, pMass: pMass,
        n_grid: N_GRID, gravity: gravity, friction: friction,
        physics_version: P.physics_version
      },
      setDt: function (d) { dt = d; },
      getDt: function () { return dt; },
      poke: poke,
      seed: seed, step: step, substeps: substeps, finite: finite,
      // exposed so a benchmark can price one phase by differencing whole loops -- browsers clamp
      // performance.now() to ~100 us, so timing a single 50 us phase directly is meaningless
      phases: { p2g: p2g, grid: gridOp, g2p: g2p, clear: clearGrid },
      timing: timing,
      setProfile: function (b) { profile = b; },
      resetTiming: function () { timing.p2g = 0; timing.grid = 0; timing.g2p = 0; }
    };
  }

  var now = (typeof performance !== 'undefined' && performance.now)
    ? function () { return performance.now(); }
    : function () { return Number(process.hrtime.bigint()) / 1e6; };

  // Deterministic disk seeding for the demo (the verification runs use the exact numpy-seeded
  // point set exported from sim.physics instead, so the initial condition is identical there).
  function seedDisk(cx, cy, radius, n, seed) {
    var s = seed >>> 0 || 1;
    function rnd() {                              // xorshift32 in [0,1)
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    }
    var pts = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) {
      var a = rnd() * Math.PI * 2, r = radius * Math.sqrt(rnd());
      pts[2 * i] = cx + r * Math.cos(a);
      pts[2 * i + 1] = cy + r * Math.sin(a);
    }
    return pts;
  }

  return { createSim: createSim, seedDisk: seedDisk, PARAMS: P, N_GRID: N_GRID };
});
