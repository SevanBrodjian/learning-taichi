// mpm-webgpu.js -- a WebGPU compute port of the canonical 2D MLS-MPM elastic step.
//
// PORTABILITY CONTRACT: no dependency on the dashboard, the data server, or the harness. It needs
// only `params.js` (generated from sim.physics by gen_params.py) and a WebGPU-capable browser.
// Load it with a <script> tag -> window.MPMWebGPU.
//
// ---------------------------------------------------------------------------------------------
// THE ONE IDEA
// ---------------------------------------------------------------------------------------------
// An explicit MPM solver costs substeps/frame x cost-per-substep, and substeps = (1/60)/dt is
// pinned by CFL: 167 for elastic. The canonical Taichi/CUDA path pays a ~56 us kernel launch per
// kernel per substep from Python, so a substep costs ~345 us and is FLAT in particle count -- the
// GPU is idle, waiting to be told what to do. The fix is not a faster kernel, it is to stop paying
// the launch: record every dispatch of every substep into ONE command buffer inside ONE compute
// pass and submit once per frame. WebGPU orders dispatches within a pass and makes each one's
// writes visible to the next, which is exactly the P2G -> grid -> G2P dependency chain, so this
// needs no explicit barriers and no CPU round trip.
//
// ---------------------------------------------------------------------------------------------
// WHAT WEBGPU FORCED, AND WHY
// ---------------------------------------------------------------------------------------------
//   1. NO ATOMIC FLOAT ADD. WGSL's atomics are u32/i32 only (`atomic<f32>` fails to compile:
//      "'atomic' only supports 'i32', 'u32' or 'vec2u'"). P2G is a scatter that atomically
//      accumulates mass and momentum as floats. Two routes are implemented and both are measured:
//        'fixed'  -- FIXED POINT. mass -> atomic<u32>, momentum -> atomic<i32> (signed: momentum
//                    goes negative). The scale is expressed in units of one particle mass, so it
//                    is scene-independent: an integer of 2^kM means "one particle's mass at this
//                    node". Mass contributions are w*2^kM with sum(w)=1 over the 3x3 stencil, so
//                    a node holding k particles holds k*2^kM and u32 saturates at 2^(32-kM)
//                    particles on one node. Conversion uses round(), not truncation: truncating a
//                    signed momentum biases it toward zero, i.e. a systematic numerical drag.
//        'casf32' -- EXACT f32 ADD via a compare-and-swap loop on the same u32 storage
//                    (bitcast -> add -> atomicCompareExchangeWeak, retry on failure). No
//                    quantisation at all; the price is retries under contention.
//   2. NO ti.svd. R comes from the closed-form 2D polar rotation, exactly as in the JS port --
//      not an approximation, since Taichi's own svd2d is built on the same rotation and the
//      elastic path never uses the singular values.
//   3. DENSE GRID SWEEP. The JS port walks a sparse active-cell list because one CPU thread cannot
//      afford 16384 cells per substep. On a GPU the dense sweep is free and a sparse list would
//      need a compaction pass, so this port sweeps every cell, like canonical Taichi does.
//   4. CLEAR IS FUSED INTO grid_op. grid_op is the only reader of the atomic accumulators, so it
//      loads them, computes the node velocity into a plain f32 buffer, and stores zeros back in
//      the same invocation. That removes a whole 16384-cell dispatch per substep (4 dispatches
//      per substep -> 3) for free.
//   5. ONLY 8 STORAGE BUFFERS PER STAGE are guaranteed by the WebGPU baseline limits. The obvious
//      layout (pos, vel, C, F, mass, momX, momY, gridV, display) is NINE and silently produces an
//      invalid bind group -- every dispatch is then dropped and the simulation runs at the speed
//      of doing nothing, which looks exactly like a spectacular result. So momentum X/Y share one
//      interleaved buffer and the grid-velocity buffer doubles as the display buffer: 7 in total.
//
// The interaction force (poke/drag) is a demo-only external body force layered on the grid update.
// It is off by default and is never enabled during verification.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./params.js'));
  } else {
    root.MPMWebGPU = factory(root.MPM_PARAMS);
  }
})(typeof self !== 'undefined' ? self : this, function (P) {
  'use strict';

  var N_GRID = P.n_grid | 0;
  var N_CELL = N_GRID * N_GRID;
  var WG_P = 64;                 // particles per workgroup
  var WG_G = 64;                 // cells per workgroup
  var GRID_WG = Math.ceil(N_CELL / WG_G);

  // ------------------------------------------------------------------ WGSL source
  // Built as a string so the frozen constants come from params.js (generated from sim.physics)
  // rather than being typed. `mode` picks the atomic strategy.
  function buildShader(mode) {
    var casf = (mode === 'casf32');
    var atomT = casf ? 'atomic<u32>' : null;

    // One inlined CAS float-add. WGSL will not let a storage pointer cross a function boundary
    // without the unrestricted_pointer_parameters extension, so this is textually inlined.
    function addF(buf, idxExpr, valExpr) {
      return [
        '  {',
        '    var _old = atomicLoad(&' + buf + '[' + idxExpr + ']);',
        '    loop {',
        '      let _nv = bitcast<u32>(bitcast<f32>(_old) + (' + valExpr + '));',
        '      let _r = atomicCompareExchangeWeak(&' + buf + '[' + idxExpr + '], _old, _nv);',
        '      if (_r.exchanged) { break; }',
        '      _old = _r.old_value;',
        '    }',
        '  }'
      ].join('\n');
    }
    var scatterMass = casf ? addF('gm', 'gi', 'wm')
      : '  atomicAdd(&gm[gi], u32(round((wm) * PR.massScale)));';
    var scatterMX = casf ? addF('gp', '2 * gi', 'mvx')
      : '  atomicAdd(&gp[2 * gi], i32(round((mvx) * PR.momScale)));';
    var scatterMY = casf ? addF('gp', '2 * gi + 1', 'mvy')
      : '  atomicAdd(&gp[2 * gi + 1], i32(round((mvy) * PR.momScale)));';

    var loadMass = casf ? 'bitcast<f32>(atomicLoad(&gm[idx]))'
                        : 'f32(atomicLoad(&gm[idx])) * PR.invMassScale';
    var loadMX = casf ? 'bitcast<f32>(atomicLoad(&gp[2 * idx]))'
                      : 'f32(atomicLoad(&gp[2 * idx])) * PR.invMomScale';
    var loadMY = casf ? 'bitcast<f32>(atomicLoad(&gp[2 * idx + 1]))'
                      : 'f32(atomicLoad(&gp[2 * idx + 1])) * PR.invMomScale';
    var zeroMom = casf ? '0u' : '0';
    var momT = casf ? 'atomic<u32>' : 'atomic<i32>';

    return [
      '// GENERATED from params.js (physics_version ' + P.physics_version + ', atomics=' + mode + ')',
      'const N_GRID : i32 = ' + N_GRID + ';',
      'const N_CELL : i32 = ' + N_CELL + ';',
      'const DX : f32 = ' + P.dx + ';',
      'const INV_DX : f32 = ' + P.inv_dx + ';',
      'const BOUND : i32 = ' + P.bound + ';',
      'const FLOOR : f32 = ' + P.floor_y + ';',
      'const CEIL : f32 = ' + (1.0 - P.floor_y) + ';',
      '',
      'struct Params {',
      '  dt : f32, pMass : f32, pVol : f32, mu : f32, la : f32,',
      '  gravity : f32, friction : f32,',
      '  massScale : f32, invMassScale : f32, momScale : f32, invMomScale : f32,',
      '  n : u32, pokeOn : u32,',
      '  pokeX : f32, pokeY : f32, pokeVX : f32, pokeVY : f32,',
      '  pokeRadius : f32, pokeRate : f32, pokeSpring : f32,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> PR : Params;',
      '@group(0) @binding(1) var<storage, read_write> pos : array<vec2<f32>>;',
      '@group(0) @binding(2) var<storage, read_write> vel : array<vec2<f32>>;',
      '@group(0) @binding(3) var<storage, read_write> Cm  : array<vec4<f32>>;',
      '@group(0) @binding(4) var<storage, read_write> Fm  : array<vec4<f32>>;',
      '@group(0) @binding(5) var<storage, read_write> gm  : array<atomic<u32>>;',
      '@group(0) @binding(6) var<storage, read_write> gp  : array<' + momT + '>;',
      // gv.xy = node velocity (what G2P gathers). gv.z = node mass in PARTICLE MASSES, gv.w =
      // |node momentum| in particle-mass*velocity -- the two quantities the mass/speed heatmaps
      // and the fixed-point headroom probe need, carried for free instead of a 9th buffer.
      '@group(0) @binding(7) var<storage, read_write> gv  : array<vec4<f32>>;',
      '',
      // ---------------------------------------------------------------- clear
      '@compute @workgroup_size(' + WG_G + ')',
      'fn clear_grid(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  atomicStore(&gm[idx], 0u);',
      '  atomicStore(&gp[2 * idx], ' + zeroMom + ');',
      '  atomicStore(&gp[2 * idx + 1], ' + zeroMom + ');',
      '  gv[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);',
      '}',
      '',
      // ---------------------------------------------------------------- P2G
      '@compute @workgroup_size(' + WG_P + ')',
      'fn p2g(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let p = gid.x;',
      '  if (p >= PR.n) { return; }',
      '  let Xp = pos[p] * INV_DX;',
      '  let base = vec2<i32>(Xp - vec2<f32>(0.5, 0.5));',   // positions clamped >= 3dx -> floor
      '  let fx = Xp - vec2<f32>(base);',
      '  let w0 = 0.5 * (vec2<f32>(1.5, 1.5) - fx) * (vec2<f32>(1.5, 1.5) - fx);',
      '  let w1 = vec2<f32>(0.75, 0.75) - (fx - vec2<f32>(1.0, 1.0)) * (fx - vec2<f32>(1.0, 1.0));',
      '  let w2 = 0.5 * (fx - vec2<f32>(0.5, 0.5)) * (fx - vec2<f32>(0.5, 0.5));',
      '',
      '  let Fp = Fm[p];',                                   // (f00, f01, f10, f11), row major
      '  let f00 = Fp.x; let f01 = Fp.y; let f10 = Fp.z; let f11 = Fp.w;',
      // closed-form 2D polar rotation R = U V^T (what Taichi's svd2d is built on)
      '  let rx = f00 + f11; let ry = f10 - f01;',
      '  let rinv = 1.0 / max(sqrt(rx * rx + ry * ry), 1e-30);',
      '  let rc = rx * rinv; let rs = ry * rinv;',
      '  let a00 = f00 - rc; let a01 = f01 + rs; let a10 = f10 - rs; let a11 = f11 - rc;',
      '  let b00 = a00 * f00 + a01 * f01; let b01 = a00 * f10 + a01 * f11;',
      '  let b10 = a10 * f00 + a11 * f01; let b11 = a10 * f10 + a11 * f11;',
      '  let Jd = f00 * f11 - f01 * f10;',
      '  let lt = PR.la * (Jd - 1.0) * Jd;',
      '  let k = -PR.dt * 4.0 * PR.pVol * INV_DX * INV_DX;',
      '  let twoMu = 2.0 * PR.mu;',
      '  let Cp = Cm[p];',
      '  let af00 = k * (twoMu * b00 + lt) + PR.pMass * Cp.x;',
      '  let af01 = k * (twoMu * b01)      + PR.pMass * Cp.y;',
      '  let af10 = k * (twoMu * b10)      + PR.pMass * Cp.z;',
      '  let af11 = k * (twoMu * b11 + lt) + PR.pMass * Cp.w;',
      '  let mv = PR.pMass * vel[p];',
      '',
      '  for (var i = 0; i < 3; i = i + 1) {',
      '    let wxi = select(select(w0.x, w1.x, i == 1), w2.x, i == 2);',
      '    let dpx = (f32(i) - fx.x) * DX;',
      '    for (var j = 0; j < 3; j = j + 1) {',
      '      let wyj = select(select(w0.y, w1.y, j == 1), w2.y, j == 2);',
      '      let dpy = (f32(j) - fx.y) * DX;',
      '      let w = wxi * wyj;',
      '      let gi = (base.x + i) * N_GRID + (base.y + j);',
      '      let wm  = w * PR.pMass;',
      '      let mvx = w * (mv.x + af00 * dpx + af01 * dpy);',
      '      let mvy = w * (mv.y + af10 * dpx + af11 * dpy);',
      scatterMass,
      scatterMX,
      scatterMY,
      '    }',
      '  }',
      '}',
      '',
      // ---------------------------------------------------------------- grid op (+ fused clear)
      '@compute @workgroup_size(' + WG_G + ')',
      'fn grid_op(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  let m = ' + loadMass + ';',
      '  let momx = ' + loadMX + ';',
      '  let momy = ' + loadMY + ';',
      '  atomicStore(&gm[idx], 0u);',                 // fused clear: grid_op is the only reader
      '  atomicStore(&gp[2 * idx], ' + zeroMom + ');',
      '  atomicStore(&gp[2 * idx + 1], ' + zeroMom + ');',
      '  var vx = momx; var vy = momy;',
      '  if (m > 0.0) { vx = vx / m; vy = vy / m; }',
      '  vy = vy - PR.dt * PR.gravity;',
      '  let i = idx / N_GRID;',
      '  let j = idx - i * N_GRID;',
      '  if (PR.pokeOn == 1u) {',
      '    let cx = PR.pokeX - f32(i) * DX;',
      '    let cy = PR.pokeY - f32(j) * DX;',
      '    let r2 = cx * cx + cy * cy;',
      '    let s2 = PR.pokeRadius * PR.pokeRadius;',
      '    if (r2 < s2 * 6.0) {',
      '      let lam = min(0.5, PR.pokeRate * PR.dt * exp(-r2 / (0.5 * s2)));',
      '      vx = vx + lam * (PR.pokeVX + cx * PR.pokeSpring - vx);',
      '      vy = vy + lam * (PR.pokeVY + cy * PR.pokeSpring - vy);',
      '    }',
      '  }',
      '  if (j < BOUND && vy < 0.0) {',              // floor: separating + Coulomb friction
      '    let cap = PR.friction * (-vy);',
      '    if (vx > 0.0) { vx = max(0.0, vx - cap); } else if (vx < 0.0) { vx = min(0.0, vx + cap); }',
      '    vy = 0.0;',
      '  }',
      '  if (j > N_GRID - BOUND && vy > 0.0) { vy = 0.0; }',
      '  if (i < BOUND && vx < 0.0) { vx = 0.0; vy = 0.0; }',
      '  if (i > N_GRID - BOUND && vx > 0.0) { vx = 0.0; vy = 0.0; }',
      '  let ipm = 1.0 / max(PR.pMass, 1e-30);',
      '  gv[idx] = vec4<f32>(vx, vy, m * ipm, length(vec2<f32>(momx, momy)) * ipm);',
      '}',
      '',
      // ---------------------------------------------------------------- G2P
      '@compute @workgroup_size(' + WG_P + ')',
      'fn g2p(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let p = gid.x;',
      '  if (p >= PR.n) { return; }',
      '  let Xp = pos[p] * INV_DX;',
      '  let base = vec2<i32>(Xp - vec2<f32>(0.5, 0.5));',
      '  let fx = Xp - vec2<f32>(base);',
      '  let w0 = 0.5 * (vec2<f32>(1.5, 1.5) - fx) * (vec2<f32>(1.5, 1.5) - fx);',
      '  let w1 = vec2<f32>(0.75, 0.75) - (fx - vec2<f32>(1.0, 1.0)) * (fx - vec2<f32>(1.0, 1.0));',
      '  let w2 = 0.5 * (fx - vec2<f32>(0.5, 0.5)) * (fx - vec2<f32>(0.5, 0.5));',
      '  var nv = vec2<f32>(0.0, 0.0);',
      '  var c00 = 0.0; var c01 = 0.0; var c10 = 0.0; var c11 = 0.0;',
      '  for (var i = 0; i < 3; i = i + 1) {',
      '    let wxi = select(select(w0.x, w1.x, i == 1), w2.x, i == 2);',
      '    let dpx = (f32(i) - fx.x) * DX;',
      '    for (var j = 0; j < 3; j = j + 1) {',
      '      let wyj = select(select(w0.y, w1.y, j == 1), w2.y, j == 2);',
      '      let dpy = (f32(j) - fx.y) * DX;',
      '      let w = wxi * wyj;',
      '      let g = gv[(base.x + i) * N_GRID + (base.y + j)].xy;',
      '      nv = nv + w * g;',
      '      let s = 4.0 * w * INV_DX * INV_DX;',
      '      c00 = c00 + s * g.x * dpx; c01 = c01 + s * g.x * dpy;',
      '      c10 = c10 + s * g.y * dpx; c11 = c11 + s * g.y * dpy;',
      '    }',
      '  }',
      '  vel[p] = nv;',
      '  let np = pos[p] + PR.dt * nv;',
      '  pos[p] = clamp(np, vec2<f32>(FLOOR, FLOOR), vec2<f32>(CEIL, CEIL));',
      '  let g00 = 1.0 + PR.dt * c00; let g01 = PR.dt * c01;',
      '  let g10 = PR.dt * c10;       let g11 = 1.0 + PR.dt * c11;',
      '  let Fp = Fm[p];',
      '  Fm[p] = vec4<f32>(g00 * Fp.x + g01 * Fp.z, g00 * Fp.y + g01 * Fp.w,',
      '                    g10 * Fp.x + g11 * Fp.z, g10 * Fp.y + g11 * Fp.w);',
      '  Cm[p] = vec4<f32>(c00, c01, c10, c11);',
      '}',
      '',
      // The direct analogue of Taichi's `noop` kernel, which costs 56.4 us to launch from Python.
      // Dispatched inside a recorded command buffer instead, its cost is the per-dispatch FLOOR --
      // the number that decides whether 167 substeps a frame is affordable at all.
      '@compute @workgroup_size(' + WG_G + ')',
      'fn empty(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  if (PR.n == 4294967295u) { gv[idx].w = 1.0; }',   // never true; keeps the store live
      '}'
    ].join('\n');
  }

  // ------------------------------------------------------------------ render shader
  function renderShader() {
    return [
      'struct RParams { radius : f32, aspect : f32, n : u32, view : u32,',
      '                 massRef : f32, vRef : f32, dimAlpha : f32, pad : f32, };',
      '@group(0) @binding(0) var<uniform> R : RParams;',
      '@group(0) @binding(1) var<storage, read> pos : array<vec2<f32>>;',
      '@group(0) @binding(2) var<storage, read> vel : array<vec2<f32>>;',
      '@group(0) @binding(3) var<storage, read> gv  : array<vec4<f32>>;',
      '',
      'struct VOut { @builtin(position) p : vec4<f32>, @location(0) uv : vec2<f32>,',
      '              @location(1) sp : f32, };',
      '',
      // the unit quad, derived arithmetically -- dynamic indexing of a const array is a
      // portability hazard across WGSL implementations and this costs nothing
      'fn quad(vi : u32) -> vec2<f32> {',
      '  let qx = select(0.0, 1.0, vi == 1u || vi == 4u || vi == 5u);',
      '  let qy = select(0.0, 1.0, vi == 2u || vi == 3u || vi == 5u);',
      '  return vec2<f32>(qx, qy) * 2.0 - vec2<f32>(1.0, 1.0);',
      '}',
      '',
      '@vertex fn vs_particles(@builtin(vertex_index) vi : u32,',
      '                        @builtin(instance_index) ii : u32) -> VOut {',
      '  let q = quad(vi);',
      '  let c = pos[ii];',
      '  let ndc = vec2<f32>(c.x * 2.0 - 1.0, c.y * 2.0 - 1.0) + q * vec2<f32>(R.radius, R.radius * R.aspect);',
      '  var o : VOut;',
      '  o.p = vec4<f32>(ndc, 0.0, 1.0);',
      '  o.uv = q;',
      '  o.sp = length(vel[ii]);',
      '  return o;',
      '}',
      '',
      '@vertex fn vs_full(@builtin(vertex_index) vi : u32) -> VOut {',
      '  let q = quad(vi);',
      '  var o : VOut;',
      '  o.p = vec4<f32>(q, 0.0, 1.0);',
      '  o.uv = q * 0.5 + vec2<f32>(0.5, 0.5);',
      '  o.sp = 0.0;',
      '  return o;',
      '}',
      '',
      'fn ramp(t : f32) -> vec3<f32> {',
      '  let u = clamp(t, 0.0, 1.0);',
      '  let c0 = vec3<f32>(0.039, 0.055, 0.078);',
      '  let c1 = vec3<f32>(0.106, 0.298, 0.451);',
      '  let c2 = vec3<f32>(0.435, 0.827, 0.933);',
      '  let c3 = vec3<f32>(1.0, 0.616, 0.361);',
      '  let c4 = vec3<f32>(1.0, 0.98, 0.92);',
      '  if (u < 0.25) { return mix(c0, c1, u / 0.25); }',
      '  if (u < 0.5)  { return mix(c1, c2, (u - 0.25) / 0.25); }',
      '  if (u < 0.75) { return mix(c2, c3, (u - 0.5) / 0.25); }',
      '  return mix(c3, c4, (u - 0.75) / 0.25);',
      '}',
      '',
      '@fragment fn fs_particles(o : VOut) -> @location(0) vec4<f32> {',
      '  let r2 = dot(o.uv, o.uv);',
      '  if (r2 > 1.0) { discard; }',
      '  let t = clamp(o.sp / max(R.vRef, 1e-6), 0.0, 1.0);',
      '  let base = mix(vec3<f32>(1.0, 0.616, 0.361), vec3<f32>(1.0, 0.98, 0.93), t);',
      '  let shade = 0.55 + 0.45 * sqrt(max(0.0, 1.0 - r2));',
      '  return vec4<f32>(base * shade, R.dimAlpha);',
      '}',
      '',
      '@fragment fn fs_grid(o : VOut) -> @location(0) vec4<f32> {',
      '  let gi = vec2<i32>(clamp(o.uv * f32(' + N_GRID + '), vec2<f32>(0.0), vec2<f32>(' + (N_GRID - 1) + '.0)));',
      '  let cell = gv[gi.x * ' + N_GRID + ' + gi.y];',
      '  var t = 0.0;',
      '  if (R.view == 1u) { t = log(1.0 + cell.z) / log(1.0 + max(R.massRef, 1e-6)); }',
      '  else { t = length(cell.xy) / max(R.vRef, 1e-6); }',
      '  return vec4<f32>(ramp(t), 1.0);',
      '}'
    ].join('\n');
  }

  // ------------------------------------------------------------------ helpers
  function supported() { return (typeof navigator !== 'undefined') && !!navigator.gpu; }

  // Deterministic disk seeding for the demo (verification uses the exact numpy point set exported
  // from sim.physics instead, so the initial condition is identical there).
  function seedDisk(cx, cy, radius, n, seed) {
    var s = (seed >>> 0) || 1;
    function rnd() { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }
    var pts = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) {
      var a = rnd() * Math.PI * 2, r = radius * Math.sqrt(rnd());
      pts[2 * i] = cx + r * Math.cos(a);
      pts[2 * i + 1] = cy + r * Math.sin(a);
    }
    return pts;
  }

  var _device = null, _adapterInfo = null, _hasTimestamp = false, _limits = null;
  var ERRORS = [];

  async function getDevice() {
    if (_device) return _device;
    if (!supported()) throw new Error('navigator.gpu is undefined (are you on a secure origin?)');
    var adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('requestAdapter returned null');
    _adapterInfo = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    var feats = [];
    _hasTimestamp = adapter.features.has('timestamp-query');
    if (_hasTimestamp) feats.push('timestamp-query');
    _device = await adapter.requestDevice({ requiredFeatures: feats });
    _limits = {};
    ['maxStorageBuffersPerShaderStage', 'maxStorageBufferBindingSize', 'maxBufferSize',
      'maxComputeWorkgroupsPerDimension', 'maxComputeInvocationsPerWorkgroup']
      .forEach(function (k) { _limits[k] = _device.limits[k]; });
    // WebGPU errors are ASYNCHRONOUS and, left alone, silent: an invalid bind group makes every
    // dispatch a no-op and the simulation "runs" at the speed of doing nothing. Never leave them
    // unlistened.
    _device.addEventListener('uncapturederror', function (e) {
      ERRORS.push(String(e.error && e.error.message || e.error));
      console.error('WebGPU uncaptured error:', e.error);
    });
    _device.lost.then(function (e) { ERRORS.push('DEVICE LOST: ' + e.message); });
    return _device;
  }

  function deviceInfo() {
    var a = _adapterInfo || {};
    return {
      vendor: a.vendor || '', architecture: a.architecture || '',
      device: a.device || '', description: a.description || '',
      timestampQuery: _hasTimestamp, limits: _limits
    };
  }
  function errors() { return ERRORS.slice(); }

  // ------------------------------------------------------------------ the simulator
  async function createSim(opts) {
    opts = opts || {};
    var device = await getDevice();
    var n = opts.n | 0;
    var area = opts.area;
    var dt = opts.dt !== undefined ? opts.dt : P.dt;
    var E = opts.E !== undefined ? opts.E : P.E;
    var mode = opts.atomics || 'fixed';
    var kM = opts.kM !== undefined ? opts.kM : P.kM;
    var kV = opts.kV !== undefined ? opts.kV : P.kV;

    var nu = P.NU;
    var pVol = area / n;
    var pMass = pVol * P.p_rho;
    var mu = E / (2.0 * (1.0 + nu));
    var la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    // Fixed-point scales in units of ONE PARTICLE MASS: an integer of 2^kM is one particle's mass
    // at a node, an integer of 2^kV is one particle-mass times one unit of velocity.
    var massScale = Math.pow(2, kM) / pMass;
    var momScale = Math.pow(2, kV) / pMass;

    device.pushErrorScope('validation');

    var U = GPUBufferUsage;
    function buf(bytes, usage) { return device.createBuffer({ size: bytes, usage: usage }); }
    var STO = U.STORAGE | U.COPY_DST | U.COPY_SRC;

    var posBuf = buf(n * 8, STO);
    var velBuf = buf(n * 8, STO);
    var CBuf = buf(n * 16, STO);
    var FBuf = buf(n * 16, STO);
    var gmBuf = buf(N_CELL * 4, STO);
    var gpBuf = buf(N_CELL * 8, STO);
    var gvBuf = buf(N_CELL * 16, STO);
    var uBuf = buf(80, U.UNIFORM | U.COPY_DST);
    var readBuf = device.createBuffer({ size: n * 8, usage: U.COPY_DST | U.MAP_READ });
    var gridReadBuf = device.createBuffer({ size: N_CELL * 16, usage: U.COPY_DST | U.MAP_READ });

    var mod = device.createShaderModule({ code: buildShader(mode), label: 'mpm-' + mode });
    var info = await mod.getCompilationInfo();
    var errs = info.messages.filter(function (m) { return m.type === 'error'; });
    if (errs.length) throw new Error('WGSL: ' + errs.map(function (m) { return m.lineNum + ': ' + m.message; }).join(' | '));

    function sb(b) {
      return { binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } };
    }
    var layout = device.createBindGroupLayout({
      label: 'mpm-compute-bgl',
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        sb(1), sb(2), sb(3), sb(4), sb(5), sb(6), sb(7)]
    });
    var bind = device.createBindGroup({
      label: 'mpm-compute-bg', layout: layout,
      entries: [uBuf, posBuf, velBuf, CBuf, FBuf, gmBuf, gpBuf, gvBuf]
        .map(function (b, i) { return { binding: i, resource: { buffer: b } }; })
    });
    var pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    function pipe(entry) {
      return device.createComputePipeline({ layout: pl, label: entry,
        compute: { module: mod, entryPoint: entry } });
    }
    var pClear = pipe('clear_grid'), pP2G = pipe('p2g'), pGrid = pipe('grid_op'), pG2P = pipe('g2p');
    var pEmpty = pipe('empty');

    var setupError = await device.popErrorScope();
    if (setupError) throw new Error('WebGPU setup: ' + setupError.message);

    var P_WG = Math.ceil(n / WG_P);
    var poke = { on: false, x: 0, y: 0, vx: 0, vy: 0, radius: 0.075, rate: 900.0, spring: 16.0 };
    var uArr = new ArrayBuffer(80);
    var uF = new Float32Array(uArr), uU = new Uint32Array(uArr);

    function writeUniform() {
      uF[0] = dt; uF[1] = pMass; uF[2] = pVol; uF[3] = mu; uF[4] = la;
      uF[5] = P.gravity; uF[6] = P.FRICTION;
      uF[7] = massScale; uF[8] = 1.0 / massScale; uF[9] = momScale; uF[10] = 1.0 / momScale;
      uU[11] = n >>> 0; uU[12] = poke.on ? 1 : 0;
      uF[13] = poke.x; uF[14] = poke.y; uF[15] = poke.vx; uF[16] = poke.vy;
      uF[17] = poke.radius; uF[18] = poke.rate; uF[19] = poke.spring;
      device.queue.writeBuffer(uBuf, 0, uArr);
    }
    writeUniform();

    // -------------------------------------------------- timestamp query (GPU-side clock)
    var qset = null, qResolve = null, qRead = null;
    if (_hasTimestamp) {
      qset = device.createQuerySet({ type: 'timestamp', count: 2 });
      qResolve = buf(16, U.QUERY_RESOLVE | U.COPY_SRC);
      qRead = device.createBuffer({ size: 16, usage: U.COPY_DST | U.MAP_READ });
    }

    function seed(pts, v0x, v0y) {
      var xs = new Float32Array(2 * n), vs = new Float32Array(2 * n);
      var Cs = new Float32Array(4 * n), Fs = new Float32Array(4 * n);
      for (var p = 0; p < n; p++) {
        xs[2 * p] = pts[2 * p]; xs[2 * p + 1] = pts[2 * p + 1];
        vs[2 * p] = v0x || 0; vs[2 * p + 1] = v0y || 0;
        Fs[4 * p] = 1; Fs[4 * p + 3] = 1;
      }
      device.queue.writeBuffer(posBuf, 0, xs);
      device.queue.writeBuffer(velBuf, 0, vs);
      device.queue.writeBuffer(CBuf, 0, Cs);
      device.queue.writeBuffer(FBuf, 0, Fs);
      device.queue.writeBuffer(gmBuf, 0, new Uint32Array(N_CELL));
      device.queue.writeBuffer(gpBuf, 0, new Int32Array(2 * N_CELL));
      device.queue.writeBuffer(gvBuf, 0, new Float32Array(4 * N_CELL));
    }

    // ------------------------------------------------------------------------------------------
    // ONE COMMAND BUFFER PER FRAME. Every substep's three dispatches go into a single compute
    // pass; WebGPU orders dispatches inside a pass and makes each one's writes visible to the
    // next, which is exactly the P2G -> grid -> G2P dependency. Nothing crosses back to the CPU.
    // ------------------------------------------------------------------------------------------
    function encodeFrame(substeps, opt) {
      opt = opt || {};
      var enc = device.createCommandEncoder();
      var desc = {};
      if (opt.timed && qset) {
        desc.timestampWrites = { querySet: qset, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
      }
      var pass = enc.beginComputePass(desc);
      pass.setBindGroup(0, bind);
      if (opt.clearFirst) { pass.setPipeline(pClear); pass.dispatchWorkgroups(GRID_WG); }
      var phases = opt.phases || 'pgG';
      var doP = phases.indexOf('p') >= 0, doG = phases.indexOf('g') >= 0,
        doQ = phases.indexOf('G') >= 0, doE = phases.indexOf('e') >= 0;
      for (var s = 0; s < substeps; s++) {
        if (doE) { pass.setPipeline(pEmpty); pass.dispatchWorkgroups(GRID_WG); }
        if (doP) { pass.setPipeline(pP2G); pass.dispatchWorkgroups(P_WG); }
        if (doG) { pass.setPipeline(pGrid); pass.dispatchWorkgroups(GRID_WG); }
        if (doQ) { pass.setPipeline(pG2P); pass.dispatchWorkgroups(P_WG); }
      }
      pass.end();
      if (opt.timed && qset) {
        enc.resolveQuerySet(qset, 0, 2, qResolve, 0);
        enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 16);
      }
      if (opt.readback) enc.copyBufferToBuffer(posBuf, 0, readBuf, 0, n * 8);
      if (opt.gridReadback) enc.copyBufferToBuffer(gvBuf, 0, gridReadBuf, 0, N_CELL * 16);
      var cmd = enc.finish();
      // noSubmit prices the CPU-side RECORDING alone: with one launch per substep gone, the next
      // candidate for the bottleneck is the JS that writes the 500-odd dispatches.
      if (opt.noSubmit) return;
      device.queue.submit([cmd]);
    }

    async function lastGpuNanos() {
      if (!qset) return null;
      await qRead.mapAsync(GPUMapMode.READ);
      var t = new BigUint64Array(qRead.getMappedRange().slice(0));
      qRead.unmap();
      return Number(t[1] - t[0]);
    }
    async function readPositions() {
      await readBuf.mapAsync(GPUMapMode.READ);
      var out = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      return out;
    }
    async function readGrid() {
      await gridReadBuf.mapAsync(GPUMapMode.READ);
      var out = new Float32Array(gridReadBuf.getMappedRange().slice(0));
      gridReadBuf.unmap();
      return out;
    }

    return {
      n: n, device: device, mode: mode,
      params: {
        E: E, dt: dt, nu: nu, mu: mu, la: la, pVol: pVol, pMass: pMass,
        n_grid: N_GRID, gravity: P.gravity, friction: P.FRICTION,
        kM: kM, kV: kV, massScale: massScale, momScale: momScale,
        atomics: mode, physics_version: P.physics_version
      },
      buffers: { pos: posBuf, vel: velBuf, gv: gvBuf },
      poke: poke,
      setDt: function (d) { dt = d; writeUniform(); },
      getDt: function () { return dt; },
      syncUniform: writeUniform,
      seed: seed, encodeFrame: encodeFrame, lastGpuNanos: lastGpuNanos,
      readPositions: readPositions, readGrid: readGrid,
      dispatchesPerFrame: function (substeps) { return 3 * substeps; },
      idle: function () { return device.queue.onSubmittedWorkDone(); },
      destroy: function () {
        [posBuf, velBuf, CBuf, FBuf, gmBuf, gpBuf, gvBuf, uBuf, readBuf, gridReadBuf]
          .forEach(function (b) { b.destroy(); });
      }
    };
  }

  // ------------------------------------------------------------------ renderer (no readback)
  async function createRenderer(canvas, sim) {
    var device = sim.device;
    var ctx = canvas.getContext('webgpu');
    var fmt = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device: device, format: fmt, alphaMode: 'opaque' });
    device.pushErrorScope('validation');
    var mod = device.createShaderModule({ code: renderShader(), label: 'mpm-render' });
    var ci = await mod.getCompilationInfo();
    var re = ci.messages.filter(function (m) { return m.type === 'error'; });
    if (re.length) throw new Error('render WGSL: ' + re.map(function (m) { return m.lineNum + ': ' + m.message; }).join(' | '));

    var rBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    var layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
      ]
    });
    var bind = device.createBindGroup({
      layout: layout,
      entries: [
        { binding: 0, resource: { buffer: rBuf } },
        { binding: 1, resource: { buffer: sim.buffers.pos } },
        { binding: 2, resource: { buffer: sim.buffers.vel } },
        { binding: 3, resource: { buffer: sim.buffers.gv } }
      ]
    });
    var pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    var blend = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
                  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
    var pParticles = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_particles' },
      fragment: { module: mod, entryPoint: 'fs_particles', targets: [{ format: fmt, blend: blend }] },
      primitive: { topology: 'triangle-list' }
    });
    var pGridView = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_full' },
      fragment: { module: mod, entryPoint: 'fs_grid', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' }
    });
    var err = await device.popErrorScope();
    if (err) throw new Error('renderer setup: ' + err.message);

    var ru = new Float32Array(8);
    var ruU = new Uint32Array(ru.buffer);

    return {
      draw: function (o) {
        o = o || {};
        var view = o.view || 'particles';                 // 'particles' | 'mass' | 'speed'
        var enc = device.createCommandEncoder();
        var pass = enc.beginRenderPass({
          colorAttachments: [{
            view: ctx.getCurrentTexture().createView(),
            clearValue: { r: 0.039, g: 0.055, b: 0.078, a: 1 },
            loadOp: 'clear', storeOp: 'store'
          }]
        });
        ru[0] = o.radius !== undefined ? o.radius : 0.010;
        ru[1] = canvas.width / canvas.height;
        ruU[2] = sim.n >>> 0;
        ruU[3] = view === 'mass' ? 1 : (view === 'speed' ? 2 : 0);
        ru[4] = o.massRef !== undefined ? o.massRef : 24.0;
        ru[5] = o.vRef !== undefined ? o.vRef : 2.5;
        ru[6] = 1.0;
        device.queue.writeBuffer(rBuf, 0, ru);
        pass.setBindGroup(0, bind);
        if (view !== 'particles') {
          pass.setPipeline(pGridView); pass.draw(6, 1);
          if (o.overlay) {
            // faint: on a grid view the GRID is the subject, so the particles are a locator, not
            // the picture. At 0.5 they simply repaint the heatmap in particle colours.
            ru[6] = 0.13; device.queue.writeBuffer(rBuf, 0, ru);
            pass.setPipeline(pParticles); pass.draw(6, sim.n);
          }
        } else {
          pass.setPipeline(pParticles); pass.draw(6, sim.n);
        }
        pass.end();
        device.queue.submit([enc.finish()]);
      }
    };
  }

  return {
    supported: supported, getDevice: getDevice, deviceInfo: deviceInfo, errors: errors,
    createSim: createSim, createRenderer: createRenderer, seedDisk: seedDisk,
    PARAMS: P, N_GRID: N_GRID, N_CELL: N_CELL, WG_P: WG_P, WG_G: WG_G,
    buildShader: buildShader
  };
});
