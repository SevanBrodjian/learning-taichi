// mpm-nn-webgpu.js -- WebGPU MLS-MPM (canonical WATER) whose GRID UPDATE can be a neural network.
//
// PORTABILITY: depends only on params.js (generated from sim.physics) and nnweights.js (the trained
// weights). No dashboard, no server, no fetch. Load with <script> -> window.MPMNN.
//
// -------------------------------------------------------------------------------------------------
// THE SEAM
// -------------------------------------------------------------------------------------------------
// P2G and G2P stay analytic and unchanged. The kernel that is replaced is the WHOLE grid update: the
// pass that turns a node's accumulated mass and momentum into its velocity. In the analytic kernel
// that is four things fused together --
//     v = p / m                              (divide out the mass)
//     v.y -= dt * g                           (gravity)
//     if at a wall and moving INTO it: zero the normal component
//     ... and cap the tangential component by Coulomb friction
// -- plus the bookkeeping zeroing of the atomic accumulators, which grid_op does because it is their
// only reader. The learned kernel does the SAME four physics steps with an MLP and keeps the same
// bookkeeping. Nothing analytic is applied to the network's output: no gravity is added afterwards,
// no wall clamp is applied afterwards. The vector the network emits IS the node velocity that G2P
// gathers.
//
// Network I/O, per cell:
//     in  (8) [ m/pMass, px/pMass, py/pMass, wallL, wallR, wallBottom, wallTop, friction ]
//     out (2) [ vx, vy ]
// The division by pMass is a multiply by ONE scalar that is uniform over the whole grid and known
// before the substep starts, so it is folded into the first layer's weights. It is not the per-cell
// division the network has to learn.
//
// -------------------------------------------------------------------------------------------------
// WHAT WEBGPU FORCES HERE
// -------------------------------------------------------------------------------------------------
//   1. EIGHT STORAGE BUFFERS PER STAGE is the baseline limit, and going over it silently invalidates
//      the bind group: every dispatch is dropped, the data stays zero, and the timing curve comes out
//      beautifully flat over nothing. Adding a weights buffer costs one slot, so the fluid state is
//      packed to leave room: velocity and the volume ratio J share one vec4 (vx, vy, J, _). Total is
//      seven storage buffers plus one uniform.
//   2. NO ATOMIC FLOAT ADD in WGSL, so P2G scatters into fixed point (mass -> atomic<u32>, momentum
//      -> atomic<i32>) in units of one particle mass, exactly as the analytic port does.
//   3. THE GRID KERNEL IS SELECTABLE at pipeline level, not by a uniform branch: 'analytic', 'nn'
//      (dense, every cell), 'nnsparse' (cells with no mass skip the network), and 'null' (loads and
//      clears the accumulators and writes the raw momentum, doing no physics at all). 'null' exists
//      so the arithmetic cost of a grid kernel can be isolated by DIFFERENCING against a kernel with
//      identical memory traffic, rather than by timing a 3 us dispatch directly.
//
// WHY 'nnsparse' IS EXACT AND NOT AN APPROXIMATION. G2P gathers from exactly the 3x3 stencil P2G
// scattered into, so a cell with zero mass cannot be read by any particle. Whatever is written there
// is unobservable, and skipping the network on those cells changes no trajectory.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./params.js'), require('./nnweights.js'));
  } else {
    root.MPMNN = factory(root.MPM_PARAMS, root.MPM_NN);
  }
})(typeof self !== 'undefined' ? self : this, function (P, NN) {
  'use strict';

  var N_GRID = P.n_grid | 0;
  var N_CELL = N_GRID * N_GRID;
  var WG_P = 64;
  var WG_G = 64;
  var GRID_WG = Math.ceil(N_CELL / WG_G);
  var N_IN = 8;

  function netSize(h) { return N_IN * h + h + h * h + h + h * 2 + 2; }

  function decodeNet(key) {
    var e = NN.nets[key];
    if (!e) throw new Error('no such net: ' + key);
    var bin = atob(e.b64);
    var u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    var f = new Float32Array(u8.buffer);
    if (f.length !== netSize(e.hidden)) throw new Error('net size mismatch ' + key);
    return { hidden: e.hidden, w: f };
  }

  // ------------------------------------------------------------------ WGSL
  // `grid` selects the grid-update kernel; `H` is baked in as a literal so the MLP loops have
  // compile-time trip counts.
  function buildShader(grid, H) {
    var oW1 = 0, oB1 = N_IN * H, oW2 = oB1 + H, oB2 = oW2 + H * H,
      oW3 = oB2 + H, oB3 = oW3 + H * 2;

    var mlp = [
      '  var xin : array<f32, 8>;',
      '  xin[0] = mh; xin[1] = phx; xin[2] = phy;',
      '  xin[3] = f32(i < BOUND); xin[4] = f32(i > N_GRID - BOUND);',
      '  xin[5] = f32(j < BOUND); xin[6] = f32(j > N_GRID - BOUND);',
      '  xin[7] = PR.friction;',
      '  var a1 : array<f32, ' + H + '>;',
      '  for (var o = 0u; o < ' + H + 'u; o = o + 1u) {',
      '    var s = W[' + oB1 + 'u + o];',
      '    for (var k = 0u; k < 8u; k = k + 1u) { s = s + xin[k] * W[' + oW1 + 'u + k * ' + H + 'u + o]; }',
      '    a1[o] = max(s, 0.0);',
      '  }',
      '  var a2 : array<f32, ' + H + '>;',
      '  for (var o = 0u; o < ' + H + 'u; o = o + 1u) {',
      '    var s = W[' + oB2 + 'u + o];',
      '    for (var k = 0u; k < ' + H + 'u; k = k + 1u) { s = s + a1[k] * W[' + oW2 + 'u + k * ' + H + 'u + o]; }',
      '    a2[o] = max(s, 0.0);',
      '  }',
      '  var vx = W[' + oB3 + 'u];',
      '  var vy = W[' + (oB3 + 1) + 'u];',
      '  for (var k = 0u; k < ' + H + 'u; k = k + 1u) {',
      '    vx = vx + a2[k] * W[' + oW3 + 'u + k * 2u];',
      '    vy = vy + a2[k] * W[' + (oW3 + 1) + 'u + k * 2u];',
      '  }'
    ].join('\n');

    var gridBody;
    if (grid === 'analytic') {
      gridBody = [
        '  var vx = momx; var vy = momy;',
        '  if (m > 0.0) { vx = vx / m; vy = vy / m; }',
        '  vy = vy - PR.dt * PR.gravity;',
        '  if (j < BOUND && vy < 0.0) {',
        '    let cap = PR.friction * (-vy);',
        '    if (vx > 0.0) { vx = max(0.0, vx - cap); } else if (vx < 0.0) { vx = min(0.0, vx + cap); }',
        '    vy = 0.0;',
        '  }',
        '  if (j > N_GRID - BOUND && vy > 0.0) {',
        '    let cap = PR.friction * vy;',
        '    if (vx > 0.0) { vx = max(0.0, vx - cap); } else if (vx < 0.0) { vx = min(0.0, vx + cap); }',
        '    vy = 0.0;',
        '  }',
        '  if (i < BOUND && vx < 0.0) {',
        '    let cap = PR.friction * (-vx);',
        '    if (vy > 0.0) { vy = max(0.0, vy - cap); } else if (vy < 0.0) { vy = min(0.0, vy + cap); }',
        '    vx = 0.0;',
        '  }',
        '  if (i > N_GRID - BOUND && vx > 0.0) {',
        '    let cap = PR.friction * vx;',
        '    if (vy > 0.0) { vy = max(0.0, vy - cap); } else if (vy < 0.0) { vy = min(0.0, vy + cap); }',
        '    vx = 0.0;',
        '  }'
      ].join('\n');
    } else if (grid === 'null') {
      // identical memory traffic, zero physics -- the differencing baseline
      gridBody = '  var vx = momx; var vy = momy;';
    } else {
      // 'nn' (dense) and 'nnsparse' (skip cells no particle can gather from)
      var pre = (grid === 'nnsparse')
        ? '  if (m <= 0.0) { gv[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0); return; }\n' : '';
      gridBody = pre +
        '  let mh = m * PR.invPMass;\n' +
        '  let phx = momx * PR.invPMass;\n' +
        '  let phy = momy * PR.invPMass;\n' + mlp;
    }

    return [
      '// GENERATED (physics_version ' + P.physics_version + ', grid=' + grid + ', H=' + H + ')',
      'const N_GRID : i32 = ' + N_GRID + ';',
      'const N_CELL : i32 = ' + N_CELL + ';',
      'const DX : f32 = ' + P.dx + ';',
      'const INV_DX : f32 = ' + P.inv_dx + ';',
      'const BOUND : i32 = ' + P.bound + ';',
      'const FLOOR : f32 = ' + P.floor_y + ';',
      'const CEIL : f32 = ' + (1.0 - P.floor_y) + ';',
      '',
      'struct Params {',
      '  dt : f32, pMass : f32, invPMass : f32, pVol : f32, E : f32,',
      '  gravity : f32, friction : f32,',
      '  massScale : f32, invMassScale : f32, momScale : f32, invMomScale : f32,',
      '  n : u32, pad0 : u32, pad1 : u32, pad2 : u32, pad3 : u32,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> PR : Params;',
      '@group(0) @binding(1) var<storage, read_write> pos : array<vec2<f32>>;',
      // (vx, vy, J, unused) -- packed so the weights buffer fits under the 8-buffer limit
      '@group(0) @binding(2) var<storage, read_write> pst : array<vec4<f32>>;',
      '@group(0) @binding(3) var<storage, read_write> Cm  : array<vec4<f32>>;',
      '@group(0) @binding(4) var<storage, read_write> gm  : array<atomic<u32>>;',
      '@group(0) @binding(5) var<storage, read_write> gp  : array<atomic<i32>>;',
      // gv.xy = node velocity (what G2P gathers); gv.z = node mass in particle masses;
      // gv.w = |node momentum| in particle-mass*velocity (the heatmaps and the headroom probe)
      '@group(0) @binding(6) var<storage, read_write> gv  : array<vec4<f32>>;',
      '@group(0) @binding(7) var<storage, read> W : array<f32>;',
      '',
      '@compute @workgroup_size(' + WG_G + ')',
      'fn clear_grid(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  atomicStore(&gm[idx], 0u);',
      '  atomicStore(&gp[2 * idx], 0);',
      '  atomicStore(&gp[2 * idx + 1], 0);',
      '  gv[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);',
      '}',
      '',
      // ------------------------------------------------------------- P2G (weakly compressible fluid)
      '@compute @workgroup_size(' + WG_P + ')',
      'fn p2g(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let p = gid.x;',
      '  if (p >= PR.n) { return; }',
      '  let Xp = pos[p] * INV_DX;',
      '  let base = vec2<i32>(Xp - vec2<f32>(0.5, 0.5));',
      '  let fx = Xp - vec2<f32>(base);',
      '  let w0 = 0.5 * (vec2<f32>(1.5, 1.5) - fx) * (vec2<f32>(1.5, 1.5) - fx);',
      '  let w1 = vec2<f32>(0.75, 0.75) - (fx - vec2<f32>(1.0, 1.0)) * (fx - vec2<f32>(1.0, 1.0));',
      '  let w2 = 0.5 * (fx - vec2<f32>(0.5, 0.5)) * (fx - vec2<f32>(0.5, 0.5));',
      '  let st = pst[p];',
      '  let pressure = PR.E * (st.z - 1.0);',           // weakly compressible: sigma = E (J-1) I
      '  let k = -PR.dt * 4.0 * PR.pVol * INV_DX * INV_DX * pressure;',
      '  let Cp = Cm[p];',
      '  let af00 = k + PR.pMass * Cp.x;',
      '  let af01 =     PR.pMass * Cp.y;',
      '  let af10 =     PR.pMass * Cp.z;',
      '  let af11 = k + PR.pMass * Cp.w;',
      '  let mv = PR.pMass * st.xy;',
      '  for (var i = 0; i < 3; i = i + 1) {',
      '    let wxi = select(select(w0.x, w1.x, i == 1), w2.x, i == 2);',
      '    let dpx = (f32(i) - fx.x) * DX;',
      '    for (var j = 0; j < 3; j = j + 1) {',
      '      let wyj = select(select(w0.y, w1.y, j == 1), w2.y, j == 2);',
      '      let dpy = (f32(j) - fx.y) * DX;',
      '      let w = wxi * wyj;',
      '      let gi = (base.x + i) * N_GRID + (base.y + j);',
      '      atomicAdd(&gm[gi], u32(round(w * PR.pMass * PR.massScale)));',
      '      atomicAdd(&gp[2 * gi], i32(round(w * (mv.x + af00 * dpx + af01 * dpy) * PR.momScale)));',
      '      atomicAdd(&gp[2 * gi + 1], i32(round(w * (mv.y + af10 * dpx + af11 * dpy) * PR.momScale)));',
      '    }',
      '  }',
      '}',
      '',
      // ------------------------------------------------------------- the grid update (the seam)
      '@compute @workgroup_size(' + WG_G + ')',
      'fn grid_op(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  let m    = f32(atomicLoad(&gm[idx])) * PR.invMassScale;',
      '  let momx = f32(atomicLoad(&gp[2 * idx])) * PR.invMomScale;',
      '  let momy = f32(atomicLoad(&gp[2 * idx + 1])) * PR.invMomScale;',
      '  atomicStore(&gm[idx], 0u);',                 // fused clear: this kernel is the only reader
      '  atomicStore(&gp[2 * idx], 0);',
      '  atomicStore(&gp[2 * idx + 1], 0);',
      '  let i = idx / N_GRID;',
      '  let j = idx - i * N_GRID;',
      gridBody,
      '  gv[idx] = vec4<f32>(vx, vy, m * PR.invPMass,',
      '                      length(vec2<f32>(momx, momy)) * PR.invPMass);',
      '}',
      '',
      // ------------------------------------------------------------- G2P (fluid: advect J)
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
      '  let st = pst[p];',
      '  let np = pos[p] + PR.dt * nv;',
      '  pos[p] = clamp(np, vec2<f32>(FLOOR, FLOOR), vec2<f32>(CEIL, CEIL));',
      '  pst[p] = vec4<f32>(nv.x, nv.y, st.z * (1.0 + PR.dt * (c00 + c11)), 0.0);',
      '  Cm[p] = vec4<f32>(c00, c01, c10, c11);',
      '}',
      '',
      // the per-dispatch floor: a dispatch that reaches memory but computes nothing
      '@compute @workgroup_size(' + WG_G + ')',
      'fn empty(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  if (PR.n == 4294967295u) { gv[idx].w = 1.0; }',
      '}'
    ].join('\n');
  }

  // ------------------------------------------------------------------ render shader
  function renderShader() {
    return [
      'struct RP { radius : f32, aspect : f32, n : u32, view : u32, vRef : f32, massRef : f32,',
      '            tint : f32, pad : f32, };',
      '@group(0) @binding(0) var<uniform> R : RP;',
      '@group(0) @binding(1) var<storage, read> pos : array<vec2<f32>>;',
      '@group(0) @binding(2) var<storage, read> pst : array<vec4<f32>>;',
      '@group(0) @binding(3) var<storage, read> gv  : array<vec4<f32>>;',
      'struct VOut { @builtin(position) p : vec4<f32>, @location(0) uv : vec2<f32>, @location(1) sp : f32, };',
      'fn quad(vi : u32) -> vec2<f32> {',
      '  let qx = select(0.0, 1.0, vi == 1u || vi == 4u || vi == 5u);',
      '  let qy = select(0.0, 1.0, vi == 2u || vi == 3u || vi == 5u);',
      '  return vec2<f32>(qx, qy) * 2.0 - vec2<f32>(1.0, 1.0);',
      '}',
      '@vertex fn vs_particles(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VOut {',
      '  let q = quad(vi);',
      '  let c = pos[ii];',
      '  var o : VOut;',
      '  o.p = vec4<f32>(vec2<f32>(c.x * 2.0 - 1.0, c.y * 2.0 - 1.0)',
      '                  + q * vec2<f32>(R.radius, R.radius * R.aspect), 0.0, 1.0);',
      '  o.uv = q;',
      '  o.sp = length(pst[ii].xy);',
      '  return o;',
      '}',
      '@vertex fn vs_full(@builtin(vertex_index) vi : u32) -> VOut {',
      '  let q = quad(vi);',
      '  var o : VOut; o.p = vec4<f32>(q, 0.0, 1.0); o.uv = q * 0.5 + vec2<f32>(0.5, 0.5); o.sp = 0.0;',
      '  return o;',
      '}',
      'fn ramp(t : f32) -> vec3<f32> {',
      '  let u = clamp(t, 0.0, 1.0);',
      '  let c0 = vec3<f32>(0.039, 0.055, 0.078); let c1 = vec3<f32>(0.106, 0.298, 0.451);',
      '  let c2 = vec3<f32>(0.435, 0.827, 0.933); let c3 = vec3<f32>(1.0, 0.616, 0.361);',
      '  let c4 = vec3<f32>(1.0, 0.98, 0.92);',
      '  if (u < 0.25) { return mix(c0, c1, u / 0.25); }',
      '  if (u < 0.5)  { return mix(c1, c2, (u - 0.25) / 0.25); }',
      '  if (u < 0.75) { return mix(c2, c3, (u - 0.5) / 0.25); }',
      '  return mix(c3, c4, (u - 0.75) / 0.25);',
      '}',
      '@fragment fn fs_particles(o : VOut) -> @location(0) vec4<f32> {',
      '  let r2 = dot(o.uv, o.uv);',
      '  if (r2 > 1.0) { discard; }',
      '  let t = clamp(o.sp / max(R.vRef, 1e-6), 0.0, 1.0);',
      '  let cool = vec3<f32>(0.30, 0.71, 1.0); let warm = vec3<f32>(1.0, 0.98, 0.93);',
      '  let tintc = vec3<f32>(1.0, 0.62, 0.36);',
      '  var base = mix(cool, warm, t);',
      '  base = mix(base, mix(tintc, warm, t), R.tint);',
      '  return vec4<f32>(base * (0.55 + 0.45 * sqrt(max(0.0, 1.0 - r2))), 1.0);',
      '}',
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

  // Untrained weights of an arbitrary width, for cost-only measurements. Values are kept O(0.1) so
  // no denormals or infinities appear -- both can change arithmetic throughput on real hardware and
  // would contaminate a timing.
  function synthNet(h) {
    var k = netSize(h), f = new Float32Array(k), s = 12345;
    for (var i = 0; i < k; i++) {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      f[i] = (s / 4294967296 - 0.5) * 0.4;
    }
    return { hidden: h, w: f };
  }

  // ------------------------------------------------------------------ device
  function supported() { return (typeof navigator !== 'undefined') && !!navigator.gpu; }
  var _device = null, _info = null, _ts = false, _limits = null, ERRORS = [];

  async function getDevice() {
    if (_device) return _device;
    if (!supported()) throw new Error('navigator.gpu is undefined (insecure origin?)');
    var adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('requestAdapter returned null');
    _info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    var feats = [];
    _ts = adapter.features.has('timestamp-query');
    if (_ts) feats.push('timestamp-query');
    _device = await adapter.requestDevice({ requiredFeatures: feats });
    _limits = {};
    ['maxStorageBuffersPerShaderStage', 'maxComputeInvocationsPerWorkgroup',
      'maxComputeWorkgroupStorageSize', 'maxBufferSize'].forEach(function (k) {
        _limits[k] = _device.limits[k];
      });
    // WebGPU errors are asynchronous and, unlistened, silent. An invalid bind group turns every
    // dispatch into a no-op, which reads as a spectacular timing result over all-zero data.
    _device.addEventListener('uncapturederror', function (e) {
      ERRORS.push(String((e.error && e.error.message) || e.error));
      console.error('WebGPU uncaptured error:', e.error);
    });
    _device.lost.then(function (e) { ERRORS.push('DEVICE LOST: ' + e.message); });
    return _device;
  }
  function deviceInfo() {
    var a = _info || {};
    return { vendor: a.vendor || '', architecture: a.architecture || '', device: a.device || '',
      description: a.description || '', timestampQuery: _ts, limits: _limits };
  }
  function errors() { return ERRORS.slice(); }

  function seedDisk(cx, cy, r, n, seed) {
    var s = (seed >>> 0) || 1;
    function rnd() { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }
    var p = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) {
      var a = rnd() * Math.PI * 2, rr = r * Math.sqrt(rnd());
      p[2 * i] = cx + rr * Math.cos(a); p[2 * i + 1] = cy + rr * Math.sin(a);
    }
    return p;
  }
  // A box at a fixed particle DENSITY, capped at the domain. Growing a disk as sqrt(n) leaves the
  // box past ~35k particles and everything outside gets clamped onto the boundary cells, which turns
  // a scaling curve into a measurement of an atomic pile-up.
  var DENSITY0 = 2048 / (Math.PI * 0.11 * 0.11);
  function boxScene(n, seed) {
    var side = Math.min(0.90, Math.sqrt(n / DENSITY0));
    var x0 = 0.5 - side / 2, y0 = 0.5 - side / 2;
    var s = (seed >>> 0) || 1;
    function rnd() { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }
    var p = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) { p[2 * i] = x0 + rnd() * side; p[2 * i + 1] = y0 + rnd() * side; }
    return { pts: p, area: side * side, side: side,
      particles_per_cell: n / (side * side * N_CELL) };
  }

  // ------------------------------------------------------------------ the simulator
  async function createSim(opts) {
    opts = opts || {};
    var device = await getDevice();
    var n = opts.n | 0;
    var area = opts.area;
    var dt = opts.dt !== undefined ? opts.dt : P.dt;
    var E = opts.E !== undefined ? opts.E : P.E;
    var netKey = opts.net || 'point16';
    // `synth` builds an UNTRAINED network of an arbitrary width. Cost does not depend on the values
    // in the weight buffer, so this is how the width axis is scanned finely enough to locate a
    // performance cliff without training a network at every point on it.
    var net = (opts.synth ? synthNet(opts.synth) : decodeNet(netKey));
    if (opts.synth) netKey = 'synth' + opts.synth;
    var H = net.hidden;
    var kM = opts.kM !== undefined ? opts.kM : P.kM;
    var kV = opts.kV !== undefined ? opts.kV : P.kV;

    var pVol = area / n;
    var pMass = pVol * P.rho;
    var massScale = Math.pow(2, kM) / pMass;
    var momScale = Math.pow(2, kV) / pMass;

    device.pushErrorScope('validation');
    var U = GPUBufferUsage, STO = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    function buf(b, u) { return device.createBuffer({ size: b, usage: u }); }
    var posBuf = buf(n * 8, STO), pstBuf = buf(n * 16, STO), CBuf = buf(n * 16, STO);
    var gmBuf = buf(N_CELL * 4, STO), gpBuf = buf(N_CELL * 8, STO), gvBuf = buf(N_CELL * 16, STO);
    var wBuf = buf(Math.max(16, net.w.byteLength), STO);
    var uBuf = buf(64, U.UNIFORM | U.COPY_DST);
    var readBuf = device.createBuffer({ size: n * 8, usage: U.COPY_DST | U.MAP_READ });
    var gridRead = device.createBuffer({ size: N_CELL * 16, usage: U.COPY_DST | U.MAP_READ });
    device.queue.writeBuffer(wBuf, 0, net.w);

    function sb(b, ro) {
      return { binding: b, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: ro ? 'read-only-storage' : 'storage' } };
    }
    var layout = device.createBindGroupLayout({
      label: 'mpmnn-bgl',
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        sb(1), sb(2), sb(3), sb(4), sb(5), sb(6), sb(7, true)]
    });
    var bind = device.createBindGroup({
      label: 'mpmnn-bg', layout: layout,
      entries: [uBuf, posBuf, pstBuf, CBuf, gmBuf, gpBuf, gvBuf, wBuf]
        .map(function (b, i) { return { binding: i, resource: { buffer: b } }; })
    });
    var pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });

    // One shader module per grid kernel; the MLP loops are compile-time sized, so a width change
    // is a different module rather than a uniform branch.
    var mods = {}, pipes = {};
    var KINDS = ['analytic', 'nn', 'nnsparse', 'null'];
    for (var ki = 0; ki < KINDS.length; ki++) {
      var kind = KINDS[ki];
      var mod = device.createShaderModule({ code: buildShader(kind, H), label: 'mpmnn-' + kind });
      var ci = await mod.getCompilationInfo();
      var errs = ci.messages.filter(function (m) { return m.type === 'error'; });
      if (errs.length) {
        throw new Error('WGSL[' + kind + ']: ' + errs.map(function (m) {
          return m.lineNum + ':' + m.message; }).join(' | '));
      }
      mods[kind] = mod;
      pipes[kind] = {};
      ['clear_grid', 'p2g', 'grid_op', 'g2p', 'empty'].forEach(function (e) {
        pipes[kind][e] = device.createComputePipeline({
          layout: pl, label: kind + ':' + e, compute: { module: mod, entryPoint: e } });
      });
    }
    var setupError = await device.popErrorScope();
    if (setupError) throw new Error('WebGPU setup: ' + setupError.message);

    var P_WG = Math.ceil(n / WG_P);
    var ua = new ArrayBuffer(64), uf = new Float32Array(ua), uu = new Uint32Array(ua);
    function writeUniform() {
      uf[0] = dt; uf[1] = pMass; uf[2] = 1.0 / pMass; uf[3] = pVol; uf[4] = E;
      uf[5] = P.gravity; uf[6] = P.fric;
      uf[7] = massScale; uf[8] = 1 / massScale; uf[9] = momScale; uf[10] = 1 / momScale;
      uu[11] = n >>> 0;
      device.queue.writeBuffer(uBuf, 0, ua);
    }
    writeUniform();

    var qset = null, qRes = null, qRead = null;
    if (_ts) {
      qset = device.createQuerySet({ type: 'timestamp', count: 2 });
      qRes = buf(16, U.QUERY_RESOLVE | U.COPY_SRC);
      qRead = device.createBuffer({ size: 16, usage: U.COPY_DST | U.MAP_READ });
    }

    function seed(pts, v0x, v0y) {
      var xs = new Float32Array(2 * n), st = new Float32Array(4 * n);
      for (var p = 0; p < n; p++) {
        xs[2 * p] = pts[2 * p]; xs[2 * p + 1] = pts[2 * p + 1];
        st[4 * p] = v0x || 0; st[4 * p + 1] = v0y || 0; st[4 * p + 2] = 1.0;   // J = 1
      }
      device.queue.writeBuffer(posBuf, 0, xs);
      device.queue.writeBuffer(pstBuf, 0, st);
      device.queue.writeBuffer(CBuf, 0, new Float32Array(4 * n));
      device.queue.writeBuffer(gmBuf, 0, new Uint32Array(N_CELL));
      device.queue.writeBuffer(gpBuf, 0, new Int32Array(2 * N_CELL));
      device.queue.writeBuffer(gvBuf, 0, new Float32Array(4 * N_CELL));
    }

    // ONE COMMAND BUFFER PER FRAME. Every substep's dispatches are recorded into a single compute
    // pass; WebGPU orders dispatches inside a pass and makes each one's writes visible to the next,
    // which is exactly the P2G -> grid -> G2P chain, so no barriers and no CPU round trip.
    function encodeFrame(substeps, opt) {
      opt = opt || {};
      var kind = opt.grid || 'analytic';
      var pp = pipes[kind];
      var enc = device.createCommandEncoder();
      var desc = {};
      if (opt.timed && qset) {
        desc.timestampWrites = { querySet: qset, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
      }
      var pass = enc.beginComputePass(desc);
      pass.setBindGroup(0, bind);
      if (opt.clearFirst) { pass.setPipeline(pp.clear_grid); pass.dispatchWorkgroups(GRID_WG); }
      var ph = opt.phases || 'pgG';
      var doP = ph.indexOf('p') >= 0, doG = ph.indexOf('g') >= 0,
        doQ = ph.indexOf('G') >= 0, doE = ph.indexOf('e') >= 0;
      // gridWG prices a COMPACTED grid dispatch: the same kernel over the number of workgroups an
      // occupied-cell list would actually need, instead of all 256. It is a cost proxy -- it does
      // not build the list, and the cells it happens to touch are not the occupied ones -- so it
      // prices the dispatch and not the compaction pass, which is priced separately.
      var gwg = opt.gridWG || GRID_WG;
      for (var s = 0; s < substeps; s++) {
        if (doE) { pass.setPipeline(pp.empty); pass.dispatchWorkgroups(GRID_WG); }
        if (doP) { pass.setPipeline(pp.p2g); pass.dispatchWorkgroups(P_WG); }
        if (doG) { pass.setPipeline(pp.grid_op); pass.dispatchWorkgroups(gwg); }
        if (doQ) { pass.setPipeline(pp.g2p); pass.dispatchWorkgroups(P_WG); }
      }
      pass.end();
      if (opt.timed && qset) {
        enc.resolveQuerySet(qset, 0, 2, qRes, 0);
        enc.copyBufferToBuffer(qRes, 0, qRead, 0, 16);
      }
      if (opt.readback) enc.copyBufferToBuffer(posBuf, 0, readBuf, 0, n * 8);
      if (opt.gridReadback) enc.copyBufferToBuffer(gvBuf, 0, gridRead, 0, N_CELL * 16);
      var cmd = enc.finish();
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
      var o = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      return o;
    }
    async function readGrid() {
      await gridRead.mapAsync(GPUMapMode.READ);
      var o = new Float32Array(gridRead.getMappedRange().slice(0));
      gridRead.unmap();
      return o;
    }

    return {
      n: n, device: device, hidden: H, netKey: netKey,
      params: { dt: dt, E: E, pVol: pVol, pMass: pMass, n_grid: N_GRID, gravity: P.gravity,
        friction: P.fric, kM: kM, kV: kV, hidden: H, net: netKey,
        net_floats: net.w.length, physics_version: P.physics_version },
      buffers: { pos: posBuf, pst: pstBuf, gv: gvBuf },
      seed: seed, encodeFrame: encodeFrame, lastGpuNanos: lastGpuNanos,
      readPositions: readPositions, readGrid: readGrid,
      idle: function () { return device.queue.onSubmittedWorkDone(); },
      shaderFor: function (k) { return buildShader(k, H); },
      destroy: function () {
        [posBuf, pstBuf, CBuf, gmBuf, gpBuf, gvBuf, wBuf, uBuf, readBuf, gridRead]
          .forEach(function (b) { b.destroy(); });
      }
    };
  }

  // ------------------------------------------------------------------ renderer
  async function createRenderer(canvas, sim) {
    var device = sim.device;
    var ctx = canvas.getContext('webgpu');
    var fmt = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device: device, format: fmt, alphaMode: 'opaque' });
    device.pushErrorScope('validation');
    var mod = device.createShaderModule({ code: renderShader(), label: 'mpmnn-render' });
    var ci = await mod.getCompilationInfo();
    var re = ci.messages.filter(function (m) { return m.type === 'error'; });
    if (re.length) throw new Error('render WGSL: ' + re.map(function (m) { return m.lineNum + ':' + m.message; }).join(' | '));
    var rBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    var layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }]
    });
    var bind = device.createBindGroup({
      layout: layout,
      entries: [{ binding: 0, resource: { buffer: rBuf } },
        { binding: 1, resource: { buffer: sim.buffers.pos } },
        { binding: 2, resource: { buffer: sim.buffers.pst } },
        { binding: 3, resource: { buffer: sim.buffers.gv } }]
    });
    var pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    var pPart = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_particles' },
      fragment: { module: mod, entryPoint: 'fs_particles', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' }
    });
    var pGrid = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_full' },
      fragment: { module: mod, entryPoint: 'fs_grid', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' }
    });
    var err = await device.popErrorScope();
    if (err) throw new Error('renderer setup: ' + err.message);
    var ru = new Float32Array(8), ruU = new Uint32Array(ru.buffer);
    return {
      draw: function (o) {
        o = o || {};
        var view = o.view || 'particles';
        var enc = device.createCommandEncoder();
        var pass = enc.beginRenderPass({
          colorAttachments: [{ view: ctx.getCurrentTexture().createView(),
            clearValue: { r: 0.039, g: 0.055, b: 0.078, a: 1 }, loadOp: 'clear', storeOp: 'store' }]
        });
        ru[0] = o.radius !== undefined ? o.radius : 0.010;
        ru[1] = canvas.width / canvas.height;
        ruU[2] = sim.n >>> 0;
        ruU[3] = view === 'mass' ? 1 : (view === 'speed' ? 2 : 0);
        ru[4] = o.vRef !== undefined ? o.vRef : 3.0;
        ru[5] = o.massRef !== undefined ? o.massRef : 24.0;
        ru[6] = o.tint !== undefined ? o.tint : 0.0;
        device.queue.writeBuffer(rBuf, 0, ru);
        pass.setBindGroup(0, bind);
        if (view !== 'particles') { pass.setPipeline(pGrid); pass.draw(6, 1); }
        else { pass.setPipeline(pPart); pass.draw(6, sim.n); }
        pass.end();
        device.queue.submit([enc.finish()]);
      }
    };
  }

  return {
    supported: supported, getDevice: getDevice, deviceInfo: deviceInfo, errors: errors,
    createSim: createSim, createRenderer: createRenderer,
    seedDisk: seedDisk, boxScene: boxScene, decodeNet: decodeNet, netSize: netSize,
    buildShader: buildShader, PARAMS: P, NETS: NN, N_GRID: N_GRID, N_CELL: N_CELL,
    WG_P: WG_P, WG_G: WG_G, GRID_WG: GRID_WG
  };
});
