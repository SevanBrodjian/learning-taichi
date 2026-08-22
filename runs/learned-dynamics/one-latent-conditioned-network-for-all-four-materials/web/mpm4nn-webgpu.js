// mpm4nn-webgpu.js -- four canonical materials on one WebGPU grid, with the per-particle
// CONSTITUTIVE MODEL replaced by ONE latent-conditioned MLP.
//
// Forked from the analytic four-material port
// (runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/web/mpm4-webgpu.js).
// The P2G/G2P transfer, the grid update, the SVD and the analytic constitutive branches are carried
// over unchanged so that `mode:'analytic'` is a like-for-like BASELINE measured by the same harness,
// in the same pass, on the same buffers, as `mode:'nn'`.
//
// =================================================================================================
// WHAT CHANGED, AND WHY EACH CHANGE IS THE POINT
// =================================================================================================
//
// 1. THE SEAM IS FUSED INTO G2P, NOT ADDED AS A DISPATCH. The network is evaluated at the bottom of
//    G2P and its stress is cached in a new buffer for the next P2G to scatter. Dispatches per
//    substep therefore stay at THREE, exactly as analytic. This matters more than it sounds:
//    T-022 replaced the GRID update with its own dispatch and found the cost was LATENCY-bound --
//    the same kernel over 7 workgroups instead of 256 cost the same. A fused seam pays no dispatch
//    at all, so what this file measures is arithmetic, not launch overhead.
//
// 2. ONE MORE STORAGE BUFFER. pos, vel, C, F, gm, gp, gv was seven; the cached stress makes eight,
//    which is exactly the WebGPU guaranteed minimum (maxStorageBuffersPerShaderStage = 8). Nine
//    silently invalidates the bind group and every dispatch becomes a no-op over zeros -- the
//    beautiful-flat-curve failure this project has already paid for once. So the WEIGHTS live in a
//    UNIFORM buffer, which is a different budget (maxUniformBuffersPerShaderStage >= 12) and is
//    also the right place for them: every lane in a warp reads the same weight address, which is
//    what a constant bank is for. A storage-buffer variant is built too, but only when the adapter
//    reports headroom for a ninth storage binding, so the A/B can be measured rather than assumed.
//
// 3. THE WEIGHT LAYOUT IS INTERLEAVED PER HIDDEN UNIT, AND THE HIDDEN VECTOR IS NEVER MATERIALISED.
//    The obvious MLP kernel writes h[0..H) to a local array and then reads it back for the output
//    layer; a dynamically indexed local array of 128 floats is exactly what spills to scratch
//    memory, and register spilling is the leading hypothesis for T-022's width cliff. Instead the
//    output accumulators (8 floats, two vec4s) are held in registers and each hidden unit's
//    contribution is added as it is computed:
//
//        for k in 0..H:  h = tanh(dot(W1_k, x));  oA += h * W2a_k;  oB += h * W2b_k;
//
//    so live state is 4 vec4 of input + 2 vec4 of output, whatever H is. Each hidden unit's six
//    vec4s (four of W1 including its bias, two of its output column) are contiguous.
//
// 4. THE BIAS RIDES IN THE INPUT PADDING. 14 features pad to 16 for the vec4 packing; slot 14 is
//    pinned to 1.0 so W1's 15th column IS b1. Free, and it removes a separate bias fetch.
//
// The seam's contract (identical to the Taichi trainer -- see train/netspec.py):
//    in  (14): S00,S01,S11 (polar stretch of the trial F), C00,C01,C10,C11, vx,vy, Jp, z0..z3
//    out  (7): tau00,tau01,tau11 (material-frame stress), dS00,dS01,dS11 (plastic correction), dJp
// F is remounted as R (S + dS); the stress is rotated back as R tau R^T. Analytic and untouched:
// the B-spline transfer, the grid update, gravity, the walls, advection.

(function (root, factory) {
  var api = factory(root.MPM_PARAMS);
  root.MPM4NN = api;
  if (typeof module === 'object' && module.exports) { module.exports = api; }
})(typeof self !== 'undefined' ? self : this, function (P) {
  'use strict';

  var N_GRID = P.n_grid | 0;
  var N_CELL = N_GRID * N_GRID;
  var WG_P = 64;
  var WG_G = 64;
  var GRID_WG = Math.ceil(N_CELL / WG_G);

  var MAT = P.materials;
  var ORDER = P.mat_order;
  var ID = P.mat_id;

  // --- network shape, mirrored from train/netspec.py (kept in sync by verify/prepare.py) ---
  var N_FEAT = 10, Z_DIM = 4, N_IN = 14, N_IN_PAD = 16, N_OUT = 7, N_OUT_PAD = 8;
  var VEC_PER_UNIT = (N_IN_PAD / 4) + (N_OUT_PAD / 4);          // 4 + 2 = 6
  function weightVecCount(H) { return VEC_PER_UNIT * H + (N_OUT_PAD / 4); }
  function weightFloatCount(H) { return 4 * weightVecCount(H); }

  function f(x) { var s = String(x); if (!/[.eE]/.test(s)) s += '.0'; return s; }

  // -------------------------------------------------------------------------------- SVD (verbatim)
  var SVD_WGSL = [
    'struct Svd2 { u : vec4<f32>, s : vec2<f32>, v : vec4<f32> };',
    'fn mul_usv(u : vec4<f32>, s : vec2<f32>, v : vec4<f32>) -> vec4<f32> {',
    '  let a = u.x * s.x; let b = u.y * s.y;',
    '  let c = u.z * s.x; let d = u.w * s.y;',
    '  return vec4<f32>(a * v.x + b * v.z, a * v.y + b * v.w,',
    '                   c * v.x + d * v.z, c * v.y + d * v.w);',
    '}',
    'fn svd2(a : vec4<f32>) -> Svd2 {',
    '  let a00 = a.x; let a01 = a.y; let a10 = a.z; let a11 = a.w;',
    '  var r = vec4<f32>(1.0, 0.0, 0.0, 1.0);',
    '  var p = a;',
    '  if (!(a00 == 0.0 && a01 == 0.0 && a10 == 0.0 && a11 == 0.0)) {',
    '    let detA = a00 * a11 - a10 * a01;',
    '    let adetA = abs(detA);',
    '    var b = vec4<f32>(a00 + a11, a01 - a10, a10 - a01, a11 + a00);',
    '    if (detA < 0.0) { b = vec4<f32>(a00 - a11, a01 + a10, a10 + a01, a11 - a00); }',
    '    let adetB = abs(b.x * b.w - b.z * b.y);',
    '    let k = 1.0 / max(sqrt(adetB), 1e-30);',
    '    r = b * k;',
    '    let t00 = a00 * a00 + a10 * a10;',
    '    let t01 = a00 * a01 + a10 * a11;',
    '    let t11 = a01 * a01 + a11 * a11;',
    '    p = vec4<f32>((t00 + adetA) * k, t01 * k, t01 * k, (t11 + adetA) * k);',
    '  }',
    '  var c = 1.0; var s = 0.0;',
    '  var s1 = p.x; var s2 = p.w;',
    '  if (abs(p.y) >= 1e-5) {',
    '    let tao = 0.5 * (p.x - p.w);',
    '    let w = sqrt(tao * tao + p.y * p.y);',
    '    var t = 0.0;',
    '    if (tao > 0.0) { t = p.y / (tao + w); } else { t = p.y / (tao - w); }',
    '    c = 1.0 / sqrt(t * t + 1.0);',
    '    s = -t * c;',
    '    s1 = c * c * p.x - 2.0 * c * s * p.y + s * s * p.w;',
    '    s2 = s * s * p.x + 2.0 * c * s * p.y + c * c * p.w;',
    '  }',
    '  var v = vec4<f32>(c, s, -s, c);',
    '  if (s1 < s2) { let tmp = s1; s1 = s2; s2 = tmp; v = vec4<f32>(-s, c, -c, -s); }',
    '  var o : Svd2;',
    '  o.u = vec4<f32>(r.x * v.x + r.y * v.z, r.x * v.y + r.y * v.w,',
    '                  r.z * v.x + r.w * v.z, r.z * v.y + r.w * v.w);',
    '  o.s = vec2<f32>(s1, s2);',
    '  o.v = v;',
    '  return o;',
    '}',
    'fn polar_r(a : vec4<f32>) -> vec4<f32> {',
    '  let a00 = a.x; let a01 = a.y; let a10 = a.z; let a11 = a.w;',
    '  if (a00 == 0.0 && a01 == 0.0 && a10 == 0.0 && a11 == 0.0) { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }',
    '  let detA = a00 * a11 - a10 * a01;',
    '  var b = vec4<f32>(a00 + a11, a01 - a10, a10 - a01, a11 + a00);',
    '  if (detA < 0.0) { b = vec4<f32>(a00 - a11, a01 + a10, a10 + a01, a11 - a00); }',
    '  let adetB = abs(b.x * b.w - b.z * b.y);',
    '  return b * (1.0 / max(sqrt(adetB), 1e-30));',
    '}'
  ].join('\n');

  // ------------------------------------------------------------------------------- shader builder
  // opt = { mode:'analytic'|'nn', hidden, f16:bool, weights:'uniform'|'storage' }
  function buildShader(opt) {
    var nn = opt.mode === 'nn';
    var H = opt.hidden | 0;
    var useF16 = !!opt.f16;
    var wStorage = opt.weights === 'storage';
    var NV = weightVecCount(H);
    var Mf = MAT.fluid, Me = MAT.elastic, Ms = MAT.snow, Ma = MAT.sand;
    var L = [];
    if (useF16) L.push('enable f16;');
    L.push('// GENERATED from params.js (physics_version ' + P.physics_version + ')  mode=' + opt.mode +
      (nn ? ('  hidden=' + H + (useF16 ? '  f16' : '  f32') + '  weights=' + (wStorage ? 'storage' : 'uniform')) : ''));
    L.push('const N_GRID : i32 = ' + N_GRID + ';');
    L.push('const N_CELL : i32 = ' + N_CELL + ';');
    L.push('const DX : f32 = ' + f(P.dx) + ';');
    L.push('const INV_DX : f32 = ' + f(P.inv_dx) + ';');
    L.push('const BOUND : i32 = ' + P.bound + ';');
    L.push('const FLOOR : f32 = ' + f(P.floor_y) + ';');
    L.push('const CEIL : f32 = ' + f(1.0 - P.floor_y) + ';');
    L.push('const M_FLUID : i32 = ' + ID.fluid + ';  const M_ELASTIC : i32 = ' + ID.elastic + ';');
    L.push('const M_SNOW : i32 = ' + ID.snow + ';   const M_SAND : i32 = ' + ID.sand + ';');
    L.push('const E_FLUID : f32 = ' + f(Mf.E) + ';');
    L.push('const MU_E : f32 = ' + f(Me.mu) + ';   const LA_E : f32 = ' + f(Me.la) + ';');
    L.push('const MU_S : f32 = ' + f(Ms.mu) + ';   const LA_S : f32 = ' + f(Ms.la) + ';');
    L.push('const XI_S : f32 = ' + f(Ms.xi) + ';   const TC_S : f32 = ' + f(Ms.tc) + ';');
    L.push('const TS_S : f32 = ' + f(Ms.ts) + ';');
    L.push('const MU_A : f32 = ' + f(Ma.mu) + ';   const LA_A : f32 = ' + f(Ma.la) + ';');
    L.push('const ALPHA_A : f32 = ' + f(Ma.alpha) + ';');
    L.push(SVD_WGSL);
    L.push('struct Params {');
    L.push('  dt : f32, pMass : f32, pVol : f32, gravity : f32, friction : f32,');
    L.push('  massScale : f32, invMassScale : f32, momScale : f32, invMomScale : f32,');
    L.push('  n : u32, hloop : u32, pad1 : f32,');
    // PER-MATERIAL DENSITY AND FRICTION, straight from sim.physics.MAT. Both used to be one number
    // for the whole domain in the earlier WebGPU port, and both are physics, not decoration:
    // rho = 0.3 is what makes snow float and -- decisive here -- what fixes canonical snow's
    // E/rho = 150. Running snow at rho = 1 makes it a 3.3x softer material wearing snow's name, and
    // it showed up immediately as a WGSL snow drop spreading 0.28 where canonical spreads 0.46.
    // fric = 0 is what stops water gripping a smooth floor.
    L.push('  rho : vec4<f32>, fric : vec4<f32>,');
    L.push('};');
    L.push('@group(0) @binding(0) var<uniform> PR : Params;');
    L.push('@group(0) @binding(1) var<storage, read_write> pos : array<vec2<f32>>;');
    L.push('@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;');
    L.push('@group(0) @binding(3) var<storage, read_write> Cm  : array<vec4<f32>>;');
    L.push('@group(0) @binding(4) var<storage, read_write> Fm  : array<vec4<f32>>;');
    L.push('@group(0) @binding(5) var<storage, read_write> gm  : array<atomic<u32>>;');
    L.push('@group(0) @binding(6) var<storage, read_write> gp  : array<atomic<i32>>;');
    L.push('@group(0) @binding(7) var<storage, read_write> gv  : array<vec4<f32>>;');
    // st.xyz = cached WORLD-frame stress (P F^T, symmetric: s00, s01, s11). st.w unused.
    L.push('@group(0) @binding(8) var<storage, read_write> st  : array<vec4<f32>>;');
    if (nn) {
      var wty = useF16 ? 'vec4<f16>' : 'vec4<f32>';
      if (wStorage) {
        L.push('@group(0) @binding(9) var<storage, read> WT : array<' + wty + '>;');
      } else {
        L.push('@group(0) @binding(9) var<uniform> WT : array<' + wty + ', ' + NV + '>;');
      }
      // the four material codes, one per row -- a uniform because every particle of a material
      // reads the same four numbers
      L.push('@group(0) @binding(10) var<uniform> ZC : array<vec4<f32>, 4>;');
    }

    // ------------------------------------------------------------------ clear
    L.push('@compute @workgroup_size(' + WG_G + ')');
    L.push('fn clear_grid(@builtin(global_invocation_id) gid : vec3<u32>) {');
    L.push('  let idx = i32(gid.x);');
    L.push('  if (idx >= N_CELL) { return; }');
    L.push('  atomicStore(&gm[idx], 0u);');
    L.push('  atomicStore(&gp[3 * idx], 0);');
    L.push('  atomicStore(&gp[3 * idx + 1], 0);');
    L.push('  atomicStore(&gp[3 * idx + 2], 0);');
    L.push('  gv[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);');
    L.push('}');

    // ------------------------------------------------------------------ analytic constitutive
    L.push('fn analytic_stress(mid : i32, Fp : vec4<f32>, Jp : f32) -> vec3<f32> {');
    L.push('  if (mid == M_FLUID) {');
    L.push('    let pres = E_FLUID * (Fp.x * Fp.w - Fp.y * Fp.z - 1.0);');
    L.push('    return vec3<f32>(pres, 0.0, pres);');
    L.push('  } else if (mid == M_SAND) {');
    L.push('    let sv = svd2(Fp);');
    L.push('    let e0 = log(max(sv.s.x, 1e-4));');
    L.push('    let e1 = log(max(sv.s.y, 1e-4));');
    L.push('    let tr = e0 + e1;');
    L.push('    let t0 = 2.0 * MU_A * e0 + LA_A * tr;');
    L.push('    let t1 = 2.0 * MU_A * e1 + LA_A * tr;');
    L.push('    let u = sv.u;');
    L.push('    return vec3<f32>(u.x * t0 * u.x + u.y * t1 * u.y,');
    L.push('                     u.x * t0 * u.z + u.y * t1 * u.w,');
    L.push('                     u.z * t0 * u.z + u.w * t1 * u.w);');
    L.push('  }');
    L.push('  var mu = MU_E; var la = LA_E;');
    L.push('  if (mid == M_SNOW) { let h = exp(XI_S * (1.0 - Jp)); mu = MU_S * h; la = LA_S * h; }');
    L.push('  let r = polar_r(Fp);');
    L.push('  let a00 = Fp.x - r.x; let a01 = Fp.y - r.y;');
    L.push('  let a10 = Fp.z - r.z; let a11 = Fp.w - r.w;');
    L.push('  let b00 = a00 * Fp.x + a01 * Fp.y; let b01 = a00 * Fp.z + a01 * Fp.w;');
    L.push('  let b11 = a10 * Fp.z + a11 * Fp.w;');
    L.push('  let Jd = Fp.x * Fp.w - Fp.y * Fp.z;');
    L.push('  let lt = la * (Jd - 1.0) * Jd;');
    L.push('  return vec3<f32>(2.0 * mu * b00 + lt, 2.0 * mu * b01, 2.0 * mu * b11 + lt);');
    L.push('}');

    // plastic update: returns the new F in .xyzw and the new Jp packed via an out-param struct
    L.push('struct Plastic { F : vec4<f32>, Jp : f32 };');
    L.push('fn analytic_plastic(mid : i32, tr : vec4<f32>, Jp : f32, dtrC : f32, dG : f32) -> Plastic {');
    L.push('  var o : Plastic;');
    L.push('  o.F = tr; o.Jp = Jp;');
    L.push('  if (mid == M_FLUID) {');
    // canonical advects the scalar J by J *= 1 + dt tr(C); det(I + dt C) is the multiplicative
    // version, so the ratio is the exact correction and the fluid's F stays isotropic
    L.push('    let Jtr = tr.x * tr.w - tr.y * tr.z;');
    L.push('    let s = sqrt(max(Jtr * (1.0 + dtrC) / max(dG, 1e-12), 1e-8));');
    L.push('    o.F = vec4<f32>(s, 0.0, 0.0, s);');
    L.push('  } else if (mid == M_SNOW) {');
    L.push('    let sv = svd2(tr);');
    L.push('    let s0 = min(max(sv.s.x, 1.0 - TC_S), 1.0 + TS_S);');
    L.push('    let s1 = min(max(sv.s.y, 1.0 - TC_S), 1.0 + TS_S);');
    L.push('    o.Jp = Jp * (sv.s.x * sv.s.y) / (s0 * s1);');
    L.push('    o.F = mul_usv(sv.u, vec2<f32>(s0, s1), sv.v);');
    L.push('  } else if (mid == M_SAND) {');
    L.push('    let sv = svd2(tr);');
    L.push('    let e0 = log(max(abs(sv.s.x), 1e-4));');
    L.push('    let e1 = log(max(abs(sv.s.y), 1e-4));');
    L.push('    let trE = e0 + e1 + Jp;');
    L.push('    var q0 = 1.0; var q1 = 1.0;');
    L.push('    if (trE >= 0.0) { o.Jp = trE; }');
    L.push('    else {');
    L.push('      o.Jp = 0.0;');
    L.push('      let eh0 = e0 - trE * 0.5;');
    L.push('      let eh1 = e1 - trE * 0.5;');
    L.push('      let ehn = sqrt(eh0 * eh0 + eh1 * eh1) + 1e-20;');
    L.push('      let dg = ehn + (2.0 * LA_A + 2.0 * MU_A) / (2.0 * MU_A) * trE * ALPHA_A;');
    L.push('      if (dg <= 0.0) { q0 = sv.s.x; q1 = sv.s.y; }');
    L.push('      else { q0 = exp(e0 - dg / ehn * eh0); q1 = exp(e1 - dg / ehn * eh1); }');
    L.push('    }');
    L.push('    o.F = mul_usv(sv.u, vec2<f32>(q0, q1), sv.v);');
    L.push('  }');
    L.push('  return o;');
    L.push('}');

    // ------------------------------------------------------------------ the MLP
    if (nn) {
      var T = useF16 ? 'f16' : 'f32';
      var V = useF16 ? 'vec4<f16>' : 'vec4<f32>';
      var cast = useF16 ? function (s) { return 'vec4<f16>(' + s + ')'; } : function (s) { return s; };
      L.push('struct Net { tau : vec3<f32>, dS : vec3<f32>, dJp : f32 };');
      L.push('fn mlp(x0 : vec4<f32>, x1 : vec4<f32>, x2 : vec4<f32>, x3 : vec4<f32>) -> Net {');
      L.push('  let y0 = ' + cast('x0') + '; let y1 = ' + cast('x1') + ';');
      L.push('  let y2 = ' + cast('x2') + '; let y3 = ' + cast('x3') + ';');
      L.push('  let NV_ : u32 = ' + (VEC_PER_UNIT * H) + 'u;');
      L.push('  var oA = WT[NV_]; var oB = WT[NV_ + 1u];');          // the output biases
      // THE LOOP BOUND IS THE EXPERIMENT. With a literal, the compiler is free to fully unroll H
      // iterations, and a fully unrolled 96-deep body is exactly the sort of thing that stops
      // fitting in the instruction cache or the register file. With `PR.hloop` -- a value the
      // compiler cannot see -- unrolling is impossible. Running both tells a cost CLIFF caused by
      // the compiler apart from one caused by the memory system, which is the difference between
      // "networks get expensive above width 88" and "this shader was compiled badly above width 88".
      L.push('  for (var k : u32 = 0u; k < ' +
        (opt.dyn ? 'PR.hloop' : (H + 'u')) + '; k = k + 1u) {');
      L.push('    let b = k * ' + VEC_PER_UNIT + 'u;');
      L.push('    let a = dot(WT[b], y0) + dot(WT[b + 1u], y1) + dot(WT[b + 2u], y2) + dot(WT[b + 3u], y3);');
      L.push('    let h = tanh(a);');
      L.push('    oA = oA + h * WT[b + 4u];');
      L.push('    oB = oB + h * WT[b + 5u];');
      L.push('  }');
      L.push('  var o : Net;');
      L.push('  o.tau = vec3<f32>(f32(oA.x), f32(oA.y), f32(oA.z));');
      L.push('  o.dS  = vec3<f32>(f32(oA.w), f32(oB.x), f32(oB.y));');
      L.push('  o.dJp = f32(oB.z);');
      L.push('  return o;');
      L.push('}');
      // features -> the four packed input vectors. Slot 14 is pinned to 1.0 so W1's 15th column is
      // the hidden bias; slot 15 is dead padding kept so the packing is a clean 4 x vec4.
      L.push('fn packx(S : vec3<f32>, C : vec4<f32>, v : vec2<f32>, Jp : f32, mid : i32)');
      L.push('    -> array<vec4<f32>, 4> {');
      L.push('  let z = ZC[mid];');
      L.push('  return array<vec4<f32>, 4>(');
      L.push('    vec4<f32>(S.x, S.y, S.z, C.x),');
      L.push('    vec4<f32>(C.y, C.z, C.w, v.x),');
      L.push('    vec4<f32>(v.y, Jp, z.x, z.y),');
      L.push('    vec4<f32>(z.z, z.w, 1.0, 0.0));');
      L.push('}');
    }

    // ------------------------------------------------------------------ P2G (reads cached stress)
    L.push('@compute @workgroup_size(' + WG_P + ')');
    L.push('fn p2g(@builtin(global_invocation_id) gid : vec3<u32>) {');
    L.push('  let p = gid.x;');
    L.push('  if (p >= PR.n) { return; }');
    L.push('  let vp = vel[p];');
    L.push('  let mid = i32(round(vp.w));');
    L.push('  if (mid > M_SAND) { return; }');
    L.push('  let Xp = pos[p] * INV_DX;');
    L.push('  let base = vec2<i32>(Xp - vec2<f32>(0.5, 0.5));');
    L.push('  let fx = Xp - vec2<f32>(base);');
    L.push('  let w0 = 0.5 * (vec2<f32>(1.5, 1.5) - fx) * (vec2<f32>(1.5, 1.5) - fx);');
    L.push('  let w1 = vec2<f32>(0.75, 0.75) - (fx - vec2<f32>(1.0, 1.0)) * (fx - vec2<f32>(1.0, 1.0));');
    L.push('  let w2 = 0.5 * (fx - vec2<f32>(0.5, 0.5)) * (fx - vec2<f32>(0.5, 0.5));');
    L.push('  let kk = -PR.dt * 4.0 * PR.pVol * INV_DX * INV_DX;');
    L.push('  let s = st[p].xyz * kk;');
    L.push('  let pm = PR.pMass * PR.rho[mid];');
    L.push('  let fr = PR.fric[mid];');
    L.push('  let Cp = Cm[p];');
    L.push('  let af00 = s.x + pm * Cp.x; let af01 = s.y + pm * Cp.y;');
    L.push('  let af10 = s.y + pm * Cp.z; let af11 = s.z + pm * Cp.w;');
    L.push('  let mv = pm * vp.xy;');
    L.push('  for (var i = 0; i < 3; i = i + 1) {');
    L.push('    let wxi = select(select(w0.x, w1.x, i == 1), w2.x, i == 2);');
    L.push('    let dpx = (f32(i) - fx.x) * DX;');
    L.push('    for (var j = 0; j < 3; j = j + 1) {');
    L.push('      let wyj = select(select(w0.y, w1.y, j == 1), w2.y, j == 2);');
    L.push('      let dpy = (f32(j) - fx.y) * DX;');
    L.push('      let w = wxi * wyj;');
    L.push('      let gi = (base.x + i) * N_GRID + (base.y + j);');
    L.push('      let wm = w * pm;');
    L.push('      atomicAdd(&gm[gi], u32(round(wm * PR.massScale)));');
    L.push('      atomicAdd(&gp[3 * gi], i32(round(w * (mv.x + af00 * dpx + af01 * dpy) * PR.momScale)));');
    L.push('      atomicAdd(&gp[3 * gi + 1], i32(round(w * (mv.y + af10 * dpx + af11 * dpy) * PR.momScale)));');
    L.push('      atomicAdd(&gp[3 * gi + 2], i32(round(wm * fr * PR.massScale)));');
    L.push('    }');
    L.push('  }');
    L.push('}');

    // ------------------------------------------------------------------ grid op (verbatim + clear)
    L.push('@compute @workgroup_size(' + WG_G + ')');
    L.push('fn grid_op(@builtin(global_invocation_id) gid : vec3<u32>) {');
    L.push('  let idx = i32(gid.x);');
    L.push('  if (idx >= N_CELL) { return; }');
    L.push('  let m = f32(atomicLoad(&gm[idx])) * PR.invMassScale;');
    L.push('  let momx = f32(atomicLoad(&gp[3 * idx])) * PR.invMomScale;');
    L.push('  let momy = f32(atomicLoad(&gp[3 * idx + 1])) * PR.invMomScale;');
    L.push('  let mfr  = f32(atomicLoad(&gp[3 * idx + 2])) * PR.invMassScale;');
    L.push('  atomicStore(&gm[idx], 0u);');
    L.push('  atomicStore(&gp[3 * idx], 0);');
    L.push('  atomicStore(&gp[3 * idx + 1], 0);');
    L.push('  atomicStore(&gp[3 * idx + 2], 0);');
    L.push('  var vx = momx; var vy = momy;');
    L.push('  var fcoef = PR.friction;');
    L.push('  if (m > 0.0) { vx = vx / m; vy = vy / m; fcoef = mfr / m; }');
    L.push('  vy = vy - PR.dt * PR.gravity;');
    L.push('  let i = idx / N_GRID;');
    L.push('  let j = idx - i * N_GRID;');
    // All four boundaries get canonical core.grid_op's treatment: separating in the normal
    // direction, a Coulomb friction cap on the tangent, using the node's own mass-weighted
    // friction. The earlier port zeroed BOTH components at the side walls, which glues material to
    // them -- water thrown against a wall could not slide back down it.
    L.push('  if (j < BOUND && vy < 0.0) {');
    L.push('    let cap = fcoef * (-vy);');
    L.push('    if (vx > 0.0) { vx = max(0.0, vx - cap); } else if (vx < 0.0) { vx = min(0.0, vx + cap); }');
    L.push('    vy = 0.0;');
    L.push('  }');
    L.push('  if (j > N_GRID - BOUND && vy > 0.0) {');
    L.push('    let cap = fcoef * vy;');
    L.push('    if (vx > 0.0) { vx = max(0.0, vx - cap); } else if (vx < 0.0) { vx = min(0.0, vx + cap); }');
    L.push('    vy = 0.0;');
    L.push('  }');
    L.push('  if (i < BOUND && vx < 0.0) {');
    L.push('    let cap = fcoef * (-vx);');
    L.push('    if (vy > 0.0) { vy = max(0.0, vy - cap); } else if (vy < 0.0) { vy = min(0.0, vy + cap); }');
    L.push('    vx = 0.0;');
    L.push('  }');
    L.push('  if (i > N_GRID - BOUND && vx > 0.0) {');
    L.push('    let cap = fcoef * vx;');
    L.push('    if (vy > 0.0) { vy = max(0.0, vy - cap); } else if (vy < 0.0) { vy = min(0.0, vy + cap); }');
    L.push('    vx = 0.0;');
    L.push('  }');
    L.push('  let ipm = 1.0 / max(PR.pMass, 1e-30);');
    L.push('  gv[idx] = vec4<f32>(vx, vy, m * ipm, length(vec2<f32>(momx, momy)) * ipm);');
    L.push('}');

    // ------------------------------------------------------------------ G2P + the seam
    L.push('@compute @workgroup_size(' + WG_P + ')');
    L.push('fn g2p(@builtin(global_invocation_id) gid : vec3<u32>) {');
    L.push('  let p = gid.x;');
    L.push('  if (p >= PR.n) { return; }');
    L.push('  var vp = vel[p];');
    L.push('  let mid = i32(round(vp.w));');
    L.push('  if (mid > M_SAND) { return; }');
    L.push('  let Xp = pos[p] * INV_DX;');
    L.push('  let base = vec2<i32>(Xp - vec2<f32>(0.5, 0.5));');
    L.push('  let fx = Xp - vec2<f32>(base);');
    L.push('  let w0 = 0.5 * (vec2<f32>(1.5, 1.5) - fx) * (vec2<f32>(1.5, 1.5) - fx);');
    L.push('  let w1 = vec2<f32>(0.75, 0.75) - (fx - vec2<f32>(1.0, 1.0)) * (fx - vec2<f32>(1.0, 1.0));');
    L.push('  let w2 = 0.5 * (fx - vec2<f32>(0.5, 0.5)) * (fx - vec2<f32>(0.5, 0.5));');
    L.push('  var nv = vec2<f32>(0.0, 0.0);');
    L.push('  var c00 = 0.0; var c01 = 0.0; var c10 = 0.0; var c11 = 0.0;');
    L.push('  for (var i = 0; i < 3; i = i + 1) {');
    L.push('    let wxi = select(select(w0.x, w1.x, i == 1), w2.x, i == 2);');
    L.push('    let dpx = (f32(i) - fx.x) * DX;');
    L.push('    for (var j = 0; j < 3; j = j + 1) {');
    L.push('      let wyj = select(select(w0.y, w1.y, j == 1), w2.y, j == 2);');
    L.push('      let dpy = (f32(j) - fx.y) * DX;');
    L.push('      let w = wxi * wyj;');
    L.push('      let g = gv[(base.x + i) * N_GRID + (base.y + j)].xy;');
    L.push('      nv = nv + w * g;');
    L.push('      let sc = 4.0 * w * INV_DX * INV_DX;');
    L.push('      c00 = c00 + sc * g.x * dpx; c01 = c01 + sc * g.x * dpy;');
    L.push('      c10 = c10 + sc * g.y * dpx; c11 = c11 + sc * g.y * dpy;');
    L.push('    }');
    L.push('  }');
    L.push('  let np = pos[p] + PR.dt * nv;');
    L.push('  pos[p] = clamp(np, vec2<f32>(FLOOR, FLOOR), vec2<f32>(CEIL, CEIL));');
    L.push('  let Cn = vec4<f32>(c00, c01, c10, c11);');
    L.push('  let Fp = Fm[p];');
    L.push('  let g00 = 1.0 + PR.dt * c00; let g01 = PR.dt * c01;');
    L.push('  let g10 = PR.dt * c10;       let g11 = 1.0 + PR.dt * c11;');
    L.push('  let tr = vec4<f32>(g00 * Fp.x + g01 * Fp.z, g00 * Fp.y + g01 * Fp.w,');
    L.push('                     g10 * Fp.x + g11 * Fp.z, g10 * Fp.y + g11 * Fp.w);');
    if (nn) {
      L.push('  let R = polar_r(tr);');
      // S = R^T tr, symmetrised
      L.push('  let s00 = R.x * tr.x + R.z * tr.z;');
      L.push('  let s11 = R.y * tr.y + R.w * tr.w;');
      L.push('  let s01 = 0.5 * ((R.x * tr.y + R.z * tr.w) + (R.y * tr.x + R.w * tr.z));');
      L.push('  let xs = packx(vec3<f32>(s00, s01, s11), Cn, nv, vp.z, mid);');
      L.push('  let o = mlp(xs[0], xs[1], xs[2], xs[3]);');
      L.push('  let n00 = s00 + o.dS.x; let n01 = s01 + o.dS.y; let n11 = s11 + o.dS.z;');
      L.push('  Fm[p] = vec4<f32>(R.x * n00 + R.y * n01, R.x * n01 + R.y * n11,');
      L.push('                    R.z * n00 + R.w * n01, R.z * n01 + R.w * n11);');
      L.push('  vp.z = vp.z + o.dJp;');
      // R tau R^T, symmetric
      L.push('  let t00 = o.tau.x; let t01 = o.tau.y; let t11 = o.tau.z;');
      L.push('  let a0 = R.x * t00 + R.y * t01; let a1 = R.x * t01 + R.y * t11;');
      L.push('  let b0 = R.z * t00 + R.w * t01; let b1 = R.z * t01 + R.w * t11;');
      L.push('  st[p] = vec4<f32>(a0 * R.x + a1 * R.y, a0 * R.z + a1 * R.w, b0 * R.z + b1 * R.w, 0.0);');
    } else {
      L.push('  let pl = analytic_plastic(mid, tr, vp.z, PR.dt * (c00 + c11),');
      L.push('                            g00 * g11 - g01 * g10);');
      L.push('  Fm[p] = pl.F;');
      L.push('  vp.z = pl.Jp;');
      L.push('  st[p] = vec4<f32>(analytic_stress(mid, pl.F, pl.Jp), 0.0);');
    }
    L.push('  vel[p] = vec4<f32>(nv, vp.z, vp.w);');
    L.push('  Cm[p] = Cn;');
    L.push('}');

    // ------------------------------------------------------------------ prime the stress cache
    // The first P2G of a run has no previous G2P to have filled `st`. Analytically that is exactly
    // zero at F = I; a NETWORK is not, so this is not optional for the learned path.
    L.push('@compute @workgroup_size(' + WG_P + ')');
    L.push('fn prime(@builtin(global_invocation_id) gid : vec3<u32>) {');
    L.push('  let p = gid.x;');
    L.push('  if (p >= PR.n) { return; }');
    L.push('  let vp = vel[p];');
    L.push('  let mid = i32(round(vp.w));');
    L.push('  if (mid > M_SAND) { return; }');
    L.push('  let Fp = Fm[p];');
    if (nn) {
      L.push('  let R = polar_r(Fp);');
      L.push('  let s00 = R.x * Fp.x + R.z * Fp.z;');
      L.push('  let s11 = R.y * Fp.y + R.w * Fp.w;');
      L.push('  let s01 = 0.5 * ((R.x * Fp.y + R.z * Fp.w) + (R.y * Fp.x + R.w * Fp.z));');
      L.push('  let xs = packx(vec3<f32>(s00, s01, s11), Cm[p], vp.xy, vp.z, mid);');
      L.push('  let o = mlp(xs[0], xs[1], xs[2], xs[3]);');
      L.push('  let t00 = o.tau.x; let t01 = o.tau.y; let t11 = o.tau.z;');
      L.push('  let a0 = R.x * t00 + R.y * t01; let a1 = R.x * t01 + R.y * t11;');
      L.push('  let b0 = R.z * t00 + R.w * t01; let b1 = R.z * t01 + R.w * t11;');
      L.push('  st[p] = vec4<f32>(a0 * R.x + a1 * R.y, a0 * R.z + a1 * R.w, b0 * R.z + b1 * R.w, 0.0);');
    } else {
      L.push('  st[p] = vec4<f32>(analytic_stress(mid, Fp, vp.z), 0.0);');
    }
    L.push('}');

    // A do-nothing kernel over the particle range: the per-dispatch floor, measured with the same
    // launch geometry as the real ones. Without it a "cost" can be launch overhead wearing a hat.
    L.push('@compute @workgroup_size(' + WG_P + ')');
    L.push('fn nop(@builtin(global_invocation_id) gid : vec3<u32>) {');
    L.push('  let p = gid.x;');
    L.push('  if (p >= 4294967295u) { pos[p] = vec2<f32>(0.0, 0.0); }');
    L.push('}');
    return L.join('\n');
  }

  // ------------------------------------------------------------- isolated MLP parity probe shader
  function probeShader(H, f16) {
    var NV = weightVecCount(H);
    var L = [];
    if (f16) L.push('enable f16;');
    var V = f16 ? 'vec4<f16>' : 'vec4<f32>';
    L.push('@group(0) @binding(0) var<storage, read> X : array<vec4<f32>>;');
    L.push('@group(0) @binding(1) var<storage, read_write> Y : array<vec4<f32>>;');
    L.push('@group(0) @binding(2) var<uniform> WT : array<' + V + ', ' + NV + '>;');
    L.push('@compute @workgroup_size(64)');
    L.push('fn main(@builtin(global_invocation_id) gid : vec3<u32>) {');
    L.push('  let i = gid.x;');
    L.push('  if (4u * i >= arrayLength(&X)) { return; }');
    var cast = f16 ? function (s) { return 'vec4<f16>(' + s + ')'; } : function (s) { return s; };
    L.push('  let y0 = ' + cast('X[4u * i]') + '; let y1 = ' + cast('X[4u * i + 1u]') + ';');
    L.push('  let y2 = ' + cast('X[4u * i + 2u]') + '; let y3 = ' + cast('X[4u * i + 3u]') + ';');
    L.push('  let NV_ : u32 = ' + (VEC_PER_UNIT * H) + 'u;');
    L.push('  var oA = WT[NV_]; var oB = WT[NV_ + 1u];');
    L.push('  for (var k : u32 = 0u; k < ' + H + 'u; k = k + 1u) {');
    L.push('    let b = k * ' + VEC_PER_UNIT + 'u;');
    L.push('    let a = dot(WT[b], y0) + dot(WT[b + 1u], y1) + dot(WT[b + 2u], y2) + dot(WT[b + 3u], y3);');
    L.push('    let h = tanh(a);');
    L.push('    oA = oA + h * WT[b + 4u];');
    L.push('    oB = oB + h * WT[b + 5u];');
    L.push('  }');
    L.push('  Y[2u * i] = vec4<f32>(f32(oA.x), f32(oA.y), f32(oA.z), f32(oA.w));');
    L.push('  Y[2u * i + 1u] = vec4<f32>(f32(oB.x), f32(oB.y), f32(oB.z), f32(oB.w));');
    L.push('}');
    return L.join('\n');
  }

  // ------------------------------------------------------------------------------- device plumbing
  function supported() { return (typeof navigator !== 'undefined') && !!navigator.gpu; }

  var _device = null, _adapterInfo = null, _hasTimestamp = false, _hasF16 = false, _limits = null;
  var ERRORS = [];

  async function getDevice() {
    if (_device) return _device;
    if (!supported()) throw new Error('navigator.gpu is undefined (secure origin?)');
    var adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('requestAdapter returned null');
    _adapterInfo = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    var feats = [];
    _hasTimestamp = adapter.features.has('timestamp-query');
    _hasF16 = adapter.features.has('shader-f16');
    if (_hasTimestamp) feats.push('timestamp-query');
    if (_hasF16) feats.push('shader-f16');
    // Ask for the adapter's real limits, not the guaranteed minimums: whether a NINTH storage
    // buffer exists at all is one of the levers being measured.
    var want = {};
    ['maxStorageBuffersPerShaderStage', 'maxUniformBufferBindingSize', 'maxStorageBufferBindingSize',
      'maxBufferSize', 'maxComputeInvocationsPerWorkgroup'].forEach(function (k) {
        if (adapter.limits[k] !== undefined) want[k] = adapter.limits[k];
      });
    _device = await adapter.requestDevice({ requiredFeatures: feats, requiredLimits: want });
    _limits = {};
    Object.keys(want).forEach(function (k) { _limits[k] = _device.limits[k]; });
    _device.addEventListener('uncapturederror', function (e) {
      ERRORS.push(String((e.error && e.error.message) || e.error));
      console.error('WebGPU uncaptured error:', e.error);
    });
    _device.lost.then(function (e) { ERRORS.push('DEVICE LOST: ' + e.message); });
    return _device;
  }

  function deviceInfo() {
    var a = _adapterInfo || {};
    return {
      vendor: a.vendor || '', architecture: a.architecture || '', device: a.device || '',
      description: a.description || '', timestampQuery: _hasTimestamp, shaderF16: _hasF16,
      limits: _limits
    };
  }
  function errors() { return ERRORS.slice(); }
  function hasF16() { return _hasF16; }
  function hasTimestamp() { return _hasTimestamp; }

  // --------------------------------------------------------------------------------------- scenes
  function rngOf(seed) {
    var s = (seed >>> 0) || 1;
    return function () { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  }
  function seedDisk(cx, cy, r, n, seed) {
    var rnd = rngOf(seed), pts = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) {
      var a = rnd() * Math.PI * 2, rr = r * Math.sqrt(rnd());
      pts[2 * i] = cx + rr * Math.cos(a); pts[2 * i + 1] = cy + rr * Math.sin(a);
    }
    return pts;
  }
  function seedBox(x0, x1, y0, y1, n, seed) {
    var rnd = rngOf(seed), pts = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) { pts[2 * i] = x0 + (x1 - x0) * rnd(); pts[2 * i + 1] = y0 + (y1 - y0) * rnd(); }
    return pts;
  }
  function sharedDt(names) {
    var d = Infinity;
    for (var i = 0; i < names.length; i++) d = Math.min(d, MAT[names[i]].dt);
    return isFinite(d) ? d : MAT.elastic.dt;
  }

  // ---------------------------------------------------------------------------------- the simulator
  // `variants` is a list of {key, mode, hidden, f16, weights}. All share ONE set of buffers, so
  // switching variant is a bind-group swap and the timings are directly comparable.
  async function createSim(opts) {
    opts = opts || {};
    var device = await getDevice();
    var cap = opts.capacity | 0 || 16384;
    var kM = opts.kM !== undefined ? opts.kM : P.kM;
    var kV = opts.kV !== undefined ? opts.kV : P.kV;
    var pVol = opts.pVol !== undefined ? opts.pVol : (Math.PI * 0.11 * 0.11) / 2048;
    var pMass = pVol * P.p_rho;
    var dt = opts.dt !== undefined ? opts.dt : MAT.snow.dt;
    var massScale = Math.pow(2, kM) / pMass;
    var momScale = Math.pow(2, kV) / pMass;
    var n = 0;
    var loopH = 0;                    // hidden width for the dynamic-loop variants

    device.pushErrorScope('validation');
    var U = GPUBufferUsage;
    var STO = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    function buf(b, u) { return device.createBuffer({ size: b, usage: u }); }
    var posBuf = buf(cap * 8, STO), velBuf = buf(cap * 16, STO), CBuf = buf(cap * 16, STO),
      FBuf = buf(cap * 16, STO), gmBuf = buf(N_CELL * 4, STO), gpBuf = buf(N_CELL * 12, STO),
      gvBuf = buf(N_CELL * 16, STO), stBuf = buf(cap * 16, STO);
    var uBuf = buf(96, U.UNIFORM | U.COPY_DST);
    var zBuf = buf(64, U.UNIFORM | U.COPY_DST);
    var readBuf = device.createBuffer({ size: cap * 8, usage: U.COPY_DST | U.MAP_READ });
    var stateReadBuf = device.createBuffer({ size: cap * 56, usage: U.COPY_DST | U.MAP_READ });

    // material codes: uploaded from netspec via setCodes(); identity, never trained
    var zArr = new Float32Array(16);
    function setCodes(codes) { zArr.set(codes.subarray(0, 16)); device.queue.writeBuffer(zBuf, 0, zArr); }
    setCodes(new Float32Array(16));

    function sbEntry(b, ro) {
      return { binding: b, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: ro ? 'read-only-storage' : 'storage' } };
    }
    var baseEntries = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }];
    for (var bi = 1; bi <= 8; bi++) baseEntries.push(sbEntry(bi, false));
    var baseRes = [uBuf, posBuf, velBuf, CBuf, FBuf, gmBuf, gpBuf, gvBuf, stBuf];

    var variants = {};
    var buildErrors = [];
    for (var vi = 0; vi < (opts.variants || []).length; vi++) {
      var vspec = opts.variants[vi];
      try {
        variants[vspec.key] = await makeVariant(vspec);
      } catch (e) {
        buildErrors.push({ key: vspec.key, error: String(e.message || e) });
      }
    }

    async function makeVariant(vs) {
      var nn = vs.mode === 'nn';
      var entries = baseEntries.slice();
      var res = baseRes.slice();
      var wBuf = null;
      if (nn) {
        var bytes = weightFloatCount(vs.hidden) * (vs.f16 ? 2 : 4);
        if (vs.weights === 'storage') {
          wBuf = buf(Math.max(bytes, 16), U.STORAGE | U.COPY_DST);
          entries.push(sbEntry(9, true));
        } else {
          wBuf = buf(Math.max(bytes, 16), U.UNIFORM | U.COPY_DST);
          entries.push({ binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
        }
        entries.push({ binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
        res.push(wBuf); res.push(zBuf);
      }
      var code = buildShader(vs);
      var mod = device.createShaderModule({ code: code, label: vs.key });
      var info = await mod.getCompilationInfo();
      var errs = info.messages.filter(function (m) { return m.type === 'error'; });
      if (errs.length) {
        throw new Error('WGSL ' + vs.key + ': ' + errs.map(function (m) {
          return m.lineNum + ':' + m.linePos + ' ' + m.message; }).join(' | '));
      }
      var layout = device.createBindGroupLayout({ entries: entries, label: vs.key + '-bgl' });
      var bind = device.createBindGroup({
        layout: layout, label: vs.key + '-bg',
        entries: res.map(function (b, i) {
          return { binding: i <= 8 ? i : (i === 9 ? 9 : 10), resource: { buffer: b } }; })
      });
      var pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      function pipe(entry) {
        return device.createComputePipeline({ layout: pl, label: vs.key + ':' + entry,
          compute: { module: mod, entryPoint: entry } });
      }
      return {
        spec: vs, bind: bind, weightBuf: wBuf, code: code,
        pClear: pipe('clear_grid'), pP2G: pipe('p2g'), pGrid: pipe('grid_op'),
        pG2P: pipe('g2p'), pPrime: pipe('prime'), pNop: pipe('nop')
      };
    }

    var setupError = await device.popErrorScope();
    if (setupError) throw new Error('WebGPU setup: ' + setupError.message);

    var uArr = new ArrayBuffer(96);
    var uF = new Float32Array(uArr), uU = new Uint32Array(uArr);
    function writeUniform() {
      uF[0] = dt; uF[1] = pMass; uF[2] = pVol; uF[3] = P.gravity; uF[4] = P.FRICTION;
      uF[5] = massScale; uF[6] = 1.0 / massScale; uF[7] = momScale; uF[8] = 1.0 / momScale;
      uU[9] = n >>> 0;
      uU[10] = loopH >>> 0;
      // slots 12..15 = rho by material id, 16..19 = fric by material id (vec4 alignment in WGSL
      // puts the two vec4 fields at byte 48 and 64)
      for (var q = 0; q < 4; q++) {
        uF[12 + q] = MAT[ORDER[q]].rho;
        uF[16 + q] = MAT[ORDER[q]].fric;
      }
      device.queue.writeBuffer(uBuf, 0, uArr);
    }
    writeUniform();

    var qset = null, qResolve = null, qRead = null;
    if (_hasTimestamp) {
      qset = device.createQuerySet({ type: 'timestamp', count: 2 });
      qResolve = buf(16, U.QUERY_RESOLVE | U.COPY_SRC);
      qRead = device.createBuffer({ size: 16, usage: U.COPY_DST | U.MAP_READ });
    }

    function zeroGrid() {
      device.queue.writeBuffer(gmBuf, 0, new Uint32Array(N_CELL));
      device.queue.writeBuffer(gpBuf, 0, new Int32Array(3 * N_CELL));
      device.queue.writeBuffer(gvBuf, 0, new Float32Array(4 * N_CELL));
    }

    function add(material, pts, v0x, v0y) {
      var k = Math.min(pts.length >> 1, cap - n);
      if (k <= 0) return 0;
      var mid = ID[material];
      var xs = new Float32Array(2 * k), vs = new Float32Array(4 * k),
        Cs = new Float32Array(4 * k), Fs = new Float32Array(4 * k), ss = new Float32Array(4 * k);
      for (var p = 0; p < k; p++) {
        xs[2 * p] = pts[2 * p]; xs[2 * p + 1] = pts[2 * p + 1];
        vs[4 * p] = v0x || 0; vs[4 * p + 1] = v0y || 0;
        vs[4 * p + 2] = (material === 'sand') ? 0.0 : 1.0;
        vs[4 * p + 3] = mid;
        // EVERY material starts at F = I here, the fluid included: this port carries the fluid's
        // volume ratio in F (det F = J), which is what gives the network one state to learn.
        Fs[4 * p] = 1; Fs[4 * p + 3] = 1;
      }
      device.queue.writeBuffer(posBuf, n * 8, xs);
      device.queue.writeBuffer(velBuf, n * 16, vs);
      device.queue.writeBuffer(CBuf, n * 16, Cs);
      device.queue.writeBuffer(FBuf, n * 16, Fs);
      device.queue.writeBuffer(stBuf, n * 16, ss);
      n += k;
      writeUniform();
      return k;
    }
    function clearAll() { n = 0; zeroGrid(); writeUniform(); }

    function setWeights(key, floats) {
      var v = variants[key];
      if (!v || !v.weightBuf) return false;
      var want = weightFloatCount(v.spec.hidden);
      var src = floats;
      if (src.length !== want) {
        var t = new Float32Array(want); t.set(src.subarray(0, Math.min(want, src.length))); src = t;
      }
      if (v.spec.f16) {
        device.queue.writeBuffer(v.weightBuf, 0, f32to16(src));
      } else {
        device.queue.writeBuffer(v.weightBuf, 0, src);
      }
      return true;
    }

    function encodeFrame(key, substeps, opt) {
      opt = opt || {};
      var v = variants[key];
      if (!v) throw new Error('unknown variant ' + key);
      if (v.spec.dyn && loopH !== v.spec.hidden) { loopH = v.spec.hidden; writeUniform(); }
      var enc = device.createCommandEncoder();
      var desc = {};
      if (opt.timed && qset) {
        desc.timestampWrites = { querySet: qset, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
      }
      var pass = enc.beginComputePass(desc);
      pass.setBindGroup(0, v.bind);
      var P_WG = Math.max(1, Math.ceil(n / WG_P));
      if (opt.clearFirst) { pass.setPipeline(v.pClear); pass.dispatchWorkgroups(GRID_WG); }
      if (opt.prime && n > 0) { pass.setPipeline(v.pPrime); pass.dispatchWorkgroups(P_WG); }
      var phases = opt.phases || 'pgG';
      var doP = phases.indexOf('p') >= 0, doG = phases.indexOf('g') >= 0,
        doQ = phases.indexOf('G') >= 0, doN = phases.indexOf('n') >= 0;
      for (var s = 0; s < substeps; s++) {
        if (doN) { pass.setPipeline(v.pNop); pass.dispatchWorkgroups(P_WG); }
        if (doP && n > 0) { pass.setPipeline(v.pP2G); pass.dispatchWorkgroups(P_WG); }
        if (doG) { pass.setPipeline(v.pGrid); pass.dispatchWorkgroups(GRID_WG); }
        if (doQ && n > 0) { pass.setPipeline(v.pG2P); pass.dispatchWorkgroups(P_WG); }
      }
      pass.end();
      if (opt.timed && qset) {
        enc.resolveQuerySet(qset, 0, 2, qResolve, 0);
        enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 16);
      }
      if (opt.readback && n > 0) enc.copyBufferToBuffer(posBuf, 0, readBuf, 0, n * 8);
      if (opt.stateReadback && n > 0) {
        enc.copyBufferToBuffer(posBuf, 0, stateReadBuf, 0, n * 8);
        enc.copyBufferToBuffer(velBuf, 0, stateReadBuf, cap * 8, n * 16);
        enc.copyBufferToBuffer(CBuf, 0, stateReadBuf, cap * 24, n * 16);
        enc.copyBufferToBuffer(FBuf, 0, stateReadBuf, cap * 40, n * 16);
      }
      device.queue.submit([enc.finish()]);
    }

    async function lastGpuNanos() {
      if (!qset) return null;
      await qRead.mapAsync(GPUMapMode.READ);
      var t = new BigUint64Array(qRead.getMappedRange().slice(0));
      qRead.unmap();
      return Number(t[1] - t[0]);
    }
    async function readPositions() {
      if (n === 0) return new Float32Array(0);
      await readBuf.mapAsync(GPUMapMode.READ, 0, n * 8);
      var out = new Float32Array(readBuf.getMappedRange(0, n * 8).slice(0));
      readBuf.unmap();
      return out;
    }
    async function readState() {
      if (n === 0) return null;
      await stateReadBuf.mapAsync(GPUMapMode.READ);
      var all = new Float32Array(stateReadBuf.getMappedRange().slice(0));
      stateReadBuf.unmap();
      return { pos: all.subarray(0, n * 2), vel: all.subarray(cap * 2, cap * 2 + n * 4),
        C: all.subarray(cap * 6, cap * 6 + n * 4), F: all.subarray(cap * 10, cap * 10 + n * 4) };
    }

    return {
      get n() { return n; }, capacity: cap, device: device, variants: variants,
      buildErrors: buildErrors,
      params: { dt: dt, pVol: pVol, pMass: pMass, n_grid: N_GRID, gravity: P.gravity,
        friction: P.FRICTION, kM: kM, kV: kV, physics_version: P.physics_version },
      buffers: { pos: posBuf, vel: velBuf, gv: gvBuf, st: stBuf },
      setDt: function (d) { dt = d; writeUniform(); },
      getDt: function () { return dt; },
      setCodes: setCodes, setWeights: setWeights,
      add: add, clear: clearAll, encodeFrame: encodeFrame, lastGpuNanos: lastGpuNanos,
      readPositions: readPositions, readState: readState,
      idle: function () { return device.queue.onSubmittedWorkDone(); },
      destroy: function () {
        [posBuf, velBuf, CBuf, FBuf, gmBuf, gpBuf, gvBuf, stBuf, uBuf, zBuf, readBuf, stateReadBuf]
          .forEach(function (b) { b.destroy(); });
        Object.keys(variants).forEach(function (k) {
          if (variants[k].weightBuf) variants[k].weightBuf.destroy(); });
      }
    };
  }

  // IEEE754 binary32 -> binary16, round-to-nearest-even, with subnormal and overflow handling.
  // Written out rather than leaned on a library because a silently wrong f16 conversion would show
  // up as "f16 is inaccurate", which is a conclusion about the hardware rather than about a bug.
  function f32to16(src) {
    var out = new Uint16Array(src.length);
    var fb = new Float32Array(1), ib = new Uint32Array(fb.buffer);
    for (var i = 0; i < src.length; i++) {
      fb[0] = src[i];
      var x = ib[0];
      var sign = (x >>> 16) & 0x8000;
      var exp = (x >>> 23) & 0xff;
      var man = x & 0x7fffff;
      var h;
      if (exp === 0xff) { h = sign | 0x7c00 | (man ? 0x200 : 0); }
      else {
        var e = exp - 127 + 15;
        if (e >= 0x1f) { h = sign | 0x7c00; }
        else if (e <= 0) {
          if (e < -10) { h = sign; }
          else {
            man = man | 0x800000;
            var shift = 14 - e;
            var mant = man >>> shift;
            if (((man >>> (shift - 1)) & 1) && ((man & ((1 << (shift - 1)) - 1)) || (mant & 1))) mant++;
            h = sign | mant;
          }
        } else {
          var m = man >>> 13;
          if ((man & 0x1000) && ((man & 0xfff) || (m & 1))) {
            m++;
            if (m === 0x400) { m = 0; e++; if (e >= 0x1f) { h = sign | 0x7c00; } }
          }
          if (h === undefined) h = sign | (e << 10) | m;
        }
      }
      out[i] = h;
      h = undefined;
    }
    return out;
  }

  // ------------------------------------------------------------- isolated MLP parity probe runner
  async function mlpProbe(H, f16, weights, X) {
    var device = await getDevice();
    var k = X.length / 16;                                  // 16 padded inputs per sample
    var U = GPUBufferUsage;
    device.pushErrorScope('validation');
    var xb = device.createBuffer({ size: X.byteLength, usage: U.STORAGE | U.COPY_DST });
    var yb = device.createBuffer({ size: k * 32, usage: U.STORAGE | U.COPY_SRC });
    var wf = weightFloatCount(H);
    var wb = device.createBuffer({ size: wf * (f16 ? 2 : 4), usage: U.UNIFORM | U.COPY_DST });
    var rb = device.createBuffer({ size: k * 32, usage: U.COPY_DST | U.MAP_READ });
    device.queue.writeBuffer(xb, 0, X);
    device.queue.writeBuffer(wb, 0, f16 ? f32to16(weights) : weights);
    var mod = device.createShaderModule({ code: probeShader(H, f16), label: 'mlp-probe' });
    var ci = await mod.getCompilationInfo();
    var errs = ci.messages.filter(function (m) { return m.type === 'error'; });
    if (errs.length) throw new Error('probe WGSL: ' + errs.map(function (m) { return m.lineNum + ': ' + m.message; }).join(' | '));
    var pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    var bg = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: xb } }, { binding: 1, resource: { buffer: yb } },
        { binding: 2, resource: { buffer: wb } }]
    });
    var enc = device.createCommandEncoder();
    var pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(k / 64));
    pass.end();
    enc.copyBufferToBuffer(yb, 0, rb, 0, k * 32);
    device.queue.submit([enc.finish()]);
    var err = await device.popErrorScope();
    if (err) throw new Error('probe setup: ' + err.message);
    await rb.mapAsync(GPUMapMode.READ);
    var out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    [xb, yb, wb, rb].forEach(function (b) { b.destroy(); });
    return out;                                             // k * 8 floats (7 used + 1 pad)
  }

  return {
    supported: supported, getDevice: getDevice, deviceInfo: deviceInfo, errors: errors,
    hasF16: hasF16, hasTimestamp: hasTimestamp,
    createSim: createSim, mlpProbe: mlpProbe, buildShader: buildShader, f32to16: f32to16,
    seedDisk: seedDisk, seedBox: seedBox, sharedDt: sharedDt,
    weightFloatCount: weightFloatCount, weightVecCount: weightVecCount,
    N_IN: N_IN, N_IN_PAD: N_IN_PAD, N_OUT: N_OUT, N_OUT_PAD: N_OUT_PAD, Z_DIM: Z_DIM,
    PARAMS: P, MAT: MAT, ORDER: ORDER, ID: ID, N_GRID: N_GRID, N_CELL: N_CELL,
    WG_P: WG_P, WG_G: WG_G
  };
});
