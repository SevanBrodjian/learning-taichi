// mpm4-webgpu.js -- a WebGPU compute port of the canonical 2D MLS-MPM step for ALL FOUR canonical
// materials (fluid, elastic, snow, sand) on ONE shared grid.
//
// PORTABILITY CONTRACT: no dependency on the dashboard, the data server, or the harness. It needs
// only `params.js` (generated from sim.physics by gen_params.py) and a WebGPU-capable browser.
// Load it with a <script> tag -> window.MPM4.
//
// =================================================================================================
// WHAT THIS ADDS OVER THE ELASTIC-ONLY PORT, AND WHAT IT COST
// =================================================================================================
//
// 1. A REAL 2x2 SVD IN WGSL. The elastic path never needs singular values -- the corotated stress
//    only wants the polar rotation R, which has a closed form. Snow and sand both need the singular
//    values themselves: snow CLAMPS them into a box (Stomakhin), sand projects the log of them onto
//    a CONE (Drucker-Prager). `svd2()` below is a line-for-line port of Taichi's `_svd2d` (polar
//    decomposition + one Jacobi rotation), not a lookalike, because the return maps are written
//    against Taichi's exact conventions -- descending singular values, U = R V, and A = U S V^T.
//    It is unit-tested against `ti.svd` on adversarial matrices before it is trusted (verify/).
//
// 2. PER-PARTICLE STATE WITHOUT NEW BUFFERS. The device guarantees only 8 storage buffers per
//    shader stage and the elastic layout already used 7. A per-particle material id and a per-
//    particle plastic record `Jp` are two more -> 9, which produces a silently INVALID BIND GROUP:
//    every dispatch is dropped, the sim "runs" at the speed of doing nothing, and the timing curve
//    looks spectacular over trajectories of pure zeros. So both are packed into the velocity
//    buffer, widened from vec2 to vec4:
//        vel[p] = (vx, vy, Jp, matId)
//    Still 7 storage buffers. The fluid's scalar volume ratio J rides in Fm[p].x, which the fluid
//    path does not otherwise use, so it costs nothing either.
//
// 3. ONE GRID MEANS ONE TIMESTEP. `sharedDt()` reproduces canonical `sim.physics.shared_dt`:
//    min(dt) over the materials actually present. Adding snow (dt = 5e-5) to any scene doubles the
//    substep count for every particle in it. That is a physical consequence of a shared grid, not a
//    tuning choice, and it is why a plastic material in a mixed scene creeps MORE than canonical
//    (plastic strain accumulates per substep, not per unit of physical time).
//
// 4. PER-MATERIAL DENSITY, POISSON RATIO AND WALL FRICTION (physics_version phys-c518316a4a05).
//    Three quantities that were single global constants are now per material, and each one needed a
//    different trick to thread through a port with no spare storage buffers:
//
//      rho  -- the particle mass is `PR.pVol * RHO[mid]`, read from a compile-time constant selected
//              by the material id that is already packed in vel.w. No buffer, no uniform. THE SINK /
//              FLOAT ORDERING IS AN OUTPUT OF THIS AND NOTHING ELSE: there is no buoyancy term
//              anywhere in this file, exactly as in canonical. Heavy material scatters more mass to
//              a node, the grid divides the scattered momentum by that mass, and gravity (applied to
//              the node VELOCITY) accelerates every node equally -- so the surrounding water's
//              pressure impulse moves a light node more than a heavy one. Archimedes falls out.
//      nu   -- folded into the emitted Lame constants MU_*/LA_* by gen_params.py, using each
//              material's own nu. Nothing in the shader sees a Poisson ratio.
//      fric -- canonical scatters a MASS-WEIGHTED friction to the grid so a node shared by two
//              materials gets the friction of what is sitting on it. That is a fourth accumulator,
//              and a fourth storage buffer would be the 8th -- one under the ceiling, but the
//              ceiling is not worth spending on bookkeeping. Instead `gm` is WIDENED to two u32 per
//              cell: gm[2i] is the fixed-point mass, gm[2i+1] the fixed-point mass*friction, and the
//              node coefficient is their ratio. Same buffer, same bind group, still 7.
//
//    Because the mass now differs per particle, the fixed-point scales can no longer be "in units of
//    the particle mass" -- two materials would scatter into one accumulator on two different scales.
//    They are in units of a fixed REFERENCE mass, pVol * 1.0 (water). See gen_params.py.
//
// 5. WALLS SEPARATE, THEY DO NOT GLUE. All four boundaries take canonical's treatment: zero the
//    normal component when it points into the wall, Coulomb-limit the tangent. The old port zeroed
//    BOTH components at the side walls, which welded material to them.
//
// The interaction force (poke/drag) is a demo-only external body force layered on the grid update.
// It is off by default and is never enabled during verification.

// Params always come from `root.MPM_PARAMS` -- set by a <script src="params.js"> tag in the
// standalone page, or by the generated ES-module wrapper in a bundled one. There is deliberately no
// CommonJS import here: a bundler that sees one in a dead branch still tries to resolve it, and this
// file has to survive being dropped into a build system it knows nothing about.
(function (root, factory) {
  var api = factory(root.MPM_PARAMS);
  root.MPM4 = api;
  if (typeof module === 'object' && module.exports) { module.exports = api; }
})(typeof self !== 'undefined' ? self : this, function (P) {
  'use strict';

  var N_GRID = P.n_grid | 0;
  var N_CELL = N_GRID * N_GRID;
  var WG_P = 64;                 // particles per workgroup
  var WG_G = 64;                 // cells per workgroup
  var GRID_WG = Math.ceil(N_CELL / WG_G);

  var MAT = P.materials;
  var ORDER = P.mat_order;                       // ['fluid','elastic','snow','sand']
  var ID = P.mat_id;                             // {fluid:0, elastic:1, snow:2, sand:3}
  var DEAD = 4;                                  // erased particle: no mass, no motion, not drawn
  // Which set of screen-space material treatments the renderer is compiled with. Stamped into every
  // benchmark row so a timing can never be attributed to the wrong renderer after the fact.
  var RENDER_TREATMENT = 't027r-per-material-water-isosurface';

  function f(x) {                                // emit a float literal WGSL will accept
    var s = String(x);
    if (!/[.eE]/.test(s)) s += '.0';
    return s;
  }

  // -----------------------------------------------------------------------------------------------
  // The 2x2 SVD, ported from taichi/_funcs.py::_svd2d and ::_polar_decompose2d.
  //
  // Returned as a flat struct because WGSL has no multiple return. Row-major throughout:
  // a vec4 (m00, m01, m10, m11) is the matrix [[x, y], [z, w]].
  //
  // Conventions that MATTER downstream and are therefore reproduced exactly:
  //   * A = U * diag(s) * V^T  (V, not V^T, is the third factor -- the same as ti.svd)
  //   * s.x >= s.y  (descending), enforced by the same swap Taichi does
  //   * the polar factor handles det(A) < 0 (a reflection) with the second B, so a particle that
  //     inverts does not silently get a rotation with the wrong handedness
  // The ONE deliberate deviation: `sqrt(adetB)` is floored at 1e-30 rather than left to divide by
  // zero. Taichi's comment argues det(B) != 0 for any non-zero A, which is true in exact arithmetic;
  // in f32 a near-degenerate A can still round it to zero, and an inf here would poison a particle.
  // -----------------------------------------------------------------------------------------------
  var SVD_WGSL = [
    'struct Svd2 { u : vec4<f32>, s : vec2<f32>, v : vec4<f32> };',
    '',
    // U diag(s) V -- reconstruction exactly as canonical sim/physics/core.py writes it (g2p:330,
    // dp_return_map:224). NOTE: canonical names ti.svd's third return `Vt`, but ti.svd returns V
    // (A = U S V^T), so canonical reconstructs U S V, i.e. the textbook F right-multiplied by V^2.
    // For an ISOTROPIC constitutive model that is unobservable -- right-multiplying F by any
    // rotation leaves the singular values, the corotated stress (F-R)F^T and the Hencky stress
    // U(...)U^T all exactly invariant -- so it is a choice of reference frame, not a physics
    // difference. It is reproduced verbatim so the port matches canonical state-for-state, not
    // merely observable-for-observable.
    'fn mul_usv(u : vec4<f32>, s : vec2<f32>, v : vec4<f32>) -> vec4<f32> {',
    '  let a = u.x * s.x; let b = u.y * s.y;',
    '  let c = u.z * s.x; let d = u.w * s.y;',
    '  return vec4<f32>(a * v.x + b * v.z, a * v.y + b * v.w,',
    '                   c * v.x + d * v.z, c * v.y + d * v.w);',
    '}',
    '',
    'fn svd2(a : vec4<f32>) -> Svd2 {',
    '  let a00 = a.x; let a01 = a.y; let a10 = a.z; let a11 = a.w;',
    '  var r = vec4<f32>(1.0, 0.0, 0.0, 1.0);',      // polar rotation R
    '  var p = a;',                                   // symmetric factor P (A = R P)
    '  if (!(a00 == 0.0 && a01 == 0.0 && a10 == 0.0 && a11 == 0.0)) {',
    '    let detA = a00 * a11 - a10 * a01;',
    '    let adetA = abs(detA);',
    '    var b = vec4<f32>(a00 + a11, a01 - a10, a10 - a01, a11 + a00);',
    '    if (detA < 0.0) { b = vec4<f32>(a00 - a11, a01 + a10, a10 + a01, a11 - a00); }',
    '    let adetB = abs(b.x * b.w - b.z * b.y);',
    '    let k = 1.0 / max(sqrt(adetB), 1e-30);',
    '    r = b * k;',
    '    let t00 = a00 * a00 + a10 * a10;',           // A^T A
    '    let t01 = a00 * a01 + a10 * a11;',
    '    let t11 = a01 * a01 + a11 * a11;',
    '    p = vec4<f32>((t00 + adetA) * k, t01 * k, t01 * k, (t11 + adetA) * k);',
    '  }',
    '  var c = 1.0; var s = 0.0;',
    '  var s1 = p.x; var s2 = p.w;',
    '  if (abs(p.y) >= 1e-5) {',                      // one Jacobi rotation diagonalises P
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
    '',
    // The polar rotation on its own: what the corotated (elastic/snow) stress needs, and all it
    // needs. Identical to svd2's `r`, so elastic and snow never pay for the Jacobi step.
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

  // ------------------------------------------------------------------ WGSL source
  function buildShader() {
    var Mf = MAT.fluid, Me = MAT.elastic, Ms = MAT.snow, Ma = MAT.sand;
    return [
      '// GENERATED from params.js (physics_version ' + P.physics_version + ')',
      'const N_GRID : i32 = ' + N_GRID + ';',
      'const N_CELL : i32 = ' + N_CELL + ';',
      'const DX : f32 = ' + f(P.dx) + ';',
      'const INV_DX : f32 = ' + f(P.inv_dx) + ';',
      'const BOUND : i32 = ' + P.bound + ';',
      'const FLOOR : f32 = ' + f(P.floor_y) + ';',
      'const CEIL : f32 = ' + f(1.0 - P.floor_y) + ';',
      '',
      '// --- canonical material ids (sim.physics.core.MAT_ID) ---',
      'const M_FLUID : i32 = ' + ID.fluid + ';',
      'const M_ELASTIC : i32 = ' + ID.elastic + ';',
      'const M_SNOW : i32 = ' + ID.snow + ';',
      'const M_SAND : i32 = ' + ID.sand + ';',
      '',
      '// --- FROZEN canonical parameters, emitted from sim.physics.MAT ---',
      'const E_FLUID : f32 = ' + f(Mf.E) + ';',
      'const MU_E : f32 = ' + f(Me.mu) + ';   const LA_E : f32 = ' + f(Me.la) + ';',
      'const MU_S : f32 = ' + f(Ms.mu) + ';   const LA_S : f32 = ' + f(Ms.la) + ';',
      'const XI_S : f32 = ' + f(Ms.xi) + ';   const TC_S : f32 = ' + f(Ms.tc) + ';',
      'const TS_S : f32 = ' + f(Ms.ts) + ';',
      'const MU_A : f32 = ' + f(Ma.mu) + ';   const LA_A : f32 = ' + f(Ma.la) + ';',
      'const ALPHA_A : f32 = ' + f(Ma.alpha) + ';',
      '',
      '// --- per-material DENSITY and WALL FRICTION (phys-c518316a4a05). Emitted as constants and',
      '// selected by the material id already packed in vel.w, so neither costs a storage buffer.',
      'const RHO_F : f32 = ' + f(Mf.rho) + ';  const RHO_E : f32 = ' + f(Me.rho) + ';',
      'const RHO_S : f32 = ' + f(Ms.rho) + ';  const RHO_A : f32 = ' + f(Ma.rho) + ';',
      'const FR_F : f32 = ' + f(Mf.fric) + ';  const FR_E : f32 = ' + f(Me.fric) + ';',
      'const FR_S : f32 = ' + f(Ms.fric) + ';  const FR_A : f32 = ' + f(Ma.fric) + ';',
      '',
      'fn matRho(m : i32) -> f32 {',
      '  if (m == M_FLUID) { return RHO_F; }',
      '  if (m == M_ELASTIC) { return RHO_E; }',
      '  if (m == M_SNOW) { return RHO_S; }',
      '  return RHO_A;',
      '}',
      'fn matFric(m : i32) -> f32 {',
      '  if (m == M_FLUID) { return FR_F; }',
      '  if (m == M_ELASTIC) { return FR_E; }',
      '  if (m == M_SNOW) { return FR_S; }',
      '  return FR_A;',
      '}',
      '',
      SVD_WGSL,
      '',
      'struct Params {',
      '  dt : f32, pMass : f32, pVol : f32, gravity : f32, friction : f32,',
      '  massScale : f32, invMassScale : f32, momScale : f32, invMomScale : f32,',
      '  n : u32, pokeOn : u32,',
      '  pokeX : f32, pokeY : f32, pokeVX : f32, pokeVY : f32,',
      '  pokeRadius : f32, pokeRate : f32, pokeSpring : f32,',
      '  eraseR : f32, pad0 : f32, pad1 : f32, pad2 : f32,',
      '};',
      '',
      '@group(0) @binding(0) var<uniform> PR : Params;',
      '@group(0) @binding(1) var<storage, read_write> pos : array<vec2<f32>>;',
      // vel.xy = velocity, vel.z = Jp (snow: plastic volume ratio, starts 1; sand: plastic
      // volumetric log-strain, starts 0), vel.w = material id as f32. Packed here rather than in
      // two new buffers because 9 storage buffers per stage silently invalidates the bind group.
      '@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;',
      '@group(0) @binding(3) var<storage, read_write> Cm  : array<vec4<f32>>;',
      // Fm = deformation gradient (row major) for elastic/snow/sand; Fm.x = J for fluid.
      '@group(0) @binding(4) var<storage, read_write> Fm  : array<vec4<f32>>;',
      // gm[2i] = fixed-point node mass; gm[2i+1] = fixed-point node mass*friction. The ratio is the
      // node's Coulomb coefficient, so a node holding water and sand gets the friction of the
      // mixture by mass, exactly as canonical's grid_fr/grid_m does. Widening this buffer rather
      // than adding an 8th is deliberate: the 8-storage-buffer ceiling is not spent on bookkeeping.
      '@group(0) @binding(5) var<storage, read_write> gm  : array<atomic<u32>>;',
      '@group(0) @binding(6) var<storage, read_write> gp  : array<atomic<i32>>;',
      // gv.xy = node velocity (what G2P gathers). gv.z = node mass in PARTICLE MASSES, gv.w =
      // |node momentum| in particle-mass*velocity -- the two quantities the mass heatmap and the
      // fixed-point headroom probe need, carried for free instead of a 9th buffer.
      '@group(0) @binding(7) var<storage, read_write> gv  : array<vec4<f32>>;',
      '',
      // ---------------------------------------------------------------- clear
      '@compute @workgroup_size(' + WG_G + ')',
      'fn clear_grid(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  atomicStore(&gm[2 * idx], 0u);',
      '  atomicStore(&gm[2 * idx + 1], 0u);',
      '  atomicStore(&gp[2 * idx], 0);',
      '  atomicStore(&gp[2 * idx + 1], 0);',
      '  gv[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);',
      '}',
      '',
      // ---------------------------------------------------------------- P2G
      '@compute @workgroup_size(' + WG_P + ')',
      'fn p2g(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let p = gid.x;',
      '  if (p >= PR.n) { return; }',
      '  let vp = vel[p];',
      '  let mid = i32(round(vp.w));',
      '  if (mid > M_SAND) { return; }',                       // erased: contributes no mass
      '  let Xp = pos[p] * INV_DX;',
      '  let base = vec2<i32>(Xp - vec2<f32>(0.5, 0.5));',
      '  let fx = Xp - vec2<f32>(base);',
      '  let w0 = 0.5 * (vec2<f32>(1.5, 1.5) - fx) * (vec2<f32>(1.5, 1.5) - fx);',
      '  let w1 = vec2<f32>(0.75, 0.75) - (fx - vec2<f32>(1.0, 1.0)) * (fx - vec2<f32>(1.0, 1.0));',
      '  let w2 = 0.5 * (fx - vec2<f32>(0.5, 0.5)) * (fx - vec2<f32>(0.5, 0.5));',
      '',
      // THE line that makes one material heavier than another (canonical: p_mass_f = p_vol * rho).
      // Everything downstream -- the scattered mass, the momentum, the affine term -- uses this,
      // and nothing anywhere adds an upward force to a light material.
      '  let pm = PR.pVol * matRho(mid);',
      '  let fr = matFric(mid);',
      '  let Fp = Fm[p];',
      '  let kk = -PR.dt * 4.0 * PR.pVol * INV_DX * INV_DX;',
      '  var st = vec4<f32>(0.0, 0.0, 0.0, 0.0);',             // -dt*4*pVol*inv_dx^2 * (P F^T)
      '  if (mid == M_FLUID) {',
      // weakly compressible: sigma = E (J - 1) I
      '    let pres = E_FLUID * (Fp.x - 1.0);',
      '    st = vec4<f32>(kk * pres, 0.0, 0.0, kk * pres);',
      '  } else if (mid == M_SAND) {',
      // Hencky (log-strain) Kirchhoff stress: tau = U diag(2 mu e + la tr(e)) U^T. Needs the
      // singular values themselves, hence the SVD, and no 1/sigma survives into the transfer.
      '    let sv = svd2(Fp);',
      '    let e0 = log(max(sv.s.x, 1e-4));',
      '    let e1 = log(max(sv.s.y, 1e-4));',
      '    let tr = e0 + e1;',
      '    let t0 = 2.0 * MU_A * e0 + LA_A * tr;',
      '    let t1 = 2.0 * MU_A * e1 + LA_A * tr;',
      '    let u = sv.u;',
      // U diag(t0,t1) U^T
      '    let m00 = u.x * t0 * u.x + u.y * t1 * u.y;',
      '    let m01 = u.x * t0 * u.z + u.y * t1 * u.w;',
      '    let m11 = u.z * t0 * u.z + u.w * t1 * u.w;',
      '    st = vec4<f32>(kk * m00, kk * m01, kk * m01, kk * m11);',
      '  } else {',
      // corotated: 2 mu (F - R) F^T + la (J-1) J I, with snow's hardening h = exp(xi (1 - Jp))
      '    var mu = MU_E; var la = LA_E;',
      '    if (mid == M_SNOW) {',
      '      let h = exp(XI_S * (1.0 - vp.z));',
      '      mu = MU_S * h; la = LA_S * h;',
      '    }',
      '    let r = polar_r(Fp);',
      '    let a00 = Fp.x - r.x; let a01 = Fp.y - r.y;',
      '    let a10 = Fp.z - r.z; let a11 = Fp.w - r.w;',
      '    let b00 = a00 * Fp.x + a01 * Fp.y; let b01 = a00 * Fp.z + a01 * Fp.w;',
      '    let b10 = a10 * Fp.x + a11 * Fp.y; let b11 = a10 * Fp.z + a11 * Fp.w;',
      '    let Jd = Fp.x * Fp.w - Fp.y * Fp.z;',
      '    let lt = la * (Jd - 1.0) * Jd;',
      '    st = vec4<f32>(kk * (2.0 * mu * b00 + lt), kk * (2.0 * mu * b01),',
      '                   kk * (2.0 * mu * b10),      kk * (2.0 * mu * b11 + lt));',
      '  }',
      '',
      '  let Cp = Cm[p];',
      '  let af00 = st.x + pm * Cp.x; let af01 = st.y + pm * Cp.y;',
      '  let af10 = st.z + pm * Cp.z; let af11 = st.w + pm * Cp.w;',
      '  let mv = pm * vp.xy;',
      '',
      '  for (var i = 0; i < 3; i = i + 1) {',
      '    let wxi = select(select(w0.x, w1.x, i == 1), w2.x, i == 2);',
      '    let dpx = (f32(i) - fx.x) * DX;',
      '    for (var j = 0; j < 3; j = j + 1) {',
      '      let wyj = select(select(w0.y, w1.y, j == 1), w2.y, j == 2);',
      '      let dpy = (f32(j) - fx.y) * DX;',
      '      let w = wxi * wyj;',
      '      let gi = (base.x + i) * N_GRID + (base.y + j);',
      '      let wm  = w * pm;',
      '      let mvx = w * (mv.x + af00 * dpx + af01 * dpy);',
      '      let mvy = w * (mv.y + af10 * dpx + af11 * dpy);',
      // round(), not truncation: truncating a signed momentum biases it toward zero, which is a
      // systematic numerical drag rather than noise.
      '      atomicAdd(&gm[2 * gi], u32(round(wm * PR.massScale)));',
      '      atomicAdd(&gm[2 * gi + 1], u32(round(wm * fr * PR.massScale)));',
      '      atomicAdd(&gp[2 * gi], i32(round(mvx * PR.momScale)));',
      '      atomicAdd(&gp[2 * gi + 1], i32(round(mvy * PR.momScale)));',
      '    }',
      '  }',
      '}',
      '',
      // ---------------------------------------------------------------- grid op (+ fused clear)
      // Canonical `coulomb`: drag the tangential velocity toward zero by at most `cap`, never past
      // it. Sliding friction removes momentum; it does not reverse it.
      'fn coulomb(vt : f32, cap : f32) -> f32 {',
      '  if (vt > 0.0) { return max(0.0, vt - cap); }',
      '  if (vt < 0.0) { return min(0.0, vt + cap); }',
      '  return vt;',
      '}',
      '',
      '@compute @workgroup_size(' + WG_G + ')',
      'fn grid_op(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  let m = f32(atomicLoad(&gm[2 * idx])) * PR.invMassScale;',
      '  let mfr = f32(atomicLoad(&gm[2 * idx + 1])) * PR.invMassScale;',
      '  let momx = f32(atomicLoad(&gp[2 * idx])) * PR.invMomScale;',
      '  let momy = f32(atomicLoad(&gp[2 * idx + 1])) * PR.invMomScale;',
      '  atomicStore(&gm[2 * idx], 0u);',             // fused clear: grid_op is the only reader
      '  atomicStore(&gm[2 * idx + 1], 0u);',
      '  atomicStore(&gp[2 * idx], 0);',
      '  atomicStore(&gp[2 * idx + 1], 0);',
      '  var vx = momx; var vy = momy;',
      // The node's own Coulomb coefficient: the mass-weighted mean of the friction of whatever is
      // sitting on it. PR.friction is only the fallback for an EMPTY node, which has no material and
      // therefore no result -- same structure as canonical grid_op.
      '  var fnode = PR.friction;',
      '  if (m > 0.0) { vx = vx / m; vy = vy / m; fnode = mfr / m; }',
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
      // ALL FOUR boundaries the same way (canonical grid_op): separating in the normal direction,
      // Coulomb friction on the tangent. The side walls used to zero BOTH components, which glued
      // material to them -- water thrown at a wall could not slide back down it. A wall is a wall.
      '  if (j < BOUND && vy < 0.0) {',              // floor
      '    vx = coulomb(vx, fnode * (-vy)); vy = 0.0;',
      '  }',
      '  if (j > N_GRID - BOUND && vy > 0.0) {',     // ceiling
      '    vx = coulomb(vx, fnode * vy); vy = 0.0;',
      '  }',
      '  if (i < BOUND && vx < 0.0) {',              // left wall
      '    vy = coulomb(vy, fnode * (-vx)); vx = 0.0;',
      '  }',
      '  if (i > N_GRID - BOUND && vx > 0.0) {',     // right wall
      '    vy = coulomb(vy, fnode * vx); vx = 0.0;',
      '  }',
      '  let ipm = 1.0 / max(PR.pMass, 1e-30);',
      '  gv[idx] = vec4<f32>(vx, vy, m * ipm, length(vec2<f32>(momx, momy)) * ipm);',
      '}',
      '',
      // ---------------------------------------------------------------- G2P
      '@compute @workgroup_size(' + WG_P + ')',
      'fn g2p(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let p = gid.x;',
      '  if (p >= PR.n) { return; }',
      '  var vp = vel[p];',
      '  let mid = i32(round(vp.w));',
      '  if (mid > M_SAND) { return; }',
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
      '  let np = pos[p] + PR.dt * nv;',
      '  pos[p] = clamp(np, vec2<f32>(FLOOR, FLOOR), vec2<f32>(CEIL, CEIL));',
      '',
      '  let Fp = Fm[p];',
      '  if (mid == M_FLUID) {',
      '    Fm[p] = vec4<f32>(Fp.x * (1.0 + PR.dt * (c00 + c11)), 0.0, 0.0, 0.0);',
      '  } else {',
      '    let g00 = 1.0 + PR.dt * c00; let g01 = PR.dt * c01;',
      '    let g10 = PR.dt * c10;       let g11 = 1.0 + PR.dt * c11;',
      '    let tr = vec4<f32>(g00 * Fp.x + g01 * Fp.z, g00 * Fp.y + g01 * Fp.w,',
      '                       g10 * Fp.x + g11 * Fp.z, g10 * Fp.y + g11 * Fp.w);',
      '    if (mid == M_ELASTIC) {',
      '      Fm[p] = tr;',
      '    } else if (mid == M_SNOW) {',
      // Stomakhin: clamp the singular values into a BOX. Cohesive -- the admissible set does not
      // shrink as the confining pressure falls, which is why snow can stand a vertical wall.
      '      let sv = svd2(tr);',
      '      let s0 = min(max(sv.s.x, 1.0 - TC_S), 1.0 + TS_S);',
      '      let s1 = min(max(sv.s.y, 1.0 - TC_S), 1.0 + TS_S);',
      '      vp.z = vp.z * (sv.s.x * sv.s.y) / (s0 * s1);',
      '      Fm[p] = mul_usv(sv.u, vec2<f32>(s0, s1), sv.v);',
      '    } else {',
      // Drucker-Prager (Klar et al. 2016, Alg. 3): project the log strain onto a CONE. Cohesionless
      // -- the admissible shear shrinks with the confining pressure, so sand cannot stand a wall.
      '      let sv = svd2(tr);',
      '      let e0 = log(max(abs(sv.s.x), 1e-4));',
      '      let e1 = log(max(abs(sv.s.y), 1e-4));',
      '      let trE = e0 + e1 + vp.z;',
      '      var q0 = 1.0; var q1 = 1.0;',
      '      if (trE >= 0.0) {',
      '        vp.z = trE;',                          // cone TIP: no tension, remember the expansion
      '      } else {',
      '        vp.z = 0.0;',
      '        let eh0 = e0 - trE * 0.5;',
      '        let eh1 = e1 - trE * 0.5;',
      '        let ehn = sqrt(eh0 * eh0 + eh1 * eh1) + 1e-20;',
      '        let dg = ehn + (2.0 * LA_A + 2.0 * MU_A) / (2.0 * MU_A) * trE * ALPHA_A;',
      '        if (dg <= 0.0) { q0 = sv.s.x; q1 = sv.s.y; }',
      '        else { q0 = exp(e0 - dg / ehn * eh0); q1 = exp(e1 - dg / ehn * eh1); }',
      '      }',
      '      Fm[p] = mul_usv(sv.u, vec2<f32>(q0, q1), sv.v);',
      '    }',
      '  }',
      '  vel[p] = vec4<f32>(nv, vp.z, vp.w);',
      '  Cm[p] = vec4<f32>(c00, c01, c10, c11);',
      '}',
      '',
      // ---------------------------------------------------------------- interactive edits
      // Marking a particle dead (id 4) rather than compacting on the GPU: compaction needs a prefix
      // sum and a second buffer, and the host compacts on pointer-up anyway, where it is free.
      '@compute @workgroup_size(' + WG_P + ')',
      'fn erase(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let p = gid.x;',
      '  if (p >= PR.n) { return; }',
      '  let d = pos[p] - vec2<f32>(PR.pokeX, PR.pokeY);',
      '  if (dot(d, d) < PR.eraseR * PR.eraseR) {',
      '    vel[p] = vec4<f32>(0.0, 0.0, 0.0, 5.0);',
      '  }',
      '}',
      '',
      '@compute @workgroup_size(' + WG_G + ')',
      'fn empty(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let idx = i32(gid.x);',
      '  if (idx >= N_CELL) { return; }',
      '  if (PR.n == 4294967295u) { gv[idx].w = 1.0; }',   // never true; keeps the store live
      '}'
    ].join('\n');
  }

  // ------------------------------------------------------------------ SVD unit-test shader
  // Deliberately a SEPARATE module and pipeline: the SVD has to be provable in isolation, on
  // matrices chosen to break it, before it is allowed anywhere near the simulation. A subtly wrong
  // SVD produces plausible-looking motion, which is exactly the failure a visual check survives.
  function svdTestShader() {
    return [
      SVD_WGSL,
      '@group(0) @binding(0) var<storage, read> A : array<vec4<f32>>;',
      '@group(0) @binding(1) var<storage, read_write> Uo : array<vec4<f32>>;',
      '@group(0) @binding(2) var<storage, read_write> So : array<vec4<f32>>;',
      '@group(0) @binding(3) var<storage, read_write> Vo : array<vec4<f32>>;',
      '@compute @workgroup_size(64)',
      'fn main(@builtin(global_invocation_id) gid : vec3<u32>) {',
      '  let i = gid.x;',
      '  if (i >= arrayLength(&A)) { return; }',
      '  let r = svd2(A[i]);',
      '  Uo[i] = r.u;',
      '  So[i] = vec4<f32>(r.s, 0.0, 0.0);',
      '  Vo[i] = r.v;',
      '}'
    ].join('\n');
  }

  // ------------------------------------------------------------------ render shader
  // Three views, all reading the simulation buffers directly -- nothing ever crosses back to the
  // CPU to be drawn.
  //   'blob'  the material view, and the one that carries all four TREATMENTS. Two passes plus an
  //           optional third:
  //             A  particles splat into an rgba16float target whose FOUR CHANNELS ARE THE FOUR
  //                MATERIALS' weights, not a premultiplied colour. This is what lets one resolve
  //                pass shade water like water and snow like snow: the colour is recoverable as the
  //                weighted mean of the frozen palette, and the per-material weight is what says
  //                WHICH treatment a pixel belongs to. It costs nothing over the old (colour*w, w)
  //                layout -- same texture, same format, same fill, same four neighbour taps.
  //             B  one full-screen resolve that reconstructs the surface (weight -> mask ->
  //                gradient -> thickness) and then branches to the treatment of whichever material
  //                dominates the pixel.
  //             C  sand only: six irregular grains per particle, drawn over the packed body.
  //   'grid'  node mass, straight out of gv.z.
  //   'pts'   one hard square per particle, material-coloured. Low sample count, crisp delivery.
  //
  // WHAT EACH TREATMENT IS (T-020 proposed these in Taichi; this is the WGSL port of the ones the
  // user chose, and T-020's Taichi costs do NOT transfer -- they are re-measured in verify/):
  //   WATER   option B, "film": the same reconstruction as the glass option but with NO background
  //           sampling and NO chromatic dispersion, and well under half the absorption. Chosen over
  //           the glass option because the demo's background is a flat dark gradient -- there is
  //           nothing behind the water worth refracting, so option A would have paid for a
  //           background sample and a three-tap dispersion to bend an almost-constant colour.
  //   SNOW    option A, "powder": the wet-plastic glint deleted, a soft powder fringe, thin snow
  //           brightened (light gets through it), packed crevices darkened, and a fine crystal
  //           grain.
  //   SAND    option A, "grains over a packed body": a matte, occluded body with six hashed grains
  //           per particle drawn on top (NOT option B, which was loose grains and no body).
  //   RUBBER  NOT a new treatment -- the shipped one, with exactly two changes: a smaller splat
  //           kernel (so particles merge into one continuous blob less aggressively) and an
  //           explicit dark border band just inside the silhouette.
  function renderShader(fmt) {
    var cols = ORDER.map(function (m) {
      var c = MAT[m].color, r = parseInt(c.slice(1, 3), 16) / 255,
        g = parseInt(c.slice(3, 5), 16) / 255, b = parseInt(c.slice(5, 7), 16) / 255;
      return 'vec3<f32>(' + f(r.toFixed(4)) + ', ' + f(g.toFixed(4)) + ', ' + f(b.toFixed(4)) + ')';
    });
    return [
      // packRef / thickChar / normAmp / featherW are the WATER RECONSTRUCTION's calibration, and
      // every one of them is a PHYSICAL quantity turned into pixels by the host (see calibrate()):
      //   packRef   the accumulated splat weight a fully-packed region of one material reaches.
      //             T-020's masks are thresholded at a FIXED FRACTION of full packing, not at a
      //             per-frame percentile, which is what keeps a thin sheet of spray reading as thin.
      //   thickChar px of distance-to-surface per unit optical depth.
      //   normAmp   how hard the surface gradient tilts the normal (T-020: 3.0 * res/1080).
      //   featherW  the width, in HALF-res px, of the band over which the normal field turns on.
      'struct RParams { radius : f32, aspect : f32, n : u32, view : u32,',
      '                 massRef : f32, iso : f32, dimAlpha : f32, odd : u32,',
      '                 packRef : f32, thickChar : f32, normAmp : f32, featherW : f32, };',
      '@group(0) @binding(0) var<uniform> R : RParams;',
      '@group(0) @binding(1) var<storage, read> pos : array<vec2<f32>>;',
      '@group(0) @binding(2) var<storage, read> vel : array<vec4<f32>>;',
      '@group(0) @binding(3) var<storage, read> gv  : array<vec4<f32>>;',
      '',
      'const C_FLUID = ' + cols[0] + ';',
      'const C_ELASTIC = ' + cols[1] + ';',
      'const C_SNOW = ' + cols[2] + ';',
      'const C_SAND = ' + cols[3] + ';',
      'const M_FLUID : i32 = ' + ID.fluid + ';',
      'const M_ELASTIC : i32 = ' + ID.elastic + ';',
      'const M_SNOW : i32 = ' + ID.snow + ';',
      'const M_SAND : i32 = ' + ID.sand + ';',
      '',
      // RUBBER'S TWO CHANGES, both rooted here. The kernel is 0.78x the radius, so a rubber
      // particle spreads its weight over 0.61x the area and neighbours merge into one smooth blob
      // far less readily -- the silhouette follows the particles instead of a heavy blur of them.
      // The weight is multiplied by 1/0.61 to compensate, so the ACCUMULATED field keeps the same
      // scale as the other three and one global iso still means the same thing for all four.
      // (Without that gain rubber would simply lose every interface pixel to whatever it touches.)
      'const RAD_E : f32 = 0.78;',
      'const GAIN_E : f32 = ' + f((1.0 / (0.78 * 0.78)).toFixed(5)) + ';',
      '',
      'fn matColor(m : i32) -> vec3<f32> {',
      '  if (m == M_FLUID) { return C_FLUID; }',
      '  if (m == M_ELASTIC) { return C_ELASTIC; }',
      '  if (m == M_SNOW) { return C_SNOW; }',
      '  return C_SAND;',
      '}',
      'fn matMask(m : i32) -> vec4<f32> {',
      '  return vec4<f32>(f32(m == M_FLUID), f32(m == M_ELASTIC),',
      '                   f32(m == M_SNOW), f32(m == M_SAND));',
      '}',
      // WATER's palette, straight from sim/material_render.py's PAL slots 4/5/6/10/11. Note that
      // C_FLUID is NOT in this list: the previously shipped water tinted the material's flat albedo,
      // which is why it came out as bright poster-paint blue. A body of water has no albedo -- what
      // you see is a shallow tint fading into a deep tint with optical depth, plus the sky.
      'const W_DEEP : vec3<f32> = vec3<f32>(0.02, 0.16, 0.30);',
      'const W_SHALLOW : vec3<f32> = vec3<f32>(0.20, 0.50, 0.60);',
      'const W_FOAM : vec3<f32> = vec3<f32>(0.93, 0.97, 1.0);',
      'const ENV_SKY : vec3<f32> = vec3<f32>(0.60, 0.74, 0.90);',
      'const ENV_GND : vec3<f32> = vec3<f32>(0.10, 0.13, 0.17);',
      'const ISO_FILL : f32 = 0.24;',
      'const THICK_MAX : f32 = 3.2;',
      'const JFA_NONE : f32 = -1.0;',
      '',
      'fn hash21(p : vec2<f32>) -> f32 {',
      '  var q = fract(p * vec2<f32>(0.1031, 0.1030));',
      '  q = q + vec2<f32>(dot(q, q.yx + vec2<f32>(33.33, 33.33)));',
      '  return fract((q.x + q.y) * q.x * 37.719);',
      '}',
      '',
      'struct VOut { @builtin(position) p : vec4<f32>, @location(0) uv : vec2<f32>,',
      '              @location(1) col : vec3<f32>, @location(2) sp : f32,',
      '              @location(3) mask : vec4<f32>, @location(4) shade : f32, };',
      '',
      // the unit quad, derived arithmetically -- dynamic indexing of a const array is a portability
      // hazard across WGSL implementations and this costs nothing
      'fn quad(vi : u32) -> vec2<f32> {',
      '  let qx = select(0.0, 1.0, vi == 1u || vi == 4u || vi == 5u);',
      '  let qy = select(0.0, 1.0, vi == 2u || vi == 3u || vi == 5u);',
      '  return vec2<f32>(qx, qy) * 2.0 - vec2<f32>(1.0, 1.0);',
      '}',
      // one clipped point, so the triangle has zero area and never reaches a fragment (uv is also
      // pushed outside the disc, so the discard would catch it anyway)
      'fn deadOut() -> VOut {',
      '  var o : VOut;',
      '  o.p = vec4<f32>(-5.0, -5.0, 0.0, 1.0); o.uv = vec2<f32>(9.0, 9.0);',
      '  o.col = vec3<f32>(0.0); o.sp = 0.0; o.mask = vec4<f32>(0.0); o.shade = 0.0;',
      '  return o;',
      '}',
      '',
      '@vertex fn vs_particles(@builtin(vertex_index) vi : u32,',
      '                        @builtin(instance_index) ii : u32) -> VOut {',
      '  let q = quad(vi);',
      '  let c = pos[ii];',
      '  let vv = vel[ii];',
      '  let mid = i32(round(vv.w));',
      '  if (mid > M_SAND) { return deadOut(); }',
      '  var o : VOut;',
      '  var rad = R.radius;',
      '  var gain = 1.0;',
      // R.dimAlpha is the treatment switch (1 = this task's, 0 = the previously shipped one).
      // Rubber's smaller kernel is half of rubber's change, so it has to be switchable too, or
      // the "before" half of the comparison would silently already contain it.
      '  if (mid == M_ELASTIC && R.dimAlpha > 0.5) { rad = rad * RAD_E; gain = GAIN_E; }',
      '  let ndc = vec2<f32>(c.x * 2.0 - 1.0, c.y * 2.0 - 1.0) + q * vec2<f32>(rad, rad * R.aspect);',
      '  o.p = vec4<f32>(ndc, 0.0, 1.0);',
      '  o.uv = q;',
      // R.odd is a single particle index the host may paint differently. It is a rendering hook
      // only -- that particle takes exactly its material's constitutive path in the solver.
      '  o.col = select(matColor(mid), vec3<f32>(0.176, 0.549, 0.365), ii == R.odd);',
      '  o.sp = length(vv.xy);',
      '  o.mask = matMask(mid) * gain;',
      '  o.shade = 1.0;',
      '  return o;',
      '}',
      '',
      // ---- SAND, pass C: six irregular grains per particle -------------------------------------
      // Drawn as n*6 instances of the unit quad; instance ii carries particle ii/6 and grain ii%6,
      // so there is no extra buffer and no sorting. Every non-sand instance degenerates in one
      // branch before it costs a fragment. Each grain gets its OWN hashed offset, radius and
      // brightness, which is what stops six copies of one sprite reading as a fat blurry particle.
      '@vertex fn vs_grain(@builtin(vertex_index) vi : u32,',
      '                    @builtin(instance_index) ii : u32) -> VOut {',
      '  let p = ii / 6u;',
      '  let g = ii % 6u;',
      '  if (p >= R.n) { return deadOut(); }',
      '  let vv = vel[p];',
      '  if (i32(round(vv.w)) != M_SAND) { return deadOut(); }',
      '  var o : VOut;',
      '  let seed = vec2<f32>(f32(p) * 0.7548 + f32(g) * 3.113, f32(p) * 0.3357 - f32(g) * 1.771);',
      '  let h0 = hash21(seed);',
      '  let h1 = hash21(seed + vec2<f32>(19.19, 7.77));',
      '  let h2 = hash21(seed + vec2<f32>(3.33, 41.41));',
      '  let ang = h0 * 6.28318;',
      '  let rr  = R.radius * (0.30 + 0.42 * h1);',              // how far out this grain sits
      '  let gr  = R.radius * (0.19 + 0.19 * h2);',              // this grain own radius
      '  let c = pos[p] + vec2<f32>(cos(ang), sin(ang)) * rr * 0.5;',
      '  let q = quad(vi);',
      '  let ndc = vec2<f32>(c.x * 2.0 - 1.0, c.y * 2.0 - 1.0) + q * vec2<f32>(gr, gr * R.aspect);',
      '  o.p = vec4<f32>(ndc, 0.0, 1.0);',
      '  o.uv = q;',
      '  o.col = C_SAND;',
      '  o.sp = 0.0;',
      '  o.mask = vec4<f32>(0.0, 0.0, 0.0, 1.0);',
      '  o.shade = 0.66 + 0.66 * h2;',                           // per-grain brightness
      '  return o;',
      '}',
      '',
      '@fragment fn fs_grain(o : VOut) -> @location(0) vec4<f32> {',
      '  let r2 = dot(o.uv, o.uv);',
      '  if (r2 > 1.0) { discard; }',
      // a grain is a little lit pebble, not a gaussian: a hard core with one lit shoulder
      '  let k = sqrt(max(0.0, 1.0 - r2));',
      '  let lit = 0.52 + 0.48 * clamp(dot(normalize(vec3<f32>(o.uv, k * 1.35)),',
      '                                    normalize(vec3<f32>(-0.42, 0.62, 0.66))), 0.0, 1.0);',
      '  let c = o.col * o.shade * lit;',
      '  return vec4<f32>(c, smoothstep(1.0, 0.45, r2) * 0.90);',
      '}',
      '',
      // ---- the odd particle, in the material view ----------------------------------------------
      // In a four-channel accumulation there is no room for a colour that is not one of the four,
      // so the one particle painted off-palette is drawn as its own quad after the resolve.
      '@vertex fn vs_odd(@builtin(vertex_index) vi : u32) -> VOut {',
      '  if (R.odd >= R.n) { return deadOut(); }',
      '  var o : VOut;',
      '  let q = quad(vi);',
      '  let c = pos[R.odd];',
      '  let rad = R.radius * 0.55;',
      '  let ndc = vec2<f32>(c.x * 2.0 - 1.0, c.y * 2.0 - 1.0) + q * vec2<f32>(rad, rad * R.aspect);',
      '  o.p = vec4<f32>(ndc, 0.0, 1.0);',
      '  o.uv = q;',
      '  o.col = vec3<f32>(0.176, 0.549, 0.365);',
      '  o.sp = 0.0; o.mask = vec4<f32>(0.0); o.shade = 1.0;',
      '  return o;',
      '}',
      '@fragment fn fs_odd(o : VOut) -> @location(0) vec4<f32> {',
      '  let r2 = dot(o.uv, o.uv);',
      '  if (r2 > 1.0) { discard; }',
      '  return vec4<f32>(o.col, 0.5 * smoothstep(1.0, 0.2, r2));',
      '}',
      '',
      '@vertex fn vs_full(@builtin(vertex_index) vi : u32) -> VOut {',
      '  let q = quad(vi);',
      '  var o : VOut;',
      '  o.p = vec4<f32>(q, 0.0, 1.0);',
      '  o.uv = q * 0.5 + vec2<f32>(0.5, 0.5);',
      '  o.col = vec3<f32>(0.0); o.sp = 0.0; o.mask = vec4<f32>(0.0); o.shade = 0.0;',
      '  return o;',
      '}',
      '',
      // ---- pass A of the material view: additive PER-MATERIAL weight accumulation ----
      '@fragment fn fs_splat(o : VOut) -> @location(0) vec4<f32> {',
      '  let r2 = dot(o.uv, o.uv);',
      '  if (r2 > 1.0) { discard; }',
      '  let w = (1.0 - r2) * (1.0 - r2);',                   // smooth compact kernel
      '  return o.mask * w;',
      '}',
      '',
      // ---- hard particles view ----
      '@fragment fn fs_particles(o : VOut) -> @location(0) vec4<f32> {',
      '  let r2 = dot(o.uv, o.uv);',
      '  if (r2 > 1.0) { discard; }',
      '  let shade = 0.55 + 0.45 * sqrt(max(0.0, 1.0 - r2));',
      '  let hot = clamp(o.sp / 2.5, 0.0, 1.0);',
      '  let c = mix(o.col, vec3<f32>(1.0, 0.99, 0.95), 0.55 * hot);',
      '  return vec4<f32>(c * shade, 0.95);',
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
      '@fragment fn fs_grid(o : VOut) -> @location(0) vec4<f32> {',
      '  let gi = vec2<i32>(clamp(o.uv * f32(' + N_GRID + '), vec2<f32>(0.0), vec2<f32>(' + (N_GRID - 1) + '.0)));',
      '  let cell = gv[gi.x * ' + N_GRID + ' + gi.y];',
      '  let t = log(1.0 + cell.z) / log(1.0 + max(R.massRef, 1e-6));',
      '  return vec4<f32>(ramp(t), 1.0);',
      '}',
      '',
      // ---- pass B of the material view: reconstruct once, then shade PER MATERIAL ----
      // The reconstruction is shared and the treatments are not, and that split is the whole idea.
      // The solver hands over one cloud of points; the reconstruction turns it into a surface with
      // a thickness and a normal; only then does the material decide what light does to it. The
      // same solver output is what reads as water, as powder or as grains.
      // group 1 is the screen-space chain's I/O. Binding 0 is "whatever this pass reads" -- the
      // accumulation for the first blur, then each intermediate in turn -- and binding 2 is the
      // finished distance field, which only the resolve reads.
      '@group(1) @binding(0) var acc : texture_2d<f32>;',
      '@group(1) @binding(1) var samp : sampler;',
      '@group(1) @binding(2) var distT : texture_2d<f32>;',
      // group 2 is the per-pass argument (blur direction + sigma, or the jump-flood step), bound at
      // a DYNAMIC OFFSET into one buffer. That is the only way to vary an argument between passes
      // inside a single command encoder: a queue.writeBuffer between two beginRenderPass calls is
      // ordered on the QUEUE, so it would apply to every pass in the submission, not just the next.
      'struct SParams { arg : vec4<f32>, dim : vec4<f32>, };',
      '@group(2) @binding(0) var<uniform> S : SParams;',
      '',
      // ======================= THE WATER RECONSTRUCTION (T-020's, ported) =======================
      // The shipped treatment already had T-020's water SHADING. What it did not have is the thing
      // the shading reads: a screen-space iso-surface. It reconstructed optical thickness from four
      // neighbour taps of the raw splat accumulation, so thickness inherited the density's
      // particle-scale lumpiness and the water stayed speckled -- the exact "smoothie" the proposal
      // existed to fix. The chain below is sim/material_render.py:build_masks in WGSL:
      //
      //   blur -> threshold to a BINARY body -> jump-flood distance transform -> optical thickness
      //
      // The load-bearing idea is the SEPARATION. One threshold cannot both decide "is there water
      // here" (wants a generous cut and pinholes sealed) and "which way does the surface face"
      // (wants fine slope). So the filled body owns opacity and thickness, and a wide band around
      // its iso-surface owns the normal. Density noise then reaches NEITHER, which is the whole
      // difference between a smoothie and a body of water.
      //
      // Everything after the first blur runs at HALF resolution. Optical thickness is a smooth,
      // low-frequency quantity -- it is the one part of the frame that does not need full res --
      // and halving costs a quarter of the pixels on eleven passes.
      '',
      // Separable Gaussian, 13 taps at sigma/2.2 spacing (+-2.95 sigma) through a LINEAR sampler,
      // so the horizontal pass also does the 2x downsample for free.
      '@fragment fn fs_blur(o : VOut) -> @location(0) f32 {',
      '  let uv = o.p.xy / S.dim.xy;',
      '  let sg = max(S.arg.z, 0.35);',
      '  let sp = sg / 2.2;',
      '  let stp = S.arg.xy * (sp / S.dim.xy);',
      '  var s = 0.0;',
      '  var wsum = 0.0;',
      '  for (var k = -6; k <= 6; k = k + 1) {',
      '    let x = f32(k) * sp;',
      '    let w = exp(-0.5 * x * x / (sg * sg));',
      '    s = s + w * textureSampleLevel(acc, samp, uv + stp * f32(k), 0.0).x;',
      '    wsum = wsum + w;',
      '  }',
      '  return s / wsum;',
      '}',
      '',
      // Threshold at a FIXED FRACTION of full packing (R.packRef, a physical quantity the host
      // computes from the particle density and the splat radius) rather than at a per-frame
      // percentile -- that is what keeps a thin sheet of spray reading as thin. Then seed every
      // pixel OUTSIDE the body with its own coordinate, so the distance that comes out is "px to
      // the nearest pixel that is not water".
      '@fragment fn fs_seed(o : VOut) -> @location(0) vec2<f32> {',
      '  let v = textureLoad(acc, vec2<i32>(o.p.xy), 0).x;',
      '  if (v > R.packRef * ISO_FILL) { return vec2<f32>(JFA_NONE, JFA_NONE); }',
      '  return o.p.xy;',
      '}',
      '',
      // One jump-flooding pass: look at the eight neighbours `step` away plus self and keep the
      // nearest seed any of them knows about. log2(range) passes give the whole field, which is why
      // a distance transform is affordable at all in a real-time frame.
      '@fragment fn fs_jfa(o : VOut) -> @location(0) vec2<f32> {',
      '  let ip = vec2<i32>(o.p.xy);',
      '  let hi = vec2<i32>(textureDimensions(acc)) - vec2<i32>(1, 1);',
      '  let st = i32(S.arg.x);',
      '  var best = textureLoad(acc, ip, 0).xy;',
      '  var bd = 1.0e18;',
      '  if (best.x >= 0.0) { let e = best - o.p.xy; bd = dot(e, e); }',
      '  for (var dy = -1; dy <= 1; dy = dy + 1) {',
      '    for (var dx = -1; dx <= 1; dx = dx + 1) {',
      '      let q = ip + vec2<i32>(dx * st, dy * st);',
      '      if (q.x >= 0 && q.y >= 0 && q.x <= hi.x && q.y <= hi.y) {',
      '        let c = textureLoad(acc, q, 0).xy;',
      '        if (c.x >= 0.0) {',
      '          let e = c - o.p.xy;',
      '          let d2 = dot(e, e);',
      '          if (d2 < bd) { bd = d2; best = c; }',
      '        }',
      '      }',
      '    }',
      '  }',
      '  return best;',
      '}',
      '',
      // Seeds -> distance, with a 3x3 box on the way out. That box plus the bilinear upsample the
      // resolve does is this pipeline's blur(distr, 2.0): a distance field straight off a
      // thresholded mask is quantised in whole pixels, and Beer-Lambert turns quantisation into
      // visible banding. A pixel no pass ever reached (deeper than the flood's range) gets the cap,
      // which is already past where the absorption saturates.
      '@fragment fn fs_dist(o : VOut) -> @location(0) f32 {',
      '  let ip = vec2<i32>(o.p.xy);',
      '  let hi = vec2<i32>(textureDimensions(acc)) - vec2<i32>(1, 1);',
      '  let cap = S.arg.y;',
      '  var s = 0.0;',
      '  for (var dy = -1; dy <= 1; dy = dy + 1) {',
      '    for (var dx = -1; dx <= 1; dx = dx + 1) {',
      '      let q = clamp(ip + vec2<i32>(dx, dy), vec2<i32>(0, 0), hi);',
      '      let c = textureLoad(acc, q, 0).xy;',
      '      var d = cap;',
      '      if (c.x >= 0.0) { d = min(length(c - (vec2<f32>(q) + vec2<f32>(0.5, 0.5))), cap); }',
      '      s = s + d;',
      '    }',
      '  }',
      '  return s / 9.0;',
      '}',
      '',
      // The normal field: ~0 outside the body, ~1 inside, turning on over featherW half-res px --
      // MUCH wider than the opacity feather. That asymmetry is deliberate and is T-020's: a hard
      // silhouette (the clean surface line) with a soft normal (a rounded rim and a genuinely FLAT
      // interior). The second term is the whisper of interior slope thickness contributes; at the
      // demo scale it tilts the normal by a fraction of a degree, and its job is only to stop a
      // dead-flat slab from having literally nothing to catch the light.
      // sim/material_render.py:tonemap, at gain 1. The other three treatments do not need it and do
      // not get it: their colours are palette tints already in display space. Water's are not --
      // Beer-Lambert and Fresnel produce a RADIANCE, and a radiance written straight to an 8-bit
      // non-sRGB swapchain comes out as the near-black pool the first attempt at this produced. The
      // curve is part of T-020's water, not decoration on top of it.
      'fn tonemapW(c : vec3<f32>) -> vec3<f32> {',
      '  let x = clamp((c / (c + vec3<f32>(0.9))) * 1.55, vec3<f32>(0.0), vec3<f32>(1.0));',
      '  return pow(x, vec3<f32>(1.0 / 1.15));',
      '}',
      'fn waterD(uv : vec2<f32>) -> f32 { return textureSampleLevel(distT, samp, uv, 0.0).x; }',
      'fn normField(d : f32) -> f32 {',
      '  return smoothstep(0.0, R.featherW, d)',
      '       + 0.6 * clamp(2.0 * d / R.thickChar, 0.0, THICK_MAX) / THICK_MAX;',
      '}',
      '',
      // ---------------- WATER, T-020 option B "film" ----------------
      // Beer-Lambert depth colour, a Fresnel-weighted sky, a grazing rim, a tight glint, and foam
      // gated to the fast/thin surface band. Option B over option A ("glass") because the demo's
      // background is a flat dark gradient: option A would pay for a background sample and a
      // three-tap chromatic dispersion to bend an almost-constant colour.
      //
      // Returns PREMULTIPLIED colour + alpha. The background does not get sampled and refracted --
      // it gets TRANSMITTED, and transmission is exactly what an alpha under premultiplied blending
      // means. So `1 - alpha` carries Beer-Lambert's exp(-absorb*t), and the water is see-through
      // where it is thin without the resolve ever reading what is behind it.
      'fn shadeWater(uv : vec2<f32>, hpx : vec2<f32>, px : vec2<f32>, motion : f32) -> vec4<f32> {',
      '  let d0 = waterD(uv);',
      '  let m = smoothstep(0.0, 1.2, d0);',
      '  if (m <= 0.002) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }',
      '  let dR = waterD(uv + vec2<f32>(hpx.x, 0.0));',
      '  let dL = waterD(uv - vec2<f32>(hpx.x, 0.0));',
      '  let dD = waterD(uv + vec2<f32>(0.0, hpx.y));',
      '  let dU = waterD(uv - vec2<f32>(0.0, hpx.y));',
      // per-FULL-res-px central difference: the taps are one half-res texel (= 2 px) either side
      '  let gx = (normField(dR) - normField(dL)) * 0.25;',
      '  let gy = (normField(dD) - normField(dU)) * 0.25;',
      // +y is DOWN in screen space and UP in the field, hence the asymmetric sign on the two
      '  let nv = normalize(vec3<f32>(-gx * R.normAmp, gy * R.normAmp, 2.1));',
      '  let tt = clamp(2.0 * d0 / R.thickChar, 0.0, THICK_MAX);',
      '  let trans = exp(-0.52 * tt);',
      '  var col = (W_SHALLOW * trans + W_DEEP * (1.0 - trans)) * (1.0 - trans);',
      '  let dim = 1.0 - 0.16 * smoothstep(0.6, THICK_MAX, tt);',
      '  col = col * dim;',
      // Fresnel: cos(theta) IS nv.z, so the flat interior stays clear and the rim turns mirror
      '  let cosT = clamp(nv.z, 0.0, 1.0);',
      '  let F = 0.02 + 0.98 * pow(1.0 - cosT, 5.0);',
      '  let sky = clamp(0.55 + 0.45 * (2.0 * nv.z * nv.y), 0.0, 1.0);',
      '  let env = ENV_SKY * sky + ENV_GND * (1.0 - sky);',
      '  var emit = col * (1.0 - F) + env * F;',
      '  emit = emit + ENV_SKY * (0.34 * pow(1.0 - cosT, 3.0));',
      '  let ld = normalize(vec3<f32>(-0.55, 0.72, 0.55));',
      '  let hv = normalize(ld + vec3<f32>(0.0, 0.0, 1.0));',
      '  let ndh = clamp(dot(nv, hv), 0.0, 1.0);',
      '  emit = emit + vec3<f32>(1.0, 1.0, 0.97) * (3.4 * pow(ndh, 70.0) + 0.10 * pow(ndh, 8.0));',
      // foam: fast AND at the surface, or genuinely thin. Ungated foam speckles the interior, which
      // is how the old water got its permanent white froth. `motion` is the GRID velocity under the
      // pixel -- already bound to this shader for the grid view, so the cue is free.
      '  let band = smoothstep(0.02, 0.14, length(vec2<f32>(gx, gy)) * R.featherW);',
      '  let thin = m * smoothstep(0.16, 0.02, tt);',
      '  var fo = clamp(0.95 * (0.9 * motion * band + 0.22 * thin), 0.0, 1.0);',
      '  fo = fo * (0.55 + 0.45 * hash21(px * 0.35));',
      '  emit = mix(emit, W_FOAM, fo);',
      '  let through = trans * dim * (1.0 - F) * (1.0 - fo);',
      '  return vec4<f32>(tonemapW(emit) * m, m * (1.0 - through));',
      '}',
      '',
      '@fragment fn fs_resolve(o : VOut) -> @location(0) vec4<f32> {',
      '  let ip = vec2<i32>(o.p.xy);',
      '  let c = textureLoad(acc, ip, 0);',
      '  let a = c.x + c.y + c.z + c.w;',
      '  if (a < R.iso * 0.30) { discard; }',
      // which material owns this pixel, and the colour of the mixture actually here
      '  var mid = M_FLUID; var best = c.x;',
      '  if (c.y > best) { mid = M_ELASTIC; best = c.y; }',
      '  if (c.z > best) { mid = M_SNOW; best = c.z; }',
      '  if (c.w > best) { mid = M_SAND; best = c.w; }',
      '  let base = (c.x * C_FLUID + c.y * C_ELASTIC + c.z * C_SNOW + c.w * C_SAND) / max(a, 1e-6);',
      // the surface, from the gradient of the TOTAL weight field. Four taps, exactly as before.
      '  let lf = textureLoad(acc, ip + vec2<i32>(-2, 0), 0);',
      '  let rt = textureLoad(acc, ip + vec2<i32>( 2, 0), 0);',
      '  let dn = textureLoad(acc, ip + vec2<i32>(0, -2), 0);',
      '  let up = textureLoad(acc, ip + vec2<i32>(0,  2), 0);',
      '  let al = lf.x + lf.y + lf.z + lf.w;',
      '  let ar = rt.x + rt.y + rt.z + rt.w;',
      '  let ad = dn.x + dn.y + dn.z + dn.w;',
      '  let au = up.x + up.y + up.z + up.w;',
      '  let grad = vec2<f32>(al - ar, ad - au);',
      '  let nrm = normalize(vec3<f32>(grad, 1.6 * R.iso));',
      '  let ld = normalize(vec3<f32>(-0.42, 0.62, 0.66));',
      '  let hv = normalize(ld + vec3<f32>(0.0, 0.0, 1.0));',
      '  let edge = smoothstep(R.iso * 0.30, R.iso * 1.25, a);',
      '  let th = a / max(R.iso, 1e-6);',                       // thickness in iso units
      // Laplacian of the weight field: positive where the surface is CONCAVE, i.e. in a crevice
      // between two lumps. Snow and sand both use it, for opposite-sounding reasons (packed shadow,
      // contact occlusion), and it costs nothing -- the four taps are already loaded.
      '  let lap = (al + ar + ad + au) - 4.0 * a;',
      '  let conc = clamp(lap / (R.iso * 1.4), 0.0, 1.0);',
      '  var col = vec3<f32>(0.0);',
      '  var alp = 1.0;',
      '',
      '  if (mid == M_FLUID && R.dimAlpha > 1.5) {',
      // ---------------- WATER, option B "film", on the RECONSTRUCTED iso-surface ----------------
      // Nothing above this line reaches the water any more. `a`, `grad`, `nrm`, `th` and `edge` are
      // all local functions of the raw splat accumulation, and a Poisson sample of particles makes
      // every one of them lumpy at the particle scale. Water reads its surface out of the distance
      // field instead, and returns premultiplied straight away.
      '    let dims = vec2<f32>(textureDimensions(acc));',
      '    let uvf = o.p.xy / dims;',
      '    let hpx = 1.0 / vec2<f32>(textureDimensions(distT));',
      '    let gi = vec2<i32>(clamp(o.uv * f32(' + N_GRID + '), vec2<f32>(0.0), vec2<f32>(' + (N_GRID - 1) + '.0)));',
      '    let motion = smoothstep(1.7, 3.7, length(gv[gi.x * ' + N_GRID + ' + gi.y].xy));',
      '    return shadeWater(uvf, hpx, o.p.xy, motion);',
      '  } else if (mid == M_FLUID) {',
      // ---- the water T-027 ORIGINALLY SHIPPED, kept live so this rework has an honest before ----
      // It is T-020's film SHADING (Beer-Lambert, a tight specular, a Fresnel rim, gated foam) over
      // a thickness and a normal read from four local taps of the raw accumulation -- which is
      // precisely the half that was missing. Keeping it costs one branch and buys a before/after
      // that changes ONE thing: snow, sand, rubber, the physics and the particle positions are
      // identical on both sides, so anything that differs is the water reconstruction.
      '    let absorb = exp(-vec3<f32>(0.46, 0.19, 0.10) * min(th, 7.0) * 0.62);',
      '    let dif = 0.62 + 0.38 * max(0.0, dot(nrm, ld));',
      '    let spec = pow(max(0.0, dot(nrm, hv)), 46.0);',
      '    let fres = pow(1.0 - clamp(nrm.z, 0.0, 1.0), 3.0);',
      '    col = base * absorb * dif;',
      '    col = col + vec3<f32>(0.78, 0.93, 1.0) * spec * 0.85;',
      '    col = col + vec3<f32>(0.30, 0.58, 0.76) * fres * 0.55;',
      '    let foam = (1.0 - edge) * smoothstep(0.10, 0.55, length(grad) / max(R.iso, 1e-6));',
      '    col = mix(col, vec3<f32>(0.86, 0.95, 1.0), foam * 0.5);',
      '    alp = min(1.0, 0.26 + 1.6 * edge);',
      '  } else if (mid == M_SNOW) {',
      // ---------------- SNOW, option A "powder" ----------------
      // No specular at all: the glint is what made the old snow read as wet plastic. Wrap lighting
      // instead (light scatters INTO the shadowed side of a powder), thin snow brightened because
      // light gets through it, crevices darkened because it does not, and a fine crystal grain.
      '    let wrap = 0.58 + 0.42 * (0.5 + 0.5 * dot(nrm, ld));',
      '    let thin = 1.0 - smoothstep(0.55, 2.8, th);',
      '    let g = hash21(o.p.xy * 0.7);',
      '    col = base * wrap * (1.0 + 0.34 * thin) * (1.0 - 0.40 * conc);',
      '    col = col + vec3<f32>(1.0, 0.99, 0.97) * (g - 0.5) * 0.075;',
      // the powder fringe: the outermost band stays bright and stays translucent, so the boundary
      // reads as loose crystals rather than as the edge of a solid
      '    col = col + vec3<f32>(0.72, 0.80, 0.95) * (1.0 - edge) * 0.22;',
      '    alp = min(1.0, 0.12 + 1.35 * edge);',
      '  } else if (mid == M_SAND) {',
      // ---------------- SAND, option A: the PACKED BODY (the grains are pass C) ----------
      // Matte -- dry sand has no specular worth drawing -- with contact occlusion in the crevices
      // and a slight darkening with depth, so a heap reads as a heap and not as a flat shape.
      '    let dif = 0.56 + 0.44 * max(0.0, dot(nrm, ld));',
      '    let ao = 1.0 - 0.50 * conc;',
      '    col = base * dif * ao * (1.0 - 0.14 * clamp(th * 0.22, 0.0, 1.0));',
      '    alp = min(1.0, 0.34 + 1.7 * edge);',
      '  } else {',
      // ---------------- RUBBER: the SHIPPED treatment, plus a border ----------------
      // Everything down to the mix() is the demo's existing resolve, unchanged. The border band is
      // the second of rubber's two changes: a dark ring in the outer shell of the silhouette, which
      // is what makes a rubber body read as an object with an edge rather than as a soft glow.
      '    let dif = 0.66 + 0.34 * max(0.0, dot(nrm, ld));',
      '    let spec = pow(max(0.0, dot(nrm, hv)), 26.0);',
      '    col = base * dif + vec3<f32>(0.9, 0.97, 1.0) * spec * 0.42;',
      '    col = mix(col * 1.30, col, edge);',
      '    let band = smoothstep(R.iso * 0.30, R.iso * 0.66, a) *',
      '               (1.0 - smoothstep(R.iso * 0.66, R.iso * 1.10, a));',
      '    col = mix(col, base * 0.20, band * 0.62);',
      '    alp = min(1.0, 0.30 + 1.7 * edge);',
      '  }',
      // PREMULTIPLIED. Snow, sand and rubber are bit-for-bit what they were -- col*alp under
      // srcFactor 'one' is the same arithmetic as col under srcFactor 'src-alpha'. The switch
      // exists for water: a transmitting body needs its background coefficient (Beer-Lambert's
      // exp(-absorb*t)) to be independent of how much colour it adds, and only premultiplied alpha
      // can express that without the shader sampling what is behind it.
      '  return vec4<f32>(col * alp, alp);',
      '}',
      '',
      // ---- the PREVIOUS single treatment, kept as a live alternative -------------------------
      // Not nostalgia: a before/after that changes two things at once is not evidence. With this
      // entry point the same build draws the same scene under the old shading and the new shading,
      // so the RENDERING comparison isolates the rendering, and the PHYSICS comparison can be drawn
      // identically on both sides. It reads the four-channel accumulation and collapses it to
      // (colour, weight) first -- exactly what the old buffer held -- so the arithmetic below is the
      // previously shipped fs_resolve verbatim.
      '@fragment fn fs_resolve_mvp(o : VOut) -> @location(0) vec4<f32> {',
      '  let ip = vec2<i32>(o.p.xy);',
      '  let cc = textureLoad(acc, ip, 0);',
      '  let a = cc.x + cc.y + cc.z + cc.w;',
      '  if (a < R.iso * 0.34) { discard; }',
      '  let base = (cc.x * C_FLUID + cc.y * C_ELASTIC + cc.z * C_SNOW + cc.w * C_SAND) / max(a, 1e-6);',
      '  let lf = textureLoad(acc, ip + vec2<i32>(-2, 0), 0);',
      '  let rt = textureLoad(acc, ip + vec2<i32>( 2, 0), 0);',
      '  let dn = textureLoad(acc, ip + vec2<i32>(0, -2), 0);',
      '  let up = textureLoad(acc, ip + vec2<i32>(0,  2), 0);',
      '  let l = lf.x + lf.y + lf.z + lf.w;',
      '  let r = rt.x + rt.y + rt.z + rt.w;',
      '  let d = dn.x + dn.y + dn.z + dn.w;',
      '  let u = up.x + up.y + up.z + up.w;',
      '  let nrm = normalize(vec3<f32>(l - r, d - u, 1.6 * R.iso));',
      '  let ld = normalize(vec3<f32>(-0.42, 0.62, 0.66));',
      '  let dif = 0.66 + 0.34 * max(0.0, dot(nrm, ld));',
      '  let spec = pow(max(0.0, dot(nrm, normalize(ld + vec3<f32>(0.0, 0.0, 1.0)))), 26.0);',
      '  let edge = smoothstep(R.iso * 0.34, R.iso * 1.25, a);',
      '  var col = base * dif + vec3<f32>(0.9, 0.97, 1.0) * spec * 0.42;',
      '  col = mix(col * 1.30, col, edge);',
      '  return vec4<f32>(col, min(1.0, 0.30 + 1.6 * edge));',
      '}'
    ].join('\n');
  }

  // ------------------------------------------------------------------ helpers
  function supported() { return (typeof navigator !== 'undefined') && !!navigator.gpu; }

  // Why navigator.gpu might be missing, told apart. `file://` and plain-HTTP LAN origins are not
  // secure contexts, so the API is HIDDEN there regardless of what the device supports -- reporting
  // that as "unsupported" blames the device for a transport problem, and has already produced one
  // wrong "this device has no WebGPU" conclusion in this project.
  function probe() {
    if (typeof navigator === 'undefined') return { ok: false, why: 'no navigator' };
    if (!navigator.gpu) {
      var sec = (typeof window !== 'undefined') && window.isSecureContext;
      return sec
        ? { ok: false, why: 'unsupported', text: 'this browser has no WebGPU' }
        : { ok: false, why: 'insecure', text: 'WebGPU is hidden outside a secure context — this page needs HTTPS or localhost' };
    }
    return { ok: true, why: 'present' };
  }

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

  // One grid means ONE timestep: min(dt) over the materials actually present, exactly as canonical
  // sim.physics.shared_dt does. Adding snow to any scene halves the timestep for everything in it.
  function sharedDt(names) {
    var d = Infinity;
    for (var i = 0; i < names.length; i++) d = Math.min(d, MAT[names[i]].dt);
    return isFinite(d) ? d : MAT.elastic.dt;
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
    // dispatch a no-op and the simulation "runs" at the speed of doing nothing. Never unlistened.
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

  // ------------------------------------------------------------------ SVD unit test (isolated)
  async function svdSelfTest(mats) {
    var device = await getDevice();
    var k = mats.length / 4;
    var U = GPUBufferUsage;
    device.pushErrorScope('validation');
    var inB = device.createBuffer({ size: k * 16, usage: U.STORAGE | U.COPY_DST });
    function ob() { return device.createBuffer({ size: k * 16, usage: U.STORAGE | U.COPY_SRC }); }
    var uB = ob(), sB = ob(), vB = ob();
    var rd = device.createBuffer({ size: k * 16 * 3, usage: U.COPY_DST | U.MAP_READ });
    device.queue.writeBuffer(inB, 0, mats);
    var mod = device.createShaderModule({ code: svdTestShader(), label: 'svd-test' });
    var ci = await mod.getCompilationInfo();
    var errs = ci.messages.filter(function (m) { return m.type === 'error'; });
    if (errs.length) throw new Error('svd WGSL: ' + errs.map(function (m) { return m.lineNum + ': ' + m.message; }).join(' | '));
    var pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    var bg = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [inB, uB, sB, vB].map(function (b, i) { return { binding: i, resource: { buffer: b } }; })
    });
    var enc = device.createCommandEncoder();
    var pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(k / 64));
    pass.end();
    enc.copyBufferToBuffer(uB, 0, rd, 0, k * 16);
    enc.copyBufferToBuffer(sB, 0, rd, k * 16, k * 16);
    enc.copyBufferToBuffer(vB, 0, rd, k * 32, k * 16);
    device.queue.submit([enc.finish()]);
    var err = await device.popErrorScope();
    if (err) throw new Error('svd test setup: ' + err.message);
    await rd.mapAsync(GPUMapMode.READ);
    var out = new Float32Array(rd.getMappedRange().slice(0));
    rd.unmap();
    [inB, uB, sB, vB, rd].forEach(function (b) { b.destroy(); });
    return { U: out.subarray(0, k * 4), S: out.subarray(k * 4, k * 8), V: out.subarray(k * 8, k * 12) };
  }

  // ------------------------------------------------------------------ the simulator
  async function createSim(opts) {
    opts = opts || {};
    var device = await getDevice();
    var cap = opts.capacity | 0 || 8192;           // buffers are allocated once, `n` grows into them
    var kM = opts.kM !== undefined ? opts.kM : P.kM;
    var kV = opts.kV !== undefined ? opts.kV : P.kV;

    // ONE density for the whole domain. Canonical simulate_multi lets each group carry its own
    // p_vol; here every particle is seeded at the same particle-per-area density, so p_vol is a
    // single uniform. That is a demo choice (constant mass density everywhere) and it is what the
    // verification scenes reproduce on the canonical side by matching area/n exactly.
    var pVol = opts.pVol !== undefined ? opts.pVol : (Math.PI * 0.11 * 0.11) / 2048;
    // REFERENCE mass, not "the" mass: since phys-c518316a4a05 each material has its own rho, and the
    // real particle mass pVol*rho[mid] is computed in the shader. This is the unit the fixed-point
    // accumulators count in (rho = 1, i.e. water) and the unit the grid-mass view is reported in.
    var pMass = pVol * P.p_rho;
    var dt = opts.dt !== undefined ? opts.dt : MAT.elastic.dt;
    var massScale = Math.pow(2, kM) / pMass;
    var momScale = Math.pow(2, kV) / pMass;
    var n = 0;

    device.pushErrorScope('validation');
    var U = GPUBufferUsage;
    function buf(bytes, usage) { return device.createBuffer({ size: bytes, usage: usage }); }
    var STO = U.STORAGE | U.COPY_DST | U.COPY_SRC;

    var posBuf = buf(cap * 8, STO);
    var velBuf = buf(cap * 16, STO);
    var CBuf = buf(cap * 16, STO);
    var FBuf = buf(cap * 16, STO);
    var gmBuf = buf(N_CELL * 8, STO);          // 2 u32 per cell: mass, and mass*friction
    var gpBuf = buf(N_CELL * 8, STO);
    var gvBuf = buf(N_CELL * 16, STO);
    var uBuf = buf(96, U.UNIFORM | U.COPY_DST);
    var readBuf = device.createBuffer({ size: cap * 8, usage: U.COPY_DST | U.MAP_READ });
    // pos (cap*8) | vel (cap*16) | C (cap*16) | F (cap*16) = cap*56, one map for the whole state
    var stateReadBuf = device.createBuffer({ size: cap * 56, usage: U.COPY_DST | U.MAP_READ });
    var gridReadBuf = device.createBuffer({ size: N_CELL * 16, usage: U.COPY_DST | U.MAP_READ });

    var mod = device.createShaderModule({ code: buildShader(), label: 'mpm4' });
    var info = await mod.getCompilationInfo();
    var errs = info.messages.filter(function (m) { return m.type === 'error'; });
    if (errs.length) throw new Error('WGSL: ' + errs.map(function (m) { return m.lineNum + ': ' + m.message; }).join(' | '));

    function sb(b) {
      return { binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } };
    }
    var layout = device.createBindGroupLayout({
      label: 'mpm4-compute-bgl',
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        sb(1), sb(2), sb(3), sb(4), sb(5), sb(6), sb(7)]
    });
    var bind = device.createBindGroup({
      label: 'mpm4-compute-bg', layout: layout,
      entries: [uBuf, posBuf, velBuf, CBuf, FBuf, gmBuf, gpBuf, gvBuf]
        .map(function (b, i) { return { binding: i, resource: { buffer: b } }; })
    });
    var pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    function pipe(entry) {
      return device.createComputePipeline({ layout: pl, label: entry,
        compute: { module: mod, entryPoint: entry } });
    }
    var pClear = pipe('clear_grid'), pP2G = pipe('p2g'), pGrid = pipe('grid_op'),
      pG2P = pipe('g2p'), pErase = pipe('erase'), pEmpty = pipe('empty');

    var setupError = await device.popErrorScope();
    if (setupError) throw new Error('WebGPU setup: ' + setupError.message);

    var poke = { on: false, x: 0, y: 0, vx: 0, vy: 0, radius: 0.075, rate: 900.0, spring: 16.0 };
    var eraseR = 0.06;
    var uArr = new ArrayBuffer(96);
    var uF = new Float32Array(uArr), uU = new Uint32Array(uArr);

    function writeUniform() {
      uF[0] = dt; uF[1] = pMass; uF[2] = pVol; uF[3] = P.gravity; uF[4] = P.FRICTION;
      uF[5] = massScale; uF[6] = 1.0 / massScale; uF[7] = momScale; uF[8] = 1.0 / momScale;
      uU[9] = n >>> 0; uU[10] = poke.on ? 1 : 0;
      uF[11] = poke.x; uF[12] = poke.y; uF[13] = poke.vx; uF[14] = poke.vy;
      uF[15] = poke.radius; uF[16] = poke.rate; uF[17] = poke.spring;
      uF[18] = eraseR;
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
      device.queue.writeBuffer(gmBuf, 0, new Uint32Array(2 * N_CELL));
      device.queue.writeBuffer(gpBuf, 0, new Int32Array(2 * N_CELL));
      device.queue.writeBuffer(gvBuf, 0, new Float32Array(4 * N_CELL));
    }

    // Append a group of particles of ONE material. Returns how many actually fit.
    function add(material, pts, v0x, v0y) {
      var k = Math.min(pts.length >> 1, cap - n);
      if (k <= 0) return 0;
      var mid = ID[material];
      var xs = new Float32Array(2 * k), vs = new Float32Array(4 * k);
      var Cs = new Float32Array(4 * k), Fs = new Float32Array(4 * k);
      for (var p = 0; p < k; p++) {
        xs[2 * p] = pts[2 * p]; xs[2 * p + 1] = pts[2 * p + 1];
        vs[4 * p] = v0x || 0; vs[4 * p + 1] = v0y || 0;
        // canonical init_state: sand's Jp is an ADDITIVE log-strain starting at 0, every other
        // material's is a MULTIPLICATIVE volume ratio starting at 1. Same field, different books.
        vs[4 * p + 2] = (material === 'sand') ? 0.0 : 1.0;
        vs[4 * p + 3] = mid;
        Fs[4 * p] = 1; Fs[4 * p + 3] = (material === 'fluid') ? 0 : 1;
      }
      device.queue.writeBuffer(posBuf, n * 8, xs);
      device.queue.writeBuffer(velBuf, n * 16, vs);
      device.queue.writeBuffer(CBuf, n * 16, Cs);
      device.queue.writeBuffer(FBuf, n * 16, Fs);
      n += k;
      writeUniform();
      return k;
    }

    function clearAll() { n = 0; zeroGrid(); writeUniform(); }

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
      if (opt.erase) { pass.setPipeline(pErase); pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / WG_P))); }
      var phases = opt.phases || 'pgG';
      var doP = phases.indexOf('p') >= 0, doG = phases.indexOf('g') >= 0,
        doQ = phases.indexOf('G') >= 0, doE = phases.indexOf('e') >= 0;
      var P_WG = Math.max(1, Math.ceil(n / WG_P));
      for (var s = 0; s < substeps; s++) {
        if (doE) { pass.setPipeline(pEmpty); pass.dispatchWorkgroups(GRID_WG); }
        if (doP && n > 0) { pass.setPipeline(pP2G); pass.dispatchWorkgroups(P_WG); }
        if (doG) { pass.setPipeline(pGrid); pass.dispatchWorkgroups(GRID_WG); }
        if (doQ && n > 0) { pass.setPipeline(pG2P); pass.dispatchWorkgroups(P_WG); }
      }
      pass.end();
      if (opt.timed && qset) {
        enc.resolveQuerySet(qset, 0, 2, qResolve, 0);
        enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 16);
      }
      if (opt.readback && n > 0) enc.copyBufferToBuffer(posBuf, 0, readBuf, 0, n * 8);
      if (opt.gridReadback) enc.copyBufferToBuffer(gvBuf, 0, gridReadBuf, 0, N_CELL * 16);
      if (opt.stateReadback && n > 0) {
        enc.copyBufferToBuffer(posBuf, 0, stateReadBuf, 0, n * 8);
        enc.copyBufferToBuffer(velBuf, 0, stateReadBuf, cap * 8, n * 16);
        enc.copyBufferToBuffer(CBuf, 0, stateReadBuf, cap * 24, n * 16);
        enc.copyBufferToBuffer(FBuf, 0, stateReadBuf, cap * 40, n * 16);
      }
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
      if (n === 0) return new Float32Array(0);
      await readBuf.mapAsync(GPUMapMode.READ, 0, n * 8);
      var out = new Float32Array(readBuf.getMappedRange(0, n * 8).slice(0));
      readBuf.unmap();
      return out;
    }
    async function readGrid() {
      await gridReadBuf.mapAsync(GPUMapMode.READ);
      var out = new Float32Array(gridReadBuf.getMappedRange().slice(0));
      gridReadBuf.unmap();
      return out;
    }
    // Full particle state, used only by the host-side compaction after an erase.
    async function readState() {
      if (n === 0) return null;
      await stateReadBuf.mapAsync(GPUMapMode.READ);
      var all = new Float32Array(stateReadBuf.getMappedRange().slice(0));
      stateReadBuf.unmap();
      return { pos: all.subarray(0, n * 2), vel: all.subarray(cap * 2, cap * 2 + n * 4),
        C: all.subarray(cap * 6, cap * 6 + n * 4), F: all.subarray(cap * 10, cap * 10 + n * 4) };
    }

    // Drop every particle the eraser marked dead, in one host round trip on pointer-up (where a
    // 1 MB readback is invisible), rather than a GPU stream compaction nobody needs 60 times a
    // second. Reclaims the slots, so the particle count in the HUD is the truth.
    // Returns {n, counts} from the SAME readback that did the compaction. Counting from a second,
    // later read of the staging buffer is a trap: the buffer still holds the pre-compaction state,
    // so the tally silently disagrees with n.
    async function compact() {
      encodeFrame(0, { stateReadback: true });
      var st = await readState();
      var counts = [0, 0, 0, 0];
      if (!st) return { n: 0, counts: counts };
      var live = 0, i, q, mid;
      var xs = new Float32Array(n * 2), vs = new Float32Array(n * 4),
        cs = new Float32Array(n * 4), fs = new Float32Array(n * 4);
      for (i = 0; i < n; i++) {
        mid = Math.round(st.vel[4 * i + 3]);
        if (mid > ID.sand) continue;
        counts[mid]++;
        xs[2 * live] = st.pos[2 * i]; xs[2 * live + 1] = st.pos[2 * i + 1];
        for (q = 0; q < 4; q++) {
          vs[4 * live + q] = st.vel[4 * i + q];
          cs[4 * live + q] = st.C[4 * i + q];
          fs[4 * live + q] = st.F[4 * i + q];
        }
        live++;
      }
      if (live !== n) {
        device.queue.writeBuffer(posBuf, 0, xs, 0, live * 2);
        device.queue.writeBuffer(velBuf, 0, vs, 0, live * 4);
        device.queue.writeBuffer(CBuf, 0, cs, 0, live * 4);
        device.queue.writeBuffer(FBuf, 0, fs, 0, live * 4);
        n = live;
        writeUniform();
      }
      return { n: n, counts: counts };
    }

    return {
      get n() { return n; },
      capacity: cap, device: device,
      params: {
        dt: dt, pVol: pVol, pMass: pMass, n_grid: N_GRID, gravity: P.gravity,
        friction: P.FRICTION, kM: kM, kV: kV, massScale: massScale, momScale: momScale,
        physics_version: P.physics_version
      },
      buffers: { pos: posBuf, vel: velBuf, gv: gvBuf },
      poke: poke,
      setEraseRadius: function (r) { eraseR = r; writeUniform(); },
      setDt: function (d) { dt = d; writeUniform(); },
      getDt: function () { return dt; },
      syncUniform: writeUniform,
      add: add, clear: clearAll, compact: compact,
      encodeFrame: encodeFrame, lastGpuNanos: lastGpuNanos,
      readPositions: readPositions, readGrid: readGrid, readState: readState,
      dispatchesPerFrame: function (substeps) { return 3 * substeps; },
      idle: function () { return device.queue.onSubmittedWorkDone(); },
      destroy: function () {
        [posBuf, velBuf, CBuf, FBuf, gmBuf, gpBuf, gvBuf, uBuf, readBuf, stateReadBuf, gridReadBuf]
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
    var mod = device.createShaderModule({ code: renderShader(fmt), label: 'mpm4-render' });
    var ci = await mod.getCompilationInfo();
    var re = ci.messages.filter(function (m) { return m.type === 'error'; });
    if (re.length) throw new Error('render WGSL: ' + re.map(function (m) { return m.lineNum + ': ' + m.message; }).join(' | '));

    var rBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    var layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
      ]
    });
    // Two group-1 shapes. The reconstruction passes read ONE texture through a linear sampler;
    // the final resolve additionally reads the finished distance field. A pipeline layout may be a
    // superset of what its entry point uses, which is why one WGSL module covers both.
    var texLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }
      ]
    });
    var texLayoutR = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
      ]
    });
    // The per-pass argument, bound at a dynamic offset. See the SParams comment in the shader.
    var stepLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 } }]
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
    var linSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    var pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    var pl2 = device.createPipelineLayout({ bindGroupLayouts: [layout, texLayout] });
    var plR = device.createPipelineLayout({ bindGroupLayouts: [layout, texLayoutR] });
    var plS = device.createPipelineLayout({ bindGroupLayouts: [layout, texLayout, stepLayout] });
    var blendAlpha = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
    // The resolve is premultiplied so water can transmit; see the return in fs_resolve.
    var blendPre = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
    var blendAdd = { color: { srcFactor: 'one', dstFactor: 'one' },
      alpha: { srcFactor: 'one', dstFactor: 'one' } };
    function recon(entry, format) {
      return device.createRenderPipeline({
        layout: plS, vertex: { module: mod, entryPoint: 'vs_full' },
        fragment: { module: mod, entryPoint: entry, targets: [{ format: format }] },
        primitive: { topology: 'triangle-list' }
      });
    }

    var pParticles = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_particles' },
      fragment: { module: mod, entryPoint: 'fs_particles', targets: [{ format: fmt, blend: blendAlpha }] },
      primitive: { topology: 'triangle-list' }
    });
    var pSplat = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_particles' },
      fragment: { module: mod, entryPoint: 'fs_splat', targets: [{ format: 'rgba16float', blend: blendAdd }] },
      primitive: { topology: 'triangle-list' }
    });
    var pResolve = device.createRenderPipeline({
      layout: plR, vertex: { module: mod, entryPoint: 'vs_full' },
      fragment: { module: mod, entryPoint: 'fs_resolve', targets: [{ format: fmt, blend: blendPre }] },
      primitive: { topology: 'triangle-list' }
    });
    // the water reconstruction, in the order it runs: blur -> threshold+seed -> flood -> distance
    var pBlur = recon('fs_blur', 'r16float');
    var pSeed = recon('fs_seed', 'rg16float');
    var pJfa = recon('fs_jfa', 'rg16float');
    var pDist = recon('fs_dist', 'r16float');
    var pResolveMvp = device.createRenderPipeline({
      layout: pl2, vertex: { module: mod, entryPoint: 'vs_full' },
      fragment: { module: mod, entryPoint: 'fs_resolve_mvp', targets: [{ format: fmt, blend: blendAlpha }] },
      primitive: { topology: 'triangle-list' }
    });
    var pGridView = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_full' },
      fragment: { module: mod, entryPoint: 'fs_grid', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' }
    });
    // sand's grains and the off-palette particle: two small alpha-blended passes over the resolve
    var pGrain = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_grain' },
      fragment: { module: mod, entryPoint: 'fs_grain', targets: [{ format: fmt, blend: blendAlpha }] },
      primitive: { topology: 'triangle-list' }
    });
    var pOdd = device.createRenderPipeline({
      layout: pl, vertex: { module: mod, entryPoint: 'vs_odd' },
      fragment: { module: mod, entryPoint: 'fs_odd', targets: [{ format: fmt, blend: blendAlpha }] },
      primitive: { topology: 'triangle-list' }
    });
    var err = await device.popErrorScope();
    if (err) throw new Error('renderer setup: ' + err.message);

    // ---- the screen-space targets -----------------------------------------------------------
    // accTex is full resolution; everything the water reconstruction touches is HALF, so the
    // eleven extra passes cost a quarter of the pixels each. RECON_DIV is the one knob: raise it
    // and the whole chain gets cheaper and softer at exactly the same rate.
    var RECON_DIV = 2;
    var STEP_STRIDE = 256;                          // minUniformBufferOffsetAlignment
    var MAX_JFA = 8;
    var accTex = null, accView = null, accBind = null, accW = 0, accH = 0;
    var reconTex = [], reconBind = [], resolveBind = null, hw = 0, hh = 0, jfaSteps = [];
    var sBuf = device.createBuffer({
      size: STEP_STRIDE * (2 + MAX_JFA), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    var sBind = device.createBindGroup({
      layout: stepLayout, entries: [{ binding: 0, resource: { buffer: sBuf, size: 32 } }] });

    function halfTex(format) {
      return device.createTexture({ size: [hw, hh], format: format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    }
    function ensureAcc() {
      if (accTex && accW === canvas.width && accH === canvas.height) return;
      if (accTex) accTex.destroy();
      reconTex.forEach(function (t) { t.destroy(); });
      accW = canvas.width; accH = canvas.height;
      hw = Math.max(1, Math.ceil(accW / RECON_DIV));
      hh = Math.max(1, Math.ceil(accH / RECON_DIV));
      accTex = device.createTexture({
        size: [accW, accH], format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      });
      accView = accTex.createView();
      // 0 blurH result | 1 blurV result (the smoothed density) | 2,3 jump-flood ping-pong |
      // 4 the finished distance field. rg16float holds a seed coordinate exactly (f16 is exact on
      // integers to 2048, and half of any sane canvas is well under that) and -1 as "no seed"; the
      // 16-bit float formats are also FILTERABLE, which 32-bit float is not, and the resolve needs
      // to bilinearly upsample the distance field.
      reconTex = ['r16float', 'r16float', 'rg16float', 'rg16float', 'r16float'].map(halfTex);
      var views = reconTex.map(function (t) { return t.createView(); });
      accBind = device.createBindGroup({ layout: texLayout,
        entries: [{ binding: 0, resource: accView }, { binding: 1, resource: linSamp }] });
      reconBind = views.map(function (v) {
        return device.createBindGroup({ layout: texLayout,
          entries: [{ binding: 0, resource: v }, { binding: 1, resource: linSamp }] });
      });
      resolveBind = device.createBindGroup({ layout: texLayoutR, entries: [
        { binding: 0, resource: accView }, { binding: 1, resource: linSamp },
        { binding: 2, resource: views[4] }
      ] });
      reconView = views;
      calW = 0;                                     // force a recalibration at the new size
    }
    var reconView = [];

    var ru = new Float32Array(12);
    var ruU = new Uint32Array(ru.buffer);
    var CLEAR = { r: 0.0235, g: 0.0353, b: 0.051, a: 1 };

    // ---- calibrating the reconstruction to the scene, in physical units ----------------------
    // T-020's masks are thresholded at a fraction of FULL PACKING and its lengths are quoted
    // relative to a 1080 px frame, so both have to be re-derived for whatever canvas this is.
    //
    //   packRef  a particle deposits (pi/3)*rpx^2 of weight (the integral of the (1-r^2)^2 kernel
    //            over its disc), and a packed region holds 1/pVol particles per unit domain area,
    //            so a packed pixel accumulates (1/pVol)/(W*H) * (pi/3) * rpx^2.
    //   sigma    T-020 blurs a POINT histogram to sigma 6.5 px at 720, i.e. 0.00903 of the frame.
    //            The demo's splat is already a disc of radius rpx, worth sigma 0.354*rpx, so only
    //            the REMAINDER in quadrature has to be added -- and it is added at half resolution.
    var sArg = new Float32Array(8 * (2 + MAX_JFA));
    var calW = 0, calH = 0, calR = 0;
    function calibrate(radius) {
      var thickChar = 55.0 * (accW / 1080.0);       // px of distance per unit optical depth
      ru[8] = (1.0 / sim.params.pVol) / (accW * accH) * (Math.PI / 3.0)
        * Math.pow(radius * 0.5 * accW, 2);         // packRef
      ru[9] = thickChar;
      ru[10] = 3.0 * (accW / 1080.0);               // normAmp, T-020's 3.0 * res/1080
      // featherW: the normal band, in HALF-res px, matched to T-020's sigma 2.97 at 720. A
      // smoothstep of width e has slope 1.5/e where a gaussian step has 0.399/sigma; equating the
      // two (and converting per-full-px to per-half-px) is where the 0.00776 comes from.
      ru[11] = Math.max(1.5, 0.00776 * accW);
      if (calW === accW && calH === accH && calR === radius) return;
      calW = accW; calH = accH; calR = radius;
      var sigSplat = 0.354 * (radius * 0.5 * accW);
      var sigWant = 0.009028 * accW;
      var sigExtra = Math.sqrt(Math.max(0, sigWant * sigWant - sigSplat * sigSplat)) / RECON_DIV;
      sigExtra = Math.max(0.4, sigExtra);
      sArg.fill(0);
      sArg[0] = 1; sArg[2] = sigExtra; sArg[4] = hw; sArg[5] = hh;   // slot 0: horizontal
      sArg[9] = 1; sArg[10] = sigExtra; sArg[12] = hw; sArg[13] = hh; // slot 1: vertical
      // How far the flood has to reach: past the cap the absorption has saturated, so a pixel the
      // flood never touched can safely be told "you are at the cap". A start step s reaches 2s-1.
      var capHalf = thickChar * 3.2 / RECON_DIV;
      var s = 1;
      while (2 * s - 1 < capHalf && s < 128) s *= 2;
      jfaSteps = [];
      for (var k = s; k >= 1 && jfaSteps.length < MAX_JFA; k = k >> 1) jfaSteps.push(k);
      jfaSteps.forEach(function (st, i) {
        var o = 8 * (2 + i);
        sArg[o] = st; sArg[o + 1] = capHalf * 1.05; sArg[o + 4] = hw; sArg[o + 5] = hh;
      });
      // fs_dist reads its cap out of the LAST jump-flood slot's argument, so no extra slot.
      for (var i = 0; i < 2 + MAX_JFA; i++) {
        device.queue.writeBuffer(sBuf, i * STEP_STRIDE, sArg, i * 8, 8);
      }
    }

    // Render cost is measured with the GPU's OWN clock, for the same reason the solver's is:
    // performance.now() is clamped to ~100 us in Chromium, and a screen-space treatment that costs
    // 0.3 ms is entirely inside that quantum. A host clock timing this reads the compositor.
    var rq = null, rqResolve = null, rqRead = null, rqBusy = false;
    if (_hasTimestamp) {
      rq = device.createQuerySet({ type: 'timestamp', count: 2 });
      rqResolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      rqRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    function tsw(on) {
      return (on && rq) ? { querySet: rq, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined;
    }

    return {
      draw: function (o) {
        o = o || {};
        var view = o.view || 'blob';                      // 'blob' | 'grid' | 'pts'
        var enc = device.createCommandEncoder();
        ru[0] = o.radius !== undefined ? o.radius : 0.012;
        ru[1] = canvas.width / canvas.height;
        ruU[2] = sim.n >>> 0;
        ruU[3] = view === 'grid' ? 1 : (view === 'pts' ? 2 : 0);
        ru[4] = o.massRef !== undefined ? o.massRef : 24.0;
        ru[5] = o.iso !== undefined ? o.iso : 1.0;
        var mvp = o.treatment === 'mvp';
        // The treatment switch, read by vs_particles (rubber's smaller kernel) and by fs_resolve
        // (which water):  0 = the pre-T-027 single treatment,  1 = T-027 as originally shipped
        // (four treatments, water off four local taps),  2 = + this rework's iso-surface water.
        var wchain = !mvp && o.water !== false;
        ru[6] = mvp ? 0.0 : (wchain ? 2.0 : 1.0);
        ruU[7] = (o.odd === undefined ? 0xffffffff : o.odd) >>> 0;

        if (view === 'blob') { ensureAcc(); calibrate(ru[0]); }  // fills ru[8..11] from the canvas
        device.queue.writeBuffer(rBuf, 0, ru);

        var timed = !!o.timed && !!rq && !rqBusy;
        if (view === 'blob') {
          var pa = enc.beginRenderPass({
            colorAttachments: [{ view: accView, clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: 'clear', storeOp: 'store' }],
            timestampWrites: timed ? { querySet: rq, beginningOfPassWriteIndex: 0 } : undefined
          });
          if (sim.n > 0) { pa.setBindGroup(0, bind); pa.setPipeline(pSplat); pa.draw(6, sim.n); }
          pa.end();
          // ---- the water reconstruction, between the splat and the shade -----------------------
          // Eleven half-resolution passes: two separable blurs (the first of which also does the
          // 2x downsample), a threshold-and-seed, the jump flood, and seeds -> distance. Skipped
          // outright when the scene holds no water, which is the honest cost model: this is
          // WATER's structure, not the frame's, and a scene with no fluid pays none of it.
          // `chainReps` is a MEASUREMENT hook and nothing else: the page always draws it once.
          // Chromium quantises timestamp-query results (65.536 us here), and the whole chain is
          // smaller than one quantum, so the only honest way to price it is to run it K times in
          // one timed region and take the slope. It writes the same textures every time, so the
          // pixels are identical and the GPU cannot skip the work.
          var creps = Math.max(1, o.chainReps || 1);
          if (wchain && sim.n > 0) {
            var chain = [
              { pipe: pBlur, src: accBind, dst: 0, slot: 0 },     // horizontal + downsample
              { pipe: pBlur, src: reconBind[0], dst: 1, slot: 1 },// vertical
              { pipe: pSeed, src: reconBind[1], dst: 2, slot: 1 } // threshold -> seeds
            ];
            var cur = 2;
            for (var j = 0; j < jfaSteps.length; j++) {
              chain.push({ pipe: pJfa, src: reconBind[cur], dst: 5 - cur, slot: 2 + j });
              cur = 5 - cur;                                      // ping-pong between 2 and 3
            }
            chain.push({ pipe: pDist, src: reconBind[cur], dst: 4,
              slot: 1 + jfaSteps.length });                       // reuses the last flood's cap
            for (var c = 0; c < chain.length * creps; c++) {
              var st = chain[c % chain.length];
              var rp = enc.beginRenderPass({ colorAttachments: [{ view: reconView[st.dst],
                clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
              rp.setBindGroup(0, bind);
              rp.setBindGroup(1, st.src);
              rp.setBindGroup(2, sBind, [st.slot * STEP_STRIDE]);
              rp.setPipeline(st.pipe); rp.draw(6, 1);
              rp.end();
            }
          }
          var pb = enc.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(),
              clearValue: CLEAR, loadOp: 'clear', storeOp: 'store' }],
            timestampWrites: timed ? { querySet: rq, endOfPassWriteIndex: 1 } : undefined
          });
          pb.setBindGroup(0, bind);
          pb.setBindGroup(1, mvp ? accBind : resolveBind);
          pb.setPipeline(mvp ? pResolveMvp : pResolve); pb.draw(6, 1);
          // Pass C, in the SAME render pass as the resolve so there is no extra attachment
          // load/store: six grains per particle over the packed sand body. Every instance whose
          // particle is not sand degenerates in the vertex shader, so this is only ever paid for
          // the sand that is actually in the scene -- but the instance COUNT is 6n either way,
          // which is why it is skipped outright when there is no sand.
          if (sim.n > 0 && !mvp && o.grains !== false) { pb.setPipeline(pGrain); pb.draw(6, sim.n * 6); }
          if (sim.n > 0) { pb.setPipeline(pOdd); pb.draw(6, 1); }
          pb.end();
        } else {
          var pass = enc.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(),
              clearValue: CLEAR, loadOp: 'clear', storeOp: 'store' }],
            timestampWrites: tsw(timed)
          });
          pass.setBindGroup(0, bind);
          if (view === 'grid') {
            pass.setPipeline(pGridView); pass.draw(6, 1);
          } else if (sim.n > 0) {
            pass.setPipeline(pParticles); pass.draw(6, sim.n);
          }
          pass.end();
        }
        if (timed) {
          rqBusy = true;
          enc.resolveQuerySet(rq, 0, 2, rqResolve, 0);
          enc.copyBufferToBuffer(rqResolve, 0, rqRead, 0, 16);
        }
        device.queue.submit([enc.finish()]);
      },
      // Device time for the LAST draw({timed:true}), across both passes of the material view.
      lastGpuNanos: async function () {
        if (!rq || !rqBusy) return null;
        await rqRead.mapAsync(GPUMapMode.READ);
        var t = new BigUint64Array(rqRead.getMappedRange().slice(0));
        rqRead.unmap();
        rqBusy = false;
        return Number(t[1] - t[0]);
      },
      // How many render passes the last blob draw actually issued, so a timing can be attributed.
      passCount: function (water) {
        return 2 + (water === false ? 0 : 3 + jfaSteps.length + 1);
      },
      destroy: function () {
        if (accTex) accTex.destroy();
        reconTex.forEach(function (t) { t.destroy(); });
        sBuf.destroy();
        if (rq) { rq.destroy(); rqResolve.destroy(); rqRead.destroy(); }
      }
    };
  }

  return {
    supported: supported, probe: probe, getDevice: getDevice, deviceInfo: deviceInfo, errors: errors,
    hasTimestamp: function () { return _hasTimestamp; },
    createSim: createSim, createRenderer: createRenderer, seedDisk: seedDisk, sharedDt: sharedDt,
    svdSelfTest: svdSelfTest,
    RENDER_TREATMENT: RENDER_TREATMENT,
    PARAMS: P, MAT: MAT, ORDER: ORDER, ID: ID, N_GRID: N_GRID, N_CELL: N_CELL,
    WG_P: WG_P, WG_G: WG_G, buildShader: buildShader, svdTestShader: svdTestShader
  };
});
