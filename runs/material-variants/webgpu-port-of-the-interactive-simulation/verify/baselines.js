// Re-time the EXISTING single-threaded JS port on exactly the scenes the WebGPU harness used, so
// the three-way comparison is one machine, one session, one scene family -- not three papers.
//
//   node verify/baselines.js <out.json>
//
// The JS port is imported unchanged from the task that produced it. Its params.js carries an older
// physics_version string, so this asserts every elastic constant is byte-identical to the one the
// WebGPU port generated; if a constant ever moves, this fails loudly instead of quietly comparing
// two different materials.
const fs = require('fs');
const path = require('path');

const RUN = path.resolve(__dirname, '..');
const JS_PORT = path.resolve(RUN, '..', 'interactive-simulation-of-one-material', 'web', 'mpm-elastic.js');
const MPM = require(JS_PORT);
const MINE = require(path.join(RUN, 'web', 'params.js'));

const SHARED = ['dim', 'n_grid', 'dx', 'inv_dx', 'p_rho', 'gravity', 'bound', 'floor_y', 'NU',
  'FRICTION', 'E', 'dt'];
const mismatch = SHARED.filter(k => MPM.PARAMS[k] !== MINE[k]);
if (mismatch.length) {
  console.error('ELASTIC CONSTANTS MOVED: ' + mismatch.map(k =>
    k + ' ' + MPM.PARAMS[k] + ' != ' + MINE[k]).join(', '));
  process.exit(2);
}

const P = MINE;
const SPF = Math.round((1 / 60) / P.dt);
const DENSITY0 = 2048 / (Math.PI * 0.11 * 0.11);
const SIDE_MAX = 0.90;

function boxScene(n, seed) {
  const side = Math.min(SIDE_MAX, Math.sqrt(n / DENSITY0));
  const x0 = 0.5 - side / 2, y0 = 0.5 - side / 2;
  let s = (seed >>> 0) || 1;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  const pts = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) { pts[2 * i] = x0 + rnd() * side; pts[2 * i + 1] = y0 + rnd() * side; }
  return { pts, area: side * side, side, particles_per_cell: n / (side * side * P.n_grid * P.n_grid) };
}

const NS = [500, 1000, 2048, 4096, 8192, 16384, 32768, 49152, 65536];
const out = { impl: 'javascript-single-thread', substeps_per_frame: SPF,
  physics_version_of_port: MPM.PARAMS.physics_version, node: process.version, rows: [] };

for (const n of NS) {
  const sc = boxScene(n, 12345);
  const sim = MPM.createSim({ n, area: sc.area, dt: P.dt });
  sim.seed(sc.pts, 0, 0);
  sim.substeps(200);                                  // warm the JIT and settle the active set
  const K = Math.max(200, Math.min(2000, Math.round(2e6 / n)));
  const t0 = process.hrtime.bigint();
  sim.substeps(K);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  const us = ms * 1000 / K;
  out.rows.push({
    n, particles_per_cell: sc.particles_per_cell, side: sc.side,
    us_per_substep: us, frame_ms: us * SPF / 1000, fps: 1000 / (us * SPF / 1000),
    active_cells: sim.activeCells()
  });
  console.error(`n=${n}\t${us.toFixed(1)} us/substep\t${(us * SPF / 1000).toFixed(2)} ms/frame`);
}

fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 2));
console.error('wrote ' + process.argv[2]);
