// Node driver: run the JS port on an initial condition exported from sim.physics and dump the
// trajectory as raw float32 so Python can score it against the canonical simulator.
//
//   node verify/run_port.js <ic.json> <outdir>
//
// It also runs a bit-for-bit sparse-vs-dense self test (the one optimization the port makes that
// is not present in the Taichi original) and a timestep sweep.

const fs = require('fs');
const path = require('path');
const MPM = require('../web/mpm-elastic.js');

const icPath = process.argv[2];
const outDir = process.argv[3];
const tag = process.argv[4] ? process.argv[4] + '_' : '';        // filename prefix for the outputs
const meta = JSON.parse(fs.readFileSync(icPath, 'utf8'));
const raw = fs.readFileSync(path.join(path.dirname(icPath), meta.pts_file));
const pts = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const V0 = meta.v0 || [0, 0];

function roll(dt, nFrames, spf, opts) {
  const sim = MPM.createSim(Object.assign({ n: meta.n, area: meta.area, dt: dt }, opts || {}));
  sim.seed(pts, V0[0], V0[1]);
  const out = new Float32Array(nFrames * meta.n * 2);
  let stable = true;
  const t0 = process.hrtime.bigint();
  for (let f = 0; f < nFrames; f++) {
    sim.substeps(spf);
    out.set(sim.x, f * meta.n * 2);
    if (stable && !sim.finite()) stable = false;
  }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  return { out, stable, ms, steps: nFrames * spf, sim };
}

const report = { n: meta.n, area: meta.area, T: meta.T, n_frames: meta.n_frames };

// ---- 1. the reference roll at canonical dt -----------------------------------------------
const dt0 = MPM.PARAMS.dt;
const spf0 = Math.max(1, Math.round((meta.T / meta.n_frames) / dt0));
const base = roll(dt0, meta.n_frames, spf0);
fs.writeFileSync(path.join(outDir, `port_${tag}canonical.f32`), Buffer.from(base.out.buffer));
report.canonical = {
  dt: dt0, spf: spf0, steps: base.steps, stable: base.stable, wall_ms: base.ms,
  us_per_step: (base.ms * 1000) / base.steps,
  physics_version: MPM.PARAMS.physics_version
};

// ---- 2. sparse vs dense: is the one optimization exact? ----------------------------------
const denseFrames = Math.min(20, meta.n_frames);
const a = roll(dt0, denseFrames, spf0, { dense: false });
const b = roll(dt0, denseFrames, spf0, { dense: true });
let maxDiff = 0, nDiff = 0;
for (let i = 0; i < a.out.length; i++) {
  const d = Math.abs(a.out[i] - b.out[i]);
  if (d > 0) nDiff++;
  if (d > maxDiff) maxDiff = d;
}
report.sparse_vs_dense = {
  frames: denseFrames, steps: a.steps, max_abs_diff: maxDiff, n_differing_values: nDiff,
  bit_identical: nDiff === 0,
  sparse_ms: a.ms, dense_ms: b.ms, speedup: b.ms / a.ms
};

// ---- 3. is the timestep a performance knob? ----------------------------------------------
report.dt_sweep = [];
for (const mult of [1, 1.5, 2, 2.5, 3, 4]) {
  const dt = dt0 * mult;
  const spf = Math.max(1, Math.round((meta.T / meta.n_frames) / dt));
  const r = roll(dt, meta.n_frames, spf);
  // spread of the final frame, a cheap blow-up detector
  let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9, bad = 0;
  const off = (meta.n_frames - 1) * meta.n * 2;
  for (let p = 0; p < meta.n; p++) {
    const px = r.out[off + 2 * p], py = r.out[off + 2 * p + 1];
    if (!isFinite(px) || !isFinite(py)) { bad++; continue; }
    if (px < minx) minx = px; if (px > maxx) maxx = px;
    if (py < miny) miny = py; if (py > maxy) maxy = py;
  }
  fs.writeFileSync(path.join(outDir, `port_${tag}dt${mult}.f32`), Buffer.from(r.out.buffer));
  report.dt_sweep.push({
    mult, dt, spf, steps: r.steps, stable: r.stable, wall_ms: r.ms,
    speedup_vs_canonical: base.ms / r.ms,
    final_width: maxx - minx, final_height: maxy - miny, n_nonfinite: bad
  });
}

// ---- 4. where does the time go, and how does it scale with particle count? ---------------
report.scaling = [];
for (const np of [500, 1000, 2000, 4000, 8000]) {
  const pts2 = MPM.seedDisk(0.5, 0.52, 0.11, np, 12345);
  const sim = MPM.createSim({ n: np, area: Math.PI * 0.11 * 0.11, dt: dt0 });
  sim.seed(pts2, 0, 0);
  sim.substeps(200);                      // warm up JIT + settle the active-cell set
  sim.setProfile(true); sim.resetTiming();
  const t0 = process.hrtime.bigint();
  const K = 1000;
  sim.substeps(K);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  report.scaling.push({
    n: np, us_per_step: (ms * 1000) / K, active_cells: sim.activeCells(),
    p2g_us: sim.timing.p2g * 1000 / K, grid_us: sim.timing.grid * 1000 / K,
    g2p_us: sim.timing.g2p * 1000 / K,
    realtime_substeps_per_sec: 1 / dt0,
    realtime_factor: 1 / ((ms / K) * 1e-3 * (1 / dt0))
  });
}

fs.writeFileSync(path.join(outDir, `port_${tag}report.json`), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
