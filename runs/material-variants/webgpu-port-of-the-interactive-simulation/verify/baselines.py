"""The other two legs of the three-way comparison, on the same machine, same session, same scenes.

  * Taichi / CUDA -- the canonical kernels from sim.physics, driven the way canonical `simulate`
    drives them: four kernel launches per substep, issued from Python. Also re-measures the empty
    launch, which is the number the whole task turns on.
  * JavaScript -- the single-threaded port from the previous task, re-timed here by baselines.js
    (imported unchanged; it asserts every elastic constant still matches).

Both use the SAME box-at-reference-density scene family the WebGPU harness used, so the three
curves are comparable rather than three separate papers.

    .venv/Scripts/python.exe runs/material-variants/webgpu-port-of-the-interactive-simulation/verify/baselines.py
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import taichi as ti                                # noqa: E402
import sim.physics as phys                         # noqa: E402
from sim.physics import core as C                  # noqa: E402

DENSITY0 = 2048 / (np.pi * 0.11 ** 2)
SIDE_MAX = 0.90
NS = [500, 1000, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 131072]


def box_scene(n, seed=12345):
    """The same constant-density box the WebGPU harness seeds, reproduced with the same xorshift32
    so the point sets are identical, not merely statistically similar."""
    side = min(SIDE_MAX, float(np.sqrt(n / DENSITY0)))
    x0, y0 = 0.5 - side / 2, 0.5 - side / 2
    s = np.uint32(seed)
    pts = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        for k in range(2):
            s ^= np.uint32(s << np.uint32(13))
            s ^= np.uint32(s >> np.uint32(17))
            s ^= np.uint32(s << np.uint32(5))
            pts[i, k] = (x0 if k == 0 else y0) + (float(s) / 4294967296.0) * side
    return pts, side * side, side, n / (side * side * C.n_grid * C.n_grid)


@ti.kernel
def noop():
    """A launch with almost no work: the floor imposed by launching a kernel at all from Python."""
    for i, j in ti.ndrange(C.n_grid, C.n_grid):
        C.grid_m[i, j] = C.grid_m[i, j]


def timeit(fn, reps, warm=20):
    for _ in range(warm):
        fn()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ti.sync()
    return (time.perf_counter() - t0) / reps * 1e6            # microseconds per call


def main():
    cfg = phys.MAT["elastic"]
    dt, E = cfg["dt"], cfg["E"]
    spf = round((1 / 60) / dt)

    res = {"impl": "taichi-cuda", "physics_version": phys.VERSION, "arch": str(ti.cfg.arch),
           "substeps_per_frame": spf, "n_grid": C.n_grid, "rows": []}

    res["empty_launch_us"] = timeit(noop, 800)
    res["grid_op_only_us"] = timeit(lambda: C.grid_op(dt, C.FRICTION, 9.8), 800)
    print("empty launch %.1f us | grid_op alone %.1f us" % (res["empty_launch_us"], res["grid_op_only_us"]))

    for n in NS:
        if n > C.MAX_P:
            print("skip n=%d (canonical MAX_P=%d -- the reference simulator cannot hold it)" % (n, C.MAX_P))
            res["rows"].append({"n": n, "us_per_substep": None,
                                "note": "exceeds canonical MAX_P=%d" % C.MAX_P})
            continue
        pts, area, side, ppc = box_scene(n)
        p_vol = area / n
        p_mass = p_vol * C.p_rho
        C._upload(pts, (0.0, 0.0), C.ELASTIC)
        C.init_state(n)

        def substep(n=n, p_vol=p_vol, p_mass=p_mass):
            C.clear_grid()
            C.p2g(C.ELASTIC, n, dt, E, 0.0, 0.0, p_vol, p_mass)
            C.grid_op(dt, C.FRICTION, 9.8)
            C.g2p(C.ELASTIC, n, dt, 0.0, 0.0, E, 0.0)      # tc, ts, E, alpha (unused by elastic)

        us = timeit(substep, 400)
        row = {"n": n, "particles_per_cell": ppc, "side": side, "us_per_substep": us,
               "frame_ms": us * spf / 1000, "fps": 1000 / (us * spf / 1000),
               "launches_per_frame": 4 * spf}
        res["rows"].append(row)
        print("n=%-7d %7.1f us/substep  %7.2f ms/frame" % (n, us, row["frame_ms"]))

    (HERE / "baseline_taichi.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n[node] re-timing the JS port")
    r = subprocess.run(["node", str(HERE / "baselines.js"), str(HERE / "baseline_js.json")],
                       capture_output=True, text=True, cwd=str(ROOT))
    print(r.stderr[-3000:])
    if r.returncode != 0:
        print(r.stdout[-2000:])
        raise SystemExit("node baselines failed")
    print("wrote baseline_taichi.json + baseline_js.json")


if __name__ == "__main__":
    main()
