"""What does the reference GPU implementation actually cost per substep, and what would a learned
grid update cost on the same device?

At 128x128 = 16384 cells and a few thousand particles the problem is far too small to saturate a
modern GPU, so the wall time is dominated by kernel launch, not arithmetic. That is the number that
decides whether a learned grid update could ever be affordable inside a real-time loop, so it is
measured here rather than argued.

    .venv/Scripts/python.exe runs/material-variants/interactive-simulation-of-one-material/verify/gpu_bench.py
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import taichi as ti                              # noqa: E402
import sim.physics as phys                       # noqa: E402
from sim.physics import core as C                # noqa: E402

NG = C.n_grid
NCELL = NG * NG
H = 32
IN, OUT = 8, 2

w1 = ti.field(float, (IN, H)); b1 = ti.field(float, H)
w2 = ti.field(float, (H, H)); b2 = ti.field(float, H)
w3 = ti.field(float, (H, OUT)); b3 = ti.field(float, OUT)
feat = ti.field(float, (NCELL, IN))
outv = ti.field(float, (NCELL, OUT))


@ti.kernel
def noop():
    """A launch with almost no work: measures the floor imposed by launching a kernel at all."""
    for i, j in ti.ndrange(NG, NG):
        C.grid_m[i, j] = C.grid_m[i, j]


@ti.kernel
def mlp(n: ti.i32):
    """One 8-32-32-2 MLP per grid cell -- a learned grid update, priced on the GPU."""
    for c in range(n):
        h1 = ti.Vector.zero(float, H)
        for j in range(H):
            s = b1[j]
            for i in range(IN):
                s += feat[c, i] * w1[i, j]
            h1[j] = ti.max(s, 0.0)
        h2 = ti.Vector.zero(float, H)
        for j in range(H):
            s = b2[j]
            for i in range(H):
                s += h1[i] * w2[i, j]
            h2[j] = ti.max(s, 0.0)
        for j in range(OUT):
            s = b3[j]
            for i in range(H):
                s += h2[i] * w3[i, j]
            outv[c, j] = s


def timeit(fn, reps, warm=20):
    for _ in range(warm):
        fn()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ti.sync()
    return (time.perf_counter() - t0) / reps * 1e6          # microseconds per call


def main():
    rng = np.random.default_rng(0)
    for f, shape in ((w1, (IN, H)), (b1, (H,)), (w2, (H, H)), (b2, (H,)), (w3, (H, OUT)), (b3, (OUT,))):
        f.from_numpy(rng.normal(0, 0.2, shape).astype(np.float32))
    feat.from_numpy(rng.normal(0, 1, (NCELL, IN)).astype(np.float32))

    res = {"n_grid": NG, "n_cells": NCELL, "physics_version": phys.VERSION,
           "arch": str(ti.cfg.arch), "hidden": H, "mlp_shape": f"{IN}-{H}-{H}-{OUT}"}

    cfg = phys.MAT["elastic"]
    dt, E = cfg["dt"], cfg["E"]
    spf = round((1 / 60) / dt)
    res["substeps_per_frame_at_60fps"] = spf

    res["per_n"] = []
    for n in (500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 16384):
        sc = phys.scene("drop", n)
        pts = sc["pts"].astype(np.float32)
        p_vol = sc["area"] / n
        p_mass = p_vol * C.p_rho
        C._upload(pts)
        C.init_state(n)

        def substep(n=n, p_vol=p_vol, p_mass=p_mass):
            C.clear_grid()
            C.p2g(C.ELASTIC, n, dt, E, 0.0, 0.0, p_vol, p_mass)
            C.grid_op(dt, C.FRICTION, 9.8)
            C.g2p(C.ELASTIC, n, dt, 0.0, 0.0)

        us = timeit(substep, 400)
        res["per_n"].append({"n": n, "us_per_step": us,
                             "realtime_factor": 1 / (us * 1e-6 / dt),
                             "frame_ms_at_60fps_realtime": us * spf / 1000})

    res["noop_launch_us"] = timeit(noop, 800)
    res["grid_op_only_us"] = timeit(lambda: C.grid_op(dt, C.FRICTION, 9.8), 800)
    for cells in (760, NCELL):
        res[f"mlp_us_{cells}cells"] = timeit(lambda cells=cells: mlp(cells), 300)

    res["mlp_frame_ms_760cells"] = res["mlp_us_760cells"] * spf / 1000
    res["mlp_frame_ms_16384cells"] = res[f"mlp_us_{NCELL}cells"] * spf / 1000

    out = RUN / "verify" / "gpu_bench.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
