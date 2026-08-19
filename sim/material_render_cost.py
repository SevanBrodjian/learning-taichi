"""Honest per-frame cost measurement for the four rendering treatments.

Why this file exists instead of a stopwatch. A first pass timed each treatment with
``time.perf_counter()`` around ``ti.sync()`` and got ~3.3 ms for every screen-space treatment --
**and the same 3.3 ms at 360x360 and at 1080x1080**, a nine-fold change in pixels. A cost that does
not move with the thing it is supposed to be proportional to is not measuring that thing. What it was
measuring is the Python-side launch of ~25-30 Taichi kernels per frame, which is a property of this
harness and not of the algorithm; a WebGPU port records its passes into a command buffer once and
does not pay it.

So this measures both, separately:
  * ``gpu_ms``  -- summed device time of every kernel in the frame, from Taichi's kernel profiler
                   (CUDA events). This is the number that scales with pixels and the only one that
                   projects onto a browser.
  * ``wall_ms`` -- end-to-end wall clock including the Python launches, reported so the gap is
                   visible rather than hidden.

Both are RTX 4090 / CUDA numbers, not WebGPU: they are a floor for the browser cost of the same
passes, and every claim built on them says so.

The profiler is enabled by wrapping ``ti.init`` BEFORE importing the physics (which owns the real
init call). Nothing in sim/physics is modified.

Usage:  python sim/material_render_cost.py [--res 456,720,1080] [--out path.json]
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import taichi as ti  # noqa: E402

_orig_init = ti.init


def _init_with_profiler(*a, **k):
    k["kernel_profiler"] = True
    return _orig_init(*a, **k)


ti.init = _init_with_profiler


def measure(res, reps=60):
    os.environ["MR_RES"] = str(res)
    for mod in [m for m in list(sys.modules) if m.startswith("sim.material_render")]:
        del sys.modules[mod]
    import sim.material_render as mr
    mr.upload_static()
    kinds = ["current", "snow_powder", "water_film", "water_glass", "rubber_flat", "rubber_tex",
             "sand_bare", "sand_grain"]
    rng = np.random.default_rng(3)
    npart = 16384
    pos = rng.uniform(0.08, 0.92, (npart, 2)).astype(np.float32)
    pos[:, 1] *= 0.55
    vel = rng.normal(0, 1.2, (npart, 2)).astype(np.float32)
    out = {}
    for kind in kinds:
        sel = {"current": 0, "water_glass": 0, "water_film": 0, "rubber_tex": 1,
               "rubber_flat": 1, "snow_powder": 2, "sand_grain": 3, "sand_bare": 3}[kind]
        mats = np.full(npart, sel, np.int32)
        mr.set_palette(False)
        n = mr.upload_frame(pos, mats, vel, pos)
        ref = mr.DEMO_DENSITY / (res * res)

        def one():
            mr.copy_v3(mr.bg, mr.img)
            mr.copy_v3(mr.bg, mr.beneath)
            mr.render_layer(kind, n, sel, ref)
            mr.composite()
            mr.tonemap(0, 1.0)

        for _ in range(12):                    # JIT + cache warm-up, excluded from every number
            one()
        ti.sync()
        ti.profiler.clear_kernel_profiler_info()
        t0 = time.perf_counter()
        for _ in range(reps):
            one()
        ti.sync()
        wall = (time.perf_counter() - t0) / reps * 1e3
        gpu = ti.profiler.get_kernel_profiler_total_time() / reps * 1e3
        out[kind] = dict(gpu_ms=round(gpu, 4), wall_ms=round(wall, 3))
        print("  res=%-5d %-13s gpu %7.3f ms   wall %6.2f ms" % (res, kind, gpu, wall), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="456,720,1080")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res_list = [int(r) for r in a.res.split(",")]
    # one resolution per process: the field shapes are baked at import time
    if len(res_list) == 1:
        d = {str(res_list[0]): measure(res_list[0])}
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(d, fh, indent=2)
    else:
        import subprocess
        all_ = {}
        for r in res_list:
            p = os.path.join(os.path.dirname(a.out or "."), "_cost_%d.json" % r)
            subprocess.run([sys.executable, os.path.abspath(__file__), "--res", str(r),
                            "--out", p], check=True, cwd=_ROOT)
            with open(p) as fh:
                all_.update(json.load(fh))
            os.remove(p)
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(all_, fh, indent=2)
        print(json.dumps(all_, indent=2))
