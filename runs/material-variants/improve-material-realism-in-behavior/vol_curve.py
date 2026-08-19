"""Volume ratio of a material as a function of time, measured exactly rather than from a raster.

For a solid the model's own volume ratio is det(F) per particle, and the body's true area is
`initial_area * mean(det F)`. For the fluid the same role is played by J. Neither is exposed per frame
by `core.simulate`, which returns positions only, so the curve is built by rolling the SAME canonical
simulator to a sweep of end times and reading the field at each end point. Slower than instrumenting
the loop, but it uses the frozen public API unchanged, which is the point.

usage:  python vol_curve.py --tag before --material elastic --scene slam
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sim.physics import core                                    # noqa: E402
from sim.physics import VERSION                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 7000

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--jobs", default="elastic/slam,elastic/drop,fluid/drop,fluid/dam")
ap.add_argument("--nt", type=int, default=22)
args = ap.parse_args()


def scene(name):
    if name in ("drop", "column", "heap"):
        return core.scene(name, N)
    if name == "slam":
        return {"pts": core.seed_disk((0.5, 0.60), 0.11, N), "area": np.pi * 0.11 ** 2,
                "v0": (0.0, -6.0), "T": 1.0}
    if name == "dam":
        return {"pts": core.seed_box(core.floor_y, 0.22, core.floor_y, 0.42, N),
                "area": (0.22 - core.floor_y) * (0.42 - core.floor_y), "v0": (0.0, 0.0), "T": 1.4}
    raise KeyError(name)


out = {"physics_version": VERSION, "tag": args.tag, "curves": {}}
for job in args.jobs.split(","):
    mat, scn = job.split("/")
    sc = scene(scn)
    ts = np.linspace(sc["T"] / args.nt, sc["T"], args.nt)
    mean_v, min_v, p01_v = [], [], []
    for T in ts:
        core.simulate(mat, sc["pts"], sc["area"], float(T), 1, v0=sc["v0"])
        d = core.J.to_numpy()[:N] if mat == "fluid" else np.linalg.det(core.F.to_numpy()[:N])
        mean_v.append(float(d.mean())); min_v.append(float(d.min()))
        p01_v.append(float(np.percentile(d, 1)))
    out["curves"][job] = {"t": [float(z) for z in ts], "mean": mean_v, "min": min_v, "p01": p01_v}
    print(f"  {job:16s} mean {min(mean_v):.4f}..{max(mean_v):.4f}   worst particle {min(min_v):.3f}"
          f"   1st pct floor {min(p01_v):.3f}", flush=True)

json.dump(out, open(os.path.join(HERE, f"volcurve_{args.tag}.json"), "w"), indent=1)
print("wrote", f"volcurve_{args.tag}.json", VERSION)
