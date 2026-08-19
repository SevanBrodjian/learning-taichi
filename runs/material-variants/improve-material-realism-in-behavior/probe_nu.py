"""Diagnosis probe: what does the Poisson ratio actually control for the elastic blob?

`core.NU` is a module-level Python constant that Taichi captures when a kernel is COMPILED, so
rebinding it immediately after import -- before any kernel has run -- sweeps it without editing the
frozen source. This is a measurement, not a change: nothing is written back.

Reports, per nu: the volume ratio det(F) at the end of the roll, the minimum footprint the blob
occupies during the impact transient, whether the body stayed in one piece, and stability.

Scenes (all task-local except `drop`, which is canonical):
  drop    canonical disk released at rest
  impact  same disk from higher up with a downward kick
  slam    a hard floor impact, the regime where a compressible solid visibly pancakes
  smash   two blobs driven head-on into each other -- the "does it break apart" probe
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--nu", type=float, default=None)
ap.add_argument("--E", type=float, default=None)
ap.add_argument("--dt", type=float, default=None)
ap.add_argument("--material", default="elastic")
ap.add_argument("--scenes", default="drop,impact")
ap.add_argument("--out", default=None)
args = ap.parse_args()

from sim.physics import core                                    # noqa: E402
if args.nu is not None:
    core.NU = args.nu                                           # BEFORE any kernel compiles
import common                                                   # noqa: E402

N = 7000
NF = 60


def scene(name):
    if name in ("drop", "column", "heap", "two_blobs"):
        return core.scene(name, N)
    if name == "impact":
        return {"pts": core.seed_disk((0.5, 0.78), 0.11, N), "area": np.pi * 0.11 ** 2,
                "v0": (0.0, -2.5), "T": 1.3}
    if name == "slam":
        return {"pts": core.seed_disk((0.5, 0.60), 0.11, N), "area": np.pi * 0.11 ** 2,
                "v0": (0.0, -6.0), "T": 1.0}
    if name == "smash":
        a = core.seed_disk((0.30, 0.30), 0.08, N // 2)
        b = core.seed_disk((0.70, 0.30), 0.08, N - N // 2)
        v = np.zeros((N, 2), np.float32)
        return {"pts": np.concatenate([a, b], 0), "area": 2 * np.pi * 0.08 ** 2,
                "v0": None, "T": 1.0, "split": (a.shape[0], 3.0)}
    raise KeyError(name)


res = {"nu": args.nu, "E": args.E, "dt": args.dt, "material": args.material}
for scene_name in args.scenes.split(","):
    sc = scene(scene_name)
    if scene_name == "smash":
        # per-particle initial velocity: the two blobs are driven at each other
        k, sp = sc["split"]
        pts = sc["pts"]
        v0 = np.zeros_like(pts)
        v0[:k, 0] = sp
        v0[k:, 0] = -sp
        snaps, times, stable = core.simulate(args.material, pts, sc["area"], sc["T"], NF,
                                             v0=(0.0, 0.0), E=args.E, dt=args.dt)
        # simulate() takes a single v0; re-run with the split by uploading through the multi path
        g = [{"material": args.material, "pts": pts[:k], "area": sc["area"] / 2, "v0": (sp, 0.0)},
             {"material": args.material, "pts": pts[k:], "area": sc["area"] / 2, "v0": (-sp, 0.0)}]
        snaps, times, _m, stable, _dt = core.simulate_multi(g, sc["T"], NF,
                                                           dt=args.dt or core.MAT[args.material]["dt"])
    else:
        snaps, times, stable = core.simulate(args.material, sc["pts"], sc["area"], sc["T"], NF,
                                             v0=sc["v0"], E=args.E, dt=args.dt)
    n = snaps.shape[1]
    if args.material == "fluid":
        d = core.J.to_numpy()[:n]
    else:
        d = np.linalg.det(core.F.to_numpy()[:n])
    ra = common.retained_area(snaps)
    res[scene_name] = {
        "stable": bool(stable),
        "detF_final_mean": float(d.mean()), "detF_final_min": float(d.min()),
        "retained_area_min": float(ra.min()), "retained_area_final": float(ra[-1]),
        "spread_width": core.spread_width(snaps[-1]), "pile_height": core.pile_height(snaps[-1]),
        "detached_fraction": common.detached_fraction(snaps[-1]),
        "fragments": common.fragment_count(snaps[-1]),
    }
    r = res[scene_name]
    print(f"{args.material} nu={args.nu} E={args.E} dt={args.dt} {scene_name:7s} stable={stable} "
          f"detF={d.mean():.4f}/{d.min():.3f} area_min={ra.min():.3f} area_end={ra[-1]:.3f} "
          f"frag={r['fragments']} detached={r['detached_fraction']:.3f}", flush=True)

if args.out:
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
