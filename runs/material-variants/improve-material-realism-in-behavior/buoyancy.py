"""The buoyancy experiment: each solid released at rest, fully submerged, in a pool of water.

Two batteries.
  materials  -- the three canonical solids at their canonical densities. Snow (0.3) should rise, rubber
                (1.2) and sand (1.6) should fall.
  control    -- ONE material (elastic) at several densities, everything else identical. If the result
                tracks density and nothing else, the mechanism is buoyancy rather than some property of
                the constitutive model.

Snapshots go to the scratchpad; the JSON summary lands in the run directory.
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sim.physics import core                                    # noqa: E402
from sim.physics import VERSION                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("TASK_SCRATCH", os.path.join(HERE, "_scratch"))
os.makedirs(SCRATCH, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="after")
ap.add_argument("--n", type=int, default=9000)
ap.add_argument("--nf", type=int, default=70)
ap.add_argument("--T", type=float, default=2.2)
args = ap.parse_args()

JOBS = ([("mat_" + m, m, None) for m in ("snow", "elastic", "sand")]
        + [("rho_%.1f" % r, "elastic", r) for r in (0.3, 0.6, 1.0, 1.6)])

out = {"physics_version": VERSION, "T": args.T, "n": args.n,
       "rho": {m: core.MAT[m]["rho"] for m in core.MAT}, "runs": {}}
store = {}
for key, solid, rho in JOBS:
    p = core.scene_pool(solid, args.n, T=args.T, rho=rho)
    t0 = time.time()
    snaps, times, mid, stable, dt = core.simulate_multi(p["groups"], p["T"], args.nf)
    sel = mid == core.MAT_ID[solid]
    if solid == "fluid":
        raise SystemExit("solid must not be the fluid")
    sol = snaps[:, sel]
    flu = snaps[:, ~sel]
    sub = [core.submerged_fraction(sol[i], flu[i]) for i in range(args.nf)]
    dep = [core.rest_depth(sol[i], flu[i]) for i in range(args.nf)]
    cy = [float(sol[i][:, 1].mean()) for i in range(args.nf)]
    rec = {"material": solid, "rho": rho if rho is not None else core.MAT[solid]["rho"],
           "stable": bool(stable), "dt": float(dt), "n_solid": int(sel.sum()),
           "submerged_curve": [round(z, 4) for z in sub],
           "rest_depth_curve": [round(z, 4) for z in dep],
           "centroid_y_curve": [round(z, 4) for z in cy],
           "submerged_final": float(np.mean(sub[-6:])),
           "rest_depth_final": float(np.mean(dep[-6:])),
           "centroid_y_start": cy[0], "centroid_y_final": float(np.mean(cy[-6:])),
           "waterline_final": float(np.mean([core.waterline(flu[i]) for i in range(args.nf - 6, args.nf)])),
           "wall_s": round(time.time() - t0, 1)}
    out["runs"][key] = rec
    store[key + "/solid"] = sol.astype(np.float16)
    store[key + "/fluid"] = flu.astype(np.float16)
    arrow = "RISES" if rec["centroid_y_final"] > rec["centroid_y_start"] + 0.01 else (
        "SINKS" if rec["centroid_y_final"] < rec["centroid_y_start"] - 0.01 else "HOVERS")
    print(f"  {key:10s} {solid:8s} rho={rec['rho']:<4} {arrow:6s} "
          f"y {rec['centroid_y_start']:.3f} -> {rec['centroid_y_final']:.3f}  "
          f"submerged {rec['submerged_final']:.2f}  waterline {rec['waterline_final']:.3f} "
          f"stable={stable} ({rec['wall_s']}s)", flush=True)

np.savez_compressed(os.path.join(SCRATCH, f"buoy_{args.tag}.npz"), **store)
json.dump(out, open(os.path.join(HERE, f"buoyancy_{args.tag}.json"), "w"), indent=1)
print("wrote buoyancy_%s.json at %s" % (args.tag, VERSION))
