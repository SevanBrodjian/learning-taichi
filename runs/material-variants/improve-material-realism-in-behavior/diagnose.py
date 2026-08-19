"""Measure the four canonical materials on a fixed battery of scenes.

Run once BEFORE editing sim/physics (`--tag before`) and once after (`--tag after`). The physics is
imported unchanged from `sim.physics`; nothing here defines a constitutive model. Snapshots go to the
scratchpad (they are large and intermediate); only the JSON summary lands in the run directory.

usage:  python diagnose.py --tag before
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sim.physics import core                                    # noqa: E402
from sim.physics import VERSION                                 # noqa: E402
import common                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("TASK_SCRATCH", os.path.join(HERE, "_scratch"))
os.makedirs(SCRATCH, exist_ok=True)

N = 7000
NF = 60
MATERIALS = ("fluid", "elastic", "snow", "sand")


def scenes():
    """The battery. Three are canonical (`core.scene`); `slam` and `dam` are task-local and stated as
    such. `slam` is the same disk as `drop` but driven hard into the floor, which is the regime where a
    compressible solid visibly loses volume. `dam` is a ONE-SIDED dam break (a block of material against
    the left wall, released), which measures how far a front runs before it stops -- the canonical
    `column` is symmetric and reaches both walls too quickly to separate a slippery material from a
    draggy one."""
    s = {k: core.scene(k, N) for k in ("drop", "column", "heap")}
    s["slam"] = {"pts": core.seed_disk((0.5, 0.60), 0.11, N), "area": np.pi * 0.11 ** 2,
                 "v0": (0.0, -6.0), "T": 1.0}
    s["dam"] = {"pts": core.seed_box(core.floor_y, 0.22, core.floor_y, 0.42, N),
                "area": (0.22 - core.floor_y) * (0.42 - core.floor_y), "v0": (0.0, 0.0), "T": 1.4}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    sc = scenes()
    out = {"physics_version": VERSION, "tag": args.tag, "N": N, "NF": NF,
           "MAT": {m: dict(core.MAT[m]) for m in MATERIALS},
           "globals": {"NU": getattr(core, "NU", None), "FRICTION": core.FRICTION,
                       "p_rho": getattr(core, "p_rho", None)},
           "runs": {}}
    store = {}
    for name, s in sc.items():
        for m in MATERIALS:
            t0 = time.time()
            snaps, times, stable = core.simulate(m, s["pts"], s["area"], s["T"], NF, v0=s["v0"])
            key = f"{name}/{m}"
            rec = common.summarize(snaps, core)
            rec["stable"] = bool(stable)
            rec["T"] = float(s["T"])
            rec["wall_s"] = round(time.time() - t0, 2)
            rec["retained_area_curve"] = [round(float(v), 4) for v in common.retained_area(snaps)]
            rec["front_curve"] = [round(common.front_position(snaps[i]), 4) for i in range(len(snaps))]
            # volumetric state at the end of the roll, read straight out of the canonical fields
            if m == "fluid":
                Jv = core.J.to_numpy()[:N]
                rec["vol_final_mean"] = float(Jv.mean())
                rec["vol_final_min"] = float(Jv.min())
                rec["vol_final_p01"] = float(np.percentile(Jv, 1))
            else:
                Fv = core.F.to_numpy()[:N]
                d = np.linalg.det(Fv)
                rec["vol_final_mean"] = float(d.mean())
                rec["vol_final_min"] = float(d.min())
                rec["vol_final_p01"] = float(np.percentile(d, 1))
            out["runs"][key] = rec
            store[key] = snaps.astype(np.float16)
            print(f"  {key:22s} stable={stable} width={rec['spread_width']:.3f} "
                  f"area_end={rec['retained_area_final']:.3f} area_min={rec['retained_area_min']:.3f} "
                  f"frag={rec['fragments_final']} det={rec['vol_final_mean']:.3f} "
                  f"({rec['wall_s']}s)", flush=True)
    store["_times"] = times
    np.savez_compressed(os.path.join(SCRATCH, f"snaps_{args.tag}.npz"), **store)
    json.dump(out, open(os.path.join(HERE, f"diag_{args.tag}.json"), "w"), indent=1)
    print("physics", VERSION, "->", f"diag_{args.tag}.json")


if __name__ == "__main__":
    main()
