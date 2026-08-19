"""Diagnosis probe for the water complaints ("too mushy and too sticky").

Three suspects are separated here:
  * floor friction  -- `core.FRICTION` is applied to EVERY material at the floor, water included.
  * stiffness E     -- the weak-compressibility bulk modulus. c = sqrt(E/rho) sets how much the
                       fluid squashes under an impact, i.e. how "mushy" it looks.
  * viscosity       -- canonically 0 for the fluid, so it cannot be the cause; measured anyway.

Both `core.FRICTION` and `core.NU` are Python constants captured at kernel-COMPILE time, so
rebinding them right after import sweeps them without editing the frozen source. Nothing is written
back; this file only measures.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--fric", type=float, default=None)
ap.add_argument("--E", type=float, default=None)
ap.add_argument("--dt", type=float, default=None)
ap.add_argument("--mu", type=float, default=0.0)
ap.add_argument("--scenes", default="drop,column")
ap.add_argument("--out", default=None)
args = ap.parse_args()

from sim.physics import core                                    # noqa: E402
if args.fric is not None:
    core.FRICTION = args.fric                                   # BEFORE any kernel compiles
import common                                                   # noqa: E402

N = 7000
NF = 60
res = {"fric": args.fric if args.fric is not None else core.FRICTION,
       "E": args.E if args.E is not None else core.MAT["fluid"]["E"],
       "dt": args.dt if args.dt is not None else core.MAT["fluid"]["dt"], "mu": args.mu}

for name in args.scenes.split(","):
    sc = core.scene(name, N)
    snaps, times, stable = core.simulate("fluid", sc["pts"], sc["area"], sc["T"], NF, v0=sc["v0"],
                                         E=args.E, dt=args.dt, mu_visc=args.mu)
    Jv = core.J.to_numpy()[:N]
    fr = np.array([common.front_position(s) for s in snaps])
    res[name] = {
        "stable": bool(stable),
        "spread_width": core.spread_width(snaps[-1]),
        "pile_height": core.pile_height(snaps[-1]),
        "front_final": float(fr[-1]),
        "front_curve": [round(float(z), 4) for z in fr],
        "J_final_mean": float(Jv.mean()), "J_final_min": float(Jv.min()),
        "J_spread": float(np.percentile(Jv, 99) - np.percentile(Jv, 1)),
        # how much material is still piled up rather than run out flat
        "residual_mound": core.pile_height(snaps[-1]),
    }
    r = res[name]
    print(f"fric={res['fric']} E={res['E']} dt={res['dt']} mu={args.mu} {name:7s} stable={stable} "
          f"width={r['spread_width']:.3f} height={r['pile_height']:.3f} front={r['front_final']:.3f} "
          f"Jmin={r['J_final_min']:.3f} Jspread={r['J_spread']:.4f}", flush=True)

if args.out:
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
