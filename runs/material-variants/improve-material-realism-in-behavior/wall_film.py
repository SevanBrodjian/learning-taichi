"""How much water is clinging to a side wall, old physics against new.

The 'sticky' complaint has two halves and they move in opposite directions, so both are measured:
  peak     -- the largest fraction of the water that is within five cells of a side wall AND well
              above the bulk surface at any sampled time. This is the splash.
  residual -- the same fraction at the END of the roll. This is what is still hanging there.

Reads the snapshot archives written by diagnose.py for both tags.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("TASK_SCRATCH", os.path.join(HERE, "_scratch"))

DX = 1.0 / 128.0
BND = 3 * DX
NEAR = BND + 5 * DX          # "within five cells of the wall"
ABOVE = 4 * DX               # "well above the local bulk surface"

out = {}
for scene in ("drop", "dam", "slam"):
    row = {}
    for tag in ("before", "after"):
        s = np.load(os.path.join(SCRATCH, f"snaps_{tag}.npz"))[f"{scene}/fluid"].astype(np.float32)
        frac = []
        for f in s:
            near = (f[:, 0] < NEAR) | (f[:, 0] > 1.0 - NEAR)
            bulk = np.percentile(f[:, 1], 60)
            frac.append(float((near & (f[:, 1] > max(bulk, 0.05) + ABOVE)).mean()))
        row[tag] = {"peak": max(frac), "residual": frac[-1],
                    "curve": [round(z, 4) for z in frac]}
    out[scene] = row
    print(f"  {scene:5s}  peak {row['before']['peak']:.3f} -> {row['after']['peak']:.3f}   "
          f"residual {row['before']['residual']:.4f} -> {row['after']['residual']:.4f}")


# --- attribution: is the bigger wall jet the STIFFNESS or the new wall boundary condition? ---------
# Only the stiffness and the friction have overrides, so the wall treatment is read as the residual.
from sim.physics import core                                    # noqa: E402

N, NF = 7000, 60


def film_curve(snaps):
    fr = []
    for f in snaps:
        near = (f[:, 0] < NEAR) | (f[:, 0] > 1.0 - NEAR)
        bulk = np.percentile(f[:, 1], 60)
        fr.append(float((near & (f[:, 1] > max(bulk, 0.05) + ABOVE)).mean()))
    return fr


ABL = [("new (canonical)", {}),
       ("revert stiffness to E/rho=180", {"E": 180.0, "dt": 1.2e-4}),
       ("revert friction to 0.5", {"fric": 0.5}),
       ("revert both", {"fric": 0.5, "E": 180.0, "dt": 1.2e-4})]
out["_attribution"] = {}
for scname in ("drop", "dam"):
    sc = core.scene(scname, N)
    out["_attribution"][scname] = {}
    for label, kw in ABL:
        snaps, _, _ = core.simulate("fluid", sc["pts"], sc["area"], sc["T"], NF, v0=sc["v0"], **kw)
        fr = film_curve(snaps)
        out["_attribution"][scname][label] = {"peak": max(fr), "residual": fr[-1]}
        print(f"  {scname:5s} {label:32s} peak {max(fr):.3f}  residual {fr[-1]:.4f}", flush=True)
    print(f"  {scname:5s} {'TRUE OLD PHYSICS':32s} peak {out[scname]['before']['peak']:.3f}  "
          f"residual {out[scname]['before']['residual']:.4f}")

json.dump(out, open(os.path.join(HERE, "wall_film.json"), "w"), indent=1)
print("wrote wall_film.json")
