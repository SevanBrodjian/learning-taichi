"""Does the Drucker-Prager friction angle actually control the slope the material holds?

The limits are the test of the implementation, not the middle: phi -> 0 must give a material with no
shear strength (a fluid that also cannot take tension), and a large phi must give something that barely
yields. If the repose angle is monotone in phi between those, the return mapping is doing its job and
what is left is a calibration choice.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import sim.physics as P  # noqa: E402
from sim.physics import core as pc  # noqa: E402

N = 5000


def block(x0, x1, y0, y1, n=N, seed=0):
    pts = pc.seed_box(x0, x1, y0, y1, n, seed)
    return {"pts": pts, "area": (x1 - x0) * (y1 - y0), "v0": (0.0, 0.0)}


SCENES = {
    # (name, aspect ratio a = H/W) -- granular column collapse runout is a strong function of a
    "tall  a=3.4": block(0.42, 0.58, pc.floor_y, 0.56),
    "mid   a=1.4": block(0.40, 0.60, pc.floor_y, 0.30),
    "squat a=0.7": block(0.35, 0.65, pc.floor_y, 0.24),
}
T = 1.7


def row(label, snap):
    print(f"    {label:14s} width={P.spread_width(snap):.3f} height={P.pile_height(snap):.3f} "
          f"repose={P.repose_angle(snap):5.1f}deg")


for sname, sc in SCENES.items():
    print(f"=== {sname} ===")
    for m, kw in [("fluid", {}), ("snow", {})]:
        s, _, ok = P.simulate(m, sc["pts"], sc["area"], T, 4, **kw)
        row(f"{m}", s[-1])
    for phi in (0.0, 15.0, 30.0, 40.0, 50.0, 65.0, 80.0):
        s, _, ok = P.simulate("sand", sc["pts"], sc["area"], T, 4, phi=phi)
        row(f"sand phi={phi:.0f}" + ("" if ok else " UNSTABLE"), s[-1])
