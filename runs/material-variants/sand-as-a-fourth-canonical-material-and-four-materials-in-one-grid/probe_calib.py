"""Calibrate the canonical sand: which friction angle, and what caps the realized repose angle.

Two questions:
  1. The realized repose angle saturates around 25 degrees no matter how large phi gets. Is that the
     FLOOR's Coulomb friction (atan(0.5) = 26.6 deg) rather than the sand's own strength? Test by
     temporarily raising the floor friction -- a diagnostic on a copy of the world constant, not a
     change to the canonical value.
  2. Does the heap scene (a squat block that collapses into a pile) separate sand from fluid robustly
     across particle counts and seeds?
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import sim.physics as P  # noqa: E402
from sim.physics import core as pc  # noqa: E402

HEAP = dict(x0=0.35, x1=0.65, y0=pc.floor_y, y1=0.24)
T = 1.7


def heap(n=5000, seed=0):
    pts = pc.seed_box(HEAP["x0"], HEAP["x1"], HEAP["y0"], HEAP["y1"], n, seed)
    return pts, (HEAP["x1"] - HEAP["x0"]) * (HEAP["y1"] - HEAP["y0"])


def show(tag, snap):
    print(f"  {tag:26s} width={P.spread_width(snap):.3f} height={P.pile_height(snap):.3f} "
          f"repose={P.repose_angle(snap):5.1f}deg")


print("=== 1. is the repose cap the FLOOR's friction? (heap scene, phi=50) ===")
pts, area = heap()
saved = pc.FRICTION
for fric in (0.2, 0.5, 1.0, 2.0, 10.0):
    pc.FRICTION = fric
    s, _, ok = P.simulate("sand", pts, area, T, 4, phi=50.0)
    show(f"floor fric={fric:<5g} (atan={np.degrees(np.arctan(fric)):.0f}deg)", s[-1])
pc.FRICTION = saved
print(f"  restored FRICTION={pc.FRICTION}")

print("\n=== 2. phi choice on the heap scene, against the other three materials ===")
for m in ("fluid", "elastic", "snow"):
    s, _, ok = P.simulate(m, pts, area, T, 4)
    show(m, s[-1])
for phi in (35.0, 40.0, 45.0, 50.0, 55.0):
    s, _, ok = P.simulate("sand", pts, area, T, 4, phi=phi)
    show(f"sand phi={phi:.0f}" + ("" if ok else "  UNSTABLE"), s[-1])

print("\n=== 3. robustness of phi=45 across seeds and particle counts (heap) ===")
for n in (3000, 5000, 9000):
    for seed in (0, 1, 2):
        pts2, area2 = heap(n, seed)
        s, _, ok = P.simulate("sand", pts2, area2, T, 4, phi=45.0)
        f, _, _ = P.simulate("fluid", pts2, area2, T, 4)
        print(f"  n={n:5d} seed={seed}  sand repose={P.repose_angle(s[-1]):5.1f} "
              f"width={P.spread_width(s[-1]):.3f} height={P.pile_height(s[-1]):.3f} | "
              f"fluid repose={P.repose_angle(f[-1]):5.1f} width={P.spread_width(f[-1]):.3f} "
              f"height={P.pile_height(f[-1]):.3f}")

print("\n=== 4. drop scene: sand must not splat like water ===")
sc = P.scene("drop", 5000)
for m in ("fluid", "elastic", "snow"):
    s, _, ok = P.simulate(m, sc["pts"], sc["area"], sc["T"], 4)
    show(m, s[-1])
for phi in (40.0, 45.0, 50.0):
    s, _, ok = P.simulate("sand", sc["pts"], sc["area"], sc["T"], 4, phi=phi)
    show(f"sand phi={phi:.0f}", s[-1])
