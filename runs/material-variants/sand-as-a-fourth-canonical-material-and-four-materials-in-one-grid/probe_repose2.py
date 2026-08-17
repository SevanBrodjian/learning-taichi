"""Fine calibration of the friction angle, plus the checks that decide whether the number is real.

  A. finer phi sweep on the steep-heap relaxation, with stability flags
  B. is the measured angle limited by grid resolution? (same test at three pile sizes)
  C. does the choice hold up across seeds and particle counts?
  D. does it stay monotone on the collapse scenes too, so it is not a one-scene artefact?
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import sim.physics as P  # noqa: E402
from sim.physics import core as pc  # noqa: E402


def steep_triangle(n=6000, half_base=0.13, flank_deg=60.0, seed=0):
    rng = np.random.default_rng(seed)
    H = half_base * np.tan(np.deg2rad(flank_deg))
    pts = []
    while len(pts) < n:
        xs = rng.uniform(0.5 - half_base, 0.5 + half_base, 4 * n)
        ys = rng.uniform(pc.floor_y, pc.floor_y + H, 4 * n)
        keep = (ys - pc.floor_y) <= H * (1.0 - np.abs(xs - 0.5) / half_base)
        pts.extend(np.stack([xs[keep], ys[keep]], 1).tolist())
    return np.array(pts[:n], dtype=np.float32), half_base * H


T = 2.0
print("=== A. finer phi sweep, steep-heap relaxation (60 deg triangle -> settled slope) ===")
PTS, AREA = steep_triangle()
for phi in (50.0, 55.0, 58.0, 60.0, 62.0, 65.0, 70.0, 75.0):
    s, _, ok = P.simulate("sand", PTS, AREA, T, 4, phi=phi)
    print(f"  phi={phi:4.0f} alpha={P.dp_alpha(phi):.3f}  repose={P.repose_angle(s[-1]):5.1f}deg  "
          f"width={P.spread_width(s[-1]):.3f} height={P.pile_height(s[-1]):.3f}"
          + ("" if ok else "  UNSTABLE"))

print("\n=== B. resolution: is the angle limited by how many cells the pile spans? (phi=60) ===")
for hb in (0.09, 0.13, 0.18, 0.24):
    pts2, area2 = steep_triangle(n=9000, half_base=hb)
    s, _, ok = P.simulate("sand", pts2, area2, T, 4, phi=60.0)
    print(f"  half_base={hb:.2f} ({hb / pc.dx:4.1f} cells)  repose={P.repose_angle(s[-1]):5.1f}deg")

print("\n=== C. phi=60 robustness across seed / particle count ===")
for n in (3000, 6000, 12000):
    for seed in (0, 1, 2):
        pts2, area2 = steep_triangle(n=n, seed=seed)
        s, _, ok = P.simulate("sand", pts2, area2, T, 4, phi=60.0)
        print(f"  n={n:6d} seed={seed}  repose={P.repose_angle(s[-1]):5.1f}deg "
              f"width={P.spread_width(s[-1]):.3f}" + ("" if ok else "  UNSTABLE"))

print("\n=== D. collapse scenes: is phi=60 still sensible where the material runs out? ===")


def block(x0, x1, y0, y1, n=6000, seed=0):
    return pc.seed_box(x0, x1, y0, y1, n, seed), (x1 - x0) * (y1 - y0)


for label, (bx) in [("heap  a=0.7", (0.35, 0.65, pc.floor_y, 0.24)),
                    ("tall  a=3.4", (0.42, 0.58, pc.floor_y, 0.56))]:
    pts2, area2 = block(*bx)
    for m in ("fluid", "snow", "elastic"):
        s, _, _ = P.simulate(m, pts2, area2, 1.7, 4)
        print(f"  {label} {m:8s} width={P.spread_width(s[-1]):.3f} "
              f"height={P.pile_height(s[-1]):.3f} repose={P.repose_angle(s[-1]):5.1f}")
    for phi in (45.0, 55.0, 60.0, 65.0):
        s, _, ok = P.simulate("sand", pts2, area2, 1.7, 4, phi=phi)
        print(f"  {label} sand{phi:4.0f} width={P.spread_width(s[-1]):.3f} "
              f"height={P.pile_height(s[-1]):.3f} repose={P.repose_angle(s[-1]):5.1f}"
              + ("" if ok else "  UNSTABLE"))
