"""The clean angle-of-repose measurement: relax an over-steep heap.

A collapsing column measures runout, and its deposit slope is systematically shallower than the true
angle of repose because the material arrives with kinetic energy. The textbook measurement instead
starts from a pile that is ALREADY steeper than the material can hold and lets it relax under gravity,
so the final flank angle is the angle the material genuinely supports.

Seeded as an isoceles triangle with 60-degree flanks, released from rest.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import sim.physics as P  # noqa: E402
from sim.physics import core as pc  # noqa: E402


def steep_triangle(n=6000, half_base=0.13, flank_deg=60.0, seed=0):
    """Rejection-sample a triangle of half-base `half_base` standing on the floor."""
    rng = np.random.default_rng(seed)
    H = half_base * np.tan(np.deg2rad(flank_deg))
    pts = []
    while len(pts) < n:
        xs = rng.uniform(0.5 - half_base, 0.5 + half_base, 4 * n)
        ys = rng.uniform(pc.floor_y, pc.floor_y + H, 4 * n)
        keep = (ys - pc.floor_y) <= H * (1.0 - np.abs(xs - 0.5) / half_base)
        pts.extend(np.stack([xs[keep], ys[keep]], 1).tolist())
    pts = np.array(pts[:n], dtype=np.float32)
    return pts, half_base * H  # area of the triangle = 0.5 * (2*hb) * H


PTS, AREA = steep_triangle()
T = 2.0
print(f"seed: 60-degree triangle, initial repose reading = {P.repose_angle(PTS):.1f} deg\n")

for phi in (35.0, 40.0, 45.0, 50.0, 55.0, 60.0):
    s, _, ok = P.simulate("sand", PTS, AREA, T, 4, phi=phi)
    print(f"  sand phi={phi:4.0f}  final repose={P.repose_angle(s[-1]):5.1f}deg  "
          f"width={P.spread_width(s[-1]):.3f} height={P.pile_height(s[-1]):.3f}"
          + ("" if ok else "  UNSTABLE"))
print()
for m in ("fluid", "snow", "elastic"):
    s, _, ok = P.simulate(m, PTS, AREA, T, 4)
    print(f"  {m:8s}      final repose={P.repose_angle(s[-1]):5.1f}deg  "
          f"width={P.spread_width(s[-1]):.3f} height={P.pile_height(s[-1]):.3f}")
