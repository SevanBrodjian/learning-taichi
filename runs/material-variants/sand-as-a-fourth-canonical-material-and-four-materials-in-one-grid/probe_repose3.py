"""Pick the canonical friction angle: the largest phi that is still MONOTONE and LOW-VARIANCE.

phi = 60 produced the steepest heap but it sits on a knee -- the repose angle falls again past it and
scatters by +-4 degrees across seeds, which is not a place to freeze a canonical parameter or hang a
golden signature. This grid reports, for each candidate phi, the mean and spread of the settled angle
over seeds and over pile sizes, so the choice is made on stability rather than on the single largest
number.
"""
import json
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
SEEDS = (0, 1, 2, 3)
SIZES = (0.10, 0.13, 0.18)
out = {}
print(f"{'phi':>5} {'alpha':>6} | {'mean':>6} {'sd':>5} {'min':>6} {'max':>6} | per-size means")
for phi in (40.0, 45.0, 50.0, 55.0, 58.0, 60.0):
    per_size = {}
    allv = []
    for hb in SIZES:
        vals = []
        for sd in SEEDS:
            pts, area = steep_triangle(half_base=hb, seed=sd)
            s, _, ok = P.simulate("sand", pts, area, T, 4, phi=phi)
            a = P.repose_angle(s[-1])
            vals.append(a)
            allv.append(a)
        per_size[hb] = float(np.mean(vals))
    a = np.array(allv)
    out[phi] = {"mean": float(a.mean()), "sd": float(a.std()), "min": float(a.min()),
                "max": float(a.max()), "per_size": per_size}
    ps = "  ".join(f"{hb:.2f}:{m:.1f}" for hb, m in per_size.items())
    print(f"{phi:5.0f} {P.dp_alpha(phi):6.3f} | {a.mean():6.1f} {a.std():5.1f} {a.min():6.1f} "
          f"{a.max():6.1f} | {ps}")

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "phi_calibration.json"), "w"), indent=1)
