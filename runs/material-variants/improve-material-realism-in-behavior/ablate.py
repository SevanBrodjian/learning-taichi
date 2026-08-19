"""Attribution: which of the changed knobs is responsible for which part of the change.

Every row runs on the CURRENT (new) physics and reverts exactly one knob to its old value, using the
overrides `core.simulate` already exposes (E, dt, nu, rho, fric). The wall boundary condition is the one
change that has no override, so its contribution is read as the residual between the fully reverted row
and the true old measurement recorded in diag_before.json.

Reported per row: dam-break front speed (water), and volume retention through a hard slam (rubber).
"""
import json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

from sim.physics import core                                    # noqa: E402
from sim.physics import VERSION                                 # noqa: E402
import common                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N, NF = 7000, 60


def front_speed(snaps, T, lo=0.05, hi=0.20):
    """Speed of the leading front while it is still running free, before it reaches the far wall.
    Least-squares slope of the 99th percentile of x against time over [lo, hi] seconds."""
    t = np.arange(1, NF + 1) * T / NF
    f = np.array([common.front_position(s) for s in snaps])
    m = (t >= lo) & (t <= hi)
    return float(np.polyfit(t[m], f[m], 1)[0])


dam = core.scene("dam", N)
out = {"physics_version": VERSION, "water": {}, "rubber": {}}

# ---- water: dam-break front speed, reverting one knob at a time -----------------------------------
WATER_ROWS = [
    ("new (canonical)", {}),
    ("revert friction to 0.5", {"fric": 0.5}),
    ("revert stiffness to E/rho=180", {"E": 180.0, "dt": 1.2e-4}),
    ("revert both", {"fric": 0.5, "E": 180.0, "dt": 1.2e-4}),
]
for label, kw in WATER_ROWS:
    snaps, _, ok = core.simulate("fluid", dam["pts"], dam["area"], dam["T"], NF, v0=dam["v0"], **kw)
    Jv = core.J.to_numpy()[:N]
    out["water"][label] = {"front_speed": front_speed(snaps, dam["T"]), "stable": bool(ok),
                           "spread_width": core.spread_width(snaps[-1]),
                           "J_spread": float(np.percentile(Jv, 99) - np.percentile(Jv, 1))}
    print(f"  water  {label:32s} front {out['water'][label]['front_speed']:.3f} dom/s  "
          f"width {out['water'][label]['spread_width']:.3f}  Jspread "
          f"{out['water'][label]['J_spread']:.4f}", flush=True)

h0 = 0.42 - core.floor_y
out["ritter_front_speed"] = 2.0 * math.sqrt(9.8 * h0)
out["sqrt_g_h0"] = math.sqrt(9.8 * h0)
print(f"  (Ritter ideal-fluid front speed 2*sqrt(g h0) = {out['ritter_front_speed']:.3f} dom/s)")

# ---- rubber: volume through a hard slam, reverting the Poisson ratio ------------------------------
slam = core.scene("slam", N)
TIMES = (0.09, 0.12, 0.16, 0.22, 0.30, 0.45)
RUB_ROWS = [
    ("new (canonical, nu=0.45)", {}),
    ("revert nu to 0.20", {"nu": 0.20}),
    ("revert nu and stiffness", {"nu": 0.20, "E": 400.0 * core.MAT["elastic"]["rho"], "dt": 1.0e-4}),
]
for label, kw in RUB_ROWS:
    curve_mean, curve_p01 = [], []
    for T in TIMES:
        core.simulate("elastic", slam["pts"], slam["area"], float(T), 1, v0=slam["v0"], **kw)
        d = np.linalg.det(core.F.to_numpy()[:N])
        curve_mean.append(float(d.mean()))
        curve_p01.append(float(np.percentile(d, 1)))
    snaps, _, ok = core.simulate("elastic", slam["pts"], slam["area"], slam["T"], NF,
                                 v0=slam["v0"], **kw)
    ra = common.retained_area(snaps)
    out["rubber"][label] = {"t": list(TIMES), "detF_mean": curve_mean, "detF_p01": curve_p01,
                            "worst_mean": min(curve_mean), "worst_p01": min(curve_p01),
                            "retained_area_min": float(ra.min()), "stable": bool(ok)}
    print(f"  rubber {label:32s} worst mean det(F) {min(curve_mean):.3f}  "
          f"worst 1st-pct {min(curve_p01):.3f}  footprint min {ra.min():.3f}", flush=True)

json.dump(out, open(os.path.join(HERE, "ablation.json"), "w"), indent=1)
print("wrote ablation.json at", VERSION)
