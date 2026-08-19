"""How much timestep margin the new parameters actually have.

Raising the fluid's stiffness and rubber's Poisson ratio both raise a wave speed, and an explicit scheme
pays for that in the largest stable timestep. This finds, per material, the largest dt at which a hard
scene still produces a sane rollout.

Detecting the blow-up needs care. `core.simulate` clamps particle positions into the domain, so a
diverging run still returns finite, in-range POSITIONS -- a check on x alone silently passes on a
material that has already exploded. The state that actually diverges is the velocity and the
deformation, so those are what is checked here, together with a behavioural check: the settled shape
must still match the shape the same scene reaches at the canonical timestep. A run can be "stable" long
after it has stopped being the same physics.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sim.physics import core                                    # noqa: E402
from sim.physics import VERSION                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 6000
FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
SHAPE_TOL = 0.15          # 15% drift in settled width counts as "no longer the same physics"


def probe(mat, sc, dt):
    snaps, _, ok = core.simulate(mat, sc["pts"], sc["area"], sc["T"], 4, v0=sc["v0"], dt=dt)
    n = snaps.shape[1]
    vel = core.v.to_numpy()[:n]
    state = core.J.to_numpy()[:n] if mat == "fluid" else np.linalg.det(core.F.to_numpy()[:n])
    finite = bool(np.isfinite(vel).all() and np.isfinite(state).all() and np.isfinite(snaps).all())
    vmax = float(np.abs(vel).max()) if finite else float("inf")
    return {"ok": bool(ok and finite and vmax < 60.0),
            "vmax": vmax if finite else None,
            "width": core.spread_width(snaps[-1]), "height": core.pile_height(snaps[-1])}


out = {"physics_version": VERSION, "shape_tol": SHAPE_TOL, "runs": {}}
for mat, scname in (("fluid", "slam"), ("elastic", "slam"), ("snow", "drop"), ("sand", "slam")):
    sc = core.scene(scname, N)
    dt0 = core.MAT[mat]["dt"]
    ref = probe(mat, sc, dt0)
    row = {"scene": scname, "dt_canonical": dt0, "reference_width": ref["width"], "factors": {}}
    stable_f, faithful_f = 1.0, 1.0
    for f in FACTORS[1:]:
        r = probe(mat, sc, dt0 * f)
        drift = (abs(r["width"] - ref["width"]) / max(ref["width"], 1e-6)) if r["ok"] else 9.9
        row["factors"][str(f)] = {"ok": r["ok"], "vmax": r["vmax"], "width": r["width"],
                                  "shape_drift": round(drift, 3)}
        if r["ok"]:
            stable_f = f
            if drift <= SHAPE_TOL:
                faithful_f = f
    row["dt_stable_max"] = dt0 * stable_f
    row["dt_faithful_max"] = dt0 * faithful_f
    row["stable_margin"] = stable_f
    row["faithful_margin"] = faithful_f
    out["runs"][mat] = row
    cap = " (capped by the sweep)" if stable_f == FACTORS[-1] else ""
    print(f"  {mat:8s} canonical dt {dt0:.1e}  bounded up to {stable_f:g}x{cap}  "
          f"shape still matches up to {faithful_f:g}x", flush=True)

json.dump(out, open(os.path.join(HERE, "stability.json"), "w"), indent=1)
print("wrote stability.json at", VERSION)
