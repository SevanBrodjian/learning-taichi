"""Where does each material's timestep actually break, and what breaks first?

The first sweep found the walls for fluid and elastic but never found one for snow or sand inside 4x
their canonical timestep, so this one pushes to 16x. It also separates two different failures, because
they do not happen at the same place and only one of them is visible in a stability flag:

  * dt_stable_max   -- the largest dt that stays finite and bounded. The loud failure.
  * dt_faithful_max -- the largest dt at which the SETTLED SHAPE still matches the converged reference.
    For sand that means it still holds its angle of repose; for a fluid it means it still spreads the
    same distance. This is the quiet failure, it always arrives first, and a stability flag cannot see
    it.

Two scenes, because they stress different things: a dropped disk (impact, the loud failure) and an
over-steep heap relaxing to its natural slope (sustained self-weight, the quiet one).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import sim.physics as P  # noqa: E402
from sim.physics import core as pc  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
N = 4000
MATS = ("fluid", "elastic", "snow", "sand")
MULTS = [16.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.125]


def steep_triangle(n=N, half_base=0.13, flank_deg=60.0, seed=0):
    rng = np.random.default_rng(seed)
    H = half_base * np.tan(np.deg2rad(flank_deg))
    pts = []
    while len(pts) < n:
        xs = rng.uniform(0.5 - half_base, 0.5 + half_base, 4 * n)
        ys = rng.uniform(pc.floor_y, pc.floor_y + H, 4 * n)
        keep = (ys - pc.floor_y) <= H * (1.0 - np.abs(xs - 0.5) / half_base)
        pts.extend(np.stack([xs[keep], ys[keep]], 1).tolist())
    return np.array(pts[:n], dtype=np.float32), half_base * H


drop = P.scene("drop", N)
tri_pts, tri_area = steep_triangle()
SCENES = {
    "drop":  dict(pts=drop["pts"], area=drop["area"], T=0.9),
    "heap":  dict(pts=tri_pts,     area=tri_area,    T=1.4),
}


def observe(material, sc, dt):
    snaps, _, finite = P.simulate(material, sc["pts"], sc["area"], sc["T"], 4, dt=dt)
    n = sc["pts"].shape[0]
    vel = pc.v.to_numpy()[:n]
    vmax = float(np.abs(vel).max()) if np.isfinite(vel).all() else float("inf")
    f = snaps[-1]
    ok = bool(finite) and np.isfinite(vmax) and vmax < 1e3
    return dict(dt=dt, ok=ok, vmax=vmax, width=P.spread_width(f), height=P.pile_height(f),
                repose=P.repose_angle(f), spf60=int(round((1 / 60) / dt)))


TOL = 0.15    # a settled shape within 15% of the converged one still counts as the same material
out = {"physics_version": P.VERSION, "n_particles": N, "tol_shape": TOL, "scenes": {}}

for sname, sc in SCENES.items():
    out["scenes"][sname] = {}
    for m in MATS:
        dtc = P.MAT[m]["dt"]
        ref = observe(m, sc, dtc * 0.125)
        rows = []
        for mu in MULTS:
            r = observe(m, sc, dtc * mu)
            r["mult"] = mu
            # relative shape drift against the converged reference: the interpretable criterion
            dw = abs(r["width"] - ref["width"]) / max(ref["width"], 1e-6)
            dh = abs(r["height"] - ref["height"]) / max(ref["height"], 1e-6)
            dr = abs(r["repose"] - ref["repose"]) / max(ref["repose"], 1.0)
            r["shape_drift"] = float(max(dw, dh, dr))
            r["faithful"] = bool(r["ok"] and r["shape_drift"] <= TOL)
            rows.append(r)
        okr = [r for r in rows if r["ok"]]
        fr = [r for r in rows if r["faithful"]]
        rec = {"ref": ref, "sweep": rows, "dt_canonical": dtc,
               "dt_stable_max": max((r["dt"] for r in okr), default=float("nan")),
               "dt_faithful_max": max((r["dt"] for r in fr), default=float("nan"))}
        out["scenes"][sname][m] = rec
        print(f"[{sname}] {m:8s} canonical={dtc:.2e}  stable<= {rec['dt_stable_max']:.2e}  "
              f"faithful<= {rec['dt_faithful_max']:.2e}   "
              f"(ref width={ref['width']:.3f} height={ref['height']:.3f} repose={ref['repose']:.1f})")
        for r in rows:
            print(f"      x{r['mult']:<6g} dt={r['dt']:.2e} ok={str(r['ok']):5s} "
                  f"vmax={r['vmax']:8.2f} w={r['width']:.3f} h={r['height']:.3f} "
                  f"rep={r['repose']:5.1f} drift={r['shape_drift']:.3f} "
                  f"{'FAITHFUL' if r['faithful'] else ''}")

# combine across scenes: the binding limit is the worse of the two
summary = {}
for m in MATS:
    st = min(out["scenes"][s][m]["dt_stable_max"] for s in SCENES)
    fa = min(out["scenes"][s][m]["dt_faithful_max"] for s in SCENES)
    summary[m] = {"E": P.MAT[m]["E"], "dt_canonical": P.MAT[m]["dt"],
                  "dt_stable_max": st, "dt_faithful_max": fa,
                  "cfl_linear": pc.dx / np.sqrt(P.MAT[m]["E"] / pc.p_rho),
                  "spf60_at_faithful": int(round((1 / 60) / fa)) if np.isfinite(fa) else None,
                  "spf60_canonical": int(round((1 / 60) / P.MAT[m]["dt"]))}
    summary[m]["frac_of_cfl_linear"] = P.MAT[m]["dt"] / summary[m]["cfl_linear"]
out["summary"] = summary
print("\n=== binding limit across both scenes ===")
for m, s in summary.items():
    print(f"  {m:8s} stable<= {s['dt_stable_max']:.2e}  faithful<= {s['dt_faithful_max']:.2e}  "
          f"canonical={s['dt_canonical']:.2e} ({s['spf60_canonical']} substeps/frame)")
json.dump(out, open(os.path.join(D, "dt_sweep2.json"), "w"), indent=1,
          default=lambda o: bool(o) if isinstance(o, (bool, np.bool_)) else float(o))
print("wrote dt_sweep2.json")
