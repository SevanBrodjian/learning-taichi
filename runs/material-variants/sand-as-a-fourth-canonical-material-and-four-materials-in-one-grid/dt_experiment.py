"""What timestep does each material force, and why?

Three questions, in order:

  1. For each of the four materials, where does the explicit solver actually break? Sweep dt on the
     canonical drop scene (impact is the stress test) and record BOTH failure modes: divergence, and
     the quieter one where the run stays finite but has stopped tracking the converged trajectory.

  2. The canonical dt values for fluid / elastic / snow were chosen by hand. Expressing each as a
     fraction of its own measured blow-up point gives a consistent safety factor, and that same factor
     applied to sand's measured blow-up point is a principled way to set sand's canonical dt rather
     than eyeballing a CFL formula.

  3. The standing explanation for snow's very small dt is that hardening h = exp(xi (1 - Jp)) makes
     COMPACTED snow far stiffer than its nominal E = 150 suggests. That is checkable two ways: measure
     the hardening factor the particles actually reach, and re-run snow with xi = 0 to see whether its
     blow-up point moves to where its E alone would put it.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import sim.physics as P  # noqa: E402
from sim.physics import core as pc  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
N = 4000
T = 0.9          # long enough to include impact, short enough for a fine sweep
NF = 6
MATS = ("fluid", "elastic", "snow", "sand")


def traj_rmse(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def run(material, pts, area, dt, **kw):
    snaps, _, finite = P.simulate(material, pts, area, T, NF, dt=dt, **kw)
    n = pts.shape[0]
    vel = pc.v.to_numpy()[:n]
    vmax = float(np.abs(vel).max()) if np.isfinite(vel).all() else float("inf")
    # a run can stay "finite" purely because simulate() clamps x into the box; count how much of the
    # cloud is pinned against that clamp, which is what a blown-up run actually looks like.
    f = snaps[-1]
    lo, hi = pc.floor_y + 1e-6, 1.0 - pc.floor_y - 1e-6
    pinned = float(((f <= lo) | (f >= hi)).any(axis=1).mean())
    ok = finite and np.isfinite(vmax) and vmax < 1e3 and pinned < 0.5
    return snaps, dict(dt=dt, finite=bool(finite), vmax=vmax, pinned=pinned, ok=bool(ok),
                       width=P.spread_width(f), height=P.pile_height(f))


def cfl_linear(E):
    """The textbook linear estimate: dt <= dx / c with the bar wave speed c = sqrt(E/rho). It is the
    number the canonical timesteps have historically been quoted against."""
    return pc.dx / np.sqrt(E / pc.p_rho)


MULTS = [4.0, 3.0, 2.5, 2.0, 1.6, 1.3, 1.0, 0.75, 0.5, 0.25, 0.125]
sc = P.scene("drop", N)
res = {}

for m in MATS:
    cfg = P.MAT[m]
    print(f"=== {m} (E={cfg['E']}, canonical dt={cfg['dt']:.2e}) ===")
    ref, _ = run(m, sc["pts"], sc["area"], cfg["dt"] * 0.125)
    rows = []
    for mu in MULTS:
        t0 = time.time()
        snaps, r = run(m, sc["pts"], sc["area"], cfg["dt"] * mu)
        r["mult"] = mu
        r["traj_rmse_vs_converged"] = traj_rmse(snaps, ref) if r["ok"] else float("nan")
        r["spf60"] = int(round((1.0 / 60.0) / r["dt"]))
        r["wall_s"] = round(time.time() - t0, 2)
        rows.append(r)
        print(f"  x{mu:<6g} dt={r['dt']:.3e} ok={str(r['ok']):5s} vmax={r['vmax']:9.2f} "
              f"pinned={r['pinned']:.2f} rmse={r['traj_rmse_vs_converged']:.3e} "
              f"spf60={r['spf60']:5d} ({r['wall_s']}s)")
    okr = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    dt_max_ok = max(r["dt"] for r in okr) if okr else float("nan")
    dt_blowup = min(r["dt"] for r in bad) if bad else float("inf")
    res[m] = {"E": cfg["E"], "dt_canonical": cfg["dt"], "sweep": rows,
              "dt_stable_max": dt_max_ok, "dt_blowup": dt_blowup,
              "safety_factor": cfg["dt"] / dt_max_ok if okr else float("nan"),
              "cfl_linear": cfl_linear(cfg["E"]),
              "frac_of_cfl_linear": cfg["dt"] / cfl_linear(cfg["E"]),
              "spf60_canonical": int(round((1.0 / 60.0) / cfg["dt"]))}
    print(f"  -> stable up to {dt_max_ok:.3e}, first failure at {dt_blowup:.3e}; "
          f"canonical is {res[m]['safety_factor']:.2f}x the stable max, "
          f"{res[m]['frac_of_cfl_linear']:.3f} of the linear CFL limit\n")

# --- 3. snow's hardening: is it really what makes snow expensive? ---
print("=== snow hardening check ===")
snaps, _, _ = P.simulate("snow", sc["pts"], sc["area"], T, NF)
n = sc["pts"].shape[0]
jp = pc.Jp.to_numpy()[:n]
h = np.exp(P.MAT["snow"]["xi"] * (1.0 - jp))
E_eff = P.MAT["snow"]["E"] * h
hard = {"Jp_min": float(jp.min()), "Jp_median": float(np.median(jp)),
        "Jp_p05": float(np.percentile(jp, 5)),
        "h_median": float(np.median(h)), "h_p95": float(np.percentile(h, 95)),
        "h_max": float(h.max()),
        "E_eff_median": float(np.median(E_eff)), "E_eff_p95": float(np.percentile(E_eff, 95)),
        "E_elastic": P.MAT["elastic"]["E"]}
print(f"  after impact: Jp median={hard['Jp_median']:.4f} (5th pct {hard['Jp_p05']:.4f}); "
      f"hardening h median={hard['h_median']:.2f}, 95th pct={hard['h_p95']:.2f}")
print(f"  effective stiffness E*h: median={hard['E_eff_median']:.0f}, "
      f"95th pct={hard['E_eff_p95']:.0f}  (elastic is {hard['E_elastic']:.0f})")

# causal test: turn hardening off and re-find the blow-up point
rows0 = []
for mu in MULTS:
    _, r = run("snow", sc["pts"], sc["area"], P.MAT["snow"]["dt"] * mu, xi=0.0)
    r["mult"] = mu
    rows0.append(r)
    print(f"  xi=0  x{mu:<6g} dt={r['dt']:.3e} ok={r['ok']} vmax={r['vmax']:.2f}")
ok0 = [r for r in rows0 if r["ok"]]
hard["xi0_dt_stable_max"] = max(r["dt"] for r in ok0) if ok0 else float("nan")
hard["xi0_sweep"] = rows0
hard["snow_dt_stable_max"] = res["snow"]["dt_stable_max"]
hard["xi0_gain"] = hard["xi0_dt_stable_max"] / res["snow"]["dt_stable_max"]
hard["cfl_linear_at_E_eff_p95"] = cfl_linear(hard["E_eff_p95"])
hard["snow_frac_of_cfl_at_E_eff_p95"] = P.MAT["snow"]["dt"] / hard["cfl_linear_at_E_eff_p95"]
print(f"  snow with hardening: stable to {res['snow']['dt_stable_max']:.3e}; "
      f"without (xi=0): {hard['xi0_dt_stable_max']:.3e}  -> {hard['xi0_gain']:.2f}x")

out = {"physics_version": P.VERSION, "scene": "drop", "n_particles": N, "T": T,
       "materials": res, "snow_hardening": hard}
json.dump(out, open(os.path.join(D, "dt_sweep.json"), "w"), indent=1)
print("\nwrote dt_sweep.json")
