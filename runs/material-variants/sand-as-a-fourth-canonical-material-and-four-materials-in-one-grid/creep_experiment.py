"""Is the slope a settled heap holds a MATERIAL property, or a per-substep artefact?

The dt sweep turned up something that has to be resolved before any angle-of-repose number can be
reported. On the heap scene, elastic gives exactly the same settled shape at every timestep from
1.25e-5 to 4e-4. Fluid, snow and sand all slump FURTHER as the timestep is made SMALLER. Snow is the
extreme case: at its canonical dt it keeps almost the whole seeded 60-degree slope, and at dt/8 it
relaxes to 17 degrees.

Two readings, with opposite consequences:
  (a) the small-dt run is the converged answer, and the canonical timesteps are simply too coarse to
      resolve the plastic flow, so the strength seen at canonical dt is partly discretisation; or
  (b) the small-dt run is the corrupted one, because each substep rectifies a little transfer noise
      into irreversible plastic strain, so eight times the substeps buys eight times the artificial
      creep and the heap sags for numerical reasons.

They are distinguishable. Under (a) the slump is a function of PHYSICAL TIME and the curves at
different dt collapse when plotted against t. Under (b) it is a function of SUBSTEP COUNT and the
curves collapse when plotted against the number of substeps taken. Run each material long past
settling at several timesteps and look at which axis collapses them.

Elastic is the control: it carries no plastic state and must be flat on both axes.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
D = os.path.dirname(os.path.abspath(__file__))
import sim.physics as P  # noqa: E402
from sim.physics import core as pc  # noqa: E402

N = 3000
T = 4.0            # long past settling: the heap scene is quiet by ~1 s
NF = 40
MULTS = (2.0, 1.0, 0.5, 0.25)
MATS = ("elastic", "snow", "sand", "fluid")

sc = P.scene("heap", N)
seeded = P.repose_angle(sc["pts"])
print(f"seeded slope {seeded:.1f} deg, T={T}s, {N} particles\n")

out = {"physics_version": P.VERSION, "seeded_slope_deg": seeded, "T": T, "n_particles": N,
       "series": {}}
for m in MATS:
    out["series"][m] = {}
    for mu in MULTS:
        dt = P.MAT[m]["dt"] * mu
        t0 = time.time()
        snaps, times, ok = P.simulate(m, sc["pts"], sc["area"], T, NF, v0=sc["v0"], dt=dt)
        ang = [P.repose_angle(snaps[i]) for i in range(NF)]
        hgt = [P.pile_height(snaps[i]) for i in range(NF)]
        steps = [(i + 1) * max(1, int(round((T / NF) / dt))) for i in range(NF)]
        out["series"][m][f"{mu:g}"] = {
            "dt": dt, "stable": bool(ok), "t": times.round(4).tolist(),
            "substeps_cumulative": steps,
            "repose_deg": [round(a, 3) for a in ang],
            "pile_height": [round(h, 5) for h in hgt],
            "wall_s": round(time.time() - t0, 1)}
        print(f"  {m:8s} dt={dt:.2e} (x{mu:g})  angle {ang[0]:5.1f} -> {ang[len(ang)//4]:5.1f} "
              f"-> {ang[-1]:5.1f} deg   height {hgt[-1]:.3f}   "
              f"{steps[-1]:7d} substeps  ({time.time() - t0:.0f}s)")

# quantify the collapse: for each material, how much does the final angle vary across dt, and does the
# angle at a FIXED substep count agree across dt?
diag = {}
for m in MATS:
    fin = {k: v["repose_deg"][-1] for k, v in out["series"][m].items()}
    spread_t = max(fin.values()) - min(fin.values())
    # sample every run at the same cumulative substep count (the largest count the coarsest run
    # reaches) and compare
    target = min(v["substeps_cumulative"][-1] for v in out["series"][m].values())
    at_steps = {}
    for k, v in out["series"][m].items():
        i = int(np.argmin(np.abs(np.array(v["substeps_cumulative"]) - target)))
        at_steps[k] = v["repose_deg"][i]
    spread_s = max(at_steps.values()) - min(at_steps.values())
    diag[m] = {"final_angle_by_dt": fin, "spread_at_equal_time": spread_t,
               "angle_at_equal_substeps": at_steps, "spread_at_equal_substeps": spread_s,
               "common_substeps": target}
    print(f"\n{m}: spread across dt at equal TIME = {spread_t:.1f} deg, "
          f"at equal SUBSTEP COUNT = {spread_s:.1f} deg")
out["collapse_diagnostic"] = diag
json.dump(out, open(os.path.join(D, "creep.json"), "w"), indent=1)
print("\nwrote creep.json")
