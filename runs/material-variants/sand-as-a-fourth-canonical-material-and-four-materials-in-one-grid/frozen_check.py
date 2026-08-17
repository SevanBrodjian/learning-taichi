"""Did promoting sand move fluid, elastic or snow? A distributional test, not a single sample.

The reference simulator is nondeterministic (GPU atomic scatter order), and the collapse scenes are
chaotic, so two runs of the SAME code already disagree by up to about 1e-2 on the fluid column. A
single before-vs-after comparison against a single self-noise number therefore proves nothing either
way. The honest test is whether the ACROSS-CODE distances are drawn from the same distribution as the
WITHIN-CODE ones.

Run in two passes so the two versions of the physics never share a process:
    python frozen_check.py before   # imports the pre-promotion package copied out of git
    python frozen_check.py after    # imports the live sim.physics
    python frozen_check.py compare
"""
import itertools
import json
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(D, "..", "..", ".."))
SCRATCH = ("C:/Users/Owner/AppData/Local/Temp/claude/"
           "C--Users-Owner-Projects-learning-taichi/18115f90-d1bb-4c00-8da9-30c4048bece9/scratchpad")
REPS = 4
N = 5000
NF = 24
MATS = ("fluid", "elastic", "snow")
SCENES = ("drop", "column")


def capture(tag):
    if tag == "before":
        sys.path.insert(0, SCRATCH)
        import physics_orig as P
    else:
        sys.path.insert(0, ROOT)
        import sim.physics as P
    out = {}
    for s in SCENES:
        sc = P.scene(s, N)
        for m in MATS:
            for r in range(REPS):
                snaps, _, ok = P.simulate(m, sc["pts"], sc["area"], sc["T"], NF, v0=sc["v0"])
                out[f"{s}.{m}.{r}"] = snaps
                print(f"  {tag} {s}/{m} rep{r} stable={ok}")
    out["_version"] = np.array([P.VERSION])
    np.savez_compressed(os.path.join(D, f"frozen_{tag}.npz"), **out)
    print(f"{tag} physics_version: {P.VERSION}")


def rmse(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def compare():
    A = np.load(os.path.join(D, "frozen_before.npz"))
    B = np.load(os.path.join(D, "frozen_after.npz"))
    rows = []
    ok_all = True
    for s in SCENES:
        for m in MATS:
            a = [A[f"{s}.{m}.{r}"] for r in range(REPS)]
            b = [B[f"{s}.{m}.{r}"] for r in range(REPS)]
            within = ([rmse(a[i], a[j]) for i, j in itertools.combinations(range(REPS), 2)]
                      + [rmse(b[i], b[j]) for i, j in itertools.combinations(range(REPS), 2)])
            across = [rmse(a[i], b[j]) for i in range(REPS) for j in range(REPS)]
            w, x = np.array(within), np.array(across)
            # the claim "unchanged" == the across-code distances sit inside the within-code spread
            passed = bool(x.mean() <= w.max() and x.max() <= w.max() * 1.6)
            ok_all = ok_all and passed
            rows.append({"scene": s, "material": m, "reps": REPS,
                         "within_code_mean": float(w.mean()), "within_code_min": float(w.min()),
                         "within_code_max": float(w.max()),
                         "across_code_mean": float(x.mean()), "across_code_min": float(x.min()),
                         "across_code_max": float(x.max()),
                         "ratio_of_means": float(x.mean() / max(w.mean(), 1e-30)),
                         "passed": passed})
            print(f"  {s:7s} {m:8s} within {w.min():.2e}-{w.max():.2e} (mean {w.mean():.2e}) | "
                  f"across {x.min():.2e}-{x.max():.2e} (mean {x.mean():.2e}) | "
                  f"ratio {x.mean()/w.mean():.2f}  {'PASS' if passed else 'FAIL'}")
    worst = max(r["ratio_of_means"] for r in rows)
    out = {"version_before": str(A["_version"][0]), "version_after": str(B["_version"][0]),
           "reps_per_side": REPS, "rows": rows, "worst_ratio_of_means": worst,
           "worst_across_code": max(r["across_code_max"] for r in rows),
           "worst_within_code": max(r["within_code_max"] for r in rows),
           "unchanged": bool(ok_all),
           "test": ("For each scene and material, all pairwise traj_rmse distances WITHIN each code "
                    "version are compared against all distances ACROSS versions. 'Unchanged' means the "
                    "across-version distances are drawn from the same spread as the within-version "
                    "ones, which is the only meaningful statement when the reference simulator is "
                    "nondeterministic and the scenes are chaotic.")}
    json.dump(out, open(os.path.join(D, "frozen_materials_check.json"), "w"), indent=1)
    print(f"\nworst ratio of means {worst:.2f}   VERDICT:",
          "UNCHANGED" if ok_all else "MOVED -- investigate")
    return ok_all


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compare":
        sys.exit(0 if compare() else 1)
    capture(cmd)
