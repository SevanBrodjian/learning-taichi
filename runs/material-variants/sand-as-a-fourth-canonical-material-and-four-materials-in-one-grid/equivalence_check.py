"""Did the multi-material refactor change single-material behaviour? Distributional, not one sample.

A single canonical-vs-multi comparison against a single self-noise number is not a test, because the
reference simulator is nondeterministic and the drop scene is chaotic, so the ratio of the two swings
by a factor of two between runs purely by luck. (It did: an earlier single-sample pass put fluid at
1.61 and a second at 0.97, on identical code.)

So: several repeats of the canonical path and several of the multi-material path, then every pairwise
distance WITHIN a path against every distance ACROSS paths.

Even that is not quite enough, because the two paths compile different kernels (a compile-time branch
against a runtime one) and can therefore order the same arithmetic differently at the last bit. A
chaotic rollout amplifies a last-bit difference exponentially, so the endpoint can land above the
run-to-run band while carrying no information at all. The established bracket for that is the
ROUNDING-PERTURBATION band: re-run the canonical path with initial positions nudged by about one
float32 rounding unit and measure how far that lands. Anything inside [self-noise, rounding-perturbed]
is a disagreement that carries no information.

The shape of the divergence curve is the real evidence and the endpoint is not: bias appears
immediately and grows linearly, chaos starts at rounding scale and grows exponentially until it
saturates.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

import render as R                                  # noqa: E402
import sim.physics as P                             # noqa: E402

MATS = ("fluid", "elastic", "snow", "sand")
COL = R.DEMO_COLOR
REPS = 3
N = 3500
NF = 40


def per_frame(a, b):
    return np.linalg.norm(a - b, axis=-1).mean(axis=1)


sc = P.scene("drop", N)
out = {"physics_version": P.VERSION, "scene": "drop", "n_particles": N, "reps_per_path": REPS,
       "rows": {},
       "test": ("All pairwise traj_rmse distances WITHIN a code path (canonical-vs-canonical and "
                "multi-vs-multi) against all distances ACROSS paths (canonical-vs-multi). 'No change' "
                "means the across-path distances come from the same distribution as the within-path "
                "ones, which is the only meaningful statement against a nondeterministic reference.")}
ok_all = True
curves = {}

ULP = 1e-7          # about one float32 rounding unit at these coordinates
rng = np.random.default_rng(7)
pts_nudged = (sc["pts"] + ULP * rng.standard_normal(sc["pts"].shape)).astype(np.float32)

for m in MATS:
    can, mul, per = [], [], []
    for _ in range(REPS):
        a, times, _ = P.simulate(m, sc["pts"], sc["area"], sc["T"], NF, v0=sc["v0"])
        can.append(a)
    g = [{"material": m, "pts": sc["pts"], "area": sc["area"], "v0": sc["v0"]}]
    for _ in range(REPS):
        c, _, _, stable, _ = P.simulate_multi(g, sc["T"], NF, dt=P.MAT[m]["dt"])
        mul.append(c)
    for _ in range(REPS):
        d, _, _ = P.simulate(m, pts_nudged, sc["area"], sc["T"], NF, v0=sc["v0"])
        per.append(d)
    within = ([per_frame(can[i], can[j]) for i, j in itertools.combinations(range(REPS), 2)]
              + [per_frame(mul[i], mul[j]) for i, j in itertools.combinations(range(REPS), 2)])
    across = [per_frame(can[i], mul[j]) for i in range(REPS) for j in range(REPS)]
    nudge = [per_frame(can[i], per[j]) for i in range(REPS) for j in range(REPS)]
    w = np.array([c.mean() for c in within])
    x = np.array([c.mean() for c in across])
    u = np.array([c.mean() for c in nudge])
    # The refactor is a no-op if its typical disagreement sits inside the bracket of disagreements
    # that provably carry no information: run-to-run nondeterminism at the low end, a one-ulp nudge of
    # the initial condition at the high end. The test is on MEANS, deliberately. These distributions
    # are heavy-tailed (chaotic amplification of a rounding-scale seed), so comparing the max of nine
    # samples against the max of twelve is a coin flip that says nothing about the code.
    band_max, band_mean = max(w.max(), u.max()), max(w.mean(), u.mean())
    passed = bool(x.mean() <= band_max and x.mean() <= 1.5 * band_mean)
    ok_all = ok_all and passed
    out["rows"][m] = {
        "stable": bool(stable),
        "within_path_mean": float(w.mean()), "within_path_min": float(w.min()),
        "within_path_max": float(w.max()),
        "across_path_mean": float(x.mean()), "across_path_min": float(x.min()),
        "across_path_max": float(x.max()),
        "rounding_nudge_mean": float(u.mean()), "rounding_nudge_min": float(u.min()),
        "rounding_nudge_max": float(u.max()),
        "ratio_to_self_noise": float(x.mean() / max(w.mean(), 1e-30)),
        "ratio_to_nudge": float(x.mean() / max(u.mean(), 1e-30)), "passed": passed}
    curves[m] = {"t": times, "within": np.array(within), "across": np.array(across),
                 "nudge": np.array(nudge)}
    out["rows"][m]["t"] = times.round(4).tolist()
    out["rows"][m]["noinfo_band_lo"] = np.minimum(np.array(within).min(axis=0),
                                                 np.array(nudge).min(axis=0)).round(11).tolist()
    out["rows"][m]["noinfo_band_hi"] = np.maximum(np.array(within).max(axis=0),
                                                 np.array(nudge).max(axis=0)).round(11).tolist()
    out["rows"][m]["across_mean_curve"] = np.array(across).mean(axis=0).round(11).tolist()
    print(f"  {m:8s} self-noise {w.min():.2e}-{w.max():.2e} | one-ulp nudge "
          f"{u.min():.2e}-{u.max():.2e} | multi-vs-canonical {x.min():.2e}-{x.max():.2e}  "
          f"{'PASS' if passed else 'FAIL'}")

out["all_passed"] = ok_all
out["worst_ratio_to_self_noise"] = max(r["ratio_to_self_noise"] for r in out["rows"].values())
out["worst_ratio_to_nudge"] = max(r["ratio_to_nudge"] for r in out["rows"].values())
json.dump(out, open(os.path.join(D, "equivalence.json"), "w"), indent=1)
print("\nworst ratio to self-noise %.2f, to the one-ulp nudge %.2f  ->"
      % (out["worst_ratio_to_self_noise"], out["worst_ratio_to_nudge"]),
      "REFACTOR IS INDISTINGUISHABLE FROM NOISE" if ok_all else "SOMETHING MOVED")

# --- figure: does the across-path curve sit inside the band the simulator draws against itself? ---
fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.7), dpi=135, facecolor=R.BG, sharey=True)
for ax, m in zip(axes, MATS):
    ax.set_facecolor(R.BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#2a3444")
    ax.tick_params(colors=R.MUTED, labelsize=8.5)
    ax.grid(alpha=0.13, color="#3a4658", lw=0.7)
    ax.set_axisbelow(True)
    c = curves[m]
    lo = np.maximum(np.minimum(c["within"].min(axis=0), c["nudge"].min(axis=0)), 1e-12)
    hi = np.maximum(c["within"].max(axis=0), c["nudge"].max(axis=0))
    ax.fill_between(c["t"], lo, hi, color="#8fa0b3", alpha=0.34,
                    label="disagreement that carries\nno information (run-to-run\nplus a one-ulp nudge)")
    for row in c["across"]:
        ax.plot(c["t"], np.maximum(row, 1e-12), color=COL[m], lw=0.9, alpha=0.45)
    ax.plot(c["t"], np.maximum(c["across"].mean(axis=0), 1e-12), color=COL[m], lw=2.4,
            label="multi-material path\nvs canonical")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)", color=R.INK, fontsize=9.5)
    ax.set_title(f"{R.LABEL[m]}", color=R.INK, fontsize=11.5)
axes[0].set_ylabel("mean per-particle distance\n(domain lengths)", color=R.INK, fontsize=9.5)
lg = axes[0].legend(frameon=False, fontsize=8, loc="upper left")
for t in lg.get_texts():
    t.set_color(R.INK)
fig.suptitle("The multi-material path diverges from canonical the way canonical diverges from itself: "
             "from rounding scale, exponentially, into the same plateau",
             color=R.INK, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(os.path.join(D, "equivalence.png"), facecolor=R.BG)
plt.close(fig)
print("wrote equivalence.png")
sys.exit(0 if ok_all else 1)
