"""GATE 1: prove the reparameterised step is still canonical, before a single weight is trained.

The learned simulator does three things differently from `sim.physics.core` (see learned_sim's
docstring): it caches the stress one kernel earlier, it remounts F as R S' instead of U diag(s') V,
and it carries the fluid's volume ratio inside F. All three are argued to be exactly equivalent for
an isotropic constitutive model. This script checks that numerically instead of taking it on trust.
If it fails, nothing downstream means anything -- a net fitted to a teacher that is not canonical
would be measured against the wrong ground truth.

TWO CHECKS, BECAUSE ONE IS NOT ENOUGH
-------------------------------------
A. STEP-LEVEL. Roll canonical forward to a non-trivial state, then advance BOTH by a few substeps
   from that identical state and compare. Over a handful of substeps nothing has had time to
   amplify, so this reads the reparameterisation's own error directly. It should be f32 rounding.

B. TRAJECTORY-LEVEL, AGAINST THE RIGHT BAND. Over a full rollout the two orderings of the same
   arithmetic diverge chaotically, so the question is not "are they equal" but "do they differ by
   more than a physically meaningless perturbation does". The band is canonical run against
   canonical seeded with positions nudged by 1e-7 -- a ten-millionth of the domain, a thousandth of
   the f32 resolution of a coordinate near 1.0. Canonical's plain run-to-run noise (atomic-add
   ordering) is the WRONG band here: the elastic path is nearly bit-deterministic, so it is ~1e-7
   and three times nothing is still nothing.

    .venv/Scripts/python.exe runs/.../train/gate_oracle.py
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))

import learned_sim as LS            # noqa: E402
import sigproxy                     # noqa: E402
from sim.physics import core        # noqa: E402

N = 4000
NF = 8
NUDGE = 1e-7
SCENES = ["drop", "column", "heap", "dam", "slam"]
MATS = ["fluid", "elastic", "snow", "sand"]


def mean_dist(a, b):
    """The registry's `traj_rmse`: MEAN PER-PARTICLE EUCLIDEAN DISTANCE, averaged over frames.
    Despite the name it is not an RMS -- see spec/registry/metrics.json."""
    return float(np.linalg.norm(a - b, axis=-1).mean())


def noise_band(material, sc, nudge=NUDGE, seed=7):
    """How far canonical moves when its initial positions are nudged by `nudge`. This is the floor
    below which no difference between two implementations of the same physics is meaningful."""
    rng = np.random.default_rng(seed)
    p0 = np.asarray(sc["pts"], np.float64)
    p1 = p0 + rng.normal(0.0, nudge, p0.shape)
    a, _, _ = core.simulate(material, p0, sc["area"], sc["T"], NF, v0=sc["v0"])
    b, _, _ = core.simulate(material, p1, sc["area"], sc["T"], NF, v0=sc["v0"])
    return a, mean_dist(a, b)


def step_level():
    """Advance canonical and the oracle from an IDENTICAL non-trivial state for a few substeps."""
    rows = []
    sc = core.scene("column", N)
    for m in MATS:
        dt = core.MAT[m]["dt"]
        # warm to a state with real deformation, plastic history and contact
        core.simulate(m, sc["pts"], sc["area"], 0.35, 1, v0=sc["v0"])
        st = {k: getattr(core, k).to_numpy()[:N].copy()
              for k in ("x", "v", "C", "F", "Jp", "J")}
        for K in (1, 4, 16):
            # canonical, K substeps from st
            _restore(st, m, sc, N)
            for _ in range(K):
                core.clear_grid()
                core.p2g(core.MAT_ID[m], N, dt, core.MAT[m]["E"], core.MAT[m]["nu"],
                         core.MAT[m]["xi"], 0.0, sc["area"] / N,
                         (sc["area"] / N) * core.MAT[m]["rho"], core.MAT[m]["fric"])
                core.grid_op(dt, core.MAT[m]["fric"], core.gravity)
                core.g2p(core.MAT_ID[m], N, dt, core.MAT[m]["tc"], core.MAT[m]["ts"],
                         core.MAT[m]["E"], core.MAT[m]["nu"], core.dp_alpha(core.MAT[m]["phi"]))
            xa = core.x.to_numpy()[:N].copy()
            # oracle, K substeps from the SAME st -- with the fluid's scalar J rebuilt into F,
            # because canonical's fluid never writes F and the learned sim reads its volume ratio
            # from there
            _restore(st, m, sc, N)
            LS.seed_fluid_F_from_J(N)
            LS.upload_params()
            LS.prime_stress(N, False, 8, False)
            for _ in range(K):
                core.clear_grid()
                LS.p2g_learned(N, dt)
                core.grid_op(dt, core.FRICTION, core.gravity)
                LS.g2p_learned(N, dt, False, 8, False, 0, 1)
            xb = core.x.to_numpy()[:N].copy()
            d = mean_dist(xa, xb)
            rows.append({"material": m, "substeps": K, "mean_pos_diff": d,
                         "pass": bool(d < 3e-6)})
            print(f"  [{'PASS' if d < 3e-6 else 'FAIL'}] step-level {m:8s} {K:3d} substeps: "
                  f"mean |dx| = {d:.3e}")
    return rows


def _restore(st, m, sc, n):
    """Put an exact saved particle state back into core's fields (and the per-particle mass/volume
    the multi-material path reads)."""
    pad = lambda a, shp: np.concatenate([a, np.zeros((core.MAX_P - n,) + shp, a.dtype)], 0)
    core.x.from_numpy(pad(st["x"], (2,)))
    core.v.from_numpy(pad(st["v"], (2,)))
    core.C.from_numpy(pad(st["C"], (2, 2)))
    core.F.from_numpy(pad(st["F"], (2, 2)))
    core.Jp.from_numpy(pad(st["Jp"], ()))
    core.J.from_numpy(pad(st["J"], ()))
    mid = np.zeros(core.MAX_P, np.int32); mid[:n] = core.MAT_ID[m]
    core.mat_id.from_numpy(mid)
    pv = np.zeros(core.MAX_P, np.float32); pv[:n] = sc["area"] / n
    core.p_vol_f.from_numpy(pv)
    core.p_mass_f.from_numpy(pv * core.MAT[m]["rho"])


def main():
    print("--- A. step-level: identical state, a few substeps, no time to amplify ---")
    steps = step_level()

    print("--- B. the golden signatures, run against the ORACLE ---")
    sigrows, sigsum = sigproxy.run(
        lambda *a, **kw: LS.simulate(*a, mode="oracle", **kw),
        lambda *a, **kw: LS.simulate_multi(*a, mode="oracle", **kw), label="oracle")
    sigproxy.show(sigrows, "the ORACLE (analytic law, reparameterised step)")

    print("--- C. trajectory divergence: the FLOOR any learned net is measured against ---")
    rows = []
    for scn in SCENES:
        sc = core.scene(scn, N)
        for m in MATS:
            a, band = noise_band(m, sc)
            g = [{"material": m, "pts": sc["pts"], "area": sc["area"], "v0": sc["v0"]}]
            c, _, _, ok, _ = LS.rollout(g, sc["T"], NF, dt=core.MAT[m]["dt"], mode="oracle")
            cross = mean_dist(a, c)
            rows.append({"scene": scn, "material": m, "oracle_vs_canonical": cross,
                         "ic_nudge_band": band, "stable": bool(ok)})
            print(f"  {scn:7s} {m:8s} oracle vs canonical {cross:.3e}   "
                  f"1e-7 IC-nudge band {band:.3e}")

    quarters = []
    for i, m in enumerate(MATS):
        pts = core.seed_disk((0.22 + 0.19 * i, 0.62), 0.055, N // 4, seed=i)
        quarters.append({"material": m, "pts": pts, "area": np.pi * 0.055 ** 2, "v0": (0.0, 0.0)})
    a, _, _, _, dtm = core.simulate_multi(quarters, 1.0, NF)
    rng = np.random.default_rng(7)
    q2 = [dict(q, pts=np.asarray(q["pts"]) + rng.normal(0, NUDGE, np.asarray(q["pts"]).shape))
          for q in quarters]
    b, _, _, _, _ = core.simulate_multi(q2, 1.0, NF)
    c, _, _, ok, _ = LS.rollout(quarters, 1.0, NF, dt=dtm, mode="oracle")
    band, cross = mean_dist(a, b), mean_dist(a, c)
    rows.append({"scene": "mixed4", "material": "all", "oracle_vs_canonical": cross,
                 "ic_nudge_band": band, "stable": bool(ok)})
    print(f"  mixed4  all      oracle vs canonical {cross:.3e}   1e-7 IC-nudge band {band:.3e}")

    allok = all(r["pass"] for r in steps) and sigsum["fail"] == 0
    out = {"gate": "oracle == canonical", "physics_version": LS.VERSION, "n": N, "frames": NF,
           "ic_nudge": NUDGE, "step_level": steps, "signatures": sigrows,
           "signature_summary": sigsum, "trajectory_floor": rows, "all_pass": allok}
    (HERE / "gate_oracle.json").write_text(json.dumps(out, indent=2))
    print("ALL PASS" if allok else "GATE FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
