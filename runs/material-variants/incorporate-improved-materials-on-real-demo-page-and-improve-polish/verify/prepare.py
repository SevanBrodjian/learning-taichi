"""Build the browser job + the canonical ground truth for the DEMO'S NEW PHYSICS (phys-c518316a4a05).

The one thing this task has to prove about the physics is that the **sink/float ordering emerges on
the demo's own WGSL solver**: snow (rho 0.3) floats, rubber (1.2) and sand (1.6) sink, with no
buoyancy force anywhere -- only the per-material particle mass p_vol*rho reaching the shared grid.

So the scenes here are BUOYANCY scenes, not the angle-of-repose scenes the MVP verified against.
(The repose scenes are still the right instrument for a single material in air; they say nothing
about density, because density is exactly unobservable for a lone material -- see the `E`/`rho`
invariance note in sim/physics/core.py.)

  pool_<solid>   canonical `scene_pool` geometry: a disk of <solid> released AT REST, fully
                 submerged at mid-depth in a pool of water. Released at rest and submerged, so the
                 only thing that can move it is its weight against the fluid pressure. One scene per
                 solid in {snow, elastic, sand}, plus a `fluid` control (a blob of water in water,
                 which must do nothing in particular).
  mixed4         the MVP's four-material regression scene, rerun on the NEW canonical physics, so a
                 trajectory-level agreement number survives the parameter change.

ONE PARTICLE DENSITY EVERYWHERE. The WebGPU engine carries a single global `p_vol`, so every group
in every scene is seeded at the same particles-per-unit-area and each group's `area` is set to
k*p_vol exactly rather than to its nominal geometric area. Rounding a particle count would otherwise
leave the two phases at slightly different p_vol, which the engine cannot represent.

    .venv/Scripts/python.exe runs/.../verify/prepare.py
"""
import json
import pathlib
import sys

import numpy as np

RUN = pathlib.Path(__file__).resolve().parents[1]
V = RUN / "verify"
sys.path.insert(0, str(RUN.parents[2]))

import sim.physics as phys                       # noqa: E402
from sim.physics import core as C                # noqa: E402

N_FRAMES = 60
T_POOL = 2.0
T_MIX = 1.6
NUDGE = 1e-7        # one float32 rounding unit at the domain scale
SOLIDS = ["snow", "elastic", "sand", "fluid"]

BLOB_R = 0.075
K_BLOB = 500                                     # -> the density every scene is seeded at
DENS = K_BLOB / (np.pi * BLOB_R ** 2)
P_VOL = 1.0 / DENS
DEPTH = 0.34


def spf_for(dt, T):
    return max(1, int(round((T / N_FRAMES) / dt)))


def pool_groups(solid, seed=0):
    """Canonical scene_pool's geometry at ONE particle density (see the module docstring)."""
    x0, x1 = C.floor_y, 1.0 - C.floor_y
    cx, cy = 0.5, 0.20
    lat = C.seed_lattice(x0, x1, C.floor_y, DEPTH,
                         int(round(DENS * (x1 - x0) * (DEPTH - C.floor_y))), seed=seed + 3)
    water = lat[np.hypot(lat[:, 0] - cx, lat[:, 1] - cy) > BLOB_R * 1.05]
    blob = C.seed_disk((cx, cy), BLOB_R, K_BLOB, seed=seed + 4)
    return [{"material": "fluid", "pts": water.astype(np.float32), "area": P_VOL * len(water)},
            {"material": solid, "pts": blob.astype(np.float32), "area": P_VOL * K_BLOB}]


def three_groups(seed=0):
    """THE HEADLINE SCENE: snow, rubber and sand released at rest, side by side, at the same depth in
    one pool. One picture, three densities, one grid -- the ordering is the whole result and it has
    to be visible without reading a number. Same geometry per blob, so the only thing that differs
    between the three is `rho`."""
    x0, x1 = C.floor_y, 1.0 - C.floor_y
    cxs = [0.25, 0.50, 0.75]
    solids = ["snow", "elastic", "sand"]
    cy = 0.20
    lat = C.seed_lattice(x0, x1, C.floor_y, DEPTH,
                         int(round(DENS * (x1 - x0) * (DEPTH - C.floor_y))), seed=seed + 3)
    keep = np.ones(len(lat), bool)
    for cx in cxs:
        keep &= np.hypot(lat[:, 0] - cx, lat[:, 1] - cy) > BLOB_R * 1.05
    water = lat[keep]
    groups = [{"material": "fluid", "pts": water.astype(np.float32), "area": P_VOL * len(water)}]
    for i, (cx, s) in enumerate(zip(cxs, solids)):
        groups.append({"material": s, "pts": C.seed_disk((cx, cy), BLOB_R, K_BLOB, seed=seed + 20 + i),
                       "area": P_VOL * K_BLOB})
    return groups


def mixed4_groups():
    R, k = BLOB_R, K_BLOB
    pool_x, pool_y = (0.10, 0.90), (C.floor_y, C.floor_y + 0.055)
    kpool = int(round(DENS * (pool_x[1] - pool_x[0]) * (pool_y[1] - pool_y[0])))
    return [
        {"material": "fluid", "pts": C.seed_box(pool_x[0], pool_x[1], pool_y[0], pool_y[1], kpool, seed=3),
         "area": P_VOL * kpool},
        {"material": "elastic", "pts": C.seed_disk((0.22, 0.62), R, k, seed=11), "area": P_VOL * k},
        {"material": "snow", "pts": C.seed_disk((0.42, 0.72), R, k, seed=12), "area": P_VOL * k},
        {"material": "sand", "pts": C.seed_disk((0.62, 0.62), R, k, seed=13), "area": P_VOL * k},
        {"material": "fluid", "pts": C.seed_disk((0.80, 0.72), R, k, seed=14), "area": P_VOL * k},
    ]


def emit(job, store, name, groups, T):
    """Run canonical three times (base / repeat / nudged) and write the shared IC for the browser.

    The repeat and the nudge are what define the SELF-NOISE BAND: GPU atomic ordering alone makes
    canonical non-deterministic, and a 1e-7 nudge is one float32 rounding unit. Any disagreement
    below that band is indistinguishable from re-running canonical itself, so it is the floor the
    browser is judged against -- not zero.
    """
    pts = np.concatenate([g["pts"] for g in groups], 0).astype(np.float32)
    mats = np.concatenate([[C.MAT_ID[g["material"]]] * len(g["pts"]) for g in groups]).astype(np.int32)
    pv = [g["area"] / len(g["pts"]) for g in groups]
    assert max(pv) - min(pv) < 1e-9 * max(pv), "groups must share one p_vol: %s" % pv
    dt = C.shared_dt([g["material"] for g in groups])
    spf = spf_for(dt, T)
    print("canonical %-14s n=%d dt=%.1e spf=%d ..." % (name, len(pts), dt, spf), flush=True)
    a, _, mid, ok, _ = C.simulate_multi(groups, T, N_FRAMES)
    b, _, _, _, _ = C.simulate_multi(groups, T, N_FRAMES)
    c, _, _, _, _ = C.simulate_multi([dict(g, pts=g["pts"] + NUDGE) for g in groups], T, N_FRAMES)
    assert (mid == mats).all()
    store[name + "_base"] = a
    store[name + "_rep"] = b
    store[name + "_nudge"] = c
    store[name + "_mat"] = mats
    (V / ("ic_%s.f32" % name)).write_bytes(pts.reshape(-1).tobytes())
    (V / ("ic_%s_mat.i32" % name)).write_bytes(mats.tobytes())
    job["ics"].append({"name": name, "kind": "multi", "pts_file": "ic_%s.f32" % name,
                       "mat_file": "ic_%s_mat.i32" % name, "n": int(len(pts)),
                       "p_vol": float(pv[0]), "dt": float(dt), "spf": spf, "n_frames": N_FRAMES,
                       "materials": [g["material"] for g in groups]})
    print("   stable:", ok, flush=True)
    return a, mats


def main():
    V.mkdir(parents=True, exist_ok=True)
    job = {"id": "mpm4-t027", "frames": N_FRAMES, "ics": [], "physics_version": phys.VERSION,
           "p_vol": float(P_VOL), "density": float(DENS)}
    store = {}
    canon = {}

    for s in SOLIDS:
        a, mats = emit(job, store, "pool_" + s, pool_groups(s), T_POOL)
        sel = mats == C.MAT_ID[s]
        fl = mats == C.MAT_ID["fluid"]
        # the solid group is the LAST run of its id; for the fluid control the blob is the tail
        idx = np.where(sel)[0]
        if s == "fluid":
            idx = idx[-K_BLOB:]
            sel = np.zeros(len(mats), bool); sel[idx] = True
            fl = np.ones(len(mats), bool); fl[idx] = False
        canon["pool_" + s] = {
            "rest_depth_0": C.rest_depth(a[0][sel], a[0][fl]),
            "rest_depth": C.rest_depth(a[-1][sel], a[-1][fl]),
            "submerged_fraction": C.submerged_fraction(a[-1][sel], a[-1][fl]),
            "rho": C.MAT[s].get("rho", C.p_rho),
        }
        canon["pool_" + s]["rest_depth_change"] = (canon["pool_" + s]["rest_depth"] -
                                                   canon["pool_" + s]["rest_depth_0"])
        print("   canonical %-8s rho=%.2f  d(rest_depth)=%+.4f  submerged=%.3f"
              % (s, canon["pool_" + s]["rho"], canon["pool_" + s]["rest_depth_change"],
                 canon["pool_" + s]["submerged_fraction"]), flush=True)

    a3, m3 = emit(job, store, "pool_three", three_groups(), T_POOL)
    fl3 = m3 == C.MAT_ID["fluid"]
    canon["pool_three"] = {}
    for s3 in ["snow", "elastic", "sand"]:
        sel = m3 == C.MAT_ID[s3]
        canon["pool_three"][s3] = {
            "rho": C.MAT[s3].get("rho", C.p_rho),
            "rest_depth_0": C.rest_depth(a3[0][sel], a3[0][fl3]),
            "rest_depth": C.rest_depth(a3[-1][sel], a3[-1][fl3]),
            "submerged_fraction": C.submerged_fraction(a3[-1][sel], a3[-1][fl3]),
            "mean_y_end": float(a3[-1][sel][:, 1].mean()),
        }
        canon["pool_three"][s3]["rest_depth_change"] = (
            canon["pool_three"][s3]["rest_depth"] - canon["pool_three"][s3]["rest_depth_0"])
        print("   three-up %-8s rho=%.2f  d(rest_depth)=%+.4f  submerged=%.3f  mean_y=%.4f"
              % (s3, canon["pool_three"][s3]["rho"], canon["pool_three"][s3]["rest_depth_change"],
                 canon["pool_three"][s3]["submerged_fraction"],
                 canon["pool_three"][s3]["mean_y_end"]), flush=True)

    emit(job, store, "mixed4", mixed4_groups(), T_MIX)

    job["bench"] = {"realtime_n": [4096, 8192, 12288, 16384], "realtime_frames": 24,
                    "render_frames": 60,
                    "render_res": [512, 1024]}
    job["canonical"] = canon
    (V / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    np.savez_compressed(V / "base.npz", **store)
    print("\nwrote job.json + base.npz")
    print(json.dumps(canon, indent=1))


if __name__ == "__main__":
    main()
