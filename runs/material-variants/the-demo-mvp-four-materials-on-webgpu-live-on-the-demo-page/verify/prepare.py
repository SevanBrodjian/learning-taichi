"""Build the browser job + the canonical ground truth it will be scored against.

Everything the browser is asked to reproduce is generated HERE, from `sim.physics`, so the two sides
start from bit-identical float32 initial conditions and run the identical number of substeps.

What it writes into verify/:
  ic_<name>.f32     the seed points, float32 (n,2), fetched by harness.html
  job.json          what the browser must run: n, area, dt, substeps/frame, frames
  base.npz          canonical trajectories + the SELF-NOISE band for each scene

THE SELF-NOISE BAND is the reference the browser is judged against, not zero. Canonical is re-run
twice: once identically (GPU atomic ordering alone makes it non-deterministic) and once with the
initial positions nudged by ~1e-7, one float32 rounding unit. Any disagreement below that band is
indistinguishable from re-running canonical itself.

Scenes:
  heap_<material>   the canonical 60-degree over-steep triangle, released from rest, at each
                    material's OWN canonical dt. This is the angle-of-repose instrument: whatever
                    slope survives is the slope the material genuinely holds.
  mixed4            all four materials on ONE grid at the shared dt = min(dt) = snow's 5e-5, run
                    through canonical `simulate_multi`. Equal particle density in every group, so a
                    single global p_vol reproduces it exactly.
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

N_HEAP = 2000
N_FRAMES = 60
T = 1.6
MATS = ["fluid", "elastic", "snow", "sand"]
NUDGE = 1e-7        # one float32 rounding unit at the domain scale


def spf_for(dt):
    return max(1, int(round((T / N_FRAMES) / dt)))


def main():
    V.mkdir(parents=True, exist_ok=True)
    job = {"id": "mpm4-verify", "frames": N_FRAMES, "T": T, "ics": [],
           "physics_version": phys.VERSION}
    store = {}

    # ---------------------------------------------------------------- per-material heap scenes
    sc = C.scene("heap", n=N_HEAP)
    pts = sc["pts"].astype(np.float32)
    area = float(sc["area"])
    (V / "ic_heap.f32").write_bytes(pts.reshape(-1).tobytes())
    store["ic_heap"] = pts

    for m in MATS:
        dt = C.MAT[m]["dt"]
        spf = spf_for(dt)
        name = "heap_" + m
        print("canonical %-14s dt=%.1e spf=%d ..." % (name, dt, spf), flush=True)
        a, _, ok_a = C.simulate(m, pts, area, T, N_FRAMES)
        b, _, ok_b = C.simulate(m, pts, area, T, N_FRAMES)
        c, _, ok_c = C.simulate(m, pts + NUDGE, area, T, N_FRAMES)
        store[name + "_base"] = a
        store[name + "_rep"] = b
        store[name + "_nudge"] = c
        print("   stable:", ok_a and ok_b and ok_c,
              " repose=%.1f deg  pile=%.3f  width=%.3f" %
              (C.repose_angle(a[-1]), C.pile_height(a[-1]), C.spread_width(a[-1])), flush=True)
        job["ics"].append({"name": name, "kind": "single", "material": m, "pts_file": "ic_heap.f32",
                           "n": N_HEAP, "area": area, "p_vol": area / N_HEAP, "dt": dt, "spf": spf,
                           "n_frames": N_FRAMES})

    # ---------------------------------------------------------------- the shared-dt consequence
    # A plastic material's settled slope decays with SUBSTEP COUNT, not physical time, so running
    # sand or snow at a *smaller* dt than its own gives it MORE creep. A shared grid forces exactly
    # that on every material when snow is present. Measured here, canonical-vs-canonical, so the
    # demo page can state the size of the effect instead of hand-waving at it.
    shared = C.shared_dt(MATS)
    creep = {}
    for m in MATS:
        own = C.MAT[m]["dt"]
        r_own = C.repose_angle(store["heap_" + m + "_base"][-1])
        if abs(own - shared) < 1e-12:
            creep[m] = {"dt_own": own, "repose_own": r_own, "dt_shared": shared,
                        "repose_shared": r_own, "delta_deg": 0.0}
            continue
        print("canonical %-8s at SHARED dt=%.1e ..." % (m, shared), flush=True)
        s, _, _ = C.simulate(m, pts, area, T, N_FRAMES, dt=shared)
        store["heap_" + m + "_shareddt"] = s
        r_sh = C.repose_angle(s[-1])
        creep[m] = {"dt_own": own, "repose_own": float(r_own), "dt_shared": shared,
                    "repose_shared": float(r_sh), "delta_deg": float(r_sh - r_own)}
        print("   repose %.1f -> %.1f deg" % (r_own, r_sh), flush=True)

    # ---------------------------------------------------------------- the mixed four-material scene
    # Four blobs at equal particle density released together over a shallow pool, so the four
    # materials both show their own character and interact through the one shared grid.
    R = 0.075
    ablob = float(np.pi * R * R)
    k = 500
    dens = k / ablob                                   # particles per unit area, shared by all groups
    pool_x, pool_y = (0.10, 0.90), (C.floor_y, C.floor_y + 0.055)
    apool = (pool_x[1] - pool_x[0]) * (pool_y[1] - pool_y[0])
    kpool = int(round(dens * apool))
    # `area` is only ever consumed as p_vol = area/k, so each group's area is set to k * p_vol
    # EXACTLY rather than to its nominal geometric area -- rounding the pool's particle count would
    # otherwise leave it at a p_vol 4e-5 relative away from the blobs', which a single global p_vol
    # in the WebGPU engine cannot represent. The physical quantity (mass per particle) is identical
    # across all five groups by construction.
    pv0 = ablob / k
    groups = [
        {"material": "fluid", "pts": C.seed_box(pool_x[0], pool_x[1], pool_y[0], pool_y[1], kpool, seed=3),
         "area": pv0 * kpool},
        {"material": "elastic", "pts": C.seed_disk((0.22, 0.62), R, k, seed=11), "area": pv0 * k},
        {"material": "snow", "pts": C.seed_disk((0.42, 0.72), R, k, seed=12), "area": pv0 * k},
        {"material": "sand", "pts": C.seed_disk((0.62, 0.62), R, k, seed=13), "area": pv0 * k},
        {"material": "fluid", "pts": C.seed_disk((0.80, 0.72), R, k, seed=14), "area": pv0 * k},
    ]
    pv = [g["area"] / g["pts"].shape[0] for g in groups]
    assert max(pv) - min(pv) < 1e-9 * max(pv), "groups must share one p_vol: %s" % pv
    mixed_pts = np.concatenate([g["pts"] for g in groups], 0).astype(np.float32)
    mixed_mat = np.concatenate([[C.MAT_ID[g["material"]]] * g["pts"].shape[0] for g in groups]).astype(np.int32)
    nm = mixed_pts.shape[0]
    dtm = C.shared_dt([g["material"] for g in groups])
    spfm = spf_for(dtm)
    print("canonical mixed4  n=%d dt=%.1e spf=%d ..." % (nm, dtm, spfm), flush=True)
    ma, _, mid, ok_m, _ = C.simulate_multi(groups, T, N_FRAMES)
    mb, _, _, _, _ = C.simulate_multi(groups, T, N_FRAMES)
    gn = [dict(g, pts=g["pts"] + NUDGE) for g in groups]
    mc, _, _, _, _ = C.simulate_multi(gn, T, N_FRAMES)
    print("   stable:", ok_m, flush=True)
    assert (mid == mixed_mat).all()
    store["mixed4_base"] = ma
    store["mixed4_rep"] = mb
    store["mixed4_nudge"] = mc
    store["mixed4_pts"] = mixed_pts
    store["mixed4_mat"] = mixed_mat
    (V / "ic_mixed4.f32").write_bytes(mixed_pts.reshape(-1).tobytes())
    (V / "ic_mixed4_mat.i32").write_bytes(mixed_mat.tobytes())
    job["ics"].append({"name": "mixed4", "kind": "multi", "pts_file": "ic_mixed4.f32",
                       "mat_file": "ic_mixed4_mat.i32", "n": int(nm), "area": float(pv[0] * nm),
                       "p_vol": float(pv[0]), "dt": dtm, "spf": spfm, "n_frames": N_FRAMES,
                       "materials": [g["material"] for g in groups]})

    # ---------------------------------------------------------------- browser-side benchmarks
    job["bench"] = {
        # the shipped scene: four materials present, snow's dt, at the particle counts the demo runs
        "realtime_n": [2048, 4096, 6144, 8192, 12288, 16384],
        "realtime_frames": 24,
        # deliberate piling: everything crushed into one corner to find the worst particles-per-cell
        # a user can actually reach, because 2^24 wraps SILENTLY at 256 particle masses on a node
        "pile_n": [4096, 8192, 12288],
        "pile_frames": 90,
        "kM": phys.__dict__.get("kM", 24),
    }
    job["shared_dt"] = float(shared)
    (V / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    np.savez_compressed(V / "base.npz", **store)
    (V / "creep.json").write_text(json.dumps(creep, indent=2), encoding="utf-8")
    print("\nshared dt =", shared, " creep:", json.dumps(creep, indent=1))
    print("wrote job.json + base.npz + creep.json")


if __name__ == "__main__":
    main()
