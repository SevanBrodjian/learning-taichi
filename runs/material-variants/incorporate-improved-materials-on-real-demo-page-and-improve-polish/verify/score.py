"""Score the browser's trajectories against canonical, and read the buoyancy ordering off BOTH.

The pass condition for this task's physics half is not "the numbers are small". It is:

    on the DEMO'S OWN WGSL solver, snow rises and rubber and sand sink, in that order by density,

with the agreement against canonical judged against canonical's own SELF-NOISE band (a repeat run
plus a 1e-7-nudged run), not against zero.

    .venv/Scripts/python.exe runs/.../verify/score.py
"""
import json
import pathlib
import sys

import numpy as np

RUN = pathlib.Path(__file__).resolve().parents[1]
V = RUN / "verify"
OUT = V / "out"
sys.path.insert(0, str(RUN.parents[2]))

from sim.physics import core as C                # noqa: E402


def per_particle_mean_dist(a, b):
    """Mean over particles and frames of |a - b|. Registered as `traj_rmse` in
    spec/registry/metrics.json -- and, as that entry warns, it is a MEAN PER-PARTICLE DISTANCE
    despite the name, not a root-mean-square and not a centre-of-mass distance."""
    return float(np.linalg.norm(a - b, axis=-1).mean())


def main():
    job = json.loads((V / "job.json").read_text())
    base = np.load(V / "base.npz")
    bench = json.loads((OUT / "bench.json").read_text())
    res = {"physics_version": job["physics_version"], "device": bench["device"],
           "user_agent": bench["user_agent"], "rollouts": {}, "buoyancy": {}}

    for ic in job["ics"]:
        name = ic["name"]
        f = OUT / ("traj_%s.f32" % name)
        if not f.exists():
            print("MISSING", f)
            continue
        n, nf = ic["n"], ic["n_frames"]
        web = np.frombuffer(f.read_bytes(), dtype=np.float32).reshape(nf, n, 2)
        a = base[name + "_base"]
        noise = max(per_particle_mean_dist(a, base[name + "_rep"]),
                    per_particle_mean_dist(a, base[name + "_nudge"]))
        d = per_particle_mean_dist(a, web)
        res["rollouts"][name] = {"traj_rmse": d, "self_noise": noise,
                                 "ratio_to_self_noise": d / max(noise, 1e-12),
                                 "n": n, "frames": nf, "dt": ic["dt"], "spf": ic["spf"],
                                 "n_nonfinite": int((~np.isfinite(web)).sum())}
        print("%-14s traj_rmse %.5f   self-noise %.5f   ratio %6.2f" % (name, d, noise, d / max(noise, 1e-12)))

        # ------------------------------------------------------------ buoyancy read-out
        mats = base[name + "_mat"]
        if not name.startswith("pool_"):
            continue
        solids = ([s for s in ["snow", "elastic", "sand"]] if name == "pool_three"
                  else [name.split("_", 1)[1]])
        row = {}
        for s in solids:
            sel = mats == C.MAT_ID[s]
            fl = mats == C.MAT_ID["fluid"]
            if s == "fluid":                      # the control: the blob is the tail group
                idx = np.where(sel)[0][-500:]
                sel = np.zeros(len(mats), bool); sel[idx] = True
                fl = np.ones(len(mats), bool); fl[idx] = False
            e = {}
            for tag, tr in (("canonical", a), ("webgpu", web)):
                e[tag] = {
                    "rest_depth_change": C.rest_depth(tr[-1][sel], tr[-1][fl]) -
                                         C.rest_depth(tr[0][sel], tr[0][fl]),
                    "submerged_fraction": C.submerged_fraction(tr[-1][sel], tr[-1][fl]),
                    "mean_y_end": float(tr[-1][sel][:, 1].mean()),
                    "mean_y_start": float(tr[0][sel][:, 1].mean()),
                }
            e["rho"] = C.MAT[s].get("rho", C.p_rho)
            e["rises_webgpu"] = bool(e["webgpu"]["rest_depth_change"] < 0)
            e["rises_canonical"] = bool(e["canonical"]["rest_depth_change"] < 0)
            e["sign_agrees"] = e["rises_webgpu"] == e["rises_canonical"]
            row[s] = e
            print("   %-8s rho %.2f | canonical d(depth) %+.4f sub %.3f  |  webgpu d(depth) %+.4f sub %.3f  | %s"
                  % (s, e["rho"], e["canonical"]["rest_depth_change"], e["canonical"]["submerged_fraction"],
                     e["webgpu"]["rest_depth_change"], e["webgpu"]["submerged_fraction"],
                     "AGREE" if e["sign_agrees"] else "DISAGREE"))
        res["buoyancy"][name] = row

    # ------------------------------------------------------------------ the pass condition
    t = res["buoyancy"].get("pool_three", {})
    if t:
        ys = {s: t[s]["webgpu"]["mean_y_end"] for s in t}
        ordered = ys.get("snow", 0) > ys.get("elastic", 0) > ys.get("sand", 0)
        res["ordering_pass"] = {
            "mean_y_end_webgpu": ys,
            "snow_above_rubber_above_sand": bool(ordered),
            "snow_rises_webgpu": bool(t["snow"]["rises_webgpu"]),
            "rubber_sinks_webgpu": bool(not t["elastic"]["rises_webgpu"]),
            "sand_sinks_webgpu": bool(not t["sand"]["rises_webgpu"]),
        }
        print("\nORDERING ON THE DEMO'S OWN SOLVER:", json.dumps(res["ordering_pass"], indent=1))

    res["solver"] = bench["realtime"]
    res["render"] = bench["render"]
    res["pile"] = bench["pile"]
    res["substep_table"] = bench["substep_table"]
    (V / "score.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nwrote", V / "score.json")


if __name__ == "__main__":
    main()
