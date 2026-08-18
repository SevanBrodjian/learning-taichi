"""Score the browser's four-material rollouts against canonical `sim.physics`.

The comparison is against the SELF-NOISE BAND, never against zero. Canonical is stochastic at the
1e-7 level (GPU atomic ordering), and an MPM rollout is chaotic, so two canonical runs of the *same*
scene separate too. Two references are computed for every scene:

    repeat   canonical vs canonical, identical inputs      -> the pure non-determinism floor
    nudge    canonical vs canonical, ICs moved by 1e-7     -> the float32-rounding band

A port that lands at or below `nudge` is doing the same arithmetic to within the precision the
ground truth itself is defined to.

`traj_rmse` is the registered name for the distance used here -- READ ITS ENTRY: it is a MEAN
PER-PARTICLE DISTANCE, not a root-mean-square, and it has no interpretable absolute scale. That is
exactly why every number below is quoted beside the band and beside a SHAPE metric
(`repose_angle`, `pile_height`, `spread_width`) that is read directly against ground truth.
"""
import json
import pathlib
import sys

import numpy as np

RUN = pathlib.Path(__file__).resolve().parents[1]
V = RUN / "verify"
OUT = V / "out"
sys.path.insert(0, str(RUN.parents[2]))

import sim.physics as phys                       # noqa: E402
from sim.physics import core as C                # noqa: E402


def traj_rmse(a, b):
    """Registered `traj_rmse` (spec/registry/metrics.json): mean over frames and particles of the
    per-particle Euclidean distance. Particle p is matched to particle p; both runs seed identically."""
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def shape(snap):
    return {"repose_angle": float(C.repose_angle(snap)),
            "pile_height": float(C.pile_height(snap)),
            "spread_width": float(C.spread_width(snap))}


def main():
    job = json.loads((V / "job.json").read_text())
    bench = json.loads((OUT / "bench.json").read_text())
    base = np.load(V / "base.npz", allow_pickle=False)
    creep = json.loads((V / "creep.json").read_text())

    rows = []
    for ic in job["ics"]:
        name = ic["name"]
        gt = base[name + "_base"]
        rep = base[name + "_rep"]
        nud = base[name + "_nudge"]
        raw = np.frombuffer((OUT / ("traj_" + name + ".f32")).read_bytes(), dtype=np.float32)
        web = raw.reshape(ic["n_frames"], ic["n"], 2)

        row = {
            "scene": name, "kind": ic["kind"], "material": ic.get("material"),
            "n": ic["n"], "dt": ic["dt"], "substeps_per_frame": ic["spf"],
            "frames": ic["n_frames"],
            "traj_rmse_web_vs_canonical": traj_rmse(web, gt),
            "self_noise_repeat": traj_rmse(rep, gt),
            "self_noise_nudge": traj_rmse(nud, gt),
            "final_frame_rmse": traj_rmse(web[-1], gt[-1]),
            "shape_canonical": shape(gt[-1]),
            "shape_web": shape(web[-1]),
            "n_nonfinite": int((~np.isfinite(web)).sum()),
        }
        row["ratio_to_nudge_band"] = row["traj_rmse_web_vs_canonical"] / max(row["self_noise_nudge"], 1e-12)
        if ic["kind"] == "multi":
            mats = base["mixed4_mat"]
            per = {}
            for nm, mid in C.MAT_ID.items():
                m = mats == mid
                if not m.any():
                    continue
                per[nm] = {
                    "n": int(m.sum()),
                    "traj_rmse_web_vs_canonical": traj_rmse(web[:, m], gt[:, m]),
                    "self_noise_nudge": traj_rmse(nud[:, m], gt[:, m]),
                    "self_noise_repeat": traj_rmse(rep[:, m], gt[:, m]),
                }
                per[nm]["ratio_to_nudge_band"] = (per[nm]["traj_rmse_web_vs_canonical"]
                                                  / max(per[nm]["self_noise_nudge"], 1e-12))
            row["per_material"] = per
        rows.append(row)

    summary = {
        "physics_version": phys.VERSION,
        "device": bench["device"],
        "user_agent": bench["user_agent"],
        "shared_dt": bench["shared_dt"],
        "substeps_per_frame_shared": bench["substeps_per_frame"],
        "scenes": rows,
        "realtime": bench["realtime"],
        "pile_headroom": bench["pile"],
        "substep_table": bench["substep_table"],
        "shared_dt_creep": creep,
        "webgpu_errors": bench["webgpu_errors"],
        "svd_unit_test": json.loads((OUT / "svd_score.json").read_text()),
    }
    (OUT / "score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("%-14s %10s %10s %10s %7s   %s" %
          ("scene", "traj_rmse", "noise(rep)", "noise(nudge)", "x band", "repose web/canon"))
    for r in rows:
        print("%-14s %10.2e %10.2e %10.2e %7.1f   %5.1f / %5.1f deg" %
              (r["scene"], r["traj_rmse_web_vs_canonical"], r["self_noise_repeat"],
               r["self_noise_nudge"], r["ratio_to_nudge_band"],
               r["shape_web"]["repose_angle"], r["shape_canonical"]["repose_angle"]))
        if "per_material" in r:
            for nm, v in r["per_material"].items():
                print("    %-10s n=%4d  rmse %.2e   band %.2e   x%.1f" %
                      (nm, v["n"], v["traj_rmse_web_vs_canonical"], v["self_noise_nudge"],
                       v["ratio_to_nudge_band"]))
    print()
    for r in bench["realtime"]:
        print("realtime n=%6d  %6.2f ms  %5.1f fps  gpu %5.2f ms  motion_ok=%s" %
              (r["n"], r["sustained_ms"], r["fps"], r["gpu_ms"], r["motion_ok"]))
    print()
    for r in bench["pile"]:
        print("pile n=%6d  max node mass %6.1f pm (saturates %d, headroom x%.2f)" %
              (r["n"], r["max_node_mass_pm"], r["mass_saturates_at_pm"], r["mass_headroom"]))
    print("\nwrote", OUT / "score.json")


if __name__ == "__main__":
    main()
