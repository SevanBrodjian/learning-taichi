"""A second, harder verification: the same elastic disk launched sideways so it bounces and rolls.

The canonical `drop` scene settles quickly. A launched disk keeps hitting the floor, so it spends
the whole rollout inside the Coulomb-friction branch of the grid update and the contact-dominated
part of the constitutive model, which is where a port is most likely to be subtly wrong.

Scene note: sim.physics does not own this initial condition (scenes are not centralised there), so
it is specified here: disk of radius 0.11 at (0.30, 0.55), n = 2048, v0 = (0.75, 0). Everything
else -- E, dt, grid, friction, gravity -- still comes from sim.physics unchanged.

    .venv/Scripts/python.exe runs/material-variants/interactive-simulation-of-one-material/verify/launch_scene.py
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sim.physics as phys                      # noqa: E402

N = 2048
T = 2.5
N_FRAMES = 150
V0 = (0.75, 0.0)
CENTER = (0.30, 0.55)
RADIUS = 0.11
PERTURB = 1e-7


def traj_rmse(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def per_frame_dist(a, b):
    return np.mean(np.linalg.norm(a - b, axis=-1), axis=1)


def main():
    pts = phys.seed_disk(CENTER, RADIUS, N, seed=3).astype(np.float32)
    area = float(np.pi * RADIUS ** 2)
    meta = {"n": N, "area": area, "T": T, "n_frames": N_FRAMES, "pts_file": "ic_launch.f32",
            "v0": list(V0), "scene": "launch (disk r=0.11 at (0.30,0.55), v0=(0.75,0))",
            "physics_version": phys.VERSION}
    (HERE / "ic_launch.f32").write_bytes(pts.tobytes())
    (HERE / "ic_launch.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[canonical] A"); snapA, times, okA = phys.simulate("elastic", pts, area, T, N_FRAMES, v0=V0)
    print("[canonical] B"); snapB, _, okB = phys.simulate("elastic", pts, area, T, N_FRAMES, v0=V0)
    rng = np.random.default_rng(11)
    ptsP = (pts + rng.normal(0, PERTURB, pts.shape)).astype(np.float32)
    print("[canonical] A' (IC nudged)"); snapP, _, okP = phys.simulate("elastic", ptsP, area, T, N_FRAMES, v0=V0)

    print("[node] port")
    r = subprocess.run(["node", str(HERE / "run_port.js"), str(HERE / "ic_launch.json"), str(HERE),
                        "launch"], capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); raise SystemExit("node failed")
    rep = json.loads((HERE / "port_launch_report.json").read_text())

    def load(tag):
        return np.fromfile(HERE / f"port_launch_{tag}.f32", dtype=np.float32).reshape(N_FRAMES, N, 2)

    port = load("canonical")
    out = {
        "scene": meta["scene"], "n_particles": N, "T": T, "n_frames": N_FRAMES, "v0": list(V0),
        "canonical_stable": bool(okA and okB and okP),
        "traj_rmse": {
            "port_vs_canonical": traj_rmse(port, snapA),
            "canonical_self_noise": traj_rmse(snapB, snapA),
            "canonical_perturbed_ic": traj_rmse(snapP, snapA),
        },
        "per_frame": {
            "times": times.tolist(),
            "port_vs_canonical": per_frame_dist(port, snapA).tolist(),
            "canonical_self_noise": per_frame_dist(snapB, snapA).tolist(),
            "canonical_perturbed_ic": per_frame_dist(snapP, snapA).tolist(),
        },
        "shape": {
            "gt_final_width": phys.spread_width(snapA[-1]),
            "port_final_width": phys.spread_width(port[-1]),
            "gt_final_height": phys.pile_height(snapA[-1]),
            "port_final_height": phys.pile_height(port[-1]),
        },
        "dt_sweep": [],
    }
    for e in rep["dt_sweep"]:
        s = load(f"dt{e['mult']}")
        finite = bool(np.isfinite(s).all())
        row = dict(e)
        row["finite"] = finite
        row["traj_rmse_vs_canonical"] = traj_rmse(np.nan_to_num(s), snapA) if finite else float("nan")
        row["final_width"] = float(row["final_width"])
        out["dt_sweep"].append(row)

    m = json.loads((RUN / "metrics.json").read_text())
    m["launch_scene"] = out
    (RUN / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    np.save(HERE / "launch_snapA.npy", snapA)
    print(json.dumps(out["traj_rmse"], indent=2))
    for e in out["dt_sweep"]:
        print(f"  x{e['mult']:<4} finite={e['finite']} rmse={e['traj_rmse_vs_canonical']:.4g}")


if __name__ == "__main__":
    main()
