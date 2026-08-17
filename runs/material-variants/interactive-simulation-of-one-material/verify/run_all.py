"""Verify the browser port against the canonical Taichi simulator, and render the evidence.

Pipeline (one process, start to finish):
  1. build the canonical `drop` initial condition from sim.physics and export it for the port
  2. roll the canonical simulator three times: A, B (same code, different atomic ordering on the
     GPU), and A' (identical code, initial condition perturbed by 1e-7 -- the scale of f32
     rounding). A-vs-B and A-vs-A' bound how much of any divergence is just chaos.
  3. invoke Node on the same initial condition to roll the JS port
  4. score everything with the registered metric traj_rmse
  5. render the comparison as motion, plus the divergence plot and the timestep sweep

Run from the repo root:
    .venv/Scripts/python.exe runs/material-variants/interactive-simulation-of-one-material/verify/run_all.py
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
N_FRAMES = 150                                  # one sample every 16.7 ms of simulated time
PERTURB = 1e-7


def traj_rmse(a, b):
    """Registered metric traj_rmse: mean over frames and particles of |x_a - x_b|.
    See spec/registry/metrics.json -- it is a mean absolute distance, not an RMS."""
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def per_frame_dist(a, b):
    return np.mean(np.linalg.norm(a - b, axis=-1), axis=1)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    sc = phys.scene("drop", N)
    pts = sc["pts"].astype(np.float32)
    area = float(sc["area"])

    meta = {"n": N, "area": area, "T": T, "n_frames": N_FRAMES,
            "pts_file": "ic.f32", "scene": "drop (disk at (0.5,0.52), r=0.11)",
            "physics_version": phys.VERSION}
    (HERE / "ic.f32").write_bytes(pts.tobytes())
    (HERE / "ic.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[canonical] run A")
    snapA, times, okA = phys.simulate("elastic", pts, area, T, N_FRAMES)
    print("[canonical] run B (same input, GPU atomics reorder)")
    snapB, _, okB = phys.simulate("elastic", pts, area, T, N_FRAMES)
    print("[canonical] run A' (IC perturbed by %g)" % PERTURB)
    rng = np.random.default_rng(7)
    ptsP = (pts + rng.normal(0, PERTURB, pts.shape)).astype(np.float32)
    snapP, _, okP = phys.simulate("elastic", ptsP, area, T, N_FRAMES)

    print("[node] rolling the JS port")
    node = subprocess.run(["node", str(HERE / "run_port.js"), str(HERE / "ic.json"), str(HERE)],
                          capture_output=True, text=True, cwd=str(ROOT))
    if node.returncode != 0:
        print(node.stdout[-4000:]); print(node.stderr[-4000:]); raise SystemExit("node failed")
    rep = json.loads((HERE / "port_report.json").read_text())

    def load(tag):
        arr = np.fromfile(HERE / f"port_{tag}.f32", dtype=np.float32)
        return arr.reshape(N_FRAMES, N, 2)

    port = load("canonical")

    out = {
        "physics_version": phys.VERSION,
        "scene": meta["scene"], "n_particles": N, "n_grid": phys.core.n_grid,
        "T": T, "n_frames": N_FRAMES,
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
            "gt_roundness": phys.circularity(snapA[-1]),
            "port_roundness": phys.circularity(port[-1]),
        },
        "node_report": rep,
    }

    # --- the timestep sweep, scored against the canonical-dt canonical run -------------------
    sweep = []
    for e in rep["dt_sweep"]:
        s = load(f"dt{e['mult']}")
        finite = np.isfinite(s).all()
        row = dict(e)
        row["traj_rmse_vs_canonical"] = traj_rmse(np.nan_to_num(s), snapA) if finite else float("nan")
        row["finite"] = bool(finite)
        row["final_width"] = float(row["final_width"])
        sweep.append(row)
    out["dt_sweep"] = sweep

    (RUN / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("traj_rmse", "shape")}, indent=2))

    np.save(HERE / "snapA.npy", snapA)
    np.save(HERE / "snapB.npy", snapB)
    np.save(HERE / "snapP.npy", snapP)
    np.save(HERE / "times.npy", times)
    print("saved metrics + trajectories")


if __name__ == "__main__":
    main()
