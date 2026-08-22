"""Write the harness job and the canonical reference trajectories it is scored against.

Phase 'cost' runs with UNTRAINED weights and answers the deployability half on its own; phase
'trained' re-runs with the real weights, adds the host-vs-shader parity probe and a learned rollout.
Splitting them is deliberate: inference cost does not depend on the weight VALUES (T-022), so the
expensive, decisive half of this task does not have to wait for training to converge -- and section 7
re-times the trained net so that claim is tested here rather than inherited.

    .venv/Scripts/python.exe runs/.../verify/prepare.py cost
    .venv/Scripts/python.exe runs/.../verify/prepare.py trained --hidden 64
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "train"))
sys.path.insert(0, str(RUN.parents[2]))

from sim.physics import core          # noqa: E402
import sim.physics as phys            # noqa: E402
import netspec as NS                  # noqa: E402

N_REF = 2048
NF_REF = 40
T_REF = 1.0
DT = min(core.MAT[m]["dt"] for m in ("fluid", "elastic", "snow", "sand"))   # the shared timestep


def mean_dist(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def make_refs():
    """One canonical reference rollout per material, at the SHARED timestep the WebGPU solver runs
    (one grid means one dt; sand's canonical solo dt is 1e-4 but a four-material solver has to run
    5e-5, and the reference has to be generated the same way or the comparison is void)."""
    refs = []
    rng = np.random.default_rng(11)
    for m in ("fluid", "elastic", "snow", "sand"):
        pts = core.seed_disk((0.5, 0.55), 0.14, N_REF, seed=7).astype(np.float32)
        a, _, ok = core.simulate(m, pts, np.pi * 0.14 ** 2, T_REF, NF_REF, dt=DT)
        b, _, _ = core.simulate(m, pts + rng.normal(0, 1e-7, pts.shape), np.pi * 0.14 ** 2,
                                T_REF, NF_REF, dt=DT)
        band = mean_dist(a, b)
        (HERE / f"ic_{m}.f32").write_bytes(pts.astype(np.float32).tobytes())
        (HERE / f"ref_{m}.f32").write_bytes(a.astype(np.float32).tobytes())
        refs.append({"material": m, "ic_file": f"ic_{m}.f32", "traj_file": f"ref_{m}.f32",
                     "n": N_REF, "n_frames": NF_REF, "T": T_REF, "stable": bool(ok),
                     "ic_nudge_band": band})
        print(f"  ref {m:8s} stable={ok}  1e-7 IC-nudge band {band:.3e}")
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["cost", "trained"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--refresh-refs", action="store_true")
    a = ap.parse_args()

    job = {
        "id": f"t028-{a.phase}",
        "phase": a.phase,
        "dt": DT,
        "n": 8192,
        # fine spacing across 64..112 on purpose: the first pass put a sharp cost jump
        # between 80 and 96, and a cliff you have only bracketed to within a factor of
        # 1.2 in width is a rumour, not a measurement
        "widths": [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 84, 88, 92, 96, 104, 112,
                   128, 160, 192, 224, 256],
        "f16_widths": [16, 32, 64, 128, 160, 192, 224, 256],
        "storage_widths": [16, 32, 64, 128],
        # same shader with the hidden loop bound read from a uniform, so the compiler
        # cannot unroll it -- the control that tells a compiler cliff from a memory one
        "dyn_widths": [16, 32, 64, 88, 96, 128, 192, 256],
        "timing_substeps": 128,
        "reps": 9,
        "n_sweep": [1024, 2048, 4096, 8192, 16384],
        "n_sweep_variants": ["analytic", "nn16", "nn64", "nn128"],
        "batch_variants": ["analytic", "nn64"],
        "batch_sizes": [1, 4, 16, 64, 256],
        "physics_version": phys.VERSION,
    }
    if a.phase == "cost" or a.refresh_refs:
        job["refs"] = make_refs()
    else:
        job["refs"] = json.loads((HERE / "refs.json").read_text())
    (HERE / "refs.json").write_text(json.dumps(job["refs"], indent=1))

    if a.phase == "trained":
        w = np.load(RUN / "train" / f"weights_h{a.hidden}.npz")
        flat = w["wgsl"].astype(np.float32)
        (HERE / "weights.f32").write_bytes(flat.tobytes())
        probe_in = np.load(RUN / "train" / "probe_inputs.npz")["X"].astype(np.float32)
        (HERE / "probe_in.f32").write_bytes(probe_in.tobytes())
        roll = json.loads((RUN / "train" / "wgsl_rollout.json").read_text())
        job["trained"] = {
            "hidden": int(a.hidden), "weights_file": "weights.f32",
            "probe_in_file": "probe_in.f32", "rollout": roll,
        }
        print(f"  trained weights: {flat.size} floats, probe inputs {probe_in.size // 16} samples")

    (HERE / "job.json").write_text(json.dumps(job, indent=1))
    print("wrote", HERE / "job.json", "phase", a.phase, "dt", DT)


if __name__ == "__main__":
    main()
