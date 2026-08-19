"""Export the canonical ground truth the browser harness is measured against, and write job.json.

Produces:
  ic_drop.f32           the exact seed point set (float32 xy pairs) sim.physics used
  ref_drop.f32          the canonical Taichi fluid trajectory from that seed  (n_frames, n, 2)
  ref_drop_b.f32        a SECOND canonical run from the same seed -- GPU atomics accumulate in a
                        different order run to run, so this is the reference's own noise floor and
                        nothing inside it is a detectable difference
  ref_stats.json        bulk shape diagnostics of the reference
  job.json              what the browser harness should run
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import sim.physics as phys                       # noqa: E402
from sim.physics import core as C                # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1]
V = RUN / "verify"
F = phys.MAT["fluid"]

N = 4096
T = 1.0
FRAMES = 60


def main():
    V.mkdir(parents=True, exist_ok=True)
    (V / "out").mkdir(exist_ok=True)
    sc = C.scene("drop", N)
    pts = sc["pts"].astype(np.float32)
    n = pts.shape[0]
    (V / "ic_drop.f32").write_bytes(pts.tobytes())

    a, _, st_a = phys.simulate("fluid", sc["pts"], sc["area"], T, FRAMES)
    b, _, st_b = phys.simulate("fluid", sc["pts"], sc["area"], T, FRAMES)
    (V / "ref_drop.f32").write_bytes(a.astype(np.float32).tobytes())
    (V / "ref_drop_b.f32").write_bytes(b.astype(np.float32).tobytes())
    self_noise = float(np.linalg.norm(a - b, axis=2).mean())
    stats = {
        "n": int(n), "T": T, "frames": FRAMES, "stable_a": bool(st_a), "stable_b": bool(st_b),
        "self_noise": self_noise,
        "final_spread_width": float(C.spread_width(a[-1])),
        "final_pile_height": float(C.pile_height(a[-1])),
        "physics_version": phys.VERSION, "dt": F["dt"], "area": float(sc["area"]),
        "substeps_per_frame_render": int(round((T / FRAMES) / F["dt"])),
    }
    (V / "ref_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

    job = {
        "id": "nn-grid-profile-1",
        "ic": {"name": "drop", "pts_file": "ic_drop.f32", "n": int(n), "area": float(sc["area"]),
               "T": T, "n_frames": FRAMES, "v0": [0.0, 0.0]},
        # ---- the sweep: network size x particle count ----
        # widths 8/16/32/64 span the range where a per-cell MLP is plausibly affordable: 8 is about
        # the smallest that can express a two-branch conditional at all, 64 is where the weight set
        # (4866 floats) stops fitting comfortably in a workgroup's share of L1.
        "widths": [8, 16, 32, 64],
        "particle_counts": [512, 2048, 8192, 16384, 32768],
        "grid_kinds": ["null", "analytic", "nn", "nnsparse"],
        "timed_substeps": 120,
        "reps": 5,
        "floor_substeps": [64, 256, 512],
        "occupancy_n": [512, 2048, 8192, 16384, 32768],
        # ---- accuracy: how long does a learned rollout survive ----
        "survival": {"n": int(n), "max_substeps": 8000, "check_every": 100,
                     "nets": ["point8", "point16", "point32", "point64",
                              "deriv8", "deriv16", "deriv32", "deriv64"]},
        "traj_nets": ["point64", "deriv64", "point16"],
    }
    (V / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    print("wrote job.json")


if __name__ == "__main__":
    main()
