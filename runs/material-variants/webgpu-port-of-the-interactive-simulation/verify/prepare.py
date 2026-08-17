"""Step 1 of verification: build the initial conditions, roll CANONICAL ground truth, write job.json.

Ground truth is a FORWARD sim from `sim.physics` -- imported, never re-derived. Three canonical rolls
per scene bound how much of any divergence is just chaos rather than a port defect:
  A   the reference
  B   the identical code again (GPU atomic ordering is non-deterministic, so B != A bit for bit)
  A'  the identical code with the initial positions nudged by 1e-7, the scale of one f32 rounding
      unit at these coordinates

Anything the WebGPU port scores between traj_rmse(B,A) and traj_rmse(A',A) is inside the simulator's
own noise. Anything above traj_rmse(A',A) is a real numerical difference.

    .venv/Scripts/python.exe runs/material-variants/webgpu-port-of-the-interactive-simulation/verify/prepare.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sim.physics as phys                       # noqa: E402

N = 2048
T = 2.5
N_FRAMES = 150
PERTURB = 1e-7

# The scale sweep. kM/kV are exponents in units of one particle mass, so they are scene-independent:
# an accumulator value of 2^kM means one particle's mass sitting at a node.
#   u32 mass     saturates at 2^(32-kM) particle masses at a single node
#   i32 momentum saturates at 2^(31-kV) particle-mass*velocity at a single node
VARIANTS = [
    {"name": "fixed_k12", "atomics": "fixed", "kM": 12, "kV": 10},
    {"name": "fixed_k16", "atomics": "fixed", "kM": 16, "kV": 14},
    {"name": "fixed_k20", "atomics": "fixed", "kM": 20, "kV": 18, "probe": True},
    {"name": "fixed_k22", "atomics": "fixed", "kM": 22, "kV": 20},
    {"name": "fixed_k24", "atomics": "fixed", "kM": 24, "kV": 22, "probe": True},
    {"name": "fixed_k26", "atomics": "fixed", "kM": 26, "kV": 24},
    {"name": "casf32", "atomics": "casf32", "kM": 20, "kV": 18},
]


def main():
    scenes = {}

    # --- scene 1: the canonical drop disk -----------------------------------------------------
    sc = phys.scene("drop", N)
    scenes["drop"] = {"pts": sc["pts"].astype(np.float32), "area": float(sc["area"]), "v0": (0.0, 0.0),
                      "desc": "drop (disk r=0.11 at (0.5,0.52), released from rest)"}

    # --- scene 2: the launched disk (the harder one: it lives in the friction branch) ----------
    pts = phys.seed_disk((0.30, 0.55), 0.11, N, seed=3).astype(np.float32)
    scenes["launch"] = {"pts": pts, "area": float(np.pi * 0.11 ** 2), "v0": (0.75, 0.0),
                        "desc": "launch (disk r=0.11 at (0.30,0.55), v0=(0.75,0)) -- bounces and rolls"}

    ics = []
    for name, s in scenes.items():
        (HERE / f"ic_{name}.f32").write_bytes(s["pts"].tobytes())
        ics.append({"name": name, "n": N, "area": s["area"], "T": T, "n_frames": N_FRAMES,
                    "v0": list(s["v0"]), "pts_file": f"ic_{name}.f32", "desc": s["desc"]})

        if (HERE / f"gt_{name}_A.npy").exists() and "--force-gt" not in sys.argv:
            print(f"[canonical {name}] cached (pass --force-gt to regenerate)")
            continue
        print(f"[canonical {name}] A")
        A, times, okA = phys.simulate("elastic", s["pts"], s["area"], T, N_FRAMES, v0=s["v0"])
        print(f"[canonical {name}] B (same code, GPU atomics reorder)")
        B, _, okB = phys.simulate("elastic", s["pts"], s["area"], T, N_FRAMES, v0=s["v0"])
        print(f"[canonical {name}] A' (IC nudged by {PERTURB:g})")
        rng = np.random.default_rng(7 if name == "drop" else 11)
        ptsP = (s["pts"] + rng.normal(0, PERTURB, s["pts"].shape)).astype(np.float32)
        Ap, _, okP = phys.simulate("elastic", ptsP, s["area"], T, N_FRAMES, v0=s["v0"])
        np.save(HERE / f"gt_{name}_A.npy", A)
        np.save(HERE / f"gt_{name}_B.npy", B)
        np.save(HERE / f"gt_{name}_P.npy", Ap)
        np.save(HERE / f"gt_{name}_t.npy", times)
        print("   stable:", bool(okA and okB and okP))

    job = {
        "id": "webgpu-verify-1",
        "physics_version": phys.VERSION,
        "ics": ics,
        "variants": VARIANTS,
        "bench": {
            # Particle counts are seeded at CONSTANT DENSITY (same particles-per-cell as the
            # reference disk) into a growing box, capped at the domain. Past the cap the density
            # rises, so the curve past that point mixes "more particles" with "more atomic
            # contention" -- the harness reports particles_per_cell so that is visible, not hidden.
            "scaling": [{"n": n, "frames": f, "atomics": "fixed"} for n, f in
                        [(500, 60), (1000, 60), (2048, 60), (4096, 60), (8192, 60), (16384, 60),
                         (32768, 40), (49152, 40), (65536, 30), (98304, 24), (131072, 20),
                         (196608, 14), (262144, 12)]],
            "atomics_n": [2048, 16384, 49152],
            "substep_sweep_n": 16384,
            "substeps": [1, 10, 42, 84, 139, 167, 333, 500],
            "floor_substeps": [16, 64, 167, 334, 668, 1336],
            "density_probe_n": [2048, 8192, 32768, 49152],
        },
    }
    (HERE / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    for stale in ("done.json", "bench.json"):
        p = out / stale
        if p.exists():
            p.unlink()
    print("wrote job.json; physics_version", phys.VERSION)


if __name__ == "__main__":
    main()
