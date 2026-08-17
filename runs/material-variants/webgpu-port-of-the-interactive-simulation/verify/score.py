"""Step 3: score every WebGPU rollout against canonical, fold in the benchmarks, write metrics.json.

The comparison protocol is the one the JS port used, because it is the only one that can tell a real
numerical difference from chaos:

    traj_rmse(port, A)  is meaningful only against  traj_rmse(B, A)  and  traj_rmse(A', A)

where B is the canonical simulator re-run (GPU atomics reorder between runs) and A' is the canonical
simulator with the initial positions nudged by 1e-7, roughly one f32 rounding unit at these
coordinates. Below the first, the port is inside the simulator's own reproducibility. Below the
second, it is inside what a single rounding difference in the initial condition would have done
anyway. Above the second, the difference is REAL.

    .venv/Scripts/python.exe runs/material-variants/webgpu-port-of-the-interactive-simulation/verify/score.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sim.physics as phys                          # noqa: E402

TARGET_MS = 1000.0 / 60.0


def traj_rmse(a, b):
    """Registered metric traj_rmse: mean over frames and particles of |x_a - x_b|.
    See spec/registry/metrics.json -- it is a mean absolute distance, not an RMS."""
    return float(np.mean(np.linalg.norm(a - b, axis=-1)))


def per_frame(a, b):
    return np.mean(np.linalg.norm(a - b, axis=-1), axis=1)


def budget_60fps(rows, xk="n", yk="frame_ms"):
    """Largest particle count whose frame still fits in 16.67 ms, by linear interpolation of the
    measured cost curve. Returns None if even the smallest measured scene misses the budget."""
    pts = sorted([(r[xk], r[yk]) for r in rows if r.get(yk) is not None])
    if not pts or pts[0][1] > TARGET_MS:
        return None
    best = pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 <= TARGET_MS:
            best = x1
        elif y0 <= TARGET_MS < y1:
            return float(x0 + (x1 - x0) * (TARGET_MS - y0) / (y1 - y0))
    return float(best)


def main():
    bench = json.loads((HERE / "out" / "bench.json").read_text())
    job = json.loads((HERE / "job.json").read_text())
    taichi = json.loads((HERE / "baseline_taichi.json").read_text())
    js = json.loads((HERE / "baseline_js.json").read_text())

    ics = {ic["name"]: ic for ic in job["ics"]}
    out = {
        "physics_version": phys.VERSION,
        "physics_version_of_params": json.loads(
            (RUN / "web" / "params.js").read_text().split("var MPM_PARAMS = ", 1)[1]
            .rsplit(";", 2)[0])["physics_version"],
        "device": bench["device"],
        "user_agent": bench["user_agent"],
        "timestamp_quantum_ns": bench["timestamp_probe"]["apparent_quantum_ns"],
        "substeps_per_frame": bench["substeps_per_frame"],
        "n_grid": phys.core.n_grid,
        "scenes": {k: v["desc"] for k, v in ics.items()},
        "webgpu_errors": bench.get("webgpu_errors", []),
    }

    # ---------------------------------------------------------------- accuracy
    acc = {}
    for name, ic in ics.items():
        A = np.load(HERE / f"gt_{name}_A.npy")
        B = np.load(HERE / f"gt_{name}_B.npy")
        Pp = np.load(HERE / f"gt_{name}_P.npy")
        times = np.load(HERE / f"gt_{name}_t.npy")
        band = {"self_noise": traj_rmse(B, A), "perturbed_ic_1e-7": traj_rmse(Pp, A)}
        rows = []
        for v in job["variants"]:
            f = HERE / "out" / f"traj_{name}_{v['name']}.f32"
            t = np.fromfile(f, dtype=np.float32).reshape(ic["n_frames"], ic["n"], 2)
            r = {
                "variant": v["name"], "atomics": v["atomics"], "kM": v["kM"], "kV": v["kV"],
                "traj_rmse": traj_rmse(t, A),
                "final_frame_dist": float(np.mean(np.linalg.norm(t[-1] - A[-1], axis=-1))),
                "vs_self_noise": traj_rmse(t, A) / band["self_noise"],
                "vs_perturbed_ic": traj_rmse(t, A) / band["perturbed_ic_1e-7"],
                "finite": bool(np.isfinite(t).all()),
                "gt_final_width": phys.spread_width(A[-1]),
                "final_width": phys.spread_width(t[-1]),
                "gt_final_height": phys.pile_height(A[-1]),
                "final_height": phys.pile_height(t[-1]),
                # a fixed-point accumulator saturates SILENTLY: mass in units of 2^kM per particle
                # mass overflows u32 at 2^(32-kM) particle masses on one node
                "mass_saturates_at_pm": 2.0 ** (32 - v["kM"]),
                "mom_saturates_at_pm": 2.0 ** (31 - v["kV"]),
                "per_frame": per_frame(t, A).tolist(),
            }
            rows.append(r)
        acc[name] = {
            "n": ic["n"], "T": ic["T"], "n_frames": ic["n_frames"], "desc": ic["desc"],
            "band": band, "times": times.tolist(), "variants": rows,
            "per_frame_self_noise": per_frame(B, A).tolist(),
            "per_frame_perturbed_ic": per_frame(Pp, A).tolist(),
        }
    out["accuracy"] = acc

    # ---------------------------------------------------------------- fixed-point headroom
    out["headroom"] = [r for r in bench["rollouts"] if "max_node_mass_pm" in r]
    out["density_probe"] = bench["density_probe"]

    # ---------------------------------------------------------------- performance
    out["dispatch_floor"] = bench["dispatch_floor"]
    out["dispatch_floor_us"] = float(np.median([r["ns_per_dispatch"] for r in bench["dispatch_floor"]]) / 1000)
    out["webgpu_scaling"] = bench["scaling"]
    out["substep_sweep"] = bench["substep_sweep"]
    out["atomics_head_to_head"] = bench["atomics_head_to_head"]

    out["taichi_cuda"] = taichi
    out["javascript"] = js

    three = []
    for r in bench["scaling"]:
        three.append({"impl": "webgpu", "n": r["n"], "frame_ms": r["sustained_ms"],
                      "us_per_substep": r["us_per_substep_sustained"],
                      "particles_per_cell": r["particles_per_cell"]})
    for r in js["rows"]:
        three.append({"impl": "javascript", "n": r["n"], "frame_ms": r["frame_ms"],
                      "us_per_substep": r["us_per_substep"],
                      "particles_per_cell": r["particles_per_cell"]})
    for r in taichi["rows"]:
        if r.get("us_per_substep") is None:
            continue
        three.append({"impl": "taichi_cuda", "n": r["n"], "frame_ms": r["frame_ms"],
                      "us_per_substep": r["us_per_substep"],
                      "particles_per_cell": r["particles_per_cell"]})
    out["three_way"] = three

    out["particle_budget_60fps"] = {
        "webgpu": budget_60fps([{"n": r["n"], "frame_ms": r["sustained_ms"]} for r in bench["scaling"]]),
        "javascript": budget_60fps(js["rows"]),
        "taichi_cuda": budget_60fps([r for r in taichi["rows"] if r.get("us_per_substep")]),
        "target_frame_ms": TARGET_MS,
        "note": ("Largest particle count whose measured frame time still fits 16.67 ms, linearly "
                 "interpolated between the two bracketing measurements. None means the budget is "
                 "missed at every measured particle count, including the smallest."),
    }
    out["launch_floor_us"] = {
        "taichi_cuda_empty_kernel_from_python": taichi["empty_launch_us"],
        "webgpu_empty_dispatch_in_recorded_buffer": out["dispatch_floor_us"],
        "ratio": taichi["empty_launch_us"] / out["dispatch_floor_us"],
    }

    (RUN / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- console summary
    print("physics_version:", out["physics_version"], "| params stamped:", out["physics_version_of_params"])
    for name, a in acc.items():
        print(f"\n=== {name}: {a['desc']}")
        print("    band: self-noise %.3g   1-ULP IC nudge %.3g" % (a["band"]["self_noise"], a["band"]["perturbed_ic_1e-7"]))
        for r in a["variants"]:
            print("    %-10s traj_rmse %.4g   = %7.1fx self-noise, %7.1fx IC-nudge" %
                  (r["variant"], r["traj_rmse"], r["vs_self_noise"], r["vs_perturbed_ic"]))
    print("\nlaunch floor: taichi %.1f us  vs  webgpu dispatch %.2f us  (%.0fx)" %
          (out["launch_floor_us"]["taichi_cuda_empty_kernel_from_python"],
           out["launch_floor_us"]["webgpu_empty_dispatch_in_recorded_buffer"],
           out["launch_floor_us"]["ratio"]))
    print("particle budget at 60 fps:", json.dumps(
        {k: (round(v) if isinstance(v, float) else v) for k, v in out["particle_budget_60fps"].items()
         if k in ("webgpu", "javascript", "taichi_cuda")}))
    print("wrote", RUN / "metrics.json")


if __name__ == "__main__":
    main()
