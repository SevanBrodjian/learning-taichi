"""Turn verify/out/bench.json into metrics.json, and check the WGSL MLP against the host weights.

Two jobs:
  1. VERIFY. The browser posted, for each network, the exact per-cell inputs it fed the shader
     (recovered through the 'null' grid kernel, which writes the raw node momentum and mass) and the
     velocities the shader produced. This re-evaluates the same weights in numpy and reports the
     largest disagreement. Nothing downstream is believed until this is at float32 rounding.
  2. SCORE. Assemble the sweep into the numbers the page and the manifest quote, including the
     real-time verdict at full and quarter GPU.
"""
import json
import pathlib
import sys

import numpy as np

RUN = pathlib.Path(__file__).resolve().parents[1]
V = RUN / "verify"
OUT = V / "out"
TRAIN = RUN / "train"

FRAME_BUDGET_MS = 1000.0 / 60.0
DERATE = 4.0            # the user's standing assumption: at most a quarter of this GPU in practice


def load_net(fn, h):
    z = np.load(TRAIN / fn)
    return [z[f"{h}_W1"], z[f"{h}_b1"], z[f"{h}_W2"], z[f"{h}_b2"], z[f"{h}_W3"], z[f"{h}_b3"]]


def net_apply(net, X):
    W1, b1, W2, b2, W3, b3 = net
    a = np.maximum(X @ W1 + b1, 0.0)
    a = np.maximum(a @ W2 + b2, 0.0)
    return a @ W3 + b3


NG, BOUND = 128, 3
i_ = np.repeat(np.arange(NG), NG)
j_ = np.tile(np.arange(NG), NG)
WALLS = np.stack([(i_ < BOUND), (i_ > NG - BOUND), (j_ < BOUND), (j_ > NG - BOUND)], 1).astype(np.float32)


def verify_inference(b):
    rows = []
    for e in b["inference_check"]:
        key = e["net"]
        tag = "weights.npz" if key.startswith("point") else "weights_grid.npz"
        h = e["hidden"]
        net = load_net(tag, h)
        gin = np.fromfile(OUT / f"infer_in_{key}.f32", dtype=np.float32).reshape(-1, 4)
        gnn = np.fromfile(OUT / f"infer_nn_{key}.f32", dtype=np.float32).reshape(-1, 4)
        ipm = 1.0 / e["pMass"]
        X = np.zeros((gin.shape[0], 8), np.float32)
        X[:, 0] = gin[:, 2]                       # gv.z = node mass in particle masses
        X[:, 1] = gin[:, 0] * ipm                 # gv.xy = raw node momentum ('null' kernel)
        X[:, 2] = gin[:, 1] * ipm
        X[:, 3:7] = WALLS
        X[:, 7] = 0.0                             # canonical water: fric = 0
        host = net_apply(net, X)
        d = np.abs(host - gnn[:, :2])
        occ = X[:, 0] > 0
        scale = max(float(np.abs(host).max()), 1e-9)
        rows.append({
            "net": key, "hidden": h, "cells": int(gin.shape[0]), "occupied": int(occ.sum()),
            "max_abs_diff": float(d.max()), "max_abs_diff_occupied": float(d[occ].max()),
            "max_rel_diff": float(d.max() / scale),
            "mean_abs_diff_occupied": float(d[occ].mean()),
            "host_output_scale": scale,
            "dense_vs_sparse_max_abs_diff": e["dense_vs_sparse_max_abs_diff"],
        })
        print(f"  {key:9s} h={h:3d}  max|wgsl-host| = {d.max():.3e} "
              f"(rel {d.max()/scale:.2e})  dense-vs-sparse {e['dense_vs_sparse_max_abs_diff']:.1e}")
    return rows


def main():
    b = json.loads((OUT / "bench.json").read_text(encoding="utf-8"))
    ref = json.loads((V / "ref_stats.json").read_text(encoding="utf-8"))
    tr1 = json.loads((TRAIN / "train_stats.json").read_text(encoding="utf-8"))
    tr2 = json.loads((TRAIN / "train_grid_stats.json").read_text(encoding="utf-8"))
    SPF = b["params"]["substeps_per_frame"]
    budget_us = FRAME_BUDGET_MS * 1000.0 / SPF

    print("--- inference verification (WGSL vs host weights) ---")
    infer = verify_inference(b)

    floor = float(np.median([r["ns_per_dispatch"] for r in b["dispatch_floor"]])) / 1000.0

    sweep = []
    for r in b["sweep"]:
        row = {"n": r["n"], "hidden": r["hidden"],
               "particles_per_cell": r["particles_per_cell"],
               "flops_per_cell": r["flops_per_cell"], "net_floats": r["net_floats"],
               "grid_us": r["grid_us"], "null_pg_us": r["null_pg_us"]}
        for k, v in r["phases"].items():
            row[f"full_us_{k}"] = v["pgG_us_per_substep"]
            row[f"pg_us_{k}"] = v["pg_us_per_substep"]
            row[f"g2p_us_{k}"] = v["pgG_us_per_substep"] - v["pg_us_per_substep"]
        for k in ("analytic", "nn", "nnsparse"):
            row[f"frame_ms_{k}"] = row[f"full_us_{k}"] * SPF / 1000.0
            row[f"frame_ms_{k}_quarter"] = row[f"frame_ms_{k}"] * DERATE
        # achieved fraction of peak, using the grid kernel's own arithmetic
        cells_per_s = 16384 / max(r["grid_us"]["nn"], 1e-12) * 1e6
        row["nn_gflops_dense"] = cells_per_s * r["flops_per_cell"] / 1e9
        sweep.append(row)

    def pick(n, h):
        for r in sweep:
            if r["n"] == n and r["hidden"] == h:
                return r
        return None

    # ---- the verdict: largest width whose FULL solver frame fits the budget, per particle count ----
    verdict = {"budget_us_per_substep": budget_us, "substeps_per_frame": SPF,
               "frame_budget_ms": FRAME_BUDGET_MS, "derate": DERATE, "rows": []}
    widths = sorted({r["hidden"] for r in sweep})
    for n in sorted({r["n"] for r in sweep}):
        row = {"n": n}
        for kind in ("nn", "nnsparse"):
            for label, lim in (("full_gpu", budget_us), ("quarter_gpu", budget_us / DERATE)):
                ok = [h for h in widths if pick(n, h)[f"full_us_{kind}"] <= lim]
                row[f"max_width_{kind}_{label}"] = max(ok) if ok else None
        row["analytic_full_us"] = pick(n, widths[0])["full_us_analytic"]
        row["analytic_fits_full_gpu"] = row["analytic_full_us"] <= budget_us
        row["analytic_fits_quarter_gpu"] = row["analytic_full_us"] <= budget_us / DERATE
        verdict["rows"].append(row)

    occ = b["occupancy"]
    surv = b["survival"]
    # The sharpest accuracy number on the page. Gravity's entire contribution to one grid update is
    # dt*g of velocity. Every trained network's own fitting error is far larger than that, so the term
    # the whole simulation is driven by sits below the network's noise floor.
    dt = b["params"]["dt"]
    g_step = dt * b["params"]["gravity"]
    grav = {"gravity_velocity_per_substep": g_step,
            "by_width": {k: {"node_v_mae_massw": v["node_v_mae_massw"],
                             "times_gravity_step": v["node_v_mae_massw"] / g_step}
                         for k, v in tr1["widths"].items()}}
    acc = {"stage1_pointwise": {k: {kk: v[kk] for kk in
                                    ("node_v_mae_massw", "node_v_mae", "node_v_rel_massw", "params",
                                     "flops_per_cell")}
                                for k, v in tr1["widths"].items()},
           "stage2_derivative": {k: {"grad_rel_before": v["before"]["grad_rel"],
                                     "grad_rel_after": v["after"]["grad_rel"],
                                     "node_v_mae_massw_before": v["before"]["node_v_mae_massw"],
                                     "node_v_mae_massw_after": v["after"]["node_v_mae_massw"],
                                     "grad_mag_true": v["after"]["grad_mag_true"]}
                                 for k, v in tr2["widths"].items()}}

    m = {
        "physics_version": b["params"]["physics_version"],
        "device": b["device"], "user_agent": b["user_agent"],
        "substeps_per_frame": SPF, "budget_us_per_substep": budget_us,
        "budget_us_per_substep_quarter_gpu": budget_us / DERATE,
        "dispatch_floor_us": floor,
        "analytic_port_check": b["analytic_check"],
        "canonical_self_noise": ref["self_noise"],
        "inference_verification": infer,
        "occupancy": occ,
        "timestamp_quantum_ns": b.get("timestamp_probe", {}).get("apparent_quantum_ns"),
        "width_scan": b.get("width_scan", []),
        "compaction": b.get("compaction", []),
        "sweep": sweep,
        "verdict": verdict,
        "survival": surv,
        "traj": b.get("traj", []),
        "accuracy": acc,
        "gravity_vs_error": grav,
        "webgpu_errors": b.get("webgpu_errors", []),
        "notes": {
            "hardware": "one RTX 4090 in Chromium; every timing is a claim about this device only",
            "timing": "GPU timestamp-query over a compute pass of %d substeps, median of %d"
                      % (b["sweep"][0]["timed_substeps"], 5),
            "grid_us": "the grid kernel's own cost, isolated by differencing against a 'null' grid "
                       "kernel with identical memory traffic and no physics",
            "derate": "quarter-GPU numbers are the measured value multiplied by 4",
        },
    }
    (RUN / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    print("\n--- headline ---")
    print(f"substeps/frame {SPF}   budget {budget_us:.1f} us/substep "
          f"({budget_us/DERATE:.1f} at a quarter GPU)   dispatch floor {floor:.2f} us")
    for r in sweep:
        if r["hidden"] in (16, 64):
            print(f"  n={r['n']:6d} h={r['hidden']:3d}  grid us: analytic {r['grid_us']['analytic']:7.2f} "
                  f" nn {r['grid_us']['nn']:8.2f}  nnsparse {r['grid_us']['nnsparse']:8.2f}   "
                  f"frame ms: analytic {r['frame_ms_analytic']:7.2f}  nn {r['frame_ms_nn']:9.2f}  "
                  f"nnsparse {r['frame_ms_nnsparse']:9.2f}")
    print("\nverdict rows:")
    for r in verdict["rows"]:
        print("  ", json.dumps(r))
    print("\nwrote", RUN / "metrics.json")


if __name__ == "__main__":
    sys.exit(main())
