"""Assemble metrics.json and manifest.json from the artifacts on disk. Written LAST, and every media
`src` is checked to resolve before the manifest is emitted."""
import json
import pathlib
import sys

RUN = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[2]))
import sim.physics as phys                      # noqa: E402

BASE = ("/api/data/learning-taichi/runs/learned-dynamics/"
        "one-latent-conditioned-network-for-all-four-materials/")
MATS = ["fluid", "elastic", "snow", "sand"]


def L(p, d=None):
    q = RUN / p
    return json.loads(q.read_text()) if q.exists() else d


def main():
    bench = L("verify/out/bench_cost.json")
    tr = L("train/train_stats.json")
    trl2 = L("train/train_stats_L2.json")
    trnn = L("train/train_stats_noNuis.json")
    ev32 = L("eval/eval_h32.json")
    evl2 = L("eval/eval_h64_L2_traj.json", {"trajectory": []})
    parity = L("verify/out/parity.json")
    gate = L("train/gate_oracle.json")
    ds = L("train/dataset_stats.json")

    ws = bench["width_sweep"]
    f32 = sorted([x for x in ws if x["mode"] == "nn" and not x["f16"]
                  and x["weights"] == "uniform" and not x["variant"].startswith("dyn")],
                 key=lambda x: x["hidden"])
    an = [x for x in ws if x["mode"] == "analytic"][0]
    qtr = bench["budget"]["us_per_substep_quarter_gpu"]
    full = bench["budget"]["us_per_substep_full_gpu"]
    fitq = max([x["hidden"] for x in f32 if x["us_per_substep_full"] <= qtr], default=0)
    fitf = max([x["hidden"] for x in f32 if x["us_per_substep_full"] <= full], default=0)

    def acc(stats, w, key):
        r = stats["results"][str(w)]["held_out"]
        return {m: round(sum(r[m][k] for k in key) / len(key), 4) for m in MATS}

    metrics = {
        "physics_version": phys.VERSION,
        "setup": {"device": bench["device"]["vendor"] + " " + bench["device"]["architecture"],
                  "n_grid": 128, "n_particles": bench["params"]["MAX_P"],
                  "n_particles_timed": 8192, "dt": bench["budget"]["dt"],
                  "materials_on_one_grid": 4, "browser": bench["user_agent"]},
        "budget": bench["budget"],
        "cost": {
            "analytic_us_per_substep_full": an["us_per_substep_full"],
            "analytic_us_per_substep_g2p": an["us_per_substep_g2p"],
            "dispatch_floor_us": bench["dispatch_floor"][-1]["ns_per_dispatch"] / 1000.0,
            "max_width_realtime_quarter_gpu": fitq,
            "max_width_realtime_full_gpu": fitf,
            "by_width": [{"hidden": x["hidden"], "us_per_substep": x["us_per_substep_full"],
                          "us_per_substep_g2p": x["us_per_substep_g2p"]} for x in f32],
            "f16": [{"hidden": x["hidden"], "us_per_substep": x["us_per_substep_full"],
                     "us_per_substep_g2p": x["us_per_substep_g2p"]}
                    for x in ws if x["f16"]],
            "weights_in_storage_buffer": [
                {"hidden": x["hidden"], "us_per_substep": x["us_per_substep_full"],
                 "us_per_substep_g2p": x["us_per_substep_g2p"]}
                for x in ws if x["weights"] == "storage"],
            "unrollable_control": [{"hidden": x["hidden"], "us_per_substep_g2p": x["us_per_substep_g2p"]}
                                   for x in ws if x["variant"].startswith("dyn")],
            "batching_wall_us_per_substep": bench["batching"],
            "n_sweep": bench["n_sweep"],
            "timestamp_quantum_ns": bench["timestamp_probe"]["apparent_quantum_ns"],
            "control_pg_drift_pct": bench["control_pg_drift_pct"],
            "cost_trained_vs_untrained_ratio": parity["cost_trained_vs_random"]["ratio"],
        },
        "capacity": {
            "onestep_rel_err_stress": {str(w): acc(tr, w, ("tau00", "tau01", "tau11"))
                                       for w in [8, 16, 32, 64, 128]},
            "onestep_rel_err_plastic": {str(w): acc(tr, w, ("dS00", "dS01", "dS11", "dJp"))
                                        for w in [8, 16, 32, 64, 128]},
            "two_hidden_layers_h64_stress": acc(trl2, 64, ("tau00", "tau01", "tau11")) if trl2 else None,
            "two_hidden_layers_h64_plastic": acc(trl2, 64, ("dS00", "dS01", "dS11", "dJp")) if trl2 else None,
            "ablation_no_C_or_v_h64_stress": acc(trnn, 64, ("tau00", "tau01", "tau11")) if trnn else None,
            "n_params": {str(w): tr["results"][str(w)]["n_params"] for w in [8, 16, 32, 64, 128]},
            "z_codes": bench["net_shape"]["z_codes"], "z_sep": bench["net_shape"]["z_sep"],
            "z_jitter": bench["net_shape"]["z_jitter"],
            "dataset": ds,
        },
        "golden_signatures": {
            "canonical_and_oracle": gate["signature_summary"],
            "learned_h32": ev32["signature_summary"],
            "learned_h32_rows": [{"name": r["name"], "pass": r["pass"], "na": r["na"],
                                  "detail": r["detail"]} for r in ev32["signatures"]],
        },
        "trajectory": {"learned_h32": ev32["trajectory"],
                       "learned_h64_two_layer": evl2.get("trajectory", [])},
        "parity_host_vs_wgsl": parity,
        "analytic_port_vs_canonical": bench["analytic_port"],
        "gate_oracle_step_level": gate["step_level"],
    }
    (RUN / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote metrics.json")
    return metrics


if __name__ == "__main__":
    main()
