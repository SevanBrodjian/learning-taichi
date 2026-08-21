"""Collect verify/ into metrics.json, then build the bespoke task page from it.

Everything the page shows is inlined from these numbers as a JS literal, not embedded as a picture of
a number (spec/style_task_page.md: prefer drawing from data over embedding images). The only images
the page carries are the ones whose subject IS an image -- the viewport screenshots -- and the videos,
which are referenced by absolute /api/data/ URL.

    .venv/Scripts/python.exe runs/.../build_run.py
"""
import json
import pathlib

RUN = pathlib.Path(__file__).resolve().parent
V = RUN / "verify"
MEDIA = "/api/data/learning-taichi/runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/"


def main():
    score = json.loads((V / "score.json").read_text())
    layout = json.loads((V / "layout.json").read_text())
    bench_new = json.loads((V / "out" / "bench.json").read_text())
    bench_old = json.loads((V / "out" / "bench_mvp_render.json").read_text())
    caps = json.loads((V / "capture_summary.json").read_text())
    job = json.loads((V / "job.json").read_text())

    def rmap(b):
        return {"%d_%s" % (r["res"], r["view"]): r["gpu_ms"] for r in b["render"]}

    m = {
        "physics_version": score["physics_version"],
        "device": score["device"],
        "user_agent": score["user_agent"],
        "canonical_reference": job["canonical"],
        "agreement": score["rollouts"],
        "buoyancy": score["buoyancy"],
        "ordering_pass": score["ordering_pass"],
        "solver": score["solver"],
        "render_gpu_ms": {"after": rmap(bench_new), "before": rmap(bench_old)},
        "render_rows": bench_new["render"],
        "pile": score["pile"],
        "substep_table": score["substep_table"],
        "layout": layout,
        "captures": caps,
        "shared_dt": bench_new["shared_dt"],
        "substeps_per_frame": bench_new["substeps_per_frame"],
    }
    # the one frame-budget statement the page makes, assembled here so the page never does arithmetic
    solver16 = [r for r in m["solver"] if r["n"] == 16384][0]
    m["frame_budget"] = {
        "n": 16384,
        "solver_gpu_ms": solver16["gpu_ms"],
        "render_gpu_ms_1024": m["render_gpu_ms"]["after"]["1024_blob"],
        "render_gpu_ms_1024_before": m["render_gpu_ms"]["before"]["1024_blob"],
        "total_gpu_ms": solver16["gpu_ms"] + m["render_gpu_ms"]["after"]["1024_blob"],
        "budget_60fps_ms": 1000.0 / 60.0,
    }
    (RUN / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print("wrote metrics.json")
    for k in ("ordering_pass", "frame_budget"):
        print(k, json.dumps(m[k], indent=1))
    return m


if __name__ == "__main__":
    main()
