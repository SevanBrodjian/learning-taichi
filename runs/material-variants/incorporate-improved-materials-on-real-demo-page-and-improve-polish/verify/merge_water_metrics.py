"""Fold the water rework's measurements into metrics.json.

Kept separate from the original run's numbers rather than overwriting them, because they were taken
with a DIFFERENT INSTRUMENT and mixing two instruments in one table is how a fake regression gets
published. The original `render_gpu_ms` block came from a sustained loop that summed device time
over N frames and divided; this block is the median of individually timed single draws. The
sustained loop reads higher (the GPU is not sitting at boost between submits), so the two are only
comparable within themselves. What IS comparable, and is the number that matters, is the difference
BETWEEN `before` and `after` inside this block: same instrument, same run, same scene, same frame.

    .venv/Scripts/python.exe runs/.../verify/merge_water_metrics.py
"""
import json
import pathlib

RUN = pathlib.Path(__file__).resolve().parents[1]
OUT = RUN / "verify" / "out"


def main():
    m = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    by_res = {}
    for res in (480, 720, 1080):
        d = json.loads((OUT / ("water_c%d_%d.json" % (res, res))).read_text(encoding="utf-8"))
        c = d["cost_ms"]
        by_res[str(res)] = {
            "pixels": res * res,
            "mvp_gpu_ms": c["mvp"]["ms"],
            "before_gpu_ms": c["before"]["ms"],
            "after_gpu_ms": c["after"]["ms"],
            "chain_gpu_ms_direct": c["chain_ms_direct"],
            "chain_gpu_ms_slope": c["chain_ms"],
            "passes_after": c["after"]["passes"],
            "passes_before": c["before"]["passes"],
            "amplification": c["amp"],
            "n": d["n"],
            "mean_abs_motion": d["mean_abs_motion"],
            "errors": len(d["errors"]),
        }
    ship = json.loads((OUT / "water_shipped_600.json").read_text(encoding="utf-8"))

    m["water_reconstruction"] = {
        "treatment": ship["treatment"],
        "instrument": ("WebGPU timestamp-query across the whole blob draw (splat pass -> reconstruction "
                       "-> resolve), median of individually timed draws, 80 draws each. Chromium "
                       "quantises timestamp results; the browser was launched with "
                       "--disable-dawn-features=timestamp_quantization and the residual granularity "
                       "was still ~16-33 us, which is why every number is cross-checked against an "
                       "amplified slope (the reconstruction run K times in one timed region)."),
        "recon_divisor": 2,
        "by_res": by_res,
        "chain_fit": {
            "fixed_ms": 0.020,
            "per_megapixel_ms": 0.0246,
            "note": ("Least-squares through the 480 and 1080 direct measurements. The fixed term is "
                     "twelve render passes' worth of attachment setup and is the DOMINANT term at "
                     "demo resolutions; the per-pixel term is what a bigger canvas buys you. The cost "
                     "does move with resolution -- x1.9 from 230k to 1.17M pixels -- so this is a GPU "
                     "reading and not a clock reading, but it is sub-linear because of that fixed "
                     "term."),
        },
        "foam_gate": {
            "peak_near_white_px_after": 237,
            "peak_near_white_px_before": 109,
            "scene": "pool + one rubber ball, 600x600, 90 frames",
            "note": ("Whitewater pixels (max channel > 190, saturation < 45) at the splash peak. This "
                     "is the check that the motion-gated foam term is actually FIRING: a gate that "
                     "never opens is indistinguishable from a gate that was never written, and "
                     "'the shading compiled' is precisely the mistake this rework exists to fix."),
        },
    }
    b = m["frame_budget"]
    b["render_gpu_ms_1024_water_rework"] = round(
        b["render_gpu_ms_1024"] + by_res["1080"]["chain_gpu_ms_direct"], 6)
    b["total_gpu_ms_water_rework"] = round(
        b["solver_gpu_ms"] + b["render_gpu_ms_1024_water_rework"], 6)
    b["water_rework_note"] = (
        "The reconstruction's marginal cost, measured with this run's instrument at 1080^2, added to "
        "the original run's 1024^2 figure. The two instruments disagree on the ABSOLUTE draw cost "
        "(0.147 ms sustained vs 0.100 ms isolated for the same build); only the delta is carried "
        "across, because only the delta was measured against a matched control.")

    (RUN / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    for r, v in by_res.items():
        print("%5s px %8d  before %.4f  after %.4f  chain %.4f ms (%d passes)"
              % (r, v["pixels"], v["before_gpu_ms"], v["after_gpu_ms"],
                 v["chain_gpu_ms_direct"], v["passes_after"]))
    print("frame budget: %.2f -> %.2f ms of %.2f" % (
        b["total_gpu_ms"], b["total_gpu_ms_water_rework"], b["budget_60fps_ms"]))


if __name__ == "__main__":
    main()
