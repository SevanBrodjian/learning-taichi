"""Figures for the learned-grid-update profile.

Every chart carries the analytic grid update as the baseline (the thing the network replaces) and the
real-time budget as a line, drawn both raw and derated to a quarter of this GPU.
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1]
BG, FG, MUTED, GRIDC = "#0a0e14", "#dfe6ee", "#7f8ea3", "#1c2430"
ACCENT = "#6fd3ee"
WCOL = {8: "#a3d977", 16: "#6fd3ee", 32: "#b48ead", 64: "#ff9d5c"}
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#2a3340", "grid.color": GRIDC, "font.size": 11,
    "legend.facecolor": "#121822", "legend.edgecolor": "#2a3340", "legend.labelcolor": FG,
})


def style(ax):
    ax.grid(True, which="both", lw=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#2a3340")


def main():
    m = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    sweep = m["sweep"]
    budget = m["budget_us_per_substep"]
    budget_q = m["budget_us_per_substep_quarter_gpu"]
    SPF = m["substeps_per_frame"]
    ns = sorted({r["n"] for r in sweep})
    hs = sorted({r["hidden"] for r in sweep})

    def series(key, h=None):
        return [next(r[key] for r in sweep if r["n"] == n and (h is None or r["hidden"] == h))
                for n in ns]

    # ---------------- 1. headline: whole-solver cost against the real-time budget ----------------
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=150)
    ax.plot(ns, series("full_us_analytic", hs[0]), "o-", color=ACCENT, lw=2.6, ms=7,
            label="analytic grid update (the baseline it replaces)", zorder=5)
    for h in hs:
        ax.plot(ns, series("full_us_nn", h), "s--", color=WCOL[h], lw=1.9, ms=5.5,
                label=f"learned grid update, width {h}")
    ax.axhline(budget, color="#7ee787", lw=1.8, ls="-")
    ax.axhline(budget_q, color="#ffd24d", lw=1.8, ls="-")
    ax.text(ns[-1] * 0.98, budget * 1.05,
            f"60 fps budget: {budget:.0f} us/substep  ({SPF} substeps/frame)",
            color="#7ee787", fontsize=10, va="bottom", ha="right")
    ax.text(ns[-1] * 0.98, budget_q * 1.05,
            f"same budget at a quarter of this GPU: {budget_q:.1f} us/substep",
            color="#ffd24d", fontsize=10, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(4.0, 130)
    ax.set_xlabel("particles")
    ax.set_ylabel("whole solver, microseconds per substep  (P2G + grid + G2P)")
    ax.set_title("Cost of replacing the grid update with a network, against the real-time budget\n"
                 "RTX 4090, Chromium, 128x128 grid, canonical water (dt = 5e-5)",
                 color=FG, fontsize=12.5, pad=12)
    style(ax)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.97)
    fig.tight_layout()
    fig.savefig(RUN / "cost_vs_budget.png")
    plt.close(fig)
    print("wrote cost_vs_budget.png")

    # ---------------- 2. the width cliff ----------------
    ws_all = m.get("width_scan", [])
    if ws_all:
        passes = sorted({r.get("pass", 0) for r in ws_all})
        p0 = [r for r in ws_all if r.get("pass", 0) == passes[0]]
        hh = [r["hidden"] for r in p0]
        fl = [r["flops_per_cell"] for r in p0]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=150)
        ref = p0[hh.index(16)]["grid_us"] / fl[hh.index(16)]
        a1.plot(hh, [ref * f for f in fl], "--", color=MUTED, lw=1.6,
                label="if cost were proportional to arithmetic")
        for pi, col in zip(passes, [ACCENT, "#ff9d5c"]):
            d = [r for r in ws_all if r.get("pass", 0) == pi]
            a1.plot([r["hidden"] for r in d], [r["grid_us"] for r in d], "o-", color=col,
                    lw=2.2, ms=5.5, label=f"measured, pass {pi + 1}")
        a1.axhline(budget, color="#7ee787", lw=1.5)
        a1.axhline(budget_q, color="#ffd24d", lw=1.5)
        a1.text(hh[0] * 1.05, budget * 1.1, "60 fps budget for the WHOLE substep",
                color="#7ee787", fontsize=9)
        a1.text(hh[0] * 1.05, budget_q * 1.1, "...at a quarter GPU", color="#ffd24d", fontsize=9)
        a1.set_xscale("log", base=2); a1.set_yscale("log")
        a1.set_xticks(hh); a1.set_xticklabels([str(x) for x in hh], fontsize=8.5)
        a1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        a1.set_xlabel("hidden width"); a1.set_ylabel("grid-update kernel, us per substep")
        a1.set_title("Cost is not proportional to the network's arithmetic", color=FG, fontsize=11.5)
        style(a1); a1.legend(fontsize=9, loc="upper left")

        for pi, col in zip(passes, [ACCENT, "#ff9d5c"]):
            d = [r for r in ws_all if r.get("pass", 0) == pi]
            a2.plot([r["hidden"] for r in d], [r["achieved_gflops"] for r in d], "o-", color=col,
                    lw=2.2, ms=5.5, label=f"pass {pi + 1}")
        a2.axvspan(22, 44, color="#b48ead", alpha=0.13)
        a2.text(30, 2000, "widths 24-40:\nthird of the throughput\nof 20 or of 48",
                color="#c9a6c4", fontsize=9.5, ha="center")
        a2.set_xscale("log", base=2)
        a2.set_xticks(hh); a2.set_xticklabels([str(x) for x in hh], fontsize=8.5)
        a2.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        a2.set_ylim(0, 4600)
        a2.set_xlabel("hidden width"); a2.set_ylabel("achieved GFLOP/s")
        a2.set_title("Throughput drops in a band, then recovers", color=FG, fontsize=11.5)
        style(a2); a2.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        fig.savefig(RUN / "width_cliff.png")
        plt.close(fig)
        print("wrote width_cliff.png")

    # ---------------- 3. why the particle-count curve is flat, and what would un-flatten it ------
    comp = m.get("compaction", [])
    if comp:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.4), dpi=150)
        ana = [r["grid_us"]["analytic"] for r in sweep if r["hidden"] == hs[0]]
        for h in hs:
            d = sorted([r for r in comp if r["hidden"] == h], key=lambda r: r["n"])
            xs = [r["n"] for r in d]
            a1.plot(xs, [r["grid_us_dense"] for r in d], "s-", color=WCOL[h], lw=2.2, ms=5.5,
                    label=f"width {h}, every cell")
            a1.plot(xs, [r["grid_us_compacted"] for r in d], "o--", color=WCOL[h], lw=1.5, ms=4,
                    alpha=0.85, label=f"width {h}, occupied cells only")
        a1.plot(ns, ana, "^-", color=ACCENT, lw=2.6, ms=8, label="analytic grid update")
        a1.axhline(budget, color="#7ee787", lw=1.4)
        a1.axhline(budget_q, color="#ffd24d", lw=1.4)
        a1.text(ns[0], budget * 1.15, "60 fps budget", color="#7ee787", fontsize=9)
        a1.text(ns[0], budget_q * 1.15, "at a quarter GPU", color="#ffd24d", fontsize=9)
        a1.set_xscale("log"); a1.set_yscale("log")
        a1.set_xticks(ns); a1.set_xticklabels([str(n) for n in ns], fontsize=9)
        a1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        a1.set_xlabel("particles"); a1.set_ylabel("grid-update kernel, us per substep")
        a1.set_title("Flat in particle count, and compaction does not tilt it", color=FG, fontsize=11.5)
        style(a1)
        a1.legend(fontsize=7.6, ncol=2, loc="center right", framealpha=0.95)

        for h in hs:
            d = sorted([r for r in comp if r["hidden"] == h], key=lambda r: r["n"])
            a2.plot([r["workgroup_reduction"] for r in d], [r["speedup"] for r in d], "o",
                    color=WCOL[h], ms=9, label=f"width {h}")
        xr = np.array([1, 40])
        a2.plot(xr, xr, "--", color=MUTED, lw=1.6, label="if time scaled with work issued")
        a2.axhline(1.0, color="#7ee787", lw=1.5)
        a2.text(1.15, 1.06, "no speedup at all", color="#7ee787", fontsize=10)
        a2.set_xscale("log"); a2.set_yscale("log")
        a2.set_xlim(1.2, 45); a2.set_ylim(0.7, 45)
        a2.set_xticks([1.5, 3, 5.6, 17, 36])
        a2.set_xticklabels(["1.5x", "3x", "5.6x", "17x", "36x"], fontsize=9)
        a2.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        a2.set_yticks([1, 2, 5, 10, 20, 40]); a2.set_yticklabels(["1x", "2x", "5x", "10x", "20x", "40x"])
        a2.set_xlabel("reduction in workgroups dispatched (dense 256 -> occupied cells only)")
        a2.set_ylabel("measured speedup")
        a2.set_title("Issuing 36x less work costs the same time:\nthe kernel is latency-bound, not throughput-bound",
                     color=FG, fontsize=11.5)
        style(a2); a2.legend(fontsize=9, loc="upper left")
        fig.tight_layout()
        fig.savefig(RUN / "flat_curve.png")
        plt.close(fig)
        print("wrote flat_curve.png")

    # ---------------- 4. accuracy: what the cost buys ----------------
    surv = m["survival"]
    acc = m["accuracy"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=150)
    for tag, col, lab in (("point", "#6fd3ee", "cell-wise loss"),
                          ("deriv", "#ff9d5c", "+ derivative loss")):
        d = sorted([r for r in surv if r["net"].startswith(tag)], key=lambda r: r["hidden"])
        a1.plot([r["hidden"] for r in d], [r["frames_60fps_tracked"] for r in d], "o-",
                color=col, lw=2.2, ms=7, label=lab)
    a1.axhline(1.0, color="#7ee787", lw=1.5, ls="--")
    a1.text(8, 1.06, "one 60 fps frame", color="#7ee787", fontsize=9.5)
    a1.set_xscale("log", base=2); a1.set_xticks([8, 16, 32, 64]); a1.set_xticklabels(["8", "16", "32", "64"])
    a1.set_xlabel("hidden width")
    a1.set_ylabel("60 fps frames tracking ground truth\n(mean particle distance under 0.05)")
    a1.set_title("How long a learned rollout stays the same fluid", color=FG, fontsize=11.5)
    style(a1); a1.legend(fontsize=9.5)

    hs2 = sorted(int(k) for k in acc["stage2_derivative"])
    gb = [acc["stage2_derivative"][str(h)]["grad_rel_before"] for h in hs2]
    ga = [acc["stage2_derivative"][str(h)]["grad_rel_after"] for h in hs2]
    pw = [acc["stage1_pointwise"][str(h)]["node_v_rel_massw"] for h in hs2]
    x = np.arange(len(hs2))
    a2.bar(x - 0.27, pw, 0.26, color="#6fd3ee", label="node velocity, relative error")
    a2.bar(x, gb, 0.26, color="#b48ead", label="its spatial derivative, cell-wise loss")
    a2.bar(x + 0.27, ga, 0.26, color="#ff9d5c", label="its spatial derivative, + derivative loss")
    a2.set_xticks(x); a2.set_xticklabels([str(h) for h in hs2])
    a2.set_xlabel("hidden width"); a2.set_ylabel("relative error against the analytic kernel")
    a2.set_title("G2P reads the derivative, and the derivative is what stays wrong",
                 color=FG, fontsize=11.5)
    a2.set_ylim(0, 1.0)
    style(a2); a2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RUN / "accuracy.png")
    plt.close(fig)
    print("wrote accuracy.png")


if __name__ == "__main__":
    main()
