"""Figures. Two questions, and the third figure is the one that answers them together.

    cost_vs_width.png      what the seam costs on WebGPU, with the real-time lines drawn on it and
                           the analytic solver it replaces marked as the baseline
    unroll_cliff.png       the same curve against the identical shader with a runtime loop bound,
                           which is the control that identifies the cliff
    capacity_vs_cost.png   accuracy and cost on ONE width axis, with the real-time cutoffs shaded --
                           the figure the whole task reduces to
    traj_error.png         per-material trajectory error against the two references that make it
                           mean something (the oracle floor and the IC-nudge band)

    .venv/Scripts/python.exe runs/.../fig/make_figs.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE.parent
BG = "#0a0e14"
FG = "#dfe6ee"
DIM = "#8fa3bf"
GRID = "#222c3c"
COLS = {"fluid": "#4db6ff", "elastic": "#ff9d5c", "snow": "#e6ecff", "sand": "#ffd24d"}


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("#0f141c")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#2a3446")
    ax.tick_params(colors=DIM, labelsize=9)
    ax.set_xlabel(xlabel, color=FG, fontsize=10)
    ax.set_ylabel(ylabel, color=FG, fontsize=10)
    if title:
        ax.set_title(title, color=FG, fontsize=12, pad=8)


def newfig(w, h):
    fig = plt.figure(figsize=(w, h), dpi=115)
    fig.patch.set_facecolor(BG)
    return fig


def load_bench():
    return json.loads((RUN / "verify" / "out" / "bench_cost.json").read_text())


def series(ws, pred, key="us_per_substep_full"):
    r = sorted([x for x in ws if pred(x)], key=lambda x: x["hidden"])
    return np.array([x["hidden"] for x in r]), np.array([x[key] for x in r])


def fig_cost(b):
    ws = b["width_sweep"]
    an = [x for x in ws if x["mode"] == "analytic"][0]
    full_budget = b["budget"]["us_per_substep_full_gpu"]
    qtr = b["budget"]["us_per_substep_quarter_gpu"]
    fig = newfig(11.5, 6.2)
    ax = fig.add_subplot(111)
    style(ax, "hidden width of the shared network",
          "microseconds per substep  (whole solver, n = 8192, one RTX 4090)",
          "What one latent-conditioned constitutive net costs on WebGPU")
    ax.axhspan(0, qtr, color="#7ee787", alpha=0.07)
    ax.axhline(qtr, color="#7ee787", lw=1.6, ls="--")
    ax.axhline(full_budget, color="#ffd24d", lw=1.6, ls="--")
    ax.text(7.4, qtr + 1.2, "60 fps at a QUARTER of this GPU", color="#7ee787", fontsize=9.5)
    ax.text(7.4, full_budget + 1.2, "60 fps at the WHOLE GPU", color="#ffd24d", fontsize=9.5)
    ax.axhline(an["us_per_substep_full"], color="#ff8f8f", lw=1.8)
    ax.annotate("the ANALYTIC four-material solver the network replaces sits at "
                f"{an['us_per_substep_full']:.1f} us --\n"
                "essentially ON the quarter-GPU line. Widths 8 and 16 are therefore FREE;\n"
                "everything above that line is what the extra capacity actually costs.",
                xy=(26, an["us_per_substep_full"]), xytext=(20, 41),
                color="#ff8f8f", fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#ff8f8f", lw=1.2))
    for pred, lab, c, mk in (
            (lambda x: x["mode"] == "nn" and not x["f16"] and x["weights"] == "uniform"
             and not x["variant"].startswith("dyn"), "f32, weights in a uniform buffer", "#6fd3ee", "o"),
            (lambda x: x["f16"], "f16 weights", "#c792ea", "s"),
            (lambda x: x["weights"] == "storage", "f32, weights in a storage buffer", "#ff9d5c", "^")):
        h, y = series(ws, pred)
        if h.size:
            ax.plot(h, y, mk + "-", color=c, lw=1.8, ms=5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 96, 128, 192, 256])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(7, 300)
    ax.set_ylim(0, 68)
    lg = ax.legend(facecolor="#0f141c", edgecolor="#2a3446", labelcolor=FG, fontsize=9,
                   loc="upper left", bbox_to_anchor=(0.0, 0.99))
    lg.get_frame().set_alpha(0.95)
    fig.tight_layout()
    fig.savefig(HERE / "cost_vs_width.png", facecolor=BG)
    plt.close(fig)


def fig_cliff(b):
    ws = b["width_sweep"]
    fig = newfig(11.0, 5.6)
    ax = fig.add_subplot(111)
    style(ax, "hidden width", "microseconds per substep  (G2P kernel alone, where the seam lives)",
          "The width cliff is the compiler giving up on unrolling, not the memory system")
    h1, y1 = series(ws, lambda x: x["mode"] == "nn" and not x["f16"] and x["weights"] == "uniform"
                    and not x["variant"].startswith("dyn"), "us_per_substep_g2p")
    h2, y2 = series(ws, lambda x: x["variant"].startswith("dyn"), "us_per_substep_g2p")
    an = [x for x in ws if x["mode"] == "analytic"][0]["us_per_substep_g2p"]
    ax.plot(h1, y1, "o-", color="#6fd3ee", lw=2.0, ms=5,
            label="loop bound is a literal -> the compiler can unroll")
    ax.plot(h2, y2, "s--", color="#ffd24d", lw=2.0, ms=5,
            label="identical shader, loop bound read from a uniform -> it cannot")
    ax.axhline(an, color="#ff8f8f", lw=1.6, label="the analytic constitutive model (SVD + Drucker-Prager)")
    # bracket the cliff
    cliff = None
    for i in range(1, len(h1)):
        if y1[i] > 1.8 * y1[i - 1]:
            cliff = (h1[i - 1], h1[i]); break
    if cliff:
        ax.axvspan(cliff[0], cliff[1], color="#ff8f8f", alpha=0.13)
        r0 = y1[list(h1).index(cliff[0])]
        r1 = y1[list(h1).index(cliff[1])]
        ax.text(8.6, max(y2) * 0.97,
                f"cost jumps {r1 / r0:.1f}x between width {cliff[0]} and {cliff[1]}\n"
                f"for 1.05x the arithmetic -- and lands\n"
                f"exactly on the curve of the shader\n"
                f"that CANNOT be unrolled.",
                color="#ff8f8f", fontsize=10.5, va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 88, 96, 128, 192, 256])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(7, 300)
    ax.legend(facecolor="#0f141c", edgecolor="#2a3446", labelcolor=FG, fontsize=9,
              loc="upper left", bbox_to_anchor=(0.26, 1.0))
    fig.tight_layout()
    fig.savefig(HERE / "unroll_cliff.png", facecolor=BG)
    plt.close(fig)


def fig_capacity(b, tr):
    """Accuracy and cost against ONE width axis. Everything this task asked reduces to whether the
    two curves have any overlap."""
    ws = b["width_sweep"]
    qtr = b["budget"]["us_per_substep_quarter_gpu"]
    full = b["budget"]["us_per_substep_full_gpu"]
    hs, cost = series(ws, lambda x: x["mode"] == "nn" and not x["f16"]
                      and x["weights"] == "uniform" and not x["variant"].startswith("dyn"))
    wmax_q = max([h for h, c in zip(hs, cost) if c <= qtr], default=0)
    wmax_f = max([h for h, c in zip(hs, cost) if c <= full], default=0)

    res = tr["results"]
    widths = sorted(int(k) for k in res)
    fig = newfig(11.5, 6.4)
    ax = fig.add_subplot(111)
    style(ax, "hidden width of the shared network",
          "held-out one-step error, in units of that material's own spread\n"
          "(1.0 = no better than predicting the mean)",
          "Capacity and cost on one axis: what the network can learn vs what it can afford")
    for m in ("fluid", "elastic", "snow", "sand"):
        y = [np.mean([res[str(w)]["held_out"][m][k] for k in ("tau00", "tau01", "tau11")])
             for w in widths]
        ax.plot(widths, y, "o-", color=COLS[m], lw=2.0, ms=5, label=m + "  (stress)")
    for m in ("snow", "sand"):
        y = [np.mean([res[str(w)]["held_out"][m][k] for k in ("dS00", "dS01", "dS11", "dJp")])
             for w in widths]
        ax.plot(widths, y, "s--", color=COLS[m], lw=1.4, ms=4, alpha=0.75,
                label=m + "  (plastic update)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(min(widths) * 0.85, max(widths) * 1.3)
    if wmax_q:
        ax.axvspan(min(widths) * 0.85, wmax_q, color="#7ee787", alpha=0.10)
        ax.axvline(wmax_q, color="#7ee787", lw=1.6, ls="--")
        ax.text(wmax_q, ax.get_ylim()[1] * 0.62, f" real time at a QUARTER GPU\n ends at width {wmax_q}",
                color="#7ee787", fontsize=9, va="top")
    if wmax_f:
        ax.axvline(wmax_f, color="#ffd24d", lw=1.6, ls="--")
        ax.text(wmax_f, ax.get_ylim()[1] * 0.20, f" whole GPU\n ends at {wmax_f}",
                color="#ffd24d", fontsize=9, va="top")
    ax.legend(facecolor="#0f141c", edgecolor="#2a3446", labelcolor=FG, fontsize=9,
              loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "capacity_vs_cost.png", facecolor=BG)
    plt.close(fig)
    return {"max_width_quarter_gpu": int(wmax_q), "max_width_full_gpu": int(wmax_f)}


def fig_traj(ev):
    rows = ev["trajectory"]
    scenes = sorted({r["scene"] for r in rows})
    fig = newfig(11.5, 4.6)
    for i, scn in enumerate(scenes):
        ax = fig.add_subplot(1, len(scenes), i + 1)
        style(ax, "", "traj_rmse vs canonical\n(mean per-particle distance)" if i == 0 else "", scn)
        sub = [r for r in rows if r["scene"] == scn]
        xs = np.arange(len(sub))
        ax.bar(xs - 0.22, [r["traj_rmse_learned"] for r in sub], 0.42,
               color=[COLS[r["material"]] for r in sub], label="learned")
        ax.bar(xs + 0.22, [r["traj_rmse_oracle"] for r in sub], 0.42,
               color="#44506a", label="oracle (analytic law, same scaffolding)")
        ax.plot(xs, [r["ic_nudge_band"] for r in sub], "_", color="#ff8f8f", ms=22, mew=2.2,
                label="1e-7 IC-nudge band")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([r["material"] for r in sub], rotation=35, ha="right", color=DIM, fontsize=9)
        if i == 0:
            ax.legend(facecolor="#0f141c", edgecolor="#2a3446", labelcolor=FG, fontsize=8, loc="upper left")
    fig.suptitle("Trajectory error, against the two references that make it mean something",
                 color=FG, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(HERE / "traj_error.png", facecolor=BG)
    plt.close(fig)


def main():
    b = load_bench()
    fig_cost(b)
    fig_cliff(b)
    summary = {}
    tp = RUN / "train" / "train_stats.json"
    if tp.exists():
        tr = json.loads(tp.read_text())
        summary.update(fig_capacity(b, tr))
    for p in sorted((RUN / "eval").glob("eval_h*.json")):
        ev = json.loads(p.read_text())
        if ev.get("trajectory"):
            fig_traj(ev)
            break
    (HERE / "fig_summary.json").write_text(json.dumps(summary, indent=2))
    print("figures written to", HERE)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
