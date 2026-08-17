"""Render the evidence: the port against canonical ground truth as motion, the timestep sweep as
motion, and the two figures the budget argument rests on.

    .venv/Scripts/python.exe runs/material-variants/interactive-simulation-of-one-material/verify/render.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
import numpy as np                               # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent

BG = "#0a0e14"
FG = "#dfe6ee"
MUT = "#7f8ea3"
GT = "#6fd3ee"
PORT = "#ff9d5c"
GRID = "#1c2430"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "#0d131b", "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": "#22303f", "grid.color": GRID, "font.size": 10,
})

M = json.loads((RUN / "metrics.json").read_text())
G = json.loads((HERE / "gpu_bench.json").read_text())
FLOOR = 3 / 128


def load(tag, pre=""):
    n, f = M["n_particles"], M["n_frames"]
    return np.fromfile(HERE / f"port_{pre}{tag}.f32", dtype=np.float32).reshape(f, n, 2)


def box(ax, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.78)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axhspan(0, FLOOR, color=GT, alpha=0.07)
    ax.axhline(FLOOR, color=GT, alpha=0.35, lw=0.9)
    ax.axvspan(0, FLOOR, color=GT, alpha=0.07); ax.axvspan(1 - FLOOR, 1, color=GT, alpha=0.07)
    ax.set_title(title, color=FG, fontsize=11, pad=6)
    ax.set_aspect("equal", adjustable="box")


def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[:, :, :3].copy()


# --------------------------------------------------------------------------- 1. GT vs port, motion
def render_overlay(scene="drop"):
    import imageio
    launch = scene == "launch"
    pre = "launch_" if launch else ""
    src = M["launch_scene"] if launch else M
    gt = np.load(HERE / ("launch_snapA.npy" if launch else "snapA.npy"))
    port = load("canonical", pre)
    pf = src["per_frame"]
    t = np.array(pf["times"])
    dP = np.array(pf["port_vs_canonical"])
    dS = np.array(pf["canonical_self_noise"])
    dI = np.array(pf["canonical_perturbed_ic"])
    nf = gt.shape[0]

    frames = []
    fig = plt.figure(figsize=(11.2, 4.4), dpi=110)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.22,
                          left=0.045, right=0.985, top=0.86, bottom=0.14)
    for k in range(nf):
        fig.clf()
        ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2])
        box(ax0, "canonical Taichi / CUDA (f32)")
        ax0.scatter(gt[k, :, 0], gt[k, :, 1], s=2.2, c=GT, alpha=0.75, lw=0)
        box(ax1, "JS port in the browser (f64 math)")
        ax1.scatter(port[k, :, 0], port[k, :, 1], s=2.2, c=PORT, alpha=0.75, lw=0)

        ax2.set_yscale("log")
        ax2.plot(t, dS, color="#c58cf0", lw=1.4, label="canonical vs itself (GPU atomics)")
        ax2.plot(t, dI, color="#5fd39a", lw=1.4, label="canonical vs IC nudged by 1e-7")
        ax2.plot(t, dP, color=PORT, lw=2.0, label="port vs canonical")
        ax2.axvline(t[k], color=FG, alpha=0.45, lw=1.0)
        ax2.set_xlim(t[0], t[-1]); ax2.set_ylim(1e-9, 1e-2 if launch else 3e-3)
        ax2.set_xlabel("simulated time (s)")
        ax2.set_ylabel("mean particle distance (domain lengths)")
        ax2.set_title("traj_rmse per frame, against canonical run A", color=FG, fontsize=11, pad=6)
        ax2.grid(alpha=0.35, lw=0.6)
        ax2.legend(loc="lower right", fontsize=8, facecolor="#0d131b", edgecolor="#22303f",
                   labelcolor=FG, framealpha=0.9)
        fig.suptitle("Same initial condition, two implementations. The port's divergence sits in the "
                     "same band as the reference's own run-to-run noise.",
                     color=MUT, fontsize=10.5, y=0.975)
        frames.append(fig_to_rgb(fig))
    plt.close(fig)
    name = "port_vs_canonical_launch.mp4" if launch else "port_vs_canonical.mp4"
    imageio.mimwrite(RUN / name, frames, fps=30, quality=8, macro_block_size=1)
    print("wrote", name, len(frames), "frames")


# --------------------------------------------------------------------------- 2. timestep sweep, motion
def render_dt_sweep(scene="drop"):
    import imageio
    launch = scene == "launch"
    pre = "launch_" if launch else ""
    src = M["launch_scene"] if launch else M
    gt = np.load(HERE / ("launch_snapA.npy" if launch else "snapA.npy"))
    runs = [("x1", load("dt1", pre), "$\\Delta t$ = 1e-4  (canonical, 167 substeps/frame)"),
            ("x2", load("dt2", pre), "$\\Delta t$ = 2e-4  (2.2x faster, 83 substeps/frame)"),
            ("x4", load("dt4", pre), "$\\Delta t$ = 4e-4  (4.4x faster, 42 substeps/frame)")]
    sweep = {f"x{e['mult']}": e for e in src["dt_sweep"]}
    nf = gt.shape[0]
    frames = []
    fig = plt.figure(figsize=(11.2, 4.4), dpi=110)
    gs = fig.add_gridspec(1, 3, wspace=0.10, left=0.02, right=0.98, top=0.80, bottom=0.10)
    for k in range(nf):
        fig.clf()
        for i, (tag, arr, title) in enumerate(runs):
            ax = fig.add_subplot(gs[i])
            box(ax, title)
            ax.scatter(gt[k, :, 0], gt[k, :, 1], s=2.4, c=GT, alpha=0.30, lw=0)
            p = np.nan_to_num(arr[k], nan=0.0, posinf=1.5, neginf=-0.5)
            ax.scatter(p[:, 0], p[:, 1], s=2.2, c=PORT, alpha=0.8, lw=0)
            e = sweep[tag]
            txt = "diverged (non-finite)" if not e["finite"] else "traj_rmse %.2e" % e["traj_rmse_vs_canonical"]
            ax.text(0.02, 0.955, txt, transform=ax.transAxes, fontsize=9.5,
                    color=PORT if e["finite"] else "#ff6b6b", va="top",
                    family="monospace")
        fig.suptitle("Raising the timestep to buy frame rate. Cyan is the canonical run in every panel; "
                     "orange is the cheaper timestep.", color=MUT, fontsize=10.5, y=0.95)
        frames.append(fig_to_rgb(fig))
    plt.close(fig)
    name = "dt_sweep_launch.mp4" if launch else "dt_sweep.mp4"
    imageio.mimwrite(RUN / name, frames, fps=30, quality=8, macro_block_size=1)
    print("wrote", name, len(frames), "frames")


# --------------------------------------------------------------------------- 3. the budget figure
JS = json.loads((RUN / "browser_bench.json").read_text())


def render_budget():
    ns = [r["n"] for r in JS["cpu_sweep"]]
    sp = [r["sparse"]["us_per_step"] for r in JS["cpu_sweep"]]
    dn = [r["dense"]["us_per_step"] for r in JS["cpu_sweep"]]
    gn = [r["n"] for r in G["per_n"]]
    gu = [r["us_per_step"] for r in G["per_n"]]
    budget = (1 / 60) / 167 * 1e6                       # us per substep that 60 fps real time allows

    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=140)
    ax.plot(ns, sp, "-o", color=PORT, ms=4, lw=2, label="browser JS, sparse grid (the port)")
    ax.plot(ns, dn, "-o", color="#c58cf0", ms=4, lw=1.6, label="browser JS, dense grid (Taichi's loop)")
    ax.plot(gn, gu, "-o", color=GT, ms=4, lw=2, label="canonical Taichi / CUDA (RTX, launch-bound)")
    ax.axhline(budget, color="#5fd39a", ls="--", lw=1.6)
    ax.text(9500, budget * 1.09, "60 fps at real time  (%.0f us / substep)" % budget,
            color="#5fd39a", fontsize=9, ha="right")
    # where the port crosses the budget
    xs = np.array(ns, float); ys = np.array(sp, float)
    cross = float(np.interp(budget, ys, xs))
    ax.plot([cross], [budget], "o", color=FG, ms=7, mec=BG)
    ax.annotate("%d particles" % round(cross), (cross, budget), textcoords="offset points",
                xytext=(8, -22), color=FG, fontsize=10, fontweight="bold")
    # where one JS thread stops beating the GPU
    gflat = float(np.mean(gu))
    xover = float(np.interp(gflat, ys, xs))
    ax.plot([xover], [gflat], "o", color=FG, ms=7, mec=BG)
    ax.annotate("one JS thread beats the RTX 4090\nbelow ~%d particles" % (round(xover / 100) * 100),
                xy=(xover, gflat), xytext=(9000, 52), color=FG, fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="-", color=FG, alpha=0.45, lw=1))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("particles"); ax.set_ylabel("microseconds per substep")
    ax.set_title("Cost of one MLS-MPM substep, elastic, 128x128 grid", color=FG, fontsize=11)
    ax.grid(alpha=0.35, lw=0.6, which="both")
    ax.legend(fontsize=8.5, facecolor="#0d131b", edgecolor="#22303f", labelcolor=FG, loc="upper left")
    fig.tight_layout()
    fig.savefig(RUN / "substep_budget.png")
    plt.close(fig)
    print("wrote substep_budget.png  (port crosses the 60fps budget at n=%.0f)" % cross)
    return cross


# --------------------------------------------------------------------------- 4. dt cost/accuracy
def render_dt_tradeoff():
    sw = [e for e in M["dt_sweep"]]
    mult = [e["mult"] for e in sw]
    speed = [e["speedup_vs_canonical"] for e in sw]
    rmse = [e["traj_rmse_vs_canonical"] for e in sw]
    floor = M["traj_rmse"]["canonical_self_noise"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    ok = [i for i, e in enumerate(sw) if e["finite"]]
    ax.plot([speed[i] for i in ok], [rmse[i] for i in ok], "-o", color=PORT, ms=6, lw=2)
    for i in ok:
        ax.annotate("  %gx dt" % mult[i], (speed[i], rmse[i]), color=FG, fontsize=9)
    bad = [i for i, e in enumerate(sw) if not e["finite"]]
    for i in bad:
        ax.axvline(speed[i], color="#ff6b6b", ls="--", lw=1.6)
        ax.text(speed[i] - 0.05, 4e-2, "%gx dt: diverged" % mult[i], color="#ff6b6b",
                rotation=90, ha="right", va="top", fontsize=9.5)
    ax.axhline(floor, color="#c58cf0", ls=":", lw=1.5)
    ax.text(4.35, floor * 1.30, "canonical simulator's own run-to-run noise (the floor)",
            color="#c58cf0", fontsize=9, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("speedup over canonical timestep (fewer substeps per frame)")
    ax.set_ylabel("traj_rmse vs canonical run")
    ax.set_title("The timestep is not a performance knob", color=FG, fontsize=11)
    ax.grid(alpha=0.35, lw=0.6)
    fig.tight_layout()
    fig.savefig(RUN / "dt_tradeoff.png")
    plt.close(fig)
    print("wrote dt_tradeoff.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "budget"):
        render_budget(); render_dt_tradeoff()
    if which in ("all", "video"):
        render_overlay(); render_dt_sweep()
