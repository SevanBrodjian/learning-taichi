"""Step 4: render the evidence as MOTION, with ground truth beside the port in every frame.

The claim under test is about a trajectory, so the artifact has to be a trajectory: three panels
side by side, playing the same physical time, on the identical initial condition --

    canonical sim.physics   |   WebGPU fixed-point 2^20   |   WebGPU fixed-point 2^24

with the canonical outline ghosted on top of each WebGPU panel so the divergence is visible in the
frame rather than inferred from a number, and a running traj_rmse read against the simulator's own
noise band underneath.

    .venv/Scripts/python.exe runs/material-variants/webgpu-port-of-the-interactive-simulation/verify/render.py
"""
import json
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.animation import FFMpegWriter       # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[4]
RUN = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

BG = "#0a0e14"
FG = "#dfe6ee"
MUTED = "#7f8ea3"
ACCENT = "#6fd3ee"
GT_COL = "#8fb7d4"
PORT_COL = "#ff9d5c"
BAD_COL = "#ff7a7a"
FLOOR = 3.0 / 128.0


def load(name, variant, n_frames, n):
    f = HERE / "out" / f"traj_{name}_{variant}.f32"
    return np.fromfile(f, dtype=np.float32).reshape(n_frames, n, 2)


def per_frame(a, b):
    return np.mean(np.linalg.norm(a - b, axis=-1), axis=1)


def view_window(arrays, pad=0.05):
    """A single crop shared by every panel, sized to hold all of the material in all of the runs.
    Without it the disk occupies a tenth of a unit-square panel and a reader cannot see a divergence
    that the numbers say is there."""
    xs = np.concatenate([a[..., 0].ravel() for a in arrays])
    ys = np.concatenate([a[..., 1].ravel() for a in arrays])
    xs = xs[np.isfinite(xs)]; ys = ys[np.isfinite(ys)]
    x0, x1 = float(xs.min()) - pad, float(xs.max()) + pad
    y0, y1 = float(ys.min()) - pad, float(ys.max()) + pad
    # keep it square so the aspect is honest
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max(x1 - x0, y1 - y0) / 2
    return (max(0.0, cx - h), min(1.0, cx + h), max(0.0, cy - h), min(1.0, cy + h))


def panel_axes(ax, title, sub, win=None):
    w = win or (0, 1, 0, 1)
    ax.set_xlim(w[0], w[1]); ax.set_ylim(w[2], w[3]); ax.set_aspect("equal")
    ax.set_facecolor("#070a0f")
    for s in ax.spines.values():
        s.set_color("#1e2733")
    ax.set_xticks([]); ax.set_yticks([])
    ax.axhspan(0, FLOOR, color="#141b25", zorder=0)
    ax.set_title(title, color=FG, fontsize=11, pad=3)
    ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center", va="top",
            color=MUTED, fontsize=8.5)


def render_compare(scene, variants, out_mp4, fps=30):
    """One row of panels, all on the same initial condition and the same physical time."""
    m = json.loads((RUN / "metrics.json").read_text())
    a = m["accuracy"][scene]
    n, nf = a["n"], a["n_frames"]
    times = np.array(a["times"])
    A = np.load(HERE / f"gt_{scene}_A.npy")
    band = a["band"]

    series = [("canonical", A, None)]
    for v in variants:
        row = next(r for r in a["variants"] if r["variant"] == v)
        series.append((v, load(scene, v, nf, n), row))

    k = len(series)
    win = view_window([s[1] for s in series])
    fig = plt.figure(figsize=(4.0 * k, 5.6), facecolor=BG)
    gs = fig.add_gridspec(2, k, height_ratios=[2.5, 1.15], hspace=0.46, wspace=0.06,
                          left=0.075, right=0.985, top=0.895, bottom=0.095)
    axes, scats, ghosts = [], [], []
    for i, (name, arr, row) in enumerate(series):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            panel_axes(ax, "canonical  sim.physics", "the ground truth, forward sim, f32 on CUDA", win)
        else:
            r = row
            lab = ("fixed point $2^{%d}$" % r["kM"]) if r["atomics"] == "fixed" \
                else "exact f32 (CAS loop)"
            sub = ("%s\ntraj_rmse %.3g  =  %.0f$\\times$ self-noise"
                   % ("quanta per particle mass" if r["atomics"] == "fixed" else "no quantisation",
                      r["traj_rmse"], r["vs_self_noise"]))
            panel_axes(ax, "WebGPU  " + lab, sub, win)
        if i > 0:
            g = ax.scatter(A[0, :, 0], A[0, :, 1], s=7.0, c=GT_COL, alpha=0.45,
                           linewidths=0, zorder=2)
            ghosts.append(g)
        col = GT_COL if i == 0 else PORT_COL
        sc = ax.scatter(arr[0, :, 0], arr[0, :, 1], s=7.0, c=col, linewidths=0, zorder=3,
                        alpha=1.0 if i == 0 else 0.85)
        axes.append(ax); scats.append(sc)

    axd = fig.add_subplot(gs[1, :])
    axd.set_facecolor("#070a0f")
    axd.set_yscale("log")
    axd.set_xlim(0, float(times[-1]))
    axd.set_xlabel("simulated time  (s)", color=MUTED, fontsize=9)
    axd.set_ylabel("mean particle distance\nfrom canonical  (domain lengths)", color=MUTED, fontsize=8.5)
    axd.tick_params(colors=MUTED, labelsize=8)
    for s in axd.spines.values():
        s.set_color("#1e2733")
    axd.axhspan(1e-12, band["self_noise"], color="#243040", alpha=0.9, zorder=0)
    axd.axhspan(band["self_noise"], band["perturbed_ic_1e-7"], color="#1a2432", alpha=0.9, zorder=0)
    axd.axhline(band["self_noise"], color=ACCENT, lw=1.0, ls=":", zorder=1)
    axd.axhline(band["perturbed_ic_1e-7"], color=ACCENT, lw=1.0, ls="--", zorder=1)

    curves, lines = [], []
    lo = band["self_noise"]
    for i, (name, arr, row) in enumerate(series):
        if i == 0:
            continue
        d = per_frame(arr, A)
        lo = min(lo, float(np.min(d[d > 0])) if np.any(d > 0) else lo)
        col = PORT_COL if row["vs_perturbed_ic"] > 3 else ACCENT
        ln, = axd.plot([], [], color=col, lw=1.6,
                       label="$2^{%d}$" % row["kM"] if row["atomics"] == "fixed" else "exact f32")
        curves.append(d); lines.append(ln)
    dsn = per_frame(np.load(HERE / f"gt_{scene}_B.npy"), A)
    lnsn, = axd.plot([], [], color=MUTED, lw=1.2, ls=":", label="canonical vs itself")
    curves.append(dsn); lines.append(lnsn)

    hi = max(float(np.max(np.concatenate(curves))), band["perturbed_ic_1e-7"])
    axd.set_ylim(max(lo * 0.5, 1e-9), hi * 6)
    from matplotlib.patches import Patch
    handles, labels = axd.get_legend_handles_labels()
    handles.append(Patch(facecolor="#1a2432", edgecolor=ACCENT, ls="--"))
    labels.append("canonical's own noise band (re-run $\\to$ one-ULP IC nudge)")
    leg = axd.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.0, ncol=4)
    for t in leg.get_texts():
        t.set_color(MUTED)

    sup = fig.suptitle("", color=FG, fontsize=12)

    writer = FFMpegWriter(fps=fps, bitrate=3600,
                          metadata={"title": "webgpu vs canonical, " + scene})
    with writer.saving(fig, str(out_mp4), dpi=110):
        for f in range(nf):
            for i, (name, arr, row) in enumerate(series):
                scats[i].set_offsets(arr[f])
            for g in ghosts:
                g.set_offsets(A[f])
            for ln, d in zip(lines, curves):
                ln.set_data(times[:f + 1], d[:f + 1])
            sup.set_text("%s   |   t = %.2f s   |   %d particles, identical initial condition"
                         % (a["desc"], times[f], n))
            writer.grab_frame()
    plt.close(fig)
    print("wrote", out_mp4)


def render_final_frames(scene, variants, out_png):
    """A still of the settled/last state, for readers who will not press play."""
    m = json.loads((RUN / "metrics.json").read_text())
    a = m["accuracy"][scene]
    n, nf = a["n"], a["n_frames"]
    A = np.load(HERE / f"gt_{scene}_A.npy")
    k = len(variants) + 1
    arrs = [load(scene, v, nf, n)[-1] for v in variants] + [A[-1]]
    win = view_window(arrs, pad=0.03)
    fig, axs = plt.subplots(1, k, figsize=(3.3 * k, 4.3), facecolor=BG)
    panel_axes(axs[0], "canonical", "sim.physics forward", win)
    axs[0].scatter(A[-1, :, 0], A[-1, :, 1], s=6.0, c=GT_COL, linewidths=0)
    for i, v in enumerate(variants):
        r = next(x for x in a["variants"] if x["variant"] == v)
        arr = load(scene, v, nf, n)
        lab = ("fixed $2^{%d}$" % r["kM"]) if r["atomics"] == "fixed" else "exact f32"
        panel_axes(axs[i + 1], "WebGPU " + lab,
                   "traj_rmse %.3g\n%.0f$\\times$ self-noise" % (r["traj_rmse"], r["vs_self_noise"]), win)
        axs[i + 1].scatter(A[-1, :, 0], A[-1, :, 1], s=6.0, c=GT_COL, alpha=0.45, linewidths=0)
        axs[i + 1].scatter(arr[-1, :, 0], arr[-1, :, 1], s=6.0, c=PORT_COL, alpha=0.85, linewidths=0)
    fig.suptitle("%s\nfinal frame (t = %.2f s); canonical ghosted in blue under every WebGPU panel"
                 % (a["desc"], a["times"][-1]), color=FG, fontsize=11)
    fig.tight_layout(rect=[0, 0.09, 1, 0.88])
    fig.savefig(out_png, dpi=130, facecolor=BG)
    plt.close(fig)
    print("wrote", out_png)


def render_overflow(out_png):
    """The other failure mode of fixed point: the accumulator silently WRAPS."""
    rp = HERE / "out" / "range.json"
    if not rp.exists():
        print("no range.json; skipping overflow figure")
        return None
    R = json.loads(rp.read_text())
    rows = R["overflow"]
    n = rows[0]["n"]
    ppc = rows[0]["particles_per_cell"]
    # what the heaviest node actually reaches at this density, measured with the exact-f32 path so
    # the measurement itself cannot saturate
    occ = min(R["occupancy"], key=lambda o: abs(o["particles_per_cell"] - ppc))
    peak = occ["max_node_mass_pm"]
    R["max_node_mass_measured"] = peak

    ref = next((r for r in rows if r["atomics"] == "casf32"), rows[0])
    refx = np.fromfile(HERE / "out" / ref["positions_file"], dtype=np.float32).reshape(n, 2)
    for r in rows:
        x = np.fromfile(HERE / "out" / r["positions_file"], dtype=np.float32).reshape(n, 2)
        r["dist_from_exact_f32"] = float(np.mean(np.linalg.norm(x - refx, axis=-1)))
        r["overflows"] = bool(r["atomics"] == "fixed" and r["mass_ceiling_pm"] < peak)
    (HERE / "out" / "range.json").write_text(json.dumps(R, indent=2), encoding="utf-8")

    fig, axs = plt.subplots(1, len(rows), figsize=(3.3 * len(rows), 4.5), facecolor=BG)
    for ax, r in zip(axs, rows):
        x = np.fromfile(HERE / "out" / r["positions_file"], dtype=np.float32).reshape(n, 2)
        lab = ("fixed $2^{%d}$" % r["kM"]) if r["atomics"] == "fixed" else "exact f32"
        if r["atomics"] == "fixed":
            sub = "node ceiling %.0f pm%s\ndist from exact f32: %.3g" % (
                r["mass_ceiling_pm"], "  --  WRAPS" if r["overflows"] else "",
                r["dist_from_exact_f32"])
        else:
            sub = "no quantisation, no ceiling\n(the reference in this figure)"
        panel_axes(ax, lab, sub)
        ax.scatter(refx[:, 0], refx[:, 1], s=2.0, c=GT_COL, alpha=0.30, linewidths=0)
        ax.scatter(x[:, 0], x[:, 1], s=2.0, c=(BAD_COL if r["overflows"] else PORT_COL), linewidths=0)
    fig.suptitle("Overrunning the fixed-point RANGE on purpose: %d particles, %.0f per cell, "
                 "heaviest node %.0f particle masses.\nA saturating u32 does not raise an error "
                 "-- it wraps.  (exact-f32 result ghosted under each panel)"
                 % (n, ppc, peak), color=FG, fontsize=10.5)
    fig.tight_layout(rect=[0, 0.10, 1, 0.88])
    fig.savefig(out_png, dpi=130, facecolor=BG)
    plt.close(fig)
    print("wrote", out_png)
    for r in rows:
        print("   %-12s ceiling %7.0f pm  dist_from_exact %.4g  wraps=%s"
              % (r["tag"], r["mass_ceiling_pm"], r["dist_from_exact_f32"], r["overflows"]))
    return out_png


def main():
    render_compare("launch", ["fixed_k20", "fixed_k24"], RUN / "launch_compare.mp4")
    render_compare("drop", ["fixed_k16", "fixed_k24"], RUN / "drop_compare.mp4")
    render_final_frames("launch", ["fixed_k12", "fixed_k16", "fixed_k20", "fixed_k24", "casf32"],
                        RUN / "launch_final_frames.png")
    render_overflow(RUN / "fixed_point_overflow.png")


if __name__ == "__main__":
    main()
