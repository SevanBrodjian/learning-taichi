"""Render the evidence as MOTION, with canonical ground truth beside the browser in every frame.

Every claim in this task is about how a material MOVES, so the artifact is a trajectory, not a pair
of final frames. Three videos:

  material_vs_canonical.mp4   4 rows (fluid / elastic / snow / sand) x 2 columns
                              (canonical sim.physics | WebGPU in the browser), same initial
                              condition, same dt, same substep count, playing the same physical
                              time. The measured angle of repose is DRAWN on both panels of every
                              row, because the claim "sand slumps and snow does not" is a claim
                              about that angle and a reader should not have to infer it.

  mixed4_vs_canonical.mp4     the four-material scene on ONE grid, canonical `simulate_multi`
                              beside the browser, particles coloured by canonical material colour.

  demo_capture.mp4            the shipped page itself, captured from the browser (written by
                              capture.py, not here).

Plus two stills: repose_bars.png (the identity claim as numbers) and cost_curve.png (frame time vs
particle count against the 60 fps budget).

    .venv/Scripts/python.exe runs/.../verify/render.py
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
PANEL = "#0e131b"
FG = "#dfe6ee"
MUTED = "#7f8ea3"
ACCENT = "#6fd3ee"
FLOOR = 3.0 / 128.0
MATS = ["fluid", "elastic", "snow", "sand"]


def ffmpeg():
    for c in ("ffmpeg", str(ROOT / ".venv" / "Scripts" / "ffmpeg.exe")):
        try:
            subprocess.run([c, "-version"], capture_output=True, check=True)
            return c
        except Exception:
            pass
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#1d2734")
    ax.tick_params(colors=MUTED, labelsize=7)


def draw_floor(ax, x0, x1):
    ax.axhspan(0, FLOOR, color="#101822", zorder=0)
    ax.plot([x0, x1], [FLOOR, FLOOR], color="#2b4658", lw=1.0, zorder=1)


def repose_overlay(ax, snap, col, C):
    """Draw the free surface and the fitted flank whose slope IS the reported repose angle, so the
    number on the panel is visibly the thing that was measured."""
    cx, hh = C.surface_profile(snap)
    if cx.size < 6:
        return 0.0
    ax.plot(cx, hh + FLOOR, color=col, lw=1.1, alpha=0.9, zorder=6)
    apex = int(np.argmax(hh))
    for sl in (slice(0, apex + 1), slice(apex, len(cx))):
        if len(cx[sl]) >= 3:
            k, b = np.polyfit(cx[sl], hh[sl], 1)
            xs = np.array([cx[sl][0], cx[sl][-1]])
            ax.plot(xs, k * xs + b + FLOOR, color="#ffffff", lw=1.4, ls="--", alpha=0.85, zorder=7)
    return float(C.repose_angle(snap))


def main():
    import sim.physics as phys                      # noqa: E402
    from sim.physics import core as C               # noqa: E402

    job = json.loads((HERE / "job.json").read_text())
    score = json.loads((HERE / "out" / "score.json").read_text())
    base = np.load(HERE / "base.npz", allow_pickle=False)
    ics = {i["name"]: i for i in job["ics"]}
    by_scene = {r["scene"]: r for r in score["scenes"]}
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg()

    def load_web(name):
        ic = ics[name]
        return np.fromfile(HERE / "out" / ("traj_%s.f32" % name),
                           dtype=np.float32).reshape(ic["n_frames"], ic["n"], 2)

    # =============================================================== 1. four materials, both sides
    NF = job["frames"]
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 5.0), facecolor=BG)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.815, bottom=0.10, hspace=0.26, wspace=0.10)
    fig.suptitle("angle-of-repose heap: an over-steep 60° pile released from rest\n"
                 "top row canonical sim.physics (ground truth)   ·   bottom row the same step in "
                 "WebGPU, in a browser   ·   identical seed, dt and substep count",
                 color=FG, fontsize=11.5, y=0.975)
    scat, texts = [], []
    for c, m in enumerate(MATS):
        name = "heap_" + m
        gt = base[name + "_base"]
        web = load_web(name)
        col = C.MAT[m]["color"]
        for r, (lab, traj) in enumerate((("canonical", gt), ("WebGPU", web))):
            ax = axes[r][c]
            style(ax)
            ax.set_xlim(0.08, 0.92)
            ax.set_ylim(0.0, 0.36)
            ax.set_aspect("equal")
            draw_floor(ax, 0.08, 0.92)
            s = ax.scatter(traj[0][:, 0], traj[0][:, 1], s=2.0, c=col, linewidths=0, alpha=0.92, zorder=4)
            scat.append((s, traj))
            ax.set_title("%s — %s" % (m.upper(), lab), color=col if r else FG, fontsize=9.4, pad=3)
            t = ax.text(0.015, 0.93, "", transform=ax.transAxes, color=FG, fontsize=8.4,
                        family="monospace", va="top")
            texts.append((t, traj, ax, col))
            if c == 0:
                ax.set_ylabel("y  (domain lengths)", color=MUTED, fontsize=8)
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("x", color=MUTED, fontsize=8)
            else:
                ax.set_xticklabels([])
        rr = by_scene[name]
        axes[1][c].text(0.985, 0.93,
                        "traj_rmse %.1e\nnoise band %.1e" % (rr["traj_rmse_web_vs_canonical"],
                                                             rr["self_noise_nudge"]),
                        transform=axes[1][c].transAxes, color=ACCENT, fontsize=7.4,
                        family="monospace", va="top", ha="right")

    overlays = [[] for _ in range(8)]

    def frame(f):
        for i, (s, traj) in enumerate(scat):
            s.set_offsets(traj[f])
        for i, (t, traj, ax, col) in enumerate(texts):
            for o in overlays[i]:
                o.remove()
            overlays[i] = []
            if f >= NF - 1:
                nb = len(ax.lines)
                ang = repose_overlay(ax, traj[f], col, C)
                overlays[i] = ax.lines[nb:]
                t.set_text("t=%.2fs   repose %.1f deg" % ((f + 1) * job["T"] / NF, ang))
            else:
                t.set_text("t=%.2fs" % ((f + 1) * job["T"] / NF))

    w = FFMpegWriter(fps=24, bitrate=4200, codec="libx264",
                     # libx264 + yuv420p needs EVEN pixel dimensions and matplotlib will happily
                     # hand it an odd one; the scale filter rounds instead of failing at the end
                     # of a long render.
                     extra_args=["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                 "-pix_fmt", "yuv420p", "-preset", "slow"])
    dst = RUN / "material_vs_canonical.mp4"
    with w.saving(fig, str(dst), dpi=118):
        for f in range(NF):
            frame(f)
            w.grab_frame(facecolor=BG)
        for _ in range(36):                          # hold the settled frame with the fit drawn
            w.grab_frame(facecolor=BG)
    plt.close(fig)
    print("wrote", dst)

    # =============================================================== 2. the mixed four-material scene
    gt = base["mixed4_base"]
    web = load_web("mixed4")
    mats = base["mixed4_mat"]
    cols = np.array([C.MAT[m]["color"] for m in MATS])[mats]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), facecolor=BG)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.845, bottom=0.115, wspace=0.06)
    rr = by_scene["mixed4"]
    fig.suptitle("all four materials on ONE grid, shared dt = %.0e (%d substeps/frame)\n"
                 "canonical simulate_multi  vs  WebGPU   ·   traj_rmse %.2e against a self-noise band of %.2e"
                 % (score["shared_dt"], score["substeps_per_frame_shared"],
                    rr["traj_rmse_web_vs_canonical"], rr["self_noise_nudge"]),
                 color=FG, fontsize=10.5, y=0.975)
    ss = []
    for c, (lab, traj) in enumerate((("canonical (ground truth)", gt), ("WebGPU (browser)", web))):
        ax = axes[c]
        style(ax)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 0.60)
        ax.set_aspect("equal")
        draw_floor(ax, 0, 1)
        ss.append(ax.scatter(traj[0][:, 0], traj[0][:, 1], s=3.0, c=cols, linewidths=0, alpha=0.95, zorder=4))
        ax.set_title(lab, color=FG, fontsize=9.5, pad=4)
        ax.set_xlabel("x  (domain lengths)", color=MUTED, fontsize=8)
        if c:
            ax.set_yticklabels([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=C.MAT[m]["color"], label=m, markersize=5)
               for m in MATS]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.15,
                   labelcolor=FG, facecolor=PANEL, edgecolor="#1d2734")
    clock = axes[1].text(0.985, 0.965, "", transform=axes[1].transAxes, color=ACCENT, fontsize=9,
                         family="monospace", va="top", ha="right")
    dst = RUN / "mixed4_vs_canonical.mp4"
    with w.saving(fig, str(dst), dpi=118):
        for f in range(NF):
            for i, traj in enumerate((gt, web)):
                ss[i].set_offsets(traj[f])
            clock.set_text("t = %.2f s" % ((f + 1) * job["T"] / NF))
            w.grab_frame(facecolor=BG)
        for _ in range(24):
            w.grab_frame(facecolor=BG)
    plt.close(fig)
    print("wrote", dst)

    # =============================================================== 3. the identity claim, as numbers
    fig, ax = plt.subplots(figsize=(8.2, 4.0), facecolor=BG)
    style(ax)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.16)
    xs = np.arange(4)
    gtv = [by_scene["heap_" + m]["shape_canonical"]["repose_angle"] for m in MATS]
    wbv = [by_scene["heap_" + m]["shape_web"]["repose_angle"] for m in MATS]
    ax.bar(xs - 0.19, gtv, 0.36, color="#4a6b80", label="canonical sim.physics")
    ax.bar(xs + 0.19, wbv, 0.36, color=[C.MAT[m]["color"] for m in MATS], label="WebGPU (browser)")
    for i in range(4):
        ax.text(xs[i] - 0.19, gtv[i] + 1.0, "%.1f" % gtv[i], ha="center", color=MUTED, fontsize=8)
        ax.text(xs[i] + 0.19, wbv[i] + 1.0, "%.1f" % wbv[i], ha="center", color=FG, fontsize=8)
    ax.axhline(60, color="#3a4a5a", ls=":", lw=1)
    ax.text(3.45, 60.8, "seeded slope 60°", color=MUTED, fontsize=7.5, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([m.upper() for m in MATS], color=FG, fontsize=9)
    ax.set_ylabel("angle of repose (degrees)", color=FG, fontsize=9)
    ax.set_ylim(0, 68)
    ax.set_title("each material is recognisably itself, and the browser agrees with canonical\n"
                 "settled slope of an over-steep 60° heap, %d particles, each at its OWN canonical dt"
                 % ics["heap_fluid"]["n"], color=FG, fontsize=10, pad=8)
    # proxy handles, because the WebGPU bars carry FOUR colours and matplotlib would put the
    # first one (fluid blue) in the legend as if that were the series colour
    proxies = [plt.Rectangle((0, 0), 1, 1, color="#4a6b80"),
               plt.Rectangle((0, 0), 1, 1, color="#b9c6d4")]
    ax.legend(proxies, ["canonical sim.physics", "WebGPU (browser, material-coloured)"],
              fontsize=8, framealpha=0.15, labelcolor=FG, facecolor=PANEL, edgecolor="#1d2734",
              loc="center right")
    dst = RUN / "repose_bars.png"
    fig.savefig(dst, facecolor=BG, dpi=150)
    plt.close(fig)
    print("wrote", dst)

    # =============================================================== 4. the cost curve
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2), facecolor=BG)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.16, wspace=0.26)
    style(ax); style(ax2)
    rt = score["realtime"]
    ns = [r["n"] for r in rt]
    ms = [r["sustained_ms"] for r in rt]
    ax.plot(ns, ms, "-o", color=ACCENT, lw=1.8, ms=5, label="measured (sustained wall clock)")
    ax.axhline(16.67, color="#ff7a7a", ls="--", lw=1.2)
    ax.text(ns[0], 17.4, "60 fps budget = 16.67 ms", color="#ff7a7a", fontsize=8)
    ax.set_ylim(0, 19)
    ax.set_xlabel("particles", color=FG, fontsize=9)
    ax.set_ylabel("compute ms per real-time frame", color=FG, fontsize=9)
    ax.set_title("four materials present → shared dt 5e-5 → %d substeps/frame"
                 % score["substeps_per_frame_shared"], color=FG, fontsize=9.5, pad=6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=FG, facecolor=PANEL, edgecolor="#1d2734")

    ph = score["pile_headroom"]
    pn = [r["n"] for r in ph]
    pm = [r["max_node_mass_pm"] for r in ph]
    ax2.plot(pn, pm, "-o", color="#ffd24d", lw=1.8, ms=5, label="worst node, deliberately piled")
    ax2.axhline(ph[0]["mass_saturates_at_pm"], color="#ff7a7a", ls="--", lw=1.2)
    ax2.text(pn[0], ph[0]["mass_saturates_at_pm"] * 0.86,
             "2^24 fixed point wraps SILENTLY at %d particle masses" % ph[0]["mass_saturates_at_pm"],
             color="#ff7a7a", fontsize=7.6)
    ax2.set_ylim(0, ph[0]["mass_saturates_at_pm"] * 1.15)
    ax2.set_xlabel("particles", color=FG, fontsize=9)
    ax2.set_ylabel("heaviest grid node (particle masses)", color=FG, fontsize=9)
    ax2.set_title("fixed-point headroom under deliberate piling", color=FG, fontsize=9.5, pad=6)
    ax2.legend(fontsize=8, framealpha=0.15, labelcolor=FG, facecolor=PANEL, edgecolor="#1d2734")
    dst = RUN / "cost_and_headroom.png"
    fig.savefig(dst, facecolor=BG, dpi=150)
    plt.close(fig)
    print("wrote", dst)


if __name__ == "__main__":
    main()
