"""Render the learned-vs-canonical comparison clips and the figures.

Rule from CLAUDE.md: any comparison shows BOTH sides against each other, in the same medium as the
claim. The claim here is about motion, so the ground truth and the learned rollout are two panels of
one video, frame-synchronised, not two final stills.
"""
import json
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import imageio.v2 as imageio             # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1]
V = RUN / "verify"
OUT = V / "out"

BG = "#0a0e14"
FG = "#dfe6ee"
MUTED = "#7f8ea3"
ACCENT = "#6fd3ee"
WARM = "#ff9d5c"
PALETTE = ["#6fd3ee", "#ff9d5c", "#b48ead", "#a3d977", "#ffd24d"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#2a3340", "grid.color": "#1c2430", "font.size": 11,
})


def style(ax):
    ax.grid(True, which="both", lw=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#2a3340")


def load_traj(path, frames, n):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(frames, n, 2)


def panel(ax, pts, title, color, sub=None):
    ax.clear()
    ax.set_facecolor(BG)
    ax.scatter(pts[:, 0], pts[:, 1], s=1.6, c=color, linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=FG, fontsize=12, pad=6)
    if sub:
        ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center", va="top",
                color=MUTED, fontsize=9.5)
    for s in ax.spines.values():
        s.set_color("#2a3340")


def make_video(dst, series, frames, fps=20, suptitle=None):
    """series: list of (label, array(frames,n,2), color, sublabel-fn(frame))"""
    fig, axes = plt.subplots(1, len(series), figsize=(4.0 * len(series), 4.35), dpi=130)
    if len(series) == 1:
        axes = [axes]
    imgs = []
    for f in range(frames):
        for ax, (lab, arr, col, subf) in zip(axes, series):
            panel(ax, arr[f], lab, col, subf(f) if subf else None)
        if suptitle:
            fig.suptitle(suptitle, color=FG, fontsize=12.5, y=0.985)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95 if suptitle else 1.0))
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        # h264 with yuv420p needs even dimensions; an odd canvas silently breaks the ffmpeg pipe
        buf = buf[: buf.shape[0] // 2 * 2, : buf.shape[1] // 2 * 2]
        imgs.append(buf.copy())
    plt.close(fig)
    imageio.mimsave(dst, imgs, fps=fps, quality=8, macro_block_size=1)
    print("wrote", dst, imgs[0].shape)


def main():
    ref_stats = json.loads((V / "ref_stats.json").read_text())
    job = json.loads((V / "job.json").read_text())
    NF, N = ref_stats["frames"], ref_stats["n"]
    ref = load_traj(V / "ref_drop.f32", NF, N)
    an = load_traj(OUT / "traj_analytic.f32", NF, N)
    nets = {}
    for k in job["traj_nets"]:
        p = OUT / f"traj_{k}.f32"
        if p.exists():
            nets[k] = load_traj(p, NF, N)

    dt = ref_stats["dt"]
    spf = ref_stats["substeps_per_frame_render"]

    def sub_of(arr, r):
        def f(fr):
            d = np.linalg.norm(arr[fr] - r[fr], axis=1).mean()
            return f"t = {(fr + 1) * spf * dt:.2f} s   mean distance from truth {d:.3f}"
        return f

    # ---- 1. the headline clip: canonical vs the best learned grid update ----
    best = "deriv64" if "deriv64" in nets else list(nets)[0]
    make_video(RUN / "learned_vs_truth.mp4",
               [("canonical grid update (ground truth)", ref, ACCENT,
                 lambda fr: f"t = {(fr + 1) * spf * dt:.2f} s"),
                (f"learned grid update, width {job['widths'][-1]}", nets[best], WARM,
                 sub_of(nets[best], ref))],
               NF, fps=20,
               suptitle="Same seed, same P2G and G2P. Only the grid update differs.")

    # ---- 2. the analytic WGSL port against canonical: the baseline is sound ----
    make_video(RUN / "analytic_port_check.mp4",
               [("canonical Taichi (CUDA)", ref, ACCENT, lambda fr: f"t = {(fr + 1) * spf * dt:.2f} s"),
                ("analytic WGSL port (the baseline)", an, ACCENT, sub_of(an, ref))],
               NF, fps=20,
               suptitle="The analytic WebGPU baseline reproduces the canonical solver.")

    # ---- 3. all learned widths at once, final frames ----
    # The blown-up rollouts pile every particle onto the domain boundary, where G2P's position clamp
    # holds it. Drawn as bare dots that reads as an EMPTY panel rather than a degenerate one, so each
    # panel states how much of the material is stacked on the clamp.
    floor_y = 3.0 / 128.0

    def clamped_frac(f):
        eps = 1e-4
        on = ((f[:, 0] <= floor_y + eps) | (f[:, 0] >= 1 - floor_y - eps)
              | (f[:, 1] <= floor_y + eps) | (f[:, 1] >= 1 - floor_y - eps))
        return float(on.mean())

    keys = [k for k in ["point16", "point64", "deriv64"] if k in nets]
    labels = {"point16": "learned, width 16\n(cell-wise loss)",
              "point64": "learned, width 64\n(cell-wise loss)",
              "deriv64": "learned, width 64\n(+ derivative loss)"}
    fig, axes = plt.subplots(1, 2 + len(keys), figsize=(3.5 * (2 + len(keys)), 4.5), dpi=140)
    panel(axes[0], ref[-1], "canonical\n(ground truth)", ACCENT,
          f"{100 * clamped_frac(ref[-1]):.0f}% on the domain edge")
    panel(axes[1], an[-1], "analytic WGSL\n(the baseline)", ACCENT,
          f"{100 * clamped_frac(an[-1]):.0f}% on the domain edge")
    for i, k in enumerate(keys):
        d = np.linalg.norm(nets[k][-1] - ref[-1], axis=1).mean()
        panel(axes[2 + i], nets[k][-1], labels[k], WARM,
              f"distance from truth {d:.2f}\n{100 * clamped_frac(nets[k][-1]):.0f}% on the domain edge")
    fig.suptitle(f"Final frame at t = {NF * spf * dt:.2f} s. Two of the learned rollouts have thrown "
                 f"every particle onto the wall, where the position clamp holds it.",
                 color=FG, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(RUN / "final_frames.png")
    plt.close(fig)
    print("wrote", RUN / "final_frames.png")

    # ---- 3b. the thing the contact sheet shows: the learned fluid does not fall ----
    # Gravity enters the grid update as v.y -= dt*g, which is 4.9e-4 of velocity per substep. The
    # network's own mass-weighted velocity error is two orders of magnitude larger than that, so the
    # single most important physical term in the kernel sits below the fitting noise and the blob
    # simply hovers. This plot is that statement as a measurement.
    tr = json.loads((RUN / "metrics.json").read_text())
    dtg = tr["accuracy"]["stage1_pointwise"]
    g_step = 9.8 * dt
    ts = (np.arange(NF) + 1) * spf * dt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=145)
    a1.plot(ts, ref[:, :, 1].mean(1), "-", color=ACCENT, lw=2.8, label="canonical (ground truth)")
    a1.plot(ts, an[:, :, 1].mean(1), "--", color="#7ee787", lw=1.8, label="analytic WGSL baseline")
    for k, col in zip(keys, ["#b48ead", "#ffd24d", WARM]):
        a1.plot(ts, nets[k][:, :, 1].mean(1), "-", color=col, lw=2.0, label=labels[k].replace("\n", " "))
    a1.axhline(floor_y, color=MUTED, lw=1.2, ls=":")
    a1.text(0.02, floor_y + 0.012, "floor", color=MUTED, fontsize=9.5)
    a1.set_xlabel("simulated seconds"); a1.set_ylabel("mean particle height")
    a1.set_title("The learned fluid does not fall", color=FG, fontsize=12)
    a1.set_ylim(0, 0.62)
    a1.text(0.12, 0.585, "these two leave the frame upward", color=MUTED, fontsize=9.5)
    style(a1); a1.legend(fontsize=8.8, loc="lower left")

    hs2 = sorted(int(k) for k in dtg)
    errs = [dtg[str(h)]["node_v_mae_massw"] for h in hs2]
    a2.bar([str(h) for h in hs2], errs, 0.55, color=[WARM if h != 64 else "#ffd24d" for h in hs2])
    a2.axhline(g_step, color="#7ee787", lw=2.0)
    a2.text(0.02, g_step * 1.35, f"gravity's whole contribution to one substep:  dt x g = {g_step:.1e}",
            color="#7ee787", fontsize=10.5, transform=a2.get_yaxis_transform(),
            bbox=dict(facecolor=BG, edgecolor="#2a3340", boxstyle="round,pad=0.3"))
    a2.set_ylim(2.4e-4, 0.32)
    for i, (h, e) in enumerate(zip(hs2, errs)):
        a2.text(i, e * 1.12, f"{e / g_step:.0f}x", ha="center", color=FG, fontsize=10.5)
    a2.set_yscale("log")
    a2.set_xlabel("hidden width")
    a2.set_ylabel("mass-weighted node velocity error")
    a2.set_title("The network's error is 55 to 300 times gravity's per-substep effect",
                 color=FG, fontsize=12)
    style(a2)
    fig.tight_layout()
    fig.savefig(RUN / "gravity_below_noise.png")
    plt.close(fig)
    print("wrote", RUN / "gravity_below_noise.png")

    # ---- 4. a contact sheet of the headline clip, so the motion can be inspected as stills ----
    picks = [0, 5, 10, 18, 30, 44, 59]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.9), dpi=140)
    for c, f in enumerate(picks):
        panel(axes[0][c], ref[f], f"t = {(f + 1) * spf * dt:.2f} s", ACCENT)
        panel(axes[1][c], nets[best][f], "", WARM)
    for a, lab, col in ((axes[0][0], "canonical", ACCENT), (axes[1][0], "learned, width 64", WARM)):
        a.text(-0.06, 0.5, lab, transform=a.transAxes, rotation=90, va="center", ha="right",
               color=col, fontsize=10.5)
    fig.suptitle("The same rollout sampled in time. Top: canonical grid update. Bottom: learned.",
                 color=FG, fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))
    fig.savefig(RUN / "clip_contact_sheet.png")
    plt.close(fig)
    print("wrote", RUN / "clip_contact_sheet.png")


if __name__ == "__main__":
    sys.exit(main())
