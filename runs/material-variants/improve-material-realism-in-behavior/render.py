"""Headless particle rendering for the four canonical materials.

Colours follow the Demo MVP palette (coordination/demo-mvp.md): elastic red, sand yellow, water blue,
snow white. These are the DEMO's distinguishing colours and are deliberately not the same object as
sim.physics.MAT[...]['color'] (the canonical figure palette, which this task leaves unchanged).

Layout notes that are not cosmetic. Every clip takes an explicit `ylim`, because these scenes put all
their material in the bottom quarter of the unit domain and a square frame would spend three quarters
of every pixel on empty space. Labels live in a dedicated header strip and the clock in a footer strip,
so no annotation is ever drawn on top of the material it is describing.

Frames are drawn with matplotlib (Agg) and encoded with imageio. No ti.GUI anywhere.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

BG = "#0a0e14"
INK = "#dfe6ee"
MUTED = "#7f8ea3"
ACCENT = "#6fd3ee"

DEMO_COLOR = {
    "elastic": "#ff4d4d",   # red
    "sand":    "#ffd24d",   # yellow
    "fluid":   "#4db6ff",   # blue  (called "water" in the demo)
    "snow":    "#f2f6fc",   # white
}
LABEL = {"elastic": "elastic", "sand": "sand", "fluid": "water", "snow": "snow"}

HDR = 32      # px reserved above each row of panels for its label
FTR = 24      # px reserved at the bottom of the figure for the clock


def _frame_axes(ax, floor_y, ylim):
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(floor_y, color="#2a3444", lw=1.2, zorder=1)


def _grab(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()


def encode(path, frames, fps=30):
    import imageio
    frames = [f[: f.shape[0] - f.shape[0] % 2, : f.shape[1] - f.shape[1] % 2] for f in frames]
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)
    return path


def render_panels(path, panels, times, floor_y, *, fps=30, dpi=110, panel_w=560, ncols=2,
                  ylim=(0.0, 1.0), sub_fn=None, pt=2.4):
    """panels = [(label, snaps (F,n,2), colour), ...] laid out in a grid of `ncols` columns.

    `sub_fn(panel_index, frame_index) -> str` is appended to the panel's header label, so a per-frame
    readout (a measured slope, a width) sits beside the name instead of on top of the particles."""
    n = len(panels)
    rows = int(np.ceil(n / ncols))
    span = ylim[1] - ylim[0]
    cell_h = panel_w * span
    W = panel_w * ncols
    H = rows * (cell_h + HDR) + FTR
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    axes = []
    for k in range(n):
        r, c = divmod(k, ncols)
        bottom = FTR + (rows - 1 - r) * (cell_h + HDR)
        axes.append(fig.add_axes([c / ncols, bottom / H, 1 / ncols, cell_h / H]))
    frames = []
    for fi in range(panels[0][1].shape[0]):
        for k, (lab, snaps, col) in enumerate(panels):
            ax = axes[k]
            ax.clear()
            _frame_axes(ax, floor_y, ylim)
            ax.scatter(snaps[fi][:, 0], snaps[fi][:, 1], s=pt, c=col, linewidths=0, alpha=0.95,
                       zorder=3)
            r, c = divmod(k, ncols)
            ypx = FTR + (rows - 1 - r) * (cell_h + HDR) + cell_h + 11
            txt = lab if sub_fn is None else f"{lab}    {sub_fn(k, fi)}"
            fig.text((c + 0.5) / ncols, ypx / H, txt, ha="center", va="center", color=INK,
                     fontsize=12.5, weight="bold")
        fig.text(0.5, 11 / H, f"t = {times[fi]:.2f} s", ha="center", va="center", color=MUTED,
                 fontsize=9.5)
        frames.append(_grab(fig))
        for t in list(fig.texts):
            t.remove()
    plt.close(fig)
    return encode(path, frames, fps)


def render_single(path, snaps, colors, times, floor_y, label, *, fps=30, dpi=110, width=900,
                  ylim=(0.0, 1.0), legend=None, pt=2.4, note=None):
    """One wide panel with a per-particle colour array. `legend` is [(name, colour), ...] drawn as a
    horizontal strip in the header, so nothing overlaps the simulation."""
    span = ylim[1] - ylim[0]
    cell_h = width * span
    H = cell_h + HDR + FTR
    fig = plt.figure(figsize=(width / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, FTR / H, 1, cell_h / H])
    frames = []
    for fi in range(snaps.shape[0]):
        ax.clear()
        _frame_axes(ax, floor_y, ylim)
        ax.scatter(snaps[fi][:, 0], snaps[fi][:, 1], s=pt, c=colors, linewidths=0, alpha=0.95,
                   zorder=3)
        ypx = FTR + cell_h + 12
        fig.text(0.012, ypx / H, label, ha="left", va="center", color=INK, fontsize=12.5,
                 weight="bold")
        if legend:
            x = 0.995
            for nm, cc in reversed(legend):
                fig.text(x, ypx / H, "● " + nm, ha="right", va="center", color=cc,
                         fontsize=11.5)
                x -= 0.075 + 0.011 * len(nm)
        fig.text(0.5, 11 / H, f"t = {times[fi]:.2f} s" + (f"     {note}" if note else ""),
                 ha="center", va="center", color=MUTED, fontsize=9.5)
        frames.append(_grab(fig))
        for t in list(fig.texts):
            t.remove()
    plt.close(fig)
    return encode(path, frames, fps)
