"""The figures. Each one carries the quantity a claim rests on, with the old physics beside the new."""
import json, math, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("TASK_SCRATCH", os.path.join(HERE, "_scratch"))
sys.path.insert(0, HERE)
import common                                                  # noqa: E402

BG, INK, MUT = "#0a0e14", "#dfe6ee", "#7f8ea3"
OLD, NEW, ACC = "#e0736a", "#6fd3ee", "#ffd24d"
FLOOR = 3.0 / 128.0

vb = json.load(open(os.path.join(HERE, "volcurve_before.json")))
va = json.load(open(os.path.join(HERE, "volcurve_after.json")))
db = json.load(open(os.path.join(HERE, "diag_before.json")))
da = json.load(open(os.path.join(HERE, "diag_after.json")))
buoy = json.load(open(os.path.join(HERE, "buoyancy_after.json")))


def style(ax, xl, yl, title=None):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#2a3444")
    ax.set_xlabel(xl, color=INK, fontsize=10)
    ax.set_ylabel(yl, color=INK, fontsize=10)
    ax.grid(alpha=0.14, color=MUT, lw=0.6)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, pad=8)


# --------------------------------------------------------------- 1. rubber holds its volume
def fig_rubber():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), facecolor=BG)
    for ax, job, name in zip(axes, ("elastic/slam", "elastic/drop"),
                             ("hard floor impact (slam)", "released from rest (drop)")):
        for cur, col, lab in ((vb, OLD, "OLD  nu = 0.20"), (va, NEW, "NEW  nu = 0.45")):
            c = cur["curves"][job]
            ax.plot(c["t"], c["mean"], color=col, lw=2.2, label=lab + "   body average")
            ax.plot(c["t"], c["p01"], color=col, lw=1.3, ls="--", alpha=0.85,
                    label=lab + "   1st-pct particle")
        ax.axhline(1.0, color=MUT, lw=0.8, ls=":")
        ax.set_ylim(0.0, 1.12)
        style(ax, "time (s)", "volume ratio  det(F)", name)
        ax.legend(fontsize=7.6, facecolor="#111823", edgecolor="#2a3444", labelcolor=INK, loc="lower right")
    fig.suptitle("Rubber: how much of its volume an elastic blob still has, over time",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(HERE, "rubber_volume.png")
    fig.savefig(p, facecolor=BG, dpi=125)
    plt.close(fig)
    return p


# --------------------------------------------------------------- 2. water is no longer squashy
def fig_water_volume():
    fig, ax = plt.subplots(figsize=(6.6, 4.3), facecolor=BG)
    for cur, col, lab in ((vb, OLD, "OLD  E/rho = 180"), (va, NEW, "NEW  E/rho = 900")):
        c = cur["curves"]["fluid/drop"]
        ax.plot(c["t"], c["mean"], color=col, lw=2.2, label=lab + "   body average")
        ax.plot(c["t"], c["p01"], color=col, lw=1.3, ls="--", alpha=0.85,
                label=lab + "   1st-pct particle")
    ax.axhline(1.0, color=MUT, lw=0.8, ls=":")
    ax.set_ylim(0.55, 1.1)
    style(ax, "time (s)", "volume ratio  J", "Water: how much a water particle is squashed\n(drop scene, impact at t ~ 0.25 s)")
    ax.legend(fontsize=8, facecolor="#111823", edgecolor="#2a3444", labelcolor=INK, loc="lower right")
    fig.tight_layout()
    p = os.path.join(HERE, "water_volume.png")
    fig.savefig(p, facecolor=BG, dpi=125)
    plt.close(fig)
    return p


# --------------------------------------------------------------- 3. dam-break front
def fig_dam_front():
    T, nf = 1.4, 60
    t = np.arange(1, nf + 1) * T / nf
    h0 = 0.42 - FLOOR
    ritter = 2.0 * math.sqrt(9.8 * h0)
    fig, ax = plt.subplots(figsize=(6.8, 4.3), facecolor=BG)
    for src, col, lab in ((db, OLD, "OLD physics"), (da, NEW, "NEW physics")):
        f = np.array(src["runs"]["dam/fluid"]["front_curve"])
        ax.plot(t, f, color=col, lw=2.2, label=lab)
    ax.plot(t, 0.22 + ritter * t, color=MUT, lw=1.2, ls="--",
            label="ideal-fluid front  2 sqrt(g h0)")
    ax.axhline(1.0 - FLOOR, color=ACC, lw=1.0, ls=":")
    ax.text(0.02, 1.0 - FLOOR + 0.012, "far wall", color=ACC, fontsize=8.5)
    ax.set_xlim(0, 0.55); ax.set_ylim(0.15, 1.05)
    style(ax, "time (s)", "front position  (99th percentile of x)",
          "Dam break: how fast the leading edge runs")
    ax.legend(fontsize=8.5, facecolor="#111823", edgecolor="#2a3444", labelcolor=INK, loc="lower right")
    fig.tight_layout()
    p = os.path.join(HERE, "dam_front.png")
    fig.savefig(p, facecolor=BG, dpi=125)
    plt.close(fig)
    return p


# --------------------------------------------------------------- 4. buoyancy: settled frames
def fig_buoyancy():
    d = np.load(os.path.join(SCRATCH, "buoy_after.npz"))
    keys = ["mat_snow", "mat_elastic", "mat_sand"]
    labs = ["snow      rho 0.3", "rubber   rho 1.2", "sand      rho 1.6"]
    cols = ["#f2f6fc", "#ff4d4d", "#ffd24d"]
    fig = plt.figure(figsize=(12.6, 3.6), facecolor=BG)
    for i, (k, lab, cc) in enumerate(zip(keys, labs, cols)):
        ax = fig.add_subplot(1, 3, i + 1)
        sol = d[k + "/solid"].astype(np.float32)[-1]
        flu = d[k + "/fluid"].astype(np.float32)[-1]
        ax.set_facecolor(BG)
        ax.scatter(flu[:, 0], flu[:, 1], s=1.1, c="#2f7fd0", linewidths=0)
        ax.scatter(sol[:, 0], sol[:, 1], s=3.0, c=cc, linewidths=0)
        wl = np.percentile(flu[:, 1], 97)
        ax.axhline(wl, color="#9fe6ff", lw=1.0, ls="--")
        ax.text(0.012, wl + 0.008, "waterline", color="#9fe6ff", fontsize=8)
        rec = buoy["runs"][k]
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.40)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a3444")
        ax.set_title("%s\ndepth below waterline  %+.3f   submerged %.0f%%"
                     % (lab, rec["rest_depth_final"], 100 * rec["submerged_final"]),
                     color=INK, fontsize=10.5, pad=7)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.5, 0.035, "Released at rest, fully submerged, in the same pool. Settled state at "
                         "t = 2.2 s. Nothing in the solver applies a buoyancy force.",
             color=MUT, fontsize=10.5, ha="center", va="center")
    p = os.path.join(HERE, "buoyancy_three.png")
    fig.savefig(p, facecolor=BG, dpi=125)
    plt.close(fig)
    return p


# --------------------------------------------------------------- 5. the density ladder
def fig_ladder():
    keys = ["rho_0.3", "rho_0.6", "rho_1.0", "rho_1.6"]
    T, nf = buoy["T"], len(buoy["runs"][keys[0]]["rest_depth_curve"])
    t = np.arange(1, nf + 1) * T / nf
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), facecolor=BG)
    cmap = plt.get_cmap("plasma")
    for i, k in enumerate(keys):
        r = buoy["runs"][k]
        axes[0].plot(t, r["rest_depth_curve"], lw=2.1, color=cmap(0.15 + 0.7 * i / 3),
                     label="rho = %.1f" % r["rho"])
    axes[0].axhline(0.0, color=MUT, lw=0.9, ls=":")
    axes[0].text(T * 0.62, -0.012, "waterline", color=MUT, fontsize=8.5)
    axes[0].invert_yaxis()
    style(axes[0], "time (s)", "depth below the waterline",
          "The SAME rubber blob at four densities\n(only the mass differs)")
    axes[0].legend(fontsize=8.5, facecolor="#111823", edgecolor="#2a3444", labelcolor=INK)

    rr = [buoy["runs"][k]["rho"] for k in keys]
    dd = [buoy["runs"][k]["rest_depth_final"] for k in keys]
    ss = [100 * buoy["runs"][k]["submerged_final"] for k in keys]
    axes[1].plot(rr, dd, "o-", color=NEW, lw=2.2, ms=7)
    for a, b, s in zip(rr, dd, ss):
        axes[1].annotate("%.0f%% under" % s, (a, b), textcoords="offset points", xytext=(9, 13),
                         color=INK, fontsize=9)
    axes[1].set_xlim(0.15, 1.85)
    axes[1].axhline(0.0, color=MUT, lw=0.9, ls=":")
    axes[1].invert_yaxis()
    style(axes[1], "density (water = 1.0)", "settled depth below the waterline",
          "Settled depth is monotone in density")
    fig.tight_layout()
    p = os.path.join(HERE, "density_ladder.png")
    fig.savefig(p, facecolor=BG, dpi=125)
    plt.close(fig)
    return p


# --------------------------------------------------------------- 6. snow / sand did not move
def fig_regression():
    scenes = ["drop", "column", "heap", "slam", "dam"]
    metrics = [("spread_width", "spread width"), ("pile_height", "pile height"),
               ("repose_angle", "angle of repose (deg)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), facecolor=BG)
    for ax, (mk, mlab) in zip(axes, metrics):
        xs, ys, cs, labels = [], [], [], []
        for mi, mat in enumerate(["snow", "sand", "fluid", "elastic"]):
            for sc in scenes:
                k = f"{sc}/{mat}"
                xs.append(db["runs"][k][mk]); ys.append(da["runs"][k][mk])
                cs.append({"snow": "#f2f6fc", "sand": "#ffd24d",
                           "fluid": "#4db6ff", "elastic": "#ff4d4d"}[mat])
                labels.append(mat)
        lim = max(max(xs), max(ys)) * 1.08 + 1e-6
        ax.plot([0, lim], [0, lim], color=MUT, lw=1.0, ls="--")
        for mat, cc in (("snow", "#f2f6fc"), ("sand", "#ffd24d"),
                        ("fluid", "#4db6ff"), ("elastic", "#ff4d4d")):
            sel = [i for i, l in enumerate(labels) if l == mat]
            ax.scatter([xs[i] for i in sel], [ys[i] for i in sel], s=52, c=cc,
                       edgecolors="#0a0e14", linewidths=0.7, label=mat, zorder=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        style(ax, "OLD physics", "NEW physics", mlab)
    axes[0].legend(fontsize=8.5, facecolor="#111823", edgecolor="#2a3444", labelcolor=INK,
                   loc="upper left")
    fig.suptitle("Every material on every scene, old physics against new. On the dashed line means unchanged.",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(HERE, "regression.png")
    fig.savefig(p, facecolor=BG, dpi=125)
    plt.close(fig)
    return p


for f in (fig_rubber, fig_water_volume, fig_dam_front, fig_buoyancy, fig_ladder, fig_regression):
    p = f()
    print("wrote", os.path.basename(p), os.path.getsize(p) // 1024, "KB")
