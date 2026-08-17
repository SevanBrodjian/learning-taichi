"""Sand as a fourth canonical material, and four materials in one grid.

Produces every figure and clip the task page rests on, plus metrics.json. Run after the dt sweeps
(dt_experiment.py, dt_experiment2.py) and the phi calibration (probe_repose3.py), whose JSON this
script reads back for the summary figures.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

import render as R                                  # noqa: E402
import sim.physics as P                             # noqa: E402
from sim.physics import core as pc                  # noqa: E402

MATS = ("fluid", "elastic", "snow", "sand")
COL = R.DEMO_COLOR
BG, INK, MUTED, ACC = R.BG, R.INK, R.MUTED, R.ACCENT
M = {}          # accumulates metrics.json


def traj_rmse(a, b):
    """Registered metric: mean over frames and particles of |x_a - x_b| (see spec/registry)."""
    return float(np.linalg.norm(a - b, axis=-1).mean())


def note(k, v):
    M[k] = v
    return v


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#2a3444")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(alpha=0.13, color="#3a4658", lw=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


# =============================================================================== 1. the heap scene
def heap_scene(n=3500):
    sc = P.scene("heap", n)
    print(f"[heap] T={sc['T']} n={n}")
    snaps, times, ok = {}, None, {}
    for m in MATS:
        t0 = time.time()
        s, tt, st = P.simulate(m, sc["pts"], sc["area"], sc["T"], 96, v0=sc["v0"])
        snaps[m], times, ok[m] = s, tt, st
        print(f"  {m:8s} stable={st} repose={P.repose_angle(s[-1]):5.1f} "
              f"width={P.spread_width(s[-1]):.3f} height={P.pile_height(s[-1]):.3f} "
              f"({time.time() - t0:.1f}s)")
    return sc, snaps, times, ok


def render_heap(sc, snaps, times):
    seeded = P.repose_angle(sc["pts"])
    ang = {m: np.array([P.repose_angle(snaps[m][i]) for i in range(snaps[m].shape[0])])
           for m in MATS}
    note("heap", {
        "seeded_slope_deg": seeded,
        "T": sc["T"],
        "final": {m: {"repose_angle": float(ang[m][-1]),
                      "spread_width": P.spread_width(snaps[m][-1]),
                      "pile_height": P.pile_height(snaps[m][-1])} for m in MATS}})

    panels = [(R.LABEL[m], snaps[m], COL[m]) for m in MATS]

    def sub(k, fi):
        return f"slope {ang[MATS[k]][fi]:4.0f}°"

    p = R.render_panels(os.path.join(D, "heap_four_alone.mp4"), panels, times, pc.floor_y,
                        fps=30, ncols=2, panel_w=560, ylim=(0.0, 0.31), sub_fn=sub, pt=2.2)
    print("  wrote", p)

    # surface-profile figure: the picture behind the repose number
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5), dpi=130, facecolor=BG)
    for ax, m in zip(axes, MATS):
        style(ax)
        f = snaps[m][-1]
        ax.scatter(f[:, 0], f[:, 1] - pc.floor_y, s=1.4, c=COL[m], alpha=0.45, linewidths=0)
        cx, hh = P.surface_profile(f)
        ax.plot(cx, hh, color="#ffffff", lw=1.4, alpha=0.85, label="free surface")
        if cx.size >= 6:
            apex = int(np.argmax(hh))
            for sl in (slice(0, apex + 1), slice(apex, len(cx))):
                if len(cx[sl]) >= 3:
                    k, b = np.polyfit(cx[sl], hh[sl], 1)
                    ax.plot(cx[sl], k * cx[sl] + b, color=ACC, lw=2.2, ls="--")
        ax.set_title(f"{R.LABEL[m]}  —  {ang[m][-1]:.0f}°", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.30)
        ax.set_xlabel("x")
        if m == MATS[0]:
            ax.set_ylabel("height above floor")
    fig.suptitle(f"Settled free surface after releasing the same {seeded:.0f}° heap "
                 f"(dashed = fitted flank slope)", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(D, "repose_profile.png"), facecolor=BG)
    plt.close(fig)
    print("  wrote repose_profile.png")

    # angle-vs-time: sand relaxes, snow/elastic do not, fluid collapses
    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=140, facecolor=BG)
    style(ax)
    for m in MATS:
        ax.plot(times, ang[m], color=COL[m], lw=2.0, label=R.LABEL[m])
    ax.axhline(seeded, color=MUTED, ls=":", lw=1.2)
    ax.text(times[-1], seeded + 1.2, f"seeded {seeded:.0f}°", color=MUTED, fontsize=9, ha="right")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("free-surface slope (degrees)")
    ax.set_title("An over-steep heap, released from rest")
    lg = ax.legend(frameon=False, fontsize=9)
    for t in lg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "repose_vs_time.png"), facecolor=BG)
    plt.close(fig)
    print("  wrote repose_vs_time.png")
    M["heap"]["angle_series"] = {"t": times.tolist(),
                                 **{m: ang[m].round(2).tolist() for m in MATS}}


# =============================================================================== 2. the drop scene
def drop_scene(n=3500):
    sc = P.scene("drop", n)
    snaps, times = {}, None
    for m in MATS:
        s, tt, st = P.simulate(m, sc["pts"], sc["area"], sc["T"], 78, v0=sc["v0"])
        snaps[m], times = s, tt
        print(f"  drop {m:8s} stable={st} width={P.spread_width(s[-1]):.3f} "
              f"height={P.pile_height(s[-1]):.3f}")
    note("drop", {"final": {m: {"spread_width": P.spread_width(snaps[m][-1]),
                                "pile_height": P.pile_height(snaps[m][-1])} for m in MATS}})
    panels = [(R.LABEL[m], snaps[m], COL[m]) for m in MATS]
    wid = {m: [P.spread_width(snaps[m][i]) for i in range(snaps[m].shape[0])] for m in MATS}

    def sub(k, fi):
        return f"width {wid[MATS[k]][fi]:.2f}"

    p = R.render_panels(os.path.join(D, "drop_four_alone.mp4"), panels, times, pc.floor_y,
                        fps=30, ncols=2, panel_w=560, ylim=(0.0, 0.70), sub_fn=sub, pt=2.2)
    print("  wrote", p)


# ================================================================= 3. four materials in ONE grid
def multi_columns(n_each=2600):
    """Four blocks standing side by side on the floor, released from rest, in one shared grid."""
    centres = [0.155, 0.385, 0.615, 0.845]
    order = ["fluid", "sand", "snow", "elastic"]
    groups = []
    for i, (m, cx) in enumerate(zip(order, centres)):
        x0, x1 = cx - 0.075, cx + 0.075
        y0, y1 = pc.floor_y, 0.44
        groups.append({"material": m, "pts": pc.seed_box(x0, x1, y0, y1, n_each, seed=i),
                       "area": (x1 - x0) * (y1 - y0), "v0": (0.0, 0.0)})
    T = 1.9
    t0 = time.time()
    snaps, times, mid, ok, dt = P.simulate_multi(groups, T, 100)
    wall = time.time() - t0
    print(f"[multi columns] n={snaps.shape[1]} dt={dt:.2e} stable={ok} ({wall:.1f}s)")
    cols = np.array([COL[order[0]]] * snaps.shape[1], dtype=object)
    for m in MATS:
        cols[mid == P.MAT_ID[m]] = COL[m]
    cols = list(cols)
    legend = [(R.LABEL[m], COL[m]) for m in order]
    p = R.render_single(os.path.join(D, "four_in_one_grid.mp4"), snaps, cols, times, pc.floor_y,
                        "four materials, one grid", legend=legend, fps=30, width=1000,
                        ylim=(0.0, 0.60), pt=1.9,
                        note=f"one shared dt = {dt:.1e} s, forced by snow")
    print("  wrote", p)
    note("multi_columns", {
        "dt_shared": dt, "n_particles": int(snaps.shape[1]), "stable": bool(ok), "T": T,
        "substeps_per_frame_60fps": int(round((1 / 60) / dt)),
        "binding_material": min(MATS, key=lambda m: P.MAT[m]["dt"]),
        "wall_seconds": round(wall, 1),
        "per_material_final": {m: {
            "spread_width": P.spread_width(snaps[-1][mid == P.MAT_ID[m]]),
            "pile_height": P.pile_height(snaps[-1][mid == P.MAT_ID[m]])} for m in MATS}})
    return snaps, mid


def multi_heaps(n_each=2600):
    """The angle-of-repose signature, run for all four materials SIMULTANEOUSLY in one grid."""
    centres = [0.14, 0.38, 0.62, 0.86]
    order = ["fluid", "sand", "snow", "elastic"]
    hb = 0.10
    H = hb * np.tan(np.deg2rad(60.0))
    groups = []
    for i, (m, cx) in enumerate(zip(order, centres)):
        groups.append({"material": m, "pts": pc.seed_triangle(cx, pc.floor_y, hb, H, n_each, seed=i),
                       "area": hb * H, "v0": (0.0, 0.0)})
    T = 1.6
    snaps, times, mid, ok, dt = P.simulate_multi(groups, T, 88)
    print(f"[multi heaps] dt={dt:.2e} stable={ok}")
    cols = np.array([COL[order[0]]] * snaps.shape[1], dtype=object)
    for m in MATS:
        cols[mid == P.MAT_ID[m]] = COL[m]
    legend = [(R.LABEL[m], COL[m]) for m in order]
    p = R.render_single(os.path.join(D, "four_heaps_one_grid.mp4"), snaps, list(cols), times,
                        pc.floor_y, "the same 60° heap, four materials, one grid",
                        legend=legend, fps=30, width=1000, ylim=(0.0, 0.30), pt=1.9,
                        note="water flattens, sand finds its repose angle, snow and elastic hold")
    print("  wrote", p)
    note("multi_heaps", {"dt_shared": dt, "stable": bool(ok),
                         "final_repose": {m: P.repose_angle(snaps[-1][mid == P.MAT_ID[m]])
                                          for m in MATS}})


# ================================================ 4. the refactor changed nothing (with a number)
def equivalence(n=3500):
    sc = P.scene("drop", n)
    rows = {}
    keep = {}
    for m in MATS:
        a, times, _ = P.simulate(m, sc["pts"], sc["area"], sc["T"], 40, v0=sc["v0"])
        b, _, _ = P.simulate(m, sc["pts"], sc["area"], sc["T"], 40, v0=sc["v0"])
        g = [{"material": m, "pts": sc["pts"], "area": sc["area"], "v0": sc["v0"]}]
        c, _, _, ok, dt = P.simulate_multi(g, sc["T"], 40, dt=P.MAT[m]["dt"])
        noise = np.linalg.norm(a - b, axis=-1).mean(axis=1)
        cross = np.linalg.norm(a - c, axis=-1).mean(axis=1)
        rows[m] = {"self_noise": float(noise.mean()), "multi_vs_canonical": float(cross.mean()),
                   "ratio": float(cross.mean() / max(noise.mean(), 1e-30)),
                   "stable": bool(ok), "dt": dt,
                   "per_frame_self_noise": noise.round(9).tolist(),
                   "per_frame_multi_vs_canonical": cross.round(9).tolist(),
                   "t": times.round(4).tolist()}
        print(f"  equiv {m:8s} multi_vs_canonical={cross.mean():.3e} "
              f"self_noise={noise.mean():.3e} ratio={rows[m]['ratio']:.2f}")
        if m == "sand":
            keep = {"canonical": a, "multi": c, "times": times}
    note("equivalence", rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=140, facecolor=BG)
    style(ax)
    for m in MATS:
        t = rows[m]["t"]
        ax.plot(t, rows[m]["per_frame_self_noise"], color=COL[m], lw=1.0, ls=":", alpha=0.8)
        ax.plot(t, rows[m]["per_frame_multi_vs_canonical"], color=COL[m], lw=2.0,
                label=R.LABEL[m])
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean per-particle distance (domain lengths)")
    ax.set_title("Multi-material path vs canonical (solid) against the simulator's\n"
                 "own run-to-run self-noise (dotted). Same curve = no change.", fontsize=11)
    lg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in lg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "equivalence.png"), facecolor=BG)
    plt.close(fig)
    print("  wrote equivalence.png")

    # and as motion, for the one material the refactor introduced
    panels = [("sand via canonical simulate()", keep["canonical"], COL["sand"]),
              ("sand via the multi-material path", keep["multi"], COL["sand"])]
    p = R.render_panels(os.path.join(D, "equivalence_sand.mp4"), panels, keep["times"],
                        pc.floor_y, fps=25, ncols=2, panel_w=560, ylim=(0.0, 0.70), pt=2.2)
    print("  wrote", p)


# ================================================================== 5. summary figures from JSON
def figure_phi():
    cal = json.load(open(os.path.join(D, "phi_calibration.json")))
    phis = sorted(float(k) for k in cal)
    mean = [cal[str(int(p)) if str(int(p)) in cal else f"{p:.1f}"]["mean"] for p in phis]
    sd = [cal[str(int(p)) if str(int(p)) in cal else f"{p:.1f}"]["sd"] for p in phis]
    per = [cal[str(int(p)) if str(int(p)) in cal else f"{p:.1f}"]["per_size"] for p in phis]
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=140, facecolor=BG)
    style(ax)
    ax.axhspan(28, 35, color="#6fd3ee", alpha=0.10)
    sizes = sorted(per[0].keys(), key=float)
    marks = ["o", "s", "^"]
    for si, k in enumerate(sizes):
        ax.plot(phis, [p[k] for p in per], marks[si], ms=5, color=MUTED, alpha=0.75,
                label=f"pile half-width {float(k):.2f} ({float(k) / pc.dx:.0f} cells)")
    ax.errorbar(phis, mean, yerr=sd, color=COL["sand"], lw=2.2, capsize=4, marker="o", ms=6,
                label="mean over 4 seeds x 3 sizes")
    ax.axvline(52.5, color="#ff6e6e", ls="--", lw=1.4)
    lo, hi = min(mean) - max(sd) - 4, max(mean) + max(sd) + 5
    ax.set_ylim(lo, hi)
    ax.text(max(phis) + 0.3, 35.4, "real dry sand, 28-35 deg", color=ACC, fontsize=9, ha="right")
    ax.text(53.2, lo + 1.0, "past here the measured angle\nstops being reproducible and\nstarts tracking the grid",
            color="#ff6e6e", fontsize=8.6, va="bottom")
    m50 = dict(zip(phis, mean))[50.0]
    ax.plot([50], [m50], "*", ms=17, color="#ffffff", zorder=6)
    ax.annotate("canonical\nphi = 50 deg", xy=(50, m50), xytext=(46.0, m50 + 5.5), color=INK,
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.0))
    ax.set_xlabel("Drucker-Prager friction angle φ (degrees, a model parameter)")
    ax.set_ylabel("measured angle of repose (degrees)")
    ax.set_title("The friction angle is not the angle of repose")
    lg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in lg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "phi_calibration.png"), facecolor=BG)
    plt.close(fig)
    print("  wrote phi_calibration.png")
    note("phi_calibration", cal)


def figure_budget():
    sw = json.load(open(os.path.join(D, "dt_sweep2.json")))
    s1 = json.load(open(os.path.join(D, "dt_sweep.json")))
    summ = sw["summary"]
    note("dt", {"summary": summ, "scenes": sw["scenes"],
                "first_sweep": {m: {k: v for k, v in s1["materials"][m].items() if k != "sweep"}
                                for m in MATS},
                "snow_hardening": {k: v for k, v in s1["snow_hardening"].items()
                                   if k != "xi0_sweep"}})
    spf = {m: summ[m]["spf60_canonical"] for m in MATS}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.1), dpi=140, facecolor=BG,
                                  gridspec_kw={"width_ratios": [1.05, 1]})
    style(ax)
    style(ax2)
    order = sorted(MATS, key=lambda m: spf[m])
    ax.bar([f"{R.LABEL[m]}\ndt={P.MAT[m]['dt']:.1e}" for m in order], [spf[m] for m in order],
           color=[COL[m] for m in order], edgecolor="#0a0e14")
    for i, m in enumerate(order):
        ax.text(i, spf[m] + 9, f"{spf[m]}", ha="center", color=INK, fontsize=11, weight="bold")
    ax.set_ylabel("substeps per frame at 60 fps")
    ax.set_title("What each material costs, alone")
    ax.set_ylim(0, max(spf.values()) * 1.22)

    combos = [("water", ["fluid"]), ("water+sand", ["fluid", "sand"]),
              ("water+sand\n+elastic", ["fluid", "sand", "elastic"]),
              ("all four\n(+snow)", list(MATS))]
    vals = [int(round((1 / 60) / P.shared_dt(c))) for _, c in combos]
    bind = [min(c, key=lambda m: P.MAT[m]["dt"]) for _, c in combos]
    ax2.bar([c[0] for c in combos], vals, color=[COL[b] for b in bind], edgecolor="#0a0e14")
    for i, (v, b) in enumerate(zip(vals, bind)):
        ax2.text(i, v + 8, f"{v}", ha="center", color=INK, fontsize=10, weight="bold")
        ax2.text(i, v * 0.5, f"set by\n{R.LABEL[b]}", ha="center", color="#0a0e14", fontsize=8.2,
                 weight="bold")
    ax2.set_ylabel("substeps per frame at 60 fps")
    ax2.set_title("One grid = one dt: the stiffest material bills everyone")
    ax2.set_ylim(0, max(vals) * 1.22)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "dt_budget.png"), facecolor=BG)
    plt.close(fig)
    print("  wrote dt_budget.png")
    note("shared_grid_cost", [{"scene": c[0].replace("\n", " "), "materials": c[1],
                               "dt": P.shared_dt(c[1]), "substeps_per_frame": v,
                               "binding": b} for (c, v, b) in zip(combos, vals, bind)])


def figure_stability():
    """The flip: 'stable' says every material tolerates a far bigger dt than it uses. 'still the same
    material' says something quite different."""
    sw = json.load(open(os.path.join(D, "dt_sweep2.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), dpi=140, facecolor=BG)
    for ax, scene in zip(axes, ("drop", "heap")):
        style(ax)
        # a diverged run has no meaningful settled shape (its positions were nan-scrubbed), so it is
        # marked at the top of the axis rather than drawn as if it were a drift value
        top = max(max(r["shape_drift"] for r in sw["scenes"][scene][m]["sweep"] if r["ok"])
                  for m in MATS) * 3.0
        for m in MATS:
            rec = sw["scenes"][scene][m]
            ok = [r for r in rec["sweep"] if r["ok"]]
            ax.plot([r["dt"] / rec["dt_canonical"] for r in ok],
                    [max(r["shape_drift"], 1e-6) for r in ok], "-o", ms=4, color=COL[m], lw=1.8,
                    label=R.LABEL[m])
            bad = [r["dt"] / rec["dt_canonical"] for r in rec["sweep"] if not r["ok"]]
            if bad:
                ax.plot(bad, [top] * len(bad), "x", ms=10, mew=2.2, color=COL[m])
        ax.axhline(top / 1.7, color="#ff6e6e", ls="-", lw=0.9, alpha=0.5)
        ax.text(0.115, top * 1.08, "diverged", color="#ff6e6e", fontsize=9)
        ax.axhline(sw["tol_shape"], color=ACC, ls="--", lw=1.3)
        ax.text(0.115, sw["tol_shape"] * 1.2, "15% shape drift", color=ACC, fontsize=9)
        ax.axvline(1.0, color=MUTED, ls=":", lw=1.2)
        ax.text(1.05, 2e-6, "canonical dt", color=MUTED, fontsize=8.5, rotation=90)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, top * 2.2)
        ax.set_xlabel("timestep, as a multiple of the canonical dt")
        ax.set_ylabel("settled-shape drift vs a dt/8 run")
        ax.set_title(f"{scene} scene")
    lg = axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for t in lg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "dt_faithfulness.png"), facecolor=BG)
    plt.close(fig)
    print("  wrote dt_faithfulness.png")


def figure_snow_stiffness():
    s1 = json.load(open(os.path.join(D, "dt_sweep.json")))
    h = s1["snow_hardening"]
    sc = P.scene("drop", 4000)
    P.simulate("snow", sc["pts"], sc["area"], 0.9, 4)
    jp = pc.Jp.to_numpy()[:4000]
    eff = P.MAT["snow"]["E"] * np.exp(P.MAT["snow"]["xi"] * (1.0 - jp))
    fig, ax = plt.subplots(figsize=(7.6, 4.1), dpi=140, facecolor=BG)
    style(ax)
    ax.hist(np.clip(eff, 0, 3000), bins=70, color=COL["snow"], alpha=0.85)
    for m, ls in (("fluid", "-"), ("sand", "-"), ("elastic", "-")):
        ax.axvline(P.MAT[m]["E"], color=COL[m], lw=2.0, ls=ls)
        ax.text(P.MAT[m]["E"] + 14, ax.get_ylim()[1] * 0.86, f"{R.LABEL[m]} E={P.MAT[m]['E']:.0f}",
                color=COL[m], fontsize=9, rotation=90, va="top")
    ax.axvline(P.MAT["snow"]["E"], color="#8899aa", lw=2.0, ls="--")
    ax.text(P.MAT["snow"]["E"] - 26, ax.get_ylim()[1] * 0.86, "snow nominal E=150", color="#8899aa",
            fontsize=9, rotation=90, va="top", ha="right")
    ax.set_xlabel("effective stiffness  E · exp(ξ(1−Jp))  after impact")
    ax.set_ylabel("particles")
    ax.set_title(f"Snow's hardening is real: median {h['E_eff_median']:.0f}, "
                 f"95th pct {h['E_eff_p95']:.0f}\n(but it is not what pins snow's timestep)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "snow_stiffness.png"), facecolor=BG)
    plt.close(fig)
    print("  wrote snow_stiffness.png")


# =============================================================================== main
if __name__ == "__main__":
    note("physics_version", P.VERSION)
    note("materials", {m: dict(P.MAT[m]) for m in MATS})
    note("dp_alpha_sand", P.dp_alpha(P.MAT["sand"]["phi"]))
    print("=== heap ===")
    sc, snaps, times, ok = heap_scene()
    render_heap(sc, snaps, times)
    print("=== drop ===")
    drop_scene()
    print("=== multi ===")
    multi_columns()
    multi_heaps()
    print("=== equivalence ===")
    equivalence()
    print("=== figures ===")
    figure_phi()
    figure_budget()
    figure_stability()
    figure_snow_stiffness()
    json.dump(M, open(os.path.join(D, "metrics.json"), "w"), indent=1)
    print("wrote metrics.json")
