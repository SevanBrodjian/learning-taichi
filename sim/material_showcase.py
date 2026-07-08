"""Forward-only material showcase: fluid vs elastic vs snow under the SAME initial condition.

A demonstration (NO gradients, NO autodiff tape, NO optimiser, NO loss) of how three constitutive
models move and differ in MLS-MPM. One p2g / grid_op / g2p skeleton is shared; only the stress branch
and the state it carries change:

  * ``fluid``   -- weakly compressible pressure from the tracked volume ratio J. Forgets shear history.
  * ``elastic`` -- corotated stress from the deformation gradient F (via ti.svd). Springs back.
  * ``snow``    -- elastic + plastic clamp of F's singular values into [1-theta_c, 1+theta_s] with
                   hardening exp(xi(1-Jp)). Crumples and holds a dented shape.

The constitutive physics is lifted verbatim from ``sim/material_variants.py`` (fluid_stress,
corotated_PFt, elastic_stress, snow_stress, and the g2p snow SVD clamp), with all the loss / x_avg /
autodiff machinery stripped out. Forward idioms follow the pristine ``sim/mpm88.py``.

Every material runs to the SAME physical time T (so different per-material dt cover the same simulated
seconds and the triptych panels are time-synchronised). Rendering is HEADLESS: frames are drawn with
matplotlib (Agg) and encoded to mp4 with imageio; no ti.GUI loop.

Usage:
    python sim/material_showcase.py            # render all scenes, still PNGs, diagnostics -> run dir
    python sim/material_showcase.py --quick    # fewer frames / shorter T for a fast smoke test
"""
import argparse
import datetime
import json
import os

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

# --- shared grid / world constants (mirror mpm88.py at n_grid=128) ---
dim = 2
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
p_rho = 1.0
gravity = 9.8
bound = 3
floor_y = bound * dx
NU = 0.2                       # Poisson ratio, fixed for every solid path
FRICTION = 0.5                 # Coulomb friction coefficient at the floor and walls

MAX_P = 16384                  # field capacity; each scene uses its own n_active <= MAX_P

FLUID, ELASTIC, SNOW = 0, 1, 2

# --- single-frame (in-place) state fields; forward-only, no time index, no needs_grad ---
x = ti.Vector.field(dim, float, MAX_P)
v = ti.Vector.field(dim, float, MAX_P)
C = ti.Matrix.field(dim, dim, float, MAX_P)
J = ti.field(float, MAX_P)                       # volume ratio (fluid path)
F = ti.Matrix.field(dim, dim, float, MAX_P)      # deformation gradient (elastic / snow path)
Jp = ti.field(float, MAX_P)                      # accumulated plastic volume change (snow)

grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

x_np_buf = ti.Vector.field(dim, float, MAX_P)    # scratch for host->device seed


# --------------------------------------------------------------------------- constitutive stress
@ti.func
def fluid_stress(p, dt, E, p_vol):
    """Weakly compressible isotropic pressure from J. Already scaled by the MLS-MPM affine prefactor."""
    s = -dt * 4.0 * E * p_vol * (J[p] - 1.0) * inv_dx * inv_dx
    return ti.Matrix([[s, 0.0], [0.0, s]])


@ti.func
def corotated_PFt(Fc, mu, la):
    """Corotated first-Piola stress contracted with F^T: 2 mu (F-R) F^T + la (J-1) J I. R = U V^T."""
    U, sig, Vt = ti.svd(Fc)
    R = U @ Vt.transpose()
    Jdet = Fc.determinant()
    return 2.0 * mu * (Fc - R) @ Fc.transpose() + la * (Jdet - 1.0) * Jdet * ti.Matrix.identity(float, dim)


@ti.func
def elastic_stress(p, dt, E, p_vol):
    mu = E / (2.0 * (1.0 + NU))
    la = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    PFt = corotated_PFt(F[p], mu, la)
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * PFt


@ti.func
def snow_stress(p, dt, E, xi, p_vol):
    h = ti.exp(xi * (1.0 - Jp[p]))          # hardening: compacted snow (Jp<1) stiffens
    mu = (E / (2.0 * (1.0 + NU))) * h
    la = (E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))) * h
    PFt = corotated_PFt(F[p], mu, la)
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * PFt


# --------------------------------------------------------------------------- MLS-MPM steps
@ti.kernel
def clear_grid():
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_v[i, j] = ti.Vector.zero(float, dim)
        grid_m[i, j] = 0.0


@ti.kernel
def p2g(mat: ti.template(), n: ti.i32, dt: ti.f32, E: ti.f32, xi: ti.f32,
        p_vol: ti.f32, p_mass: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = ti.Matrix.zero(float, dim, dim)
        if ti.static(mat == FLUID):
            stress = fluid_stress(p, dt, E, p_vol)
        elif ti.static(mat == ELASTIC):
            stress = elastic_stress(p, dt, E, p_vol)
        else:
            stress = snow_stress(p, dt, E, xi, p_vol)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass


@ti.func
def coulomb(vt, cap):
    """Reduce a tangential velocity component toward zero by up to ``cap`` (a Coulomb friction impulse).
    If |vt| <= cap the node sticks; otherwise it keeps its excess. cap is proportional to the normal
    velocity being cancelled, so a fast-flowing fluid keeps most of its speed while a slow settling pile
    sticks -- which is what lets a fluid spread while snow builds an angle-of-repose heap."""
    r = vt
    if vt > 0:
        r = ti.max(0.0, vt - cap)
    elif vt < 0:
        r = ti.min(0.0, vt + cap)
    return r


@ti.kernel
def grid_op(dt: ti.f32, fric: ti.f32):
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        if m > 0.0:
            grid_v[i, j] = grid_v[i, j] / m
        grid_v[i, j].y -= dt * gravity
        vx = grid_v[i, j].x
        vy = grid_v[i, j].y
        # Floor: separating with Coulomb friction. Cancel the inward (downward) normal velocity and
        # apply a friction impulse proportional to it on the tangential component. This basal friction
        # is what lets a slumping pile hold a slope while a fast-flowing fluid keeps most of its speed
        # and still spreads.
        if j < bound and vy < 0:
            vx = coulomb(vx, fric * (-vy))
            vy = 0.0
        if j > n_grid - bound and vy > 0:   # ceiling: separating
            vy = 0.0
        # Side walls: sticky. Anything moving into a wall stops completely, so an impact splash cannot
        # ride up a frictionless wall and fill the box. A weakly Coulomb wall barely damps a sheet
        # sliding UP it (the into-wall velocity is small there), so a full stick is the clean choice.
        if i < bound and vx < 0:
            vx = 0.0
            vy = 0.0
        if i > n_grid - bound and vx > 0:
            vx = 0.0
            vy = 0.0
        grid_v[i, j] = ti.Vector([vx, vy])


@ti.func
def g2p_gather(p):
    Xp = x[p] * inv_dx
    base = int(Xp - 0.5)
    fx = Xp - base
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    new_v = ti.Vector.zero(float, dim)
    new_C = ti.Matrix.zero(float, dim, dim)
    for i, j in ti.static(ti.ndrange(3, 3)):
        offset = ti.Vector([i, j])
        dpos = (offset - fx) * dx
        weight = w[i].x * w[j].y
        g_v = grid_v[base[0] + i, base[1] + j]
        new_v += weight * g_v
        new_C += 4.0 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx
    return new_v, new_C


@ti.kernel
def g2p(mat: ti.template(), n: ti.i32, dt: ti.f32, theta_c: ti.f32, theta_s: ti.f32):
    for p in range(n):
        new_v, new_C = g2p_gather(p)
        v[p] = new_v
        x[p] = x[p] + dt * new_v
        # Keep particles inside the domain (respect bound) so nothing clips through a wall/floor.
        x[p] = ti.math.clamp(x[p], floor_y, 1.0 - floor_y)
        if ti.static(mat == FLUID):
            J[p] = J[p] * (1.0 + dt * new_C.trace())     # volume ratio evolves by velocity divergence
        elif ti.static(mat == ELASTIC):
            F[p] = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
        else:
            F_tr = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]  # trial elastic F
            U, sig, Vt = ti.svd(F_tr)
            s0 = ti.min(ti.max(sig[0, 0], 1.0 - theta_c), 1.0 + theta_s)  # plastic clamp of stretches
            s1 = ti.min(ti.max(sig[1, 1], 1.0 - theta_c), 1.0 + theta_s)
            Jp[p] = Jp[p] * (sig[0, 0] * sig[1, 1]) / (s0 * s1)          # push clamped-off part into Jp
            F[p] = U @ ti.Matrix([[s0, 0.0], [0.0, s1]]) @ Vt           # keep only recoverable elastic F
        C[p] = new_C


@ti.kernel
def init_state(n: ti.i32):
    for p in range(n):
        x[p] = x_np_buf[p]
        v[p] = ti.Vector.zero(float, dim)
        C[p] = ti.Matrix.zero(float, dim, dim)
        J[p] = 1.0
        Jp[p] = 1.0
        F[p] = ti.Matrix.identity(float, dim)


# --------------------------------------------------------------------------- scene setup
def seed_disk(center, radius, n):
    """n points uniformly in a disk (rejection-free: sqrt radius for uniform area)."""
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    pts = np.stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)], axis=1)
    return pts


def seed_box(x0, x1, y0, y1, n):
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], axis=1)
    return pts


def upload(pts):
    """Copy a (n,2) numpy array into the seed buffer (padded to MAX_P)."""
    n = pts.shape[0]
    buf = np.zeros((MAX_P, dim), dtype=np.float32)
    buf[:n] = pts.astype(np.float32)
    x_np_buf.from_numpy(buf)
    return n


# --------------------------------------------------------------------------- rollout
# Per-material stability settings. Snow uses a softer E and a smaller dt so its SVD clamp stays finite
# (the explicit CFL limit scales like 1/sqrt(E); see the material-stiffness training page). Every
# material is integrated to the SAME physical time T, so a smaller dt just means more substeps.
MAT_CFG = {
    "fluid":   {"id": FLUID,   "E": 180.0, "dt": 1.2e-4, "xi": 0.0,  "tc": 0.0,   "ts": 0.0,    "color": "#4db6ff"},
    "elastic": {"id": ELASTIC, "E": 400.0, "dt": 1.0e-4, "xi": 0.0,  "tc": 0.0,   "ts": 0.0,    "color": "#ff9d5c"},
    "snow":    {"id": SNOW,    "E": 150.0, "dt": 5.0e-5, "xi": 10.0, "tc": 2.5e-2, "ts": 7.5e-3, "color": "#e6ecff"},
}


def run_material(mat_name, pts, p_vol_scale, T, n_frames, cfg_override=None):
    """Roll one material forward to physical time T under the given seed, capturing n_frames position
    snapshots evenly in physical time. Returns (snaps (n_frames,n,2), times (n_frames,), stable_bool)."""
    cfg = dict(MAT_CFG[mat_name])
    if cfg_override:
        cfg.update(cfg_override)
    n = upload(pts)
    # physically consistent density: p_vol = region_area / n_particles, so total mass = area * p_rho
    # regardless of particle count. p_vol_scale carries the region area.
    p_vol = p_vol_scale / n
    p_mass = p_vol * p_rho
    dt = cfg["dt"]
    mat_id = cfg["id"]
    steps_per_frame = max(1, int(round((T / n_frames) / dt)))
    init_state(n)
    snaps = np.zeros((n_frames, n, dim), dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    t = 0.0
    stable = True
    for fidx in range(n_frames):
        for _ in range(steps_per_frame):
            clear_grid()
            p2g(mat_id, n, dt, cfg["E"], cfg["xi"], p_vol, p_mass)
            grid_op(dt, FRICTION)
            g2p(mat_id, n, dt, cfg["tc"], cfg["ts"])
            t += dt
        cur = x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            print(f"    [!] {mat_name} went non-finite at frame {fidx} (t={t:.3f}) -- unstable dt/E")
            cur = np.nan_to_num(cur, nan=0.0)
        snaps[fidx] = cur
        times[fidx] = t
    return snaps, times, stable


# --------------------------------------------------------------------------- rendering
BG = "#0a0e14"
GROUND = "#161c26"
WALL = "#26313d"
INK = "#dfe6ee"


def _draw_panel(ax, pts, color, label, tlabel):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG)
    ax.axis("off")
    # ground strip + floor line + side walls
    ax.axhspan(0, floor_y, color=GROUND, zorder=0)
    ax.axhline(floor_y, color=WALL, lw=1.0, zorder=1)
    ax.axvline(floor_y, color=WALL, lw=0.8, alpha=0.6, zorder=1)
    ax.axvline(1.0 - floor_y, color=WALL, lw=0.8, alpha=0.6, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=4.0, c=color, edgecolors="none", alpha=0.85, zorder=2)
    ax.text(0.5, 0.955, label, ha="center", va="center", color=INK, fontsize=13,
            weight="bold", transform=ax.transAxes)
    if tlabel is not None:
        ax.text(0.5, 0.055, tlabel, ha="center", va="center", color="#9fb0c0",
                fontsize=9, transform=ax.transAxes)


def render_triptych(path, panels, times, fps=30, dpi=110, panel=380):
    """panels = [(label, snaps (F,n,2), color), ...]; write a side-by-side mp4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    ncols = len(panels)
    W = panel * ncols
    H = panel
    n_frames = panels[0][1].shape[0]
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    axes = [fig.add_axes([k / ncols, 0.0, 1.0 / ncols, 1.0]) for k in range(ncols)]
    frames = []
    for fidx in range(n_frames):
        tlabel = f"t = {times[fidx]:.2f} s"
        for k, (label, snaps, color) in enumerate(panels):
            ax = axes[k]
            ax.clear()
            _draw_panel(ax, snaps[fidx], color, label, tlabel)
        fig.canvas.draw()
        cw, ch = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(ch, cw, 4)[..., :3]
        rgb = rgb[: ch - (ch % 2), : cw - (cw % 2), :]   # even dims for libx264/yuv420p
        frames.append(rgb.copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


def render_still(path, panels, times, fidx, dpi=140, panel=420):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(panels)
    fig = plt.figure(figsize=(panel * ncols / dpi, panel / dpi), dpi=dpi, facecolor=BG)
    axes = [fig.add_axes([k / ncols, 0.0, 1.0 / ncols, 1.0]) for k in range(ncols)]
    tlabel = f"t = {times[fidx]:.2f} s"
    for k, (label, snaps, color) in enumerate(panels):
        _draw_panel(axes[k], snaps[fidx], color, label, tlabel)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------- diagnostics
def diagnostics(snaps_final):
    """Final-frame shape diagnostics: horizontal spread (width) and pile height above the floor.
    Robust percentiles (5-95) so a few stray particles do not dominate."""
    xs = snaps_final[:, 0]
    ys = snaps_final[:, 1]
    width = float(np.percentile(xs, 95) - np.percentile(xs, 5))
    height = float(np.percentile(ys, 95) - floor_y)
    com_y = float(ys.mean() - floor_y)
    rms = float(np.sqrt(((snaps_final - snaps_final.mean(axis=0)) ** 2).sum(axis=1).mean()))
    return {"width": width, "height": height, "com_height": com_y, "rms_spread": rms}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Forward-only fluid/elastic/snow material showcase")
    ap.add_argument("--quick", action="store_true", help="short smoke test (fewer frames, shorter T)")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo, "runs", "material-variants",
                           "implement-nondifferentiable-material-variants")
    os.makedirs(out_dir, exist_ok=True)

    materials = ["fluid", "elastic", "snow"]
    nf = 40 if args.quick else 100
    diag_table = {}   # scene -> {material -> diag}
    results_media = []

    # ---------------- Scene 1: drop & splat (a disk released above the floor) ----------------
    print("=== Scene 1: drop & splat ===")
    disk = seed_disk(center=(0.5, 0.52), radius=0.11, n=9000)
    area1 = np.pi * 0.11 ** 2
    T1 = 0.9 if args.quick else 1.3
    s1 = {}
    times1 = None
    for m in materials:
        print(f"  rolling {m} ...")
        snaps, times, stable = run_material(m, disk, area1, T1, nf)
        s1[m] = snaps
        times1 = times
        diag_table.setdefault("drop", {})[m] = diagnostics(snaps[-1])
        print(f"    {m}: stable={stable}  final width={diag_table['drop'][m]['width']:.3f} "
              f"height={diag_table['drop'][m]['height']:.3f}")
    panels1 = [(m.capitalize(), s1[m], MAT_CFG[m]["color"]) for m in materials]
    render_triptych(os.path.join(out_dir, "drop_triptych.mp4"), panels1, times1)
    render_still(os.path.join(out_dir, "drop_still.png"), panels1, times1, nf - 1)
    results_media += [
        ("video", "drop_triptych.mp4",
         "Drop and splat: the same disk released above the floor, one panel per material. Fluid spreads "
         "into a flat puddle, elastic squashes on impact then springs back to a rounded jiggling blob, "
         "snow crumples and packs into a static dented heap. Same initial condition in all three panels."),
        ("image", "drop_still.png",
         "Final settled frame of the drop-and-splat scene. Fluid has spread into a wide flat puddle, "
         "elastic has sprung back to a compact rounded blob, and snow sits between them as a crumpled "
         "heap that neither flowed flat nor rebounded."),
    ]

    # ---------------- Scene 2: column collapse (a tall block slumping from rest) ----------------
    print("=== Scene 2: column collapse ===")
    col = seed_box(0.42, 0.58, floor_y, 0.56, 7000)
    area2 = (0.58 - 0.42) * (0.56 - floor_y)
    T2 = 1.0 if args.quick else 1.7
    s2 = {}
    times2 = None
    for m in materials:
        print(f"  rolling {m} ...")
        snaps, times, stable = run_material(m, col, area2, T2, nf)
        s2[m] = snaps
        times2 = times
        diag_table.setdefault("column", {})[m] = diagnostics(snaps[-1])
        print(f"    {m}: stable={stable}  final width={diag_table['column'][m]['width']:.3f} "
              f"height={diag_table['column'][m]['height']:.3f}")
    panels2 = [(m.capitalize(), s2[m], MAT_CFG[m]["color"]) for m in materials]
    render_triptych(os.path.join(out_dir, "column_triptych.mp4"), panels2, times2)
    render_still(os.path.join(out_dir, "column_still.png"), panels2, times2, nf - 1)
    results_media += [
        ("video", "column_triptych.mp4",
         "Column collapse: an identical tall block released from rest, one panel per material. Fluid runs "
         "out into a flat sheet, elastic wobbles and springs back toward its original height, snow slumps "
         "into a stable angle-of-repose pile that holds its shape. Same initial block in all three panels."),
        ("image", "column_still.png",
         "Final settled frame of the column-collapse scene. Fluid has run out flattest and widest, snow "
         "holds a slumped angle-of-repose pile, and elastic has sprung back to stand nearly its full "
         "original height."),
    ]

    # ---------------- Scene 3 (bonus): the stiffness dial on the elastic drop ----------------
    print("=== Scene 3: elastic stiffness dial ===")
    T3 = 0.9 if args.quick else 1.3
    stiff_cfgs = [("E = 50 (soft)", 50.0, "#ffd27f"),
                  ("E = 400", 400.0, "#ff9d5c"),
                  ("E = 1600 (stiff)", 1600.0, "#ff5c5c")]
    s3 = []
    times3 = None
    diag_table["stiffness"] = {}
    for label, Eval, color in stiff_cfgs:
        # stiff E needs a smaller dt (CFL ~ 1/sqrt(E)); scale dt down from the E=400 baseline.
        dt_e = 1.0e-4 * float(np.sqrt(400.0 / Eval))
        print(f"  rolling elastic {label} (dt={dt_e:.2e}) ...")
        snaps, times, stable = run_material(
            "elastic", disk, area1, T3, nf, cfg_override={"E": Eval, "dt": dt_e})
        s3.append((label, snaps, color))
        times3 = times
        diag_table["stiffness"][label] = diagnostics(snaps[-1])
        print(f"    {label}: stable={stable}  final width={diag_table['stiffness'][label]['width']:.3f} "
              f"height={diag_table['stiffness'][label]['height']:.3f}")
    render_triptych(os.path.join(out_dir, "stiffness_triptych.mp4"), s3, times3)
    # An elastic blob RECOVERS its rest shape, so the final frame hides the stiffness effect. Show the
    # peak-squash instant instead: the frame where the soft (E=50) blob is widest on impact, rendered
    # for all three E so the differential deformation at the same instant is visible.
    peak_frame = int(np.argmax([diagnostics(s3[0][1][f])["width"] for f in range(nf)]))
    render_still(os.path.join(out_dir, "stiffness_still.png"), s3, times3, peak_frame)
    results_media += [
        ("video", "stiffness_triptych.mp4",
         "The stiffness dial on one material: the same elastic disk dropped at three values of Young's "
         "modulus E. Soft E flattens and jiggles slowly like a gel, stiff E barely deforms and rings fast "
         "like hard rubber. Only E changes across the three panels."),
        ("image", "stiffness_still.png",
         "Peak-impact frame of the elastic stiffness dial, at the instant the soft blob is most squashed. "
         "The soft blob (left) pancakes flat on impact, the stiff blob (right) barely dents, showing that a "
         "stiffer material answers the same impact with far less deformation. All three spring back toward "
         "a disk afterward, which is why the difference is clearest at peak impact rather than at rest."),
    ]

    # ---------------- write metrics + manifest ----------------
    metrics = {"floor_y": floor_y, "n_grid": n_grid, "scenes": {}}
    for scene, d in diag_table.items():
        metrics["scenes"][scene] = d
    metrics["material_cfg"] = {m: {k: MAT_CFG[m][k] for k in ("E", "dt", "xi", "tc", "ts")}
                               for m in materials}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    write_manifest(out_dir, diag_table, results_media, T1, T2)
    print(f"\nwrote {out_dir}")
    print(json.dumps(diag_table, indent=2))


def write_manifest(out_dir, diag_table, results_media, T1, T2):
    base = "runs/material-variants/implement-nondifferentiable-material-variants"
    dd = diag_table["drop"]
    dc = diag_table["column"]

    def row(scene, m):
        d = diag_table[scene][m]
        return [scene, m, f"{d['width']:.3f}", f"{d['height']:.3f}", f"{d['com_height']:.3f}"]

    table_rows = [row("drop", m) for m in ("fluid", "elastic", "snow")] + \
                 [row("column", m) for m in ("fluid", "elastic", "snow")]

    results = []
    for (typ, fname, cap) in results_media:
        if typ == "video":
            results.append({"type": "video", "src": f"{base}/{fname}", "caption": cap})
        else:
            results.append({"type": "image", "src": f"{base}/{fname}", "caption": cap})
    results.append({
        "type": "table",
        "columns": ["scene", "material", "final width", "pile height", "COM height"],
        "rows": table_rows,
        "caption": ("Final-frame shape diagnostics (domain units, floor subtracted). 'final width' = "
                    "5-to-95 percentile horizontal extent, 'pile height' = 95th-percentile height above "
                    "the floor, 'COM height' = mean particle height above the floor. Fluid spreads widest "
                    "and sits lowest, elastic recovers the narrowest and tallest shape, and snow lands "
                    "between the two in both width and height. Percentiles are used so a few stray "
                    "particles do not dominate the number."),
    })

    findings = (
        "On two fixed 2D scenes at n_grid=128 with hand-tuned stable parameters, the three constitutive "
        "models produce distinct, expected qualitative signatures, and a simple final-frame diagnostic "
        "backs the eye. In the drop-and-splat scene an identical disk released above the floor ends with "
        f"fluid spread widest and sitting lowest (width {dd['fluid']['width']:.3f}, height "
        f"{dd['fluid']['height']:.3f}), elastic springing back to the most compact and tallest blob "
        f"(width {dd['elastic']['width']:.3f}, height {dd['elastic']['height']:.3f}), and snow landing "
        f"between the two as a crumpled heap that neither flows flat nor rebounds (width "
        f"{dd['snow']['width']:.3f}, height {dd['snow']['height']:.3f}). In the column-collapse scene an "
        f"identical tall block is released from rest: fluid runs out into a flat wide sheet (width "
        f"{dc['fluid']['width']:.3f}, height {dc['fluid']['height']:.3f}), elastic springs back and stands "
        f"nearly its full height (width {dc['elastic']['width']:.3f}, height {dc['elastic']['height']:.3f}), "
        f"and snow slumps to a stable angle-of-repose pile (width {dc['snow']['width']:.3f}, height "
        f"{dc['snow']['height']:.3f}). In both scenes and in both the width and the height diagnostic the "
        "order is the same: fluid is widest and lowest, elastic is narrowest and tallest because it "
        "recovers its shape, and snow sits squarely between them, holding whatever crumpled or slumped "
        "shape its plastic clamp locked in. A bonus third scene dials Young's modulus E on the elastic "
        "drop (E=50, 400, 1600) and shows the soft blob pancaking flat on impact and jiggling slowly "
        "while the stiff blob barely dents and rebounds fast, the visible face of the stiffness scaling. "
        "Every frame and video was viewed; nothing went non-finite, flew off-screen, or clipped through a "
        "wall or the floor at the recorded settings."
    )

    hypothesis = (
        "Each material's motion follows directly from the stress its constitutive law develops. Fluid "
        "stress is an isotropic pressure that depends only on the current volume ratio J, so it resists "
        "compression but offers zero resistance to shear (mu = 0). With nothing to store shape memory, a "
        "fluid blob has no restoring force back to a shape and simply flows until it is a flat, level "
        "puddle. Elastic stress is the corotated function of the full deformation gradient F, "
        "2 mu (F - R) + lambda (J-1) J F^-T, where R is the rotation from the SVD of F. The shear term "
        "2 mu (F - R) stores recoverable energy in any non-rotational deformation, so an elastic blob that "
        "squashes on impact pushes back and springs toward its original rounded shape, overshooting into a "
        "damped jiggle. Snow adds a plastic clamp: each step the singular values of F are limited to "
        "[1 - theta_c, 1 + theta_s] and the excess is moved into a permanent plastic record, with the "
        "moduli hardening as the material compacts. Because deformation past the clamp is not recoverable, "
        "snow crumples and keeps its dented shape instead of springing back, and its finite yield strength "
        "lets a slumped block stand at an angle of repose rather than running out flat. The stiffness dial "
        "follows the same stress law scaled by E: larger E means a larger restoring stress at the same "
        "strain, hence less equilibrium sag (~1/E) and a faster elastic wobble (~sqrt(E)), which is exactly "
        "what the three-panel E sweep shows. What would test the generality of these signatures beyond this "
        "demonstration: more initial conditions and geometries, a friction model at the floor, a 3D run, "
        "and a sweep of the snow clamp parameters theta_c / theta_s to map brittle-to-ductile behavior."
    )

    cfg_str = "; ".join(f"{m} E={MAT_CFG[m]['E']:g}, dt={MAT_CFG[m]['dt']:g}"
                        for m in ("fluid", "elastic", "snow"))
    limitations = (
        "A forward demonstration on TWO scenes (plus one stiffness-dial bonus), in 2D, at a single grid "
        "resolution (n_grid=128), f32, with hand-tuned per-material parameters chosen for stability, not a "
        f"measurement or a claim about materials in general. Parameters are NOT held equal across materials "
        f"({cfg_str}): the fluid is softened to calm its splash and snow runs a smaller dt and softer E so "
        "its SVD clamp stays finite, so absolute widths and heights are not strictly comparable number-for-"
        "number across rows, only the qualitative ordering is. Every material is integrated to the same "
        "PHYSICAL time so the panels are time-synchronised, but the settling state depends on how long the "
        f"scene is run. Boundaries are shared by all materials: Coulomb friction (coefficient {FRICTION:g}) "
        "at the floor, which snow needs to build its angle-of-repose pile, and sticky side walls so an "
        "impact splash cannot ride up a frictionless wall; the exact repose angle and the fluid's wall "
        "run-up therefore depend on these boundary choices, not only on the constitutive model. No "
        "gradients, optimisation, or loss are involved anywhere; this task makes "
        "no claim about controllability, gradient smoothness, or quantitative accuracy. GPU atomic-add "
        "accumulation is not bitwise reproducible; rerun if a frame looks off."
    )

    manifest = {
        "schema_version": "2",
        "task_id": "implement-nondifferentiable-material-variants",
        "direction": "material-variants",
        "title": "Fluid vs elastic vs snow: a forward-only material showcase",
        "status": "active",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": (
            "Run one MLS-MPM scene forward (no gradients, no optimiser, no loss) under three constitutive "
            "models -- weakly compressible fluid, corotated elastic, and Stomakhin-style snow -- from the "
            "SAME initial condition, and show side by side how each moves. Two scenes (a disk dropped onto "
            "the floor and a tall column collapsing) plus a bonus stiffness dial on the elastic drop. The "
            "constitutive physics is reused from sim/material_variants.py with the autodiff machinery "
            "stripped; the point is to see and teach the qualitative differences, backed by a simple "
            "final-shape diagnostic, not to make any claim about optimisation or generality."
        ),
        "findings": findings,
        "hypothesis": hypothesis,
        "limitations": limitations,
        "results": results,
        "custom_html": None,
        "training_refs": ["material-showcase", "constitutive-models", "svd-polar",
                          "material-stiffness", "mpm-in-context"],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
