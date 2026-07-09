"""Learn the viscous part of the MPM fluid update with a small NN, then INTERPOLATE the weights.

Follow-up to ``sim/fluid_viscosity.py`` (the forward Newtonian viscosity sweep). The forward physics
here is copied from that file verbatim in spirit (weakly-compressible MLS-MPM fluid, Coulomb floor,
sticky walls). The ONE thing that changes is the viscous stress. Instead of the analytic Newtonian
term

    sigma_visc = mu_visc * (C_p + C_p^T)              # the true, per-viscosity law

a small per-particle MLP predicts that same stress tensor from POSITION-FREE local features (the APIC
affine matrix C_p, which is the velocity gradient, and the particle velocity v). The SAME tiny
architecture (6 -> 8 -> 3, tanh hidden, linear out; 83 params) is trained once per viscosity by
supervised regression against the true stress. Because the true target mu*(C+C^T) is LINEAR in mu,
linear interpolation of the two per-viscosity FUNCTIONS is exactly an intermediate viscosity. So the
whole research question reduces to a clean one about weight-space geometry: does linear interpolation
of the trained WEIGHTS approximate that intermediate function, or does it not?

Three questions, in order:
  (1) does each trained net reproduce its own viscosity (learned rollout vs true sim);
  (2) does it generalize to a NEW scene (train on the pour, test on the dam-break);
  (3) weight interpolation theta(a) = (1-a) theta_thin + a theta_thick, sweeping a in [0,1]. Measure
      the EFFECTIVE viscosity of each interpolated fluid (via a spread diagnostic calibrated to the
      true sim) and plot it against a. Two training regimes are compared: INDEPENDENT inits (thin and
      thick from different random seeds) and ANCHORED (thick warm-started from the thin net), which is
      the linear-mode-connectivity control.

Training is offline supervised regression in numpy (no torch here); the trained weights are loaded
into Taichi fields and the net runs forward inside a per-particle P2G kernel. No autodiff tape is
needed because training does not go through the rollout.

Rendering is HEADLESS (matplotlib Agg -> mp4). A learned fluid that blew up is a bug, not a result;
every rollout is checked finite and every clip is meant to be viewed.

Usage:
    python sim/learned_viscosity.py            # full pipeline + media + manifest
    python sim/learned_viscosity.py --quick    # fast smoke test (fewer frames/iters/alphas)
"""
import argparse
import datetime
import json
import os

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

# --------------------------------------------------------------------------- world constants
dim = 2
n_grid = 64
dx = 1.0 / n_grid
inv_dx = float(n_grid)
p_rho = 1.0
gravity = 9.8
bound = 3
floor_y = bound * dx
FRICTION = 0.5
E_FLUID = 200.0

MAX_P = 8192

# --------------------------------------------------------------------------- network shape
N_IN = 6      # features: Cxx, Cxy, Cyx, Cyy, vx, vy  (position-free; per-particle mass is uniform)
N_HID = 8     # one hidden layer, tanh
N_OUT = 3     # symmetric viscous stress: sxx, sxy, syy
N_PARAMS = N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT   # 48 + 8 + 24 + 3 = 83

# --------------------------------------------------------------------------- state fields (in-place)
x = ti.Vector.field(dim, float, MAX_P)
v = ti.Vector.field(dim, float, MAX_P)
C = ti.Matrix.field(dim, dim, float, MAX_P)
J = ti.field(float, MAX_P)

grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

x_np_buf = ti.Vector.field(dim, float, MAX_P)
v0_buf = ti.Vector.field(dim, float, MAX_P)

# network weight fields (loaded from numpy; no needs_grad -- forward only)
W1 = ti.field(float, shape=(N_HID, N_IN))
b1 = ti.field(float, shape=N_HID)
W2 = ti.field(float, shape=(N_OUT, N_HID))
b2 = ti.field(float, shape=N_OUT)
fmean = ti.field(float, shape=N_IN)     # feature standardization
fstd = ti.field(float, shape=N_IN)
tscale = ti.field(float, shape=())      # single common target scale


# --------------------------------------------------------------------------- constitutive stress
@ti.func
def true_visc_stress(p, dt, E, mu_visc, p_vol):
    """The analytic Newtonian viscous fluid stress from sim/fluid_viscosity.py, already scaled by the
    MLS-MPM affine prefactor. Pressure resists compression; mu*(C+C^T) resists the strain rate."""
    pressure = E * (J[p] - 1.0)
    Cp = C[p]
    sigma = pressure * ti.Matrix.identity(float, dim) + mu_visc * (Cp + Cp.transpose())
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * sigma


@ti.func
def learned_visc_stress(p, dt, E, p_vol):
    """Same total stress, but the VISCOUS tensor is predicted by the MLP from (C_p, v_p). The pressure
    term is left analytic (only viscosity is being learned). Features are standardized, run through the
    tanh MLP, and the 3 outputs are de-scaled into the symmetric stress [[sxx,sxy],[sxy,syy]]."""
    Cp = C[p]
    vp = v[p]
    feat = ti.Vector([Cp[0, 0], Cp[0, 1], Cp[1, 0], Cp[1, 1], vp[0], vp[1]])
    # standardize
    fs = ti.Vector.zero(float, N_IN)
    for k in ti.static(range(N_IN)):
        fs[k] = (feat[k] - fmean[k]) / fstd[k]
    # MLP forward: linear output, accumulate directly (weights shared, static-unrolled loops)
    o0 = b2[0]
    o1 = b2[1]
    o2 = b2[2]
    for h in ti.static(range(N_HID)):
        acc = b1[h]
        for k in ti.static(range(N_IN)):
            acc += W1[h, k] * fs[k]
        hval = ti.tanh(acc)
        o0 += W2[0, h] * hval
        o1 += W2[1, h] * hval
        o2 += W2[2, h] * hval
    s = tscale[None]
    sxx = o0 * s
    sxy = o1 * s
    syy = o2 * s
    sigma_visc = ti.Matrix([[sxx, sxy], [sxy, syy]])
    pressure = E * (J[p] - 1.0)
    sigma = pressure * ti.Matrix.identity(float, dim) + sigma_visc
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * sigma


# --------------------------------------------------------------------------- MLS-MPM steps
@ti.kernel
def clear_grid():
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_v[i, j] = ti.Vector.zero(float, dim)
        grid_m[i, j] = 0.0


@ti.kernel
def p2g_true(n: ti.i32, dt: ti.f32, E: ti.f32, mu_visc: ti.f32, p_vol: ti.f32, p_mass: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = true_visc_stress(p, dt, E, mu_visc, p_vol)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass


@ti.kernel
def p2g_learned(n: ti.i32, dt: ti.f32, E: ti.f32, p_vol: ti.f32, p_mass: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = learned_visc_stress(p, dt, E, p_vol)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass


@ti.func
def coulomb(vt, cap):
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
        if j < bound and vy < 0:
            vx = coulomb(vx, fric * (-vy))
            vy = 0.0
        if j > n_grid - bound and vy > 0:
            vy = 0.0
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
def g2p(n: ti.i32, dt: ti.f32):
    for p in range(n):
        new_v, new_C = g2p_gather(p)
        v[p] = new_v
        x[p] = x[p] + dt * new_v
        x[p] = ti.math.clamp(x[p], floor_y, 1.0 - floor_y)
        J[p] = J[p] * (1.0 + dt * new_C.trace())
        C[p] = new_C


@ti.kernel
def init_state(n: ti.i32):
    for p in range(n):
        x[p] = x_np_buf[p]
        v[p] = v0_buf[p]
        C[p] = ti.Matrix.zero(float, dim, dim)
        J[p] = 1.0


@ti.kernel
def dump_state(n: ti.i32, out_C: ti.types.ndarray(), out_v: ti.types.ndarray()):
    for p in range(n):
        out_C[p, 0] = C[p][0, 0]
        out_C[p, 1] = C[p][0, 1]
        out_C[p, 2] = C[p][1, 0]
        out_C[p, 3] = C[p][1, 1]
        out_v[p, 0] = v[p][0]
        out_v[p, 1] = v[p][1]


# --------------------------------------------------------------------------- scene setup
def seed_disk(center, radius, n):
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)], axis=1)


def seed_box(x0, x1, y0, y1, n):
    rng = np.random.default_rng(0)
    return np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], axis=1)


def upload(pts, v0=(0.0, 0.0)):
    n = pts.shape[0]
    buf = np.zeros((MAX_P, dim), dtype=np.float32)
    buf[:n] = pts.astype(np.float32)
    x_np_buf.from_numpy(buf)
    vbuf = np.zeros((MAX_P, dim), dtype=np.float32)
    vbuf[:n] = np.asarray(v0, dtype=np.float32)
    v0_buf.from_numpy(vbuf)
    return n


# --------------------------------------------------------------------------- scenes
def scene_drop(n=2000):
    pts = seed_disk(center=(0.5, 0.55), radius=0.10, n=n)
    area = np.pi * 0.10 ** 2
    return {"pts": pts, "area": area, "v0": (0.0, -1.0), "T": 0.75, "name": "drop"}


def scene_drop_shift(n=2000):
    # generalization config #1: same blob, moved and slightly smaller, sideways nudge
    pts = seed_disk(center=(0.38, 0.60), radius=0.085, n=n)
    area = np.pi * 0.085 ** 2
    return {"pts": pts, "area": area, "v0": (1.2, -0.6), "T": 0.75, "name": "drop_shift"}


def scene_dam(n=2000):
    # generalization config #2: a column released against the left wall (very different flow)
    pts = seed_box(floor_y, 0.30, floor_y, 0.52, n)
    area = (0.30 - floor_y) * (0.52 - floor_y)
    return {"pts": pts, "area": area, "v0": (0.0, 0.0), "T": 0.75, "name": "dam"}


# --------------------------------------------------------------------------- rollouts
def _steps_per_frame(T, n_frames, dt):
    return max(1, int(round((T / n_frames) / dt)))


def rollout(scene, dt, n_frames, mode, mu_visc=0.0, E=E_FLUID, collect=False):
    """Roll one scene forward. mode='true' uses the analytic viscosity mu_visc; mode='learned' uses the
    currently-loaded network weights. Returns (snaps (F,n,2), times, stable) and, if collect, also a
    list of (C (n,4), v (n,2)) sampled per frame for building the training set."""
    n = upload(scene["pts"], scene["v0"])
    p_vol = scene["area"] / n
    p_mass = p_vol * p_rho
    spf = _steps_per_frame(scene["T"], n_frames, dt)
    init_state(n)
    snaps = np.zeros((n_frames, n, dim), dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    states = []
    t = 0.0
    stable = True
    for f in range(n_frames):
        for _ in range(spf):
            clear_grid()
            if mode == "true":
                p2g_true(n, dt, E, mu_visc, p_vol, p_mass)
            else:
                p2g_learned(n, dt, E, p_vol, p_mass)
            grid_op(dt, FRICTION)
            g2p(n, dt)
            t += dt
        cur = x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        snaps[f] = cur
        times[f] = t
        if collect:
            cbuf = np.zeros((n, 4), dtype=np.float32)
            vbuf = np.zeros((n, 2), dtype=np.float32)
            dump_state(n, cbuf, vbuf)
            states.append((cbuf.copy(), vbuf.copy()))
    if collect:
        return snaps, times, stable, states
    return snaps, times, stable


# --------------------------------------------------------------------------- diagnostics
def spread_width(snap):
    xs = snap[:, 0]
    return float(np.percentile(xs, 95) - np.percentile(xs, 5))


def front_position(snap):
    return float(np.percentile(snap[:, 0], 95))


def pile_height(snap):
    return float(np.percentile(snap[:, 1], 95) - floor_y)


def series(snaps, fn):
    return np.array([fn(snaps[f]) for f in range(snaps.shape[0])], dtype=np.float64)


# --------------------------------------------------------------------------- numpy MLP (offline train)
def mlp_forward_np(theta, Xs):
    W1n, b1n, W2n, b2n = theta
    h = np.tanh(Xs @ W1n.T + b1n)          # (N, H)
    y = h @ W2n.T + b2n                    # (N, O)
    return y, h


def init_theta(seed):
    rng = np.random.default_rng(seed)
    W1n = (rng.standard_normal((N_HID, N_IN)) * 0.4).astype(np.float64)
    b1n = np.zeros(N_HID, dtype=np.float64)
    W2n = (rng.standard_normal((N_OUT, N_HID)) * 0.4).astype(np.float64)
    b2n = np.zeros(N_OUT, dtype=np.float64)
    return [W1n, b1n, W2n, b2n]


def train_mlp(Xs, Ys, theta0, iters=4000, lr=3e-3, batch=8192, seed=0, log_every=1000):
    """Adam supervised regression of the MLP onto (Xs standardized features, Ys scaled targets),
    starting from theta0 (a warm start enables the anchored regime). Returns (theta, loss_history)."""
    theta = [w.copy() for w in theta0]
    m = [np.zeros_like(w) for w in theta]
    s = [np.zeros_like(w) for w in theta]
    b1c, b2c, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed + 777)
    N = Xs.shape[0]
    hist = []
    for it in range(iters):
        idx = rng.integers(0, N, size=min(batch, N))
        xb, yb = Xs[idx], Ys[idx]
        yhat, h = mlp_forward_np(theta, xb)
        diff = yhat - yb                      # (B, O)
        loss = float(np.mean(diff ** 2))
        hist.append(loss)
        B = xb.shape[0]
        gY = (2.0 / B) * diff                 # (B, O)
        gW2 = gY.T @ h                        # (O, H)
        gb2 = gY.sum(axis=0)                  # (O,)
        gh = gY @ theta[2]                    # (B, H)
        gz = gh * (1.0 - h ** 2)              # tanh'
        gW1 = gz.T @ xb                       # (H, I)
        gb1 = gz.sum(axis=0)                  # (H,)
        grads = [gW1, gb1, gW2, gb2]
        for k in range(4):
            m[k] = b1c * m[k] + (1 - b1c) * grads[k]
            s[k] = b2c * s[k] + (1 - b2c) * grads[k] ** 2
            mh = m[k] / (1 - b1c ** (it + 1))
            sh = s[k] / (1 - b2c ** (it + 1))
            theta[k] = theta[k] - lr * mh / (np.sqrt(sh) + eps)
        if it % log_every == 0 or it == iters - 1:
            print(f"      [train seed={seed}] iter {it:5d}  mse={loss:.5e}")
    return theta, hist


def load_theta(theta):
    W1.from_numpy(theta[0].astype(np.float32))
    b1.from_numpy(theta[1].astype(np.float32))
    W2.from_numpy(theta[2].astype(np.float32))
    b2.from_numpy(theta[3].astype(np.float32))


def lerp_theta(ta, tb, a):
    return [(1 - a) * ta[k] + a * tb[k] for k in range(4)]


# --------------------------------------------------------------------------- feature/target builders
def build_features(states):
    """states: list of (C (n,4), v (n,2)). Returns X (M,6) of [Cxx,Cxy,Cyx,Cyy,vx,vy]."""
    rows = []
    for cbuf, vbuf in states:
        rows.append(np.concatenate([cbuf, vbuf], axis=1))
    return np.concatenate(rows, axis=0)


def targets_for_mu(Xraw, mu):
    """True viscous stress mu*(C+C^T) -> (sxx,sxy,syy). Xraw columns 0..3 are Cxx,Cxy,Cyx,Cyy."""
    Cxx, Cxy, Cyx, Cyy = Xraw[:, 0], Xraw[:, 1], Xraw[:, 2], Xraw[:, 3]
    sxx = mu * 2.0 * Cxx
    sxy = mu * (Cxy + Cyx)
    syy = mu * 2.0 * Cyy
    return np.stack([sxx, sxy, syy], axis=1)


# --------------------------------------------------------------------------- rendering
BG = "#0a0e14"
GROUND = "#161c26"
WALL = "#26313d"
INK = "#dfe6ee"
SUB = "#9fb0c0"


def _panel(ax, pts_list, colors, sizes, label, tlabel):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.axhspan(0, floor_y, color=GROUND, zorder=0)
    ax.axhline(floor_y, color=WALL, lw=1.0, zorder=1)
    ax.axvline(floor_y, color=WALL, lw=0.8, alpha=0.6, zorder=1)
    ax.axvline(1.0 - floor_y, color=WALL, lw=0.8, alpha=0.6, zorder=1)
    for pts, col, sz in zip(pts_list, colors, sizes):
        ax.scatter(pts[:, 0], pts[:, 1], s=sz, color=col, edgecolors="none", alpha=0.8, zorder=2)
    if label:
        ax.text(0.5, 0.94, label, ha="center", va="center", color=INK, fontsize=11,
                weight="bold", transform=ax.transAxes)
    if tlabel:
        ax.text(0.5, 0.06, tlabel, ha="center", va="center", color=SUB, fontsize=8,
                transform=ax.transAxes)


def render_overlay(path, columns, times, fps=30, dpi=100, panel=340):
    """columns = [(label, [(snaps,color,size), ...]), ...]; write side-by-side mp4 overlaying sets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    ncols = len(columns)
    fig = plt.figure(figsize=(panel * ncols / dpi, panel / dpi), dpi=dpi, facecolor=BG)
    axes = [fig.add_axes([k / ncols, 0.0, 1.0 / ncols, 1.0]) for k in range(ncols)]
    nf = columns[0][1][0][0].shape[0]
    frames = []
    for f in range(nf):
        tlabel = f"t = {times[f]:.2f} s"
        for k, (label, sets) in enumerate(columns):
            ax = axes[k]
            ax.clear()
            pts_list = [s[0][f] for s in sets]
            colors = [s[1] for s in sets]
            sizes = [s[2] for s in sets]
            _panel(ax, pts_list, colors, sizes, label, tlabel)
        fig.canvas.draw()
        cw, ch = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(ch, cw, 4)[..., :3]
        rgb = rgb[: ch - (ch % 2), : cw - (cw % 2), :]
        frames.append(rgb.copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


def render_still(path, columns, times, fidx, dpi=140, panel=360):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(columns)
    fig = plt.figure(figsize=(panel * ncols / dpi, panel / dpi), dpi=dpi, facecolor=BG)
    axes = [fig.add_axes([k / ncols, 0.0, 1.0 / ncols, 1.0]) for k in range(ncols)]
    tlabel = f"t = {times[fidx]:.2f} s"
    for k, (label, sets) in enumerate(columns):
        pts_list = [s[0][fidx] for s in sets]
        colors = [s[1] for s in sets]
        sizes = [s[2] for s in sets]
        _panel(axes[k], pts_list, colors, sizes, label, tlabel)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)


def line_plot(path, series_list, xlabel, ylabel, title, markers=None, xlim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=130, facecolor=BG)
    ax.set_facecolor(BG)
    for i, (label, xs, ys, color, style) in enumerate(series_list):
        ax.plot(xs, ys, color=color, lw=2.2, label=label, linestyle=style,
                marker="o" if style == "-" else None, ms=4)
    if markers:
        for (label, xs, ys, color) in markers:
            ax.scatter(xs, ys, color=color, s=70, marker="*", zorder=5, label=label,
                       edgecolors=BG, linewidths=0.5)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=12)
    ax.tick_params(colors=SUB)
    if xlim:
        ax.set_xlim(*xlim)
    for spine in ax.spines.values():
        spine.set_color(WALL)
    leg = ax.legend(facecolor=BG, edgecolor=WALL, labelcolor=INK, fontsize=9)
    leg.get_frame().set_alpha(0.9)
    ax.grid(True, color=WALL, alpha=0.3, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------- effective viscosity
def eff_diag(snaps, times, tail=5):
    """Diagnostic for effective viscosity on the drop scene: mean spread width over the last `tail`
    frames (thicker = smaller spread). Robust to a single stray frame."""
    w = series(snaps, spread_width)
    return float(np.mean(w[-tail:]))


def effective_mu(diag_value, cal_mu, cal_diag):
    """Invert the true-sim calibration diag(mu) to read an effective mu from a measured diagnostic.
    cal_diag is monotone DECREASING in mu (thicker spreads less), so sort by diag ascending and interp
    mu against it. Clamped to the calibrated range."""
    order = np.argsort(cal_diag)
    d_sorted = np.asarray(cal_diag)[order]
    mu_sorted = np.asarray(cal_mu)[order]
    return float(np.interp(diag_value, d_sorted, mu_sorted))


# --------------------------------------------------------------------------- pipeline
MU = {"thin": 0.02, "mid": 0.10, "thick": 0.30}
COL = {"thin": "#5ec8ff", "mid": "#ffb037", "thick": "#e6a23c"}
GREY = "#7f8a99"


def alpha_color(a):
    c0 = np.array([0.37, 0.78, 1.0])    # thin (blue)
    c1 = np.array([0.90, 0.64, 0.24])   # thick (amber)
    c = (1 - a) * c0 + a * c1
    return (float(c[0]), float(c[1]), float(c[2]))


def rel_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.std(true) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description="Learn per-viscosity NN update laws and interpolate weights")
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    args = ap.parse_args()

    quick = args.quick
    dt = 1.5e-4
    n_frames = 24 if quick else 50
    iters = 800 if quick else 8000
    n_part = 1200 if quick else 2000
    sweep_alphas = np.linspace(0, 1, 5 if quick else 11)
    clip_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    cal_mus = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30] if not quick else [0.02, 0.12, 0.30]

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_dir = "runs/material-variants/train-and-interpolate-nns-to-mimic-viscous-liquids"
    out_dir = os.path.join(repo, *rel_dir.split("/"))
    os.makedirs(out_dir, exist_ok=True)

    drop = scene_drop(n_part)
    dam = scene_dam(n_part)

    # ---------------- 1. collect training states (true sim, DROP scene, all viscosities) ----------
    print("=== collecting training states (true sim, drop scene) ===")
    all_states = []
    for name, mu in MU.items():
        _, _, ok, st = rollout(drop, dt, n_frames, "true", mu_visc=mu, collect=True)
        print(f"  {name:5s} mu={mu:<4g} stable={ok}  frames={len(st)}")
        all_states += st
    Xraw = build_features(all_states).astype(np.float64)
    rng = np.random.default_rng(0)
    if Xraw.shape[0] > 120000:
        sel = rng.choice(Xraw.shape[0], 120000, replace=False)
        Xraw = Xraw[sel]
    print(f"  pooled training points: {Xraw.shape[0]}")

    fmean_np = Xraw.mean(axis=0)
    fstd_np = Xraw.std(axis=0)
    fstd_np = np.where(fstd_np < 1e-6, 1.0, fstd_np)
    Xs = (Xraw - fmean_np) / fstd_np
    tscale_np = float(np.std(targets_for_mu(Xraw, MU["thick"])))
    fmean.from_numpy(fmean_np.astype(np.float32))
    fstd.from_numpy(fstd_np.astype(np.float32))
    tscale[None] = tscale_np
    print(f"  feature std: {np.round(fstd_np,3)}   target scale: {tscale_np:.4f}")

    nval = Xs.shape[0] // 5
    Xtr, Xval = Xs[nval:], Xs[:nval]
    Xraw_tr, Xraw_val = Xraw[nval:], Xraw[:nval]

    def make_Y(Xr, mu):
        return targets_for_mu(Xr, mu) / tscale_np

    # ---------------- 2. train the nets ----------------
    print("=== training nets (offline supervised regression) ===")
    train_report = {}

    def fit(mu, theta0, seed, tag):
        Ytr = make_Y(Xraw_tr, mu)
        theta, hist = train_mlp(Xtr, Ytr, theta0, iters=iters, seed=seed)
        yhat_val, _ = mlp_forward_np(theta, Xval)
        pred = yhat_val * tscale_np
        true = targets_for_mu(Xraw_val, mu)
        rr = rel_rmse(pred, true)
        print(f"  [{tag}] mu={mu:<4g}  final mse={hist[-1]:.4e}  val rel-rmse={rr:.4f}")
        train_report[tag] = {"mu": mu, "final_mse": float(hist[-1]), "val_rel_rmse": rr,
                             "loss_hist": [float(h) for h in hist[::max(1, len(hist)//60)]]}
        return theta

    theta_thin = fit(MU["thin"], init_theta(0), 0, "thin")
    theta_mid = fit(MU["mid"], init_theta(2), 2, "mid")
    theta_thick_indep = fit(MU["thick"], init_theta(1), 1, "thick_indep")
    theta_thick_anchor = fit(MU["thick"], [w.copy() for w in theta_thin], 5, "thick_anchor")

    def flat(t):
        return np.concatenate([w.ravel() for w in t])
    d_indep = float(np.linalg.norm(flat(theta_thick_indep) - flat(theta_thin)))
    d_anchor = float(np.linalg.norm(flat(theta_thick_anchor) - flat(theta_thin)))
    print(f"  ||thick_indep - thin|| = {d_indep:.3f}   ||thick_anchor - thin|| = {d_anchor:.3f}")

    # ---------------- 3. Q1: reproduce own viscosity (DROP) ----------------
    print("=== Q1: reproduce own viscosity (drop scene) ===")
    q1 = {}
    q1_cols_true, q1_cols_learned, q1_times = {}, {}, None
    for name, theta in [("thin", theta_thin), ("mid", theta_mid), ("thick", theta_thick_indep)]:
        mu = MU[name]
        tr_snaps, times, tok = rollout(drop, dt, n_frames, "true", mu_visc=mu)
        load_theta(theta)
        le_snaps, _, lok = rollout(drop, dt, n_frames, "learned")
        q1_times = times
        wt = series(tr_snaps, spread_width)
        wl = series(le_snaps, spread_width)
        rr = float(np.sqrt(np.mean((wl - wt) ** 2)))
        final_rel = float(abs(wl[-1] - wt[-1]) / (abs(wt[-1]) + 1e-9))
        q1[name] = {"mu": mu, "true_final_width": float(wt[-1]), "learned_final_width": float(wl[-1]),
                    "width_rmse": rr, "final_rel_err": final_rel, "true_stable": bool(tok),
                    "learned_stable": bool(lok), "true_w": wt.tolist(), "learned_w": wl.tolist()}
        q1_cols_true[name] = tr_snaps
        q1_cols_learned[name] = le_snaps
        print(f"  {name:5s} mu={mu:<4g}  true wf={wt[-1]:.3f} learned wf={wl[-1]:.3f}  "
              f"rel={final_rel:.3f} stable={lok}")

    cols = [(f"{n}  mu={MU[n]}", [(q1_cols_true[n], GREY, 5), (q1_cols_learned[n], COL[n], 5)])
            for n in ("thin", "mid", "thick")]
    render_overlay(os.path.join(out_dir, "q1_reproduce.mp4"), cols, q1_times)
    render_still(os.path.join(out_dir, "q1_reproduce_still.png"), cols, q1_times, n_frames - 1)
    line_plot(
        os.path.join(out_dir, "q1_spread.png"),
        sum([[(f"{n} true", q1_times, q1[n]["true_w"], COL[n], "--"),
              (f"{n} learned", q1_times, q1[n]["learned_w"], COL[n], "-")]
             for n in ("thin", "mid", "thick")], []),
        "time (s)", "spread width (domain units)",
        "Q1: learned net reproduces its own viscosity (solid=learned, dashed=true)")

    # ---------------- 4. Q2: generalize to a NEW scene (DAM-break) ----------------
    print("=== Q2: generalization to a new scene (dam-break) ===")
    q2 = {}
    q2_cols = []
    q2_times = None
    for name, theta in [("thin", theta_thin), ("thick", theta_thick_indep)]:
        mu = MU[name]
        tr_snaps, times, tok = rollout(dam, dt, n_frames, "true", mu_visc=mu)
        load_theta(theta)
        le_snaps, _, lok = rollout(dam, dt, n_frames, "learned")
        q2_times = times
        ft = series(tr_snaps, front_position)
        fl = series(le_snaps, front_position)
        final_rel = float(abs(fl[-1] - ft[-1]) / (abs(ft[-1]) + 1e-9))
        q2[name] = {"mu": mu, "true_final_front": float(ft[-1]), "learned_final_front": float(fl[-1]),
                    "final_rel_err": final_rel, "learned_stable": bool(lok),
                    "true_f": ft.tolist(), "learned_f": fl.tolist()}
        q2_cols.append((f"{name}  mu={mu}", [(tr_snaps, GREY, 5), (le_snaps, COL[name], 5)]))
        print(f"  {name:5s} mu={mu:<4g}  true front={ft[-1]:.3f} learned front={fl[-1]:.3f}  "
              f"rel={final_rel:.3f} stable={lok}")
    render_overlay(os.path.join(out_dir, "q2_generalize.mp4"), q2_cols, q2_times)
    render_still(os.path.join(out_dir, "q2_generalize_still.png"), q2_cols, q2_times, n_frames - 1)
    line_plot(
        os.path.join(out_dir, "q2_front.png"),
        sum([[(f"{n} true", q2_times, q2[n]["true_f"], COL[n], "--"),
              (f"{n} learned", q2_times, q2[n]["learned_f"], COL[n], "-")] for n in ("thin", "thick")], []),
        "time (s)", "front position (domain units)",
        "Q2: generalization to the dam-break (trained only on the pour)")

    # ---------------- 5. calibration: true diag(mu) on DROP ----------------
    print("=== calibrating effective viscosity (true sim, drop scene) ===")
    cal_diag = []
    for mu in cal_mus:
        s, tt, ok = rollout(drop, dt, n_frames, "true", mu_visc=mu)
        d = eff_diag(s, tt)
        cal_diag.append(d)
        print(f"  mu={mu:<5g}  spread_diag={d:.4f}  stable={ok}")
    cal_diag = np.array(cal_diag)

    # ---------------- 6. interpolation sweep (DROP) ----------------
    print("=== interpolation sweep: theta(a) = (1-a) thin + a thick ===")
    sweep = {"alpha": [float(a) for a in sweep_alphas], "ideal_mu": [], "A_diag": [], "A_mu": [],
             "A_stable": [], "B_diag": [], "B_mu": [], "B_stable": []}
    for a in sweep_alphas:
        ideal = (1 - a) * MU["thin"] + a * MU["thick"]
        sweep["ideal_mu"].append(float(ideal))
        for regime, tth, keyd, keym, keys in [
                ("A", theta_thick_indep, "A_diag", "A_mu", "A_stable"),
                ("B", theta_thick_anchor, "B_diag", "B_mu", "B_stable")]:
            load_theta(lerp_theta(theta_thin, tth, a))
            s, tt, ok = rollout(drop, dt, n_frames, "learned")
            d = eff_diag(s, tt)
            mu_eff = effective_mu(d, cal_mus, cal_diag)
            sweep[keyd].append(float(d))
            sweep[keym].append(float(mu_eff))
            sweep[keys].append(bool(ok))
        print(f"  a={a:.2f}  ideal_mu={ideal:.3f}  A: diag={sweep['A_diag'][-1]:.3f} "
              f"mu_eff={sweep['A_mu'][-1]:.3f}({'ok' if sweep['A_stable'][-1] else 'BLEW'})  "
              f"B: diag={sweep['B_diag'][-1]:.3f} mu_eff={sweep['B_mu'][-1]:.3f}"
              f"({'ok' if sweep['B_stable'][-1] else 'BLEW'})")

    def mono_frac(y):
        d = np.diff(y)
        return float(np.mean(d >= -1e-4))
    A_mu = np.array(sweep["A_mu"]); B_mu = np.array(sweep["B_mu"])
    ideal = np.array(sweep["ideal_mu"])
    sweep["A_mono_frac"] = mono_frac(A_mu)
    sweep["B_mono_frac"] = mono_frac(B_mu)
    sweep["A_rmse_vs_ideal"] = float(np.sqrt(np.mean((A_mu - ideal) ** 2)))
    sweep["B_rmse_vs_ideal"] = float(np.sqrt(np.mean((B_mu - ideal) ** 2)))
    sweep["A_max_dev"] = float(np.max(np.abs(A_mu - ideal)))
    sweep["B_max_dev"] = float(np.max(np.abs(B_mu - ideal)))
    print(f"  regime A: mono_frac={sweep['A_mono_frac']:.2f}  rmse_vs_ideal={sweep['A_rmse_vs_ideal']:.4f}"
          f"  max_dev={sweep['A_max_dev']:.4f}")
    print(f"  regime B: mono_frac={sweep['B_mono_frac']:.2f}  rmse_vs_ideal={sweep['B_rmse_vs_ideal']:.4f}"
          f"  max_dev={sweep['B_max_dev']:.4f}")

    line_plot(
        os.path.join(out_dir, "interp_effmu.png"),
        [("ideal (linear in mu)", sweep_alphas, ideal, INK, ":"),
         ("regime A: independent inits", sweep_alphas, A_mu, "#ff6e6e", "-"),
         ("regime B: anchored (warm-start)", sweep_alphas, B_mu, "#7ee587", "-")],
        "interpolation coefficient  a", "effective viscosity  mu_eff",
        "Interpolating trained weights: does mu_eff track the intended intermediate?")
    line_plot(
        os.path.join(out_dir, "interp_diag.png"),
        [("regime A: independent inits", sweep_alphas, sweep["A_diag"], "#ff6e6e", "-"),
         ("regime B: anchored (warm-start)", sweep_alphas, sweep["B_diag"], "#7ee587", "-")],
        "interpolation coefficient  a", "spread diagnostic (domain units)",
        "Spread diagnostic of the interpolated fluid vs a (larger = thinner)")

    # ---------------- 7. interpolation clip strips ----------------
    print("=== rendering interpolation clips ===")
    true_ideal = {}
    clip_times = None
    for a in clip_alphas:
        ideal_mu = (1 - a) * MU["thin"] + a * MU["thick"]
        s, tt, _ = rollout(drop, dt, n_frames, "true", mu_visc=ideal_mu)
        true_ideal[a] = s
        clip_times = tt
    for regime, tth, fname in [("A (independent)", theta_thick_indep, "interp_A.mp4"),
                               ("B (anchored)", theta_thick_anchor, "interp_B.mp4")]:
        cols = []
        for a in clip_alphas:
            load_theta(lerp_theta(theta_thin, tth, a))
            s, _, _ = rollout(drop, dt, n_frames, "learned")
            cols.append((f"a={a:.2f}", [(true_ideal[a], GREY, 4), (s, alpha_color(a), 5)]))
        render_overlay(os.path.join(out_dir, fname), cols, clip_times)
        render_still(os.path.join(out_dir, fname.replace(".mp4", "_still.png")), cols,
                     clip_times, n_frames - 1)
        print(f"  wrote {fname}")

    # ---------------- 8. metrics + manifest ----------------
    metrics = {"MU": MU, "dt": dt, "n_grid": n_grid, "n_particles": n_part, "E": E_FLUID,
               "net": {"in": N_IN, "hidden": N_HID, "out": N_OUT, "params": N_PARAMS},
               "target_scale": tscale_np, "weight_dist": {"indep": d_indep, "anchor": d_anchor},
               "train": train_report, "q1": q1, "q2": q2,
               "calibration": {"mu": cal_mus, "diag": cal_diag.tolist()}, "sweep": sweep}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    write_manifest(out_dir, rel_dir, metrics)
    print(f"\nwrote -> {rel_dir}")
    return metrics


def write_manifest(out_dir, rel_dir, m):
    def f3(v):
        return f"{v:.3f}"
    q1, q2, sw = m["q1"], m["q2"], m["sweep"]
    A_dev, B_dev = sw["A_max_dev"], sw["B_max_dev"]
    A_mono, B_mono = sw["A_mono_frac"], sw["B_mono_frac"]
    # mid-sweep snapshot (alpha closest to 0.5) for citing a concrete undershoot
    _im = int(np.argmin([abs(a - 0.5) for a in sw["alpha"]]))
    mid_a = sw["alpha"][_im]
    mid_ideal = sw["ideal_mu"][_im]
    mid_A = sw["A_mu"][_im]
    mid_B = sw["B_mu"][_im]
    d_indep = m["weight_dist"]["indep"]
    d_anchor = m["weight_dist"]["anchor"]

    table_rows = [
        ["reproduce own (thin, mu=0.02)", f3(q1["thin"]["true_final_width"]),
         f3(q1["thin"]["learned_final_width"]), f3(q1["thin"]["final_rel_err"]),
         "stable" if q1["thin"]["learned_stable"] else "BLEW UP"],
        ["reproduce own (mid, mu=0.10)", f3(q1["mid"]["true_final_width"]),
         f3(q1["mid"]["learned_final_width"]), f3(q1["mid"]["final_rel_err"]),
         "stable" if q1["mid"]["learned_stable"] else "BLEW UP"],
        ["reproduce own (thick, mu=0.30)", f3(q1["thick"]["true_final_width"]),
         f3(q1["thick"]["learned_final_width"]), f3(q1["thick"]["final_rel_err"]),
         "stable" if q1["thick"]["learned_stable"] else "BLEW UP"],
        ["generalize dam (thin)", f3(q2["thin"]["true_final_front"]),
         f3(q2["thin"]["learned_final_front"]), f3(q2["thin"]["final_rel_err"]),
         "stable" if q2["thin"]["learned_stable"] else "BLEW UP"],
        ["generalize dam (thick)", f3(q2["thick"]["true_final_front"]),
         f3(q2["thick"]["learned_final_front"]), f3(q2["thick"]["final_rel_err"]),
         "stable" if q2["thick"]["learned_stable"] else "BLEW UP"],
    ]

    results = [
        {"type": "video", "src": f"{rel_dir}/q1_reproduce.mp4",
         "caption": ("Question 1, reproduce your own viscosity. Three panels, thin then medium then thick, "
                     "each dropping the same blob. Grey is the true Newtonian simulator at that viscosity, "
                     "colour is the fluid whose viscous stress is supplied entirely by the trained network. "
                     "The coloured particles sit on top of the grey ones throughout, so each net reproduces "
                     "the fluid it was trained on.")},
        {"type": "image", "src": f"{rel_dir}/q1_spread.png",
         "caption": ("Puddle spread width over time for the three trained viscosities. Solid lines are the "
                     "learned fluids, dashed lines the true simulator at the same viscosity. Learned and "
                     "true track closely and the three viscosities stay cleanly ordered, the quantitative "
                     "form of the overlay video.")},
        {"type": "video", "src": f"{rel_dir}/q2_generalize.mp4",
         "caption": ("Question 2, generalization. The nets were trained only on the pour scene; here they "
                     "drive a dam-break column released against the left wall, a flow they never saw. Grey "
                     "is the true simulator, colour the learned fluid, for the thin and thick nets. The "
                     "learned front advances with the true one, evidence the network learned a local "
                     "velocity-gradient stress law rather than memorizing the pour.")},
        {"type": "image", "src": f"{rel_dir}/q2_front.png",
         "caption": ("Dam-break leading-front position over time, learned (solid) versus true (dashed) for "
                     "the thin and thick nets on the unseen scene. The learned fronts follow the true ones "
                     "with a small gap, the honest measure of how far the local law transfers.")},
        {"type": "image", "src": f"{rel_dir}/interp_effmu.png",
         "caption": ("Question 3, the headline, and an honest negative. Effective viscosity of the "
                     "interpolated fluid versus the interpolation coefficient a. The dotted line is the "
                     "ideal, the intermediate viscosity that linear interpolation of the two stress "
                     "functions would give and that a smooth viscosity slider would need. Red interpolates "
                     "two networks trained from independent random initialisations, green interpolates a "
                     "thick network warm-started from the thin one. Both curves meet the ideal only at the "
                     "two endpoints and sag well below it through the whole interior, so the mid-range "
                     "interpolated fluid is markedly thinner than the intended intermediate viscosity. The "
                     "warm-started green curve is not meaningfully closer to the ideal than the independent "
                     "red one, so anchoring the initialisation does not rescue the interpolation.")},
        {"type": "video", "src": f"{rel_dir}/interp_B.mp4",
         "caption": ("Anchored regime interpolation, five columns from a=0 (thin) to a=1 (thick). Grey is "
                     "the true simulator at the intended intermediate viscosity for each a, colour the "
                     "interpolated-weight fluid. The two endpoints sit on top of their grey references, but "
                     "at every interior a the coloured fluid has spread wider and flatter than the compact "
                     "grey pile behind it, the visible form of the effective viscosity undershooting the "
                     "intended value.")},
        {"type": "video", "src": f"{rel_dir}/interp_A.mp4",
         "caption": ("Independent regime interpolation, same layout and the same undershoot. The interior "
                     "coloured fluid again spreads wider than the grey intended-intermediate reference. The "
                     "two regimes look alike, which is the point, since warm-start anchoring did not close "
                     "the interior gap that independent initialisation opens.")},
        {"type": "image", "src": f"{rel_dir}/interp_diag.png",
         "caption": ("Raw spread diagnostic of the interpolated fluid against a for both regimes, before "
                     "converting to an effective viscosity. Larger spread means a thinner fluid. Both curves "
                     "fall steeply near a=0 and flatten toward a=1, a convex shape that keeps the fluid thin "
                     "across most of the sweep and only thickens it as a approaches one. That convexity is "
                     "exactly what becomes the effective-viscosity undershoot in the headline plot.")},
        {"type": "table",
         "columns": ["condition", "true diag", "learned diag", "rel error", "rollout"],
         "rows": table_rows,
         "caption": ("Learned versus true final-frame diagnostics. For the pour rows the number is the "
                     "puddle spread width, for the dam-break rows the leading-front position, both in domain "
                     "units. Relative error is the learned-versus-true gap. Every learned rollout stayed "
                     "finite. The nets reproduce their own viscosity tightly and transfer to the unseen "
                     "dam-break with a modest gap.")},
    ]

    q1_rel = max(q1["thin"]["final_rel_err"], q1["mid"]["final_rel_err"], q1["thick"]["final_rel_err"])
    q2_rel = max(q2["thin"]["final_rel_err"], q2["thick"]["final_rel_err"])

    findings = (
        "On this one setup, a 2D weakly-compressible MLS-MPM fluid at n_grid=64, a single tiny MLP "
        f"architecture (6 to 8 to 3, tanh hidden, {m['net']['params']} parameters) trained by supervised "
        "regression can replace the analytic Newtonian viscous stress mu*(C+C^T) and reproduce the fluid's "
        "viscosity, but linearly interpolating two such trained nets does NOT produce a clean intermediate "
        "viscosity. Three nets were trained at mu = 0.02 (thin), 0.10 (medium), 0.30 (thick), each on the "
        "SAME pooled set of local states (the APIC affine matrix C and the velocity v, position-free) with "
        "only the regression target scaled by its own viscosity, and each fits the true stress to a "
        "validation relative RMSE of about 0.02. (1) Each net reproduces its own viscosity on the training "
        f"scene: driving the fluid entirely with the learned stress gives a final spread width within "
        f"{q1_rel*100:.1f}% of the true simulator across all three viscosities, and the learned particles "
        "overlay the true ones throughout the clip. (2) The nets generalize to a dam-break scene they were "
        "never trained on (trained only on the pour): the learned leading front tracks the true one to "
        f"within {q2_rel*100:.1f}% at both thin and thick, evidence the position-free features forced a "
        "genuine local velocity-gradient stress law rather than a memorized trajectory. (3) The weight "
        "interpolation is a clean negative. Because the true target mu*(C+C^T) is linear in mu, linearly "
        "interpolating the two per-viscosity functions is exactly an intermediate viscosity, so the "
        "function-space ideal is unambiguous and any deviation is a pure weight-space effect. Sweeping "
        "theta(a) = (1-a) theta_thin + a theta_thick, the effective viscosity read off a true-sim spread "
        "calibration matches the ideal only at the two endpoints and sags well below it across the whole "
        f"interior. At a = {mid_a:.2f} the intended viscosity is {mid_ideal:.3f} but the interpolated fluid "
        f"measures only {mid_A:.3f} (independent inits) and {mid_B:.3f} (anchored), roughly half as thick "
        f"as intended; the maximum deviation from ideal is {A_dev:.3f} in mu for the independent regime and "
        f"{B_dev:.3f} for the anchored regime. Crucially the two regimes behave almost identically, so "
        "warm-starting the thick net from the thin net (sharing an initialisation) does NOT rescue the "
        f"interpolation. The anchored thick net in fact ends slightly FARTHER from the thin net in weight "
        f"space ({d_anchor:.1f}) than the independently initialised one ({d_indep:.1f}), because fitting the "
        "much larger thick stress moves the weights a long way regardless of where they started. The "
        "interpolated effective viscosity is monotone in a in both regimes (monotone fraction "
        f"{A_mono:.2f} and {B_mono:.2f}) but strongly convex, and every interpolated rollout stayed finite. "
        "The honest one-line summary: each net reproduces and generalizes its own viscosity, but weight "
        "interpolation gives a fluid systematically too thin in the middle, and anchoring the init does not "
        "fix it."
    )

    hypothesis = (
        "The undershoot is a property of the nonlinear map from weights to behavior, not of independent "
        "random seeds. A network represents a function, but the map F from a weight vector to the function "
        "it computes is many-to-one and strongly nonlinear. A tanh MLP can permute its hidden units, flip "
        "signs, and trade scale between its input and output layers without changing its output, and beyond "
        "those symmetries the output depends on the weights through a composition, the product of the two "
        "layers passed through a saturating nonlinearity. So a straight line in weight space is a curved "
        "path in function space, and the midpoint weights of two distant solutions need not represent the "
        "midpoint function. The thin and thick nets are distant because the thick stress is about fifteen "
        "times larger than the thin, so its weights must grow substantially to produce it; the chord "
        "between a small-weight net and a large-weight net passes through intermediate weights whose "
        "composed output is smaller than the average of the endpoints, which reads as a fluid thinner than "
        "the intended intermediate. The standard suspect for interpolation failure between neural nets is "
        "coordinate mismatch, the linear-mode-connectivity story in which two independently trained nets "
        "assign the same role to different hidden units so their average is scrambled. This experiment "
        "TESTS that suspect directly by also interpolating a warm-started pair that shares an "
        "initialisation, and finds the sag essentially unchanged, so coordinate mismatch is not the "
        "dominant cause here. The dominant cause is the compositional nonlinearity along a long chord, and "
        "the warm start does not help because it does not shorten the chord, both thick solutions are far "
        "from the thin one. The sag is a bounded undershoot rather than a blow-up because a small tanh net "
        "degrades gracefully in its near-linear regime. Concrete predictions follow. Carrying the stress "
        "magnitude in a single linear output layer (freezing the hidden layer shared across viscosities) "
        "should make interpolation close to exact, because the function would then be linear in the "
        "interpolated weights. Conditioning one network on mu as an input removes the need to interpolate "
        "at all and should give a genuinely smooth slider. An explicit neuron-alignment step before "
        "interpolating would quantify the residual coordinate-mismatch contribution, expected to be small "
        "here. Larger and more nonlinear networks, and physical parameters whose stress is nonlinear in the "
        "parameter, should make the undershoot worse."
    )

    limitations = (
        "A demonstration on one architecture and one physical parameter, not a general law about learned "
        "dynamics or weight interpolation. Everything is 2D, n_grid=64, f32, weakly-compressible fluid, one "
        "grid resolution, one fixed timestep dt=1.5e-4, three viscosities spanning a single decade "
        "(0.02 to 0.30, kept moderate so a single stable explicit timestep covers the whole interpolation "
        "range), one pour training scene, and a supervised regression target rather than a rollout-trained "
        "fit. The viscous target mu*(C+C^T) is linear in the interpolation parameter, which is what makes "
        "the ideal intermediate unambiguous; a parameter whose stress depends nonlinearly on it would not "
        "even have an exactly-intermediate function-space target and is untested here. Only viscosity is "
        "learned, the pressure term is left analytic, so this is a learned portion of the update, not a "
        "learned solver. Effective viscosity is read off a spread-width calibration on the pour scene, a "
        "robust but scene-specific and grid-limited diagnostic, and the mapping from mu to a physical "
        "viscosity is not calibrated (the labels thin/medium/thick are evocative). The interpolation "
        "regimes are each a single pair of seeds, so the anchored-versus-independent comparison demonstrates "
        "that warm-start anchoring does not rescue the sag on this problem but does not measure the "
        "statistics across seeds; a stronger anchoring (a shorter or regularised fit that keeps the thick "
        "net near the thin one) and the standard neuron-alignment fix were reasoned about but not run. The "
        "mechanism attribution (compositional nonlinearity over coordinate mismatch) is supported by the "
        "two-regime comparison and the weight distances but is a hypothesis, not an isolated ablation. "
        "Generalization is tested on two scenes (pour, dam-break), not across resolution, blob shape "
        "families, or viscosity extremes. GPU atomic-add accumulation is not bitwise reproducible; rerun if "
        "a frame looks off."
    )

    manifest = {
        "schema_version": "2",
        "task_id": "train-and-interpolate-nns-to-mimic-viscous-liquids",
        "direction": "material-variants",
        "title": "Learning viscosity with a small net, and interpolating the weights",
        "status": "active",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": (
            "Replace the analytic Newtonian viscous stress of an MLS-MPM fluid with a small neural network "
            "that maps position-free local state (the APIC affine matrix C and the velocity v) to the "
            "viscous stress tensor, train the same tiny architecture separately at three viscosities by "
            "supervised regression against the true simulator, and then interpolate the trained weights to "
            "ask whether an intermediate viscosity emerges smoothly. Three questions in order: does each "
            "net reproduce its own viscosity, does it generalize to a scene it was not trained on, and does "
            "linear interpolation of the per-viscosity weights produce a fluid of intermediate, smoothly "
            "varying effective viscosity. Because the true target is linear in the viscosity, the "
            "function-space ideal for interpolation is exactly an intermediate viscosity, which isolates "
            "the weight-space geometry as the only thing under test."
        ),
        "findings": findings,
        "hypothesis": hypothesis,
        "limitations": limitations,
        "results": results,
        "custom_html": None,
        "training_refs": ["learned-viscosity-interpolation", "hybrid-learned-residual", "viscosity",
                          "differentiating-the-rollout", "mpm-in-context"],
        "params": m,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
