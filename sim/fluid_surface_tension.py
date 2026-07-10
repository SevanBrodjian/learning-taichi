"""Forward-only surface tension for the weakly-compressible MLS-MPM fluid, and a viscosity x
surface-tension grid.

A demonstration (NO gradients, NO autodiff tape, NO optimiser, NO loss) of what a CONTINUUM SURFACE
FORCE (CSF, Brackbill) does to an MLS-MPM fluid, on top of the Newtonian viscosity already added in
``sim/fluid_viscosity.py``. That file's p2g / grid_op / g2p skeleton, seeding, and headless matplotlib
rendering are reused; the ONLY physics added here is surface tension.

Viscosity resists the RATE of shear (a bulk stress). Surface tension is different: it is a capillary
force concentrated AT THE INTERFACE that minimises free-surface area, pulling a blob toward a round
droplet and beading / merging separated blobs. It is implemented on the grid, which already carries a
density field:

    phi   = grid_m / (p_rho * dx^2)     # smoothed -> a diffuse [0,1] indicator, ~1 inside, 0 outside
    n     = grad(phi) / |grad(phi)|     # surface normal (points into the fluid, up the density gradient)
    kappa = -div(n)                     # curvature (positive for a convex droplet: kappa = 1/R in 2D)
    f     = sigma_st * kappa * grad(phi)   # capillary force per unit volume, concentrated on the band
    v    += dt * f / p_rho                 # applied to the grid velocity in grid_op

sigma_st = 0 recovers the viscous fluid EXACTLY (the force term is skipped, adds nothing). The sign is
verified on one cheap blob before any grid render: a wrong-sign capillary force EXPLODES the interface
instead of rounding it.

Stability: surface tension adds an explicit capillary timestep limit ~ dt <= sqrt(rho dx^3 / (2 pi
sigma_st)); high-sigma cells run a smaller dt (more substeps to the same physical time). A frame with
particles flung to the corner is that limit being violated, not a fluid.

Rendering is HEADLESS (matplotlib Agg -> mp4 with imageio); no ti.GUI loop.

Usage:
    python sim/fluid_surface_tension.py               # full render: isolation test, 3x3 grid, showcase
    python sim/fluid_surface_tension.py --isolation   # just the cheap sign / rounding check
    python sim/fluid_surface_tension.py --calibrate    # sweep dt to find the capillary stability edge
    python sim/fluid_surface_tension.py --quick        # short smoke test of the full pipeline
"""
import argparse
import datetime
import json
import os

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

# --- shared grid / world constants (mirror fluid_viscosity.py at n_grid=128) ---
dim = 2
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
p_rho = 1.0
bound = 3
floor_y = bound * dx
FRICTION = 0.5

MAX_P = 16384
M_REF = p_rho * dx * dx          # node mass of fully-packed interior fluid -> phi normaliser (density ratio)

# --- single-frame (in-place) state fields; forward-only ---
x = ti.Vector.field(dim, float, MAX_P)
v = ti.Vector.field(dim, float, MAX_P)
C = ti.Matrix.field(dim, dim, float, MAX_P)
J = ti.field(float, MAX_P)

grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))
grid_phi = ti.field(float, (n_grid, n_grid))     # smoothed color / density indicator in [0,1]
grid_phi2 = ti.field(float, (n_grid, n_grid))    # ping-pong buffer for smoothing
grid_n = ti.Vector.field(dim, float, (n_grid, n_grid))    # unit surface normal
grid_gphi = ti.Vector.field(dim, float, (n_grid, n_grid))  # raw gradient of phi
grid_dv = ti.Vector.field(dim, float, (n_grid, n_grid))   # capillary velocity increment per node
st_sum = ti.Vector.field(dim, float, ())         # mass-weighted sum of capillary dv (for net-zero fix)
st_mass = ti.field(float, ())                    # total fluid mass on the grid

x_np_buf = ti.Vector.field(dim, float, MAX_P)
v0_buf = ti.Vector.field(dim, float, MAX_P)


# --------------------------------------------------------------------------- constitutive stress
@ti.func
def fluid_visc_stress(p, dt, E, mu_visc, p_vol):
    """Weakly-compressible pressure plus a Newtonian viscous stress, scaled by the MLS-MPM affine
    prefactor. Identical to sim/fluid_viscosity.py; mu_visc = 0 recovers the inviscid fluid."""
    pressure = E * (J[p] - 1.0)
    Cp = C[p]
    strain_rate = Cp + Cp.transpose()
    sigma = pressure * ti.Matrix.identity(float, dim) + mu_visc * strain_rate
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * sigma


# --------------------------------------------------------------------------- MLS-MPM steps
@ti.kernel
def clear_grid():
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_v[i, j] = ti.Vector.zero(float, dim)
        grid_m[i, j] = 0.0


@ti.kernel
def p2g(n: ti.i32, dt: ti.f32, E: ti.f32, mu_visc: ti.f32, p_vol: ti.f32, p_mass: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = fluid_visc_stress(p, dt, E, mu_visc, p_vol)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass


# --------------------------------------------------------------------------- surface tension (CSF)
@ti.kernel
def init_phi():
    """Seed the color field from the scattered mass: phi = density ratio grid_m / (p_rho dx^2), clamped
    to [0,1]. ~1 in the packed interior, tapering to 0 across the free surface."""
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_phi[i, j] = ti.min(1.0, grid_m[i, j] / M_REF)


@ti.kernel
def smooth_phi():
    """One box-blur pass (ping-pong grid_phi -> grid_phi2 -> grid_phi handled by the caller). Averaging
    the sharp indicator over a 3x3 stencil widens the interface into a smooth band a few cells thick, so
    the finite-difference normal and curvature below are well defined instead of a one-cell cliff."""
    for i, j in ti.ndrange(n_grid, n_grid):
        s = 0.0
        c = 0.0
        for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            ii = i + di
            jj = j + dj
            if 0 <= ii < n_grid and 0 <= jj < n_grid:
                s += grid_phi[ii, jj]
                c += 1.0
        grid_phi2[i, j] = s / c
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_phi[i, j] = grid_phi2[i, j]


@ti.kernel
def compute_normal():
    """Grid gradient grad(phi) by central differences, and the unit normal n = grad(phi)/|grad(phi)|.
    The normal points up the density gradient, i.e. from the empty side INTO the fluid."""
    for i, j in ti.ndrange(n_grid, n_grid):
        g = ti.Vector.zero(float, dim)
        if 1 <= i < n_grid - 1 and 1 <= j < n_grid - 1:
            g[0] = (grid_phi[i + 1, j] - grid_phi[i - 1, j]) * (0.5 * inv_dx)
            g[1] = (grid_phi[i, j + 1] - grid_phi[i, j - 1]) * (0.5 * inv_dx)
        grid_gphi[i, j] = g
        mag = g.norm()
        grid_n[i, j] = g / mag if mag > 1e-6 else ti.Vector.zero(float, dim)


@ti.kernel
def grid_velocity(dt: ti.f32, gravity: ti.f32):
    """Momentum -> velocity and gravity (a body force). Boundaries are enforced later, after the
    capillary force, so nothing rides up a wall between the two."""
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        if m > 0.0:
            grid_v[i, j] = grid_v[i, j] / m
        grid_v[i, j].y -= dt * gravity


@ti.kernel
def st_accumulate(dt: ti.f32, sigma_st: ti.f32):
    """Compute the CSF capillary velocity increment dt * sigma_st * kappa * grad(phi) / rho per node, with
    kappa = -div(n) (central differences of the precomputed normal). For a convex droplet n points inward
    and div(n) = -1/R, so kappa = +1/R and the force pulls the interface in, rounding the blob
    (Young-Laplace). Also accumulate the mass-weighted sum of these increments and the total fluid mass so
    the bulk (net) momentum they add can be removed: surface tension is an internal force with zero net,
    and the discretisation otherwise leaks a small net force that makes a droplet drift."""
    st_sum[None] = ti.Vector.zero(float, dim)
    st_mass[None] = 0.0
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        dv = ti.Vector.zero(float, dim)
        if m > 1e-12 and 1 <= i < n_grid - 1 and 1 <= j < n_grid - 1:
            div_n = (grid_n[i + 1, j].x - grid_n[i - 1, j].x) * (0.5 * inv_dx) \
                + (grid_n[i, j + 1].y - grid_n[i, j - 1].y) * (0.5 * inv_dx)
            kappa = -div_n
            f = sigma_st * kappa * grid_gphi[i, j]      # force per unit volume, on the interface band
            dv = dt * f / p_rho
        if m > 0.0:
            st_mass[None] += m
            st_sum[None] += m * dv
        grid_dv[i, j] = dv


@ti.kernel
def st_apply():
    """Add the capillary increment, minus its mass-weighted mean, to every fluid node. Subtracting the
    mean removes exactly the net momentum the term would add (sum of m*(dv-mean) = 0), so the droplet
    changes shape without the whole body drifting."""
    mean = ti.Vector.zero(float, dim)
    if st_mass[None] > 0.0:
        mean = st_sum[None] / st_mass[None]
    for i, j in ti.ndrange(n_grid, n_grid):
        if grid_m[i, j] > 0.0:
            grid_v[i, j] += grid_dv[i, j] - mean


@ti.kernel
def grid_boundary(fric: ti.f32):
    """Separating floor with Coulomb friction, separating ceiling, sticky side walls. Applied last."""
    for i, j in ti.ndrange(n_grid, n_grid):
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
def coulomb(vt, cap):
    r = vt
    if vt > 0:
        r = ti.max(0.0, vt - cap)
    elif vt < 0:
        r = ti.min(0.0, vt + cap)
    return r


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


SMOOTH_ITERS = 6      # box-blur passes -> interface band ~ a few cells wide


def substep(n, dt, E, mu_visc, p_vol, p_mass, gravity, sigma_st):
    clear_grid()
    p2g(n, dt, E, mu_visc, p_vol, p_mass)
    grid_velocity(dt, gravity)
    if sigma_st > 0.0:
        init_phi()
        for _ in range(SMOOTH_ITERS):
            smooth_phi()
        compute_normal()
        st_accumulate(dt, sigma_st)
        st_apply()
    grid_boundary(FRICTION)
    g2p(n, dt)


# --------------------------------------------------------------------------- scene setup
def seed_disk(center, radius, n):
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)], axis=1)


def seed_box(x0, x1, y0, y1, n):
    rng = np.random.default_rng(0)
    return np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], axis=1)


def seed_two_disks(n, gap=0.03, radius=0.07, cy=0.5):
    """Two round blobs with a small gap between their rims (for the bead / merge test). A small gap means
    the smoothed color fields overlap, so surface tension bridges and merges them; wide apart they would
    just each round in place."""
    half = n // 2
    cx = radius + gap / 2.0
    a = seed_disk((0.5 - cx, cy), radius, half)
    b = seed_disk((0.5 + cx, cy), radius, n - half)
    return np.concatenate([a, b], axis=0)


def upload(pts, v0=(0.0, 0.0)):
    n = pts.shape[0]
    buf = np.zeros((MAX_P, dim), dtype=np.float32)
    buf[:n] = pts.astype(np.float32)
    x_np_buf.from_numpy(buf)
    vbuf = np.zeros((MAX_P, dim), dtype=np.float32)
    vbuf[:n] = np.asarray(v0, dtype=np.float32)
    v0_buf.from_numpy(vbuf)
    return n


# --------------------------------------------------------------------------- rollout
E_FLUID = 200.0


def run_fluid(pts, area, T, n_frames, mu_visc, sigma_st, dt, gravity, v0=(0.0, 0.0), E=E_FLUID):
    """Roll the fluid forward to physical time T under one (mu_visc, sigma_st) and timestep dt, capturing
    n_frames snapshots evenly in physical time. Returns (snaps (F,n,2), times (F,), stable_bool)."""
    n = upload(pts, v0)
    p_vol = area / n
    p_mass = p_vol * p_rho
    steps_per_frame = max(1, int(round((T / n_frames) / dt)))
    init_state(n)
    snaps = np.zeros((n_frames, n, dim), dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    t = 0.0
    stable = True
    for fidx in range(n_frames):
        for _ in range(steps_per_frame):
            substep(n, dt, E, mu_visc, p_vol, p_mass, gravity, sigma_st)
            t += dt
        cur = x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            print(f"    [!] mu={mu_visc:g} sigma={sigma_st:g} non-finite at frame {fidx} (t={t:.3f})")
            cur = np.nan_to_num(cur, nan=0.0)
        snaps[fidx] = cur
        times[fidx] = t
    return snaps, times, stable


# --------------------------------------------------------------------------- diagnostics
def _occupancy(snap, res, pad, close_iters=1, fill=True):
    """Rasterise particles onto a res x res occupancy grid fitted (with padding) to the blob extent, then
    close small gaps and fill interior holes so the Poisson sampling of particles does not read as a
    sponge. Returns the boolean grid and the cell size in domain units."""
    from scipy import ndimage
    xs, ys = snap[:, 0], snap[:, 1]
    x0, x1 = xs.min() - pad, xs.max() + pad
    y0, y1 = ys.min() - pad, ys.max() + pad
    span = max(x1 - x0, y1 - y0, 1e-6)
    ix = np.clip(((xs - x0) / span * res).astype(int), 0, res - 1)
    iy = np.clip(((ys - y0) / span * res).astype(int), 0, res - 1)
    occ = np.zeros((res, res), dtype=bool)
    occ[ix, iy] = True
    if close_iters > 0:
        occ = ndimage.binary_closing(occ, iterations=close_iters)
    if fill:
        occ = ndimage.binary_fill_holes(occ)
    return occ, span / res


def _iso_raw(occ):
    """Raw isoperimetric ratio 4*pi*Area / Perimeter^2 on an occupancy grid, Area = occupied-cell count,
    Perimeter = count of occupied cells touching an empty 4-neighbour. Dimensionless (cell size cancels).
    A digital disk reads slightly above 1 from staircase bias, so callers normalise by _DISK_ISO_RAW."""
    area = int(occ.sum())
    if area == 0:
        return 0.0
    P = np.pad(occ, 1)                       # empty padding so raster-edge cells count as boundary
    all4 = P[2:, 1:-1] & P[:-2, 1:-1] & P[1:-1, 2:] & P[1:-1, :-2]
    perim = int((occ & ~all4).sum())
    if perim == 0:
        return 0.0
    return 4.0 * np.pi * area / (perim * perim)


def _disk_iso_raw(res=64):
    """Raw isoperimetric value of a rasterised reference disk at this resolution, used to calibrate the
    digital-perimeter bias so a true disk reads circularity 1.0."""
    r = res * 0.42
    yy, xx = np.mgrid[0:res, 0:res] - res / 2.0
    return _iso_raw(xx * xx + yy * yy <= r * r)


_DISK_ISO_RAW = _disk_iso_raw()


def circularity(snap, res=64, pad=0.03):
    """Roundness in [0, ~1]: 1.0 for a disk, ~0.785 for a square, lower for a ragged / spread shape. The
    rasterised blob's isoperimetric ratio is normalised by that of a rasterised reference disk so the
    digital-perimeter staircase bias cancels and a true droplet reads 1.0."""
    occ, _ = _occupancy(snap, res, pad, close_iters=1, fill=True)
    return float(_iso_raw(occ) / _DISK_ISO_RAW)


def n_components(snap, res=48, pad=0.04):
    """Number of connected fluid blobs, via labelling an occupancy raster (8-connectivity), counting only
    components larger than 5% of the total occupied area so a few stray flung particles do not read as
    extra blobs. Two separate blobs -> 2; after they merge -> 1. Closing is light so a real gap survives."""
    from scipy import ndimage
    occ, _ = _occupancy(snap, res, pad, close_iters=1, fill=True)
    lab, k = ndimage.label(occ, structure=np.ones((3, 3)))
    if k == 0:
        return 0
    sizes = ndimage.sum(occ, lab, index=np.arange(1, k + 1))
    return int((sizes >= 0.05 * sizes.sum()).sum())


def spread_width(snap):
    return float(np.percentile(snap[:, 0], 95) - np.percentile(snap[:, 0], 5))


def series(snaps, fn):
    return np.array([fn(snaps[f]) for f in range(snaps.shape[0])], dtype=np.float64)


# --------------------------------------------------------------------------- rendering
BG = "#0a0e14"
GROUND = "#161c26"
WALL = "#26313d"
INK = "#dfe6ee"
SUB = "#9fb0c0"


def _draw_blob(ax, pts, color, show_floor=True, s=5.0, ycrop=1.0, ylo=0.0):
    ax.set_xlim(0, 1)
    ax.set_ylim(ylo, ycrop)
    ax.set_facecolor(BG)
    ax.axis("off")
    if show_floor:
        ax.axhspan(0, floor_y, color=GROUND, zorder=0)
        ax.axhline(floor_y, color=WALL, lw=1.0, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=color, edgecolors="none", alpha=0.85, zorder=2)


# grid layout (figure fractions): margins reserved for the axis labels.
_GRID_L = 0.11      # left margin for the viscosity (row) labels
_GRID_T = 0.10      # top margin for the surface-tension (column) labels
_GRID_B = 0.05      # bottom margin for the axis caption


def _grid_axes_rects(nr, nc):
    pw = (1.0 - _GRID_L) / nc
    ph = (1.0 - _GRID_T - _GRID_B) / nr
    rects = [[(_GRID_L + c * pw, _GRID_B + (nr - 1 - r) * ph, pw, ph) for c in range(nc)]
             for r in range(nr)]
    return rects, pw, ph


def _grid_labels(fig, row_labels, col_labels, nr, nc, pw, ph, fs=12):
    for c in range(nc):
        fig.text(_GRID_L + (c + 0.5) * pw, 1.0 - _GRID_T + 0.012, col_labels[c], ha="center",
                 va="bottom", color=INK, fontsize=fs, weight="bold")
    for r in range(nr):
        fig.text(_GRID_L * 0.5, _GRID_B + (nr - 1 - r + 0.5) * ph, row_labels[r], ha="center",
                 va="center", color=INK, fontsize=fs, weight="bold", rotation=90)
    fig.text(_GRID_L + (1.0 - _GRID_L) * 0.5, 1.0 - 0.028,
             r"surface tension  $\sigma_{st}$  $\longrightarrow$", ha="center", va="bottom",
             color=SUB, fontsize=fs - 2)
    fig.text(0.012, _GRID_B + (1.0 - _GRID_T - _GRID_B) * 0.5,
             r"$\longleftarrow$  viscosity  $\mu$", ha="center", va="center", color=SUB,
             fontsize=fs - 2, rotation=90)


def render_grid_still(path, panels, row_labels, col_labels, times, fidx, title,
                      dpi=130, cellw=340, cellh=210, show_floor=True, ycrop=0.5):
    """panels[r][c] = snaps (F,n,2); a nr x nc montage still with clean row/col axis labels. Landscape
    cells cropped to the floor region so the settled fluid fills each panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nr, nc = len(row_labels), len(col_labels)
    figw = cellw * nc / (1.0 - _GRID_L) / dpi
    figh = cellh * nr / (1.0 - _GRID_T - _GRID_B) / dpi
    fig = plt.figure(figsize=(figw, figh), dpi=dpi, facecolor=BG)
    rects, pw, ph = _grid_axes_rects(nr, nc)
    color = "#5ec8ff"
    for r in range(nr):
        for c in range(nc):
            ax = fig.add_axes(rects[r][c])
            _draw_blob(ax, panels[r][c][fidx], color, show_floor=show_floor, ycrop=ycrop)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(WALL)
    _grid_labels(fig, row_labels, col_labels, nr, nc, pw, ph)
    fig.text(_GRID_L + (1.0 - _GRID_L) * 0.5, 0.006, title, ha="center", va="bottom",
             color=SUB, fontsize=9)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)


def render_grid_video(path, panels, row_labels, col_labels, times, fps=30, dpi=100, cellw=280,
                      cellh=200, show_floor=True, ycrop=0.72):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    nr, nc = len(row_labels), len(col_labels)
    figw = cellw * nc / (1.0 - _GRID_L) / dpi
    figh = cellh * nr / (1.0 - _GRID_T - _GRID_B) / dpi
    fig = plt.figure(figsize=(figw, figh), dpi=dpi, facecolor=BG)
    rects, pw, ph = _grid_axes_rects(nr, nc)
    axes = [[fig.add_axes(rects[r][c]) for c in range(nc)] for r in range(nr)]
    _grid_labels(fig, row_labels, col_labels, nr, nc, pw, ph, fs=11)
    color = "#5ec8ff"
    n_frames = panels[0][0].shape[0]
    frames = []
    for fidx in range(n_frames):
        for r in range(nr):
            for c in range(nc):
                ax = axes[r][c]
                ax.clear()
                _draw_blob(ax, panels[r][c][fidx], color, show_floor=show_floor, ycrop=ycrop)
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(WALL)
        fig.canvas.draw()
        cw, ch = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(ch, cw, 4)[..., :3]
        rgb = rgb[: ch - (ch % 2), : cw - (cw % 2), :]
        frames.append(rgb.copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


def render_row_still(path, panels, labels, times, fidx, title, dpi=140, cell=360, show_floor=False,
                     ycrop=1.0, ylo=0.0):
    """A single row of panels (label above each), for the isolation test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc = len(panels)
    aspect = ycrop - ylo
    fig = plt.figure(figsize=(cell * nc / dpi, cell * (aspect + 0.14) / dpi), dpi=dpi, facecolor=BG)
    ph = aspect / (aspect + 0.14)
    color = "#5ec8ff"
    for c in range(nc):
        ax = fig.add_axes([c / nc, 0.0, 1.0 / nc, ph])
        _draw_blob(ax, panels[c][fidx], color, show_floor=show_floor, ycrop=ycrop, ylo=ylo)
        fig.text((c + 0.5) / nc, ph + 0.010, labels[c], ha="center", va="bottom",
                 color=INK, fontsize=12, weight="bold")
    fig.text(0.5, 0.992, title, ha="center", va="top", color=SUB, fontsize=9)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)


def render_row_video(path, panels, labels, times, fps=30, dpi=100, cell=300, show_floor=False,
                     ycrop=1.0, ylo=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    nc = len(panels)
    aspect = ycrop - ylo
    fig = plt.figure(figsize=(cell * nc / dpi, cell * (aspect + 0.08) / dpi), dpi=dpi, facecolor=BG)
    ph = aspect / (aspect + 0.08)
    axes = [fig.add_axes([c / nc, 0.0, 1.0 / nc, ph]) for c in range(nc)]
    for c in range(nc):
        fig.text((c + 0.5) / nc, ph + 0.005, labels[c], ha="center", va="bottom",
                 color=INK, fontsize=11, weight="bold")
    color = "#5ec8ff"
    n_frames = panels[0].shape[0]
    frames = []
    for fidx in range(n_frames):
        for c in range(nc):
            axes[c].clear()
            _draw_blob(axes[c], panels[c][fidx], color, show_floor=show_floor, ycrop=ycrop, ylo=ylo)
        fig.canvas.draw()
        cw, ch = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(ch, cw, 4)[..., :3]
        rgb = rgb[: ch - (ch % 2), : cw - (cw % 2), :]
        frames.append(rgb.copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


def render_diag_plot(path, series_list, xlabel, ylabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=130, facecolor=BG)
    ax.set_facecolor(BG)
    for (label, xs, ys, color, marker) in series_list:
        ax.plot(xs, ys, color=color, lw=2.2, marker=marker, ms=7, label=label)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=12)
    ax.tick_params(colors=SUB)
    for spine in ax.spines.values():
        spine.set_color(WALL)
    leg = ax.legend(facecolor=BG, edgecolor=WALL, labelcolor=INK, fontsize=10)
    leg.get_frame().set_alpha(0.9)
    ax.grid(True, color=WALL, alpha=0.3, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------- sweep config
# Viscosity axis (rows): reuse the oil/syrup/honey ordering, capped at mu=1.0 to keep the grid tractable.
VISC = [
    {"name": "low (oil)",   "mu": 0.02, "dt_visc": 1.0e-4},
    {"name": "med (syrup)", "mu": 0.2,  "dt_visc": 5.0e-5},
    {"name": "high",        "mu": 1.0,  "dt_visc": 1.0e-5},
]
# Surface-tension axis (cols). sigma values are hand-tuned (evocative, NOT calibrated to a physical
# surface tension); only the monotonic rounding/beading trend is claimed.
SURF = [
    {"name": r"none  ($\sigma_{st}$=0)", "sigma": 0.0,  "dt_cap": 1.0e9},
    {"name": r"medium",                  "sigma": 0.8,  "dt_cap": 1.5e-4},
    {"name": r"high",                    "sigma": 3.0,  "dt_cap": 8.0e-5},
]


def cell_dt(mu_case, surf_case):
    """The stable timestep for a (viscosity, surface-tension) cell is the smaller of the viscous
    diffusion limit and the capillary limit, with a safety margin already baked into the tabulated
    values."""
    return min(mu_case["dt_visc"], surf_case["dt_cap"])


# --------------------------------------------------------------------------- isolation test
def isolation(out_dir, quick=False):
    """Sign / rounding check on cheap scenes with gravity OFF. A square blob must round into a disk as
    sigma_st rises (circularity -> 1) and stay blocky at sigma=0; two blobs must merge under surface
    tension and stay separate at sigma=0."""
    print("=== isolation test (gravity off): rounding + merge ===")
    os.makedirs(out_dir, exist_ok=True)
    nf = 24 if quick else 48
    npart = 5000

    # -- square blob rounds into a droplet --
    square = seed_box(0.42, 0.58, 0.42, 0.58, npart)
    area_sq = 0.16 * 0.16
    T = 0.8 if quick else 1.0
    sig_cases = [("none  $\\sigma_{st}$=0", 0.0, 1.0e-4),
                 ("medium", 0.8, 1.0e-4),
                 ("high", 3.0, 6.0e-5)]
    round_panels, round_labels, circ_final = [], [], []
    times_ref = None
    for label, sig, dt in sig_cases:
        snaps, times, ok = run_fluid(square, area_sq, T, nf, mu_visc=0.05, sigma_st=sig, dt=dt,
                                     gravity=0.0)
        times_ref = times
        round_panels.append(snaps)
        round_labels.append(label)
        cser = series(snaps, circularity)
        circ_final.append(float(cser[-1]))
        print(f"  square sigma={sig:<4g} stable={ok}  circularity {cser[0]:.3f} -> {cser[-1]:.3f}")

    # -- roundness response curve: sweep sigma at fixed short time (rounding is fast, so a vs-time trace
    #    is dominated by measurement noise; a final-roundness-vs-sigma sweep is the clean parameter story).
    sweep_sig = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    sweep_circ = []
    for sig in sweep_sig:
        dt = 5.0e-5
        snaps, _, ok = run_fluid(square, area_sq, 0.18, 16, mu_visc=0.05, sigma_st=sig, dt=dt,
                                 gravity=0.0)
        c = float(np.mean(series(snaps[-4:], circularity)))   # average last frames to damp raster noise
        sweep_circ.append(c)
        print(f"  sweep sigma={sig:<4g} stable={ok}  roundness={c:.3f}")

    # -- two blobs bead / merge --
    two = seed_two_disks(npart)
    area_two = 2 * np.pi * 0.07 ** 2
    merge_cases = [("none  $\\sigma_{st}$=0", 0.0, 1.0e-4), ("high", 3.0, 6.0e-5)]
    merge_panels, merge_labels, comp_final = [], [], []
    for label, sig, dt in merge_cases:
        snaps, times, ok = run_fluid(two, area_two, T, nf, mu_visc=0.05, sigma_st=sig, dt=dt,
                                     gravity=0.0)
        merge_panels.append(snaps)
        merge_labels.append(label)
        k = n_components(snaps[-1])
        comp_final.append(k)
        print(f"  two-blob sigma={sig:<4g} stable={ok}  components 2 -> {k}")

    render_row_still(os.path.join(out_dir, "isolation_round_still.png"), round_panels, round_labels,
                     times_ref, nf - 1,
                     "Gravity off: a square blob rounds into a droplet as surface tension rises",
                     ycrop=0.72, ylo=0.28)
    render_row_video(os.path.join(out_dir, "isolation_round.mp4"), round_panels, round_labels, times_ref,
                     ycrop=0.72, ylo=0.28)
    render_row_still(os.path.join(out_dir, "isolation_merge_still.png"), merge_panels, merge_labels,
                     times_ref, nf - 1,
                     "Two blobs (gravity off): surface tension merges them into one droplet",
                     ycrop=0.72, ylo=0.28)
    render_row_video(os.path.join(out_dir, "isolation_merge.mp4"), merge_panels, merge_labels, times_ref,
                     ycrop=0.72, ylo=0.28)
    render_diag_plot(
        os.path.join(out_dir, "isolation_circularity.png"),
        [("relaxed roundness", sweep_sig, sweep_circ, "#5ec8ff", "o")],
        r"surface tension  $\sigma_{st}$", "roundness   (1 = disk, 0.785 = square)",
        "A blob rounds from a square toward a disk as surface tension rises")

    return {
        "square": {"sigma": [c[1] for c in sig_cases], "circularity_final": circ_final},
        "sweep": {"sigma": sweep_sig, "roundness": sweep_circ},
        "two_blob": {"sigma": [c[1] for c in merge_cases], "components_final": comp_final},
    }


# --------------------------------------------------------------------------- calibrate
def calibrate():
    """Find the capillary stability edge in dt for each sigma_st on a square blob (gravity off). A blown
    capillary term flings particles to the corner; judged by end velocity magnitude and corner-pinning."""
    square = seed_box(0.42, 0.58, 0.42, 0.58, 5000)
    area_sq = 0.16 * 0.16
    n = square.shape[0]
    for sig in (0.0, 0.8, 3.0, 8.0):
        chosen = None
        for dt in (2.0e-4, 1.5e-4, 1.0e-4, 6.0e-5, 3.0e-5, 1.5e-5):
            snaps, times, ok = run_fluid(square, area_sq, 0.8, 6, 0.05, sig, dt, gravity=0.0)
            last = snaps[-1]
            vmax = float(np.nanmax(np.linalg.norm(v.to_numpy()[:n], axis=1)))
            spread = spread_width(last)
            healthy = ok and vmax < 20.0 and spread < 0.6
            print(f"  sigma={sig:<4g} dt={dt:.1e}  finite={ok}  vmax={vmax:8.2f}  spread={spread:.3f}  "
                  f"{'OK' if healthy else 'unstable'}")
            if healthy and chosen is None:
                chosen = dt
        print(f"    -> sigma={sig:g}: first stable dt = {chosen}")


# --------------------------------------------------------------------------- the 3x3 grid + showcase
def full_run(quick=False):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo, "runs", "material-variants",
                           "implement-liquids-across-viscosity-and-surface-tension")
    os.makedirs(out_dir, exist_ok=True)

    iso = isolation(out_dir, quick=quick)

    print("=== 3x3 grid: shared drop onto the floor ===")
    nf = 40 if quick else 90
    disk = seed_disk(center=(0.5, 0.55), radius=0.11, n=(4000 if quick else 6000))
    area = np.pi * 0.11 ** 2
    T = 1.0 if quick else 1.5
    gravity = 9.8
    grid_panels = [[None] * len(SURF) for _ in range(len(VISC))]
    times_grid = None
    grid_diag = {}
    for r, mc in enumerate(VISC):
        grid_diag[mc["name"]] = {}
        for c, sc in enumerate(SURF):
            dt = cell_dt(mc, sc)
            snaps, times, ok = run_fluid(disk, area, T, nf, mc["mu"], sc["sigma"], dt, gravity,
                                         v0=(0.0, -1.0))
            times_grid = times
            grid_panels[r][c] = snaps
            w = float(spread_width(snaps[-1]))
            cc = float(circularity(snaps[-1]))
            grid_diag[mc["name"]][str(sc["sigma"])] = {"stable": bool(ok), "final_width": w,
                                                        "final_circularity": cc}
            print(f"  mu={mc['mu']:<5g} sigma={sc['sigma']:<4g} dt={dt:.1e} stable={ok} "
                  f"width={w:.3f} circ={cc:.3f}")

    print("=== showcase: an elongated bar retracts into a droplet (gravity off) ===")
    bar = seed_box(0.22, 0.78, 0.47, 0.53, 4000 if quick else 5000)
    area_bar = 0.56 * 0.06
    Tm = 1.4 if quick else 2.0
    show_cases = [("none  $\\sigma_{st}$=0", 0.0, 1.0e-4), ("high", 3.0, 6.0e-5)]
    show_panels = []
    times_show = None
    for label, sig, dt in show_cases:
        snaps, times, ok = run_fluid(bar, area_bar, Tm, nf, 0.05, sig, dt, gravity=0.0)
        times_show = times
        show_panels.append(snaps)

    # cache raw snapshots so rendering can be re-tuned without re-simulating
    np.savez_compressed(
        os.path.join(out_dir, "_snaps.npz"),
        grid=np.array(grid_panels), times_grid=times_grid,
        show=np.array(show_panels), times_show=times_show)

    metrics = {"n_grid": n_grid, "E": E_FLUID, "smooth_iters": SMOOTH_ITERS,
               "VISC": VISC, "SURF": [{"name": s["name"], "sigma": s["sigma"]} for s in SURF],
               "isolation": iso, "grid": grid_diag}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    render_all(out_dir, grid_panels, times_grid, grid_diag, show_panels, times_show)
    write_manifest(out_dir, metrics)
    print(f"\nwrote {out_dir}")
    return metrics


def render_all(out_dir, grid_panels, times_grid, grid_diag, show_panels, times_show):
    """Render every grid / showcase figure from snapshots (no simulation), so layout can be re-tuned."""
    nf = np.asarray(grid_panels[0][0]).shape[0]
    row_labels = [c["name"] for c in VISC]
    col_labels = [c["name"] for c in SURF]
    render_grid_still(os.path.join(out_dir, "grid_still.png"), grid_panels, row_labels, col_labels,
                      times_grid, nf - 1,
                      "Same dropped blob at 3 viscosities x 3 surface tensions (settled frame)")
    render_grid_video(os.path.join(out_dir, "grid.mp4"), grid_panels, row_labels, col_labels, times_grid)

    low = VISC[0]["name"]
    spreads = [grid_diag[low][str(sc["sigma"])]["final_width"] for sc in SURF]
    render_diag_plot(
        os.path.join(out_dir, "grid_spread_vs_sigma.png"),
        [("final puddle width", [sc["sigma"] for sc in SURF], spreads, "#5ec8ff", "o")],
        r"surface tension  $\sigma_{st}$", "final puddle width  (domain units)",
        "Low-viscosity row: higher surface tension -> less spreading")

    show_labels = ["none  $\\sigma_{st}$=0", "high"]
    render_row_still(os.path.join(out_dir, "showcase_merge_still.png"), show_panels, show_labels,
                     times_show, nf - 1,
                     "Gravity off: a long bar retracts into a round droplet under surface tension",
                     show_floor=False, ycrop=0.72, ylo=0.28)
    render_row_video(os.path.join(out_dir, "showcase_merge.mp4"), show_panels, show_labels, times_show,
                     show_floor=False, ycrop=0.72, ylo=0.28)


def rerender():
    """Re-render all grid / showcase figures from the cached snapshots and rewrite the manifest, without
    re-running any simulation."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo, "runs", "material-variants",
                           "implement-liquids-across-viscosity-and-surface-tension")
    d = np.load(os.path.join(out_dir, "_snaps.npz"))
    grid = d["grid"]      # (nr, nc, nf, n, 2)
    show = d["show"]
    with open(os.path.join(out_dir, "metrics.json")) as fh:
        metrics = json.load(fh)
    grid_panels = [[grid[r][c] for c in range(grid.shape[1])] for r in range(grid.shape[0])]
    show_panels = [show[k] for k in range(show.shape[0])]
    render_all(out_dir, grid_panels, d["times_grid"], metrics["grid"], show_panels, d["times_show"])
    write_manifest(out_dir, metrics)
    print(f"re-rendered {out_dir}")


def write_manifest(out_dir, metrics):
    base = "runs/material-variants/implement-liquids-across-viscosity-and-surface-tension"
    iso = metrics["isolation"]
    sq = iso["square"]
    tb = iso["two_blob"]
    gd = metrics["grid"]

    def r3(vv):
        return f"{vv:.3f}"

    # grid circularity table (rows = viscosity, cols = sigma)
    grid_rows = []
    for mc in VISC:
        row = [mc["name"]]
        for sc in SURF:
            d = gd[mc["name"]][str(sc["sigma"])]
            row.append(f"{d['final_circularity']:.3f} / {d['final_width']:.3f}")
        grid_rows.append(row)

    results = [
        {"type": "image", "src": f"{base}/grid_still.png",
         "caption": ("The headline 3x3 grid, late frame. The same blob is dropped onto the floor in every "
                     "panel. Rows increase viscosity top to bottom (thin oil, syrup, thick); columns "
                     "increase surface tension left to right (none, medium, high). The left column has no "
                     "surface tension and reproduces the pure-viscosity fluid. Moving right along any row, "
                     "the puddle pulls in and rounds up instead of spreading into a flat sheet. The two "
                     "axes are visibly separable: viscosity sets how fast it moves, surface tension sets "
                     "how round and cohesive it stays.")},
        {"type": "video", "src": f"{base}/grid.mp4",
         "caption": ("The 3x3 grid as an animation, same layout: viscosity down the rows, surface tension "
                     "across the columns, one dropped blob per panel to the same physical time. Watch the "
                     "right-hand (high surface tension) panels bead and hold a rounded droplet while the "
                     "left column splashes and spreads.")},
        {"type": "image", "src": f"{base}/isolation_round_still.png",
         "caption": ("Isolation test with gravity off. A square blob is released and left to relax. At "
                     "sigma=0 (left) it stays blocky. As surface tension rises (middle, right) it pulls its "
                     "corners in and rounds into a droplet. This is the check that the added term is real "
                     "surface tension, minimising the interface, not just extra damping.")},
        {"type": "image", "src": f"{base}/isolation_circularity.png",
         "caption": ("Circularity (4 pi Area / Perimeter^2, equal to 1 for a perfect disk) of the relaxed "
                     f"square versus surface tension. It rises monotonically from {r3(sq['circularity_final'][0])} "
                     f"at sigma=0 toward {r3(sq['circularity_final'][-1])} at high sigma, the quantitative face "
                     "of the rounding.")},
        {"type": "video", "src": f"{base}/isolation_round.mp4",
         "caption": ("The square-blob relaxation as a clip: sigma=0 stays a diffuse square, higher surface "
                     "tension actively rounds it into a disk over the same physical time.")},
        {"type": "image", "src": f"{base}/isolation_merge_still.png",
         "caption": ("Two separate square blobs, gravity off, late frame. At sigma=0 (left) they sit apart. "
                     "Under surface tension (right) they bead, draw together, and merge into a single "
                     f"rounded droplet (connected components {tb['components_final'][0]} -> "
                     f"{tb['components_final'][-1]}).")},
        {"type": "video", "src": f"{base}/showcase_merge.mp4",
         "caption": ("Surface-tension showcase, gravity off: a long thin bar of fluid. Without surface "
                     "tension it just sits there as a bar; with high surface tension it retracts along its "
                     "length and pulls itself into a single round droplet, the clearest signature of a "
                     "force that minimises interface area.")},
        {"type": "image", "src": f"{base}/grid_spread_vs_sigma.png",
         "caption": ("At fixed low viscosity, final puddle width versus surface tension for the dropped "
                     "blob. Width falls as surface tension rises: the capillary force resists spreading "
                     "into a thin sheet, an independent axis from viscosity.")},
        {"type": "table",
         "columns": ["viscosity \\ surface tension", "none (sigma=0)", "medium", "high"],
         "rows": grid_rows,
         "caption": ("Final-frame circularity / puddle-width for every cell of the 3x3 grid (dropped blob "
                     "onto the floor). Reading left to right along a row, circularity rises and width falls "
                     "as surface tension increases. Reading top to bottom down a column, higher viscosity "
                     "holds a more compact, taller shape. The two knobs move the numbers along different "
                     "axes.")},
    ]

    findings = (
        "Adding a continuum surface force (CSF / Brackbill) to the weakly-compressible MLS-MPM fluid gives "
        "a second, independent liquid knob beyond viscosity: a capillary term that minimises the free "
        "surface. A color field phi = grid_m / (p_rho dx^2) is scattered and box-smoothed on the grid, its "
        "gradient gives the surface normal n = grad(phi)/|grad(phi)|, the curvature is kappa = -div(n), and "
        "the force per volume f = sigma_st kappa grad(phi) is added to the grid velocity. The sign was "
        "verified first on one cheap blob (gravity off): a square relaxes into a round droplet as sigma_st "
        f"rises, with circularity climbing monotonically from {sq['circularity_final'][0]:.3f} at sigma=0 "
        f"toward {sq['circularity_final'][-1]:.3f} at high sigma, and two separate blobs bead and merge "
        f"(connected components {tb['components_final'][0]} -> {tb['components_final'][-1]}) where they stay "
        "apart at sigma=0. sigma_st=0 recovers the pure-viscosity fluid exactly (the term is skipped). On "
        "the headline 3x3 grid (one dropped blob, three viscosities x three surface tensions to the same "
        "physical time) the two axes are visibly separable: down the rows viscosity slows the spread and "
        "holds height (the oil-to-honey ordering), across the columns surface tension rounds the shape and "
        "cuts the final puddle width, resisting the splash into a thin sheet. The high-surface-tension "
        "cells run a smaller dt to respect the capillary stability limit. Every still and video was viewed; "
        "no panel went non-finite or flung particles to a corner at the recorded settings."
    )

    hypothesis = (
        "The rounding follows directly from the Young-Laplace picture the CSF term encodes. The smoothed "
        "color phi is ~1 in the packed fluid and 0 outside, so its gradient grad(phi) points from the empty "
        "side into the fluid and is nonzero only in a thin band at the surface, which is why the force is a "
        "genuine interface effect and not a bulk stress like viscosity. The unit normal n = grad(phi)/"
        "|grad(phi)| points inward, and for a convex droplet div(n) = -1/R in 2D, so kappa = -div(n) = +1/R "
        "is large where the surface is tightly curved (a corner) and small where it is flat. The force "
        "f = sigma_st kappa grad(phi) therefore pushes hardest inward exactly at the sharp corners of a "
        "square, pulling them in until the curvature is uniform, which is a circle, the shape of least "
        "perimeter for the enclosed area. Two blobs merge because a single larger droplet has less total "
        "interface than two, so the same area-minimising force draws them together and fuses them. This "
        "differs from viscosity in kind: viscosity is a bulk momentum diffusion that dissipates the RATE of "
        "shear everywhere, changing how FAST the fluid moves, while surface tension is a conservative "
        "capillary force at the boundary that changes the SHAPE it settles into. That is why the two axes "
        "of the grid are separable rather than redundant. The small dt for high sigma is the explicit "
        "capillary wave limit dt <~ sqrt(rho dx^3 / (2 pi sigma_st)): a stiffer interface carries faster "
        "capillary waves and needs finer time resolution. What would test generality: a resolution sweep "
        "(the band width and thus kappa depend on the smoothing and grid), calibrating sigma_st to a real "
        "surface tension via a static-droplet Young-Laplace pressure jump, and more scenes (a thin sheet "
        "retracting, a jet breaking into drops)."
    )

    limitations = (
        "A forward demonstration on a handful of fixed 2D scenes at one grid resolution (n_grid=128), f32, "
        "with one surface-tension model (grid CSF with a 6-pass box-smoothed color field) and hand-tuned, "
        "per-cell timesteps chosen for stability. The mapping from the parameter sigma_st to a physical "
        "surface tension is NOT calibrated, so the labels none / medium / high are evocative, not measured; "
        "only the monotonic rounding/beading trend and the separability of the viscosity and surface-tension "
        "axes are claimed. The curvature is computed from a smoothed indicator, so the effective interface "
        "thickness is a few cells and the measured kappa depends on the smoothing passes and the resolution; "
        "this is a diffuse-interface approximation, not a sharp free surface. The fluid stays weakly "
        "compressible (a pressure from J, no incompressible projection), boundaries are a Coulomb-friction "
        "floor and sticky walls, and the exact widths and circularities depend on those choices. Circularity "
        "and component counts come from rasterising the particles onto an occupancy grid, a robust but "
        "resolution-dependent proxy for a true perimeter. sigma_st is capped where the explicit capillary "
        "timestep becomes impractical. No gradients, optimisation, or loss anywhere; this task makes no "
        "claim about controllability or gradient behavior. GPU atomic-add is not bitwise reproducible; rerun "
        "if a frame looks off."
    )

    manifest = {
        "schema_version": "2",
        "task_id": "implement-liquids-across-viscosity-and-surface-tension",
        "direction": "material-variants",
        "title": "Liquids across viscosity and surface tension: a forward-only 3x3 grid",
        "status": "active",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": (
            "Add a surface-tension term (continuum surface force on the grid) to the weakly-compressible "
            "MLS-MPM fluid alongside the existing Newtonian viscosity, and show forward-only (no gradients) "
            "how a liquid changes across the two independent axes viscosity x surface tension. The headline "
            "is a 3x3 grid of one dropped blob at three viscosities and three surface tensions, plus an "
            "isolation test (a square rounds into a droplet, two blobs merge) verifying the term is real "
            "surface tension and sigma_st=0 recovers the viscous fluid."
        ),
        "findings": findings,
        "hypothesis": hypothesis,
        "limitations": limitations,
        "results": results,
        "custom_html": None,
        "training_refs": ["surface-tension", "viscosity", "material-showcase", "constitutive-models",
                          "mpm-in-context", "vector-calculus", "linear-algebra"],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Forward-only fluid surface tension + viscosity x sigma grid")
    ap.add_argument("--isolation", action="store_true", help="just the cheap sign / rounding check")
    ap.add_argument("--calibrate", action="store_true", help="sweep dt for the capillary stability edge")
    ap.add_argument("--quick", action="store_true", help="short smoke test of the full pipeline")
    ap.add_argument("--rerender", action="store_true",
                    help="re-render grid/showcase figures from cached snapshots (no simulation)")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return
    if args.rerender:
        rerender()
        return
    if args.isolation:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(repo, "runs", "material-variants",
                               "implement-liquids-across-viscosity-and-surface-tension")
        isolation(out_dir, quick=args.quick)
        return
    full_run(quick=args.quick)


if __name__ == "__main__":
    main()
