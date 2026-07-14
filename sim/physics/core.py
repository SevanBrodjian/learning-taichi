"""Canonical MLS-MPM physics — the single, frozen source of truth for this project.

Every task IMPORTS this and uses it unchanged; a task that re-derives the MPM step or a material's
parameters is a defect (see CLAUDE.md -> "Canonical physics"). This is what kills ground-truth drift:
there is exactly ONE fluid, ONE elastic, ONE snow, and every task that needs a material as ground truth
(or to learn against) gets the same one.

Scope of this module:
  * The MLS-MPM transfer skeleton (P2G / grid update with Coulomb friction / G2P) at n_grid=128.
  * Three constitutive models, with FROZEN canonical parameters (MAT):
      - fluid   : weakly-compressible pressure from J, plus an optional Newtonian viscous stress
                  mu_visc (C + C^T), plus an optional continuum-surface-force surface tension.
      - elastic : corotated stress from the deformation gradient F (via ti.svd).
      - snow    : elastic + Stomakhin plastic clamp of F's singular values + hardening. The canonical
                  snow CRUMBLES and holds an angle of repose (asserted in physics/signatures.py).
  * Canonical scenes (drop disk, collapse column, two blobs).
  * `simulate(...)` : roll a material forward and return position snapshots. NO gradients — ground truth
    is the simplest correct FORWARD sim; nothing here needs autodiff. A task that must optimize *through*
    the physics builds its own differentiable variant and says so.

Building-block `@ti.func`s (fluid_visc_stress, corotated_PFt, ...) are exported so a learned-dynamics
task can reuse the transfer skeleton while swapping only the stress or the whole update — the SEAM it
replaces is declared in the task, not re-implemented here.

This module owns `ti.init`; import it before declaring your own Taichi kernels in the same process.
"""
# NOTE: do NOT add `from __future__ import annotations` here — Taichi introspects real annotation
# objects (e.g. ti.template()) on its @ti.kernel signatures, and stringized annotations break that.
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

# --------------------------------------------------------------------------- world constants (frozen)
dim = 2
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
p_rho = 1.0
gravity = 9.8
bound = 3
floor_y = bound * dx
NU = 0.2                 # Poisson ratio, fixed for every solid path
FRICTION = 0.5           # Coulomb friction at the floor (what lets snow hold an angle of repose)

MAX_P = 16384

FLUID, ELASTIC, SNOW = 0, 1, 2
MAT_ID = {"fluid": FLUID, "elastic": ELASTIC, "snow": SNOW}

# FROZEN canonical per-material parameters. Changing any of these is a deliberate, version-bumping,
# test-gated event (CLAUDE.md -> "Canonical physics" / promotion criteria), never a per-task tweak.
MAT = {
    "fluid":   {"E": 180.0, "dt": 1.2e-4, "xi": 0.0,  "tc": 0.0,    "ts": 0.0,    "color": "#4db6ff"},
    "elastic": {"E": 400.0, "dt": 1.0e-4, "xi": 0.0,  "tc": 0.0,    "ts": 0.0,    "color": "#ff9d5c"},
    "snow":    {"E": 150.0, "dt": 5.0e-5, "xi": 10.0, "tc": 2.5e-2, "ts": 7.5e-3, "color": "#e6ecff"},
}
E_FLUID = MAT["fluid"]["E"]

# NOTE: surface tension (the continuum-surface-force capillary term) is a FLUID knob that is NOT yet
# part of the canonical library — its working implementation lives in sim/fluid_surface_tension.py and
# is a documented "promote later" item (needs a careful port + a golden test before it is canonical).
# Do not add it here without a passing signature test.

# --------------------------------------------------------------------------- state fields
x = ti.Vector.field(dim, float, MAX_P)
v = ti.Vector.field(dim, float, MAX_P)
C = ti.Matrix.field(dim, dim, float, MAX_P)
J = ti.field(float, MAX_P)                       # volume ratio (fluid path)
F = ti.Matrix.field(dim, dim, float, MAX_P)      # deformation gradient (elastic / snow path)
Jp = ti.field(float, MAX_P)                      # accumulated plastic volume change (snow)

grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

x_np_buf = ti.Vector.field(dim, float, MAX_P)
v0_buf = ti.Vector.field(dim, float, MAX_P)


# --------------------------------------------------------------------------- constitutive stress
@ti.func
def fluid_visc_stress(p, dt, E, mu_visc, p_vol):
    """Weakly-compressible pressure E(J-1) plus a Newtonian viscous stress mu_visc (C + C^T), scaled by
    the MLS-MPM affine prefactor. mu_visc = 0 recovers the inviscid fluid exactly."""
    pressure = E * (J[p] - 1.0)
    Cp = C[p]
    strain_rate = Cp + Cp.transpose()
    sigma = pressure * ti.Matrix.identity(float, dim) + mu_visc * strain_rate
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * sigma


@ti.func
def corotated_PFt(Fc, mu, la):
    """Corotated first-Piola stress contracted with F^T: 2 mu (F-R) F^T + la (J-1) J I, R = U V^T."""
    U, sig, Vt = ti.svd(Fc)
    R = U @ Vt.transpose()
    Jdet = Fc.determinant()
    return 2.0 * mu * (Fc - R) @ Fc.transpose() + la * (Jdet - 1.0) * Jdet * ti.Matrix.identity(float, dim)


@ti.func
def elastic_stress(p, dt, E, p_vol):
    mu = E / (2.0 * (1.0 + NU))
    la = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * corotated_PFt(F[p], mu, la)


@ti.func
def snow_stress(p, dt, E, xi, p_vol):
    h = ti.exp(xi * (1.0 - Jp[p]))          # hardening: compacted snow (Jp<1) stiffens
    mu = (E / (2.0 * (1.0 + NU))) * h
    la = (E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))) * h
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * corotated_PFt(F[p], mu, la)


# --------------------------------------------------------------------------- MLS-MPM steps
@ti.kernel
def clear_grid():
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_v[i, j] = ti.Vector.zero(float, dim)
        grid_m[i, j] = 0.0


@ti.kernel
def p2g(mat: ti.template(), n: ti.i32, dt: ti.f32, E: ti.f32, xi: ti.f32,
        mu_visc: ti.f32, p_vol: ti.f32, p_mass: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = ti.Matrix.zero(float, dim, dim)
        if ti.static(mat == FLUID):
            stress = fluid_visc_stress(p, dt, E, mu_visc, p_vol)
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
    r = vt
    if vt > 0:
        r = ti.max(0.0, vt - cap)
    elif vt < 0:
        r = ti.min(0.0, vt + cap)
    return r


@ti.kernel
def grid_op(dt: ti.f32, fric: ti.f32, grav: ti.f32):
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        if m > 0.0:
            grid_v[i, j] = grid_v[i, j] / m
        grid_v[i, j].y -= dt * grav
        vx = grid_v[i, j].x
        vy = grid_v[i, j].y
        if j < bound and vy < 0:                 # floor: separating, Coulomb friction on the tangent
            vx = coulomb(vx, fric * (-vy))
            vy = 0.0
        if j > n_grid - bound and vy > 0:        # ceiling: separating
            vy = 0.0
        if i < bound and vx < 0:                 # sticky side walls
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
        x[p] = ti.math.clamp(x[p], floor_y, 1.0 - floor_y)
        if ti.static(mat == FLUID):
            J[p] = J[p] * (1.0 + dt * new_C.trace())
        elif ti.static(mat == ELASTIC):
            F[p] = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
        else:
            F_tr = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
            U, sig, Vt = ti.svd(F_tr)
            s0 = ti.min(ti.max(sig[0, 0], 1.0 - theta_c), 1.0 + theta_s)
            s1 = ti.min(ti.max(sig[1, 1], 1.0 - theta_c), 1.0 + theta_s)
            Jp[p] = Jp[p] * (sig[0, 0] * sig[1, 1]) / (s0 * s1)
            F[p] = U @ ti.Matrix([[s0, 0.0], [0.0, s1]]) @ Vt
        C[p] = new_C


# --------------------------------------------------------------------------- init / upload
@ti.kernel
def init_state(n: ti.i32):
    for p in range(n):
        x[p] = x_np_buf[p]
        v[p] = v0_buf[p]
        C[p] = ti.Matrix.zero(float, dim, dim)
        J[p] = 1.0
        Jp[p] = 1.0
        F[p] = ti.Matrix.identity(float, dim)


def _upload(pts, v0=(0.0, 0.0)):
    n = pts.shape[0]
    buf = np.zeros((MAX_P, dim), dtype=np.float32)
    buf[:n] = pts.astype(np.float32)
    x_np_buf.from_numpy(buf)
    vb = np.zeros((MAX_P, dim), dtype=np.float32)
    vb[:n] = np.asarray(v0, dtype=np.float32)
    v0_buf.from_numpy(vb)
    return n


# --------------------------------------------------------------------------- canonical scenes
def seed_disk(center, radius, n, seed=0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)], axis=1)


def seed_box(x0, x1, y0, y1, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], axis=1)


def scene(name, n=9000):
    """A canonical initial condition. Returns dict(pts, area, v0, T)."""
    if name == "drop":
        return {"pts": seed_disk((0.5, 0.52), 0.11, n), "area": np.pi * 0.11 ** 2,
                "v0": (0.0, 0.0), "T": 1.3}
    if name == "column":
        return {"pts": seed_box(0.42, 0.58, floor_y, 0.56, n), "area": (0.58 - 0.42) * (0.56 - floor_y),
                "v0": (0.0, 0.0), "T": 1.7}
    if name == "two_blobs":
        a = seed_disk((0.42, 0.5), 0.07, n // 2)
        b = seed_disk((0.58, 0.5), 0.07, n - n // 2)
        return {"pts": np.concatenate([a, b], 0), "area": 2 * np.pi * 0.07 ** 2, "v0": (0.0, 0.0), "T": 1.0}
    raise KeyError(name)


# --------------------------------------------------------------------------- the forward simulator
def simulate(material, pts, area, T, n_frames, *, v0=(0.0, 0.0), dt=None, E=None,
             mu_visc=0.0, gravity_on=True):
    """Roll `material` ("fluid"|"elastic"|"snow") forward to physical time T from seed `pts` (area for
    density), capturing n_frames snapshots evenly in physical time. Canonical frozen params unless
    overridden. mu_visc is a fluid-only knob (Newtonian viscosity). Returns (snaps (n_frames,n,2),
    times, stable)."""
    cfg = MAT[material]
    mat_id = MAT_ID[material]
    dt = cfg["dt"] if dt is None else dt
    E = cfg["E"] if E is None else E
    grav = 9.8 if gravity_on else 0.0
    n = _upload(pts, v0)
    p_vol = area / n
    p_mass = p_vol * p_rho
    spf = max(1, int(round((T / n_frames) / dt)))
    init_state(n)
    snaps = np.zeros((n_frames, n, dim), dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    t = 0.0
    stable = True
    for fidx in range(n_frames):
        for _ in range(spf):
            clear_grid()
            p2g(mat_id, n, dt, E, cfg["xi"], mu_visc, p_vol, p_mass)
            grid_op(dt, FRICTION, grav)
            g2p(mat_id, n, dt, cfg["tc"], cfg["ts"])
            t += dt
        cur = x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        snaps[fidx] = cur
        times[fidx] = t
    return snaps, times, stable


# --------------------------------------------------------------------------- shape diagnostics
def spread_width(snap):
    return float(np.percentile(snap[:, 0], 95) - np.percentile(snap[:, 0], 5))


def pile_height(snap):
    return float(np.percentile(snap[:, 1], 95) - floor_y)


def circularity(snap, res=96):
    """Isoperimetric roundness of the occupied region (1 = disk). Used to check surface tension."""
    xs = np.clip(((snap[:, 0]) * res).astype(int), 0, res - 1)
    ys = np.clip(((snap[:, 1]) * res).astype(int), 0, res - 1)
    occ = np.zeros((res, res), bool)
    occ[xs, ys] = True
    area = occ.sum()
    if area < 4:
        return 0.0
    edge = occ & ~(
        np.roll(occ, 1, 0) & np.roll(occ, -1, 0) & np.roll(occ, 1, 1) & np.roll(occ, -1, 1))
    per = edge.sum()
    return float(min(1.0, 4 * np.pi * area / (per * per + 1e-9)))
