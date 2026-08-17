"""Canonical MLS-MPM physics — the single, frozen source of truth for this project.

Every task IMPORTS this and uses it unchanged; a task that re-derives the MPM step or a material's
parameters is a defect (see CLAUDE.md -> "Canonical physics"). This is what kills ground-truth drift:
there is exactly ONE fluid, ONE elastic, ONE snow, and every task that needs a material as ground truth
(or to learn against) gets the same one.

Scope of this module:
  * The MLS-MPM transfer skeleton (P2G / grid update with Coulomb friction / G2P) at n_grid=128.
  * Four constitutive models, with FROZEN canonical parameters (MAT):
      - fluid   : weakly-compressible pressure from J, plus an optional Newtonian viscous stress
                  mu_visc (C + C^T), plus an optional continuum-surface-force surface tension.
      - elastic : corotated stress from the deformation gradient F (via ti.svd).
      - snow    : elastic + Stomakhin plastic clamp of F's singular values + hardening. The canonical
                  snow CRUMBLES and holds an angle of repose (asserted in physics/signatures.py).
      - sand    : Hencky (log-strain) elasticity + Drucker-Prager return mapping (Klar et al. 2016).
                  Cohesionless and PRESSURE-DEPENDENT: shear strength is proportional to confining
                  pressure, so canonical sand cannot stand a vertical column but DOES hold an angle of
                  repose once it has collapsed into a heap (asserted in physics/signatures.py).
  * Canonical scenes (drop disk, collapse column, two blobs).
  * `simulate(...)` : roll ONE material forward and return position snapshots. NO gradients — ground
    truth is the simplest correct FORWARD sim; nothing here needs autodiff. A task that must optimize
    *through* the physics builds its own differentiable variant and says so.
  * `simulate_multi(...)` : roll SEVERAL materials forward in ONE shared grid, with a per-particle
    material id and a runtime branch. Each material still takes exactly its canonical path; the only
    thing the shared grid forces is a single shared timestep, which is min(dt) over the materials
    present (the stiffest material pays for everyone).

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

FLUID, ELASTIC, SNOW, SAND = 0, 1, 2, 3
MAT_ID = {"fluid": FLUID, "elastic": ELASTIC, "snow": SNOW, "sand": SAND}
N_MAT = 4


def dp_alpha(phi_deg):
    """Drucker-Prager cone slope from a friction angle, in the 2D/3D-agnostic form used by
    Klar et al. 2016 "Drucker-Prager Elastoplasticity for Sand Animation":

        alpha = sqrt(2/3) * 2 sin(phi) / (3 - sin(phi))

    phi is the internal friction angle of the granular pack. It is the ONE parameter that sets how
    steep a slope the material can hold, i.e. the angle of repose. phi = 0 gives alpha = 0, a material
    with no shear strength at all (a fluid that also cannot take tension)."""
    s = np.sin(np.deg2rad(phi_deg))
    return float(np.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s))


# FROZEN canonical per-material parameters. Changing any of these is a deliberate, version-bumping,
# test-gated event (CLAUDE.md -> "Canonical physics" / promotion criteria), never a per-task tweak.
MAT = {
    "fluid":   {"E": 180.0, "dt": 1.2e-4, "xi": 0.0,  "tc": 0.0,    "ts": 0.0,    "phi": 0.0,
                "color": "#4db6ff"},
    "elastic": {"E": 400.0, "dt": 1.0e-4, "xi": 0.0,  "tc": 0.0,    "ts": 0.0,    "phi": 0.0,
                "color": "#ff9d5c"},
    "snow":    {"E": 150.0, "dt": 5.0e-5, "xi": 10.0, "tc": 2.5e-2, "ts": 7.5e-3, "phi": 0.0,
                "color": "#e6ecff"},
    "sand":    {"E": 300.0, "dt": 1.0e-4, "xi": 0.0,  "tc": 0.0,    "ts": 0.0,    "phi": 50.0,
                "color": "#ffd24d"},
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
Jp = ti.field(float, MAX_P)                      # snow: accumulated plastic volume change (starts 1)
                                                 # sand: accumulated plastic VOLUMETRIC STRAIN (starts 0)

grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

x_np_buf = ti.Vector.field(dim, float, MAX_P)
v0_buf = ti.Vector.field(dim, float, MAX_P)

# --- multi-material state: one grid, several materials, a per-particle id and a runtime branch ---
mat_id = ti.field(ti.i32, MAX_P)                 # which constitutive model particle p obeys
p_vol_f = ti.field(float, MAX_P)                 # per-particle volume (differs between groups)
p_mass_f = ti.field(float, MAX_P)
m_E = ti.field(float, N_MAT)                     # per-material params, indexed by material id
m_xi = ti.field(float, N_MAT)
m_tc = ti.field(float, N_MAT)
m_ts = ti.field(float, N_MAT)
m_alpha = ti.field(float, N_MAT)
m_muv = ti.field(float, N_MAT)


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


@ti.func
def hencky_tau(Fc, mu, la):
    """Kirchhoff stress of a St-Venant-Kirchhoff-Hencky solid: energy mu*||eps||^2 + (la/2)*tr(eps)^2
    in the LOG strain eps = ln(sigma) of F = U sigma V^T.

    Why log strain for sand and not the corotated model used for elastic/snow: the Drucker-Prager yield
    surface is a cone in principal-STRESS space, and with a Hencky energy the stress is linear in the
    principal log strains, so the plastic projection onto that cone is a closed-form projection of a
    2-vector. With corotated elasticity the same projection has no closed form.

    Contracting the first Piola stress with F^T collapses to the Kirchhoff stress itself:
        P F^T = U (2 mu eps + la tr(eps) I) U^T
    so no inverse of sigma survives into the transfer (the 1/sigma terms cancel), which is what keeps
    this stable when a particle gets strongly compressed."""
    U, sig, _Vt = ti.svd(Fc)
    e0 = ti.log(ti.max(sig[0, 0], 1e-4))
    e1 = ti.log(ti.max(sig[1, 1], 1e-4))
    tr = e0 + e1
    t0 = 2.0 * mu * e0 + la * tr
    t1 = 2.0 * mu * e1 + la * tr
    return U @ ti.Matrix([[t0, 0.0], [0.0, t1]]) @ U.transpose()


@ti.func
def sand_stress(p, dt, E, p_vol):
    mu = E / (2.0 * (1.0 + NU))
    la = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * hencky_tau(F[p], mu, la)


@ti.func
def dp_return_map(p, F_tr, mu, la, alpha):
    """Drucker-Prager return mapping (Klar et al. 2016, Alg. 3) with the volume correction.

    Sand's defining property is that its shear strength is proportional to the confining pressure and
    that it cannot carry tension at all. Written on the log strain eps of the trial deformation, with
    deviatoric part eps_hat, the admissible set is the cone

        ||eps_hat|| <= -(d*la + 2 mu) / (2 mu) * alpha * tr(eps),      tr(eps) <= 0.

    Three cases, all handled below:
      * expansion (tr(eps) >= 0)  -> project to the TIP of the cone: F_E := U V^T, all stress released.
        The volumetric part that was thrown away is accumulated in Jp[p] so that the same amount of
        compression has to be undone before the grain pack carries stress again. Without this bookkeeping
        the material silently gains volume every time it is thrown apart and never re-packs.
      * inside the cone (dgamma <= 0) -> purely elastic, keep the trial F.
      * outside -> slide the log strain back onto the cone along the deviatoric direction.
    """
    U, sig, Vt = ti.svd(F_tr)
    e0 = ti.log(ti.max(ti.abs(sig[0, 0]), 1e-4))
    e1 = ti.log(ti.max(ti.abs(sig[1, 1]), 1e-4))
    tr = e0 + e1 + Jp[p]
    s0 = 1.0
    s1 = 1.0
    if tr >= 0.0:                                   # cone tip: no tension, remember the lost expansion
        Jp[p] = tr
    else:
        Jp[p] = 0.0
        eh0 = e0 - tr / dim
        eh1 = e1 - tr / dim
        ehn = ti.sqrt(eh0 * eh0 + eh1 * eh1) + 1e-20
        dgamma = ehn + (dim * la + 2.0 * mu) / (2.0 * mu) * tr * alpha
        if dgamma <= 0.0:                           # inside the yield cone -> elastic
            s0 = sig[0, 0]
            s1 = sig[1, 1]
        else:                                       # project back onto the cone
            s0 = ti.exp(e0 - dgamma / ehn * eh0)
            s1 = ti.exp(e1 - dgamma / ehn * eh1)
    return U @ ti.Matrix([[s0, 0.0], [0.0, s1]]) @ Vt


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
        elif ti.static(mat == SNOW):
            stress = snow_stress(p, dt, E, xi, p_vol)
        else:
            stress = sand_stress(p, dt, E, p_vol)
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
def g2p(mat: ti.template(), n: ti.i32, dt: ti.f32, theta_c: ti.f32, theta_s: ti.f32,
        E: ti.f32, alpha: ti.f32):
    for p in range(n):
        new_v, new_C = g2p_gather(p)
        v[p] = new_v
        x[p] = x[p] + dt * new_v
        x[p] = ti.math.clamp(x[p], floor_y, 1.0 - floor_y)
        if ti.static(mat == FLUID):
            J[p] = J[p] * (1.0 + dt * new_C.trace())
        elif ti.static(mat == ELASTIC):
            F[p] = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
        elif ti.static(mat == SNOW):
            F_tr = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
            U, sig, Vt = ti.svd(F_tr)
            s0 = ti.min(ti.max(sig[0, 0], 1.0 - theta_c), 1.0 + theta_s)
            s1 = ti.min(ti.max(sig[1, 1], 1.0 - theta_c), 1.0 + theta_s)
            Jp[p] = Jp[p] * (sig[0, 0] * sig[1, 1]) / (s0 * s1)
            F[p] = U @ ti.Matrix([[s0, 0.0], [0.0, s1]]) @ Vt
        else:
            F_tr = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
            mu = E / (2.0 * (1.0 + NU))
            la = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
            F[p] = dp_return_map(p, F_tr, mu, la, alpha)
        C[p] = new_C


# --------------------------------------------------------------------------- multi-material: one grid
# Same physics, one runtime branch instead of four compile-time ones, so a single grid can carry all
# four materials at once. Every material still takes EXACTLY its canonical path; the parameters come
# from the same frozen MAT table, uploaded into the m_* fields.
@ti.kernel
def p2g_multi(n: ti.i32, dt: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        m = mat_id[p]
        p_vol = p_vol_f[p]
        p_mass = p_mass_f[p]
        stress = ti.Matrix.zero(float, dim, dim)
        if m == FLUID:
            stress = fluid_visc_stress(p, dt, m_E[FLUID], m_muv[FLUID], p_vol)
        elif m == ELASTIC:
            stress = elastic_stress(p, dt, m_E[ELASTIC], p_vol)
        elif m == SNOW:
            stress = snow_stress(p, dt, m_E[SNOW], m_xi[SNOW], p_vol)
        else:
            stress = sand_stress(p, dt, m_E[SAND], p_vol)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass


@ti.kernel
def g2p_multi(n: ti.i32, dt: ti.f32):
    for p in range(n):
        new_v, new_C = g2p_gather(p)
        v[p] = new_v
        x[p] = x[p] + dt * new_v
        x[p] = ti.math.clamp(x[p], floor_y, 1.0 - floor_y)
        m = mat_id[p]
        if m == FLUID:
            J[p] = J[p] * (1.0 + dt * new_C.trace())
        elif m == ELASTIC:
            F[p] = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
        elif m == SNOW:
            F_tr = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
            U, sig, Vt = ti.svd(F_tr)
            s0 = ti.min(ti.max(sig[0, 0], 1.0 - m_tc[SNOW]), 1.0 + m_ts[SNOW])
            s1 = ti.min(ti.max(sig[1, 1], 1.0 - m_tc[SNOW]), 1.0 + m_ts[SNOW])
            Jp[p] = Jp[p] * (sig[0, 0] * sig[1, 1]) / (s0 * s1)
            F[p] = U @ ti.Matrix([[s0, 0.0], [0.0, s1]]) @ Vt
        else:
            F_tr = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
            Es = m_E[SAND]
            mu = Es / (2.0 * (1.0 + NU))
            la = Es * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
            F[p] = dp_return_map(p, F_tr, mu, la, m_alpha[SAND])
        C[p] = new_C


# --------------------------------------------------------------------------- init / upload
@ti.kernel
def init_state(n: ti.i32):
    for p in range(n):
        x[p] = x_np_buf[p]
        v[p] = v0_buf[p]
        C[p] = ti.Matrix.zero(float, dim, dim)
        J[p] = 1.0
        # snow's Jp is a multiplicative plastic VOLUME ratio (starts at 1); sand's is an additive
        # plastic volumetric STRAIN in log space (starts at 0). Same field, different bookkeeping.
        Jp[p] = 0.0 if mat_id[p] == SAND else 1.0
        F[p] = ti.Matrix.identity(float, dim)


def _upload(pts, v0=(0.0, 0.0), mat=FLUID):
    n = pts.shape[0]
    buf = np.zeros((MAX_P, dim), dtype=np.float32)
    buf[:n] = pts.astype(np.float32)
    x_np_buf.from_numpy(buf)
    vb = np.zeros((MAX_P, dim), dtype=np.float32)
    vb[:n] = np.asarray(v0, dtype=np.float32)
    v0_buf.from_numpy(vb)
    mb = np.zeros(MAX_P, dtype=np.int32)
    mb[:n] = mat
    mat_id.from_numpy(mb)
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


def seed_triangle(cx, y0, half_base, height, n, seed=0):
    """Uniform samples in the isoceles triangle standing on the floor with apex above (cx, y0).
    Uniformity comes from barycentric sampling with a sqrt on the first coordinate; sampling the
    bounding box and rejecting would leave a density gradient near the apex at small n."""
    rng = np.random.default_rng(seed)
    r1 = np.sqrt(rng.uniform(0, 1, n))
    r2 = rng.uniform(0, 1, n)
    ax, ay = cx - half_base, y0
    bx, by = cx + half_base, y0
    tx, ty = cx, y0 + height
    xs = (1 - r1) * ax + r1 * (1 - r2) * bx + r1 * r2 * tx
    ys = (1 - r1) * ay + r1 * (1 - r2) * by + r1 * r2 * ty
    return np.stack([xs, ys], axis=1)


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
    if name == "heap":
        # The angle-of-repose test: an OVER-STEEP pile released from rest. Seeded as a 60-degree
        # triangle, which is steeper than any granular material can support, so whatever slope is left
        # at the end is the slope the material genuinely holds. A collapsing column measures runout
        # instead, and its deposit is systematically shallower because the material arrives moving.
        hb = 0.13
        H = hb * np.tan(np.deg2rad(60.0))
        return {"pts": seed_triangle(0.5, floor_y, hb, H, n), "area": hb * H,
                "v0": (0.0, 0.0), "T": 1.6}
    raise KeyError(name)


# --------------------------------------------------------------------------- the forward simulator
def simulate(material, pts, area, T, n_frames, *, v0=(0.0, 0.0), dt=None, E=None, xi=None,
             phi=None, mu_visc=0.0, gravity_on=True):
    """Roll `material` ("fluid"|"elastic"|"snow"|"sand") forward to physical time T from seed `pts`
    (area for density), capturing n_frames snapshots evenly in physical time. Canonical frozen params
    unless overridden. mu_visc is a fluid-only knob (Newtonian viscosity), xi a snow-only one (hardening)
    and phi a sand-only one (friction angle); the last two are exposed so their cost and their effect can
    be measured against the canonical value. Returns (snaps (n_frames,n,2), times, stable)."""
    cfg = MAT[material]
    mid = MAT_ID[material]
    dt = cfg["dt"] if dt is None else dt
    E = cfg["E"] if E is None else E
    xi = cfg["xi"] if xi is None else xi
    alpha = dp_alpha(cfg.get("phi", 0.0) if phi is None else phi)
    grav = 9.8 if gravity_on else 0.0
    n = _upload(pts, v0, mid)
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
            p2g(mid, n, dt, E, xi, mu_visc, p_vol, p_mass)
            grid_op(dt, FRICTION, grav)
            g2p(mid, n, dt, cfg["tc"], cfg["ts"], E, alpha)
            t += dt
        cur = x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        snaps[fidx] = cur
        times[fidx] = t
    return snaps, times, stable


def shared_dt(materials):
    """The timestep a shared grid is forced to run at: one grid means ONE timestep, so the stiffest
    material present pays for every particle in the scene."""
    return min(MAT[m]["dt"] for m in materials)


def simulate_multi(groups, T, n_frames, *, dt=None, gravity_on=True, mu_visc=0.0):
    """Roll SEVERAL materials forward in ONE shared grid.

    groups: list of dicts with keys `material`, `pts` (n_i,2), `area`, and optional `v0`.
    dt defaults to shared_dt(materials present) -- the physically forced choice, not a preference.

    Returns (snaps (n_frames, N, 2), times, mats (N,) int32, stable, dt). Particle order is the
    concatenation of the groups in the order given, so `mats` selects each group's rows.

    Contact between different materials is whatever the shared grid produces (a single velocity field
    per node, so momentum is exchanged as if the node held one blended material). That is not a
    calibrated multi-phase contact model and is not claimed to be one."""
    mats = [g["material"] for g in groups]
    dt = shared_dt(mats) if dt is None else dt
    grav = 9.8 if gravity_on else 0.0

    pts = np.concatenate([np.asarray(g["pts"], dtype=np.float32) for g in groups], 0)
    n = pts.shape[0]
    if n > MAX_P:
        raise ValueError("simulate_multi: %d particles exceeds MAX_P=%d" % (n, MAX_P))
    mid = np.zeros(n, dtype=np.int32)
    v0s = np.zeros((n, dim), dtype=np.float32)
    pvol = np.zeros(n, dtype=np.float32)
    off = 0
    for g in groups:
        k = np.asarray(g["pts"]).shape[0]
        mid[off:off + k] = MAT_ID[g["material"]]
        v0s[off:off + k] = np.asarray(g.get("v0", (0.0, 0.0)), dtype=np.float32)
        pvol[off:off + k] = g["area"] / k
        off += k

    xb = np.zeros((MAX_P, dim), dtype=np.float32); xb[:n] = pts
    vb = np.zeros((MAX_P, dim), dtype=np.float32); vb[:n] = v0s
    mb = np.zeros(MAX_P, dtype=np.int32); mb[:n] = mid
    vo = np.zeros(MAX_P, dtype=np.float32); vo[:n] = pvol
    x_np_buf.from_numpy(xb)
    v0_buf.from_numpy(vb)
    mat_id.from_numpy(mb)
    p_vol_f.from_numpy(vo)
    p_mass_f.from_numpy(vo * p_rho)

    for name, i in MAT_ID.items():
        c = MAT[name]
        m_E[i] = c["E"]; m_xi[i] = c["xi"]; m_tc[i] = c["tc"]; m_ts[i] = c["ts"]
        m_alpha[i] = dp_alpha(c.get("phi", 0.0))
        m_muv[i] = mu_visc if name == "fluid" else 0.0

    spf = max(1, int(round((T / n_frames) / dt)))
    init_state(n)
    snaps = np.zeros((n_frames, n, dim), dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    t = 0.0
    stable = True
    for fidx in range(n_frames):
        for _ in range(spf):
            clear_grid()
            p2g_multi(n, dt)
            grid_op(dt, FRICTION, grav)
            g2p_multi(n, dt)
            t += dt
        cur = x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        snaps[fidx] = cur
        times[fidx] = t
    return snaps, times, mid, stable, dt


# --------------------------------------------------------------------------- shape diagnostics
def spread_width(snap):
    return float(np.percentile(snap[:, 0], 95) - np.percentile(snap[:, 0], 5))


def pile_height(snap):
    return float(np.percentile(snap[:, 1], 95) - floor_y)


def surface_profile(snap, nbins=20, q=90.0, min_count=8):
    """Free-surface height above the floor as a function of x. Bins particles by x between their 1st
    and 99th percentile and takes the q-th percentile of y inside each bin, so a little spray does not
    become "the surface". Returns (bin centres, heights) for bins with enough particles."""
    xs, ys = snap[:, 0], snap[:, 1]
    lo, hi = np.percentile(xs, 1), np.percentile(xs, 99)
    if hi - lo < 1e-6:
        return np.zeros(0), np.zeros(0)
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, nbins - 1)
    cx, hh = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= min_count:
            cx.append(0.5 * (edges[b] + edges[b + 1]))
            hh.append(np.percentile(ys[m], q) - floor_y)
    return np.array(cx), np.array(hh)


def repose_angle(snap, nbins=20, q=90.0, min_count=8):
    """Slope of the settled free surface, in DEGREES: the signature that separates a heap from a puddle.

    The surface profile is split at its apex and a straight line is least-squares fitted to each whole
    flank; the reported angle is atan of the mean |slope| over the flanks that have at least 3 bins.
    Fitting the WHOLE flank rather than the steep part is deliberate: a flat puddle also has a sharp
    drop at its rim, and only a full-flank fit tells "flat with an edge" apart from "sloped all the way
    down"."""
    cx, hh = surface_profile(snap, nbins, q, min_count)
    if cx.size < 6:
        return 0.0
    apex = int(np.argmax(hh))
    slopes = []
    for sl in (slice(0, apex + 1), slice(apex, len(cx))):
        if len(cx[sl]) >= 3:
            slopes.append(abs(np.polyfit(cx[sl], hh[sl], 1)[0]))
    if not slopes:
        return 0.0
    return float(np.degrees(np.arctan(float(np.mean(slopes)))))


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
