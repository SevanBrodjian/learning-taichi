"""Canonical MLS-MPM physics — the single, frozen source of truth for this project.

Every task IMPORTS this and uses it unchanged; a task that re-derives the MPM step or a material's
parameters is a defect (see CLAUDE.md -> "Canonical physics"). This is what kills ground-truth drift:
there is exactly ONE fluid, ONE elastic, ONE snow, and every task that needs a material as ground truth
(or to learn against) gets the same one.

Scope of this module:
  * The MLS-MPM transfer skeleton (P2G / grid update with Coulomb friction / G2P) at n_grid=128.
  * PER-MATERIAL DENSITY. Every material carries `rho`, so a particle's mass is p_vol*rho and a heavy
    material is genuinely heavier than a light one on the shared grid. Nothing applies a buoyancy
    force: sinking and floating fall out of the mass ratio alone (see the MAT comment below).
  * PER-MATERIAL Poisson ratio `nu` (how incompressible the solid is) and per-material floor/wall
    friction `fric`, both of which used to be single global constants shared by every material.
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
gravity = 9.8
bound = 3
floor_y = bound * dx

# The three DEFAULTS below exist only as a fallback for code that names no material. Every canonical
# material overrides all three in MAT, and the simulators read the material's value, not these.
p_rho = 1.0              # density  -> MAT[...]["rho"]
NU = 0.2                 # Poisson ratio -> MAT[...]["nu"]
FRICTION = 0.5           # Coulomb friction at the floor and the side walls -> MAT[...]["fric"]

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
#
# rho  -- density, with water as the unit. This is what makes sand and rubber sink and snow float when
#         they share a grid. There is NO buoyancy force anywhere in this file: a particle's mass is
#         p_vol*rho, P2G scatters that mass to the grid, and the grid update divides the scattered
#         momentum by it. Gravity is applied to the VELOCITY, so it accelerates every node equally,
#         while the surrounding fluid's pressure reaches a node as an impulse divided by that node's
#         mass. A heavy node therefore feels less of the fluid's upward push and sinks; a light one
#         feels more and rises. Archimedes is an OUTPUT here, not an input.
#
# WHY E MOVED WHEN rho WAS INTRODUCED. One material on its own is EXACTLY invariant under
# (rho, E) -> (k rho, k E). The momentum balance rho Dv/Dt = div(sigma) + rho g has sigma proportional
# to E, so only E/rho survives; in the discrete transfer the stress reaches the grid divided by the
# node mass, which is proportional to rho, so it enters as E/rho there too. Absolute density is
# unobservable for a lone material and only becomes physical when two materials share a grid. So snow,
# sand and elastic keep their old E/rho exactly and their solo behaviour does not move (asserted by a
# golden signature); the old E numbers live on as E/rho -- snow 150, sand 300.
#
# nu   -- Poisson ratio: how much the material resists a change of VOLUME as opposed to a change of
#         shape. la = E nu / ((1+nu)(1-2nu)) diverges as nu -> 1/2, which is what "incompressible"
#         means numerically, and it is why rubber runs a smaller timestep than a squashy solid would.
#         Rubber is nearly incompressible in reality (nu ~ 0.5); the granular/plastic materials are not.
# fric -- Coulomb friction coefficient at the floor and the side walls. Water is frictionless against a
#         smooth boundary; a granular pack is not. This used to be one global number, which made water
#         drag along the floor and glue itself to the walls.
MAT = {
    # E/rho = 900: five times the old 180. The weakly-compressible fluid's density varies like
    # rho v^2 / E, so the old value let particles compress by tens of percent on impact, which is what
    # "mushy" looked like. fric = 0 is the other half: water does not grip a smooth floor.
    "fluid":   {"E": 900.0, "rho": 1.0, "nu": 0.20, "fric": 0.0, "dt": 5.0e-5,
                "xi": 0.0,  "tc": 0.0,    "ts": 0.0,    "phi": 0.0,  "color": "#4db6ff"},
    # rubber: nu 0.45 (was the global 0.20). E is raised so that mu/rho -- the SHEAR response, which is
    # what the material's shape dynamics actually depend on -- is unchanged: mu = E/(2(1+nu)) = 200 at
    # rho = 1.2 matches the old mu = 166.7 at rho = 1. Only the volumetric stiffness la went up.
    "elastic": {"E": 580.0, "rho": 1.2, "nu": 0.45, "fric": 0.5, "dt": 5.0e-5,
                "xi": 0.0,  "tc": 0.0,    "ts": 0.0,    "phi": 0.0,  "color": "#ff9d5c"},
    # settled snow is about 0.3x the density of water, so it floats. E/rho = 150, exactly as before.
    "snow":    {"E": 45.0,  "rho": 0.3, "nu": 0.20, "fric": 0.5, "dt": 5.0e-5,
                "xi": 10.0, "tc": 2.5e-2, "ts": 7.5e-3, "phi": 0.0,  "color": "#e6ecff"},
    # dry sand packs at about 1.6x water, so it sinks. E/rho = 300, exactly as before.
    "sand":    {"E": 480.0, "rho": 1.6, "nu": 0.20, "fric": 0.5, "dt": 1.0e-4,
                "xi": 0.0,  "tc": 0.0,    "ts": 0.0,    "phi": 50.0, "color": "#ffd24d"},
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
# Mass-weighted friction scattered to the grid alongside the mass, so a node shared by two materials
# gets the friction of whatever is actually sitting on it. grid_fr/grid_m is the node's coefficient.
grid_fr = ti.field(float, (n_grid, n_grid))

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
m_nu = ti.field(float, N_MAT)
m_fric = ti.field(float, N_MAT)


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
def elastic_stress(p, dt, E, nu, p_vol):
    mu = E / (2.0 * (1.0 + nu))
    la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * corotated_PFt(F[p], mu, la)


@ti.func
def snow_stress(p, dt, E, nu, xi, p_vol):
    h = ti.exp(xi * (1.0 - Jp[p]))          # hardening: compacted snow (Jp<1) stiffens
    mu = (E / (2.0 * (1.0 + nu))) * h
    la = (E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))) * h
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
def sand_stress(p, dt, E, nu, p_vol):
    mu = E / (2.0 * (1.0 + nu))
    la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
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
        grid_fr[i, j] = 0.0


@ti.kernel
def p2g(mat: ti.template(), n: ti.i32, dt: ti.f32, E: ti.f32, nu: ti.f32, xi: ti.f32,
        mu_visc: ti.f32, p_vol: ti.f32, p_mass: ti.f32, fric: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = ti.Matrix.zero(float, dim, dim)
        if ti.static(mat == FLUID):
            stress = fluid_visc_stress(p, dt, E, mu_visc, p_vol)
        elif ti.static(mat == ELASTIC):
            stress = elastic_stress(p, dt, E, nu, p_vol)
        elif ti.static(mat == SNOW):
            stress = snow_stress(p, dt, E, nu, xi, p_vol)
        else:
            stress = sand_stress(p, dt, E, nu, p_vol)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass
            grid_fr[base[0] + i, base[1] + j] += weight * p_mass * fric


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
    """Grid update. `fric` is only the fallback for a node that carries no mass (and therefore no
    material and no result); every node that holds material uses grid_fr/grid_m, the mass-weighted
    friction of what is actually sitting on it.

    All four boundaries get the SAME treatment: separating in the normal direction, Coulomb friction on
    the tangent. The side walls used to zero BOTH components, which glued material to them -- water
    thrown against a wall could not slide back down it, which is the artefact that reads as water being
    sticky. A wall is a wall, not glue, and it is the material's own `fric` that decides how much it
    drags along one."""
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        f = fric
        if m > 0.0:
            grid_v[i, j] = grid_v[i, j] / m
            f = grid_fr[i, j] / m
        grid_v[i, j].y -= dt * grav
        vx = grid_v[i, j].x
        vy = grid_v[i, j].y
        if j < bound and vy < 0:                 # floor: separating, Coulomb friction on the tangent
            vx = coulomb(vx, f * (-vy))
            vy = 0.0
        if j > n_grid - bound and vy > 0:        # ceiling: separating
            vx = coulomb(vx, f * vy)
            vy = 0.0
        if i < bound and vx < 0:                 # left wall: separating, NOT glued
            vy = coulomb(vy, f * (-vx))
            vx = 0.0
        if i > n_grid - bound and vx > 0:        # right wall
            vy = coulomb(vy, f * vx)
            vx = 0.0
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
        E: ti.f32, nu: ti.f32, alpha: ti.f32):
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
            mu = E / (2.0 * (1.0 + nu))
            la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
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
            stress = elastic_stress(p, dt, m_E[ELASTIC], m_nu[ELASTIC], p_vol)
        elif m == SNOW:
            stress = snow_stress(p, dt, m_E[SNOW], m_nu[SNOW], m_xi[SNOW], p_vol)
        else:
            stress = sand_stress(p, dt, m_E[SAND], m_nu[SAND], p_vol)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass
            grid_fr[base[0] + i, base[1] + j] += weight * p_mass * m_fric[m]


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
            nus = m_nu[SAND]
            mu = Es / (2.0 * (1.0 + nus))
            la = Es * nus / ((1.0 + nus) * (1.0 - 2.0 * nus))
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


def seed_lattice(x0, x1, y0, y1, n, seed=0, jitter=0.25):
    """A jittered regular lattice covering the box, with roughly `n` points.

    Use this instead of `seed_box` for a body of fluid that is supposed to START AT REST. A uniform
    random sample has Poisson clumping at the sub-cell scale, and the weakly-compressible fluid has no
    way to push back on it: its pressure comes from the advected volume ratio J, not from the actual
    particle packing, so a randomly seeded pool quietly compacts as it settles and its free surface
    creeps downward for the whole run. Seeding on a lattice starts the pack near its own rest density
    and cuts that drift by more than half. The jitter (a fraction of the lattice spacing) breaks the
    grid alignment that would otherwise show up as banding artefacts in the transfer."""
    a = (x1 - x0) * (y1 - y0)
    s = np.sqrt(a / max(n, 1))
    xs = np.arange(x0 + s / 2, x1, s)
    ys = np.arange(y0 + s / 2, y1, s)
    X, Y = np.meshgrid(xs, ys)
    p = np.stack([X.ravel(), Y.ravel()], axis=1)
    rng = np.random.default_rng(seed)
    return p + rng.uniform(-jitter * s, jitter * s, p.shape)


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
    if name == "slam":
        # A HARD floor impact: the drop disk released from higher up and given a downward kick. Gentle
        # settling barely compresses anything, so a solid's VOLUMETRIC response -- the thing the Poisson
        # ratio governs -- is only visible in a scene carrying enough kinetic energy to squash it.
        return {"pts": seed_disk((0.5, 0.60), 0.11, n), "area": np.pi * 0.11 ** 2,
                "v0": (0.0, -6.0), "T": 1.0}
    if name == "dam":
        # A ONE-SIDED dam break: a block held against the left wall and released. Unlike the symmetric
        # `column`, which reaches both side walls almost at once, this measures RUNOUT -- how far the
        # leading front travels before it stops -- which is what separates a material that slides along
        # the floor from one that drags on it.
        xr, yt = 0.22, 0.42
        return {"pts": seed_box(floor_y, xr, floor_y, yt, n),
                "area": (xr - floor_y) * (yt - floor_y), "v0": (0.0, 0.0), "T": 1.4}
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


def scene_pool(solid, n=9000, *, depth=0.34, blob_r=0.075, blob_cx=0.5, blob_cy=0.20,
               x0=None, x1=None, T=2.0, seed=0, rho=None):
    """The canonical BUOYANCY scene: a disk of `solid`, at rest, fully submerged at mid-depth in a pool
    of water, with the water seeded AROUND it so neither phase starts overlapping the other.

    Starting the blob submerged and at rest is deliberate. Dropping it in from above would measure a
    splash, and whatever the blob did afterwards would be confounded with the momentum it arrived with.
    Released at rest inside the pool, the only thing that can move it is the balance between its weight
    and the fluid pressure around it, which is exactly the quantity under test.

    Both groups are given the same particle density (particles per unit area), so `p_vol` matches across
    the interface and one phase is not silently resolved better than the other.

    `rho` overrides the solid's density, which is what turns this scene into a controlled experiment:
    the same material at three densities must float, hover and sink.

    The pool spans the whole tank by default, so the water starts at rest against both walls instead of
    collapsing sideways and sloshing -- a wave crossing the tank would move the free surface by more than
    the effect being measured.

    Returns dict(groups, T, water_area, solid_area, n_solid).
    """
    x0 = floor_y if x0 is None else x0
    x1 = 1.0 - floor_y if x1 is None else x1
    box_area = (x1 - x0) * (depth - floor_y)
    disk_area = np.pi * blob_r ** 2
    n_solid = max(120, int(round(n * disk_area / box_area)))
    lat = seed_lattice(x0, x1, floor_y, depth, n, seed=seed)
    water = lat[np.hypot(lat[:, 0] - blob_cx, lat[:, 1] - blob_cy) > blob_r * 1.05]
    blob = seed_disk((blob_cx, blob_cy), blob_r, n_solid, seed=seed + 1)
    g_solid = {"material": solid, "pts": blob, "area": disk_area, "v0": (0.0, 0.0)}
    if rho is not None:
        g_solid["rho"] = float(rho)
    return {"groups": [{"material": "fluid", "pts": water, "area": box_area - disk_area,
                        "v0": (0.0, 0.0)}, g_solid],
            "T": T, "water_area": box_area - disk_area, "solid_area": disk_area, "n_solid": n_solid}


# --------------------------------------------------------------------------- the forward simulator
def simulate(material, pts, area, T, n_frames, *, v0=(0.0, 0.0), dt=None, E=None, xi=None,
             phi=None, nu=None, rho=None, fric=None, mu_visc=0.0, gravity_on=True):
    """Roll `material` ("fluid"|"elastic"|"snow"|"sand") forward to physical time T from seed `pts`
    (whose `area` fixes the per-particle volume), capturing n_frames snapshots evenly in physical time.
    Canonical frozen params unless overridden. mu_visc is a fluid-only knob (Newtonian viscosity), xi a
    snow-only one (hardening) and phi a sand-only one (friction angle); nu, rho and fric are exposed so
    that the Poisson ratio, the density and the boundary friction can each be measured against the
    canonical value. Returns (snaps (n_frames,n,2), times, stable)."""
    cfg = MAT[material]
    mid = MAT_ID[material]
    dt = cfg["dt"] if dt is None else dt
    E = cfg["E"] if E is None else E
    xi = cfg["xi"] if xi is None else xi
    nu = cfg.get("nu", NU) if nu is None else nu
    rho = cfg.get("rho", p_rho) if rho is None else rho
    fric = cfg.get("fric", FRICTION) if fric is None else fric
    alpha = dp_alpha(cfg.get("phi", 0.0) if phi is None else phi)
    grav = 9.8 if gravity_on else 0.0
    n = _upload(pts, v0, mid)
    p_vol = area / n
    p_mass = p_vol * rho
    spf = max(1, int(round((T / n_frames) / dt)))
    init_state(n)
    snaps = np.zeros((n_frames, n, dim), dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    t = 0.0
    stable = True
    for fidx in range(n_frames):
        for _ in range(spf):
            clear_grid()
            p2g(mid, n, dt, E, nu, xi, mu_visc, p_vol, p_mass, fric)
            grid_op(dt, fric, grav)
            g2p(mid, n, dt, cfg["tc"], cfg["ts"], E, nu, alpha)
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

    groups: list of dicts with keys `material`, `pts` (n_i,2), `area`, and optional `v0` and `rho`.
    dt defaults to shared_dt(materials present) -- the physically forced choice, not a preference.

    `rho` overrides that group's canonical density. It exists so a buoyancy result can be shown to
    depend on DENSITY and nothing else: the same material, same stiffness, same scene, three densities.

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
    prho = np.zeros(n, dtype=np.float32)
    off = 0
    for g in groups:
        k = np.asarray(g["pts"]).shape[0]
        mid[off:off + k] = MAT_ID[g["material"]]
        v0s[off:off + k] = np.asarray(g.get("v0", (0.0, 0.0)), dtype=np.float32)
        pvol[off:off + k] = g["area"] / k
        prho[off:off + k] = g.get("rho", MAT[g["material"]].get("rho", p_rho))
        off += k

    xb = np.zeros((MAX_P, dim), dtype=np.float32); xb[:n] = pts
    vb = np.zeros((MAX_P, dim), dtype=np.float32); vb[:n] = v0s
    mb = np.zeros(MAX_P, dtype=np.int32); mb[:n] = mid
    vo = np.zeros(MAX_P, dtype=np.float32); vo[:n] = pvol
    ro = np.zeros(MAX_P, dtype=np.float32); ro[:n] = prho
    x_np_buf.from_numpy(xb)
    v0_buf.from_numpy(vb)
    mat_id.from_numpy(mb)
    p_vol_f.from_numpy(vo)
    p_mass_f.from_numpy(vo * ro)          # THE line that makes one material heavier than another

    for name, i in MAT_ID.items():
        c = MAT[name]
        m_E[i] = c["E"]; m_xi[i] = c["xi"]; m_tc[i] = c["tc"]; m_ts[i] = c["ts"]
        m_alpha[i] = dp_alpha(c.get("phi", 0.0))
        m_nu[i] = c.get("nu", NU)
        m_fric[i] = c.get("fric", FRICTION)
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


def waterline(fluid_snap, q=97.0):
    """Height of a pool's free surface: the q-th percentile of the fluid particles' y. The percentile
    rather than the max, so a little spray thrown off the top does not become "the surface"."""
    return float(np.percentile(fluid_snap[:, 1], q))


def submerged_fraction(solid_snap, fluid_snap, q=97.0):
    """Fraction of a body's particles sitting below the free surface of the fluid around it.

    This is the direct read-out of Archimedes' principle. A body at rest floating in equilibrium
    displaces its own weight, so the submerged fraction settles at rho_solid/rho_fluid; a body denser
    than the fluid cannot reach equilibrium at all and ends fully submerged at 1."""
    return float((solid_snap[:, 1] < waterline(fluid_snap, q)).mean())


def rest_depth(solid_snap, fluid_snap, q=97.0):
    """How deep a body has settled: the waterline minus the body's mean height, in domain units.
    Positive means the body's centre is under water, negative means it is riding on top."""
    return float(waterline(fluid_snap, q) - solid_snap[:, 1].mean())


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
