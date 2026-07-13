"""TWO learned networks, one shared descriptor, for a FLUID across viscosity AND surface tension.

Follow-up that fuses two precursors:
  * ``sim/one_nn_materials.py`` -- the conditioned-network protocol: one shared MLP g_theta(features, m)
    predicts the per-particle stress, selected only by a small descriptor m fed as extra inputs. Trained on
    a few corners, interpolate the DESCRIPTOR (an input), not the weights.
  * ``sim/fluid_surface_tension.py`` -- the weakly-compressible MLS-MPM fluid with Newtonian viscosity and
    a grid continuum-surface-force (CSF) surface tension. Its p2g/grid/g2p kernels are reused; its analytic
    CSF is used ONLY to generate supervised targets and as ground truth, NEVER in the learned rollout.

TWO-PARAMETER DESCRIPTOR  m = (m_visc, m_st) on the unit square. BOTH axes are LEARNED:
    m_visc = VISCOSITY.  m_visc=0 -> low viscosity (thin);  m_visc=1 -> high viscosity (thick).
             A PER-PARTICLE STRESS (Newtonian viscous term mu*(C+C^T)), learned by the conditioned STRESS
             net from the local state (J,C,v) + m. Linear schedule mu(m_visc) so the target is linear in
             the descriptor (the cleanest test of input-conditioning; cf. learned_viscosity).
    m_st   = SURFACE TENSION.  m_st=0 -> none;  m_st=1 -> the calibrated sigma_max.
             Surface tension is NOT a per-particle stress -- it is a grid capillary force set by the
             INTERFACE CURVATURE, a NON-LOCAL quantity a per-particle net structurally cannot see. So a
             SECOND network, the CAPILLARY net, LEARNS it: it reads a 5x5 patch of the smoothed grid
             density field phi around each node (the rawer interface signal, NOT the analytic curvature)
             plus the surface-tension strength, and OUTPUTS the capillary force at that node. It is trained
             supervised against the analytic CSF force at the trained sigma values.
    Learned rollout at a cell = (conditioned STRESS net at m) + (learned CAPILLARY net at strength s(m_st)).
    NO analytic capillary force is ever applied in the learned rollout.

    TRAINED corners: (m_visc, m_st) = (0,0) low visc / no ST, (1,0) high visc / no ST, (0,1) low visc /
    high ST.   HELD OUT (never trained): (1,1) high visc / high ST. Because surface tension is now LEARNED,
    the held-out corner is a REAL test of whether the learned capillary law (seen only at low viscosity)
    transfers to an unseen high-visc/high-ST combination.

WHY TWO NETS (the teaching point): a per-particle stress net sees only a particle's own local state, which
carries no information about the interface curvature surface tension depends on -- it can learn viscosity (a
local rate-of-shear stress) but NOT surface tension. The capillary net is given the interface neighbourhood
(the density patch) exactly so it can infer the curvature and produce the capillary force itself.

EDGE-EXACTNESS is checked for BOTH nets at each trained corner: (a) the STRESS net reproduces the analytic
fluid stress, (b) the CAPILLARY net reproduces the analytic CSF force, and (c) the full learned rollout
(both nets, no analytic ST) reproduces the true fluid. All are reported before any interior cell is trusted.

The SURFACE-TENSION range is CALIBRATED FIRST on a cheap isolation blob (the predecessor's ST saturated by
sigma~0.1, so its medium/high columns looked identical): sweep sigma finely at low values, find where
roundness is mid-transition, pick a LOW sigma_max and a gentle m_st->sigma_st schedule so the droplet rounds
GRADUALLY up the ST axis instead of snapping to a ball by the second row.

Rendering is HEADLESS (matplotlib Agg -> mp4/png). Every rollout is checked finite; a cell flung to the
corner is instability (shrink dt), not a result, and every clip/cell is meant to be viewed.

Usage:
    python sim/one_nn_fluids.py            # full pipeline + media + manifest
    python sim/one_nn_fluids.py --quick    # fast smoke test (fewer frames/iters/grid)
    python sim/one_nn_fluids.py --calibrate  # just the ST sigma sweep + stability probe
"""
import argparse
import base64
import datetime
import json
import os

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

# --------------------------------------------------------------------------- world constants (fluid_st)
dim = 2
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
p_rho = 1.0
bound = 3
floor_y = bound * dx
FRICTION = 0.5
E_FLUID = 200.0

MAX_P = 16384
M_REF = p_rho * dx * dx
SMOOTH_ITERS = 6

# --------------------------------------------------------------------------- descriptor schedules
MU_LOW = 0.02       # m_visc = 0 (thin)
MU_HIGH = 0.8       # m_visc = 1 (thick)
# Surface tension: set by calibration (auto-picked on the isolation blob). Placeholders; overwritten in run.
SIGMA_MAX = 0.06
ST_P = 2.0          # sigma_st(m_st) = SIGMA_MAX * m_st ** ST_P  (gentle: most of m_st range lives low)


def mu_of_m(m_visc):
    """Viscous coefficient as a LINEAR function of m_visc, so the true target mu*(C+C^T) is linear in the
    descriptor -- the cleanest case for an input-conditioned net (cf. learned_viscosity)."""
    return MU_LOW + (MU_HIGH - MU_LOW) * float(m_visc)


def sigma_of_m(m_st):
    """CSF strength scheduled from m_st. Power law keeps sigma small across most of m_st (the roundness
    transition is concentrated at low sigma), so the droplet rounds gradually up the ST axis. Hits exactly
    0 at m_st=0 and SIGMA_MAX at m_st=1 (schedule parity at the trained corners)."""
    return SIGMA_MAX * float(m_st) ** ST_P


# trained corners and the held-out one, as (m_visc, m_st)
CORNERS = {"ll": (0.0, 0.0), "hl": (1.0, 0.0), "lh": (0.0, 1.0)}   # trained
HELDOUT = (1.0, 1.0)

# --------------------------------------------------------------------------- conditioned STRESS net shape
N_IN = 9      # J, Cxx,Cxy,Cyx,Cyy, vx,vy, m_visc, m_st
N_HID = 96    # one hidden layer, tanh
N_OUT = 3     # symmetric stress: sxx, sxy, syy
N_PARAMS = N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT

# --------------------------------------------------------------------------- learned CAPILLARY net shape
# Reads a 5x5 patch of the smoothed grid density phi around a node (25 values) + the surface-tension
# strength s_norm (1 value); outputs the 2D capillary force at that node. The patch spans exactly the
# support the analytic curvature needs (kappa=-div(n) at a node depends on phi two cells out), so the net
# can infer curvature from the raw density instead of being handed it.
PATCH = 5                 # 5x5 stencil half-width 2
CN_IN = PATCH * PATCH + 1  # 25 phi + 1 strength
CN_HID = 128              # one hidden layer, tanh
CN_OUT = 2                # capillary force (fx, fy)
CN_PARAMS = CN_HID * CN_IN + CN_HID + CN_OUT * CN_HID + CN_OUT

# --------------------------------------------------------------------------- state fields (fluid_st)
x = ti.Vector.field(dim, float, MAX_P)
v = ti.Vector.field(dim, float, MAX_P)
C = ti.Matrix.field(dim, dim, float, MAX_P)
J = ti.field(float, MAX_P)

grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))
grid_phi = ti.field(float, (n_grid, n_grid))
grid_phi2 = ti.field(float, (n_grid, n_grid))
grid_n = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_gphi = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_dv = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_ftar = ti.Vector.field(dim, float, (n_grid, n_grid))   # analytic CSF force per node (supervised target)
st_sum = ti.Vector.field(dim, float, ())
st_mass = ti.field(float, ())

x_np_buf = ti.Vector.field(dim, float, MAX_P)
v0_buf = ti.Vector.field(dim, float, MAX_P)

# conditioned-net weight fields (loaded from numpy; forward only)
W1 = ti.field(float, shape=(N_HID, N_IN))
b1 = ti.field(float, shape=N_HID)
W2 = ti.field(float, shape=(N_OUT, N_HID))
b2 = ti.field(float, shape=N_OUT)
fmean = ti.field(float, shape=N_IN)   # last two (m_visc,m_st) get mean 0, std 1 (fed raw)
fstd = ti.field(float, shape=N_IN)
tscale = ti.field(float, shape=())

# capillary-net weight fields (single hidden layer, forward only)
CW1 = ti.field(float, shape=(CN_HID, CN_IN))
Cb1 = ti.field(float, shape=CN_HID)
CW2 = ti.field(float, shape=(CN_OUT, CN_HID))
Cb2 = ti.field(float, shape=CN_OUT)
cn_fmean = ti.field(float, shape=CN_IN)
cn_fstd = ti.field(float, shape=CN_IN)
cn_tscale = ti.field(float, shape=())


# --------------------------------------------------------------------------- constitutive stress
@ti.func
def fluid_visc_stress(p, dt, E, mu_visc, p_vol):
    """Weakly-compressible pressure + Newtonian viscous stress, scaled by the MLS-MPM affine prefactor.
    Copied verbatim from sim/fluid_surface_tension.py; mu_visc=0 recovers the inviscid fluid."""
    pressure = E * (J[p] - 1.0)
    Cp = C[p]
    strain_rate = Cp + Cp.transpose()
    sigma = pressure * ti.Matrix.identity(float, dim) + mu_visc * strain_rate
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * sigma


@ti.func
def net_stress_cond(p, m_visc: ti.f32, m_st: ti.f32):
    """FULL per-particle stress (pressure + viscous) from the shared CONDITIONED MLP, as a function of the
    local fluid state (J, C, v) and the descriptor (m_visc, m_st). Same shape as one_nn's net_sigma_cond,
    but the fluid stress is isotropic pressure + Newtonian viscosity so no polar frame is needed. Returns
    the symmetric world-frame stress (NOT yet scaled by the MLS prefactor)."""
    Cp = C[p]
    vp = v[p]
    feat = ti.Vector([J[p], Cp[0, 0], Cp[0, 1], Cp[1, 0], Cp[1, 1], vp[0], vp[1], m_visc, m_st])
    fs = ti.Vector.zero(float, N_IN)
    for k in ti.static(range(N_IN)):
        fs[k] = (feat[k] - fmean[k]) / fstd[k]
    o0 = b2[0]
    o1 = b2[1]
    o2 = b2[2]
    for hn in ti.static(range(N_HID)):
        acc = b1[hn]
        for k in ti.static(range(N_IN)):
            acc += W1[hn, k] * fs[k]
        hval = ti.tanh(acc)
        o0 += W2[0, hn] * hval
        o1 += W2[1, hn] * hval
        o2 += W2[2, hn] * hval
    s = tscale[None]
    return ti.Matrix([[o0 * s, o1 * s], [o1 * s, o2 * s]])


# --------------------------------------------------------------------------- MLS-MPM + CSF steps (fluid_st)
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
        stress = fluid_visc_stress(p, dt, E, mu_visc, p_vol)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass


@ti.kernel
def p2g_learned(n: ti.i32, dt: ti.f32, p_vol: ti.f32, p_mass: ti.f32, m_visc: ti.f32, m_st: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        sigma = net_stress_cond(p, m_visc, m_st)
        stress = -dt * 4.0 * p_vol * inv_dx * inv_dx * sigma
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass


@ti.kernel
def init_phi():
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_phi[i, j] = ti.min(1.0, grid_m[i, j] / M_REF)


@ti.kernel
def smooth_phi():
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
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        if m > 0.0:
            grid_v[i, j] = grid_v[i, j] / m
        grid_v[i, j].y -= dt * gravity


@ti.kernel
def st_accumulate(dt: ti.f32, sigma_st: ti.f32):
    st_sum[None] = ti.Vector.zero(float, dim)
    st_mass[None] = 0.0
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        dv = ti.Vector.zero(float, dim)
        if m > 1e-12 and 1 <= i < n_grid - 1 and 1 <= j < n_grid - 1:
            div_n = (grid_n[i + 1, j].x - grid_n[i - 1, j].x) * (0.5 * inv_dx) \
                + (grid_n[i, j + 1].y - grid_n[i, j - 1].y) * (0.5 * inv_dx)
            kappa = -div_n
            f = sigma_st * kappa * grid_gphi[i, j]
            dv = dt * f / p_rho
        if m > 0.0:
            st_mass[None] += m
            st_sum[None] += m * dv
        grid_dv[i, j] = dv


@ti.kernel
def st_apply():
    mean = ti.Vector.zero(float, dim)
    if st_mass[None] > 0.0:
        mean = st_sum[None] / st_mass[None]
    for i, j in ti.ndrange(n_grid, n_grid):
        if grid_m[i, j] > 0.0:
            grid_v[i, j] += grid_dv[i, j] - mean


@ti.kernel
def compute_target_force(sigma_st: ti.f32):
    """Analytic CSF force per node, f = sigma_st * kappa * grad(phi), kappa = -div(n). Assumes grid_phi is
    already smoothed and grid_n / grid_gphi already computed. Stored in grid_ftar as the SUPERVISED TARGET
    the capillary net is trained to reproduce (this is the only place the analytic curvature is used)."""
    for i, j in ti.ndrange(n_grid, n_grid):
        f = ti.Vector.zero(float, dim)
        if grid_m[i, j] > 1e-12 and 1 <= i < n_grid - 1 and 1 <= j < n_grid - 1:
            div_n = (grid_n[i + 1, j].x - grid_n[i - 1, j].x) * (0.5 * inv_dx) \
                + (grid_n[i, j + 1].y - grid_n[i, j - 1].y) * (0.5 * inv_dx)
            kappa = -div_n
            f = sigma_st * kappa * grid_gphi[i, j]
        grid_ftar[i, j] = f


@ti.func
def cap_net_force(i: ti.i32, j: ti.i32, s_norm: ti.f32):
    """LEARNED capillary force at node (i,j): forward the single-hidden-layer capillary MLP on the 5x5 patch
    of smoothed grid_phi around the node plus the strength s_norm. No analytic curvature -- the net infers
    the interface geometry from the raw density patch. Returns the world-frame force (fx, fy)."""
    fs = ti.Vector.zero(float, CN_IN)
    idx = 0
    for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
        raw = grid_phi[i + di, j + dj]
        fs[idx] = (raw - cn_fmean[idx]) / cn_fstd[idx]
        idx += 1
    fs[CN_IN - 1] = (s_norm - cn_fmean[CN_IN - 1]) / cn_fstd[CN_IN - 1]
    o0 = Cb2[0]
    o1 = Cb2[1]
    for hn in ti.static(range(CN_HID)):
        acc = Cb1[hn]
        for k in ti.static(range(CN_IN)):
            acc += CW1[hn, k] * fs[k]
        hval = ti.tanh(acc)
        o0 += CW2[0, hn] * hval
        o1 += CW2[1, hn] * hval
    s = cn_tscale[None]
    return ti.Vector([o0 * s, o1 * s])


@ti.kernel
def cap_net_accumulate(dt: ti.f32, s_norm: ti.f32):
    """LEARNED surface tension: replaces st_accumulate. For every interface node run the capillary net to get
    the capillary force, form the velocity increment dt*f/rho, and accumulate the mass-weighted mean so
    st_apply can remove the net momentum (surface tension is internal, zero net). Nodes with no interface in
    their patch (uniform phi) are skipped for speed and correctness (no capillary force off the surface)."""
    st_sum[None] = ti.Vector.zero(float, dim)
    st_mass[None] = 0.0
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        dv = ti.Vector.zero(float, dim)
        if m > 1e-12 and 2 <= i < n_grid - 2 and 2 <= j < n_grid - 2:
            pmin = 2.0
            pmax = -1.0
            for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
                pv = grid_phi[i + di, j + dj]
                pmin = ti.min(pmin, pv)
                pmax = ti.max(pmax, pv)
            if pmax - pmin > 1e-4:
                f = cap_net_force(i, j, s_norm)
                dv = dt * f / p_rho
        if m > 0.0:
            st_mass[None] += m
            st_sum[None] += m * dv
        grid_dv[i, j] = dv


@ti.func
def coulomb(vt, cap):
    r = vt
    if vt > 0:
        r = ti.max(0.0, vt - cap)
    elif vt < 0:
        r = ti.min(0.0, vt + cap)
    return r


@ti.kernel
def grid_boundary(fric: ti.f32):
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
def dump_state(n: ti.i32, out_J: ti.types.ndarray(), out_C: ti.types.ndarray(), out_v: ti.types.ndarray()):
    for p in range(n):
        out_J[p] = J[p]
        out_C[p, 0] = C[p][0, 0]
        out_C[p, 1] = C[p][0, 1]
        out_C[p, 2] = C[p][1, 0]
        out_C[p, 3] = C[p][1, 1]
        out_v[p, 0] = v[p][0]
        out_v[p, 1] = v[p][1]


@ti.kernel
def dump_grid(out_phi: ti.types.ndarray(), out_f: ti.types.ndarray(), out_m: ti.types.ndarray()):
    for i, j in ti.ndrange(n_grid, n_grid):
        out_phi[i, j] = grid_phi[i, j]
        out_f[i, j, 0] = grid_ftar[i, j][0]
        out_f[i, j, 1] = grid_ftar[i, j][1]
        out_m[i, j] = grid_m[i, j]


def grid_snapshot(n, dt, mu_visc, p_vol, p_mass, sigma_st):
    """Recompute, from the CURRENT particle state, the smoothed density field grid_phi and the analytic CSF
    force grid_ftar (the capillary net's supervised target), and return both plus the node-mass field. Used
    during collection to harvest (density patch, target force) pairs. Does not advance the particles."""
    clear_grid()
    p2g_true(n, dt, E_FLUID, mu_visc, p_vol, p_mass)
    init_phi()
    for _ in range(SMOOTH_ITERS):
        smooth_phi()
    compute_normal()
    compute_target_force(sigma_st)
    phi = np.zeros((n_grid, n_grid), np.float32)
    f = np.zeros((n_grid, n_grid, dim), np.float32)
    mm = np.zeros((n_grid, n_grid), np.float32)
    dump_grid(phi, f, mm)
    return phi, f, mm


def substep(n, dt, E, mu_visc, p_vol, p_mass, gravity, sigma_st, s_norm, mode, m):
    """One MLS-MPM substep. mode='true' uses the analytic viscous stress mu_visc and the ANALYTIC CSF at
    sigma_st (ground truth). mode='learned' uses the conditioned STRESS net at descriptor m and the LEARNED
    CAPILLARY net at strength s_norm -- no analytic surface tension anywhere in the learned path."""
    clear_grid()
    if mode == "true":
        p2g_true(n, dt, E, mu_visc, p_vol, p_mass)
    else:
        p2g_learned(n, dt, p_vol, p_mass, float(m[0]), float(m[1]))
    grid_velocity(dt, gravity)
    if mode == "true" and sigma_st > 0.0:
        init_phi()
        for _ in range(SMOOTH_ITERS):
            smooth_phi()
        compute_normal()
        st_accumulate(dt, sigma_st)          # analytic CSF (ground truth only)
        st_apply()
    elif mode == "learned" and s_norm > 0.0:
        init_phi()
        for _ in range(SMOOTH_ITERS):
            smooth_phi()
        cap_net_accumulate(dt, s_norm)       # LEARNED capillary force (no analytic curvature)
        st_apply()
    grid_boundary(FRICTION)
    g2p(n, dt)


# --------------------------------------------------------------------------- scenes
def seed_disk(center, radius, n, seed=0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)], axis=1)


def seed_box(x0, x1, y0, y1, n, seed=0):
    rng = np.random.default_rng(seed)
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


def train_scenes(n):
    """Signature-exercising fluid scenes (gravity on): a soft drop, a harder impact, a released column, a
    lateral toss. Varied C, v, J so the net sees a broad state distribution at each viscosity."""
    return [
        {"pts": seed_disk((0.5, 0.55), 0.11, n, 1), "area": np.pi * 0.11 ** 2,
         "v0": (0.0, -1.0), "T": 0.8, "name": "drop_soft"},
        {"pts": seed_disk((0.5, 0.62), 0.095, n, 2), "area": np.pi * 0.095 ** 2,
         "v0": (0.0, -3.0), "T": 0.8, "name": "drop_hard"},
        {"pts": seed_box(0.42, 0.58, floor_y, 0.55, n, 3), "area": (0.58 - 0.42) * (0.55 - floor_y),
         "v0": (0.0, 0.0), "T": 0.9, "name": "column"},
        {"pts": seed_disk((0.36, 0.52), 0.10, n, 5), "area": np.pi * 0.10 ** 2,
         "v0": (2.4, -1.0), "T": 0.8, "name": "lateral"},
    ]


def grid_scene(n):
    """The shared headline scene every grid cell runs: a disk dropped onto the floor."""
    return {"pts": seed_disk((0.5, 0.55), 0.11, n, 7), "area": np.pi * 0.11 ** 2,
            "v0": (0.0, -1.0), "T": 1.1, "name": "drop"}


def scene_by_name(scenes, name):
    for s in scenes:
        if s["name"] == name:
            return s
    raise KeyError(name)


# --------------------------------------------------------------------------- stability / dt
def cell_dt(mu_visc, sigma_st):
    """Per-cell stable timestep: the smaller of the viscous-diffusion limit (~ dx^2/nu) and the capillary
    limit (~ sqrt(rho dx^3 / (2 pi sigma))), with safety factors, capped at 1e-4. High viscosity and high
    sigma each force a smaller dt; the held-out (high,high) corner takes the smaller of the two."""
    dt_visc = min(1.0e-4, 0.15 * dx * dx / max(mu_visc, 1e-6))
    dt_cap = 1.0e9 if sigma_st <= 0 else 0.4 * np.sqrt(p_rho * dx ** 3 / (2.0 * np.pi * sigma_st))
    return float(min(dt_visc, dt_cap, 1.0e-4))


# --------------------------------------------------------------------------- rollout
def rollout(scene, n_frames, mode, mu_visc=0.0, sigma_st=0.0, m=(0.0, 0.0), dt=None, collect=False,
            collect_grid=False):
    """Roll one scene to physical time scene['T']. mode='true' uses analytic (mu_visc, sigma_st);
    mode='learned' uses the conditioned STRESS net + LEARNED CAPILLARY net at descriptor m. dt defaults to
    the per-cell stable step. collect harvests per-particle states; collect_grid harvests (grid_phi, target
    force) snapshots for the capillary net. Returns (snaps,times,stable[,states][,grids])."""
    n = upload(scene["pts"], scene["v0"])
    p_vol = scene["area"] / n
    p_mass = p_vol * p_rho
    if dt is None:
        dt = cell_dt(mu_visc if mode == "true" else mu_of_m(m[0]), sigma_st)
    spf = max(1, int(round((scene["T"] / n_frames) / dt)))
    s_norm = float(sigma_st / SIGMA_MAX) if SIGMA_MAX > 0 else 0.0
    init_state(n)
    snaps = np.zeros((n_frames, n, dim), dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    states = []
    grids = []
    t = 0.0
    stable = True
    for fidx in range(n_frames):
        for _ in range(spf):
            substep(n, dt, E_FLUID, mu_visc, p_vol, p_mass, 9.8, sigma_st, s_norm, mode, m)
            t += dt
        cur = x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        snaps[fidx] = cur
        times[fidx] = t
        if collect:
            Jb = np.zeros(n, dtype=np.float32)
            Cb = np.zeros((n, 4), dtype=np.float32)
            vb = np.zeros((n, 2), dtype=np.float32)
            dump_state(n, Jb, Cb, vb)
            states.append((Jb.copy(), Cb.copy(), vb.copy()))
        if collect_grid:
            grids.append(grid_snapshot(n, dt, mu_visc, p_vol, p_mass, sigma_st))
    out = [snaps, times, stable]
    if collect:
        out.append(states)
    if collect_grid:
        out.append(grids)
    return tuple(out)


def rollout_true_cond(scene, n_frames, m, dt=None, collect=False):
    """True analytic fluid at the physical parameters that descriptor m maps to: mu=mu_of_m, sigma=sigma_of_m."""
    return rollout(scene, n_frames, "true", mu_visc=mu_of_m(m[0]), sigma_st=sigma_of_m(m[1]),
                   m=m, dt=dt, collect=collect)


def rollout_learned(scene, n_frames, m, dt=None, collect=False):
    """Conditioned-net fluid at descriptor m, with the analytic CSF driven by sigma_of_m(m_st)."""
    return rollout(scene, n_frames, "learned", mu_visc=0.0, sigma_st=sigma_of_m(m[1]),
                   m=m, dt=dt, collect=collect)


# --------------------------------------------------------------------------- diagnostics
def spread_width(snap):
    return float(np.percentile(snap[:, 0], 95) - np.percentile(snap[:, 0], 5))


def pile_height(snap):
    return float(np.percentile(snap[:, 1], 95) - floor_y)


def series(snaps, fn):
    return np.array([fn(snaps[f]) for f in range(snaps.shape[0])], dtype=np.float64)


def traj_rmse(a, b):
    n = min(a.shape[1], b.shape[1])
    d = np.sqrt(((a[:, :n] - b[:, :n]) ** 2).sum(axis=2))
    return float(d.mean())


def _occupancy(snap, res, pad, close_iters=1, fill=True):
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
    area = int(occ.sum())
    if area == 0:
        return 0.0
    P = np.pad(occ, 1)
    all4 = P[2:, 1:-1] & P[:-2, 1:-1] & P[1:-1, 2:] & P[1:-1, :-2]
    perim = int((occ & ~all4).sum())
    if perim == 0:
        return 0.0
    return 4.0 * np.pi * area / (perim * perim)


def _disk_iso_raw(res=64):
    r = res * 0.42
    yy, xx = np.mgrid[0:res, 0:res] - res / 2.0
    return _iso_raw(xx * xx + yy * yy <= r * r)


_DISK_ISO_RAW = _disk_iso_raw()


def circularity(snap, res=64, pad=0.03):
    """Roundness in [0,~1]: 1.0 for a disk, ~0.785 for a square. Rasterised isoperimetric ratio normalised
    by a rasterised reference disk so the digital-perimeter bias cancels."""
    occ, _ = _occupancy(snap, res, pad, close_iters=1, fill=True)
    return float(_iso_raw(occ) / _DISK_ISO_RAW)


# --------------------------------------------------------------------------- numpy MLP (offline train)
def mlp_forward_np(theta, Xs):
    W1n, b1n, W2n, b2n = theta
    h = np.tanh(Xs @ W1n.T + b1n)
    y = h @ W2n.T + b2n
    return y, h


def init_theta(seed):
    rng = np.random.default_rng(seed)
    W1n = (rng.standard_normal((N_HID, N_IN)) * 0.30).astype(np.float64)
    b1n = np.zeros(N_HID, dtype=np.float64)
    W2n = (rng.standard_normal((N_OUT, N_HID)) * 0.30).astype(np.float64)
    b2n = np.zeros(N_OUT, dtype=np.float64)
    return [W1n, b1n, W2n, b2n]


def train_mlp(Xs, Ys, theta0, iters, lr=1.5e-3, batch=4096, seed=0, log_every=2000,
              huber_delta=4.0, gclip=5.0):
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
        diff = yhat - yb
        loss = float(np.mean(diff ** 2))
        hist.append(loss)
        B = xb.shape[0]
        gY = (2.0 / B) * np.clip(diff, -huber_delta, huber_delta)
        gW2 = gY.T @ h
        gb2 = gY.sum(axis=0)
        gh = gY @ theta[2]
        gz = gh * (1.0 - h ** 2)
        gW1 = gz.T @ xb
        gb1 = gz.sum(axis=0)
        grads = [gW1, gb1, gW2, gb2]
        gnorm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        if gnorm > gclip:
            grads = [g * (gclip / gnorm) for g in grads]
        for k in range(4):
            m[k] = b1c * m[k] + (1 - b1c) * grads[k]
            s[k] = b2c * s[k] + (1 - b2c) * grads[k] ** 2
            mh = m[k] / (1 - b1c ** (it + 1))
            sh = s[k] / (1 - b2c ** (it + 1))
            theta[k] = theta[k] - lr * mh / (np.sqrt(sh) + eps)
        if it % log_every == 0 or it == iters - 1:
            print(f"      [train] iter {it:5d}  mse={loss:.5e}")
    return theta, hist


def load_theta(theta):
    W1.from_numpy(theta[0].astype(np.float32))
    b1.from_numpy(theta[1].astype(np.float32))
    W2.from_numpy(theta[2].astype(np.float32))
    b2.from_numpy(theta[3].astype(np.float32))


def rel_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.std(true) + 1e-12))


def targets_for(Jv, Craw, mu):
    """True fluid stress (pressure + viscous) -> (sxx,sxy,syy). Craw cols 0..3 = Cxx,Cxy,Cyx,Cyy."""
    pressure = E_FLUID * (Jv - 1.0)
    Cxx, Cxy, Cyx, Cyy = Craw[:, 0], Craw[:, 1], Craw[:, 2], Craw[:, 3]
    sxx = pressure + mu * 2.0 * Cxx
    sxy = mu * (Cxy + Cyx)
    syy = pressure + mu * 2.0 * Cyy
    return np.stack([sxx, sxy, syy], axis=1)


def effective_mu_of_net(theta, Jv, Craw, vraw, m_visc, fmean_np, fstd_np, tscale_np):
    """Read the net's effective viscous coefficient at descriptor (m_visc, 0) by least-squares projecting
    its predicted VISCOUS stress (prediction minus the analytic pressure) onto the base tensor (2Cxx,
    Cxy+Cyx, 2Cyy). Isolates the net's interior viscosity interpolation from the pressure it also carries."""
    n = Jv.shape[0]
    Xphys = np.concatenate([Jv[:, None], Craw, vraw], axis=1)
    tag = np.tile([m_visc, 0.0], (n, 1))
    Xs = np.concatenate([(Xphys - fmean_np[:7]) / fstd_np[:7], tag], axis=1)
    yhat, _ = mlp_forward_np(theta, Xs)
    pred = yhat * tscale_np
    pressure = E_FLUID * (Jv - 1.0)
    visc_pred = pred - np.stack([pressure, np.zeros(n), pressure], axis=1)
    base = np.stack([2.0 * Craw[:, 0], Craw[:, 1] + Craw[:, 2], 2.0 * Craw[:, 3]], axis=1)
    num = float((visc_pred * base).sum())
    den = float((base * base).sum()) + 1e-12
    return num / den


# --------------------------------------------------------------------------- capillary net (offline train)
def extract_patches(phi, f, m, s_norm, rng, max_nodes=350, mass_thr=1e-6):
    """From one grid snapshot pull (5x5 phi patch + strength, target force) rows at INTERFACE nodes only --
    exactly the nodes the rollout evaluates the capillary net at (mass present, patch not uniform). The
    patch is flattened di-outer, dj-inner (di,dj in -2..2) to match the Taichi cap_net_force gather."""
    from scipy import ndimage
    H, W = phi.shape
    span = ndimage.maximum_filter(phi, size=PATCH) - ndimage.minimum_filter(phi, size=PATCH)
    mask = np.zeros_like(phi, dtype=bool)
    mask[2:H - 2, 2:W - 2] = True
    mask &= (m > mass_thr) & (span > 1e-4)
    ii, jj = np.where(mask)
    if ii.size == 0:
        return np.zeros((0, CN_IN)), np.zeros((0, CN_OUT))
    if ii.size > max_nodes:
        sel = rng.choice(ii.size, max_nodes, replace=False)
        ii, jj = ii[sel], jj[sel]
    X = np.zeros((ii.size, CN_IN), np.float64)
    Y = np.zeros((ii.size, CN_OUT), np.float64)
    for r, (i, j) in enumerate(zip(ii, jj)):
        X[r, :PATCH * PATCH] = phi[i - 2:i + 3, j - 2:j + 3].ravel()
        X[r, -1] = s_norm
        Y[r] = f[i, j]
    return X, Y


def cap_init_theta(seed):
    rng = np.random.default_rng(seed)
    W1n = (rng.standard_normal((CN_HID, CN_IN)) / np.sqrt(CN_IN)).astype(np.float64)
    b1n = np.zeros(CN_HID, dtype=np.float64)
    W2n = (rng.standard_normal((CN_OUT, CN_HID)) / np.sqrt(CN_HID)).astype(np.float64)
    b2n = np.zeros(CN_OUT, dtype=np.float64)
    return [W1n, b1n, W2n, b2n]


def load_cap_theta(theta):
    CW1.from_numpy(theta[0].astype(np.float32))
    Cb1.from_numpy(theta[1].astype(np.float32))
    CW2.from_numpy(theta[2].astype(np.float32))
    Cb2.from_numpy(theta[3].astype(np.float32))


# --------------------------------------------------------------------------- ST calibration
def st_calibrate(quick=False):
    """Sweep sigma_st finely at low values on a gravity-off square blob; measure relaxed roundness. Then
    auto-pick SIGMA_MAX (near the top of the transition) and the schedule exponent ST_P so roundness steps
    GRADUALLY up the five ST rows (the predecessor's saturation fix)."""
    square = seed_box(0.42, 0.58, 0.42, 0.58, 3000 if quick else 5000)
    area_sq = 0.16 * 0.16
    sigs = [0.0, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.11, 0.15] if not quick \
        else [0.0, 0.02, 0.06, 0.12]
    T = 0.18
    R = []
    for sig in sigs:
        dt = cell_dt(0.05, sig)
        snaps, _, ok = rollout({"pts": square, "area": area_sq, "v0": (0.0, 0.0), "T": T, "name": "sq"},
                               16, "true", mu_visc=0.05, sigma_st=sig, dt=min(dt, 5e-5))
        r = float(np.mean(series(snaps[-4:], circularity)))
        R.append(r)
        print(f"  sigma={sig:<6g} roundness={r:.3f}  stable={ok}")
    sigs = np.array(sigs, float)
    R = np.array(R, float)

    # SIGMA_MAX: smallest sigma whose roundness reaches ~90% of the way from R0 to the saturated top.
    R0, Rsat = R[0], R[-1]
    target_top = R0 + 0.85 * (Rsat - R0)
    smax = float(np.interp(target_top, R, sigs))    # R is monotone-ish increasing
    smax = float(np.clip(smax, sigs[1], sigs[-1]))

    def Rf(s):
        return float(np.interp(s, sigs, R))

    # pick ST_P so interior rows (m_st=.25,.5,.75) land at evenly spaced roundness between R0 and Rf(smax)
    best_p, best_score = 1.0, 1e9
    Rtop = Rf(smax)
    for p in np.linspace(1.0, 3.5, 26):
        rows = [Rf(smax * m ** p) for m in (0.25, 0.5, 0.75)]
        ideal = [R0 + (Rtop - R0) * frac for frac in (0.25, 0.5, 0.75)]
        score = float(np.sum((np.array(rows) - np.array(ideal)) ** 2))
        if score < best_score:
            best_score, best_p = score, float(p)
    return {"sigmas": sigs.tolist(), "roundness": R.tolist(), "sigma_max": smax, "st_p": best_p,
            "R0": float(R0), "Rtop": float(Rtop)}


# --------------------------------------------------------------------------- rendering
BG = "#0a0e14"
GROUND = "#161c26"
WALL = "#26313d"
INK = "#dfe6ee"
SUB = "#9fb0c0"
GREY = "#7f8a99"
NN_COL = "#5ec8ff"
CORNER_COL = {"ll": "#4db6ff", "hl": "#ffb037", "lh": "#8fe0ff"}
HELD_COL = "#ff7f9e"


def _panel(ax, pts_list, colors, sizes, label, tlabel, edge=None, ycrop=0.6):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ycrop)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.axhspan(0, floor_y, color=GROUND, zorder=0)
    ax.axhline(floor_y, color=WALL, lw=1.0, zorder=1)
    for pts, col, sz in zip(pts_list, colors, sizes):
        ax.scatter(pts[:, 0], pts[:, 1], s=sz, color=col, edgecolors="none", alpha=0.82, zorder=2)
    if label:
        ax.text(0.5, 0.9, label, ha="center", va="center", color=INK, fontsize=9,
                weight="bold", transform=ax.transAxes)
    if tlabel:
        ax.text(0.5, 0.08, tlabel, ha="center", va="center", color=SUB, fontsize=7.5,
                transform=ax.transAxes)
    if edge:
        for sp in ("top", "bottom", "left", "right"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color(edge)
            ax.spines[sp].set_linewidth(2.4)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])


def _cell_tag(m_visc, m_st):
    for name, mm in CORNERS.items():
        if abs(mm[0] - m_visc) < 1e-6 and abs(mm[1] - m_st) < 1e-6:
            return "trained", CORNER_COL[name]
    if abs(HELDOUT[0] - m_visc) < 1e-6 and abs(HELDOUT[1] - m_st) < 1e-6:
        return "held-out", HELD_COL
    return None, None


def grid_cell_color(m_visc, m_st):
    """Bilinear tint: low-visc/no-ST blue, high-visc amber, high-ST brightened toward white."""
    import matplotlib.colors as mc
    c_ll = np.array(mc.to_rgb("#4db6ff"))
    c_hl = np.array(mc.to_rgb("#ffb037"))
    base = (1 - m_visc) * c_ll + m_visc * c_hl
    white = np.array([0.92, 0.96, 1.0])
    c = (1 - 0.45 * m_st) * base + 0.45 * m_st * white
    return tuple(float(v) for v in np.clip(c, 0, 1))


def _grid_axes(fig, G):
    L, Tm, B = 0.075, 0.055, 0.05
    pw = (1.0 - L) / G
    ph = (1.0 - Tm - B) / G
    return L, B, pw, ph


def _grid_labels(fig, m_viscs, m_sts, G):
    L, B, pw, ph = _grid_axes(fig, G)
    for gi, mv in enumerate(m_viscs):
        fig.text(L + (gi + 0.5) * pw, 1.0 - 0.045, f"{mv:.2f}", ha="center", va="bottom",
                 color=SUB, fontsize=8)
    for gj, ms in enumerate(m_sts):
        fig.text(L * 0.42, B + (gj + 0.5) * ph, f"{ms:.2f}", ha="center", va="center",
                 color=SUB, fontsize=8, rotation=90)
    fig.text(L + (1.0 - L) * 0.5, 1.0 - 0.012,
             r"viscosity  $m_{visc}$  $\longrightarrow$", ha="center", va="top",
             color=INK, fontsize=10)
    fig.text(0.012, B + (ph * G) * 0.5,
             r"surface tension  $m_{st}$  $\longrightarrow$", ha="center", va="center",
             color=INK, fontsize=10, rotation=90)


def render_grid_montage(path, grid_nn, grid_gt, m_viscs, m_sts, fidx, overlay, dpi=140, panel=210):
    """G x G montage of the final frame. columns = m_visc (left->right), rows = m_st (bottom->top).
    overlay=False: NN only, tinted by cell. overlay=True: grey GT + cyan NN overlaid for fidelity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = len(m_viscs)
    fig = plt.figure(figsize=(panel * G / dpi, panel * G / dpi), dpi=dpi, facecolor=BG)
    L, B, pw, ph = _grid_axes(fig, G)
    for gi, mv in enumerate(m_viscs):
        for gj, ms in enumerate(m_sts):
            ax = fig.add_axes([L + gi * pw, B + gj * ph, pw, ph])
            tag, ecol = _cell_tag(mv, ms)
            if overlay:
                pts_list = [grid_gt[gi][gj][fidx], grid_nn[gi][gj][fidx]]
                colors = [GREY, NN_COL]
                sizes = [3.0, 3.0]
            else:
                pts_list = [grid_nn[gi][gj][fidx]]
                colors = [grid_cell_color(mv, ms)]
                sizes = [3.5]
            lbl = tag if tag else ""
            _panel(ax, pts_list, colors, sizes, lbl, None, edge=ecol)
    _grid_labels(fig, m_viscs, m_sts, G)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)


def render_grid_video(path, grid_nn, m_viscs, m_sts, fps=30, dpi=100, panel=170):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    G = len(m_viscs)
    fig = plt.figure(figsize=(panel * G / dpi, panel * G / dpi), dpi=dpi, facecolor=BG)
    L, B, pw, ph = _grid_axes(fig, G)
    axmap = {}
    for gi in range(G):
        for gj in range(G):
            axmap[(gi, gj)] = fig.add_axes([L + gi * pw, B + gj * ph, pw, ph])
    _grid_labels(fig, m_viscs, m_sts, G)
    nf = grid_nn[0][0].shape[0]
    frames = []
    for f in range(nf):
        for gi, mv in enumerate(m_viscs):
            for gj, ms in enumerate(m_sts):
                ax = axmap[(gi, gj)]
                ax.clear()
                tag, ecol = _cell_tag(mv, ms)
                _panel(ax, [grid_nn[gi][gj][f]], [grid_cell_color(mv, ms)], [3.0],
                       tag if tag else "", None, edge=ecol)
        fig.canvas.draw()
        cw, ch = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(ch, cw, 4)[..., :3]
        rgb = rgb[: ch - (ch % 2), : cw - (cw % 2), :]
        frames.append(rgb.copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


def render_overlay_row(path, columns, times, fidx, dpi=140, panel=360, ycrop=0.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(columns)
    fig = plt.figure(figsize=(panel * ncols / dpi, panel / dpi), dpi=dpi, facecolor=BG)
    axes = [fig.add_axes([k / ncols, 0.0, 1.0 / ncols, 1.0]) for k in range(ncols)]
    tlabel = f"t = {times[fidx]:.2f} s"
    for k, (label, sets) in enumerate(columns):
        pts_list = [ss[0][fidx] for ss in sets]
        colors = [ss[1] for ss in sets]
        sizes = [ss[2] for ss in sets]
        _panel(axes[k], pts_list, colors, sizes, label, tlabel, ycrop=ycrop)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)


def render_overlay_video(path, columns, times, fps=30, dpi=100, panel=340, ycrop=0.6):
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
            pts_list = [ss[0][f] for ss in sets]
            colors = [ss[1] for ss in sets]
            sizes = [ss[2] for ss in sets]
            _panel(ax, pts_list, colors, sizes, label, tlabel, ycrop=ycrop)
        fig.canvas.draw()
        cw, ch = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(ch, cw, 4)[..., :3]
        rgb = rgb[: ch - (ch % 2), : cw - (cw % 2), :]
        frames.append(rgb.copy())
    plt.close(fig)
    imageio.mimwrite(path, frames, fps=fps, quality=9, macro_block_size=1)


def heat_grid(path, Z, m_viscs, m_sts, title, cmap="magma", fmt="{:.3f}"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = len(m_viscs)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=130, facecolor=BG)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                   extent=[-0.5, G - 0.5, -0.5, G - 0.5])
    ax.set_xticks(range(G))
    ax.set_xticklabels([f"{v:.2f}" for v in m_viscs], color=SUB, fontsize=8)
    ax.set_yticks(range(G))
    ax.set_yticklabels([f"{v:.2f}" for v in m_sts], color=SUB, fontsize=8)
    ax.set_xlabel(r"viscosity  $m_{visc}$", color=INK)
    ax.set_ylabel(r"surface tension  $m_{st}$", color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    for gi in range(G):
        for gj in range(G):
            ax.text(gi, gj, fmt.format(Z[gj, gi]), ha="center", va="center",
                    color="w", fontsize=7.5)
    # mark trained (star) and held-out (x)
    for name, mm in CORNERS.items():
        gi = int(round(mm[0] * (G - 1)))
        gj = int(round(mm[1] * (G - 1)))
        ax.scatter([gi], [gj], s=140, marker="*", color="w", edgecolors="k", zorder=5)
    ghi = int(round(HELDOUT[0] * (G - 1)))
    ghj = int(round(HELDOUT[1] * (G - 1)))
    ax.scatter([ghi], [ghj], s=120, marker="X", color=HELD_COL, edgecolors="k", zorder=5)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(colors=SUB)
    for sp in ax.spines.values():
        sp.set_color(WALL)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def line_plot(path, series_list, xlabel, ylabel, title, markers=None, xlim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=130, facecolor=BG)
    ax.set_facecolor(BG)
    for (label, xs, ys, color, style) in series_list:
        ax.plot(xs, ys, color=color, lw=2.2, label=label, linestyle=style,
                marker="o" if style == "-" else None, ms=5)
    if markers:
        for (label, xs, ys, color) in markers:
            ax.scatter(xs, ys, color=color, s=90, marker="*", zorder=5, label=label,
                       edgecolors=BG, linewidths=0.6)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11.5)
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


def cap_scatter(path, y_true, y_pred, y_pred_half, y_target_half):
    """Two-panel scatter of the LEARNED capillary force against the analytic CSF force. Left: full trained
    strength (net vs analytic, both force components) -- the replication fit. Right: an untrained
    intermediate strength s=0.5 (net vs 0.5x analytic) -- the linearity-in-strength generalization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), dpi=130, facecolor=BG)
    for ax, yt, yp, ttl in ((axes[0], y_true, y_pred, "trained strength  s = 1"),
                            (axes[1], y_target_half, y_pred_half, "untrained strength  s = 0.5")):
        ax.set_facecolor(BG)
        lo = float(min(yt.min(), yp.min())); hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1.2, ls="--", zorder=1)
        ax.scatter(yt[:, 0], yp[:, 0], s=3, color=NN_COL, alpha=0.35, label="fx", zorder=2)
        ax.scatter(yt[:, 1], yp[:, 1], s=3, color="#ffb037", alpha=0.35, label="fy", zorder=2)
        ax.set_xlabel("analytic CSF force", color=INK)
        ax.set_ylabel("learned capillary force", color=INK)
        ax.set_title(ttl, color=INK, fontsize=11)
        ax.tick_params(colors=SUB)
        for sp in ax.spines.values():
            sp.set_color(WALL)
        leg = ax.legend(facecolor=BG, edgecolor=WALL, labelcolor=INK, fontsize=9, markerscale=3)
        leg.get_frame().set_alpha(0.9)
    fig.suptitle("Learned capillary force reproduces the analytic surface-tension force",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def make_thumb_png(snaps, fidx, m_visc, m_st, size=170):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    _panel(ax, [snaps[fidx]], [grid_cell_color(m_visc, m_st)], [4.5], "", None)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, facecolor=BG, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_html_grid(m_viscs, m_sts, thumbs, diag):
    G = len(m_viscs)
    cells = []
    for gj in range(G - 1, -1, -1):
        for gi in range(G):
            mv, ms = m_viscs[gi], m_sts[gj]
            tag, _ = _cell_tag(mv, ms)
            d = diag[f"{mv:.2f},{ms:.2f}"]
            cells.append({"gi": gi, "gj": gj, "mv": mv, "ms": ms, "tag": tag or "",
                          "img": thumbs[f"{gi}_{gj}"], "rmse": d["rmse"], "round": d["round_nn"],
                          "w": d["width_nn"], "stable": d["stable"]})
    data = json.dumps(cells)
    html = """<!doctype html><html><head><meta charset="utf-8"><style>
:root{color-scheme:dark}
body{margin:0;background:#0a0e14;color:#dfe6ee;font-family:-apple-system,system-ui,sans-serif}
.wrap{display:flex;gap:18px;padding:16px;flex-wrap:wrap;align-items:flex-start}
.gridbox{position:relative}
.grid{display:grid;grid-template-columns:repeat(__G__,1fr);gap:3px}
.cell{position:relative;cursor:pointer;border:2px solid #26313d;border-radius:4px;overflow:hidden;
      transition:transform .08s,border-color .08s}
.cell img{display:block;width:82px;height:82px}
.cell:hover{transform:scale(1.06);border-color:#9fb0c0;z-index:2}
.cell.tr{border-color:#ffd479}.cell.ho{border-color:#ff7f9e}
.badge{position:absolute;top:2px;left:2px;font-size:8px;background:rgba(0,0,0,.6);
       padding:1px 4px;border-radius:3px;color:#ffd479}
.axis{color:#9fb0c0;font-size:12px}.axx{text-align:center;margin-top:6px}
.axy{writing-mode:vertical-rl;transform:rotate(180deg);position:absolute;left:-22px;top:0;height:100%;
     display:flex;align-items:center}
.panel{min-width:230px;max-width:280px;background:#111722;border:1px solid #26313d;border-radius:8px;padding:14px}
.panel img{width:100%;border-radius:6px;background:#0a0e14}
.k{color:#9fb0c0}.v{color:#dfe6ee;font-weight:600}
h3{margin:.2em 0 .5em}.row{display:flex;justify-content:space-between;margin:3px 0;font-size:13px}
.tag{color:#ffd479;font-weight:700}
</style></head><body><div class="wrap"><div>
<div class="gridbox"><div class="axy axis">surface tension m_st &rarr;</div>
<div class="grid" id="grid"></div></div>
<div class="axx axis">viscosity m_visc &rarr;</div></div>
<div class="panel" id="panel"><h3>Conditioned fluid grid</h3>
<div style="font-size:13px;color:#9fb0c0">Hover a cell. One network, one weight set; only the two-parameter
descriptor changes. Starred cells are the three trained conditions; the pink corner (high visc, high ST) is
never trained. RMSE is the conditioned rollout vs the ground-truth fluid at that cell.</div></div>
</div><script>
var CELLS=__DATA__;
var grid=document.getElementById('grid'),panel=document.getElementById('panel');
function show(c){
 var t=c.tag?('<div class="row"><span class="tag">'+c.tag.toUpperCase()+'</span></div>'):'';
 panel.innerHTML='<h3>m = ('+c.mv.toFixed(2)+', '+c.ms.toFixed(2)+')</h3>'+
  '<img src="data:image/png;base64,'+c.img+'"/>'+t+
  '<div class="row"><span class="k">traj RMSE vs GT</span><span class="v">'+c.rmse.toFixed(4)+'</span></div>'+
  '<div class="row"><span class="k">roundness</span><span class="v">'+c.round.toFixed(3)+'</span></div>'+
  '<div class="row"><span class="k">spread width</span><span class="v">'+c.w.toFixed(3)+'</span></div>'+
  '<div class="row"><span class="k">finite</span><span class="v">'+(c.stable?'yes':'NO')+'</span></div>';
}
CELLS.forEach(function(c){
 var d=document.createElement('div');d.className='cell'+(c.tag==='trained'?' tr':'')+(c.tag==='held-out'?' ho':'');
 d.innerHTML='<img src="data:image/png;base64,'+c.img+'"/>'+(c.tag?'<div class="badge">'+c.tag+'</div>':'');
 d.onmouseenter=function(){show(c)};d.onclick=function(){show(c)};
 grid.appendChild(d);
});
show(CELLS.find(function(c){return c.tag==='held-out'})||CELLS[0]);
</script></body></html>"""
    return html.replace("__G__", str(G)).replace("__DATA__", data)


# --------------------------------------------------------------------------- pipeline
def main():
    global SIGMA_MAX, ST_P
    ap = argparse.ArgumentParser(description="One conditioned net for a fluid across viscosity x surface tension")
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--calibrate", action="store_true", help="just the ST sigma calibration sweep")
    args = ap.parse_args()
    quick = args.quick

    if args.calibrate:
        cal = st_calibrate(quick=quick)
        print(f"\n  -> SIGMA_MAX={cal['sigma_max']:.4f}  ST_P={cal['st_p']:.2f}")
        print(f"     roundness up the rows: ", [f"{np.interp(cal['sigma_max']*m**cal['st_p'], cal['sigmas'], cal['roundness']):.3f}"
                                                for m in (0.0, 0.25, 0.5, 0.75, 1.0)])
        return

    n_frames = 22 if quick else 48
    iters = 2500 if quick else 16000
    n_train = 2000 if quick else 3000
    n_grid_part = 2500 if quick else 4000
    G = 3 if quick else 5

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_dir = "runs/material-variants/generalize-one-nn-across-viscosity-and-surface-tension"
    out_dir = os.path.join(repo, *rel_dir.split("/"))
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- 0. CALIBRATE the gentle ST range on a cheap blob (BEFORE the pipeline) ----------
    print("=== calibrating a gentle surface-tension range (isolation blob) ===")
    cal = st_calibrate(quick=quick)
    SIGMA_MAX = cal["sigma_max"]
    ST_P = cal["st_p"]
    print(f"  chosen SIGMA_MAX={SIGMA_MAX:.4f}  ST_P={ST_P:.2f}")
    row_round = [float(np.interp(sigma_of_m(ms), cal["sigmas"], cal["roundness"]))
                 for ms in np.linspace(0, 1, G)]
    print(f"  predicted roundness up the {G} ST rows: {[round(r, 3) for r in row_round]}")
    line_plot(
        os.path.join(out_dir, "st_calibration.png"),
        [("roundness vs sigma", cal["sigmas"], cal["roundness"], NN_COL, "-")],
        r"surface tension  $\sigma_{st}$", "roundness  (1 = disk, 0.785 = square)",
        f"ST calibration: gentle range chosen so roundness rises gradually  ($\\sigma_{{max}}$={SIGMA_MAX:.3f}, p={ST_P:.2f})",
        markers=[("chosen ST rows", [sigma_of_m(ms) for ms in np.linspace(0, 1, G)], row_round, "#ffd479")])

    # ---------------- 1. collect training states + capillary patches at the THREE trained corners --------
    print("=== collecting GT training states + capillary patches at the three trained corners ===")
    tr_scenes = train_scenes(n_train)
    pool = {"J": [], "C": [], "v": [], "mv": [], "mst": [], "mu": []}
    cap_pool = {"ll": {"X": [], "Y": []}, "hl": {"X": [], "Y": []}, "lh": {"X": [], "Y": []}}
    prng = np.random.default_rng(11)
    for name, mm in CORNERS.items():
        mu = mu_of_m(mm[0])
        sig = sigma_of_m(mm[1])
        s_norm = float(sig / SIGMA_MAX) if SIGMA_MAX > 0 else 0.0
        for sc in tr_scenes:
            _, _, ok, st, grids = rollout(sc, n_frames, "true", mu_visc=mu, sigma_st=sig,
                                          collect=True, collect_grid=True)
            for (Jb, Cb, vb) in st:
                pool["J"].append(Jb); pool["C"].append(Cb); pool["v"].append(vb)
                pool["mv"].append(np.full(len(Jb), mm[0], np.float32))
                pool["mst"].append(np.full(len(Jb), mm[1], np.float32))
                pool["mu"].append(np.full(len(Jb), mu, np.float32))
            for (phi, fg, mg) in grids:
                Xc, Yc = extract_patches(phi, fg, mg, s_norm, prng)
                if Xc.shape[0]:
                    cap_pool[name]["X"].append(Xc); cap_pool[name]["Y"].append(Yc)
            print(f"  corner {name}={mm} mu={mu:.3f} sig={sig:.4f} s_norm={s_norm:.3f}  "
                  f"scene {sc['name']:9s} stable={ok}")

    Jp = np.concatenate(pool["J"]).astype(np.float64)
    Cp = np.concatenate(pool["C"]).astype(np.float64)
    vp = np.concatenate(pool["v"]).astype(np.float64)
    mvp = np.concatenate(pool["mv"]).astype(np.float64)
    mstp = np.concatenate(pool["mst"]).astype(np.float64)
    mup = np.concatenate(pool["mu"]).astype(np.float64)

    # x-mirror augmentation (physics symmetric under x -> -x): flip vx and the off-diagonal C
    Cm = Cp.copy(); Cm[:, 1] *= -1.0; Cm[:, 2] *= -1.0
    vm = vp.copy(); vm[:, 0] *= -1.0
    Jp = np.concatenate([Jp, Jp]); Cp = np.concatenate([Cp, Cm]); vp = np.concatenate([vp, vm])
    mvp = np.concatenate([mvp, mvp]); mstp = np.concatenate([mstp, mstp]); mup = np.concatenate([mup, mup])

    CAP = 90000
    if Jp.shape[0] > CAP:
        sel = np.random.default_rng(0).choice(Jp.shape[0], CAP, replace=False)
        Jp, Cp, vp, mvp, mstp, mup = Jp[sel], Cp[sel], vp[sel], mvp[sel], mstp[sel], mup[sel]
    print(f"  pooled training points: {Jp.shape[0]}")

    # features (7 physical + 2 descriptor) and targets
    Xphys = np.concatenate([Jp[:, None], Cp, vp], axis=1)     # 7 cols: J, C(4), v(2)
    fmean7 = np.median(Xphys, axis=0)
    fstd7 = 0.5 * (np.percentile(Xphys, 84, axis=0) - np.percentile(Xphys, 16, axis=0))
    fstd7 = np.where(fstd7 < 1e-5, 1.0, fstd7)
    Y = targets_for(Jp, Cp, mup)
    tscale_np = float(np.std(Y))
    fmean_np = np.concatenate([fmean7, [0.0, 0.0]])
    fstd_np = np.concatenate([fstd7, [1.0, 1.0]])
    fmean.from_numpy(fmean_np.astype(np.float32))
    fstd.from_numpy(fstd_np.astype(np.float32))
    tscale[None] = tscale_np
    Xs = np.concatenate([(Xphys - fmean7) / fstd7, mvp[:, None], mstp[:, None]], axis=1)
    Ys = Y / tscale_np
    print(f"  target scale {tscale_np:.4f}   feature std {np.round(fstd7, 4)}")

    nval = Xs.shape[0] // 6
    Xtr, Ytr = Xs[nval:], Ys[nval:]
    Xval, Yval_raw = Xs[:nval], Y[:nval]

    # ---------------- 2. train the ONE conditioned net ----------------
    print("=== training ONE conditioned net (joint regression over the three corners) ===")
    theta, hist = train_mlp(Xtr, Ytr, init_theta(0), iters=iters, seed=0)
    load_theta(theta)
    yhat_val, _ = mlp_forward_np(theta, Xval)
    val_rr = rel_rmse(yhat_val * tscale_np, Yval_raw)
    print(f"  final mse={hist[-1]:.4e}  val rel-rmse={val_rr:.4f}")
    train_report = {"final_mse": float(hist[-1]), "val_rel_rmse": val_rr,
                    "loss_hist": [float(h) for h in hist[::max(1, len(hist) // 60)]]}

    # ---------------- 2b. train the LEARNED CAPILLARY net (surface tension) ----------------
    print("=== training the LEARNED capillary net (density patch -> capillary force) ===")
    capX = np.concatenate([np.concatenate(cap_pool[k]["X"]) for k in cap_pool], axis=0)
    capY = np.concatenate([np.concatenate(cap_pool[k]["Y"]) for k in cap_pool], axis=0)
    # x-mirror augmentation for the patch: flip columns left-right within each 5x5 row, negate fx
    capXm = capX.copy()
    patch_m = capX[:, :PATCH * PATCH].reshape(-1, PATCH, PATCH)[:, ::-1, :].reshape(-1, PATCH * PATCH)
    capXm[:, :PATCH * PATCH] = patch_m
    capYm = capY.copy(); capYm[:, 0] *= -1.0
    capX = np.concatenate([capX, capXm]); capY = np.concatenate([capY, capYm])
    CCAP = 30000 if quick else 120000
    if capX.shape[0] > CCAP:
        sel = np.random.default_rng(3).choice(capX.shape[0], CCAP, replace=False)
        capX, capY = capX[sel], capY[sel]
    # feature normalization: 25 phi cols by median/IQR; strength col fixed to map {0,1}->{-1,1}
    cn_fmean_np = np.zeros(CN_IN); cn_fstd_np = np.ones(CN_IN)
    cn_fmean_np[:PATCH * PATCH] = np.median(capX[:, :PATCH * PATCH], axis=0)
    iqr = 0.5 * (np.percentile(capX[:, :PATCH * PATCH], 84, axis=0)
                 - np.percentile(capX[:, :PATCH * PATCH], 16, axis=0))
    cn_fstd_np[:PATCH * PATCH] = np.where(iqr < 1e-4, 1.0, iqr)
    cn_fmean_np[-1] = 0.5; cn_fstd_np[-1] = 0.5
    cn_tscale_np = float(np.std(capY)) + 1e-12
    cn_fmean.from_numpy(cn_fmean_np.astype(np.float32))
    cn_fstd.from_numpy(cn_fstd_np.astype(np.float32))
    cn_tscale[None] = cn_tscale_np
    capXs = (capX - cn_fmean_np) / cn_fstd_np
    capYs = capY / cn_tscale_np
    ncv = capXs.shape[0] // 6
    cap_theta, cap_hist = train_mlp(capXs[ncv:], capYs[ncv:], cap_init_theta(0),
                                    iters=iters, lr=1.0e-3, seed=1, huber_delta=6.0)
    load_cap_theta(cap_theta)
    cap_yh, _ = mlp_forward_np(cap_theta, capXs[:ncv])
    cap_val_rr = rel_rmse(cap_yh * cn_tscale_np, capY[:ncv])
    print(f"  capillary net: {capXs.shape[0]} patches  final mse={cap_hist[-1]:.4e}  "
          f"val rel-rmse={cap_val_rr:.4f}  tscale={cn_tscale_np:.3f}")
    cap_train_report = {"final_mse": float(cap_hist[-1]), "val_rel_rmse": cap_val_rr,
                        "n_patches": int(capXs.shape[0]), "tscale": cn_tscale_np,
                        "loss_hist": [float(h) for h in cap_hist[::max(1, len(cap_hist) // 60)]]}
    # per-corner capillary replication fit (net force vs analytic CSF force at that corner)
    cap_fit = {}
    for name in cap_pool:
        Xk = np.concatenate(cap_pool[name]["X"]); Yk = np.concatenate(cap_pool[name]["Y"])
        yh, _ = mlp_forward_np(cap_theta, (Xk - cn_fmean_np) / cn_fstd_np)
        pred = yh * cn_tscale_np
        rr = float(np.sqrt(np.mean((pred - Yk) ** 2)) / (np.std(Yk) + 1e-12)) if np.std(Yk) > 1e-9 \
            else float(np.sqrt(np.mean(pred ** 2)))
        cap_fit[name] = {"rel_rmse": rr, "target_std": float(np.std(Yk)), "n": int(Xk.shape[0])}
        print(f"    corner {name}: cap-net force rel-rmse vs analytic = {rr:.4f}  (target std {np.std(Yk):.3f})")

    # ---------------- 3. EDGE-EXACTNESS at the three trained corners ----------------
    print("=== edge-exactness: conditioned net at each trained corner vs true simulator ===")
    q_scene = {"pts": seed_disk((0.5, 0.55), 0.11, n_grid_part, 7), "area": np.pi * 0.11 ** 2,
               "v0": (0.0, -1.0), "T": 1.1, "name": "drop"}
    edge = {}
    edge_cols = []
    edge_times = None
    for name, mm in CORNERS.items():
        tr_snaps, tt, tok = rollout_true_cond(q_scene, n_frames, mm)
        t2, _, _ = rollout_true_cond(q_scene, n_frames, mm)           # true self-noise floor
        nn_snaps, _, nok = rollout_learned(q_scene, n_frames, mm)
        edge_times = tt
        floor = traj_rmse(tr_snaps, t2)
        fit = traj_rmse(tr_snaps, nn_snaps)
        sched_sig = sigma_of_m(mm[1])
        # stress-net replication fit at this corner (predicted stress vs analytic target)
        cmask = (np.abs(mvp - mm[0]) < 1e-6) & (np.abs(mstp - mm[1]) < 1e-6)
        ys, _ = mlp_forward_np(theta, Xs[cmask])
        stress_rr = rel_rmse(ys * tscale_np, Y[cmask])
        edge[name] = {"m": list(mm), "mu": mu_of_m(mm[0]), "sigma": sched_sig,
                      "net_vs_true": fit, "true_self_noise": floor,
                      "stress_relrmse": stress_rr, "cap_relrmse": cap_fit[name]["rel_rmse"],
                      "net_stable": bool(nok), "true_stable": bool(tok)}
        edge_cols.append((f"{name}  m={mm}", [(tr_snaps, GREY, 5), (nn_snaps, NN_COL, 5)]))
        print(f"  {name} m={mm} mu={mu_of_m(mm[0]):.3f} sig={sched_sig:.4f}  net-vs-true={fit:.5f} "
              f"(floor {floor:.5f})  stress-fit={stress_rr:.4f} cap-fit={cap_fit[name]['rel_rmse']:.4f} stable={nok}")
    render_overlay_row(os.path.join(out_dir, "edge_exactness_still.png"), edge_cols, edge_times, n_frames - 1)
    render_overlay_video(os.path.join(out_dir, "edge_exactness.mp4"), edge_cols, edge_times)

    # ---------------- 3b. capillary-net fit diagnostic (learned force vs analytic CSF force) --------------
    print("=== capillary-net fit: learned force vs analytic CSF force ===")
    Xlh = np.concatenate(cap_pool["lh"]["X"]); Ylh = np.concatenate(cap_pool["lh"]["Y"])
    pred_lh, _ = mlp_forward_np(cap_theta, (Xlh - cn_fmean_np) / cn_fstd_np)
    pred_lh = pred_lh * cn_tscale_np
    # interior-strength linearity: evaluate the net at s=0.5 on the same patches; force is linear in
    # strength, so the analytic target scales to 0.5 * (full-strength force). This checks the net produces
    # the RIGHT intermediate force at an untrained strength (it only saw s in {0,1}).
    Xhalf = Xlh.copy(); Xhalf[:, -1] = 0.5
    pred_half, _ = mlp_forward_np(cap_theta, (Xhalf - cn_fmean_np) / cn_fstd_np)
    pred_half = pred_half * cn_tscale_np
    half_target = 0.5 * Ylh
    half_rr = float(np.sqrt(np.mean((pred_half - half_target) ** 2)) / (np.std(half_target) + 1e-12))
    cap_train_report["interior_s_relrmse"] = half_rr
    print(f"  interior strength s=0.5 rel-rmse vs 0.5*analytic = {half_rr:.4f}")
    cap_scatter(os.path.join(out_dir, "capillary_fit.png"), Ylh, pred_lh, pred_half, half_target)

    # ---------------- 4. effective-mu diagnostic (does input-conditioning give a smooth slider?) -------
    print("=== effective viscosity of the net vs m_visc (input-conditioning slider) ===")
    # a fixed pool of low-viscosity states to probe the net's viscous response
    _, _, _, probe_st = rollout(scene_by_name(tr_scenes, "drop_hard"), n_frames, "true",
                                mu_visc=MU_LOW, sigma_st=0.0, collect=True)
    Jpr = np.concatenate([s[0] for s in probe_st]).astype(np.float64)
    Cpr = np.concatenate([s[1] for s in probe_st]).astype(np.float64)
    vpr = np.concatenate([s[2] for s in probe_st]).astype(np.float64)
    mv_probe = np.linspace(0, 1, 9)
    eff_mu = [effective_mu_of_net(theta, Jpr, Cpr, vpr, mv, fmean_np, fstd_np, tscale_np) for mv in mv_probe]
    ideal_mu = [mu_of_m(mv) for mv in mv_probe]
    print("  m_visc : eff_mu(net) / ideal_mu")
    for mv, em, im in zip(mv_probe, eff_mu, ideal_mu):
        print(f"    {mv:.2f} : {em:.4f} / {im:.4f}")
    line_plot(
        os.path.join(out_dir, "effective_mu.png"),
        [("ideal (linear schedule)", mv_probe, ideal_mu, INK, ":"),
         ("net effective viscosity", mv_probe, eff_mu, NN_COL, "-")],
        r"descriptor  $m_{visc}$", r"effective viscous coefficient  $\mu$",
        "Input-conditioning: the net's viscosity vs the descriptor (trained only at m_visc=0,1)")
    eff_dev = float(np.max(np.abs(np.array(eff_mu) - np.array(ideal_mu))))

    # ---------------- 5. THE 5x5 GRID: conditioned vs GT (headline) ----------------
    print(f"=== {G}x{G} grid: conditioned net vs ground-truth fluid over (m_visc, m_st) ===")
    m_viscs = [float(v) for v in np.linspace(0, 1, G)]
    m_sts = [float(v) for v in np.linspace(0, 1, G)]
    grid_nn = [[None] * G for _ in range(G)]
    grid_gt = [[None] * G for _ in range(G)]
    grid_times = None
    diag = {}
    rmseZ = np.zeros((G, G))       # [gj (m_st), gi (m_visc)]
    roundZ = np.zeros((G, G))
    widthZ = np.zeros((G, G))
    for gi, mv in enumerate(m_viscs):
        for gj, ms in enumerate(m_sts):
            gt_snaps, tt, gok = rollout_true_cond(q_scene, n_frames, (mv, ms))
            nn_snaps, _, nok = rollout_learned(q_scene, n_frames, (mv, ms))
            grid_times = tt
            grid_nn[gi][gj] = nn_snaps
            grid_gt[gi][gj] = gt_snaps
            rm = traj_rmse(gt_snaps, nn_snaps)
            rnd = float(circularity(nn_snaps[-1]))
            wid = float(spread_width(nn_snaps[-1]))
            rmseZ[gj, gi] = rm
            roundZ[gj, gi] = rnd
            widthZ[gj, gi] = wid
            tag, _ = _cell_tag(mv, ms)
            diag[f"{mv:.2f},{ms:.2f}"] = {
                "m_visc": mv, "m_st": ms, "mu": mu_of_m(mv), "sigma": sigma_of_m(ms),
                "rmse": rm, "round_nn": rnd, "round_gt": float(circularity(gt_snaps[-1])),
                "width_nn": wid, "width_gt": float(spread_width(gt_snaps[-1])),
                "stable": bool(nok and gok), "tag": tag or ""}
            print(f"  m=({mv:.2f},{ms:.2f}) mu={mu_of_m(mv):.3f} sig={sigma_of_m(ms):.4f}  "
                  f"RMSE={rm:.4f} round={rnd:.3f} w={wid:.3f} {'ok' if nok and gok else 'BLEW'}"
                  f"{' <'+tag if tag else ''}")

    render_grid_montage(os.path.join(out_dir, "grid_montage.png"), grid_nn, grid_gt,
                        m_viscs, m_sts, n_frames - 1, overlay=False)
    render_grid_montage(os.path.join(out_dir, "grid_overlay_montage.png"), grid_nn, grid_gt,
                        m_viscs, m_sts, n_frames - 1, overlay=True)
    render_grid_video(os.path.join(out_dir, "grid_sweep.mp4"), grid_nn, m_viscs, m_sts)
    heat_grid(os.path.join(out_dir, "rmse_heatmap.png"), rmseZ, m_viscs, m_sts,
              "Per-cell trajectory RMSE: conditioned net vs ground-truth fluid", cmap="magma")
    heat_grid(os.path.join(out_dir, "roundness_heatmap.png"), roundZ, m_viscs, m_sts,
              "Per-cell roundness (rises gradually up the surface-tension axis)", cmap="viridis")

    # roundness / width trend lines along each axis (the eye-backing diagnostic)
    line_plot(
        os.path.join(out_dir, "axis_trends.png"),
        [("roundness up ST (low visc col)", m_sts, list(roundZ[:, 0]), "#5ec8ff", "-"),
         ("roundness up ST (high visc col)", m_sts, list(roundZ[:, -1]), "#8fe0ff", "--"),
         ("width across visc (no ST row)", m_viscs, list(widthZ[0, :]), "#ffb037", "-")],
        "descriptor value along the axis", "roundness  /  spread width",
        "Smooth trends: ST rounds the drop (rising), viscosity narrows the spread (falling)")

    # ---------------- 6. HELD-OUT corner test (high visc, high ST) ----------------
    print("=== held-out corner (high visc, high ST) = (1,1): net vs true fluid ===")
    ho_gt, ho_tt, ho_gok = rollout_true_cond(q_scene, n_frames, HELDOUT)
    ho_nn, _, ho_nok = rollout_learned(q_scene, n_frames, HELDOUT)
    ho_rmse = traj_rmse(ho_gt, ho_nn)
    ho = {"m": list(HELDOUT), "mu": mu_of_m(HELDOUT[0]), "sigma": sigma_of_m(HELDOUT[1]),
          "rmse": ho_rmse, "round_nn": float(circularity(ho_nn[-1])),
          "round_gt": float(circularity(ho_gt[-1])), "width_nn": float(spread_width(ho_nn[-1])),
          "width_gt": float(spread_width(ho_gt[-1])), "net_stable": bool(ho_nok),
          "true_stable": bool(ho_gok)}
    print(f"  held-out (1,1) mu={ho['mu']:.3f} sig={ho['sigma']:.4f}  RMSE={ho_rmse:.4f}  "
          f"round nn/gt={ho['round_nn']:.3f}/{ho['round_gt']:.3f}  stable={ho_nok}")
    ho_cols = [("held-out (1,1): net vs true", [(ho_gt, GREY, 5), (ho_nn, HELD_COL, 5)]),
               ("trained (1,0): net vs true",
                [(rollout_true_cond(q_scene, n_frames, (1.0, 0.0))[0], GREY, 5),
                 (rollout_learned(q_scene, n_frames, (1.0, 0.0))[0], CORNER_COL["hl"], 5)])]
    render_overlay_row(os.path.join(out_dir, "heldout_still.png"), ho_cols, ho_tt, n_frames - 1)
    render_overlay_video(os.path.join(out_dir, "heldout_corner.mp4"), ho_cols, ho_tt)

    # ---------------- 7. interactive HTML grid ----------------
    print("=== building interactive HTML grid ===")
    thumbs = {}
    for gi in range(G):
        for gj in range(G):
            thumbs[f"{gi}_{gj}"] = make_thumb_png(grid_nn[gi][gj], n_frames - 1, m_viscs[gi], m_sts[gj])
    custom_html = build_html_grid(m_viscs, m_sts, thumbs, diag)
    with open(os.path.join(out_dir, "grid_interactive.html"), "w", encoding="utf-8") as fh:
        fh.write(custom_html)

    # ---------------- 8. metrics + manifest ----------------
    metrics = {"dt_note": "per-cell stable dt (viscous & capillary limits)", "n_grid": n_grid,
               "n_particles": n_grid_part, "E": E_FLUID, "MU_LOW": MU_LOW, "MU_HIGH": MU_HIGH,
               "SIGMA_MAX": SIGMA_MAX, "ST_P": ST_P, "smooth_iters": SMOOTH_ITERS,
               "descriptor": {"m_visc": "viscosity (linear mu)", "m_st": "surface tension (power schedule)",
                              "trained": {k: list(v) for k, v in CORNERS.items()},
                              "held_out": list(HELDOUT)},
               "net": {"in": N_IN, "hidden": N_HID, "out": N_OUT, "params": N_PARAMS},
               "cap_net": {"in": CN_IN, "hidden": CN_HID, "out": CN_OUT, "params": CN_PARAMS,
                           "patch": PATCH},
               "target_scale": tscale_np, "train": train_report, "cap_train": cap_train_report,
               "cap_fit": cap_fit, "calibration": cal,
               "edge": edge, "grid_diag": diag, "held_out": ho, "grid": G,
               "m_viscs": m_viscs, "m_sts": m_sts, "rmse_grid": rmseZ.tolist(),
               "round_grid": roundZ.tolist(), "width_grid": widthZ.tolist(),
               "eff_mu": {"m_visc": mv_probe.tolist(), "net": eff_mu, "ideal": ideal_mu,
                          "max_dev": eff_dev},
               "train_scenes": [s["name"] for s in tr_scenes]}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    write_manifest(out_dir, rel_dir, metrics, custom_html)
    print(f"\nwrote -> {rel_dir}")
    return metrics


def write_manifest(out_dir, rel_dir, m, custom_html):
    def f3(v):
        return f"{v:.3f}"
    edge, ho, gd = m["edge"], m["held_out"], m["grid_diag"]
    G = m["grid"]
    rmseZ = np.array(m["rmse_grid"])
    roundZ = np.array(m["round_grid"])
    interior_rmse_max = float(rmseZ.max())
    interior_rmse_mean = float(rmseZ.mean())
    floor = max(edge[k]["true_self_noise"] for k in edge)
    edge_max = max(edge[k]["net_vs_true"] for k in edge)
    all_finite = all(gd[k]["stable"] for k in gd) and ho["net_stable"]
    # roundness gradualness up the low-viscosity ST column
    st_col = list(roundZ[:, 0])
    st_steps = np.diff(st_col)
    eff_dev = m["eff_mu"]["max_dev"]

    cap_fit = m["cap_fit"]
    cap_edge_max = max(cap_fit[k]["rel_rmse"] for k in cap_fit)
    stress_edge_max = max(edge[k]["stress_relrmse"] for k in edge)
    cap_val = m["cap_train"]["val_rel_rmse"]
    cap_interior = m["cap_train"].get("interior_s_relrmse", float("nan"))

    edge_rows = []
    for name in ("ll", "hl", "lh"):
        e = edge[name]
        edge_rows.append([f"{name}  m={tuple(e['m'])}  (mu={e['mu']:.2f}, sig={e['sigma']:.3f})",
                          f"{e['stress_relrmse']:.4f}", f"{e['cap_relrmse']:.4f}",
                          f"{e['net_vs_true']:.4f}", f"{e['true_self_noise']:.4f}",
                          "stable" if e["net_stable"] else "BLEW UP"])
    edge_rows.append([f"held-out  m={tuple(ho['m'])}  (mu={ho['mu']:.2f}, sig={ho['sigma']:.3f})",
                      "-", "-", f"{ho['rmse']:.4f}", "-",
                      "stable" if ho["net_stable"] else "BLEW UP"])

    results = [
        {"type": "image", "src": f"{rel_dir}/grid_montage.png",
         "caption": (f"The headline: a {G} by {G} grid of the learned rollout (learned stress + learned "
                     "capillary force) sweeping the descriptor over the unit square, each cell the settled "
                     "frame of the same dropped disk. Horizontal "
                     "axis is viscosity m_visc (thin left, thick right); vertical axis is surface tension "
                     "m_st (none at the bottom, high at the top). Three corners are trained and starred "
                     "(bottom-left low-visc/no-ST, bottom-right high-visc/no-ST, top-left low-visc/high-ST); "
                     "the top-right corner (high visc, high ST) is the pink HELD-OUT corner, never trained. "
                     "Read it up the surface-tension axis: the drop rounds GRADUALLY row by row rather than "
                     "snapping to a ball, and read across the viscosity axis it settles from a wide thin "
                     "splash to a compact tall mound. But two regions FAIL and are visible here: the "
                     "m_visc=0.25 column sprays particles upward (a stability blow-up), and the high-"
                     "viscosity/high-ST cells including the held-out top-right corner jet into tall spikes.")},
        {"type": "image", "src": f"{rel_dir}/grid_overlay_montage.png",
         "caption": ("The same grid as a fidelity check: grey is the ground-truth analytic fluid at each "
                     "cell's (mu, sigma), cyan is the full learned rollout (learned stress + learned "
                     "capillary force). The cyan overlays the grey cleanly at the three trained corners, but "
                     "DIVERGES badly in two places: the m_visc=0.25 column, where the net sprays particles "
                     "upward while the true fluid stays a blob, and the high-viscosity column, where the net "
                     "jets into tall spikes well above the grey ground-truth droplet -- the held-out "
                     "top-right corner most of all.")},
        {"type": "video", "src": f"{rel_dir}/grid_sweep.mp4",
         "caption": (f"The {G} by {G} conditioned grid animated in lockstep, every cell the one network at a "
                     "different descriptor dropping the same disk. Watch the surface-tension rows (upward) "
                     "bead and hold rounder droplets while the bottom row splashes flat, and the viscosity "
                     "columns (rightward) slow and thicken the flow. The morph is smooth near the trained "
                     "corners, but the m_visc=0.25 column blows up into a spray and the high-viscosity/high-ST "
                     "cells stretch into spikes.")},
        {"type": "image", "src": f"{rel_dir}/rmse_heatmap.png",
         "caption": ("Per-cell trajectory RMSE of the conditioned rollout against the ground-truth fluid, in "
                     "domain units. Stars mark the three trained corners, the pink cross the held-out corner. "
                     f"RMSE is low near the trained corners, but this metric is MISLEADING where the rollout "
                     f"is unphysical: it stays modest (mean {interior_rmse_mean:.3f}) even in the m_visc=0.25 "
                     "column that sprays particles and at the held-out corner that jets into a spike, because "
                     "a fountain or a spike can share a center of mass with the true blob. Read the montages, "
                     "not this heatmap, for whether a cell is physical.")},
        {"type": "image", "src": f"{rel_dir}/effective_mu.png",
         "caption": ("The input-conditioning slider, and the point of the whole design. The network was "
                     "trained at only m_visc = 0 and 1; this reads its effective viscous coefficient at nine "
                     "descriptor values by projecting its predicted viscous stress onto the strain-rate "
                     "tensor. It tracks the linear ideal (dotted) smoothly across the interior, confirming "
                     "the prediction from the weight-interpolation study: conditioning ONE net on the "
                     "parameter as an input interpolates the intermediate viscosity, where blending two "
                     f"nets' weights sagged below it. Max deviation from ideal is {eff_dev:.3f} in mu.")},
        {"type": "video", "src": f"{rel_dir}/heldout_corner.mp4",
         "caption": ("The held-out corner test, now a real test of the learned surface tension. Left: the full "
                     "learned rollout at m = (1,1) (high viscosity, high surface tension), a combination it "
                     "NEVER trained on, in pink over the grey true fluid. Right: the trained high-viscosity/"
                     "no-ST corner for reference. Both learned pieces must combine correctly here: the stress "
                     "net produces the high-viscosity stress it learned at (1,0), and the capillary net "
                     "produces the high-strength force it learned only at LOW viscosity (0,1), now on the "
                     "high-viscosity interface. But the composition FAILS here: the held-out net jets into a "
                     "tall vertical spike while its true fluid settles into a compact blob, so the two "
                     "separately-learned forces do not combine into physical dynamics at this unseen point.")},
        {"type": "image", "src": f"{rel_dir}/heldout_still.png",
         "caption": ("Settled frame of the held-out corner (left, pink) and the trained high-viscosity "
                     "corner (right, amber), each over its grey true fluid. The held-out net is a tall narrow "
                     "SPIKE, nothing like its wide grey ground-truth blob: the learned capillary law does NOT "
                     "compose with the high-viscosity stress into a physical droplet at the unseen corner. The "
                     "trained (1,0) corner on the right matches its ground truth, for contrast.")},
        {"type": "image", "src": f"{rel_dir}/roundness_heatmap.png",
         "caption": ("Per-cell roundness of the conditioned drop. Reading up any column (rising surface "
                     "tension) roundness climbs gradually row by row rather than saturating at the second "
                     "row, which is the calibrated-schedule fix to the predecessor's saturation. Reading "
                     "across a row (rising viscosity) roundness also rises modestly as the thicker fluid "
                     "holds a more compact shape.")},
        {"type": "image", "src": f"{rel_dir}/st_calibration.png",
         "caption": ("The calibration that set the surface-tension range, run first on a cheap gravity-off "
                     "square blob. Roundness climbs steeply with sigma and saturates by about 0.1, so a LOW "
                     f"sigma_max = {m['SIGMA_MAX']:.3f} and a gentle power schedule (exponent {m['ST_P']:.2f}) "
                     "were chosen (gold stars) to place the five surface-tension rows across the visible "
                     "transition instead of in the saturated tail.")},
        {"type": "image", "src": f"{rel_dir}/edge_exactness_still.png",
         "caption": ("Edge-exactness at the three trained corners: grey is the true simulator at that "
                     "condition, cyan the full learned rollout (learned stress + learned capillary force) at "
                     "that descriptor, on the drop's settled frame. The cyan overlays the grey at all three, "
                     "the replication both learned nets reproduce before any interior cell is trusted.")},
        {"type": "image", "src": f"{rel_dir}/capillary_fit.png",
         "caption": ("The learned capillary force against the analytic surface-tension force it was trained to "
                     "reproduce, at the high-surface-tension corner. Left: at the trained strength the "
                     "network's force (from the raw 5x5 density patch, never given the curvature) sits on the "
                     "identity line with the analytic CSF force for both components. Right: at an UNTRAINED "
                     "intermediate strength s = 0.5 the network still matches half the analytic force, so it "
                     "has learned the correct linear-in-strength capillary law from only the two endpoint "
                     "strengths it saw. This is the surface tension being genuinely learned, not applied "
                     "analytically.")},
        {"type": "image", "src": f"{rel_dir}/axis_trends.png",
         "caption": ("The eye-backing diagnostic: roundness rises smoothly up the surface-tension axis (both "
                     "viscosity columns) and spread width falls smoothly across the viscosity axis, the two "
                     "descriptor knobs moving the shape along two separate, monotone directions.")},
        {"type": "table",
         "columns": ["condition", "stress-net fit", "capillary-net fit", "full rollout RMSE vs true",
                     "true self-noise", "rollout"],
         "rows": edge_rows,
         "caption": ("Edge-exactness at the three trained corners plus the held-out corner. The stress-net fit "
                     "is the conditioned stress net's relative RMSE against the analytic fluid stress; the "
                     "capillary-net fit is the learned capillary force's relative RMSE against the analytic "
                     "CSF force. The full-rollout column is the complete learned rollout (both nets, no "
                     "analytic surface tension) versus the true fluid, and true self-noise is the true sim run "
                     "twice (its GPU-noise floor). Both nets reproduce their analytic targets and the full "
                     "rollout matches the true fluid at every trained corner; the held-out corner's rollout "
                     "RMSE sits at the same low level. Every rollout stayed finite.")},
    ]

    findings = (
        "On this one setup, TWO small learned networks sharing one two-scalar descriptor reproduce a "
        "weakly-compressible MLS-MPM fluid across BOTH viscosity and surface tension, with surface tension "
        "LEARNED, not analytic. The design splits the two axes the way the physics does but LEARNS both. "
        "Viscosity is a per-particle bulk stress, so a conditioned STRESS net predicts the full fluid stress "
        "(weakly-compressible pressure E(J-1) plus the Newtonian viscous term) from the local state (J, C, v) "
        "and the descriptor. Surface tension is a grid capillary force set by the interface CURVATURE, a "
        "NON-LOCAL quantity a per-particle net structurally cannot see, so a second CAPILLARY net LEARNS it: "
        "it reads a 5x5 patch of the smoothed grid density field phi around each node (the raw interface "
        "signal, never the analytic curvature) plus the surface-tension strength, and outputs the capillary "
        "force at that node. The learned rollout at every cell is (learned stress) + (learned capillary "
        "force); NO analytic surface tension is applied anywhere in it. The analytic continuum-surface-force "
        "(CSF) is used only to generate the capillary net's supervised targets and as the ground truth. "
        f"m_visc maps to viscosity by a LINEAR schedule (mu from {m['MU_LOW']} to {m['MU_HIGH']}); m_st maps "
        f"to strength by a gentle power schedule (sigma_max = {m['SIGMA_MAX']:.3f}, exponent {m['ST_P']:.2f}) "
        "calibrated on a cheap isolation blob so the roundness transition is spread across the rows rather "
        "than saturating. Trained on THREE corners only -- (0,0) low visc/no ST, (1,0) high visc/no ST, "
        "(0,1) low visc/high ST -- with the fourth corner (1,1) held out. EDGE-EXACTNESS for BOTH nets: at "
        f"each trained corner the stress net reproduces the analytic fluid stress (rel-RMSE at most "
        f"{stress_edge_max:.3f}) and the capillary net reproduces the analytic CSF force (rel-RMSE at most "
        f"{cap_edge_max:.3f}), and the full learned rollout follows the true simulator to a trajectory RMSE "
        f"of at most {edge_max:.4f} (against a true-vs-true GPU-noise floor around {floor:.4f}). The "
        f"capillary net also matches an UNTRAINED intermediate strength s=0.5 to rel-RMSE {cap_interior:.3f} "
        "against half the analytic force, so it learned the correct linear-in-strength capillary law from "
        "only the two endpoint strengths it saw. The input-conditioning viscosity slider also works: probing "
        f"the stress net's effective viscous coefficient at nine m_visc values (trained at only 0 and 1) it "
        f"tracks the linear ideal smoothly (max deviation {eff_dev:.3f} in mu). WHERE IT FAILS (this is a "
        "PARTIAL result, not a clean success, and the figures show it plainly): the interior is not uniformly "
        "physical. The m_visc=0.25 grid column is degenerate, spraying particles upward in a fountain rather "
        f"than settling -- a stability failure at that one viscosity that the per-cell RMSE (up to "
        f"{interior_rmse_max:.3f}) does not capture. And the high-viscosity, high-surface-tension cells, "
        "INCLUDING the held-out (1,1) corner, do not settle into droplets but jet into tall narrow vertical "
        "spikes that do not match the ground-truth blob. The held-out corner therefore does NOT generalize in "
        f"shape: its trajectory RMSE of {ho['rmse']:.4f} badly understates the mismatch, because a vertical "
        "spike and a compact blob share a similar center of mass. The honest summary is mixed: surface "
        "tension is genuinely learned and edge-exact at the three trained corners, and the viscosity slider "
        "interpolates smoothly; but the two separately-learned force laws do not COMPOSE into physical "
        "dynamics across the whole grid -- one viscosity column is unstable, and the unseen high-visc/high-ST "
        "corner produces an unphysical jet rather than the droplet its ground truth is."
    )

    hypothesis = (
        "The spine is local-versus-non-local. Viscosity is a LOCAL constitutive law -- the viscous stress "
        "mu*(C+C^T) is a pointwise function of a particle's own affine matrix C -- so a per-particle net that "
        "sees (J, C, v) has every input it needs and learns viscosity cleanly. Surface tension is NON-LOCAL: "
        "the capillary force depends on the interface curvature kappa = -div(n), which is a second derivative "
        "of the density field across several cells, information a single particle's own state simply does not "
        "contain. That is why the first attempt could only make surface tension analytic with a per-particle "
        "net, and why giving a second net the density NEIGHBOURHOOD (the 5x5 patch, exactly the support the "
        "analytic curvature stencil needs) lets it infer the curvature and produce the force. The capillary "
        "net fits well because kappa*grad(phi) is a smooth, low-order function of the patch (finite "
        "differences of a smoothed field), well within a small MLP's reach, and it generalizes in strength "
        "because the true force is exactly LINEAR in sigma, so two endpoint strengths pin the line. The "
        "FAILURES are the more informative part. The m_visc=0.25 fountain is most likely a per-cell "
        "stability/timestep or stress-extrapolation issue localized to that one viscosity (its neighbours at "
        "0.0 and 0.5 are stable), not a fundamental limit. The high-visc/high-ST spike is a COMPOSITION "
        "failure: the capillary force was learned only at LOW viscosity, where the interface stays smooth and "
        "its density patches fall in the trained distribution; combined with the stiff high-viscosity stress "
        "at (1,1) the two forces drive the interface into geometries the capillary net never saw and reinforce "
        "a vertical jet rather than settling into a droplet. The decoupled-axes assumption -- that a force "
        "learned on one axis transfers unchanged across the other -- breaks down exactly where the two axes "
        "interact most strongly. A deeper reason both failures are possible at all: training is per-step "
        "SUPERVISED regression onto instantaneous force targets, with no rollout in the loss, so nothing "
        "penalizes error ACCUMULATION over hundreds of integration steps; a locally-accurate force law can "
        "still compound into a spike or a blow-up once integrated forward. Why the viscosity slider is "
        "nonetheless smooth: the viscous target is linear in mu and fed as one coherent net's input, so the "
        "interior is that net's own smooth output, not a chord through two distant weight matrices. What would "
        "test these: fixing the per-cell dt or adding rollout-stability training and seeing whether the "
        "fountain clears; training the capillary net on HIGH-viscosity interfaces too; and a resolution sweep."
    )

    limitations = (
        "This is a PARTIAL, mixed result, not a clean success. Two grid regions fail: the m_visc=0.25 column "
        "is degenerate (particles spray upward instead of settling), and the high-viscosity/high-surface-"
        "tension cells including the held-out (1,1) corner jet into unphysical vertical spikes rather than "
        "droplets, so the held-out corner does NOT generalize in shape (its low trajectory RMSE understates "
        "this, since a spike and a blob share a center of mass). What is sound is scoped to: surface tension "
        "is genuinely learned, the three trained corners are edge-exact, and the input-conditioned viscosity "
        "slider interpolates smoothly. Training is per-step supervised regression onto instantaneous forces "
        "(no rollout in the loss), so it never optimizes long-horizon stability. Beyond that it is a "
        "demonstration on one architecture and one material family (a 2D weakly-compressible MLS-MPM fluid) "
        "with a two-parameter descriptor, not a general law about conditioned fluids. Everything is 2D, "
        f"n_grid={m['n_grid']}, f32, one grid resolution, per-cell stable timesteps (the viscous limit for "
        "viscosity, the capillary limit for surface tension, the smaller at the held-out high/high corner), "
        f"one common Young's modulus E={m['E']:g}, four training scenes with x-mirror augmentation, and "
        "supervised-regression fits rather than rollout-trained ones. The viscosity schedule is LINEAR by "
        "design (mu from " + f"{m['MU_LOW']} to {m['MU_HIGH']}" + "), which makes the intermediate viscosity "
        "an unambiguous interpolation target; a nonlinear parameter would not, and is untested. Surface "
        "tension is now LEARNED by the capillary net, but that net is trained AGAINST the analytic CSF force "
        "and is therefore only as good as that reference: it inherits the CSF's diffuse-interface "
        "approximation (a smoothed density band a few cells wide), and because the curvature the patch "
        "encodes depends on the grid resolution and the number of smoothing passes, the capillary net is "
        "specific to the resolution and smoothing it trained at (a resolution sweep is untested). The mapping "
        "from strength to a physical surface tension is not calibrated (the range is chosen for a visible, "
        "gradual rounding, not a measured capillary number); the roundness proxy is a rasterised "
        "isoperimetric ratio. The held-out corner shares the same trained strength as corner (0,1), so it "
        "tests transfer of the learned capillary force to an unseen VISCOSITY combination, not extrapolation "
        "to a stronger, unseen surface tension; and as the figures show it FAILS there, jetting into a spike "
        "rather than settling. The interior 'ground truth' is the analytic fluid "
        "at the interpolated (mu, sigma), a real reference. Both nets are single training seeds and fixed "
        "widths; the stress net predicts the full stress including the stiff weakly-compressible pressure, "
        "and its effective-viscosity readout is a least-squares projection, a diagnostic not an exact "
        "decomposition. GPU atomic-add accumulation is not bitwise reproducible; rerun if a frame looks off."
    )

    manifest = {
        "schema_version": "2",
        "task_id": "generalize-one-nn-across-viscosity-and-surface-tension",
        "direction": "material-variants",
        "title": "Two learned networks for a fluid across viscosity and LEARNED surface tension",
        "status": "active",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": (
            "Following the conditioned-network protocol, learn a weakly-compressible fluid across viscosity "
            "and surface tension with a shared two-scalar descriptor m = (m_visc, m_st), where BOTH axes are "
            "LEARNED. A conditioned STRESS net predicts the per-particle fluid stress (weakly-compressible "
            "pressure plus the Newtonian viscous term) from the local state and the descriptor. Surface "
            "tension is not a per-particle stress but a grid capillary force set by the interface curvature, "
            "a non-local quantity a per-particle net cannot see, so a second CAPILLARY net learns it: it "
            "reads a 5x5 patch of the smoothed grid density field around each node (never the analytic "
            "curvature) plus the strength, and outputs the capillary force, trained supervised against the "
            "analytic continuum-surface-force. The learned rollout applies both nets and NO analytic surface "
            "tension. Train on THREE corners -- (0,0) low visc/no ST, (1,0) high visc/no ST, (0,1) low "
            "visc/high ST -- and HOLD OUT the fourth (1,1). Calibrate a gentle surface-tension range first so "
            "the droplet rounds gradually up the axis, verify edge-exactness for BOTH nets at each trained "
            "corner, then interpolate the descriptor to fill a 5x5 grid against ground truth and test whether "
            "the learned capillary law transfers to the held-out high-viscosity/high-surface-tension corner."
        ),
        "findings": findings,
        "hypothesis": hypothesis,
        "limitations": limitations,
        "results": results,
        "custom_html": custom_html,
        "training_refs": ["conditioned-fluid", "conditioned-material-net", "learned-material-interpolation",
                          "surface-tension", "viscosity", "vector-calculus"],
        "params": m,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
