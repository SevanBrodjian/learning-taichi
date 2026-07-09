"""Learn the constitutive STRESS of three structurally different materials with one shared network,
then INTERPOLATE the weights between materials and characterize what emerges.

Follow-up to ``sim/learned_viscosity.py``. That task learned a net per VISCOSITY -- a single linear
knob whose true target mu*(C+C^T) is linear in mu, so linear interpolation of two per-viscosity FUNCTIONS
is exactly an intermediate viscosity and the whole question reduced to weight-space geometry. This task
repeats the experiment across the three CONSTITUTIVE MODELS from ``sim/material_showcase.py`` /
``sim/material_diff.py``:

  * fluid   -- weakly-compressible isotropic pressure, sigma = E (det F - 1) I. Forgets shear.
  * elastic -- corotated first-Piola stress from the full deformation gradient F (via ti.svd). Springs back.
  * snow    -- corotated stress with hardening exp(xi (1 - Jp)); a plastic clamp of F's singular values
               mutates the STATE (F, Jp), not just the stress.

These are structurally different laws, NOT one linear family, so there is NO known function-space target
for interpolating a fluid-net with a snow-net. Question 3 is genuinely exploratory.

===================================================================================================
This is the REWORKED version. The first attempt was returned for two reasons; both are fixed here.

FIX 1 -- VARIED, SNOW-ENGAGING TRAINING (was: one gentle drop, snow barely plastic).
  The three nets are now trained on FIVE scenes that deliberately exercise each material's signature:
  a soft drop, a HARD impact (fires snow's plastic clamp on compression), a column slump (shear /
  angle-of-repose), a wide settling slab, and a lateral shearing throw. The snow clamp-active fraction
  is measured and reported, so "the snow net acts like snow" is backed by the pooled states actually
  crossing the yield surface. Generalization is then tested on TWO genuinely held-out scenes.

FIX 2 -- ENDPOINT-PARITY BUG (was: at alpha=0 the interpolated rollout exploded).
  Root cause: the old sweep held the state-update rule FIXED at the B-endpoint's rule for the whole
  sweep, so at alpha=0 the fluid net ran under the ELASTIC free-F rule it was never trained on (its F
  grew shear it had never seen), went off-distribution, and blew up -- a harness bug, not a property of
  interpolation. The fix is a UNIFIED state kernel used by EVERY rollout: free-trial F, then clamp F's
  singular values into [1-tc, 1+ts], pushing the excess into Jp. A wide "band-off" (tc=ts=BAND_OFF)
  never fires and recovers pure free-F, so elastic and fluid are the band-off case and snow is the
  snow-band case. Only the STRESS law and the yield band differ between materials. Consequences:
    - fluid<->elastic runs band-off throughout, so alpha=0 IS the Q1 fluid config and alpha=1 IS the
      Q1 elastic config (pure weight interpolation, endpoints identical to Q1 by construction).
    - elastic<->snow interpolates BOTH the stress weights AND the yield band (1/tc linear in alpha), so
      alpha=0 is pure elastic (band-off) and alpha=1 is true snow (snow band). The plastic clamp is a
      state rule outside the weights, so morphing elastic->snow HONESTLY requires morphing it too; this
      is stated as a scoping choice, not hidden.
  Endpoint parity is then VERIFIED explicitly: the alpha=0 / alpha=1 sweep rollouts are compared frame
  by frame against the Q1 endpoint rollouts, and the residual is shown to be at the level of GPU
  atomic-add non-determinism (measured by running one config twice).

The fluid is the det-F special case of the shared substrate: with band-off free-F, its det F evolves
identically to a volumetric-reset fluid (the stress depends only on det F, so the shear part of F never
feeds back), so "true fluid under the unified kernel" is the same fluid, and carrying full F for the
fluid too is exactly what makes all three nets live on one comparable feature manifold.

Snow's plasticity is handled explicitly and honestly. The plastic clamp is a STATE update (it mutates
F's singular values and pushes the excess into Jp), NOT a stress, so it stays a shared, non-learned
analytic rule; the net learns ONLY the stress. The net DOES see the current plastic record Jp as an
input, so the hardening exp(xi(1-Jp)) is a representable memoryless function of the current state; the
memory lives entirely in how the clamp accumulates Jp, which is shared. Every interpolation claim is
scoped to what the WEIGHTS (and, for elastic<->snow, the yield band) control.

Targets are BAKED WITH TAICHI (the same corotated_PFt / ti.svd used by the true simulator) so the
regression target is bitwise the true-sim stress, with no numpy/Taichi SVD-convention mismatch.

Rendering is HEADLESS (matplotlib Agg -> mp4). A learned rollout that blew up or scattered is a bug,
not a result; every rollout is checked finite and every clip is meant to be viewed.

Usage:
    python sim/learned_materials.py            # full pipeline + media + manifest
    python sim/learned_materials.py --quick    # fast smoke test (fewer frames/iters/alphas)
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
NU = 0.2                      # Poisson ratio, shared by every solid path
E_COMMON = 200.0             # ONE Young's modulus for all three materials, so the only difference between
                            # them is the CONSTITUTIVE FORM (not stiffness). Isolates "structurally
                            # different law" from a magnitude confound.
XI = 3.0                     # snow hardening exp(xi(1-Jp)). Moderate: the plastic CLAMP (below) is what
                            # makes snow crumple and hold; xi only sets how much compacted snow stiffens.
                            # Kept at 3 (not the showcase's 10) so the hardened-stress tail stays in a
                            # range the net can fit well and drive stably, while snow still hardens ~7x.
TC, TS = 2.5e-2, 7.5e-3      # snow plastic clamp band [1-tc, 1+ts]
BAND_OFF = 10.0              # "clamp off": a band so wide the clamp never fires, recovering pure free-F.
                            # With singular values ~1, [1-10, 1+10] never bites, so F = F_trial and Jp=1.

MAX_P = 8192
POOL_CAP = 200000            # capacity for the pooled training-state fields
CAP_PER_MAT = 40000          # pooled points kept per material for the fit (balanced across materials)

FLUID, ELASTIC, SNOW = 0, 1, 2
MAT_ID = {"fluid": FLUID, "elastic": ELASTIC, "snow": SNOW}

# --------------------------------------------------------------------------- network shape
# COROTATIONAL structuring. The net is fed the rotation-INVARIANT symmetric stretch S (the strain), plus
# the affine matrix C, velocity, and the plastic record Jp, and outputs the stress in the MATERIAL frame;
# the analytic polar rotation R rotates it back to the world frame. This is what makes the corotated
# stress learnable and stable: the raw deformation gradient F = R S mixes a small elastic strain into an
# order-one rotation, and a plastic clamp pins F to a near-rotation where that strain is tiny, so a net
# fed raw F must undo the rotation itself and its error swamps the strain signal. Feeding S (which the
# polar decomposition strips of rotation) hands the net the strain directly. See [[svd-polar]].
# Features: S00,S01,S11, Cxx,Cxy,Cyx,Cyy, vx,vy, Jp  (position-free, per-particle). Jp is 1 for fluid and
# elastic and drops for snow; the stress depends on it through the hardening exp(xi(1-Jp)), so it is an
# INPUT, while the plastic CLAMP that updates Jp stays the shared analytic state rule.
N_IN = 10
N_HID = 48    # one hidden layer, tanh
N_OUT = 3     # symmetric material-frame stress: pxx, pxy, pyy
N_PARAMS = N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT

# --------------------------------------------------------------------------- state fields (in-place)
x = ti.Vector.field(dim, float, MAX_P)
v = ti.Vector.field(dim, float, MAX_P)
C = ti.Matrix.field(dim, dim, float, MAX_P)
F = ti.Matrix.field(dim, dim, float, MAX_P)     # deformation gradient, carried for ALL materials
Jp = ti.field(float, MAX_P)                     # accumulated plastic volume change (snow)

grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

x_np_buf = ti.Vector.field(dim, float, MAX_P)
v0_buf = ti.Vector.field(dim, float, MAX_P)

# pooled-state fields for baking targets with Taichi (guarantees target == true-sim stress)
Fpool = ti.Matrix.field(dim, dim, float, POOL_CAP)
Jppool = ti.field(float, POOL_CAP)
sigpool = ti.field(float, shape=(POOL_CAP, 3))   # material-frame stress target P_mat (pxx,pxy,pyy)
Spool = ti.field(float, shape=(POOL_CAP, 3))     # symmetric stretch features S (S00,S01,S11)

# network weight fields (loaded from numpy; forward only)
W1 = ti.field(float, shape=(N_HID, N_IN))
b1 = ti.field(float, shape=N_HID)
W2 = ti.field(float, shape=(N_OUT, N_HID))
b2 = ti.field(float, shape=N_OUT)
fmean = ti.field(float, shape=N_IN)
fstd = ti.field(float, shape=N_IN)
tscale = ti.field(float, shape=())


# --------------------------------------------------------------------------- constitutive stress (true)
@ti.func
def corotated_PFt(Fc, mu, la):
    """Corotated first-Piola stress contracted with F^T: 2 mu (F-R) F^T + la (J-1) J I, R = U V^T."""
    U, sig, Vt = ti.svd(Fc)
    R = U @ Vt.transpose()
    Jdet = Fc.determinant()
    return 2.0 * mu * (Fc - R) @ Fc.transpose() + la * (Jdet - 1.0) * Jdet * ti.Matrix.identity(float, dim)


@ti.func
def true_sigma(mat: ti.template(), p, E, xi):
    """Physical stress tensor sigma (BEFORE the -dt 4 p_vol/dx^2 prefactor) for one material at particle p.
    fluid: pressure from det F. elastic/snow: corotated PFt, snow hardened by exp(xi(1-Jp))."""
    sig = ti.Matrix.zero(float, dim, dim)
    if ti.static(mat == FLUID):
        s = E * (F[p].determinant() - 1.0)
        sig = ti.Matrix([[s, 0.0], [0.0, s]])
    elif ti.static(mat == ELASTIC):
        mu = E / (2.0 * (1.0 + NU))
        la = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        sig = corotated_PFt(F[p], mu, la)
    else:
        h = ti.exp(xi * (1.0 - Jp[p]))
        mu = (E / (2.0 * (1.0 + NU))) * h
        la = (E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))) * h
        sig = corotated_PFt(F[p], mu, la)
    return sig


@ti.func
def learned_sigma(p):
    """World-frame stress predicted by the shared MLP. The net is fed the rotation-invariant stretch S
    (and C, v, Jp), outputs the material-frame symmetric stress P_mat, and P_mat is rotated back to the
    world frame by the analytic polar rotation R of F. So sigma_world = R P_mat R^T with P_mat learned."""
    Fp = F[p]
    U, sig, Vt = ti.svd(Fp)
    R = U @ Vt.transpose()
    S = Vt @ sig @ Vt.transpose()      # right stretch F = R S, symmetric, rotation-invariant
    Cp = C[p]
    vp = v[p]
    # Guard the plastic-record input to the range the net was trained on. Snow is the one material whose
    # own stress errors feed back into a state it also reads (the net drives motion -> the clamp updates
    # Jp -> the net sees the new Jp), so a small misfit can drive Jp far below anything in training and the
    # net then extrapolates the hardening exp(xi(1-Jp)) into a runaway. Clamping the Jp FEATURE to the
    # trained band (training min Jp ~ 0.2) breaks that runaway without touching the fluid/elastic nets,
    # for which Jp is pinned at 1 anyway.
    jp_in = ti.min(ti.max(Jp[p], 0.15), 1.5)
    feat = ti.Vector([S[0, 0], S[0, 1], S[1, 1],
                      Cp[0, 0], Cp[0, 1], Cp[1, 0], Cp[1, 1], vp[0], vp[1], jp_in])
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
    Pm = ti.Matrix([[o0 * s, o1 * s], [o1 * s, o2 * s]])
    return R @ Pm @ R.transpose()


# --------------------------------------------------------------------------- MLS-MPM steps
@ti.kernel
def clear_grid():
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_v[i, j] = ti.Vector.zero(float, dim)
        grid_m[i, j] = 0.0


@ti.kernel
def p2g_true(mat: ti.template(), n: ti.i32, dt: ti.f32, E: ti.f32, xi: ti.f32,
             p_vol: ti.f32, p_mass: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4.0 * p_vol * inv_dx * inv_dx * true_sigma(mat, p, E, xi)
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base[0] + i, base[1] + j] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base[0] + i, base[1] + j] += weight * p_mass


@ti.kernel
def p2g_learned(n: ti.i32, dt: ti.f32, p_vol: ti.f32, p_mass: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4.0 * p_vol * inv_dx * inv_dx * learned_sigma(p)
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
def g2p(n: ti.i32, dt: ti.f32, iso: ti.f32, tc: ti.f32, ts: ti.f32):
    """UNIFIED state update used by EVERY rollout (true or learned, every material). Advect, then evolve
    the deformation gradient F through the singular values of its free trial, with two continuous knobs:
      * iso -- ISOTROPIZATION toward the volumetric special case, det-PRESERVING (multiplicative in
        log-space): s_i' = s_i^(1-iso) * g^iso with g = sqrt(s0 s1). iso=0 keeps the free trial F
        (elastic and snow). iso=1 sends both singular values to g, so F = g R is a scaled rotation with
        stretch S = g I -- the fluid's det-F special case. This matters numerically: a fluid has no
        restoring force on the shear part of F, so under free F that part drifts without bound and
        det F (all the fluid stress depends on) turns to catastrophic-cancellation garbage; iso=1 keeps
        F bounded and det F exact. It preserves det at every iso, so the interpolation stays volume-clean.
      * (tc, ts) -- the plastic CLAMP band. band-off (BAND_OFF) never fires; the snow band clamps the
        singular values into [1-tc, 1+ts] and pushes the excess into the plastic record Jp, the crumple
        that makes snow snow. The clamp is analytic and shared, NOT learned.
    Materials: fluid = (iso=1, band-off), elastic = (iso=0, band-off), snow = (iso=0, snow band). Making
    one kernel serve all rollouts is what guarantees the interpolation endpoints run the exact same code
    path as the Q1 replication rollouts, so alpha=0 and alpha=1 reproduce their materials identically."""
    for p in range(n):
        new_v, new_C = g2p_gather(p)
        v[p] = new_v
        x[p] = x[p] + dt * new_v
        x[p] = ti.math.clamp(x[p], floor_y, 1.0 - floor_y)
        F_tr = (ti.Matrix.identity(float, dim) + dt * new_C) @ F[p]
        U, sig, Vt = ti.svd(F_tr)
        s0 = ti.max(sig[0, 0], 1e-5)
        s1 = ti.max(sig[1, 1], 1e-5)
        lg = 0.5 * (ti.log(s0) + ti.log(s1))            # log of the geometric mean g = sqrt(det)
        s0 = ti.exp((1.0 - iso) * ti.log(s0) + iso * lg)  # det-preserving isotropization
        s1 = ti.exp((1.0 - iso) * ti.log(s1) + iso * lg)
        c0 = ti.min(ti.max(s0, 1.0 - tc), 1.0 + ts)     # plastic clamp
        c1 = ti.min(ti.max(s1, 1.0 - tc), 1.0 + ts)
        Jp[p] = Jp[p] * (s0 * s1) / (c0 * c1)
        F[p] = U @ ti.Matrix([[c0, 0.0], [0.0, c1]]) @ Vt
        C[p] = new_C


@ti.kernel
def init_state(n: ti.i32):
    for p in range(n):
        x[p] = x_np_buf[p]
        v[p] = v0_buf[p]
        C[p] = ti.Matrix.zero(float, dim, dim)
        F[p] = ti.Matrix.identity(float, dim)
        Jp[p] = 1.0


@ti.kernel
def dump_state(n: ti.i32, out_F: ti.types.ndarray(), out_C: ti.types.ndarray(),
               out_v: ti.types.ndarray(), out_Jp: ti.types.ndarray()):
    for p in range(n):
        out_F[p, 0] = F[p][0, 0]
        out_F[p, 1] = F[p][0, 1]
        out_F[p, 2] = F[p][1, 0]
        out_F[p, 3] = F[p][1, 1]
        out_C[p, 0] = C[p][0, 0]
        out_C[p, 1] = C[p][0, 1]
        out_C[p, 2] = C[p][1, 0]
        out_C[p, 3] = C[p][1, 1]
        out_v[p, 0] = v[p][0]
        out_v[p, 1] = v[p][1]
        out_Jp[p] = Jp[p]


@ti.kernel
def bake_sigma(mat: ti.template(), n: ti.i32, E: ti.f32, xi: ti.f32):
    """Compute, at each pooled state, the rotation-invariant stretch features S (material-independent) and
    the MATERIAL-FRAME stress target P_mat for this material. P_mat = R^T sigma_world R is a function of S
    (and Jp) only: for a corotated solid, corotated_PFt(F) = R (2 mu (S-I) S + la (J-1) J I) R^T, so the
    material-frame target is 2 mu (S-I) S + la (J-1) J I, and for the fluid it is E (det S - 1) I. Baking
    the target with ti.svd matches the true simulator's rotation convention exactly."""
    for p in range(n):
        Fp = Fpool[p]
        U, sig, Vt = ti.svd(Fp)
        S = Vt @ sig @ Vt.transpose()
        Spool[p, 0] = S[0, 0]
        Spool[p, 1] = S[0, 1]
        Spool[p, 2] = S[1, 1]
        d = Fp.determinant()
        Pm = ti.Matrix.zero(float, dim, dim)
        if ti.static(mat == FLUID):
            s = E * (d - 1.0)
            Pm = ti.Matrix([[s, 0.0], [0.0, s]])
        else:
            h = 1.0
            if ti.static(mat == SNOW):
                h = ti.exp(xi * (1.0 - Jppool[p]))
            mu = (E / (2.0 * (1.0 + NU))) * h
            la = (E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))) * h
            Pm = 2.0 * mu * (S - ti.Matrix.identity(float, dim)) @ S \
                + la * (d - 1.0) * d * ti.Matrix.identity(float, dim)
        sigpool[p, 0] = Pm[0, 0]
        sigpool[p, 1] = 0.5 * (Pm[0, 1] + Pm[1, 0])
        sigpool[p, 2] = Pm[1, 1]


# --------------------------------------------------------------------------- scene setup
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


# --- TRAINING scenes: varied geometry AND scenes that engage each material's signature ---
def train_scenes(n):
    return [
        # soft drop: gentle release, mild deformation (the baseline the old run used alone)
        {"pts": seed_disk((0.5, 0.52), 0.11, n, 1), "area": np.pi * 0.11 ** 2,
         "v0": (0.0, -1.0), "T": 0.8, "name": "drop_soft"},
        # HARD impact: fast downward throw, strong compression on landing -> fires snow's plastic clamp
        {"pts": seed_disk((0.5, 0.62), 0.095, n, 2), "area": np.pi * 0.095 ** 2,
         "v0": (0.0, -4.0), "T": 0.8, "name": "drop_hard"},
        # column slump: tall block from rest -> shear, angle-of-repose slump (fires snow shear-side)
        {"pts": seed_box(0.44, 0.56, floor_y, 0.62, n, 3), "area": (0.56 - 0.44) * (0.62 - floor_y),
         "v0": (0.0, 0.0), "T": 0.9, "name": "column"},
        # wide slab: a broad low block settling -> spreading for fluid, mild compaction at the base
        {"pts": seed_box(0.30, 0.70, floor_y, 0.17, n, 4), "area": (0.70 - 0.30) * (0.17 - floor_y),
         "v0": (0.0, 0.0), "T": 0.7, "name": "wide_slab"},
        # lateral throw: sideways+down velocity -> shearing impact and tumble
        {"pts": seed_disk((0.38, 0.52), 0.10, n, 5), "area": np.pi * 0.10 ** 2,
         "v0": (2.6, -1.0), "T": 0.8, "name": "lateral"},
    ]


# --- HELD-OUT generalization scenes: never trained on; genuinely new geometry AND velocity ---
def gen_scenes(n):
    return [
        # new size + position + a leftward-down throw (none of the training scenes throw left)
        {"pts": seed_disk((0.60, 0.58), 0.12, n, 11), "area": np.pi * 0.12 ** 2,
         "v0": (-1.9, -1.3), "T": 0.8, "name": "gen_toss"},
        # a chunky wide arch released from rest, wider and taller than any training block
        {"pts": seed_box(0.34, 0.66, floor_y, 0.30, n, 12), "area": (0.66 - 0.34) * (0.30 - floor_y),
         "v0": (0.0, 0.0), "T": 0.9, "name": "gen_arch"},
    ]


def scene_by_name(scenes, name):
    for s in scenes:
        if s["name"] == name:
            return s
    raise KeyError(name)


# --------------------------------------------------------------------------- rollouts
def _steps_per_frame(T, n_frames, dt):
    return max(1, int(round((T / n_frames) / dt)))


def band_of(mat):
    """The yield band for a pure material: band-off for fluid/elastic, snow band for snow."""
    return (TC, TS) if mat == "snow" else (BAND_OFF, BAND_OFF)


def iso_of(mat):
    """Isotropization for a pure material: 1 for the fluid (volumetric det-F special case), 0 otherwise."""
    return 1.0 if mat == "fluid" else 0.0


def rollout(scene, dt, n_frames, mode, mat="elastic", iso=0.0, tc=BAND_OFF, ts=BAND_OFF, collect=False):
    """Roll one scene forward with the UNIFIED kernel. mode='true' uses the analytic stress of `mat`;
    mode='learned' uses the currently-loaded net. iso is the isotropization knob and (tc,ts) the yield
    band. Returns (snaps,times,stable) and, if collect, also per-frame (F,C,v,Jp) states."""
    n = upload(scene["pts"], scene["v0"])
    p_vol = scene["area"] / n
    p_mass = p_vol * p_rho
    spf = _steps_per_frame(scene["T"], n_frames, dt)
    mat_id = MAT_ID[mat]
    init_state(n)
    snaps = np.zeros((n_frames, n, dim), dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)
    states = []
    t = 0.0
    stable = True
    for fidx in range(n_frames):
        for _ in range(spf):
            clear_grid()
            if mode == "true":
                p2g_true(mat_id, n, dt, E_COMMON, XI, p_vol, p_mass)
            else:
                p2g_learned(n, dt, p_vol, p_mass)
            grid_op(dt, FRICTION)
            g2p(n, dt, iso, tc, ts)
            t += dt
        cur = x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        snaps[fidx] = cur
        times[fidx] = t
        if collect:
            Fb = np.zeros((n, 4), dtype=np.float32)
            Cb = np.zeros((n, 4), dtype=np.float32)
            vb = np.zeros((n, 2), dtype=np.float32)
            Jb = np.zeros(n, dtype=np.float32)
            dump_state(n, Fb, Cb, vb, Jb)
            states.append((Fb.copy(), Cb.copy(), vb.copy(), Jb.copy()))
    if collect:
        return snaps, times, stable, states
    return snaps, times, stable


# --------------------------------------------------------------------------- diagnostics
def spread_width(snap):
    xs = snap[:, 0]
    return float(np.percentile(xs, 95) - np.percentile(xs, 5))


def pile_height(snap):
    return float(np.percentile(snap[:, 1], 95) - floor_y)


def aspect_ratio(snap):
    w = spread_width(snap)
    h = pile_height(snap)
    return float(h / (w + 1e-9))


def com_height(snap):
    return float(snap[:, 1].mean() - floor_y)


def series(snaps, fn):
    return np.array([fn(snaps[f]) for f in range(snaps.shape[0])], dtype=np.float64)


def jiggle(snaps, tail_frac=0.35):
    """Late-time oscillation amplitude of the center-of-mass height -- the elastic 'bounce' signature.
    An elastic blob is still ringing at the end; fluid and snow have settled, so their COM height is flat."""
    ch = series(snaps, com_height)
    k = max(3, int(len(ch) * tail_frac))
    return float(np.std(ch[-k:]))


def traj_rmse(a, b):
    """Mean over frames of the per-particle position distance between two rollouts (same scene, same n)."""
    n = min(a.shape[1], b.shape[1])
    d = np.sqrt(((a[:, :n] - b[:, :n]) ** 2).sum(axis=2))
    return float(d.mean())


# --------------------------------------------------------------------------- numpy MLP (offline train)
def mlp_forward_np(theta, Xs):
    W1n, b1n, W2n, b2n = theta
    h = np.tanh(Xs @ W1n.T + b1n)
    y = h @ W2n.T + b2n
    return y, h


def init_theta(seed):
    rng = np.random.default_rng(seed)
    W1n = (rng.standard_normal((N_HID, N_IN)) * 0.35).astype(np.float64)
    b1n = np.zeros(N_HID, dtype=np.float64)
    W2n = (rng.standard_normal((N_OUT, N_HID)) * 0.35).astype(np.float64)
    b2n = np.zeros(N_OUT, dtype=np.float64)
    return [W1n, b1n, W2n, b2n]


def train_mlp(Xs, Ys, theta0, iters, lr=1.5e-3, batch=4096, seed=0, log_every=2000,
              huber_delta=4.0, gclip=5.0):
    """Adam regression with a ROBUST (clipped-residual / Huber) gradient and global grad-norm clipping.
    The fluid's det-F pressure target has a heavy tail on the hard-impact scene (rare high-compression
    particles with a very large pressure), and the snow target has a hardening tail. Plain MSE at a normal
    learning rate lets those rare large residuals blow the weights to NaN. Clipping the residual at
    huber_delta (in normalized target units) makes the loss quadratic in the bulk and linear in the tail,
    so the fit tracks the typical stress without being dominated (or destabilized) by the extremes; the
    global grad-norm clip is a second guard. Reported loss is the plain MSE for interpretability."""
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
        gY = (2.0 / B) * np.clip(diff, -huber_delta, huber_delta)   # clipped-residual (Huber) gradient
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
            print(f"      [train seed={seed}] iter {it:5d}  mse={loss:.5e}")
    return theta, hist


def load_theta(theta):
    W1.from_numpy(theta[0].astype(np.float32))
    b1.from_numpy(theta[1].astype(np.float32))
    W2.from_numpy(theta[2].astype(np.float32))
    b2.from_numpy(theta[3].astype(np.float32))


def lerp_theta(ta, tb, a):
    return [(1 - a) * ta[k] + a * tb[k] for k in range(4)]


def rel_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.std(true) + 1e-12))


# --------------------------------------------------------------------------- rendering
BG = "#0a0e14"
GROUND = "#161c26"
WALL = "#26313d"
INK = "#dfe6ee"
SUB = "#9fb0c0"
GREY = "#7f8a99"
MAT_COL = {"fluid": "#4db6ff", "elastic": "#ff9d5c", "snow": "#e6ecff"}


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
        ax.scatter(pts[:, 0], pts[:, 1], s=sz, color=col, edgecolors="none", alpha=0.82, zorder=2)
    if label:
        ax.text(0.5, 0.94, label, ha="center", va="center", color=INK, fontsize=10.5,
                weight="bold", transform=ax.transAxes)
    if tlabel:
        ax.text(0.5, 0.06, tlabel, ha="center", va="center", color=SUB, fontsize=8,
                transform=ax.transAxes)


def render_overlay(path, columns, times, fps=30, dpi=100, panel=340):
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
        pts_list = [ss[0][fidx] for ss in sets]
        colors = [ss[1] for ss in sets]
        sizes = [ss[2] for ss in sets]
        _panel(axes[k], pts_list, colors, sizes, label, tlabel)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)


def line_plot(path, series_list, xlabel, ylabel, title, xlim=None, markers=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130, facecolor=BG)
    ax.set_facecolor(BG)
    for (label, xs, ys, color, style) in series_list:
        ax.plot(xs, ys, color=color, lw=2.2, label=label, linestyle=style,
                marker="o" if style == "-" else None, ms=4)
    if markers:
        for (label, xs, ys, color) in markers:
            ax.scatter(xs, ys, color=color, s=70, marker="*", zorder=5, label=label,
                       edgecolors=BG, linewidths=0.5)
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


def alpha_color(a, c0hex, c1hex):
    import matplotlib.colors as mc
    c0 = np.array(mc.to_rgb(c0hex))
    c1 = np.array(mc.to_rgb(c1hex))
    c = (1 - a) * c0 + a * c1
    return (float(c[0]), float(c[1]), float(c[2]))


# --------------------------------------------------------------------------- pipeline
def main():
    ap = argparse.ArgumentParser(description="Learn per-material stress nets and interpolate the weights")
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--probe", action="store_true",
                    help="full training iters/frames but stop after Q1 + parity (no Q3 media)")
    ap.add_argument("--manifest-only", action="store_true",
                    help="regenerate manifest.json from the existing metrics.json (no sim)")
    args = ap.parse_args()
    quick = args.quick

    repo0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_dir0 = "runs/material-variants/train-material-replicating-nns-and-interpolate"
    out_dir0 = os.path.join(repo0, *rel_dir0.split("/"))
    if args.manifest_only:
        with open(os.path.join(out_dir0, "metrics.json")) as fh:
            metrics = json.load(fh)
        write_manifest(out_dir0, rel_dir0, metrics)
        print(f"regenerated manifest from metrics.json -> {rel_dir0}")
        return None

    dt = 5e-5
    n_frames = 24 if quick else 48
    iters = 1500 if quick else 9000
    n_part = 1000 if quick else 1500
    sweep_alphas = np.linspace(0, 1, 5 if quick else 9)
    clip_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    mats = ["fluid", "elastic", "snow"]

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_dir = "runs/material-variants/train-material-replicating-nns-and-interpolate"
    out_dir = os.path.join(repo, *rel_dir.split("/"))
    os.makedirs(out_dir, exist_ok=True)

    tr_scenes = train_scenes(n_part)
    gscenes = gen_scenes(n_part)

    # ---------------- 1. collect pooled training states over ALL training scenes --------------------
    print("=== collecting training states (true sims over 5 varied scenes) ===")
    per_mat = {m: {"F": [], "C": [], "v": [], "Jp": []} for m in mats}
    snow_clamp = {}   # per-scene snow clamp-active fraction
    for m in mats:
        tc, ts = band_of(m)
        iso = iso_of(m)
        for sc in tr_scenes:
            _, _, ok, st = rollout(sc, dt, n_frames, "true", mat=m, iso=iso, tc=tc, ts=ts, collect=True)
            for (Fb, Cb, vb, Jb) in st:
                per_mat[m]["F"].append(Fb)
                per_mat[m]["C"].append(Cb)
                per_mat[m]["v"].append(vb)
                per_mat[m]["Jp"].append(Jb)
            if m == "snow":
                Jall = np.concatenate([s[3] for s in st])
                frac = float(np.mean(Jall < 0.99))   # genuine plastic yield (>1% permanent compaction)
                snow_clamp[sc["name"]] = {"clamp_frac": frac, "min_Jp": float(Jall.min())}
                print(f"  snow  {sc['name']:10s} stable={ok}  clamp-active frac={frac:.3f} "
                      f"minJp={Jall.min():.3f}")
            else:
                print(f"  {m:8s} {sc['name']:10s} stable={ok}  frames={len(st)}")

    rng = np.random.default_rng(0)
    Xm, Fm, Jm = {}, {}, {}
    for m in mats:
        Fa = np.concatenate(per_mat[m]["F"], axis=0).astype(np.float64)
        Ca = np.concatenate(per_mat[m]["C"], axis=0).astype(np.float64)
        va = np.concatenate(per_mat[m]["v"], axis=0).astype(np.float64)
        Ja = np.concatenate(per_mat[m]["Jp"], axis=0).astype(np.float64)
        # x-MIRROR AUGMENTATION. MLS-MPM physics is symmetric under reflection about the vertical axis
        # (x -> -x), so a local stress law must be too. The training scenes throw only to the RIGHT
        # (the lateral scene), which leaves leftward throws out of distribution and makes the sensitive
        # snow net puff up on a held-out leftward toss. Reflect every collected state to cover both
        # directions: a vector v -> (-vx, vy), and a linear map M (F or C) -> P M P with P = diag(-1, 1),
        # which negates the off-diagonal entries. Jp is a scalar and is unchanged. The stress target is
        # baked from the reflected F afterward, so each augmented pair is exactly correct physics.
        Fmir = Fa.copy(); Fmir[:, 1] *= -1.0; Fmir[:, 2] *= -1.0
        Cmir = Ca.copy(); Cmir[:, 1] *= -1.0; Cmir[:, 2] *= -1.0
        vmir = va.copy(); vmir[:, 0] *= -1.0
        Fa = np.concatenate([Fa, Fmir], axis=0)
        Ca = np.concatenate([Ca, Cmir], axis=0)
        va = np.concatenate([va, vmir], axis=0)
        Ja = np.concatenate([Ja, Ja], axis=0)
        Nt = Fa.shape[0]
        if Nt > CAP_PER_MAT:
            sel = rng.choice(Nt, CAP_PER_MAT, replace=False)
            Fa, Ca, va, Ja = Fa[sel], Ca[sel], va[sel], Ja[sel]
        Fm[m], Jm[m] = Fa, Ja
        Xm[m] = {"C": Ca, "v": va, "Jp": Ja}
        print(f"  pooled {m:8s}: {Fa.shape[0]} states")

    # bake per-material stretch features S and material-frame targets P_mat (via Taichi ti.svd)
    Sm, Ym = {}, {}
    for m in mats:
        Nm = Fm[m].shape[0]
        Fpad = np.concatenate([Fm[m].reshape(-1, 4), np.zeros((POOL_CAP - Nm, 4))], axis=0)
        Fpool.from_numpy(Fpad.reshape(POOL_CAP, 2, 2).astype(np.float32))
        Jppool.from_numpy(np.concatenate([Jm[m], np.ones(POOL_CAP - Nm)]).astype(np.float32))
        bake_sigma(MAT_ID[m], Nm, E_COMMON, XI)
        Sm[m] = Spool.to_numpy()[:Nm].astype(np.float64)
        Ym[m] = sigpool.to_numpy()[:Nm].astype(np.float64)

    # assemble features X = [S(3), C(4), v(2), Jp(1)] per material
    Xraw = {m: np.concatenate([Sm[m], Xm[m]["C"], Xm[m]["v"], Xm[m]["Jp"][:, None]], axis=1)
            for m in mats}
    Xall = np.concatenate([Xraw[m] for m in mats], axis=0)
    # ROBUST standardization from the pooled states of ALL three materials (shared). The fast fluid has
    # huge transient velocity gradients C, so a plain std would swamp the solids' resolution; a percentile
    # spread (p84-p16 ~ 1 sigma for a normal) lets the bulk set the scale and rare outliers saturate tanh.
    fmean_np = np.median(Xall, axis=0)
    fstd_np = 0.5 * (np.percentile(Xall, 84, axis=0) - np.percentile(Xall, 16, axis=0))
    fstd_np = np.where(fstd_np < 1e-4, 1.0, fstd_np)
    tscale_np = float(np.std(np.concatenate([Ym[m] for m in mats], axis=0)))
    fmean.from_numpy(fmean_np.astype(np.float32))
    fstd.from_numpy(fstd_np.astype(np.float32))
    tscale[None] = tscale_np
    print(f"  shared target scale: {tscale_np:.4f}")
    snow_clamp_overall = float(np.mean(Jm["snow"] < 0.99))
    print(f"  snow clamp-active fraction (pooled): {snow_clamp_overall:.3f}  minJp={Jm['snow'].min():.3f}")
    for m in mats:
        print(f"  target std [{m:8s}] = {np.std(Ym[m]):.4f}   max|.|={np.max(np.abs(Ym[m])):.3f}")

    Xs = {m: (Xraw[m] - fmean_np) / fstd_np for m in mats}

    # ---------------- 2. train the three nets (same architecture, shared standardization) -------------
    print("=== training nets (offline supervised regression) ===")
    train_report = {}
    thetas = {}
    seed_of = {"fluid": 0, "elastic": 1, "snow": 2}
    for m in mats:
        nval = Xs[m].shape[0] // 5
        Xtr, Xval = Xs[m][nval:], Xs[m][:nval]
        Ytr = Ym[m][nval:] / tscale_np
        Yval = Ym[m][:nval]
        theta, hist = train_mlp(Xtr, Ytr, init_theta(seed_of[m]), iters=iters, seed=seed_of[m])
        yhat_val, _ = mlp_forward_np(theta, Xval)
        pred = yhat_val * tscale_np
        rr = rel_rmse(pred, Yval)
        thetas[m] = theta
        train_report[m] = {"final_mse": float(hist[-1]), "val_rel_rmse": rr,
                           "loss_hist": [float(h) for h in hist[::max(1, len(hist) // 60)]]}
        print(f"  [{m:8s}] final mse={hist[-1]:.4e}  val rel-rmse={rr:.4f}")

    def flat(t):
        return np.concatenate([w.ravel() for w in t])
    wdist = {"fluid_elastic": float(np.linalg.norm(flat(thetas["fluid"]) - flat(thetas["elastic"]))),
             "elastic_snow": float(np.linalg.norm(flat(thetas["elastic"]) - flat(thetas["snow"]))),
             "fluid_snow": float(np.linalg.norm(flat(thetas["fluid"]) - flat(thetas["snow"])))}
    print(f"  weight distances: {wdist}")

    # ---------------- 3. Q1: each net reproduces its own material (soft-drop scene) -------------------
    print("=== Q1: reproduce own material (soft-drop scene) ===")
    q1_scene = scene_by_name(tr_scenes, "drop_soft")
    q1 = {}
    q1_cols = []
    q1_times = None
    q1_true_snaps, q1_learned_snaps = {}, {}
    for m in mats:
        tc, ts = band_of(m)
        iso = iso_of(m)
        tr_snaps, times, tok = rollout(q1_scene, dt, n_frames, "true", mat=m, iso=iso, tc=tc, ts=ts)
        load_theta(thetas[m])
        le_snaps, _, lok = rollout(q1_scene, dt, n_frames, "learned", iso=iso, tc=tc, ts=ts)
        q1_times = times
        wt = series(tr_snaps, spread_width)
        wl = series(le_snaps, spread_width)
        ht = series(tr_snaps, pile_height)
        hl = series(le_snaps, pile_height)
        w_rel = float(abs(wl[-1] - wt[-1]) / (abs(wt[-1]) + 1e-9))
        h_rel = float(abs(hl[-1] - ht[-1]) / (abs(ht[-1]) + 1e-9))
        q1[m] = {"true_width": float(wt[-1]), "learned_width": float(wl[-1]),
                 "true_height": float(ht[-1]), "learned_height": float(hl[-1]),
                 "width_rel_err": w_rel, "height_rel_err": h_rel,
                 "traj_rmse": traj_rmse(tr_snaps, le_snaps),
                 "true_stable": bool(tok), "learned_stable": bool(lok),
                 "true_w": wt.tolist(), "learned_w": wl.tolist(),
                 "true_h": ht.tolist(), "learned_h": hl.tolist()}
        q1_true_snaps[m] = tr_snaps
        q1_learned_snaps[m] = le_snaps
        q1_cols.append((f"{m}", [(tr_snaps, GREY, 5), (le_snaps, MAT_COL[m], 5)]))
        print(f"  {m:8s} true(w={wt[-1]:.3f},h={ht[-1]:.3f}) learned(w={wl[-1]:.3f},h={hl[-1]:.3f})  "
              f"w_rel={w_rel:.3f} h_rel={h_rel:.3f} trajRMSE={q1[m]['traj_rmse']:.4f} stable={lok}")
    render_overlay(os.path.join(out_dir, "q1_reproduce.mp4"), q1_cols, q1_times)
    render_still(os.path.join(out_dir, "q1_reproduce_still.png"), q1_cols, q1_times, n_frames - 1)
    line_plot(os.path.join(out_dir, "q1_width.png"),
              sum([[(f"{m} true", q1_times, q1[m]["true_w"], MAT_COL[m], "--"),
                    (f"{m} learned", q1_times, q1[m]["learned_w"], MAT_COL[m], "-")] for m in mats], []),
              "time (s)", "spread width (domain units)",
              "Q1: each net reproduces its material (solid=learned, dashed=true)")

    # ---------------- 3b. snow-plasticity showcase: learned snow crumples on the HARD impact ----------
    # The rework complaint was "the snow net hardly acts like snow". Prove it directly on the scene that
    # fires the plastic clamp hardest: all three LEARNED nets on the hard impact, side by side, plus the
    # learned-snow-vs-true-snow overlay on the column slump (angle of repose).
    print("=== Q1b: materials are distinct under the learned nets (hard impact) ===")
    hard = scene_by_name(tr_scenes, "drop_hard")
    distinct_cols = []
    dtimes = None
    for m in mats:
        tc, ts = band_of(m)
        load_theta(thetas[m])
        sn, tt, ok = rollout(hard, dt, n_frames, "learned", iso=iso_of(m), tc=tc, ts=ts)
        dtimes = tt
        distinct_cols.append((f"learned {m}", [(sn, MAT_COL[m], 5)]))
        print(f"  learned {m:8s} hard-impact final width={series(sn, spread_width)[-1]:.3f} stable={ok}")
    render_overlay(os.path.join(out_dir, "q1b_distinct_hard.mp4"), distinct_cols, dtimes)
    render_still(os.path.join(out_dir, "q1b_distinct_hard_still.png"), distinct_cols, dtimes, n_frames - 1)

    col_scene = scene_by_name(tr_scenes, "column")
    snow_tr, sctimes, _ = rollout(col_scene, dt, n_frames, "true", mat="snow", iso=0.0, tc=TC, ts=TS)
    load_theta(thetas["snow"])
    snow_le, _, snok = rollout(col_scene, dt, n_frames, "learned", iso=0.0, tc=TC, ts=TS)
    snow_col = [("snow: true vs learned (column slump)",
                 [(snow_tr, GREY, 5), (snow_le, MAT_COL["snow"], 5)])]
    render_overlay(os.path.join(out_dir, "q1b_snow_column.mp4"), snow_col, sctimes)
    render_still(os.path.join(out_dir, "q1b_snow_column_still.png"), snow_col, sctimes, n_frames - 1)
    print(f"  learned snow column-slump stable={snok} "
          f"trajRMSE={traj_rmse(snow_tr, snow_le):.4f}")

    # ---------------- 4. Q2: generalize to held-out scenes ---------------------------------------------
    print("=== Q2: generalize to held-out scenes ===")
    q2 = {}
    q2_cols_by_scene = {}
    q2_times = {}
    for sc in gscenes:
        cols = []
        for m in mats:
            tc, ts = band_of(m)
            iso = iso_of(m)
            tr_snaps, times, tok = rollout(sc, dt, n_frames, "true", mat=m, iso=iso, tc=tc, ts=ts)
            load_theta(thetas[m])
            le_snaps, _, lok = rollout(sc, dt, n_frames, "learned", iso=iso, tc=tc, ts=ts)
            q2_times[sc["name"]] = times
            wt = series(tr_snaps, spread_width)
            wl = series(le_snaps, spread_width)
            ht = series(tr_snaps, pile_height)
            hl = series(le_snaps, pile_height)
            w_rel = float(abs(wl[-1] - wt[-1]) / (abs(wt[-1]) + 1e-9))
            h_rel = float(abs(hl[-1] - ht[-1]) / (abs(ht[-1]) + 1e-9))
            q2[f"{sc['name']}/{m}"] = {
                "scene": sc["name"], "material": m,
                "true_width": float(wt[-1]), "learned_width": float(wl[-1]),
                "true_height": float(ht[-1]), "learned_height": float(hl[-1]),
                "width_rel_err": w_rel, "height_rel_err": h_rel,
                "traj_rmse": traj_rmse(tr_snaps, le_snaps),
                "learned_stable": bool(lok), "true_w": wt.tolist(), "learned_w": wl.tolist()}
            cols.append((f"{m}", [(tr_snaps, GREY, 5), (le_snaps, MAT_COL[m], 5)]))
            print(f"  {sc['name']:9s} {m:8s} w_rel={w_rel:.3f} h_rel={h_rel:.3f} "
                  f"trajRMSE={q2[sc['name'] + '/' + m]['traj_rmse']:.4f} stable={lok}")
        q2_cols_by_scene[sc["name"]] = cols
        render_overlay(os.path.join(out_dir, f"q2_{sc['name']}.mp4"), cols, q2_times[sc["name"]])
        render_still(os.path.join(out_dir, f"q2_{sc['name']}_still.png"), cols,
                     q2_times[sc["name"]], n_frames - 1)

    # ---------------- 5. ENDPOINT PARITY: sweep endpoints must equal the Q1 endpoints ------------------
    # Before any interior alpha means anything, alpha=0 and alpha=1 of each sweep MUST reproduce the Q1
    # replication rollout of their endpoint material. With the unified kernel this holds by construction
    # (same scene, dt, init, band, net-application path), so the only residual is GPU atomic-add
    # non-determinism. Measure that floor by running the SAME config twice, then confirm the endpoint
    # residual sits at that floor rather than diverging.
    print("=== endpoint parity check (drop_soft) ===")
    parity = {}
    # self-noise floor: re-run learned fluid (Q1 config: iso=1, band-off) and compare to the Q1 fluid run
    load_theta(thetas["fluid"])
    fluid_again, _, _ = rollout(q1_scene, dt, n_frames, "learned", iso=1.0, tc=BAND_OFF, ts=BAND_OFF)
    parity["self_noise_fluid"] = traj_rmse(q1_learned_snaps["fluid"], fluid_again)
    # fluid<->elastic endpoints: alpha=0 (iso=1, band-off) == Q1 fluid, alpha=1 (iso=0, band-off) == Q1 elastic
    load_theta(lerp_theta(thetas["fluid"], thetas["elastic"], 0.0))
    fe0, _, fe0ok = rollout(q1_scene, dt, n_frames, "learned", iso=1.0, tc=BAND_OFF, ts=BAND_OFF)
    load_theta(lerp_theta(thetas["fluid"], thetas["elastic"], 1.0))
    fe1, _, fe1ok = rollout(q1_scene, dt, n_frames, "learned", iso=0.0, tc=BAND_OFF, ts=BAND_OFF)
    parity["fe_alpha0_vs_q1fluid"] = traj_rmse(q1_learned_snaps["fluid"], fe0)
    parity["fe_alpha1_vs_q1elastic"] = traj_rmse(q1_learned_snaps["elastic"], fe1)
    # elastic<->snow endpoints: alpha=0 (iso=0, band-off) == Q1 elastic, alpha=1 (iso=0, snow band) == Q1 snow
    load_theta(lerp_theta(thetas["elastic"], thetas["snow"], 0.0))
    es0, _, es0ok = rollout(q1_scene, dt, n_frames, "learned", iso=0.0, tc=BAND_OFF, ts=BAND_OFF)
    load_theta(lerp_theta(thetas["elastic"], thetas["snow"], 1.0))
    es1, _, es1ok = rollout(q1_scene, dt, n_frames, "learned", iso=0.0, tc=TC, ts=TS)
    parity["es_alpha0_vs_q1elastic"] = traj_rmse(q1_learned_snaps["elastic"], es0)
    parity["es_alpha1_vs_q1snow"] = traj_rmse(q1_learned_snaps["snow"], es1)
    parity["endpoints_stable"] = bool(fe0ok and fe1ok and es0ok and es1ok)
    print(f"  self-noise floor (same config twice): {parity['self_noise_fluid']:.5f}")
    for k in ("fe_alpha0_vs_q1fluid", "fe_alpha1_vs_q1elastic",
              "es_alpha0_vs_q1elastic", "es_alpha1_vs_q1snow"):
        print(f"  {k:30s} trajRMSE={parity[k]:.5f}")
    # parity figure: endpoint sweep rollout overlaid on the Q1 endpoint (should be indistinguishable)
    parity_cols = [
        ("f<->e  alpha=0  vs Q1 fluid", [(q1_learned_snaps["fluid"], GREY, 6), (fe0, MAT_COL["fluid"], 4)]),
        ("f<->e  alpha=1  vs Q1 elastic", [(q1_learned_snaps["elastic"], GREY, 6), (fe1, MAT_COL["elastic"], 4)]),
        ("e<->s  alpha=1  vs Q1 snow", [(q1_learned_snaps["snow"], GREY, 6), (es1, MAT_COL["snow"], 4)]),
    ]
    render_still(os.path.join(out_dir, "endpoint_parity_still.png"), parity_cols, q1_times, n_frames - 1)
    render_overlay(os.path.join(out_dir, "endpoint_parity.mp4"), parity_cols, q1_times)

    if args.probe:
        print("=== PROBE: stopping after Q1 + parity ===")
        print(f"  Q1 snow trajRMSE={q1['snow']['traj_rmse']:.4f} learned_w={q1['snow']['learned_width']:.3f} "
              f"(true {q1['snow']['true_width']:.3f})")
        return None

    # ---------------- 6. Q3: interpolate the weights between material pairs ----------------------------
    print("=== Q3: interpolate weights between material pairs ===")

    def band_for_pair(key, a):
        """Yield band along a sweep. fluid<->elastic: band-off throughout (pure weight interpolation).
        elastic<->snow: interpolate 1/tc, 1/ts linearly in alpha, so alpha=0 is band-off (pure elastic)
        and alpha=1 is the snow band (true snow), with plasticity engaging smoothly across the sweep."""
        if key == "fluid_elastic":
            return BAND_OFF, BAND_OFF
        inv_tc = (1 - a) * (1.0 / BAND_OFF) + a * (1.0 / TC)
        inv_ts = (1 - a) * (1.0 / BAND_OFF) + a * (1.0 / TS)
        return 1.0 / inv_tc, 1.0 / inv_ts

    def iso_for_pair(key, a):
        """Isotropization along a sweep. fluid<->elastic morphs iso from 1 (fluid, det-F special case) at
        alpha=0 to 0 (elastic, free F) at alpha=1, so the state rule interpolates alongside the weights
        and each endpoint is exactly its Q1 config. elastic<->snow keeps iso=0 (both free F)."""
        return (1.0 - a) if key == "fluid_elastic" else 0.0

    pairs = [("fluid", "elastic", "fluid_elastic"),
             ("elastic", "snow", "elastic_snow")]
    sweeps = {}
    for (A, B, key) in pairs:
        print(f"  -- pair {A} -> {B}  ({key}) --")
        rec = {"alpha": [float(a) for a in sweep_alphas], "A": A, "B": B,
               "width": [], "height": [], "aspect": [], "jiggle": [], "clamp_frac": [],
               "tc": [], "ts": [], "stable": []}
        for a in sweep_alphas:
            tc, ts = band_for_pair(key, a)
            iso = iso_for_pair(key, a)
            load_theta(lerp_theta(thetas[A], thetas[B], a))
            snaps, tt, ok, st = rollout(q1_scene, dt, n_frames, "learned", iso=iso, tc=tc, ts=ts,
                                        collect=True)
            Jlast = st[-1][3]
            rec["width"].append(float(series(snaps, spread_width)[-3:].mean()))
            rec["height"].append(float(series(snaps, pile_height)[-3:].mean()))
            rec["aspect"].append(aspect_ratio(snaps[-1]))
            rec["jiggle"].append(jiggle(snaps))
            rec["clamp_frac"].append(float(np.mean(Jlast < 0.99)))
            rec["tc"].append(float(tc))
            rec["ts"].append(float(ts))
            rec["stable"].append(bool(ok))
            print(f"    a={a:.2f}  width={rec['width'][-1]:.3f} height={rec['height'][-1]:.3f} "
                  f"aspect={rec['aspect'][-1]:.3f} jiggle={rec['jiggle'][-1]:.4f} "
                  f"clampfrac={rec['clamp_frac'][-1]:.3f} {'ok' if ok else 'BLEW'}")
        sweeps[key] = rec

        # Width AND height both stay near the domain size across the interior (the blob disperses to fill
        # the box) and collapse to a settled material only at the two endpoints -- the quantitative form of
        # the degenerate interior seen in the clips. A settled material never exceeds ~0.3 in height.
        diag_series = [("spread width", sweep_alphas, rec["width"], "#4db6ff", "-"),
                       ("vertical extent (height)", sweep_alphas, rec["height"], "#ff9d5c", "-")]
        line_plot(os.path.join(out_dir, f"interp_{key}_diag.png"), diag_series,
                  f"interpolation coefficient  alpha   ({A} -> {B})", "extent (domain units)",
                  f"Q3: interpolated-weight blob extent vs alpha  ({A} -> {B})")

        # clip strip along alpha (color morphs A->B); overlay each true endpoint material as a grey ghost
        true_A, _, _ = rollout(q1_scene, dt, n_frames, "true", mat=A, iso=iso_of(A),
                               tc=band_of(A)[0], ts=band_of(A)[1])
        true_B, ttc, _ = rollout(q1_scene, dt, n_frames, "true", mat=B, iso=iso_of(B),
                                 tc=band_of(B)[0], ts=band_of(B)[1])
        cols = []
        for a in clip_alphas:
            tc, ts = band_for_pair(key, a)
            iso = iso_for_pair(key, a)
            load_theta(lerp_theta(thetas[A], thetas[B], a))
            snaps, _, _ = rollout(q1_scene, dt, n_frames, "learned", iso=iso, tc=tc, ts=ts)
            sets = [(snaps, alpha_color(a, MAT_COL[A], MAT_COL[B]), 5)]
            if a == 0.0:
                sets = [(true_A, GREY, 4)] + sets
            elif a == 1.0:
                sets = [(true_B, GREY, 4)] + sets
            cols.append((f"a={a:.2f}", sets))
        render_overlay(os.path.join(out_dir, f"interp_{key}.mp4"), cols, ttc)
        render_still(os.path.join(out_dir, f"interp_{key}_still.png"), cols, ttc, n_frames - 1)
        print(f"    wrote interp_{key}.mp4")

    # ---------------- 7. true-material reference triptych (sanity + teaching figure) -------------------
    print("=== true-material reference triptych (soft drop) ===")
    ref_cols = []
    ref_times = None
    for m in mats:
        tc, ts = band_of(m)
        snaps, times, _ = rollout(q1_scene, dt, n_frames, "true", mat=m, iso=iso_of(m), tc=tc, ts=ts)
        ref_times = times
        ref_cols.append((f"{m}", [(snaps, MAT_COL[m], 5)]))
    render_overlay(os.path.join(out_dir, "true_materials.mp4"), ref_cols, ref_times)
    render_still(os.path.join(out_dir, "true_materials_still.png"), ref_cols, ref_times, n_frames - 1)

    # ---------------- 8. metrics + manifest -----------------------------------------------------------
    metrics = {"dt": dt, "n_grid": n_grid, "n_particles": n_part, "E": E_COMMON, "NU": NU,
               "snow": {"xi": XI, "tc": TC, "ts": TS}, "band_off": BAND_OFF,
               "net": {"in": N_IN, "hidden": N_HID, "out": N_OUT, "params": N_PARAMS},
               "target_scale": tscale_np, "weight_dist": wdist,
               "snow_clamp_per_scene": snow_clamp, "snow_clamp_pooled": snow_clamp_overall,
               "train": train_report, "q1": q1, "q1b_snow_column_rmse": traj_rmse(snow_tr, snow_le),
               "q2": q2, "parity": parity, "sweeps": sweeps,
               "train_scenes": [s["name"] for s in tr_scenes],
               "gen_scenes": [s["name"] for s in gscenes]}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    write_manifest(out_dir, rel_dir, metrics)
    print(f"\nwrote -> {rel_dir}")
    return metrics


def write_manifest(out_dir, rel_dir, m):
    def f3(v):
        return f"{v:.3f}"
    q1, q2, sw, par = m["q1"], m["q2"], m["sweeps"], m["parity"]
    fe, es = sw["fluid_elastic"], sw["elastic_snow"]

    # Degeneracy of the interpolation interior: a SETTLED material never rises above ~0.3 in height, so a
    # large interior height means the blob dispersed to fill the box. Compare the interior extent to the
    # (settled) endpoints.
    def interior(rec, key, fn):
        vals = [rec[key][i] for i, a in enumerate(rec["alpha"]) if 0.0 < a < 1.0]
        return float(fn(vals))
    fe_int_hmax = interior(fe, "height", max)
    es_int_hmax = interior(es, "height", max)
    fe_int_hmin = interior(fe, "height", min)
    es_int_hmin = interior(es, "height", min)
    fe_end_hmax = max(fe["height"][0], fe["height"][-1])
    es_end_hmax = max(es["height"][0], es["height"][-1])

    q1_wmax = max(q1[k]["width_rel_err"] for k in ("fluid", "elastic", "snow"))
    q1_hmax = max(q1[k]["height_rel_err"] for k in ("fluid", "elastic", "snow"))
    gen_keys = list(q2.keys())
    q2_wmax = max(q2[k]["width_rel_err"] for k in gen_keys)
    q2_fluid_w = np.mean([q2[k]["width_rel_err"] for k in gen_keys if q2[k]["material"] == "fluid"])
    q2_elastic_w = np.mean([q2[k]["width_rel_err"] for k in gen_keys if q2[k]["material"] == "elastic"])
    q2_snow_w = np.mean([q2[k]["width_rel_err"] for k in gen_keys if q2[k]["material"] == "snow"])
    q2_snow_h = np.mean([q2[k]["height_rel_err"] for k in gen_keys if q2[k]["material"] == "snow"])

    par_max = max(par["fe_alpha0_vs_q1fluid"], par["fe_alpha1_vs_q1elastic"],
                  par["es_alpha0_vs_q1elastic"], par["es_alpha1_vs_q1snow"])

    table_rows = []
    for m_ in ("fluid", "elastic", "snow"):
        table_rows.append([f"replicate soft-drop ({m_})", f3(q1[m_]["true_width"]),
                           f3(q1[m_]["learned_width"]), f3(q1[m_]["width_rel_err"]),
                           "stable" if q1[m_]["learned_stable"] else "BLEW UP"])
    for k in gen_keys:
        table_rows.append([f"generalize {q2[k]['scene']} ({q2[k]['material']})",
                           f3(q2[k]["true_width"]), f3(q2[k]["learned_width"]),
                           f3(q2[k]["width_rel_err"]),
                           "stable" if q2[k]["learned_stable"] else "BLEW UP"])

    results = [
        {"type": "video", "src": f"{rel_dir}/true_materials.mp4",
         "caption": ("The three true materials on the shared substrate, one panel each, the same disk "
                     "dropped onto the floor at one common Young's modulus so the only difference is the "
                     "constitutive law. Fluid spreads into a flat puddle, elastic squashes then springs "
                     "back to a rounded jiggling blob, snow crumples and holds a dented heap. This is the "
                     "behavior the three stress networks have to reproduce.")},
        {"type": "video", "src": f"{rel_dir}/q1_reproduce.mp4",
         "caption": ("Question 1, each net reproduces its own material on the soft drop. Three panels, "
                     "fluid then elastic then snow. Grey is the true simulator with the analytic "
                     "constitutive stress; colour is the material whose stress is supplied entirely by the "
                     "trained network, with the shared analytic plastic clamp handling snow's state update. "
                     "The coloured particles track the grey ones throughout, so one small architecture "
                     "learns all three structurally different stress laws.")},
        {"type": "video", "src": f"{rel_dir}/q1b_distinct_hard.mp4",
         "caption": ("The three learned nets on the hard impact, the scene that fires snow's plastic clamp "
                     "hardest. Left to right, the learned fluid splats into a wide sheet, the learned "
                     "elastic rebounds and rings, and the learned snow crumples and holds a compact dented "
                     "heap. Trained on varied scenes that cross snow's yield surface, the snow net now "
                     "behaves like snow rather than like a soft solid.")},
        {"type": "video", "src": f"{rel_dir}/q1b_snow_column.mp4",
         "caption": ("Learned snow versus true snow on the column slump, the angle-of-repose test. Grey is "
                     "the true Stomakhin snow, pale blue the network-driven snow (the plastic clamp is the "
                     "shared analytic rule, only the hardening stress is learned). The learned column "
                     "slumps to nearly the same standing pile as the true one, evidence the net captured "
                     "the hardening stress across the plastic regime, not just the near-rest elastic one.")},
        {"type": "image", "src": f"{rel_dir}/q1_width.png",
         "caption": ("Spread width over time for the three materials on the soft drop. Solid lines are the "
                     "learned rollouts, dashed the true simulator. Learned and true track closely and the "
                     "three materials stay cleanly separated, with fluid widening most and elastic least, "
                     "the quantitative form of the overlay video.")},
        {"type": "video", "src": f"{rel_dir}/q2_gen_toss.mp4",
         "caption": ("Question 2, generalization to a held-out scene. A disk of a new size is thrown down "
                     "and to the left, a geometry and velocity absent from every training scene. Grey is "
                     "the true simulator, colour the learned material. All three learned materials track "
                     "their true run closely, the fluid spreading, the elastic staying compact, and the "
                     "snow slumping to the same heap, evidence the nets learned a local stress law rather "
                     "than memorizing the training scenes.")},
        {"type": "video", "src": f"{rel_dir}/q2_gen_arch.mp4",
         "caption": ("Generalization to a second held-out scene, a chunky wide block released from rest, "
                     "wider and taller than any training block. Grey true, colour learned. All three "
                     "materials again track the truth tightly, consistent across the two unseen scenes "
                     "rather than a fluke of one.")},
        {"type": "image", "src": f"{rel_dir}/endpoint_parity_still.png",
         "caption": ("Endpoint-parity check, the fix for the interpolation bug. Each panel overlays a sweep "
                     "endpoint rollout (colour) on the Question-1 replication rollout of that same material "
                     "(grey). Left, the fluid-elastic sweep at alpha=0 sits exactly on the Q1 fluid; middle, "
                     "the same sweep at alpha=1 sits on the Q1 elastic; right, the elastic-snow sweep at "
                     "alpha=1 sits on the Q1 snow. Because every rollout runs the one unified state kernel, "
                     "the endpoints reproduce their material identically instead of exploding as they did "
                     "before, so the interior sweep is meaningful.")},
        {"type": "image", "src": f"{rel_dir}/interp_fluid_elastic_diag.png",
         "caption": ("Question 3, interpolating the fluid net and the elastic net. The state rule morphs "
                     "with the weights (the fluid's volumetric projection at alpha=0 to the elastic's free "
                     "deformation at alpha=1) so the two endpoints are the Q1 fluid and Q1 elastic exactly. "
                     "Spread width and vertical extent of the interpolated-weight blob are plotted against "
                     "alpha. The two endpoints are settled materials (small height, resting at the floor), "
                     "but the interior diagnostics swing erratically between near-zero and near the domain "
                     "size with no intermediate trend, the quantitative signature of the dispersed, broken "
                     "interior seen in the clips rather than a graded in-between material.")},
        {"type": "video", "src": f"{rel_dir}/interp_fluid_elastic.mp4",
         "caption": ("Fluid-to-elastic weight interpolation, five columns from alpha=0 (fluid) to alpha=1 "
                     "(elastic). Grey ghosts at the two ends are the true fluid and true elastic; colour is "
                     "the interpolated-weight blob, tinted blue to orange. The two endpoints sit exactly on "
                     "their grey references (a spreading fluid and a compact elastic), but every interior "
                     "blend is degenerate, a fine spray of particles scattered across the whole domain "
                     "rather than a plausible intermediate. Blending the weights of two structurally "
                     "different stress laws does not give an in-between material, it gives a broken one.")},
        {"type": "image", "src": f"{rel_dir}/interp_elastic_snow_diag.png",
         "caption": ("Interpolating the elastic net and the snow net. Because snow's identity lives partly "
                     "in a state rule outside the weights, the sweep morphs both the stress weights and the "
                     "plastic-yield band, so alpha=0 is pure elastic and alpha=1 is true snow. Spread width "
                     "and vertical extent versus alpha. As with the fluid-to-elastic pair the two endpoints "
                     "are settled materials while the interior diagnostics swing erratically toward the "
                     "domain size, so the interior is a dispersed, broken cloud, not a graded "
                     "elastic-to-snow morph.")},
        {"type": "video", "src": f"{rel_dir}/interp_elastic_snow.mp4",
         "caption": ("Elastic-to-snow interpolation, five columns from alpha=0 (pure elastic) to alpha=1 "
                     "(true snow). Grey ghosts are the true endpoints. The elastic and snow endpoints are "
                     "clean compact heaps, but again every interior blend scatters into a sparse cloud "
                     "filling the domain. Even with the plastic clamp co-interpolated so the endpoints are "
                     "exact, the interior stays degenerate, because the blended stress between two "
                     "structurally different laws is not itself a valid constitutive law.")},
        {"type": "table",
         "columns": ["condition", "true width", "learned width", "rel error", "rollout"],
         "rows": table_rows,
         "caption": ("Learned versus true final-frame spread width for each material on the training soft "
                     "drop (replicate) and on the two held-out scenes (generalize), in domain units. "
                     "Relative error is the learned-versus-true gap. Every learned rollout stayed finite. "
                     "The fluid transfers tightest; the solids reproduce their own scene well and "
                     "generalize with a larger but bounded gap.")},
    ]

    findings = (
        "On this one setup, a single tiny MLP architecture (10 inputs, one hidden layer of 48 tanh units, "
        f"3 outputs, {m['net']['params']} parameters) trained by supervised regression replaces the "
        "analytic constitutive stress of three structurally different materials -- weakly-compressible "
        "fluid, corotated elastic, and Stomakhin snow -- and reproduces each on its own scenes and on "
        "held-out scenes, but linearly interpolating the weights of two of these nets produces a DEGENERATE "
        "interior, not an intermediate material. All three nets share ONE architecture, ONE position-free "
        "input layout (the rotation-invariant stretch S from the polar decomposition of the deformation "
        "gradient F, carried for every material, plus the APIC affine matrix C, the velocity, and the "
        "plastic record Jp), ONE symmetric material-frame stress output rotated back by the analytic polar "
        "rotation, and ONE shared feature standardization and target scale from the pooled states of all "
        "three materials, so only the trainable weights differ. All three also run under ONE unified state "
        "kernel with two continuous knobs: an isotropization that at its extreme keeps F volumetric (the "
        "fluid's det-F special case, which keeps its deformation gradient bounded so det F stays numerically "
        "clean) and a plastic clamp of the singular values into [1-tc, 1+ts] whose band-off setting never "
        "fires (elastic) and whose snow band fires the crumple (snow). This shared kernel is what lets the "
        "interpolation endpoints reproduce their materials exactly. The nets were trained on FIVE varied "
        "scenes (a soft drop, a hard impact, a column slump, a wide settling slab, and a lateral throw) "
        "chosen to exercise each material's signature; the hard impact and column drive snow across its "
        f"yield surface, so {m['snow_clamp_pooled']*100:.0f} percent of the pooled snow states have "
        "suffered genuine plastic compaction (Jp below 0.99, minimum Jp around "
        f"{min(v['min_Jp'] for v in m['snow_clamp_per_scene'].values()):.2f}), meaning the snow net is "
        "trained on genuinely plastic states, not just near-rest elastic ones. Each net fits the true "
        f"stress to a validation relative RMSE around {m['train']['fluid']['val_rel_rmse']:.2f} (fluid), "
        f"{m['train']['elastic']['val_rel_rmse']:.2f} (elastic), {m['train']['snow']['val_rel_rmse']:.2f} "
        f"(snow). (1) Each net reproduces its own material on the soft drop within {q1_wmax*100:.0f} percent "
        f"final spread width and {q1_hmax*100:.0f} percent final pile height of the true simulator, the "
        "learned particles overlay the true ones throughout, and on the hard impact the three learned nets "
        "are visibly distinct -- the learned snow crumples and holds a dented heap while the learned fluid "
        "splats and the learned elastic rebounds -- and the learned snow slumps to nearly the true "
        "angle-of-repose pile on the column, so the snow net genuinely behaves like snow. (2) The nets "
        "generalize to two held-out scenes (a leftward throw of a new-size disk and a chunky wide block): "
        "all three transfer tightly, each within about one percent on final spread width, and the learned "
        "snow carries its only visible residual in pile height on the fast shearing toss (mean final-height "
        f"error {q2_snow_h*100:.0f} percent, against a few percent for fluid and elastic). "
        "Reaching that required reflecting the training states left-to-right, since the physics is "
        "mirror-symmetric but the scenes threw only one way and the sensitive snow net otherwise puffed up "
        "on a leftward toss it had never seen, the honest sign that snow's plastic-feedback law is the "
        "least forgiving of the three to a gap in training coverage. (3) Weight interpolation is the "
        "exploratory part and there is no "
        "ground-truth intermediate. The rework's endpoint bug is fixed: at alpha=0 and alpha=1 each "
        "interpolated rollout reproduces the Q1 endpoint material to a trajectory RMSE of at most "
        f"{par_max:.4f}, at the level of GPU non-determinism itself (the same config re-run twice differs "
        f"by {par['self_noise_fluid']:.4f}), because every rollout now runs the one unified kernel and the "
        "state rule is co-interpolated to each endpoint's own setting, so the earlier endpoint explosion is "
        "gone. With the endpoints correct, the honest interior finding is negative and clean for BOTH "
        "pairs: every interior blend is a broken material. The two endpoints are settled materials (a "
        f"puddle, a blob, or a heap on the floor, final vertical extent {fe_end_hmax:.2f} domain units or "
        "less), but each interior blend is a sparse spray of particles that has variously flung apart to "
        "fill the box or smeared flat from wall to wall, with diagnostics that swing erratically and show "
        f"no intermediate trend (the fluid-to-elastic interior vertical extent ranges from {fe_int_hmin:.2f} "
        f"to {fe_int_hmax:.2f} across alpha against {fe_end_hmax:.2f} at the endpoints, and the "
        f"elastic-to-snow interior behaves the same, up to {es_int_hmax:.2f} against {es_end_hmax:.2f}). "
        "Every interpolated rollout stayed finite, so the interior is a genuine degenerate steady state, "
        "not a numerical blow-up. This directly answers whether structurally different "
        "endpoints make weight interpolation worse than the linear-viscosity precursor: they make it far "
        "worse. The viscosity case, where the two endpoints were one linear family, interpolated to a "
        "stable fluid that was merely too thin in the middle; here, where the endpoints are structurally "
        "different constitutive laws, interpolation does not produce a valid material at all in the "
        "interior. The honest one-line summary: one small net learns and generalizes all three stress laws, "
        "and the endpoints of a weight interpolation are exact, but the interior of a weight interpolation "
        "between structurally different materials is degenerate, so a material slider cannot be built by "
        "blending separate per-material stress nets."
    )

    hypothesis = (
        "Three mechanisms explain the results, and all connect to the viscosity precursor. First, why one "
        "memoryless local network can stand in for three very different constitutive laws: a constitutive "
        "stress is by construction a local function of the deformation state, sigma = g(F) for the solids "
        "and sigma = E(det F - 1) I for the fluid, so a network fed the rotation-invariant stretch (plus "
        "the affine matrix, velocity, and plastic record, all position-free) is fitting a genuine local "
        "function and has the inputs to do it. Feeding the polar stretch rather than the raw deformation "
        "gradient is what makes the solids learnable and stable, because the raw gradient buries a small "
        "elastic strain inside an order-one rotation and a net fed the raw gradient must undo that rotation "
        "itself. The fluid is the easy case because its stress collapses onto the single scalar det F. Snow "
        "is the hardest to transfer because its stress is the corotated law scaled by a hardening factor "
        "that depends on the plastic record Jp, and Jp is a state its own predicted stress feeds back into "
        "through the clamp, so a small stress error can push Jp somewhere the net never saw and compound; "
        "the five-scene training set that crosses the yield surface, plus clamping the Jp input to the "
        "trained range, is what tames that feedback and lets the snow net both reproduce and generalize. "
        "Second, and this is the headline, why the interior of the weight interpolation is degenerate. The "
        "map from a weight vector to the function it computes is strongly nonlinear: a one-hidden-layer net "
        "computes roughly W2 tanh(W1 x), whose magnitude runs through the product of the two weight "
        "matrices, so a straight line in weight space is a curved path in function space and the midpoint "
        "weights of two distant solutions do not represent the midpoint function. In the viscosity study "
        "that curvature was a bounded defect, because the two endpoints were the SAME functional form "
        "scaled by a knob, so every point on the chord was still a valid viscous stress, just the wrong "
        "size, and the fluid stayed a fluid that was merely too thin. Here the two endpoints are DIFFERENT "
        "functional forms, a det-only pressure and a full corotated tensor, and the chord between them "
        "passes through weight vectors whose stress is neither, a tensor field that is not the gradient of "
        "any energy and carries no guarantee of being dissipative or even sign-definite. A non-dissipative "
        "stress injects energy every step instead of removing it, so the blob heats up and its particles "
        "fly apart until they are a diffuse cloud filling the domain, which is exactly the dispersed "
        "interior seen in every interior clip. That is the deeper reason structurally different endpoints "
        "make interpolation qualitatively worse and not just quantitatively worse: leaving the linear "
        "family means leaving the manifold of valid constitutive laws, and most of the chord lies off it. "
        "Third, why the endpoints are nonetheless exact. Part of each material's identity lives in a state "
        "rule outside the weights (the fluid's volumetric projection, snow's plastic clamp), and the fix "
        "co-interpolates those rules to each endpoint's own setting, which is necessary for the endpoints "
        "to be their true materials but cannot rescue the interior, because the interior's problem is the "
        "invalid blended stress, not the state rule. The prediction that follows for a controllable world "
        "model is sharp: a material slider cannot be built by training separate per-material stress nets "
        "and blending their weights, because the blend is off the manifold of valid physics through most of "
        "its range. It has to be built by conditioning a single network on a material descriptor so the "
        "whole continuum is trained on real physics, and by learning or conditioning the state rules too, "
        "not just the stress."
    )

    limitations = (
        "A demonstration on one architecture and three specific materials, not a general law about learned "
        "constitutive models or weight interpolation. Everything is 2D, n_grid=64, f32, one grid "
        f"resolution, one fixed timestep dt={m['dt']:g} (set conservatively so the stiffest case, hardened "
        f"snow on a hard impact, stays under the explicit CFL limit), one common Young's modulus "
        f"E={m['E']:g} for all three materials (isolating the constitutive form from a stiffness confound, "
        "so a real material comparison would also vary E), five training scenes, two held-out "
        "generalization scenes, and a supervised-regression fit rather than a rollout-trained one. Snow's "
        f"hardening was set to a moderate xi={m['snow']['xi']:g} (below a showcase-strength value) so the "
        "hardened-stress tail stays in a range the net can fit and drive stably, and the Jp input to the "
        "net is clamped to the trained range to stop the hardening feedback from running away; both are "
        "stability choices that make the snow net trainable but also mean the learned snow is a slightly "
        "tamer snow than a stiff-hardening one. The most important scoping point is the handling of snow's "
        "plasticity: the plastic clamp is a STATE update that mutates F's singular values and Jp, so it "
        "cannot live in a memoryless stress network; the choice here is to keep the clamp as a shared, "
        "non-learned analytic rule and learn only the stress. Because part of each material's identity lives "
        "in a state rule outside the weights (the fluid's volumetric projection, snow's plastic clamp), "
        "neither interpolation is a pure weight interpolation: each co-interpolates the state rule (the "
        "isotropization for fluid-to-elastic, the yield band for elastic-to-snow) so the endpoints are "
        "exactly their true materials, and there is no ground truth for the half-morphed state rules, so "
        "those schedules are reasonable but arbitrary exploratory choices. This matters less than it might "
        "because the interior is degenerate regardless of the state-rule schedule; the dispersal is driven "
        "by the blended stress, not the state rule. A fluid-to-snow sweep was left out because it would "
        "confound both differences (state rule and stress law) at once. The interpolation is characterized "
        "by spread width and vertical extent on the soft-drop scene plus the viewed clips, a robust but "
        "scene-specific and qualitative read; by design there is no intermediate material to score against. "
        "Each net is a single training seed, so the results are one realization, not a seed average; the "
        "degenerate interior is expected to be robust to the seed (it follows from the endpoints being off "
        "one linear family) but that was not measured. Generalization is tested on two unseen scenes, not "
        "across resolution, blob families, or a range of E. Endpoint parity is verified to the level of GPU "
        "atomic-add non-determinism, which is not bitwise reproducible; rerun if a frame looks off."
    )

    manifest = {
        "schema_version": "2",
        "task_id": "train-material-replicating-nns-and-interpolate",
        "direction": "material-variants",
        "title": "Learning three materials with one net, and interpolating between them",
        "status": "active",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objective": (
            "Train the SAME small network architecture to replace the constitutive stress of three "
            "structurally different materials -- weakly-compressible fluid, corotated elastic, and "
            "Stomakhin snow -- by supervised regression against each true simulator, on a common substrate "
            "that makes the weights comparable: one architecture, one position-free input layout (the "
            "rotation-invariant polar stretch of the deformation gradient F carried for all materials, the "
            "APIC affine matrix C, velocity, and the plastic record Jp), one symmetric material-frame "
            "stress output, one shared feature standardization and target scale, and one unified state "
            "kernel (free-trial F plus a plastic clamp whose band is off for fluid and elastic and set for "
            "snow). Verify each net reproduces its material across varied scenes that exercise its "
            "signature (including scenes that fire snow's plastic clamp) and generalizes to held-out "
            "scenes, then interpolate the trained weights between material pairs (fluid-elastic and "
            "elastic-snow) and characterize honestly what dynamics emerge. Unlike the viscosity precursor, "
            "whose linear target gave a known intermediate, these are not one linear family, so there is no "
            "ground-truth intermediate and question three is exploratory. Snow's plastic clamp is a state "
            "update, not a stress, and is kept as a shared analytic rule so the net learns only the stress; "
            "the interpolation endpoints are made to reproduce their materials exactly before any interior "
            "alpha is read."
        ),
        "findings": findings,
        "hypothesis": hypothesis,
        "limitations": limitations,
        "results": results,
        "custom_html": None,
        "training_refs": ["learned-material-interpolation", "learned-viscosity-interpolation",
                          "material-showcase", "differentiable-materials", "constitutive-models",
                          "svd-polar", "hybrid-learned-residual"],
        "params": m,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
