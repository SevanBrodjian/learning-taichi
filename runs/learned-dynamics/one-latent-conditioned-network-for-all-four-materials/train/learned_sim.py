"""The learned MLS-MPM simulator: canonical scaffolding, ONE latent-conditioned net at the seam.

WHAT IS CANONICAL AND UNTOUCHED
-------------------------------
`sim.physics.core` owns ti.init, the grid fields, the frozen MAT table and the grid update. This
module imports it and uses:
    core.clear_grid, core.grid_op            (verbatim -- the grid update is NOT the seam)
    core.x / v / C / F / Jp / mat_id / p_vol_f / p_mass_f / grid_v / grid_m / grid_fr
    core.MAT, core.MAT_ID, core.dp_alpha     (parameters, never retyped)
It re-implements only P2G's scatter and G2P's gather, because the seam sits between them.

THE DECLARED DEVIATIONS FROM THE CANONICAL *STEP* (the constitutive LAW and every PARAMETER are
canonical; CLAUDE.md requires a variant to say exactly what differs):

1. STRESS IS CACHED, NOT RECOMPUTED. Canonical P2G computes the stress from F at the top of every
   substep. Here the constitutive evaluation happens at the BOTTOM of G2P and its stress is stored in
   `stress_f` for the next P2G to scatter. Same quantity, one kernel earlier -- the stress P2G needs
   at step n is a function of the F that G2P produced at step n-1 and of nothing else. This makes the
   whole seam ONE network evaluation per substep instead of two.

2. F IS CARRIED AS (R, S) IMPLICITLY. F' is remounted as R S' with R the polar rotation of the trial
   F and S' the corrected symmetric stretch, i.e. F' = U diag(s') V^T. Canonical stores U diag(s') V
   (see the note in the WebGPU port: ti.svd returns V, and canonical reconstructs with V, not V^T).
   The two differ by a rotation applied on the RIGHT of F. For an ISOTROPIC constitutive model that
   is unobservable: right-multiplying F by an orthogonal Q leaves the singular values, the left polar
   factor and therefore every stress in this library exactly invariant, and the difference propagates
   as a right rotation forever without ever reaching a position. `oracle` mode exists to prove that
   numerically rather than argue it.

3. THE FLUID CARRIES ITS VOLUME RATIO IN F, NOT IN A SEPARATE SCALAR J. Canonical's fluid ignores F
   and advects J by J *= 1 + dt tr(C). Here every material has an F and the fluid's constitutive
   update ISOTROPISES it, S' = sqrt(J_new) I, with J_new set to exactly canonical's value
   det(S_tr) * (1 + dt tr C) / det(I + dt C). That keeps the oracle bit-comparable with canonical
   while giving the network ONE state representation to learn for all four materials.

Everything else -- B-spline weights, the affine APIC transfer, gravity, the separating walls, the
Coulomb friction cap, the per-material density and volume -- is canonical.

MODES
-----
    'oracle'  the analytic canonical constitutive law, in the reparameterised frame. Its job is to
              prove that (1)-(3) change nothing, and to LABEL training data.
    'nn'      the latent-conditioned MLP.
"""
# NOTE: no `from __future__ import annotations` -- Taichi introspects real annotation objects on
# @ti.kernel signatures and stringised annotations break it (same note as sim/physics/core.py).
import pathlib
import sys

import numpy as np
import taichi as ti

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # netspec lives beside this file
sys.path.insert(0, str(_HERE.parents[3]))            # repo root, for sim.physics

import sim.physics as phys                           # noqa: E402
from sim.physics import core                         # noqa: E402

import netspec as NS                                 # noqa: E402

dim = 2
MAX_P = core.MAX_P
H_MAX = 256                      # scratch width; the trained width is a runtime value <= this
N_IN = NS.N_IN
N_OUT = NS.N_OUT
Z_DIM = NS.Z_DIM

FLUID, ELASTIC, SNOW, SAND = core.FLUID, core.ELASTIC, core.SNOW, core.SAND

# ---------------------------------------------------------------------------- extra state
stress_f = ti.Matrix.field(dim, dim, float, MAX_P)     # cached world-frame stress (P F^T / sigma)
z_of_mat = ti.Vector.field(Z_DIM, float, 4)            # the four material codes

W1 = ti.field(float, (H_MAX, N_IN))
b1 = ti.field(float, H_MAX)
W2h = ti.field(float, (H_MAX, H_MAX))                  # optional second hidden layer
b2h = ti.field(float, H_MAX)
W3 = ti.field(float, (N_OUT, H_MAX))
b3 = ti.field(float, N_OUT)
hid1 = ti.field(float, (MAX_P, H_MAX))
hid2 = ti.field(float, (MAX_P, H_MAX))

# capture pool for DAgger: the states the LEARNED rollout actually visits
CAP = 400000
cap_S = ti.Vector.field(3, float, CAP)                 # S_tr: S00, S01, S11
cap_C = ti.Vector.field(4, float, CAP)
cap_v = ti.Vector.field(2, float, CAP)
cap_Jp = ti.field(float, CAP)
cap_m = ti.field(ti.i32, CAP)
cap_n = ti.field(ti.i32, ())

# labelling buffers (oracle applied to an arbitrary batch of trial states)
lab_S = ti.Vector.field(3, float, CAP)
lab_Jp = ti.field(float, CAP)
lab_m = ti.field(ti.i32, CAP)
lab_out = ti.Vector.field(N_OUT, float, CAP)


# ---------------------------------------------------------------------------- small 2x2 helpers
@ti.func
def sym_eig2(s00, s01, s11):
    """Eigen-decomposition of the symmetric 2x2 [[s00,s01],[s01,s11]], descending eigenvalues.
    Returns (l0, l1, q00, q10) where the first eigenvector is (q00, q10) and the second is
    (-q10, q00), i.e. Q = [[q00, -q10], [q10, q00]] is a rotation with S = Q diag(l) Q^T."""
    tr = s00 + s11
    df = s00 - s11
    r = ti.sqrt(df * df + 4.0 * s01 * s01)
    l0 = 0.5 * (tr + r)
    l1 = 0.5 * (tr - r)
    # eigenvector for l0: (s01, l0 - s00) or (l0 - s11, s01), whichever is better conditioned
    ex, ey = 0.0, 0.0
    if ti.abs(s01) > 1e-12:
        ex, ey = l0 - s11, s01
    else:
        ex, ey = 1.0, 0.0
    nrm = ti.sqrt(ex * ex + ey * ey) + 1e-30
    return l0, l1, ex / nrm, ey / nrm


@ti.func
def polar2(Fm):
    """F = R S with R orthogonal (handles det F < 0) and S symmetric. Same closed form as the WGSL
    `polar_r` and Taichi's `_polar_decompose2d`, so host and shader agree to f32 rounding."""
    a00, a01, a10, a11 = Fm[0, 0], Fm[0, 1], Fm[1, 0], Fm[1, 1]
    R = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
    if not (a00 == 0.0 and a01 == 0.0 and a10 == 0.0 and a11 == 0.0):
        detA = a00 * a11 - a10 * a01
        b = ti.Matrix([[a00 + a11, a01 - a10], [a10 - a01, a11 + a00]])
        if detA < 0.0:
            b = ti.Matrix([[a00 - a11, a01 + a10], [a10 + a01, a11 - a00]])
        adetB = ti.abs(b[0, 0] * b[1, 1] - b[1, 0] * b[0, 1])
        R = b * (1.0 / ti.max(ti.sqrt(adetB), 1e-30))
    S = R.transpose() @ Fm
    S = 0.5 * (S + S.transpose())
    return R, S


# ---------------------------------------------------------------------------- the analytic teacher
@ti.func
def oracle_step(m, S00, S01, S11, trC, detIdtC, Jpold):
    """THE CANONICAL CONSTITUTIVE LAW, in the material frame.

    In: the trial symmetric stretch S_tr (S00,S01,S11), the plastic record Jp, and -- for the fluid
    only -- tr(C) and det(I + dt C), which reproduce canonical's linearised J advection exactly.
    Out: (tau00, tau01, tau11, dS00, dS01, dS11, dJp) -- the material-frame stress the NEXT P2G
    scatters, the correction that turns the trial stretch into the plastic one, and the change in Jp.

    Every parameter comes from sim.physics (core.m_*), never from a literal here.
    """
    l0, l1, q0, q1 = sym_eig2(S00, S01, S11)
    n0, n1 = l0, l1                                        # plastic-corrected principal stretches
    dJp = 0.0
    t0, t1 = 0.0, 0.0                                      # principal stress in the eigenframe
    iso = 0.0                                              # isotropic add-on (fluid / la term)

    if m == FLUID:
        Jtr = l0 * l1
        Jnew = Jtr * (1.0 + trC) / ti.max(detIdtC, 1e-12)
        s = ti.sqrt(ti.max(Jnew, 1e-8))
        n0, n1 = s, s
        iso = core.m_E[FLUID] * (Jnew - 1.0)               # weakly-compressible pressure E (J - 1)
    else:
        if m == SNOW:
            c0 = ti.min(ti.max(l0, 1.0 - core.m_tc[SNOW]), 1.0 + core.m_ts[SNOW])
            c1 = ti.min(ti.max(l1, 1.0 - core.m_tc[SNOW]), 1.0 + core.m_ts[SNOW])
            dJp = Jpold * (l0 * l1) / (c0 * c1) - Jpold
            n0, n1 = c0, c1
        elif m == SAND:
            e0 = ti.log(ti.max(ti.abs(l0), 1e-4))
            e1 = ti.log(ti.max(ti.abs(l1), 1e-4))
            trE = e0 + e1 + Jpold
            mu = core.m_E[SAND] / (2.0 * (1.0 + core.m_nu[SAND]))
            la = core.m_E[SAND] * core.m_nu[SAND] / \
                ((1.0 + core.m_nu[SAND]) * (1.0 - 2.0 * core.m_nu[SAND]))
            if trE >= 0.0:
                dJp = trE - Jpold
                n0, n1 = 1.0, 1.0                          # cone tip: all stress released
            else:
                dJp = -Jpold
                eh0 = e0 - trE * 0.5
                eh1 = e1 - trE * 0.5
                ehn = ti.sqrt(eh0 * eh0 + eh1 * eh1) + 1e-20
                dg = ehn + (2.0 * la + 2.0 * mu) / (2.0 * mu) * trE * core.m_alpha[SAND]
                if dg <= 0.0:
                    n0, n1 = l0, l1
                else:
                    n0 = ti.exp(e0 - dg / ehn * eh0)
                    n1 = ti.exp(e1 - dg / ehn * eh1)

        # --- stress at the PLASTIC-CORRECTED state, exactly as the next canonical P2G would ---
        if m == SAND:
            mu = core.m_E[SAND] / (2.0 * (1.0 + core.m_nu[SAND]))
            la = core.m_E[SAND] * core.m_nu[SAND] / \
                ((1.0 + core.m_nu[SAND]) * (1.0 - 2.0 * core.m_nu[SAND]))
            g0 = ti.log(ti.max(n0, 1e-4))
            g1 = ti.log(ti.max(n1, 1e-4))
            tre = g0 + g1
            t0 = 2.0 * mu * g0 + la * tre                  # Hencky Kirchhoff stress
            t1 = 2.0 * mu * g1 + la * tre
        else:
            mid = ELASTIC if m == ELASTIC else SNOW
            mu = core.m_E[mid] / (2.0 * (1.0 + core.m_nu[mid]))
            la = core.m_E[mid] * core.m_nu[mid] / \
                ((1.0 + core.m_nu[mid]) * (1.0 - 2.0 * core.m_nu[mid]))
            if m == SNOW:
                h = ti.exp(core.m_xi[SNOW] * (1.0 - (Jpold + dJp)))
                mu = mu * h
                la = la * h
            Jd = n0 * n1
            # corotated P F^T in the material frame: 2 mu (S - I) S + la (J-1) J I
            t0 = 2.0 * mu * (n0 - 1.0) * n0
            t1 = 2.0 * mu * (n1 - 1.0) * n1
            iso = la * (Jd - 1.0) * Jd

    # rebuild the symmetric tensors in the ORIGINAL material frame (eigenvectors of S_tr)
    aa, bb = q0 * q0, q1 * q1
    ab = q0 * q1
    tau00 = t0 * aa + t1 * bb + iso
    tau01 = (t0 - t1) * ab
    tau11 = t0 * bb + t1 * aa + iso
    d0, d1 = n0 - l0, n1 - l1
    dS00 = d0 * aa + d1 * bb
    dS01 = (d0 - d1) * ab
    dS11 = d0 * bb + d1 * aa
    return ti.Vector([tau00, tau01, tau11, dS00, dS01, dS11, dJp])


@ti.kernel
def label_batch(n: ti.i32, dt: ti.f32):
    """Apply the analytic teacher to `n` arbitrary trial states in lab_*, writing lab_out.
    trC / det(I + dt C) are not available for a free-floating state, so the fluid's tiny linearisation
    correction is taken at its identity value (1.0), which differs from the exact factor by O(dt^2)."""
    for i in range(n):
        s = lab_S[i]
        lab_out[i] = oracle_step(lab_m[i], s[0], s[1], s[2], 0.0, 1.0, lab_Jp[i])


# ---------------------------------------------------------------------------- the network
@ti.func
def mlp_eval(p, S00, S01, S11, Cm, vv, Jpv, zc, H: ti.template(), L2: ti.template()):
    for k in range(H):
        a = (W1[k, 0] * S00 + W1[k, 1] * S01 + W1[k, 2] * S11
             + W1[k, 3] * Cm[0, 0] + W1[k, 4] * Cm[0, 1] + W1[k, 5] * Cm[1, 0] + W1[k, 6] * Cm[1, 1]
             + W1[k, 7] * vv[0] + W1[k, 8] * vv[1] + W1[k, 9] * Jpv + b1[k])
        for d in ti.static(range(Z_DIM)):
            a += W1[k, 10 + d] * zc[d]
        hid1[p, k] = ti.tanh(a)
    if ti.static(L2):
        for k in range(H):
            a = b2h[k]
            for j in range(H):
                a += W2h[k, j] * hid1[p, j]
            hid2[p, k] = ti.tanh(a)
        for k in range(H):
            hid1[p, k] = hid2[p, k]
    o = ti.Vector.zero(float, N_OUT)
    for j in ti.static(range(N_OUT)):
        a = b3[j]
        for k in range(H):
            a += W3[j, k] * hid1[p, k]
        o[j] = a
    return o


# ---------------------------------------------------------------------------- MPM kernels
@ti.kernel
def p2g_learned(n: ti.i32, dt: ti.f32):
    """Canonical P2G with ONE change: the stress comes out of `stress_f` instead of being
    recomputed. The -dt * 4 * p_vol * inv_dx^2 prefactor is applied here, so `stress_f` holds the
    raw material stress and nothing about the timestep leaks into the cache."""
    for p in range(n):
        Xp = core.x[p] * core.inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        p_vol = core.p_vol_f[p]
        p_mass = core.p_mass_f[p]
        kk = -dt * 4.0 * p_vol * core.inv_dx * core.inv_dx
        affine = kk * stress_f[p] + p_mass * core.C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * core.dx
            weight = w[i].x * w[j].y
            core.grid_v[base[0] + i, base[1] + j] += weight * (p_mass * core.v[p] + affine @ dpos)
            core.grid_m[base[0] + i, base[1] + j] += weight * p_mass
            core.grid_fr[base[0] + i, base[1] + j] += weight * p_mass * core.m_fric[core.mat_id[p]]


@ti.kernel
def g2p_learned(n: ti.i32, dt: ti.f32, use_nn: ti.template(), H: ti.template(),
                L2: ti.template(), capture: ti.i32, cap_stride: ti.i32):
    for p in range(n):
        new_v, new_C = core.g2p_gather(p)
        core.v[p] = new_v
        core.x[p] = core.x[p] + dt * new_v
        core.x[p] = ti.math.clamp(core.x[p], core.floor_y, 1.0 - core.floor_y)
        core.C[p] = new_C

        G = ti.Matrix.identity(float, dim) + dt * new_C
        F_tr = G @ core.F[p]
        R, S = polar2(F_tr)
        m = core.mat_id[p]
        Jpold = core.Jp[p]

        if capture == 1 and (p % cap_stride) == 0:
            idx = ti.atomic_add(cap_n[None], 1)
            if idx < CAP:
                cap_S[idx] = ti.Vector([S[0, 0], S[0, 1], S[1, 1]])
                cap_C[idx] = ti.Vector([new_C[0, 0], new_C[0, 1], new_C[1, 0], new_C[1, 1]])
                cap_v[idx] = new_v
                cap_Jp[idx] = Jpold
                cap_m[idx] = m

        o = ti.Vector.zero(float, N_OUT)
        if ti.static(use_nn):
            o = mlp_eval(p, S[0, 0], S[0, 1], S[1, 1], new_C, new_v, Jpold, z_of_mat[m], H, L2)
        else:
            o = oracle_step(m, S[0, 0], S[0, 1], S[1, 1], dt * new_C.trace(), G.determinant(), Jpold)

        Sn = ti.Matrix([[S[0, 0] + o[3], S[0, 1] + o[4]], [S[0, 1] + o[4], S[1, 1] + o[5]]])
        core.F[p] = R @ Sn
        # Keep canonical's scalar J in step with det F. The learned simulator has ONE state
        # representation, so the fluid's volume ratio lives in F -- but sim/physics/signatures.py's
        # incompressibility check reads core.J for the fluid, and a stale J would let that signature
        # pass on a number the learned sim never computed.
        core.J[p] = Sn.determinant()
        core.Jp[p] = Jpold + o[6]
        tau = ti.Matrix([[o[0], o[1]], [o[1], o[2]]])
        stress_f[p] = R @ tau @ R.transpose()


@ti.kernel
def prime_stress(n: ti.i32, use_nn: ti.template(), H: ti.template(), L2: ti.template()):
    """Fill the stress cache from the current state without advancing anything. Needed because the
    first P2G of a run has no previous G2P to have written it -- and because a NETWORK does not
    output exactly zero at F = I the way the analytic law does."""
    for p in range(n):
        R, S = polar2(core.F[p])
        m = core.mat_id[p]
        o = ti.Vector.zero(float, N_OUT)
        if ti.static(use_nn):
            o = mlp_eval(p, S[0, 0], S[0, 1], S[1, 1], core.C[p], core.v[p], core.Jp[p],
                         z_of_mat[m], H, L2)
        else:
            o = oracle_step(m, S[0, 0], S[0, 1], S[1, 1], 0.0, 1.0, core.Jp[p])
        tau = ti.Matrix([[o[0], o[1]], [o[1], o[2]]])
        stress_f[p] = R @ tau @ R.transpose()


# ---------------------------------------------------------------------------- host driver
_uploaded = {"H": 0, "L2": False}


def upload_params():
    """Push the frozen canonical per-material parameters into core's m_* fields, exactly as
    core.simulate_multi does. Called before every rollout so nothing can be stale."""
    for name, i in core.MAT_ID.items():
        c = core.MAT[name]
        core.m_E[i] = c["E"]
        core.m_xi[i] = c["xi"]
        core.m_tc[i] = c["tc"]
        core.m_ts[i] = c["ts"]
        core.m_alpha[i] = core.dp_alpha(c.get("phi", 0.0))
        core.m_nu[i] = c.get("nu", core.NU)
        core.m_fric[i] = c.get("fric", core.FRICTION)
        core.m_muv[i] = 0.0
    z = NS.Z_CODES.astype(np.float32)
    for i in range(4):
        for d in range(Z_DIM):
            z_of_mat[i][d] = float(z[i, d])


def upload_weights(ps):
    """ps is netspec's parameter list with the normalisation already folded in."""
    H = ps[0][0].shape[0]
    L2 = len(ps) == 3
    assert H <= H_MAX, f"hidden width {H} exceeds H_MAX {H_MAX}"
    w1 = np.zeros((H_MAX, N_IN), np.float32); w1[:H] = ps[0][0]
    bb1 = np.zeros(H_MAX, np.float32); bb1[:H] = ps[0][1]
    W1.from_numpy(w1); b1.from_numpy(bb1)
    if L2:
        w2 = np.zeros((H_MAX, H_MAX), np.float32); w2[:H, :H] = ps[1][0]
        bb2 = np.zeros(H_MAX, np.float32); bb2[:H] = ps[1][1]
        W2h.from_numpy(w2); b2h.from_numpy(bb2)
    w3 = np.zeros((N_OUT, H_MAX), np.float32); w3[:, :H] = ps[-1][0]
    W3.from_numpy(w3); b3.from_numpy(ps[-1][1].astype(np.float32))
    _uploaded["H"] = H
    _uploaded["L2"] = L2
    return H, L2


def _seed(groups):
    """Upload a list of {material, pts, area, v0, rho?} groups into core's particle fields,
    reproducing core.simulate_multi's initialisation exactly."""
    pts = np.concatenate([np.asarray(g["pts"], np.float32) for g in groups], 0)
    n = pts.shape[0]
    assert n <= MAX_P, f"{n} particles exceeds MAX_P {MAX_P}"
    mid = np.zeros(n, np.int32)
    v0s = np.zeros((n, dim), np.float32)
    pvol = np.zeros(n, np.float32)
    prho = np.zeros(n, np.float32)
    off = 0
    for g in groups:
        k = np.asarray(g["pts"]).shape[0]
        mid[off:off + k] = core.MAT_ID[g["material"]]
        v0s[off:off + k] = np.asarray(g.get("v0", (0.0, 0.0)), np.float32)
        pvol[off:off + k] = g["area"] / k
        prho[off:off + k] = g.get("rho", core.MAT[g["material"]].get("rho", core.p_rho))
        off += k
    xb = np.zeros((MAX_P, dim), np.float32); xb[:n] = pts
    vb = np.zeros((MAX_P, dim), np.float32); vb[:n] = v0s
    mb = np.zeros(MAX_P, np.int32); mb[:n] = mid
    vo = np.zeros(MAX_P, np.float32); vo[:n] = pvol
    ro = np.zeros(MAX_P, np.float32); ro[:n] = prho
    core.x_np_buf.from_numpy(xb)
    core.v0_buf.from_numpy(vb)
    core.mat_id.from_numpy(mb)
    core.p_vol_f.from_numpy(vo)
    core.p_mass_f.from_numpy(vo * ro)
    core.init_state(n)
    stress_f.from_numpy(np.zeros((MAX_P, dim, dim), np.float32))
    return n, mid


def seed_fluid_F_from_J(n):
    """Canonical's fluid never writes F; its volume ratio lives in the scalar J. When a canonical
    state is handed to the learned simulator mid-flight, the fluid's F must be rebuilt from that J
    or the learned run silently restarts the fluid at J = 1."""
    Fn = core.F.to_numpy()
    Jn = core.J.to_numpy()
    mi = core.mat_id.to_numpy()
    sel = (mi[:n] == FLUID)
    s = np.sqrt(np.maximum(Jn[:n][sel], 1e-8))
    Fn[:n][sel] = np.stack([np.stack([s, np.zeros_like(s)], -1),
                            np.stack([np.zeros_like(s), s], -1)], -2)
    core.F.from_numpy(Fn)


def rollout(groups, T, n_frames, *, dt=None, mode="oracle", gravity_on=True,
            capture=False, cap_stride=997, hidden=None, l2=None):
    """Roll the learned (or oracle) simulator. Same signature shape as core.simulate_multi and the
    same return: (snaps, times, mats, stable, dt)."""
    upload_params()
    mats = [g["material"] for g in groups]
    dt = core.shared_dt(mats) if dt is None else dt
    grav = core.gravity if gravity_on else 0.0
    n, mid = _seed(groups)
    use_nn = (mode == "nn")
    H = hidden if hidden is not None else _uploaded["H"]
    L2 = _uploaded["L2"] if l2 is None else l2
    if not use_nn:
        H, L2 = 8, False                      # unused, but must be a concrete template value
    prime_stress(n, use_nn, H, L2)
    spf = max(1, int(round((T / n_frames) / dt)))
    snaps = np.zeros((n_frames, n, dim), np.float32)
    times = np.zeros(n_frames, np.float32)
    t, stable = 0.0, True
    cflag = 1 if capture else 0
    for fidx in range(n_frames):
        for _ in range(spf):
            core.clear_grid()
            p2g_learned(n, dt)
            core.grid_op(dt, core.FRICTION, grav)
            g2p_learned(n, dt, use_nn, H, L2, cflag, cap_stride)
            t += dt
        cur = core.x.to_numpy()[:n]
        if not np.isfinite(cur).all():
            stable = False
            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)
        snaps[fidx] = cur
        times[fidx] = t
    return snaps, times, mid, stable, dt


def simulate(material, pts, area, T, n_frames, *, v0=(0.0, 0.0), dt=None, rho=None,
             fric=None, mode="oracle", gravity_on=True, **unsupported):
    """Drop-in for core.simulate, so sim.physics.signatures can be pointed at the LEARNED simulator.

    `rho` and `fric` are honoured (both live in the analytic half of the step). `E`, `nu`, `xi`,
    `phi` and `mu_visc` are NOT: those are constitutive parameters, and in the learned simulator the
    constitutive law lives inside the network's weights, where there is no knob to turn. A signature
    that overrides one is reported N/A rather than silently run with the wrong physics.
    """
    for k, v in unsupported.items():
        if v is not None and not (k == "mu_visc" and v == 0.0):
            raise NotImplementedError(f"the learned constitutive net has no '{k}' knob")
    g = {"material": material, "pts": pts, "area": area, "v0": v0}
    if rho is not None:
        g["rho"] = float(rho)
    old_fric = core.MAT[material].get("fric")
    if fric is not None:
        core.MAT[material]["fric"] = float(fric)
    try:
        dt = core.MAT[material]["dt"] if dt is None else dt
        snaps, times, _, stable, _ = rollout([g], T, n_frames, dt=dt, mode=mode,
                                             gravity_on=gravity_on)
    finally:
        if fric is not None:
            core.MAT[material]["fric"] = old_fric
    return snaps, times, stable


def simulate_multi(groups, T, n_frames, *, dt=None, gravity_on=True, mu_visc=0.0, mode="oracle"):
    assert mu_visc == 0.0, "the learned net has no viscosity knob"
    return rollout(groups, T, n_frames, dt=dt, mode=mode, gravity_on=gravity_on)


def reset_capture():
    cap_n[None] = 0


def take_capture():
    k = min(int(cap_n[None]), CAP)
    return (cap_S.to_numpy()[:k], cap_C.to_numpy()[:k], cap_v.to_numpy()[:k],
            cap_Jp.to_numpy()[:k], cap_m.to_numpy()[:k])


def label(S3, Jp, mid):
    """Analytic targets for arbitrary trial states. S3 (N,3), Jp (N,), mid (N,) -> (N, N_OUT)."""
    upload_params()
    n = S3.shape[0]
    assert n <= CAP
    buf = np.zeros((CAP, 3), np.float32); buf[:n] = S3
    lab_S.from_numpy(buf)
    b = np.zeros(CAP, np.float32); b[:n] = Jp
    lab_Jp.from_numpy(b)
    bi = np.zeros(CAP, np.int32); bi[:n] = mid
    lab_m.from_numpy(bi)
    label_batch(n, 0.0)
    return lab_out.to_numpy()[:n]


VERSION = phys.VERSION
