"""Long, diverse, realistic fluid showcase built on the GPU screen-space renderer.

This module IMPORTS the committed pieces and does not mutate any of them:
  * ``fluid_render_gpu.GPUFluidRenderer`` -- the fast Taichi screen-space renderer (metaball splat,
    filled-interior no-holes mask, JFA distance thickness, Beer-Lambert depth color, refraction,
    Fresnel, specular, foam, caustics, bloom, tone map). Rendered in ~1-2 s per scene on a 4090.
  * ``fluid_render2`` -- imported transitively by the GPU renderer; it owns the single ``ti.init`` and
    the tank/background builder (``make_tank``) and the mp4/png helpers.
  * the Newtonian viscous stress from ``fluid_viscosity`` is REUSED (copied, not imported, because that
    file would call ``ti.init`` a second time and destroy the shared fields): the particle stress is
    ``sigma = E (J-1) I + mu_visc (C + C^T)``, the pressure plus the strain-rate stress from the APIC
    affine matrix.

Two things are added on top of the imported renderer:

  1. A small emitter-capable MLS-MPM fluid so scenes can run LONG (10-15 s) with sustained forcing: a
     faucet keeps a stream falling, two jets collide, a tank fills. A single release in a closed box
     goes quiet in about a second; sustained motion needs sustained forcing. A thin viscosity is kept
     on every scene to drain grid-scale velocity noise and keep the long weakly-compressible rollout
     from slowly drifting or blowing up; every rollout is checked finite and sampled at late frames.

  2. PER-PARTICLE COLOR (a passive dye). Each particle carries an RGB color, set once at birth and
     never touched by the physics. At render time the color is atomic-scattered per channel into GPU
     fields, blurred with the SAME Gaussian the density uses, and divided by the (equally blurred)
     density to give a smooth local mean hue. That local color drives the Beer-Lambert body tint in
     place of the renderer's single fixed water color, so two dyed liquids braid together and blend to
     an intermediate hue where their particle populations interleave. The blend is passive color
     advection, not a diffusion/mixing chemistry.

Headless only (no ti.GUI); non-differentiable and offline by design.

Usage:
    python sim/fluid_showcase_gpu.py            # full deliverable: all scenes + stills + manifest
    python sim/fluid_showcase_gpu.py --quick    # low-res short smoke test of one scene
    python sim/fluid_showcase_gpu.py --scenes color_faucets water_pour   # subset
"""
import argparse
import json
import os
import time

import numpy as np
import taichi as ti

import fluid_render_gpu as frg           # noqa: E402  (this import runs fr2's ti.init)
import fluid_render2 as fr2              # noqa: E402

# ------------------------------------------------------------------------- shared world constants
dim = 2
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
p_rho = 1.0
gravity = 9.8
bound = 3
floor_y = bound * dx

CAP = fr2.MAX_P          # particle-field capacity (== renderer capacity, 60000)
BATCH = 8192             # max particles activated in one host call

# ------------------------------------------------------------------------- sim fields (own context)
x = ti.Vector.field(dim, float, CAP)
v = ti.Vector.field(dim, float, CAP)
C = ti.Matrix.field(dim, dim, float, CAP)
J = ti.field(float, CAP)
grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

x_stage = ti.Vector.field(dim, float, BATCH)     # host -> device staging for a birth batch
v_stage = ti.Vector.field(dim, float, BATCH)


# ------------------------------------------------------------------------- constitutive stress
@ti.func
def visc_stress(p, dt, E, mu_visc, p_vol):
    """Weakly-compressible pressure PLUS a Newtonian viscous stress, both scaled by the MLS-MPM affine
    prefactor. pressure E (J-1) resists compression; mu_visc (C + C^T) resists the strain RATE (the
    symmetric part of grad v carried in the APIC affine matrix C). mu_visc = 0 recovers the inviscid
    fluid exactly. (Copied from sim/fluid_viscosity.py.)"""
    pressure = E * (J[p] - 1.0)
    Cp = C[p]
    strain_rate = Cp + Cp.transpose()
    sigma = pressure * ti.Matrix.identity(float, dim) + mu_visc * strain_rate
    return -dt * 4.0 * p_vol * inv_dx * inv_dx * sigma


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
        stress = visc_stress(p, dt, E, mu_visc, p_vol)
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
def grid_op(dt: ti.f32, fric: ti.f32, obs_x: ti.f32, obs_y: ti.f32, obs_r: ti.f32):
    for i, j in ti.ndrange(n_grid, n_grid):
        m = grid_m[i, j]
        if m > 0.0:
            grid_v[i, j] = grid_v[i, j] / m
        grid_v[i, j].y -= dt * gravity
        vx = grid_v[i, j].x
        vy = grid_v[i, j].y
        # circular obstacle (obs_r <= 0 disables): stop the normal component of any inflow.
        if obs_r > 0.0:
            gx = i * dx - obs_x
            gy = j * dx - obs_y
            if gx * gx + gy * gy < obs_r * obs_r:
                nrm = ti.Vector([gx, gy]).normalized(1e-6)
                vn = vx * nrm.x + vy * nrm.y
                if vn < 0.0:
                    vx -= vn * nrm.x
                    vy -= vn * nrm.y
        if j < bound and vy < 0:
            vx = coulomb(vx, fric * (-vy))
            vy = 0.0
        if j > n_grid - bound and vy > 0:
            vy = 0.0
        if i < bound and vx < 0:
            vx = 0.0
        if i > n_grid - bound and vx > 0:
            vx = 0.0
        grid_v[i, j] = ti.Vector([vx, vy])


@ti.kernel
def g2p(n: ti.i32, dt: ti.f32, flip: ti.f32):
    for p in range(n):
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
        v[p] = new_v + flip * (v[p] - new_v)
        x[p] = x[p] + dt * new_v
        x[p] = ti.math.clamp(x[p], floor_y, 1.0 - floor_y)
        J[p] = J[p] * (1.0 + dt * new_C.trace())
        C[p] = new_C


@ti.kernel
def _init_head(n: ti.i32):
    for p in range(n):
        x[p] = x_stage[p]      # only valid for n <= BATCH; used by activate() batches
        v[p] = v_stage[p]
        C[p] = ti.Matrix.zero(float, dim, dim)
        J[p] = 1.0


@ti.kernel
def _activate(start: ti.i32, count: ti.i32):
    for k in range(count):
        p = start + k
        x[p] = x_stage[k]
        v[p] = v_stage[k]
        C[p] = ti.Matrix.zero(float, dim, dim)
        J[p] = 1.0


# ------------------------------------------------------------------------- color-aware GPU renderer
class ColorFluidRenderer(frg.GPUFluidRenderer):
    """Subclass that adds a passive per-particle color field to the imported GPU renderer. The parent
    pipeline is untouched; only the body tint is driven by a smooth local mean of the particle colors
    instead of the single fixed water color."""

    def __init__(self, res, look=None, close_r=10, deep_k=0.42, sh_k=0.60, sh_add=(0.20, 0.22, 0.24),
                 cblur_mult=2.5):
        super().__init__(res, look, close_r=close_r)
        f = ti.f32
        r = res
        self.pcol = ti.Vector.field(3, f, self.MAXP)      # per-particle RGB dye (uploaded per frame)
        self.craw = ti.Vector.field(3, f, (r, r))         # atomic-scattered per-channel color sums
        self.cblur = ti.Vector.field(3, f, (r, r))        # blurred color sums (numerator)
        self.cbt = ti.Vector.field(3, f, (r, r))          # vec3 blur temp (horizontal pass)
        self.cden = ti.field(f, (r, r))                   # count blurred at the SAME wide sigma (denom)
        self.lcol = ti.Vector.field(3, f, (r, r))         # normalized local mean color (the dye)
        self.deep_k = deep_k
        self.sh_k = sh_k
        self.sh_add = sh_add
        # the color field is blurred WIDER than the density mask so a red/blue interface reads as a soft
        # purple blend band rather than a razor seam; the denominator count is blurred at the same width
        # so the ratio is still a proper local mean and pure regions keep their source hue.
        self.cblur_mult = cblur_mult

    # ---- color splat + normalize --------------------------------------------------------------
    @ti.kernel
    def _clear_craw(self):
        for i, j in ti.ndrange(self.res, self.res):
            self.craw[i, j] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _splat_color(self, n: ti.i32):
        res = self.res
        for p in range(n):
            c = ti.min(ti.max(int(self.pp[p].x * res), 0), res - 1)
            rr = ti.min(ti.max(int((1.0 - self.pp[p].y) * res), 0), res - 1)
            ti.atomic_add(self.craw[rr, c], self.pcol[p])

    @ti.kernel
    def _blur3_h(self, src: ti.template(), dst: ti.template(), sid: ti.i32, r: ti.i32):
        for i, j in ti.ndrange(self.res, self.res):
            acc = ti.Vector.zero(ti.f32, 3)
            for k in range(-r, r + 1):
                jj = ti.min(ti.max(j + k, 0), self.res - 1)
                acc += self.gkers[sid, k + r] * src[i, jj]
            dst[i, j] = acc

    @ti.kernel
    def _blur3_v(self, src: ti.template(), dst: ti.template(), sid: ti.i32, r: ti.i32):
        for i, j in ti.ndrange(self.res, self.res):
            acc = ti.Vector.zero(ti.f32, 3)
            for k in range(-r, r + 1):
                ii = ti.min(ti.max(i + k, 0), self.res - 1)
                acc += self.gkers[sid, k + r] * src[ii, j]
            dst[i, j] = acc

    def _blur3(self, src, dst, sigma):
        sid, r = self._sigma_id(sigma)
        self._blur3_h(src, self.cbt, sid, r)
        self._blur3_v(self.cbt, dst, sid, r)

    @ti.kernel
    def _normalize_color(self):
        # local mean dye = blur(sum color) / blur(count), both at the same wide color sigma (cden).
        for i, j in ti.ndrange(self.res, self.res):
            d = self.cden[i, j] + 1e-4
            cc = self.cblur[i, j] / d
            self.lcol[i, j] = ti.Vector([ti.min(ti.max(cc[0], 0.0), 1.0),
                                         ti.min(ti.max(cc[1], 0.0), 1.0),
                                         ti.min(ti.max(cc[2], 0.0), 1.0)])

    # ---- color-aware shading (copy of parent _shade, deep/shallow driven by lcol) --------------
    @ti.kernel
    def _shade_color(self):
        res = self.res
        deep_k = self.deep_k
        sh_k = self.sh_k
        sh_add = ti.Vector([self.sh_add[0], self.sh_add[1], self.sh_add[2]])
        F0 = self.L["F0"]
        lx = self.L["light"][0]; ly = self.L["light"][1]; lz = 0.55
        ln = ti.sqrt(lx * lx + ly * ly + lz * lz)
        lx /= ln; ly /= ln; lz /= ln
        hx = lx; hy = ly; hz = lz + 1.0
        hn = ti.sqrt(hx * hx + hy * hy + hz * hz)
        hx /= hn; hy /= hn; hz /= hn
        for i, j in ti.ndrange(res, res):
            nx = self.nx[i, j]; ny = self.ny[i, j]; nz = self.nz[i, j]
            dye = self.lcol[i, j]
            deep = dye * deep_k                                  # saturated dark body color
            shallow = dye * sh_k + sh_add                        # pale lifted thin-film color
            tt = ti.min(ti.max(self.thick[i, j], 0.0), self.L["thick_max"])
            transmit = ti.exp(-self.L["absorb"] * tt)
            body_tint = shallow * transmit + deep * (1.0 - transmit)
            col = self.refr[i, j] * transmit + body_tint * (1.0 - transmit)
            ao = 1.0 - 0.16 * frg._smoothstep(0.6, self.L["thick_max"], self.thick[i, j])
            col = col * ao
            cos_t = ti.min(ti.max(nz, 0.0), 1.0)
            fres = F0 + (1.0 - F0) * (1.0 - cos_t) ** 5
            rup = 2.0 * nz * ny
            sky = ti.min(ti.max(0.55 + 0.45 * rup, 0.0), 1.0)
            env = ti.Vector([0.60, 0.74, 0.90]) * sky + ti.Vector([0.10, 0.13, 0.17]) * (1.0 - sky)
            col = col * (1.0 - fres) + env * fres
            rimg = self.L["rim"] * (1.0 - cos_t) ** 3
            col = col + rimg * ti.Vector([0.35, 0.58, 0.78])
            ndh = ti.min(ti.max(nx * hx + ny * hy + nz * hz, 0.0), 1.0)
            spec = self.L["spec_gain"] * ndh ** self.L["shininess"]
            sheen = self.L["sheen"] * ndh ** 8.0
            col = col + (spec + sheen) * ti.Vector([1.0, 1.0, 0.97])
            self.img[i, j] = col

    # ---- per-frame driver (override to inject color) ------------------------------------------
    def render(self, pos, speed, color, frame=0):
        n = pos.shape[0]
        buf = np.zeros((self.MAXP, 2), np.float32); buf[:n] = pos.astype(np.float32)
        self.pp.from_numpy(buf)
        sbuf = np.zeros(self.MAXP, np.float32); sbuf[:n] = speed.astype(np.float32)
        self.psp.from_numpy(sbuf)
        cbuf = np.zeros((self.MAXP, 3), np.float32); cbuf[:n] = color.astype(np.float32)
        self.pcol.from_numpy(cbuf)
        self._render_device(n, frame)
        return self.out.to_numpy()

    def _render_device(self, n, frame):
        res = self.res
        self._build_masks(n)
        # local per-particle color: numerator = color splat blurred wide; denominator = the count splat
        # (self.raw, filled by _build_masks) blurred at the SAME wide sigma, so the ratio is a valid mean.
        self._clear_craw()
        self._splat_color(n)
        sc = self.L["sigma_px"] * self.cblur_mult
        self._blur3(self.craw, self.cblur, sc)
        self._blur(self.raw, self.cden, sc)
        self._normalize_color()
        # normals
        self._blur(self.fill, self.bt2, 2.5)
        shift = (frame * 5) % res
        self._make_base_field(self.bt2, shift)
        self._blur(self.ds, self.bt2, 1.6)
        self._copy_scalar(self.bt2, self.ds)
        self._normals()
        # refraction
        self._blur(self.thick, self.bt2, 2.0)
        self._refraction(self.bt2)
        # color-aware shading
        self._shade_color()
        # foam (identical to parent)
        self._blur(self.spsum, self.bt2, self.L["sigma_px"])
        self._copy_scalar(self.bt2, self.spsum)
        self._blur(self.fill, self.bt3, 3.0)
        self._foam_build(self.dens, self.bt3)
        self._blur(self.foam, self.bt2, 1.8)
        self._copy_scalar(self.bt2, self.foam)
        self._foam_apply()
        # composite over background
        self._composite()
        # caustics + contact
        self._col_reduce()
        self._blur1d_s(self.nx_mean, self.tmp1d, 4.0)
        self._copy_1d(self.tmp1d, self.nx_mean)
        self._focus_pre(self.nx_mean)
        self._blur1d_s(self.tmp1d, self.focus, 6.0)
        self._focus_norm()
        self._footprint_pre()
        self._blur1d_s(self.tmp1d, self.footprint, 12.0)
        self._caustics_contact()
        self._reflection()
        # bloom
        bloom_sigma = res / 130.0
        for ch in range(3):
            self._extract_bright(ch)
            self._blur(self.chan, self.chanb, bloom_sigma)
            self._add_bloom(ch, self.L["bloom"])
        self._tonemap()
        ti.sync()


# ------------------------------------------------------------------------- seeding helpers
def seed_disk(center, radius, n, rng):
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)], axis=1)


def seed_box(x0, x1, y0, y1, n, rng):
    return np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], axis=1)


# ------------------------------------------------------------------------- the simulation harness
class FluidSim:
    """Emitter-capable MLS-MPM fluid. Holds an active particle count and a host-side color array; the
    sim never reads the color (it is a passive tracer). Emitters activate a jittered batch each frame,
    spreading the batch along the emission direction over the frame interval so a stream reads as a
    continuous ribbon rather than a clump."""

    def __init__(self, E, dt, mu_visc, fric=0.06, flip=0.0, obstacle=None, cap=None):
        self.E = E
        self.dt = dt
        self.mu_visc = mu_visc
        self.fric = fric
        self.flip = flip
        self.obstacle = obstacle          # (x, y, r) in domain units, or None
        self.cap = cap or (CAP - 200)
        self.n = 0
        self.pcolor = np.zeros((CAP, 3), np.float32)
        self.emitters = []
        # physically consistent mass: fixed per-particle volume from a reference density of particles
        self.p_vol = None
        self.p_mass = None

    def set_volume(self, area, n_ref):
        self.p_vol = area / n_ref
        self.p_mass = self.p_vol * p_rho

    def seed_pool(self, pts, color, v0=(0.0, 0.0)):
        n = pts.shape[0]
        assert n <= self.cap
        buf = np.zeros((BATCH, dim), np.float32)
        vbuf = np.zeros((BATCH, dim), np.float32)
        vbuf[:] = np.asarray(v0, np.float32)
        # seed in BATCH-sized chunks
        start = self.n
        k = 0
        while k < n:
            m = min(BATCH, n - k)
            buf[:m] = pts[k:k + m].astype(np.float32)
            vbuf[:m] = np.asarray(v0, np.float32)
            x_stage.from_numpy(buf)
            v_stage.from_numpy(vbuf)
            _activate(start + k, m)
            k += m
        self.pcolor[start:start + n] = np.asarray(color, np.float32)
        self.n += n

    def add_emitter(self, pos, vel, half_w, rate, color, t0=0.0, t1=1e9, jitter=0.004, sweep=None):
        """pos=(x,y) emission center; vel=(vx,vy); half_w = half-width of the nozzle transverse to vel;
        rate = particles per second; color = RGB; active in [t0, t1) seconds. sweep=(amp, freq, phase)
        adds amp*sin(2 pi freq t + phase) to vx so the stream sweeps horizontally over time."""
        self.emitters.append(dict(pos=np.array(pos, np.float64), vel=np.array(vel, np.float64),
                                  half_w=half_w, rate=rate, color=np.array(color, np.float32),
                                  t0=t0, t1=t1, jitter=jitter, sweep=sweep, carry=0.0))

    def _emit(self, t, frame_dt, rng):
        ox = self.obstacle
        for em in self.emitters:
            if t < em["t0"] or t >= em["t1"]:
                continue
            em["carry"] += em["rate"] * frame_dt
            count = int(em["carry"])
            if count <= 0:
                continue
            em["carry"] -= count
            if self.n + count > self.cap:
                count = self.cap - self.n
                if count <= 0:
                    continue
            vel = em["vel"].copy()
            sw = em.get("sweep")
            if sw is not None:
                # time-varying horizontal sweep: makes a faucet paint back and forth so two colored
                # streams cross each other's territory and interleave instead of pooling side by side.
                vel[0] = vel[0] + sw[0] * np.sin(2.0 * np.pi * sw[1] * t + sw[2])
            speed = np.linalg.norm(vel) + 1e-9
            vdir = vel / speed
            # transverse direction to the flow
            tdir = np.array([-vdir[1], vdir[0]])
            # spread the batch along the flow over the frame interval so it is a ribbon, not a clump
            s = rng.uniform(0.0, 1.0, count)            # 0=leading edge (fell furthest) .. 1=nozzle
            along = -s * speed * frame_dt
            trans = rng.uniform(-em["half_w"], em["half_w"], count)
            jit = rng.uniform(-em["jitter"], em["jitter"], (count, 2))
            px = em["pos"][0] + vdir[0] * along + tdir[0] * trans + jit[:, 0]
            py = em["pos"][1] + vdir[1] * along + tdir[1] * trans + jit[:, 1]
            pts = np.stack([px, py], axis=1)
            pts = np.clip(pts, floor_y + 1e-3, 1.0 - floor_y - 1e-3)
            buf = np.zeros((BATCH, dim), np.float32)
            vbuf = np.zeros((BATCH, dim), np.float32)
            m = min(count, BATCH)
            buf[:m] = pts[:m]
            vbuf[:m] = vel.astype(np.float32)
            x_stage.from_numpy(buf)
            v_stage.from_numpy(vbuf)
            _activate(self.n, m)
            self.pcolor[self.n:self.n + m] = em["color"]
            self.n += m

    def substeps_per_frame(self, frame_dt):
        return max(1, int(round(frame_dt / self.dt)))

    def step_frame(self, t, frame_dt, rng):
        self._emit(t, frame_dt, rng)
        ox, oy, orr = (self.obstacle if self.obstacle else (0.0, 0.0, -1.0))
        nsub = self.substeps_per_frame(frame_dt)
        for _ in range(nsub):
            clear_grid()
            p2g(self.n, self.dt, self.E, self.mu_visc, self.p_vol, self.p_mass)
            grid_op(self.dt, self.fric, ox, oy, orr)
            g2p(self.n, self.dt, self.flip)
        return t + nsub * self.dt

    def readout(self):
        pos = x.to_numpy()[:self.n]
        vel = v.to_numpy()[:self.n]
        col = self.pcolor[:self.n].copy()
        finite = np.isfinite(pos).all() and np.isfinite(vel).all()
        if not finite:
            pos = np.nan_to_num(pos)
            vel = np.nan_to_num(vel)
        speed = np.linalg.norm(vel, axis=1)
        return pos, speed, col, finite


# ------------------------------------------------------------------------- scene definitions
WATER = (0.16, 0.52, 0.78)      # single-color water dye
HONEY = (0.96, 0.63, 0.11)
RED = (0.92, 0.13, 0.14)
BLUE = (0.12, 0.34, 0.95)
GREEN = (0.20, 0.80, 0.30)
YELLOW = (0.95, 0.80, 0.12)
TEAL = (0.10, 0.72, 0.74)
CLEARW = (0.16, 0.50, 0.72)


def _reset_sim():
    # zero the whole field range once so stale particles from a previous scene never render.
    _zero_all()


@ti.kernel
def _zero_all():
    for p in range(CAP):
        x[p] = ti.Vector([-1.0, -1.0])
        v[p] = ti.Vector.zero(float, dim)
        C[p] = ti.Matrix.zero(float, dim, dim)
        J[p] = 1.0


def scene_water_pour(quick=False):
    """Thin blue water: a single top faucet fills a tank, splashing and settling. Sustained forcing."""
    rng = np.random.default_rng(10)
    sim = FluidSim(E=420.0, dt=8.0e-5, mu_visc=0.03, fric=0.06, flip=0.0)
    sim.set_volume(area=0.16, n_ref=26000)
    # a shallow starting puddle so the first splashes have water to hit
    pool = seed_box(floor_y, 1 - floor_y, floor_y, 0.06, 5000 if not quick else 1500, rng)
    sim.seed_pool(pool, WATER)
    rate = 3500 if not quick else 2500
    sim.add_emitter(pos=(0.5, 0.9), vel=(0.0, -1.6), half_w=0.028, rate=rate, color=WATER)
    return dict(sim=sim, T=13.0 if not quick else 2.0, clip_s=13.0 if not quick else 2.0,
                fps=30, rng=rng, title="water_pour", hero_t=6.0 if not quick else 1.2)


def scene_triple_drop(quick=False):
    """Three colored dollops (red, green, blue) dropped in sequence into a clear pool. Timed emitter
    bursts stagger the impacts across the clip so a splash lands every couple of seconds and the colors
    interleave where the crowns overlap, rather than three simultaneous plops that just settle."""
    rng = np.random.default_rng(11)
    sim = FluidSim(E=480.0, dt=7.0e-5, mu_visc=0.03, fric=0.06, flip=0.02)
    pool = seed_box(floor_y, 1 - floor_y, floor_y, 0.10, 8000 if not quick else 2500, rng)
    sim.set_volume(area=0.15 + 4 * np.pi * 0.055 ** 2, n_ref=20000)
    sim.seed_pool(pool, CLEARW)
    burst = 15000 if not quick else 9000     # particles / second during a dollop's release window
    # four colored dollops dropped in sequence, spaced across the clip so a splash lands every ~2.7 s
    sim.add_emitter(pos=(0.28, 0.84), vel=(0.05, -1.0), half_w=0.045, rate=burst, color=RED,
                    t0=0.4, t1=0.66)
    sim.add_emitter(pos=(0.46, 0.84), vel=(0.0, -1.0), half_w=0.045, rate=burst, color=GREEN,
                    t0=3.1, t1=3.36)
    sim.add_emitter(pos=(0.62, 0.84), vel=(0.0, -1.0), half_w=0.045, rate=burst, color=BLUE,
                    t0=5.8, t1=6.06)
    sim.add_emitter(pos=(0.76, 0.84), vel=(-0.05, -1.0), half_w=0.045, rate=burst, color=YELLOW,
                    t0=8.5, t1=8.76)
    return dict(sim=sim, T=12.0 if not quick else 1.6, clip_s=12.0 if not quick else 1.6,
                fps=30, rng=rng, title="triple_drop", hero_t=6.2 if not quick else 0.8)


def scene_dam_break(quick=False):
    """Classic tall column released against the left wall; collapses, races across, jets up the far
    wall, returns and sloshes. Run long enough to slosh several times and settle."""
    rng = np.random.default_rng(12)
    sim = FluidSim(E=520.0, dt=7.0e-5, mu_visc=0.04, fric=0.04, flip=0.02)
    n = 30000 if not quick else 9000
    col = seed_box(floor_y, 0.32, floor_y, 0.82, n, rng)
    sim.set_volume(area=(0.32 - floor_y) * (0.82 - floor_y), n_ref=n)
    sim.seed_pool(col, WATER)
    return dict(sim=sim, T=11.0 if not quick else 1.6, clip_s=11.0 if not quick else 1.6,
                fps=30, rng=rng, title="dam_break", hero_t=1.6 if not quick else 0.9)


def scene_honey_coil(quick=False):
    """Thick amber honey: a faucet pours a slow rope of honey onto the floor that piles and coils.
    Thick viscosity (small dt), softer E so the timestep stays feasible over the rollout."""
    rng = np.random.default_rng(13)
    sim = FluidSim(E=200.0, dt=1.5e-5, mu_visc=0.5, fric=0.30, flip=0.0)
    sim.set_volume(area=0.05, n_ref=12000)
    rate = 3400 if not quick else 1800
    sim.add_emitter(pos=(0.5, 0.74), vel=(0.0, -0.7), half_w=0.02, rate=rate, color=HONEY)
    # honey is intrinsically slow, so it is shown in mild slow motion: 6.5 s of physics stretched to a
    # 12 s clip, which reads naturally for a coiling rope and keeps the tiny-dt sim cost bounded.
    return dict(sim=sim, T=6.5 if not quick else 1.0, clip_s=12.0 if not quick else 1.6,
                fps=30, rng=rng, title="honey_coil", hero_t=5.2 if not quick else 0.8)


def scene_color_faucets(quick=False):
    """THE color-mixing hero. A red faucet and a blue faucet pour together into one tank; the streams
    stay legibly red and blue where they fall, and a purple band grows in the churn between them."""
    rng = np.random.default_rng(14)
    sim = FluidSim(E=440.0, dt=8.0e-5, mu_visc=0.035, fric=0.05, flip=0.0)
    sim.set_volume(area=0.16, n_ref=24000)
    pool = seed_box(floor_y, 1 - floor_y, floor_y, 0.04, 2600 if not quick else 1200, rng)
    sim.seed_pool(pool, (0.5 * RED[0] + 0.5 * BLUE[0], 0.5 * RED[1] + 0.5 * BLUE[1],
                         0.5 * RED[2] + 0.5 * BLUE[2]))
    rate = 1450 if not quick else 1300
    # the two faucets SWEEP horizontally in opposite phase so each paints back and forth across the
    # centre, laying interleaved red and blue that the wide colour blur reads as a purple blend band.
    # A lower fill keeps the surface churning for the whole clip instead of settling into a full tank.
    sim.add_emitter(pos=(0.42, 0.9), vel=(0.12, -1.5), half_w=0.024, rate=rate, color=RED,
                    sweep=(1.8, 0.45, 0.0))
    sim.add_emitter(pos=(0.58, 0.9), vel=(-0.12, -1.5), half_w=0.024, rate=rate, color=BLUE,
                    sweep=(1.8, 0.45, np.pi))
    return dict(sim=sim, T=14.0 if not quick else 2.2, clip_s=14.0 if not quick else 2.2,
                fps=30, rng=rng, title="color_faucets", hero_t=8.5 if not quick else 1.4)


def scene_jet_collide(quick=False):
    """Two opposing horizontal jets, red from the left and yellow from the right, collide mid-air, throw
    a fan of spray, and rain down blending to orange where they meet. A distinct jet/stream starting
    condition plus mixing. Red and yellow are chosen because their average is a clean vivid orange,
    where near-complementary dyes would blend to a muddy grey."""
    rng = np.random.default_rng(15)
    sim = FluidSim(E=460.0, dt=7.0e-5, mu_visc=0.03, fric=0.05, flip=0.0)
    sim.set_volume(area=0.10, n_ref=22000)
    pool = seed_box(floor_y, 1 - floor_y, floor_y, 0.04, 3000 if not quick else 1200, rng)
    sim.seed_pool(pool, (0.5 * RED[0] + 0.5 * YELLOW[0], 0.5 * RED[1] + 0.5 * YELLOW[1],
                         0.5 * RED[2] + 0.5 * YELLOW[2]))
    rate = 2400 if not quick else 1400
    sim.add_emitter(pos=(0.07, 0.62), vel=(2.1, 0.15), half_w=0.022, rate=rate, color=RED)
    sim.add_emitter(pos=(0.93, 0.62), vel=(-2.1, 0.15), half_w=0.022, rate=rate, color=YELLOW)
    return dict(sim=sim, T=11.0 if not quick else 2.0, clip_s=11.0 if not quick else 2.0,
                fps=30, rng=rng, title="jet_collide", hero_t=4.5 if not quick else 1.2)


SCENES = {
    "water_pour": scene_water_pour,
    "triple_drop": scene_triple_drop,
    "dam_break": scene_dam_break,
    "honey_coil": scene_honey_coil,
    "color_faucets": scene_color_faucets,
    "jet_collide": scene_jet_collide,
}


# ------------------------------------------------------------------------- run one scene
def run_scene(name, renderer, out_dir, quick=False, probe_dir=None, n_probes=6):
    cfg = SCENES[name](quick=quick)
    sim = cfg["sim"]
    fps = cfg["fps"]
    T = cfg["T"]                       # physical seconds simulated
    clip_s = cfg["clip_s"]            # playback length of the mp4 (> T means mild slow motion)
    rng = cfg["rng"]
    n_frames = int(round(clip_s * fps))
    frame_dt = T / n_frames           # physical seconds advanced per rendered frame
    title = cfg["title"]
    hero_frame = min(int(round(cfg["hero_t"] / T * n_frames)), n_frames - 1)
    probe_idxs = set(int(round(k)) for k in np.linspace(6, n_frames - 1, n_probes))
    probe_idxs.add(hero_frame)

    import imageio
    mp4_path = os.path.join(out_dir, f"{title}.mp4")
    writer = imageio.get_writer(mp4_path, fps=fps, quality=9, macro_block_size=1, codec="libx264")

    t = 0.0
    t0 = time.perf_counter()
    peak_speed = 0.0
    any_nonfinite = False
    hero_img = None
    max_n = 0
    for f in range(n_frames):
        t = sim.step_frame(t, frame_dt, rng)
        pos, speed, col, finite = sim.readout()
        any_nonfinite = any_nonfinite or (not finite)
        peak_speed = max(peak_speed, float(speed.max()) if speed.size else 0.0)
        max_n = max(max_n, sim.n)
        img = renderer.render(pos, speed, col, frame=f)
        even = img[: img.shape[0] - img.shape[0] % 2, : img.shape[1] - img.shape[1] % 2]
        writer.append_data(even)
        if f == hero_frame:
            hero_img = img.copy()
        if probe_dir is not None and f in probe_idxs:
            fr2.save_png(os.path.join(probe_dir, f"{title}_{f:04d}.png"), img)
    writer.close()
    if hero_img is not None:
        fr2.save_png(os.path.join(out_dir, f"{title}_hero.png"), hero_img)
    wall = time.perf_counter() - t0
    info = dict(title=title, n_frames=n_frames, fps=fps, T_phys=T, duration_s=n_frames / fps,
                slowmo=round(clip_s / T, 2), peak_speed=peak_speed, nonfinite=bool(any_nonfinite),
                n_particles=int(max_n), hero_frame=hero_frame, wall_s=wall,
                E=sim.E, dt=sim.dt, mu_visc=sim.mu_visc, flip=sim.flip)
    print(f"[{title}] frames={n_frames} dur={n_frames/fps:.1f}s (T_phys={T}s slowmo={clip_s/T:.2f}x) "
          f"Np<= {max_n} peak|v|={peak_speed:.2f} nonfinite={any_nonfinite} wall={wall:.1f}s")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--no-probes", action="store_true")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo, "runs", "realistic-rendering", "more-realistic-basic-fluid-sims")
    os.makedirs(out_dir, exist_ok=True)
    probe_dir = None if args.no_probes else os.path.join(out_dir, "_probes")
    if probe_dir:
        os.makedirs(probe_dir, exist_ok=True)

    res = args.res or (420 if args.quick else 1000)
    L = dict(fr2.DEFAULT_LOOK)
    L["res"] = res
    # showcase look: the dye demo wants the falling streams to read as COLORED liquid, not whitewater,
    # so foam is pulled back to genuine fast crests / spray tips only, and absorption is lifted a touch
    # so even a moderately thin colored stream shows its hue rather than transmitting straight through.
    L.update(dict(absorb=1.7, foam_speed=2.6, foam_gain=0.7, foam_thin=(0.12, 0.02), foam_thin_w=0.18))
    tank = fr2.make_tank(res, L)

    print(f"[gpu] building color renderer at res={res} ...")
    renderer = ColorFluidRenderer(res, L, close_r=10)
    renderer.upload_tank(tank)

    names = args.scenes or list(SCENES.keys())
    info_path = os.path.join(out_dir, "scene_info.json")
    # merge with any existing info so separate invocations (rendered in groups) accumulate.
    merged = {}
    if os.path.exists(info_path):
        try:
            for d in json.load(open(info_path)):
                merged[d["title"]] = d
        except Exception:
            merged = {}
    for name in names:
        _reset_sim()
        d = run_scene(name, renderer, out_dir, quick=args.quick, probe_dir=probe_dir)
        merged[d["title"]] = d
        with open(info_path, "w") as fh:
            json.dump([merged[k] for k in merged], fh, indent=2)
    print("[done] scenes now on disk:", list(merged.keys()))


if __name__ == "__main__":
    main()
