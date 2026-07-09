"""GPU (Taichi) port of the screen-space fluid renderer ``sim/fluid_render2.py``.

The v2 renderer (``sim/fluid_render2.py``) is pure numpy/scipy on the CPU: a stack of full-image passes
(metaball splat + Gaussian blur, a ``distance_transform_edt`` for thickness, per-pixel background
refraction, Fresnel, specular, foam, bloom, tone map) run single-threaded. That is ~0.5-2 s per 1080**2
frame, so a multi-scene showcase takes tens of minutes while the physics (a few thousand MLS-MPM
particles) is a sub-millisecond GPU job. The *render*, not the physics, is the bottleneck.

This module reimplements the **same pipeline** in Taichi kernels, keeping every intermediate on device and
reading back only the final RGB frame. The stages map onto the GPU as:

  * particle -> density : atomic **scatter** of particle counts into a grid field (the metaball splat),
    then a **separable Gaussian blur** (two 1D kernel passes) instead of ``scipy.ndimage.gaussian_filter``.
  * filled interior / no holes : the CPU fix (``distance_transform_edt`` + a connected-component small-hole
    fill) is replaced by a GPU-friendly equivalent -- a bounded morphological **closing** (grey close +
    binary close via disk max/min kernels) that seals the sub-particle-spacing Poisson pinholes while
    leaving genuine large cavities (splash craters, a breaking-wave barrel) open, plus a **jump-flooding
    (JFA) Euclidean distance transform** for the smooth optical thickness. The JFA is O(log n) full-image
    passes and is exact on device.
  * shading : surface normals from the density-field gradient, Beer-Lambert depth color, background
    refraction (normal + thickness-gradient lensing with chromatic dispersion), Fresnel, rim, Blinn-Phong
    specular, surface-gated foam, floor caustics, contact shadow, floor reflection, bloom, vignette, tone
    map -- all the same formulas as the CPU version, each a per-pixel Taichi kernel.

The physics/sim and the CPU reference renderer are imported from ``sim/fluid_render2.py`` (which is left
untouched); this file adds only the GPU renderer plus a benchmark / visual-parity driver.

Usage:
    python sim/fluid_render_gpu.py               # full deliverable: bench + parity still + GPU clip + manifest
    python sim/fluid_render_gpu.py --quick       # low-res smoke test
    python sim/fluid_render_gpu.py --bench-only   # just the CPU-vs-GPU timing table
"""
import argparse
import os
import time

import numpy as np
import taichi as ti

# Importing the v2 module runs its ``ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)`` and defines
# the MLS-MPM sim (run_scene) plus the CPU reference renderer (render_frame, pure numpy/scipy). This module
# reuses that single Taichi context and adds its own fields; it does NOT call ti.init again (a second init
# would destroy the imported sim fields).
import fluid_render2 as fr2

RMAX = 64          # max Gaussian blur radius supported (covers sigma up to ~16 with truncate=4)
NBINS = 1024       # histogram bins for the on-device percentile normalization
BIG = 1e18


@ti.func
def _smoothstep(a, b, x):
    t = ti.math.clamp((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.data_oriented
class GPUFluidRenderer:
    """Screen-space fluid renderer, entirely on the GPU. Allocate once for a fixed resolution, upload the
    static background + noise (computed once by the CPU code), then call ``render(pos, speed, frame)`` per
    frame; only the final uint8 image is read back."""

    def __init__(self, res, look=None, close_r=10):
        self.res = res
        self.MAXP = fr2.MAX_P
        L = dict(fr2.DEFAULT_LOOK)
        if look:
            L.update(look)
        L["res"] = res
        self.L = L
        self.close_r = close_r
        r = res
        f = ti.f32
        vec2i = lambda: ti.Vector.field(2, ti.i32, (r, r))
        sc = lambda: ti.field(f, (r, r))
        v3 = lambda: ti.Vector.field(3, f, (r, r))

        # particle inputs
        self.pp = ti.Vector.field(2, f, self.MAXP)     # positions (x,y) in [0,1]
        self.psp = ti.field(f, self.MAXP)              # per-particle speed

        # scatter / density
        self.raw = sc()          # splatted particle count
        self.dens = sc()         # blurred count (metaball density); also foam denominator
        self.work = sc()         # normalized density scratch (grey close etc.)
        self.body = sc()         # filled interior mask (0/1)
        self.fill = sc()         # feathered opacity in [0,1]
        self.thick = sc()        # smooth optical thickness
        self.bt = sc()           # blur temp (horizontal pass output)
        self.bt2 = sc()          # general scalar scratch
        self.bt3 = sc()          # general scalar scratch
        self.ds = sc()           # smoothed field used for normals
        self.nx = sc(); self.ny = sc(); self.nz = sc()
        self.spsum = sc()        # blurred speed splat (foam numerator)
        self.foam = sc()

        # images
        self.bg = v3()           # static background (uploaded once)
        self.refr = v3()         # refracted background
        self.img = v3()          # working color / composited image
        self.chan = sc()         # single-channel scratch for image blurs (bloom)
        self.chanb = sc()

        # static noise (uploaded once)
        self.ripple = sc()
        self.foamn = sc()

        # jump-flooding seed buffers (nearest-seed coordinates)
        self.sa = vec2i()
        self.sb = vec2i()
        self.jsrc = sc()         # 1 where a JFA seed sits

        # gaussian kernel weights, one row per distinct sigma. Built once (during warm-up) and reused, so
        # no host->device kernel upload happens on the steady-state per-frame path.
        self.NSIG = 32
        self.gkers = ti.field(f, (self.NSIG, 2 * RMAX + 1))
        self._gkers_np = np.zeros((self.NSIG, 2 * RMAX + 1), np.float32)
        self._sigma_ids = {}
        self._next_sid = 0

        # histogram percentile
        self.hist = ti.field(ti.i32, NBINS)
        self.dmax = ti.field(f, ())
        self.htot = ti.field(ti.i32, ())
        self.ref = ti.field(f, ())

        # 1D column functions (caustics / contact shadow)
        self.col_water = ti.field(f, r)
        self.nx_mean = ti.field(f, r)
        self.tmp1d = ti.field(f, r)
        self.focus = ti.field(f, r)
        self.footprint = ti.field(f, r)
        self.cw_max = ti.field(f, ())
        self.foc_max = ti.field(f, ())

        # final image (uint8, read back once per frame)
        self.out = ti.field(ti.u8, (r, r, 3))

        self._sigma_cache = -1.0
        self.floor_row = int((1.0 - fr2.floor_y) * res)

    # ------------------------------------------------------------------ static setup
    def upload_tank(self, tank):
        """Upload the one-time background image and noise fields the CPU builds in ``make_tank``."""
        self.bg.from_numpy(tank["bg"].astype(np.float32))
        self.ripple.from_numpy(tank["ripple_noise"].astype(np.float32))
        self.foamn.from_numpy(tank["foam_noise"].astype(np.float32))
        self.floor_row = tank.get("floor_row", self.floor_row)

    def _sigma_id(self, sigma):
        """Return (row-id, radius) for a Gaussian of this sigma, building the row once and caching it."""
        key = round(float(sigma), 4)
        if key in self._sigma_ids:
            return self._sigma_ids[key]
        sid = self._next_sid
        self._next_sid += 1
        r = max(1, min(int(4.0 * sigma + 0.5), RMAX))
        k = np.arange(-r, r + 1, dtype=np.float64)
        w = np.exp(-(k * k) / (2.0 * sigma * sigma))
        w /= w.sum()
        self._gkers_np[sid, : 2 * r + 1] = w.astype(np.float32)
        self.gkers.from_numpy(self._gkers_np)   # only during warm-up, when a new sigma first appears
        self._sigma_ids[key] = (sid, r)
        return sid, r

    def _blur(self, src, dst, sigma):
        sid, r = self._sigma_id(sigma)
        self._blur_h(src, self.bt, sid, r)
        self._blur_v(self.bt, dst, sid, r)

    # ------------------------------------------------------------------ kernels: blur
    @ti.kernel
    def _blur_h(self, src: ti.template(), dst: ti.template(), sid: ti.i32, r: ti.i32):
        for i, j in ti.ndrange(self.res, self.res):
            acc = 0.0
            for k in range(-r, r + 1):
                jj = ti.min(ti.max(j + k, 0), self.res - 1)
                acc += self.gkers[sid, k + r] * src[i, jj]
            dst[i, j] = acc

    @ti.kernel
    def _blur_v(self, src: ti.template(), dst: ti.template(), sid: ti.i32, r: ti.i32):
        for i, j in ti.ndrange(self.res, self.res):
            acc = 0.0
            for k in range(-r, r + 1):
                ii = ti.min(ti.max(i + k, 0), self.res - 1)
                acc += self.gkers[sid, k + r] * src[ii, j]
            dst[i, j] = acc

    @ti.kernel
    def _blur1d(self, src: ti.template(), dst: ti.template(), sid: ti.i32, r: ti.i32):
        for i in range(self.res):
            acc = 0.0
            for k in range(-r, r + 1):
                ii = ti.min(ti.max(i + k, 0), self.res - 1)
                acc += self.gkers[sid, k + r] * src[ii]
            dst[i] = acc

    def _blur1d_s(self, src, dst, sigma):
        sid, r = self._sigma_id(sigma)
        self._blur1d(src, dst, sid, r)

    # ------------------------------------------------------------------ kernels: splat
    @ti.kernel
    def _clear_raw(self):
        for i, j in ti.ndrange(self.res, self.res):
            self.raw[i, j] = 0.0
            self.spsum[i, j] = 0.0

    @ti.kernel
    def _splat(self, n: ti.i32):
        # metaball splat == histogram of particle counts (and speed-weighted counts for foam), row 0 = top.
        res = self.res
        for p in range(n):
            c = ti.min(ti.max(int(self.pp[p].x * res), 0), res - 1)
            rr = ti.min(ti.max(int((1.0 - self.pp[p].y) * res), 0), res - 1)
            ti.atomic_add(self.raw[rr, c], 1.0)
            ti.atomic_add(self.spsum[rr, c], self.psp[p])

    # ------------------------------------------------------------------ kernels: percentile
    @ti.kernel
    def _dens_max(self):
        self.dmax[None] = 0.0
        for i, j in ti.ndrange(self.res, self.res):
            ti.atomic_max(self.dmax[None], self.dens[i, j])

    @ti.kernel
    def _build_hist(self):
        for b in range(NBINS):
            self.hist[b] = 0
        self.htot[None] = 0
        for i, j in ti.ndrange(self.res, self.res):
            dv = self.dens[i, j]
            if dv > 1e-4:
                b = int(dv / (self.dmax[None] + 1e-12) * (NBINS - 1))
                b = ti.min(ti.max(b, 0), NBINS - 1)
                ti.atomic_add(self.hist[b], 1)
                ti.atomic_add(self.htot[None], 1)

    @ti.kernel
    def _compute_ref(self):
        ref = 1.0
        if self.htot[None] > 0:
            thresh = 0.8 * self.htot[None]
            acc = 0
            idx = NBINS - 1
            found = 0
            ti.loop_config(serialize=True)
            for b in range(NBINS):
                acc += self.hist[b]
                if found == 0 and acc >= thresh:
                    idx = b
                    found = 1
            ref = (idx + 0.5) / NBINS * self.dmax[None]
        self.ref[None] = ref + 1e-9

    @ti.kernel
    def _normalize(self):
        for i, j in ti.ndrange(self.res, self.res):
            self.work[i, j] = self.dens[i, j] / self.ref[None]

    # ------------------------------------------------------------------ kernels: morphology (disk max/min)
    @ti.kernel
    def _dilate(self, src: ti.template(), dst: ti.template(), r: ti.i32):
        res = self.res
        for i, j in ti.ndrange(res, res):
            m = -BIG
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if di * di + dj * dj <= r * r:
                        ii = ti.min(ti.max(i + di, 0), res - 1)
                        jj = ti.min(ti.max(j + dj, 0), res - 1)
                        m = ti.max(m, src[ii, jj])
            dst[i, j] = m

    @ti.kernel
    def _erode(self, src: ti.template(), dst: ti.template(), r: ti.i32):
        res = self.res
        for i, j in ti.ndrange(res, res):
            m = BIG
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if di * di + dj * dj <= r * r:
                        ii = ti.min(ti.max(i + di, 0), res - 1)
                        jj = ti.min(ti.max(j + dj, 0), res - 1)
                        m = ti.min(m, src[ii, jj])
            dst[i, j] = m

    @ti.kernel
    def _threshold(self, src: ti.template(), dst: ti.template(), t: ti.f32):
        for i, j in ti.ndrange(self.res, self.res):
            dst[i, j] = 1.0 if src[i, j] > t else 0.0

    # ------------------------------------------------------------------ kernels: jump-flooding EDT
    @ti.kernel
    def _jfa_init(self):
        # seed = pixels where jsrc>0.5; store own coords, else sentinel.
        for i, j in ti.ndrange(self.res, self.res):
            if self.jsrc[i, j] > 0.5:
                self.sa[i, j] = ti.Vector([i, j])
            else:
                self.sa[i, j] = ti.Vector([-1, -1])

    @ti.kernel
    def _jfa_pass(self, src: ti.template(), dst: ti.template(), step: ti.i32):
        res = self.res
        for i, j in ti.ndrange(res, res):
            best = src[i, j]
            bd = BIG
            if best.x >= 0:
                bd = float((i - best.x) ** 2 + (j - best.y) ** 2)
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    ii = i + di * step
                    jj = j + dj * step
                    if 0 <= ii < res and 0 <= jj < res:
                        cand = src[ii, jj]
                        if cand.x >= 0:
                            d = float((i - cand.x) ** 2 + (j - cand.y) ** 2)
                            if d < bd:
                                bd = d
                                best = cand
            dst[i, j] = best

    @ti.kernel
    def _copy_seed_from(self, src: ti.template()):
        for i, j in ti.ndrange(self.res, self.res):
            self.sa[i, j] = src[i, j]

    def _jfa(self, max_step):
        # ping-pong between sa/sb so no per-pass copy is needed; leave the result in sa.
        self._jfa_init()
        step = 1
        while step < max_step:
            step *= 2
        cur, nxt = self.sa, self.sb
        while step >= 1:
            self._jfa_pass(cur, nxt, step)
            cur, nxt = nxt, cur
            step //= 2
        if cur is not self.sa:
            self._copy_seed_from(cur)

    @ti.kernel
    def _edt_from_seed(self, dst: ti.template(), body_mask: ti.template()):
        # distance to nearest seed, zeroed outside the body (matches distance_transform_edt(body)).
        for i, j in ti.ndrange(self.res, self.res):
            d = 0.0
            s = self.sa[i, j]
            if s.x >= 0 and body_mask[i, j] > 0.5:
                d = ti.sqrt(float((i - s.x) ** 2 + (j - s.y) ** 2))
            dst[i, j] = d

    # ------------------------------------------------------------------ mask pipeline
    def _build_masks(self, n):
        self._clear_raw()
        self._splat(n)
        self._blur(self.raw, self.dens, self.L["sigma_px"])
        self._dens_max()
        self._build_hist()
        self._compute_ref()
        self._normalize()                       # work = dens / ref
        # grey closing (disk 3): grey dilate then grey erode
        self._dilate(self.work, self.bt2, 3)
        self._erode(self.bt2, self.work, 3)
        # low-threshold body, then a bounded binary closing to seal Poisson pinholes / small pockets while
        # leaving genuine large cavities open (the GPU-friendly stand-in for close + small-hole fill).
        self._threshold(self.work, self.body, self.L["iso_fill"])
        cr = self.close_r
        self._dilate(self.body, self.bt2, cr)
        self._erode(self.bt2, self.body, cr)
        # feathered opacity from a smoothed body
        self._blur(self.body, self.bt2, 2.0)
        self._feather(self.bt2, self.fill, 0.5 - self.L["edge"], 0.5 + self.L["edge"])
        # smooth optical thickness: JFA Euclidean distance transform of the filled body
        self._air_seed(self.body)               # jsrc = 1 where air (body==0)
        maxd = int(self.L["thick_char"] * self.L["thick_max"] + 8)
        self._jfa(maxd)
        self._edt_from_seed(self.bt2, self.body)
        self._scale_clip(self.bt2, self.bt3, 1.0 / self.L["thick_char"], self.L["thick_max"])
        self._blur(self.bt3, self.thick, 2.0)
        self._clip_field(self.thick, self.L["thick_max"])

    @ti.kernel
    def _feather(self, src: ti.template(), dst: ti.template(), a: ti.f32, b: ti.f32):
        for i, j in ti.ndrange(self.res, self.res):
            dst[i, j] = _smoothstep(a, b, src[i, j])

    @ti.kernel
    def _air_seed(self, body: ti.template()):
        for i, j in ti.ndrange(self.res, self.res):
            self.jsrc[i, j] = 1.0 if body[i, j] < 0.5 else 0.0

    @ti.kernel
    def _scale_clip(self, src: ti.template(), dst: ti.template(), s: ti.f32, hi: ti.f32):
        for i, j in ti.ndrange(self.res, self.res):
            dst[i, j] = ti.min(ti.max(src[i, j] * s, 0.0), hi)

    @ti.kernel
    def _clip_field(self, f: ti.template(), hi: ti.f32):
        for i, j in ti.ndrange(self.res, self.res):
            f[i, j] = ti.min(ti.max(f[i, j], 0.0), hi)

    # ------------------------------------------------------------------ normals
    @ti.kernel
    def _make_base_field(self, blurred_fill: ti.template(), shift: ti.i32):
        res = self.res
        for i, j in ti.ndrange(res, res):
            base = blurred_fill[i, j] + 0.6 * ti.min(ti.max(self.thick[i, j], 0.0),
                                                     self.L["thick_max"]) / self.L["thick_max"]
            if ti.static(self.L["ripple_amp"] > 0):
                jj = (j + shift) % res
                ripple = self.ripple[i, jj] - 0.5
                interior = _smoothstep(0.2, 0.9, self.thick[i, j] / self.L["thick_max"])
                base += self.L["ripple_amp"] * ripple * interior
            self.ds[i, j] = base

    @ti.kernel
    def _normals(self):
        res = self.res
        for i, j in ti.ndrange(res, res):
            i0 = ti.max(i - 1, 0); i1 = ti.min(i + 1, res - 1)
            j0 = ti.max(j - 1, 0); j1 = ti.min(j + 1, res - 1)
            grow = (self.ds[i1, j] - self.ds[i0, j]) * 0.5
            gcol = (self.ds[i, j1] - self.ds[i, j0]) * 0.5
            nx = -gcol * self.L["normal_amp"]
            ny = grow * self.L["normal_amp"]
            nz = self.L["normal_k"]
            nn = ti.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
            self.nx[i, j] = nx / nn
            self.ny[i, j] = ny / nn
            self.nz[i, j] = nz / nn

    # ------------------------------------------------------------------ refraction + shading
    @ti.func
    def _sample_bg(self, col, row):
        res = self.res
        c = ti.min(ti.max(col, 0.0), res - 1.001)
        rr = ti.min(ti.max(row, 0.0), res - 1.001)
        c0 = int(ti.floor(c)); r0 = int(ti.floor(rr))
        c1 = c0 + 1; r1 = r0 + 1
        fc = c - c0; fr = rr - r0
        a = self.bg[r0, c0]; b = self.bg[r0, c1]
        cc = self.bg[r1, c0]; d = self.bg[r1, c1]
        return (a * (1 - fc) * (1 - fr) + b * fc * (1 - fr)
                + cc * (1 - fc) * fr + d * fc * fr)

    @ti.kernel
    def _refraction(self, tblur: ti.template()):
        # tblur = gaussian_filter(thick, 2.0); its gradient drives interior lensing.
        res = self.res
        chroma = self.L["chroma"]
        for i, j in ti.ndrange(res, res):
            i0 = ti.max(i - 1, 0); i1 = ti.min(i + 1, res - 1)
            j0 = ti.max(j - 1, 0); j1 = ti.min(j + 1, res - 1)
            tgr = (tblur[i1, j] - tblur[i0, j]) * 0.5
            tgc = (tblur[i, j1] - tblur[i, j0]) * 0.5
            off_n = self.L["refract"] * ti.min(ti.max(self.thick[i, j], 0.0), 1.6)
            lx = self.nx[i, j] * off_n - tgc * self.L["refract_lens"]
            ly = self.ny[i, j] * off_n + tgr * self.L["refract_lens"]
            # chromatic dispersion: R/G/B sampled at slightly different offset scales.
            sr = 1.0 + chroma
            sg = 1.0
            sb = 1.0 - chroma
            rr = self._sample_bg(j + lx * sr, i - ly * sr)[0]
            gg = self._sample_bg(j + lx * sg, i - ly * sg)[1]
            bb = self._sample_bg(j + lx * sb, i - ly * sb)[2]
            self.refr[i, j] = ti.Vector([rr, gg, bb])

    @ti.kernel
    def _shade(self):
        res = self.res
        liquid = ti.Vector([self.L["liquid"][0], self.L["liquid"][1], self.L["liquid"][2]])
        shallow = ti.Vector([self.L["shallow"][0], self.L["shallow"][1], self.L["shallow"][2]])
        F0 = self.L["F0"]
        # key light + half vector (constant over the image)
        lx = self.L["light"][0]; ly = self.L["light"][1]; lz = 0.55
        ln = ti.sqrt(lx * lx + ly * ly + lz * lz)
        lx /= ln; ly /= ln; lz /= ln
        hx = lx; hy = ly; hz = lz + 1.0
        hn = ti.sqrt(hx * hx + hy * hy + hz * hz)
        hx /= hn; hy /= hn; hz /= hn
        for i, j in ti.ndrange(res, res):
            nx = self.nx[i, j]; ny = self.ny[i, j]; nz = self.nz[i, j]
            tt = ti.min(ti.max(self.thick[i, j], 0.0), self.L["thick_max"])
            transmit = ti.exp(-self.L["absorb"] * tt)
            body_tint = shallow * transmit + liquid * (1.0 - transmit)
            col = self.refr[i, j] * transmit + body_tint * (1.0 - transmit)
            # ambient occlusion for volume
            ao = 1.0 - 0.16 * _smoothstep(0.6, self.L["thick_max"], self.thick[i, j])
            col = col * ao
            # Fresnel environment reflection
            cos_t = ti.min(ti.max(nz, 0.0), 1.0)
            fres = F0 + (1.0 - F0) * (1.0 - cos_t) ** 5
            rup = 2.0 * nz * ny
            sky = ti.min(ti.max(0.55 + 0.45 * rup, 0.0), 1.0)
            env = ti.Vector([0.60, 0.74, 0.90]) * sky + ti.Vector([0.10, 0.13, 0.17]) * (1.0 - sky)
            col = col * (1.0 - fres) + env * fres
            # rim light at the grazing waterline
            rimg = self.L["rim"] * (1.0 - cos_t) ** 3
            col = col + rimg * ti.Vector([0.35, 0.58, 0.78])
            # Blinn-Phong specular (tight glint + broad wet sheen)
            ndh = ti.min(ti.max(nx * hx + ny * hy + nz * hz, 0.0), 1.0)
            spec = self.L["spec_gain"] * ndh ** self.L["shininess"]
            sheen = self.L["sheen"] * ndh ** 8.0
            col = col + (spec + sheen) * ti.Vector([1.0, 1.0, 0.97])
            self.img[i, j] = col

    # ------------------------------------------------------------------ foam
    @ti.kernel
    def _foam_build(self, avgden: ti.template(), edgeblur: ti.template()):
        res = self.res
        for i, j in ti.ndrange(res, res):
            avg_speed = self.spsum[i, j] / (avgden[i, j] + 1e-6)
            motion = _smoothstep(self.L["foam_speed"], self.L["foam_speed"] * 2.2, avg_speed)
            i0 = ti.max(i - 1, 0); i1 = ti.min(i + 1, res - 1)
            j0 = ti.max(j - 1, 0); j1 = ti.min(j + 1, res - 1)
            er = (edgeblur[i1, j] - edgeblur[i0, j]) * 0.5
            ec = (edgeblur[i, j1] - edgeblur[i, j0]) * 0.5
            edgemag = ti.sqrt(er * er + ec * ec)
            band = _smoothstep(0.004, 0.02, edgemag)
            thinfoam = self.fill[i, j] * _smoothstep(self.L["foam_thin"][0], self.L["foam_thin"][1], self.thick[i, j])
            motionfoam = motion * band
            fo = ti.min(ti.max(self.L["foam_gain"] * (0.9 * motionfoam + self.L["foam_thin_w"] * thinfoam), 0.0), 1.0)
            fo = fo * (1.0 - self.L["foam_tex"] * (1.0 - self.foamn[i, j]))
            self.foam[i, j] = fo

    @ti.kernel
    def _foam_apply(self):
        fc = ti.Vector([0.93, 0.97, 1.0])
        for i, j in ti.ndrange(self.res, self.res):
            f = self.foam[i, j]
            self.img[i, j] = self.img[i, j] * (1.0 - f) + fc * f

    @ti.kernel
    def _composite(self):
        for i, j in ti.ndrange(self.res, self.res):
            m = self.fill[i, j]
            self.img[i, j] = self.bg[i, j] * (1.0 - m) + self.img[i, j] * m

    # ------------------------------------------------------------------ caustics + contact (1D columns)
    @ti.kernel
    def _col_reduce(self):
        res = self.res
        for c in range(res):
            self.col_water[c] = 0.0
            self.nx_mean[c] = 0.0
        for i, j in ti.ndrange(res, res):
            ti.atomic_add(self.col_water[j], self.fill[i, j])
            ti.atomic_add(self.nx_mean[j], self.nx[i, j])
        self.cw_max[None] = 1e-6
        for c in range(res):
            self.nx_mean[c] = self.nx_mean[c] / res
            ti.atomic_max(self.cw_max[None], self.col_water[c])

    @ti.kernel
    def _focus_pre(self, nxb: ti.template()):
        # slope of the (blurred) column-mean normal x; times normalized water column.
        res = self.res
        for c in range(res):
            c0 = ti.max(c - 1, 0); c1 = ti.min(c + 1, res - 1)
            slope = ti.abs((nxb[c1] - nxb[c0]) * 0.5)
            self.tmp1d[c] = (self.col_water[c] / (self.cw_max[None] + 1e-6)) * slope

    @ti.kernel
    def _focus_norm(self):
        res = self.res
        self.foc_max[None] = 1e-6
        for c in range(res):
            ti.atomic_max(self.foc_max[None], self.focus[c])
        for c in range(res):
            self.focus[c] = self.focus[c] / (self.foc_max[None] + 1e-6)

    @ti.kernel
    def _footprint_pre(self):
        for c in range(self.res):
            self.tmp1d[c] = 1.0 if self.col_water[c] > 3.0 else 0.0

    @ti.kernel
    def _caustics_contact(self):
        res = self.res
        floor_row = self.floor_row
        band_h = int(0.05 * res)
        cg = self.L["caustic_gain"]
        contact = self.L["contact"]
        ccol = ti.Vector([0.75, 0.85, 1.0])
        s_band = 2.0 * (band_h * 0.5) ** 2 + 1e-6
        s_cont = 2.0 * (0.02 * res) ** 2 + 1e-6
        for i, j in ti.ndrange(res, res):
            if cg > 0 and floor_row < res - 4:
                prof = ti.exp(-((i - floor_row) ** 2) / s_band)
                caustic = prof * self.focus[j] * cg
                self.img[i, j] = self.img[i, j] + caustic * ccol
            if contact > 0 and floor_row < res - 4:
                profc = ti.exp(-((i - (floor_row + 6)) ** 2) / s_cont)
                shadow = profc * self.footprint[j]
                self.img[i, j] = self.img[i, j] * (1.0 - contact * shadow)

    @ti.kernel
    def _reflection(self):
        res = self.res
        fy = self.floor_row
        refl_h = ti.min(fy, res - fy)
        for i, j in ti.ndrange(res, res):
            if fy <= i < fy + refl_h:
                k = i - fy
                src = 2 * fy - 1 - i
                if 0 <= src < res:
                    fade = (1.0 - k / (refl_h - 1.0 + 1e-9)) * 0.20
                    self.img[i, j] = self.img[i, j] * (1.0 - fade) + self.img[src, j] * fade

    # ------------------------------------------------------------------ finish: bloom, vignette, tone map
    @ti.kernel
    def _extract_bright(self, ch: ti.i32):
        for i, j in ti.ndrange(self.res, self.res):
            self.chan[i, j] = ti.max(self.img[i, j][ch] - 0.72, 0.0)

    @ti.kernel
    def _add_bloom(self, ch: ti.i32, gain: ti.f32):
        for i, j in ti.ndrange(self.res, self.res):
            self.img[i, j][ch] = self.img[i, j][ch] + gain * self.chanb[i, j]

    @ti.kernel
    def _tonemap(self):
        res = self.res
        cc = res / 2.0
        vig = self.L["vignette"]
        for i, j in ti.ndrange(res, res):
            r2 = ((j - cc) / res) ** 2 + ((i - cc) / res) ** 2
            v = 1.0 - vig * _smoothstep(0.12, 0.42, r2)
            col = self.img[i, j] * v
            col = col / (col + 0.9)
            for ch in ti.static(range(3)):
                val = ti.min(ti.max(col[ch] * 1.55, 0.0), 1.0) ** (1.0 / 1.15)
                self.out[i, j, ch] = ti.cast(ti.min(ti.max(val, 0.0), 1.0) * 255.0, ti.u8)

    # ------------------------------------------------------------------ per-frame driver
    def render(self, pos, speed, frame=0):
        n = pos.shape[0]
        buf = np.zeros((self.MAXP, 2), np.float32)
        buf[:n] = pos.astype(np.float32)
        self.pp.from_numpy(buf)
        sbuf = np.zeros(self.MAXP, np.float32)
        sbuf[:n] = speed.astype(np.float32)
        self.psp.from_numpy(sbuf)
        self._render_device(n, frame)
        return self.out.to_numpy()

    def _render_device(self, n, frame):
        """All-on-device rendering (no host roundtrips). Split out so the benchmark can time exactly the
        GPU compute with the particle data already resident."""
        res = self.res
        self._build_masks(n)
        # normals
        self._blur(self.fill, self.bt2, 2.5)
        shift = (frame * 5) % res
        self._make_base_field(self.bt2, shift)
        self._blur(self.ds, self.bt2, 1.6)
        # copy blurred field back into ds for the gradient
        self._copy_scalar(self.bt2, self.ds)
        self._normals()
        # refraction
        self._blur(self.thick, self.bt2, 2.0)
        self._refraction(self.bt2)
        # shading
        self._shade()
        # foam: numerator = blur(speed splat); denominator = blur(count) which is exactly ``dens`` already
        # computed at the top of the mask pass, so it is reused rather than re-blurred.
        self._blur(self.spsum, self.bt2, self.L["sigma_px"])   # blurred speed splat
        self._copy_scalar(self.bt2, self.spsum)
        self._blur(self.fill, self.bt3, 3.0)                   # edge field
        self._foam_build(self.dens, self.bt3)                  # avgden = dens = blur(count)
        self._blur(self.foam, self.bt2, 1.8)
        self._copy_scalar(self.bt2, self.foam)
        self._foam_apply()
        # composite liquid over background
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
        # floor reflection
        self._reflection()
        # bloom
        bloom_sigma = res / 130.0
        for ch in range(3):
            self._extract_bright(ch)
            self._blur(self.chan, self.chanb, bloom_sigma)
            self._add_bloom(ch, self.L["bloom"])
        # tone map -> uint8
        self._tonemap()
        ti.sync()

    @ti.kernel
    def _copy_scalar(self, src: ti.template(), dst: ti.template()):
        for i, j in ti.ndrange(self.res, self.res):
            dst[i, j] = src[i, j]

    @ti.kernel
    def _copy_1d(self, src: ti.template(), dst: ti.template()):
        for i in range(self.res):
            dst[i] = src[i]


# =========================================================================== driver / benchmark
def _sim_scene(scene_name, n_frames, quick=False):
    cfg = fr2.SCENES[scene_name](quick=quick)
    xs, vs, stable = fr2.run_scene(cfg["pts"], cfg["E"], cfg["dt"], cfg["T"], n_frames,
                                   fric=cfg["fric"], area=cfg["area"], flip=cfg["flip"])
    return xs, vs, stable, cfg


def benchmark(renderer, xs, vs, frame_idx, gpu_iters=50, cpu_iters=3, tank=None, L=None):
    """Time CPU (fr2.render_frame) vs GPU on the SAME particle data at the renderer's resolution."""
    pos = xs[frame_idx]
    vel = vs[frame_idx]
    speed = np.linalg.norm(vel, axis=1)

    # ---- GPU: warm up (compiles all kernels), then time steady-state ----
    t_compile0 = time.perf_counter()
    _ = renderer.render(pos, speed, frame=frame_idx)
    ti.sync()
    compile_s = time.perf_counter() - t_compile0
    # a couple more warmups to settle clocks
    for _ in range(3):
        renderer.render(pos, speed, frame=frame_idx)
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(gpu_iters):
        renderer.render(pos, speed, frame=frame_idx)
    ti.sync()
    gpu_ms = (time.perf_counter() - t0) / gpu_iters * 1000.0

    # time device-only (particle data resident) to separate upload from compute
    n = pos.shape[0]
    buf = np.zeros((renderer.MAXP, 2), np.float32); buf[:n] = pos.astype(np.float32)
    renderer.pp.from_numpy(buf)
    sbuf = np.zeros(renderer.MAXP, np.float32); sbuf[:n] = speed.astype(np.float32)
    renderer.psp.from_numpy(sbuf)
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(gpu_iters):
        renderer._render_device(n, frame_idx)
    ti.sync()
    gpu_dev_ms = (time.perf_counter() - t0) / gpu_iters * 1000.0

    # ---- CPU reference ----
    _ = fr2.render_frame(pos, vel, tank, L, frame=frame_idx)   # warm caches
    t0 = time.perf_counter()
    for _ in range(cpu_iters):
        fr2.render_frame(pos, vel, tank, L, frame=frame_idx)
    cpu_ms = (time.perf_counter() - t0) / cpu_iters * 1000.0

    return {
        "res": renderer.res,
        "n_particles": int(n),
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
        "gpu_dev_ms": gpu_dev_ms,
        "speedup": cpu_ms / gpu_ms,
        "speedup_dev": cpu_ms / gpu_dev_ms,
        "compile_s": compile_s,
    }


def build_side_by_side(cpu_img, gpu_img, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.0), facecolor="#0a0e14")
    axes[0].imshow(cpu_img); axes[0].set_title("CPU  (numpy/scipy, fluid_render2.py)",
                                               color="#e6a15a", fontsize=15, pad=10)
    axes[1].imshow(gpu_img); axes[1].set_title("GPU  (Taichi, fluid_render_gpu.py)",
                                               color="#7fd0ff", fontsize=15, pad=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#26313d")
    fig.suptitle(title, color="#dfe6ee", fontsize=17, y=0.98)
    fig.subplots_adjust(left=0.006, right=0.994, top=0.9, bottom=0.01, wspace=0.03)
    fig.savefig(out_path, dpi=95, facecolor="#0a0e14")
    plt.close(fig)


def build_bench_figure(bench, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor="#0a0e14")
    ax.set_facecolor("#0a0e14")
    labels = ["CPU\n(numpy/scipy)", "GPU\n(Taichi, full)", "GPU\n(device-only)"]
    vals = [bench["cpu_ms"], bench["gpu_ms"], bench["gpu_dev_ms"]]
    colors = ["#e6a15a", "#7fd0ff", "#5bd6a0"]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(min(vals) * 0.45, max(vals) * 6.0)      # headroom so labels clear the title
    ax.set_ylabel("milliseconds per 1080$^2$ frame (log scale)", color="#dfe6ee")
    ax.set_title(f"CPU vs GPU render time  -  {bench['speedup']:.0f}x faster "
                 f"({bench['n_particles']} particles, res {bench['res']})",
                 color="#dfe6ee", fontsize=13, pad=16)
    for b, v in zip(bars, vals):
        fps = 1000.0 / v
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.1f} ms\n{fps:.0f} fps",
                ha="center", va="bottom", color="#dfe6ee", fontsize=11)
    ax.tick_params(colors="#9fb0c0")
    for s in ax.spines.values():
        s.set_color("#26313d")
    ax.grid(True, axis="y", color="#1a222c", lw=0.6)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.14)
    fig.savefig(out_path, dpi=110, facecolor="#0a0e14")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="low-res smoke test")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--nframes", type=int, default=None)
    ap.add_argument("--close-r", type=int, default=10)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo, "runs", "realistic-rendering", "gpu-accelerate-fluid-renderer")
    os.makedirs(out_dir, exist_ok=True)

    res = 384 if args.quick else 1080
    n_frames = args.nframes or (40 if args.quick else 130)

    L = dict(fr2.DEFAULT_LOOK)
    L["res"] = res
    tank = fr2.make_tank(res, L)

    print(f"[sim] rolling balldrop + dambreak ({n_frames} frames each) ...")
    xs_bd, vs_bd, st_bd, cfg_bd = _sim_scene("balldrop", n_frames, quick=args.quick)
    xs_db, vs_db, st_db, cfg_db = _sim_scene("dambreak", n_frames, quick=args.quick)
    print(f"[sim] stable balldrop={st_bd} dambreak={st_db}")

    print(f"[gpu] building renderer at res={res} ...")
    renderer = GPUFluidRenderer(res, L, close_r=args.close_r)
    renderer.upload_tank(tank)

    # ---------------- benchmark ----------------
    hero_db = min(118 if not args.quick else 30, n_frames - 1)
    hero_bd = min(58 if not args.quick else 30, n_frames - 1)
    print("[bench] timing CPU vs GPU on the dam-break hero frame ...")
    bench = benchmark(renderer, xs_db, vs_db, hero_db,
                      gpu_iters=30 if not args.quick else 20,
                      cpu_iters=3 if not args.quick else 2, tank=tank, L=L)
    print(f"[bench] res={bench['res']} N={bench['n_particles']}  "
          f"CPU {bench['cpu_ms']:.1f} ms  GPU {bench['gpu_ms']:.2f} ms "
          f"(device-only {bench['gpu_dev_ms']:.2f} ms)  speedup {bench['speedup']:.0f}x  "
          f"compile {bench['compile_s']:.1f}s")

    if args.bench_only:
        return

    # ---------------- visual parity stills ----------------
    print("[parity] rendering CPU and GPU stills for side-by-side ...")
    for scene, xs, vs, hero, label in [
            ("dambreak", xs_db, vs_db, hero_db, "Dam break (breaking-wave barrel)"),
            ("balldrop", xs_bd, vs_bd, hero_bd, "Ball drop (peak splash crown)")]:
        pos, vel = xs[hero], vs[hero]
        speed = np.linalg.norm(vel, axis=1)
        cpu_img = fr2.render_frame(pos, vel, tank, L, frame=hero)
        gpu_img = renderer.render(pos, speed, frame=hero)
        fr2.save_png(os.path.join(out_dir, f"{scene}_cpu.png"), cpu_img)
        fr2.save_png(os.path.join(out_dir, f"{scene}_gpu.png"), gpu_img)
        build_side_by_side(cpu_img, gpu_img,
                           os.path.join(out_dir, f"cpu_vs_gpu_{scene}.png"),
                           f"{label}  -  same particles, same formulas, {bench['speedup']:.0f}x faster on GPU")

    build_bench_figure(bench, os.path.join(out_dir, "benchmark.png"))

    # ---------------- GPU clip ----------------
    print("[clip] rendering full GPU clip (dam break) ...")
    t0 = time.perf_counter()
    frames = []
    for f in range(n_frames):
        speed = np.linalg.norm(vs_db[f], axis=1)
        frames.append(renderer.render(xs_db[f], speed, frame=f))
    ti.sync()
    render_wall = time.perf_counter() - t0
    print(f"[clip] rendered {n_frames} GPU frames in {render_wall:.2f}s "
          f"({render_wall / n_frames * 1000:.1f} ms/frame incl. host loop)")
    t0 = time.perf_counter()
    fr2.encode_mp4(os.path.join(out_dir, "dambreak_gpu.mp4"), frames, fps=cfg_db["fps"])
    encode_wall = time.perf_counter() - t0
    print(f"[clip] encoded mp4 in {encode_wall:.2f}s (I/O + codec bound)")

    # also a balldrop clip
    frames_bd = []
    for f in range(n_frames):
        speed = np.linalg.norm(vs_bd[f], axis=1)
        frames_bd.append(renderer.render(xs_bd[f], speed, frame=f))
    ti.sync()
    fr2.encode_mp4(os.path.join(out_dir, "balldrop_gpu.mp4"), frames_bd, fps=cfg_bd["fps"])

    # save timing summary for the manifest builder
    import json
    summary = dict(bench)
    summary["n_frames"] = n_frames
    summary["render_wall_s"] = render_wall
    summary["encode_wall_s"] = encode_wall
    summary["scene_render_ms_gpu"] = bench["gpu_dev_ms"] * n_frames
    summary["scene_render_s_cpu"] = bench["cpu_ms"] * n_frames / 1000.0
    with open(os.path.join(out_dir, "bench.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[done] wrote stills, clips, benchmark figure, bench.json to", out_dir)


if __name__ == "__main__":
    main()
