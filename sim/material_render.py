"""Rendering PROPOSALS for the four canonical materials — water, rubber, snow, sand.

The complaint this file answers: in the shipped demo all four materials are drawn by exactly one
shader (a soft particle splat resolved into an iso-surface) and therefore differ **only in hue**. The
treatment is roughly right for snow and wrong for everything else. This module proposes a distinct
treatment per material and renders each one **against the current look, same scene, same seed**.

SCOPE. This file renders; it never simulates its own physics and never touches the demo. Motion comes
from ``sim.physics`` unchanged (a forward sim; no gradients are needed anywhere here). The demo's
WebGSL renderer in ``harness/dashboard/src/components/mpm/mpm4.js`` is READ, reproduced here as the
mandatory baseline, and left untouched.

WHAT IS REPRODUCED, AND WHY IT IS FAITHFUL
  * ``T_CURRENT`` is a line-by-line port of the demo's ``fs_splat`` + ``fs_resolve`` pair: the same
    compact kernel w = (1-r^2)^2, the same additive (colour*w, w) accumulation, the same central-
    difference normal with nz = 1.6*iso, the same diffuse/specular/edge terms and the same rim
    brightening. The demo runs radius = 0.034 in NDC, i.e. 0.017 of the unit domain, and iso = 2.6.
  * Every scene is seeded at the demo's OWN areal particle density,
    ``DENSITY = 500 / (pi * 0.075^2) ~= 28294`` particles per unit area (demo4.js), because the
    accumulated weight `a` that the iso threshold is compared against scales with that density. A
    proposal that only looks good at 8x the demo's particle count is not a proposal for this demo.

The water treatment is NOT invented here. It is a port of the screen-space iso-surface pipeline this
repo already built and measured in ``sim/fluid_render2.py`` / ``sim/fluid_render_gpu.py`` (runs
``realistic-rendering/*``): metaball density -> filled interior mask (opacity) separated from the
density gradient (normals), a distance-transform optical thickness, Beer-Lambert depth colour,
background refraction with chromatic dispersion, Fresnel, specular, surface-gated foam.

Usage:
    python sim/material_render.py --smoke     # one frame of every treatment, for eyeballing
    python sim/material_render.py             # full deliverable: all clips, stills, greyscale, costs
    python sim/material_render.py --bench     # just the per-treatment frame-cost measurement
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import taichi as ti  # noqa: E402
import sim.physics as physics  # noqa: E402  (this triggers ti.init inside sim.physics.core)

MAT = physics.MAT
MAT_ID = physics.MAT_ID
ORDER = ["fluid", "elastic", "snow", "sand"]
LABEL = {"fluid": "WATER", "elastic": "RUBBER", "snow": "SNOW", "sand": "SAND"}

# demo4.js: DENSITY = 500 / (pi * 0.075^2). One areal density for the whole domain.
DEMO_DENSITY = 500.0 / (np.pi * 0.075 ** 2)
# mpm4.js render params, as demo4.js calls them: radius 0.034 NDC (= 0.017 of the unit domain), iso 2.6
DEMO_RADIUS = 0.017
DEMO_ISO = 2.6

# Render resolution. Overridable from the environment so the cost measurement can be run at
# several resolutions in separate processes -- the screen-space passes scale with PIXELS, not
# with particles, and that is the single most important fact for fitting a frame budget.
RES = int(os.environ.get("MR_RES", "720"))
MAXP = 20000
f32 = ti.f32

# ============================================================================ fields
pp = ti.Vector.field(2, f32, MAXP)          # position in [0,1]^2
pmt = ti.field(ti.i32, MAXP)                # material id
psp = ti.field(f32, MAXP)                   # speed
puv = ti.Vector.field(2, f32, MAXP)         # material coordinates (rest position)

_sc = lambda: ti.field(f32, (RES, RES))     # noqa: E731
_v3 = lambda: ti.Vector.field(3, f32, (RES, RES))  # noqa: E731
_v2 = lambda: ti.Vector.field(2, f32, (RES, RES))  # noqa: E731

acc3 = _v3()      # demo baseline: sum(colour * w)
accw = _sc()      # demo baseline: sum(w)
dens = _sc()      # raw particle histogram
densb = _sc()     # blurred density (metaball field)
tmpa = _sc()
tmpb = _sc()
tmpc = _sc()
fillm = _sc()     # feathered opacity mask in [0,1]
bodym = _sc()     # binary interior (0/1)
thick = _sc()     # optical thickness from the distance transform
nrm = _v3()       # surface normal
spd = _sc()       # speed splat
spdb = _sc()
uacc = _v2()      # sum(material-coords * w)
uf = _v2()        # material-coordinate field
prio = _sc()      # sand: winning grain's priority hash (-1 = no grain)
gcov = _sc()      # sand: grain coverage (AA)
distr = _sc()     # UNBLURRED distance to the nearest outside pixel (a crisp border needs a crisp d)
img = _v3()       # working image
bg = _v3()        # static background
beneath = _v3()   # what a transparent layer sees behind it
layer = _v3()     # one material's shaded colour
lalpha = _sc()    # one material's coverage
noise1 = _sc()    # low-frequency value noise (ripple / mottle)
noise2 = _sc()    # high-frequency value noise (foam / sparkle)
sa = ti.Vector.field(2, ti.i32, (RES, RES))   # jump-flooding buffers
sb = ti.Vector.field(2, ti.i32, (RES, RES))

# palette, host-settable so the greyscale test can neutralise every albedo without touching shading
# 0..3 material base colour | 4 water deep | 5 water shallow | 6 foam | 7 rubber outline
# 8 rubber inner bevel | 9 snow sparkle | 10 env sky | 11 env ground
PAL = ti.Vector.field(3, f32, 16)

NSIG, RMAX = 12, 48
gker = ti.field(f32, (NSIG, 2 * RMAX + 1))
grad_ = ti.field(ti.i32, NSIG)
_sig_ids = {}


def sigma_id(sigma):
    """Register a Gaussian sigma (in pixels) and return its slot. Built once, reused every frame, so
    no host->device kernel upload happens on the steady-state path."""
    key = round(float(sigma), 3)
    if key in _sig_ids:
        return _sig_ids[key]
    sid = len(_sig_ids)
    if sid >= NSIG:
        raise RuntimeError("out of Gaussian slots")
    r = int(min(RMAX, max(1, np.ceil(4.0 * key))))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / max(key, 1e-6)) ** 2)
    k /= k.sum()
    row = np.zeros(2 * RMAX + 1, np.float32)
    row[RMAX - r:RMAX + r + 1] = k
    host = gker.to_numpy()
    host[sid] = row
    gker.from_numpy(host)
    grad_[sid] = r
    _sig_ids[key] = sid
    return sid


# ============================================================================ small helpers
@ti.func
def hash1(n):
    x = ti.sin(n * 127.1 + 311.7) * 43758.5453
    return x - ti.floor(x)


@ti.func
def smoothstep(a, b, x):
    t = ti.math.clamp((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.kernel
def clear_s(fld: ti.template(), val: f32):
    for i, j in fld:
        fld[i, j] = val


@ti.kernel
def clear_v3(fld: ti.template(), r: f32, g: f32, b: f32):
    for i, j in fld:
        fld[i, j] = ti.Vector([r, g, b])


@ti.kernel
def clear_v2(fld: ti.template()):
    for i, j in fld:
        fld[i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def copy_v3(src: ti.template(), dst: ti.template()):
    for i, j in src:
        dst[i, j] = src[i, j]


@ti.kernel
def blur_h(src: ti.template(), dst: ti.template(), sid: ti.i32):
    r = grad_[sid]
    for i, j in ti.ndrange(RES, RES):
        s = 0.0
        for k in range(-r, r + 1):
            s += gker[sid, k + RMAX] * src[ti.min(ti.max(i + k, 0), RES - 1), j]
        dst[i, j] = s


@ti.kernel
def blur_v(src: ti.template(), dst: ti.template(), sid: ti.i32):
    r = grad_[sid]
    for i, j in ti.ndrange(RES, RES):
        s = 0.0
        for k in range(-r, r + 1):
            s += gker[sid, k + RMAX] * src[i, ti.min(ti.max(j + k, 0), RES - 1)]
        dst[i, j] = s


def blur(src, dst, sigma, tmp):
    sid = sigma_id(sigma)
    blur_h(src, tmp, sid)
    blur_v(tmp, dst, sid)


@ti.kernel
def blur3_h(src: ti.template(), dst: ti.template(), sid: ti.i32):
    r = grad_[sid]
    for i, j in ti.ndrange(RES, RES):
        s = ti.Vector([0.0, 0.0, 0.0])
        for k in range(-r, r + 1):
            s += gker[sid, k + RMAX] * src[ti.min(ti.max(i + k, 0), RES - 1), j]
        dst[i, j] = s


@ti.kernel
def blur3_v(src: ti.template(), dst: ti.template(), sid: ti.i32):
    r = grad_[sid]
    for i, j in ti.ndrange(RES, RES):
        s = ti.Vector([0.0, 0.0, 0.0])
        for k in range(-r, r + 1):
            s += gker[sid, k + RMAX] * src[i, ti.min(ti.max(j + k, 0), RES - 1)]
        dst[i, j] = s


@ti.kernel
def morph(src: ti.template(), dst: ti.template(), r: ti.i32, is_max: ti.i32):
    """Grey dilation (is_max=1) / erosion (is_max=0) over a DISK of radius r. A disk rather than a
    square because a square structuring element leaves visibly boxed corners on a silhouette."""
    for i, j in ti.ndrange(RES, RES):
        best = -1e18 if is_max == 1 else 1e18
        for dj in range(-r, r + 1):
            w = int(ti.sqrt(ti.max(0.0, float(r * r - dj * dj))))
            for di in range(-w, w + 1):
                v = src[ti.min(ti.max(i + di, 0), RES - 1), ti.min(ti.max(j + dj, 0), RES - 1)]
                best = ti.max(best, v) if is_max == 1 else ti.min(best, v)
        dst[i, j] = best


# ---- jump flooding distance transform ------------------------------------------------------------
@ti.kernel
def jfa_seed(mask: ti.template(), inside: ti.i32):
    """Seed every pixel that is NOT part of `mask` (inside=1) so the resulting distance is 'distance
    to the nearest pixel outside the body' — exactly scipy's distance_transform_edt(body)."""
    for i, j in ti.ndrange(RES, RES):
        m = mask[i, j] > 0.5
        seed = (not m) if inside == 1 else m
        sa[i, j] = ti.Vector([i, j]) if seed else ti.Vector([-1, -1])


@ti.kernel
def jfa_pass(step: ti.i32):
    for i, j in ti.ndrange(RES, RES):
        best = sa[i, j]
        bd = 1e18
        if best[0] >= 0:
            bd = float((best[0] - i) ** 2 + (best[1] - j) ** 2)
        for dy in ti.static(range(-1, 2)):
            for dx in ti.static(range(-1, 2)):
                ii, jj = i + dx * step, j + dy * step
                if 0 <= ii < RES and 0 <= jj < RES:
                    c = sa[ii, jj]
                    if c[0] >= 0:
                        d = float((c[0] - i) ** 2 + (c[1] - j) ** 2)
                        if d < bd:
                            bd = d
                            best = c
        sb[i, j] = best
    for i, j in ti.ndrange(RES, RES):
        sa[i, j] = sb[i, j]


@ti.kernel
def jfa_dist(dst: ti.template()):
    for i, j in ti.ndrange(RES, RES):
        c = sa[i, j]
        dst[i, j] = 0.0 if c[0] < 0 else ti.sqrt(float((c[0] - i) ** 2 + (c[1] - j) ** 2))


def dist_transform(mask, dst, inside=1):
    jfa_seed(mask, inside)
    s = 1
    while s < RES:
        s *= 2
    s //= 2
    while s >= 1:
        jfa_pass(s)
        s //= 2
    jfa_dist(dst)


# ============================================================================ splatting
@ti.kernel
def splat_hist(n: ti.i32, sel: ti.i32, dst: ti.template()):
    """One unit of mass into the pixel a particle lands in. Blurring this is identical to summing a
    Gaussian metaball on every particle, and far cheaper (fluid_render2.splat_field)."""
    for p in range(n):
        if sel < 0 or pmt[p] == sel:
            i = ti.min(ti.max(int(pp[p][0] * RES), 0), RES - 1)
            j = ti.min(ti.max(int(pp[p][1] * RES), 0), RES - 1)
            dst[i, j] += 1.0


@ti.kernel
def splat_hist_w(n: ti.i32, sel: ti.i32, dst: ti.template()):
    for p in range(n):
        if sel < 0 or pmt[p] == sel:
            i = ti.min(ti.max(int(pp[p][0] * RES), 0), RES - 1)
            j = ti.min(ti.max(int(pp[p][1] * RES), 0), RES - 1)
            dst[i, j] += psp[p]


@ti.kernel
def splat_disc(n: ti.i32, sel: ti.i32, rpx: f32):
    """The demo's pass A, verbatim: additive (colour*w, w) with w = (1-r^2)^2 over a disc."""
    for p in range(n):
        if sel < 0 or pmt[p] == sel:
            cx, cy = pp[p][0] * RES, pp[p][1] * RES
            col = PAL[pmt[p]]
            lo_i = ti.max(int(cx - rpx) - 1, 0)
            hi_i = ti.min(int(cx + rpx) + 1, RES - 1)
            lo_j = ti.max(int(cy - rpx) - 1, 0)
            hi_j = ti.min(int(cy + rpx) + 1, RES - 1)
            for i in range(lo_i, hi_i + 1):
                for j in range(lo_j, hi_j + 1):
                    dx = (i + 0.5 - cx) / rpx
                    dy = (j + 0.5 - cy) / rpx
                    r2 = dx * dx + dy * dy
                    if r2 < 1.0:
                        w = (1.0 - r2) * (1.0 - r2)
                        acc3[i, j] += col * w
                        accw[i, j] += w


@ti.kernel
def splat_uv(n: ti.i32, sel: ti.i32, rpx: f32):
    """Material coordinates, weight-averaged with the same compact kernel. Dividing by the weight
    gives a smooth field u(x) whose level sets are painted ON the material and therefore stretch and
    shear with it — the cue that makes a body read as ONE object rather than a cloud of dots."""
    for p in range(n):
        if sel < 0 or pmt[p] == sel:
            cx, cy = pp[p][0] * RES, pp[p][1] * RES
            lo_i = ti.max(int(cx - rpx) - 1, 0)
            hi_i = ti.min(int(cx + rpx) + 1, RES - 1)
            lo_j = ti.max(int(cy - rpx) - 1, 0)
            hi_j = ti.min(int(cy + rpx) + 1, RES - 1)
            for i in range(lo_i, hi_i + 1):
                for j in range(lo_j, hi_j + 1):
                    dx = (i + 0.5 - cx) / rpx
                    dy = (j + 0.5 - cy) / rpx
                    r2 = dx * dx + dy * dy
                    if r2 < 1.0:
                        w = (1.0 - r2) * (1.0 - r2)
                        uacc[i, j] += puv[p] * w
                        accw[i, j] += w


@ti.kernel
def resolve_uv():
    for i, j in uacc:
        w = accw[i, j]
        uf[i, j] = uacc[i, j] / w if w > 1e-6 else ti.Vector([0.0, 0.0])


# ============================================================================ T_CURRENT (the demo)
@ti.kernel
def resolve_current(iso: f32, off: ti.i32):
    """The demo's pass B, verbatim (mpm4.js fs_resolve), writing colour+alpha into layer/lalpha."""
    for i, j in accw:
        a = accw[i, j]
        if a < iso * 0.34:
            lalpha[i, j] = 0.0
            layer[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            base = acc3[i, j] / ti.max(a, 1e-6)
            lft = accw[ti.max(i - off, 0), j]
            rgt = accw[ti.min(i + off, RES - 1), j]
            dwn = accw[i, ti.min(j + off, RES - 1)]
            up_ = accw[i, ti.max(j - off, 0)]
            nv = ti.Vector([lft - rgt, dwn - up_, 1.6 * iso]).normalized()
            ld = ti.Vector([-0.42, 0.62, 0.66]).normalized()
            dif = 0.66 + 0.34 * ti.max(0.0, nv.dot(ld))
            h = (ld + ti.Vector([0.0, 0.0, 1.0])).normalized()
            spec = ti.pow(ti.max(0.0, nv.dot(h)), 26.0)
            edge = smoothstep(iso * 0.34, iso * 1.25, a)
            col = base * dif + ti.Vector([0.9, 0.97, 1.0]) * spec * 0.42
            col = col * 1.30 * (1.0 - edge) + col * edge
            layer[i, j] = col
            lalpha[i, j] = ti.min(1.0, 0.30 + 1.6 * edge)


def render_current(n, sel, iso=DEMO_ISO):
    clear_v3(acc3, 0.0, 0.0, 0.0)
    clear_s(accw, 0.0)
    splat_disc(n, sel, DEMO_RADIUS * RES)
    resolve_current(iso, max(1, int(round(2.0 * RES / 900.0))))


# ============================================================================ shared reconstruction
def build_masks(n, sel, ref, sigma, iso_fill, close_r, edge=0.09, round_sigma=0.0):
    """Density -> filled interior mask -> feathered opacity -> distance-transform thickness.

    The separation is the load-bearing idea from the fluid-rendering lineage: ONE threshold cannot
    both decide 'is there material here' (wants a generous threshold, and repair of the Poisson
    pinholes a random particle sample leaves) and 'which way does the surface face' (wants the fine
    slope information). So the filled mask owns opacity and the density gradient owns normals.
    `ref` is the blurred count a fully-packed region reaches: areal density / RES^2. Using that fixed
    physical reference instead of a per-frame percentile keeps a thin sheet of spray reading as thin.
    """
    clear_s(dens, 0.0)
    splat_hist(n, sel, dens)
    blur(dens, densb, sigma, tmpa)
    clear_s(tmpb, 0.0)
    normalize_by(densb, ref)
    morph(densb, tmpb, 3, 1)          # grey close: dilate ...
    morph(tmpb, tmpc, 3, 0)           # ... then erode, sealing sub-spacing pinholes
    threshold(tmpc, bodym, iso_fill)
    if close_r > 0:
        morph(bodym, tmpb, close_r, 1)
        morph(tmpb, bodym, close_r, 0)
    if round_sigma > 0.0:
        # blur-and-rethreshold: a low-pass on the SILHOUETTE. Curvature above the cutoff is removed,
        # so a lumpy particle boundary becomes one smooth closed outline — which is what makes a body
        # read as a single object instead of a cluster.
        blur(bodym, tmpb, round_sigma, tmpa)
        threshold(tmpb, bodym, 0.5)
    blur(bodym, tmpb, 2.0, tmpa)
    feather(tmpb, fillm, edge)
    dist_transform(bodym, distr, inside=1)
    blur(distr, thick, 2.0, tmpa)


@ti.kernel
def normalize_by(fld: ti.template(), ref: f32):
    for i, j in fld:
        fld[i, j] = fld[i, j] / (ref + 1e-9)


@ti.kernel
def threshold(src: ti.template(), dst: ti.template(), t: f32):
    for i, j in src:
        dst[i, j] = 1.0 if src[i, j] > t else 0.0


@ti.kernel
def feather(src: ti.template(), dst: ti.template(), e: f32):
    for i, j in src:
        dst[i, j] = smoothstep(0.5 - e, 0.5 + e, src[i, j])


@ti.kernel
def normals_from(fld: ti.template(), amp: f32, k: f32):
    for i, j in fld:
        gx = 0.5 * (fld[ti.min(i + 1, RES - 1), j] - fld[ti.max(i - 1, 0), j])
        gy = 0.5 * (fld[i, ti.min(j + 1, RES - 1)] - fld[i, ti.max(j - 1, 0)])
        nrm[i, j] = ti.Vector([-gx * amp, -gy * amp, k]).normalized()


@ti.func
def sample3(fld: ti.template(), fx: f32, fy: f32):
    x = ti.math.clamp(fx, 0.0, RES - 1.001)
    y = ti.math.clamp(fy, 0.0, RES - 1.001)
    i0, j0 = int(x), int(y)
    tx, ty = x - i0, y - j0
    a = fld[i0, j0] * (1 - tx) + fld[i0 + 1, j0] * tx
    b = fld[i0, j0 + 1] * (1 - tx) + fld[i0 + 1, j0 + 1] * tx
    return a * (1 - ty) + b * ty


# ============================================================================ WATER
@ti.kernel
def shade_water(absorb: f32, refract: f32, chroma: f32, spec_gain: f32, shininess: f32,
                rim: f32, foam_gain: f32, thick_char: f32, thick_max: f32, glassy: ti.i32):
    """Screen-space iso-surface water, ported from sim/fluid_render2.render_frame.

    Beer-Lambert depth colour over a REFRACTED background, Fresnel-weighted environment reflection,
    a grazing rim, a tight Blinn-Phong glint, and foam gated to the thin/fast surface band. `glassy`
    selects the full treatment (refraction + dispersion, option A) or the cheap one (option B: no
    background sampling at all, so the body is tinted rather than see-through)."""
    for i, j in fillm:
        m = fillm[i, j]
        if m <= 0.0:
            lalpha[i, j] = 0.0
            layer[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            nv = nrm[i, j]
            tt = ti.math.clamp(thick[i, j] / thick_char, 0.0, thick_max)
            refr = beneath[i, j]
            if glassy == 1:
                off = refract * ti.min(tt, 1.6)
                ox, oy = nv[0] * off, nv[1] * off
                r = sample3(beneath, i + ox * (1.0 + chroma), j + oy * (1.0 + chroma))[0]
                g = sample3(beneath, i + ox, j + oy)[1]
                b = sample3(beneath, i + ox * (1.0 - chroma), j + oy * (1.0 - chroma))[2]
                refr = ti.Vector([r, g, b])
            trans = ti.exp(-absorb * tt)
            col = refr * trans + (PAL[5] * trans + PAL[4] * (1.0 - trans)) * (1.0 - trans)
            col *= 1.0 - 0.16 * smoothstep(0.6, thick_max, tt)
            # Fresnel: cos(theta) IS nz, so the flat interior stays clear and the rim turns mirror
            cos_t = ti.math.clamp(nv[2], 0.0, 1.0)
            F = 0.02 + 0.98 * ti.pow(1.0 - cos_t, 5.0)
            sky = ti.math.clamp(0.55 + 0.45 * (2.0 * nv[2] * nv[1]), 0.0, 1.0)
            env = PAL[10] * sky + PAL[11] * (1.0 - sky)
            col = col * (1.0 - F) + env * F
            col += PAL[10] * (rim * ti.pow(1.0 - cos_t, 3.0))
            ld = ti.Vector([-0.55, 0.72, 0.55]).normalized()
            h = (ld + ti.Vector([0.0, 0.0, 1.0])).normalized()
            ndh = ti.math.clamp(nv.dot(h), 0.0, 1.0)
            col += ti.Vector([1.0, 1.0, 0.97]) * (spec_gain * ti.pow(ndh, shininess)
                                                  + 0.10 * ti.pow(ndh, 8.0))
            # foam: fast AND at the surface, or genuinely thin. Ungated foam speckles the interior.
            avg_sp = spdb[i, j] / (densb[i, j] + 1e-6)
            motion = smoothstep(1.7, 3.7, avg_sp)
            gx = 0.5 * (fillm[ti.min(i + 1, RES - 1), j] - fillm[ti.max(i - 1, 0), j])
            gy = 0.5 * (fillm[i, ti.min(j + 1, RES - 1)] - fillm[i, ti.max(j - 1, 0)])
            band = smoothstep(0.004, 0.02, ti.sqrt(gx * gx + gy * gy))
            thin = m * smoothstep(0.16, 0.02, tt)
            fo = ti.math.clamp(foam_gain * (0.9 * motion * band + 0.22 * thin), 0.0, 1.0)
            fo *= 1.0 - 0.5 * (1.0 - noise2[i, j])
            col = col * (1.0 - fo) + PAL[6] * fo
            layer[i, j] = col
            lalpha[i, j] = m


def render_water(n, sel, ref, glassy=True):
    sig = 5.0 if glassy else 6.5
    # iso_fill is deliberately LOW. It is the threshold that decides "is there water here", and
    # raising it is how the airborne spray a splash throws off quietly disappears from the frame --
    # the one thing the current splat renderer does better than a reconstruction.
    build_masks(n, sel, ref, sig, iso_fill=0.24, close_r=6 if glassy else 0)
    # normals: the smooth body field plus a low-amplitude interior ripple, so a flat slab of water
    # still has something to refract. Density noise never reaches the normal.
    blur(fillm, tmpc, 2.5, tmpa)
    ripple_into(0.22 if glassy else 0.0)
    blur(tmpb, tmpc, 1.6, tmpa)
    normals_from(tmpc, 3.0 * (RES / 1080.0), 2.1)
    clear_s(spd, 0.0)
    splat_hist_w(n, sel, spd)
    blur(spd, spdb, sig, tmpa)
    if glassy:
        shade_water(1.30, 60.0 * (RES / 1080.0), 0.35, 2.8, 90.0, 0.18, 0.95,
                    55.0 * (RES / 1080.0), 3.2, 1)
    else:
        # option B trades the two expensive cues (background refraction, chromatic dispersion) for a
        # much lower absorption, so instead of a deep body you get a thin, pale, clearly LIT liquid.
        shade_water(0.52, 0.0, 0.0, 3.4, 70.0, 0.34, 0.95,
                    55.0 * (RES / 1080.0), 3.2, 0)


@ti.kernel
def ripple_into(amp: f32):
    for i, j in fillm:
        base = tmpc[i, j] + 0.6 * ti.math.clamp(thick[i, j] / (55.0 * (RES / 1080.0)) / 3.2, 0.0, 1.0)
        if amp > 0.0:
            interior = smoothstep(0.2, 0.9, ti.math.clamp(thick[i, j] / (55.0 * (RES / 1080.0)) / 3.2, 0.0, 1.0))
            base += amp * (noise1[i, j] - 0.5) * interior
        tmpb[i, j] = base


# ============================================================================ RUBBER
@ti.kernel
def shade_rubber(border_px: f32, tex: f32, cell: f32, sheen: f32):
    """One coherent solid: a CONSTANT-WIDTH dark border from the signed distance to the silhouette,
    a flat interior that carries no particle-scale variation at all, a broad soft sheen, and (tex>0)
    a lattice painted in MATERIAL coordinates so the interior stretches and shears with the body."""
    for i, j in fillm:
        m = fillm[i, j]
        if m <= 0.0:
            lalpha[i, j] = 0.0
            layer[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            d = distr[i, j]                       # px from this pixel to the nearest outside pixel
            base = PAL[1]
            if tex > 0.0:
                u = uf[i, j]
                g = ti.max(ti.cos(u[0] * 6.2831853 / cell), ti.cos(u[1] * 6.2831853 / cell))
                base = base * (1.0 - tex * smoothstep(0.88, 1.0, g))
            nv = nrm[i, j]
            ld = ti.Vector([-0.42, 0.62, 0.66]).normalized()
            dif = 0.86 + 0.14 * ti.max(0.0, nv.dot(ld))
            h = (ld + ti.Vector([0.0, 0.0, 1.0])).normalized()
            ndh = ti.math.clamp(nv.dot(h), 0.0, 1.0)
            col = base * dif + ti.Vector([1.0, 0.96, 0.90]) * sheen * ti.pow(ndh, 10.0)
            # a broad inner bevel, then a HARD border of constant pixel width. Constant width is the
            # whole point: it is a drawn outline, so it does not thin out where the body is thin, and
            # the eye reads the closed line as the boundary of one object.
            bev = 1.0 - smoothstep(border_px, border_px * 5.0, d)
            col = col * (1.0 - 0.30 * bev) + PAL[8] * (0.30 * bev)
            edge = 1.0 - smoothstep(border_px * 0.55, border_px * 1.45, d)
            col = col * (1.0 - edge) + PAL[7] * edge
            layer[i, j] = col
            lalpha[i, j] = m


def render_rubber(n, sel, ref, textured=True):
    build_masks(n, sel, ref, 6.5, iso_fill=0.30, close_r=6, edge=0.055, round_sigma=5.0)
    if textured:
        clear_v2(uacc)
        clear_s(accw, 0.0)
        splat_uv(n, sel, DEMO_RADIUS * RES)
        resolve_uv()
    blur(fillm, tmpc, 4.0, tmpa)
    normals_from(tmpc, 1.4 * (RES / 1080.0), 0.55)
    shade_rubber(max(3.0, 0.0058 * RES), 0.24 if textured else 0.0, 0.062, 0.60)


# ============================================================================ SNOW
@ti.kernel
def shade_snow(iso: f32, off: ti.i32, sparkle: f32, halo: f32, crevice: f32, grain: f32):
    """Snow is the REFERENCE the others diverge from, so it is built on the same mushy splat the demo
    already uses rather than on a solid reconstruction. What it gains is only what a snow surface has
    and the current shader denies it: it is MATTE (the demo's tight glint is a wet-plastic cue), its
    edge is a soft powder fringe instead of a hard iso cut, thin snow glows because a strongly
    scattering medium leaks light through it, packed crevices sit in shadow, and the surface carries
    a sparse crystal sparkle."""
    for i, j in accw:
        a = accw[i, j]
        m = smoothstep(iso * 0.06, iso * 0.50, a)              # soft powder fringe, not a hard cut
        haze = halo * smoothstep(iso * 0.02, iso * 0.22, a) * (1.0 - m)
        if m <= 0.002 and haze <= 0.002:
            lalpha[i, j] = 0.0
            layer[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            lft = accw[ti.max(i - off, 0), j]
            rgt = accw[ti.min(i + off, RES - 1), j]
            dwn = accw[i, ti.min(j + off, RES - 1)]
            up_ = accw[i, ti.max(j - off, 0)]
            nv = ti.Vector([lft - rgt, dwn - up_, 1.6 * iso]).normalized()
            ld = ti.Vector([-0.42, 0.62, 0.66]).normalized()
            wrap = 0.74 + 0.26 * (0.5 + 0.5 * nv.dot(ld))      # multiple scattering flattens shading
            depth = ti.math.clamp(a / (iso * 3.3), 0.0, 1.0)
            translu = 1.0 + 0.30 * (1.0 - depth)               # thin fringe glows
            shadow = 1.0 - crevice * (1.0 - depth) * depth * 4.0
            col = PAL[2] * wrap * translu * shadow
            surf = smoothstep(0.90, 0.20, depth)
            # fine crystal grain, gated to the surface: snow IS granular, just at a much finer
            # scale than sand, and without it snow is only recognisable by elimination.
            col = col * (1.0 + grain * (noise2[i, j] - 0.5) * surf)
            col += PAL[9] * (sparkle * surf * smoothstep(0.80, 0.985, noise2[i, j]))
            layer[i, j] = col
            lalpha[i, j] = ti.min(1.0, ti.max(m, haze))


def render_snow(n, sel, ref, powder=True):
    clear_v3(acc3, 0.0, 0.0, 0.0)
    clear_s(accw, 0.0)
    splat_disc(n, sel, DEMO_RADIUS * RES)
    if powder:
        shade_snow(DEMO_ISO, max(1, int(round(2.0 * RES / 900.0))), 1.05, 0.42, 0.20, 0.40)
    else:
        shade_snow(DEMO_ISO, max(1, int(round(2.0 * RES / 900.0))), 0.0, 0.0, 0.0, 0.0)


# ============================================================================ SAND
@ti.kernel
def splat_grains(n: ti.i32, sel: ti.i32, k: ti.i32, rmean: f32, spread: f32):
    """K irregular grains per particle. Each grain gets its own jittered offset, radius, ellipse
    aspect and rotation from a hash of (particle, grain). Overlaps are resolved by an atomic max on
    the grain's random PRIORITY — and because the grain's shade is a deterministic function of that
    same priority, the winner's shade is recoverable in the resolve pass without a second buffer."""
    for p in range(n):
        if sel < 0 or pmt[p] == sel:
            for g in range(k):
                sd = float(p * 7 + g * 131)
                a0 = hash1(sd)
                a1 = hash1(sd + 17.3)
                a2 = hash1(sd + 41.9)
                a3 = hash1(sd + 63.1)
                a4 = hash1(sd + 91.7)
                ang = a0 * 6.2831853
                rad = spread * ti.sqrt(a1)
                cx = pp[p][0] * RES + rad * ti.cos(ang)
                cy = pp[p][1] * RES + rad * ti.sin(ang)
                rr = rmean * (0.55 + 0.95 * a2 * a2)      # skewed small: many fines, few coarse
                asp = 0.62 + 0.55 * a3
                th = a4 * 3.14159
                ct, st = ti.cos(th), ti.sin(th)
                ext = int(rr * 1.7) + 2
                lo_i = ti.max(int(cx) - ext, 0)
                hi_i = ti.min(int(cx) + ext, RES - 1)
                lo_j = ti.max(int(cy) - ext, 0)
                hi_j = ti.min(int(cy) + ext, RES - 1)
                pr = hash1(sd + 5.11)
                for i in range(lo_i, hi_i + 1):
                    for j in range(lo_j, hi_j + 1):
                        dx = i + 0.5 - cx
                        dy = j + 0.5 - cy
                        ex = (dx * ct + dy * st) / (rr * asp)
                        ey = (-dx * st + dy * ct) / (rr / asp)
                        r = ti.sqrt(ex * ex + ey * ey)
                        cov = ti.math.clamp((1.0 - r) * rr * 1.6 + 0.5, 0.0, 1.0)
                        if cov > 0.0:
                            ti.atomic_max(gcov[i, j], cov)
                            if cov > 0.5:
                                ti.atomic_max(prio[i, j], pr)


@ti.kernel
def shade_sand(under: ti.i32, ao: f32, thick_char: f32):
    """Per-grain albedo from the winning grain's priority hash, modulated by how deep in the pack the
    pixel sits (grains at the surface catch the light, grains inside the heap sit in shadow)."""
    for i, j in gcov:
        c = gcov[i, j]
        m = fillm[i, j]
        if c <= 0.0 and (under == 0 or m <= 0.0):
            lalpha[i, j] = 0.0
            layer[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            pr = prio[i, j]
            shade = 0.62 + 0.78 * hash1(pr * 91.7 + 3.3) if pr >= 0.0 else 0.70
            tt = ti.math.clamp(thick[i, j] / thick_char, 0.0, 1.0)
            depth = 1.0 - ao * tt
            nv = nrm[i, j]
            ld = ti.Vector([-0.42, 0.62, 0.66]).normalized()
            form = 0.74 + 0.34 * ti.max(0.0, nv.dot(ld))
            col = PAL[3] * shade * depth * form
            a = c
            if under == 1:
                body_col = PAL[3] * 0.42 * depth
                col = body_col * (1.0 - c) + col * c
                a = ti.max(c, m)
            layer[i, j] = col
            lalpha[i, j] = a


def render_sand(n, sel, ref, under=True):
    build_masks(n, sel, ref, 5.0, iso_fill=0.24, close_r=4, edge=0.10)
    blur(fillm, tmpc, 4.0, tmpa)
    normals_from(tmpc, 1.6 * (RES / 1080.0), 1.1)
    clear_s(gcov, 0.0)
    clear_s(prio, -1.0)
    # grain scale is tied to the particle spacing, so the pack stays opaque at the demo's density
    sp_px = np.sqrt(1.0 / DEMO_DENSITY) * RES
    splat_grains(n, sel, 6, float(0.46 * sp_px), float(1.05 * sp_px))
    shade_sand(1 if under else 0, 0.42, 30.0 * (RES / 1080.0))


# ============================================================================ compositing / output
@ti.kernel
def composite():
    for i, j in img:
        a = ti.math.clamp(lalpha[i, j], 0.0, 1.0)
        img[i, j] = img[i, j] * (1.0 - a) + layer[i, j] * a


@ti.kernel
def tonemap(grey: ti.i32, gain: f32):
    for i, j in img:
        c = img[i, j] * gain
        c = c / (c + 0.9)
        c = ti.math.clamp(c * 1.55, 0.0, 1.0) ** (1.0 / 1.15)
        if grey == 1:
            y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            c = ti.Vector([y, y, y])
        img[i, j] = c


def to_image():
    a = img.to_numpy()                      # [i=x, j=y, 3], y up
    a = np.transpose(a, (1, 0, 2))[::-1]    # -> [row=y down, col=x, 3]
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


# ============================================================================ palette / background
def _hex(c):
    return np.array([int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)], np.float32) / 255.0


NEUTRAL = np.array([0.68, 0.68, 0.68], np.float32)


def set_palette(grey=False):
    p = np.zeros((16, 3), np.float32)
    if grey:
        for k in range(4):
            p[k] = NEUTRAL
        p[4] = NEUTRAL * 0.14          # water deep
        p[5] = NEUTRAL * 0.62          # water shallow
        p[6] = np.array([0.96, 0.96, 0.96])
        p[7] = np.array([0.05, 0.05, 0.05])
        p[8] = np.array([1.0, 1.0, 1.0])
        p[9] = np.array([1.0, 1.0, 1.0])
        p[10] = np.array([0.74, 0.74, 0.74])
        p[11] = np.array([0.13, 0.13, 0.13])
    else:
        for k, m in enumerate(ORDER):
            p[k] = _hex(MAT[m]["color"])
        p[4] = np.array([0.02, 0.16, 0.30])
        p[5] = np.array([0.20, 0.50, 0.60])
        p[6] = np.array([0.93, 0.97, 1.0])
        p[7] = np.array([0.06, 0.03, 0.02])
        p[8] = np.array([1.0, 0.86, 0.72])
        p[9] = np.array([1.0, 1.0, 1.0])
        p[10] = np.array([0.60, 0.74, 0.90])
        p[11] = np.array([0.10, 0.13, 0.17])
    PAL.from_numpy(p)


def value_noise(res, cells, rng, octaves=3):
    out = np.zeros((res, res), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = cells * (2 ** o)
        base = rng.standard_normal((c + 1, c + 1)).astype(np.float32)
        ys = np.linspace(0, c, res, endpoint=False)
        xs = np.linspace(0, c, res, endpoint=False)
        y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
        fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
        a = base[np.ix_(y0, x0)]; b = base[np.ix_(y0, x0 + 1)]
        cc = base[np.ix_(y0 + 1, x0)]; d = base[np.ix_(y0 + 1, x0 + 1)]
        out += amp * (a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + cc * fy * (1 - fx) + d * fy * fx)
        total += amp
        amp *= 0.5
    out /= total
    return ((out - out.min()) / (np.ptp(out) + 1e-9)).astype(np.float32)


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def build_background(res, floor_y):
    """A dark studio backdrop with real structure in it. Structure is not decoration: refraction can
    only be SEEN if there is something behind the water to displace, and a flat wall refracts to a
    flat wall. Kept in the demo's dark register."""
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    vv = 1.0 - yy / res      # world y, 0 at bottom
    uu = xx / res
    top = np.array([0.030, 0.045, 0.062])
    bot = np.array([0.055, 0.075, 0.092])
    g = _ss(0.0, 1.0, vv)[..., None]
    b = bot * (1 - g) + top * g
    for cx, cy, wu, wv, inten, tint in [
            (0.18, 0.80, 0.10, 0.13, 0.085, (0.62, 0.80, 1.0)),
            (0.84, 0.84, 0.11, 0.13, 0.065, (1.0, 0.86, 0.66)),
            (0.50, 0.94, 0.22, 0.09, 0.040, (0.80, 0.90, 1.0))]:
        b = b + np.exp(-((uu - cx) ** 2) / (2 * wu ** 2)
                       - ((vv - cy) ** 2) / (2 * wv ** 2))[..., None] * inten * np.array(tint)
    # faint, broken vertical detail. It has to be there — refraction that displaces a flat wall
    # produces a flat wall, and the whole point of the water treatment is that you can see through
    # it — but at an amplitude that never reads as light shafts.
    streak = 0.5 + 0.5 * np.cos(2 * np.pi * (uu * 5.0 + 0.35 * np.sin(vv * 5 + uu * 3)))
    streak = _ss(0.86, 1.0, streak) * _ss(0.10, 0.45, vv)
    b = b + streak[..., None] * 0.006 * np.array([0.85, 0.92, 1.0])
    mott = value_noise(res, 7, np.random.default_rng(7), 3)
    b = b + (mott - 0.5)[..., None] * 0.010 * np.array([0.7, 0.85, 1.0])
    b = b + np.exp(-((vv - floor_y) ** 2) / (2 * 0.030 ** 2))[..., None] * np.array([0.05, 0.07, 0.09])
    fm = _ss(floor_y + 0.004, floor_y - 0.004, vv)
    b = b * (1 - fm[..., None]) + np.array([0.028, 0.034, 0.046]) * fm[..., None]
    return np.clip(b, 0, 1).astype(np.float32)


def upload_static():
    rng = np.random.default_rng(7)
    n1 = value_noise(RES, 9, rng, 3)
    n2 = value_noise(RES, 40, rng, 3)
    # fields are [i=x, j=y]; the noise arrays are [row=y down, col=x]
    noise1.from_numpy(np.transpose(n1[::-1], (1, 0)).copy())
    noise2.from_numpy(np.transpose(n2[::-1], (1, 0)).copy())
    b = build_background(RES, physics.core.floor_y)
    bg.from_numpy(np.transpose(b[::-1], (1, 0, 2)).copy())


def upload_frame(pos, mats, vel=None, uv=None):
    n = pos.shape[0]
    buf = np.zeros((MAXP, 2), np.float32); buf[:n] = pos
    pp.from_numpy(buf)
    mb = np.zeros(MAXP, np.int32); mb[:n] = mats
    pmt.from_numpy(mb)
    sb_ = np.zeros(MAXP, np.float32)
    if vel is not None:
        sb_[:n] = np.linalg.norm(vel, axis=1)
    psp.from_numpy(sb_)
    ub = np.zeros((MAXP, 2), np.float32)
    ub[:n] = uv if uv is not None else pos
    puv.from_numpy(ub)
    return n


# ============================================================================ treatment dispatch
def treatment_ref(mat, dens_scale=1.0):
    return DEMO_DENSITY * dens_scale / (RES * RES)


PROPOSAL = {"fluid": "water_glass", "elastic": "rubber_tex", "snow": "snow_powder",
            "sand": "sand_grain"}
ALTERNATE = {"fluid": "water_film", "elastic": "rubber_flat", "snow": "snow_current",
             "sand": "sand_bare"}


def render_layer(kind, n, sel, ref):
    if kind == "current":
        render_current(n, sel)
    elif kind == "water_glass":
        render_water(n, sel, ref, glassy=True)
    elif kind == "water_film":
        render_water(n, sel, ref, glassy=False)
    elif kind == "rubber_tex":
        render_rubber(n, sel, ref, textured=True)
    elif kind == "rubber_flat":
        render_rubber(n, sel, ref, textured=False)
    elif kind == "snow_powder":
        render_snow(n, sel, ref, powder=True)
    elif kind == "snow_current":
        render_current(n, sel)
    elif kind == "sand_grain":
        render_sand(n, sel, ref, under=True)
    elif kind == "sand_bare":
        render_sand(n, sel, ref, under=False)
    else:
        raise KeyError(kind)


def render_frame(pos, mats, vel, uv, kinds, ref, grey=False, gain=1.0):
    """kinds: dict material-name -> treatment key. Solids composite first, then water, so submerged
    material is what the water absorbs and refracts."""
    set_palette(grey)
    n = upload_frame(pos, mats, vel, uv)
    copy_v3(bg, img)
    present = [m for m in ORDER if np.any(mats == MAT_ID[m])]
    for m in [x for x in present if x != "fluid"] + [x for x in present if x == "fluid"]:
        if m == "fluid":
            copy_v3(img, beneath)
        render_layer(kinds[m], n, MAT_ID[m], ref)
        composite()
    tonemap(1 if grey else 0, gain)
    return to_image()


# ============================================================================ scenes (canonical)
def demo_scene(name):
    """A canonical scene from sim.physics, re-seeded at the DEMO's areal particle density. The demo
    runs one global density (~1.7 particles per grid cell); a proposal that needs 8x that is not a
    proposal for this demo, and the iso threshold it is compared against is calibrated to it."""
    probe = physics.scene(name, n=16)
    n = int(round(DEMO_DENSITY * probe["area"]))
    return physics.scene(name, n=n)


def run_solo(name, material, n_frames):
    sc = demo_scene(name)
    snaps, times, stable = physics.simulate(material, sc["pts"], sc["area"], sc["T"],
                                            n_frames, v0=sc["v0"])
    mats = np.full(snaps.shape[1], MAT_ID[material], np.int32)
    return dict(snaps=snaps, times=times, mats=mats, stable=bool(stable), scene=name,
                material=material)


def run_fill(material, n_frames, depth=0.26, blob_r=0.13, blob_y=0.72, T=1.3):
    """TASK-LOCAL SCENE, declared as such. A slab of the material filling the bottom of the tank plus
    a disk of the same material dropped into it.

    Why it is here at all: the canonical scenes hold ~1-2.5k particles at the demo's areal density,
    and at that volume a fluid ends its life as a one-cell-thick film. A film is a fair test of the
    CURRENT look but a useless test of a water treatment whose whole claim is about depth, thickness
    and see-through. This scene is the one the demo actually shows once a visitor paints for a few
    seconds (the demo caps at 16384 particles, i.e. 0.58 unit area at this density), and it is the
    only geometry here that is not lifted from sim.physics.scene. Seeding still uses the canonical
    helpers (seed_lattice for a body meant to start at rest, seed_disk for the blob).
    """
    x0, x1 = physics.core.floor_y, 1.0 - physics.core.floor_y
    slab_area = (x1 - x0) * (depth - physics.core.floor_y)
    disk_area = float(np.pi * blob_r ** 2)
    n_slab = int(round(DEMO_DENSITY * slab_area))
    n_blob = int(round(DEMO_DENSITY * disk_area))
    # seed_lattice only for the fluid (it exists to start a POOL near its rest density). For
    # the others a lattice is actively harmful here: sand's per-particle grain jitter inherits
    # the lattice and the slab renders as vertical corduroy.
    if material == 'fluid':
        slab = physics.core.seed_lattice(x0, x1, physics.core.floor_y, depth, n_slab, seed=3)
    else:
        slab = physics.seed_box(x0, x1, physics.core.floor_y, depth, n_slab, seed=3)
    blob = physics.seed_disk((0.42, blob_y), blob_r, n_blob, seed=5)
    pts = np.concatenate([slab, blob], 0)
    snaps, times, stable = physics.simulate(material, pts, slab_area + disk_area, T, n_frames)
    mats = np.full(snaps.shape[1], MAT_ID[material], np.int32)
    return dict(snaps=snaps, times=times, mats=mats, stable=bool(stable), scene="fill",
                material=material)


def run_fourup(n_frames, r=0.10, y=0.62, T=1.6):
    """All four in ONE shared grid, side by side. Distinctness is a property of the SET, so the four
    have to be seen together and not only one at a time."""
    cxs = [0.15, 0.3833, 0.6167, 0.85]
    groups = []
    for k, (cx, m) in enumerate(zip(cxs, ORDER)):
        area = float(np.pi * r * r)
        n = int(round(DEMO_DENSITY * area))
        groups.append({"material": m, "pts": physics.seed_disk((cx, y), r, n, seed=11 + k),
                       "area": area, "v0": (0.0, 0.0)})
    snaps, times, mats, stable, dt = physics.simulate_multi(groups, T, n_frames)
    return dict(snaps=snaps, times=times, mats=mats, stable=bool(stable), scene="four_up",
                material="all", dt=float(dt))


def run_pool(solid, n_frames, T=2.0):
    probe = physics.scene_pool(solid, n=16)
    n = int(round(DEMO_DENSITY * probe["water_area"]))
    sc = physics.scene_pool(solid, n=n, T=T)
    snaps, times, mats, stable, dt = physics.simulate_multi(sc["groups"], sc["T"], n_frames)
    return dict(snaps=snaps, times=times, mats=mats, stable=bool(stable), scene="pool_" + solid,
                material=solid, dt=float(dt))


def velocities(snaps, times):
    v = np.zeros_like(snaps)
    dtv = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    v[1:] = (snaps[1:] - snaps[:-1]) / np.maximum(dtv[1:, None, None], 1e-6)
    v[0] = v[1]
    return v


# ============================================================================ output helpers
def save_png(path, arr):
    import imageio.v2 as imageio
    imageio.imwrite(path, arr)


def encode(path, frames, fps=30):
    import imageio.v2 as imageio
    even = [f[: f.shape[0] - f.shape[0] % 2, : f.shape[1] - f.shape[1] % 2] for f in frames]
    imageio.mimwrite(path, even, fps=fps, quality=8, macro_block_size=1, codec="libx264")


_FONT = None


def _font(sz):
    global _FONT
    from PIL import ImageFont
    try:
        return ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", sz)
    except Exception:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", sz)
        except Exception:
            return ImageFont.load_default()


def label_panels(panels, titles, gap=6, band=34):
    """Stack panels horizontally with a caption band. The caption is not decoration — a comparison
    frame with no labels is unreadable the moment it leaves the page it was built for."""
    from PIL import Image, ImageDraw
    h, w = panels[0].shape[:2]
    W = w * len(panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (W, h + band), (10, 14, 20))
    d = ImageDraw.Draw(out)
    fnt = _font(max(13, int(0.026 * w)))
    for k, (p, t) in enumerate(zip(panels, titles)):
        x = k * (w + gap)
        out.paste(Image.fromarray(p), (x, band))
        d.text((x + 8, band // 2), t, font=fnt, fill=(190, 214, 232), anchor="lm")
    return np.asarray(out)


# ============================================================================ cost measurement
def bench(kind, n_particles=16384, reps=40, res_note=None):
    """MEASURED per-frame render cost. Taichi/CUDA on this machine's GPU, NOT WebGPU — so this is a
    lower bound on the browser cost of the same passes, and it is reported as such. Its use here is
    the RATIO to the current renderer measured in the identical harness, which is the number that
    actually decides whether a treatment fits the demo's ~10 ms drawing budget."""
    rng = np.random.default_rng(3)
    pos = rng.uniform(0.08, 0.92, (n_particles, 2)).astype(np.float32)
    pos[:, 1] *= 0.55
    mats = np.zeros(n_particles, np.int32)
    sel = {"current": 0, "water_glass": 0, "water_film": 0, "rubber_tex": 1, "rubber_flat": 1,
           "snow_powder": 2, "snow_current": 2, "sand_grain": 3, "sand_bare": 3}[kind]
    mats[:] = sel
    vel = rng.normal(0, 1.2, (n_particles, 2)).astype(np.float32)
    set_palette(False)
    n = upload_frame(pos, mats, vel, pos)
    ref = DEMO_DENSITY / (RES * RES)
    copy_v3(bg, img)
    copy_v3(bg, beneath)
    for _ in range(6):                      # warm-up: JIT compile excluded, reported separately
        render_layer(kind, n, sel, ref)
        composite()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        copy_v3(bg, img)
        copy_v3(bg, beneath)
        render_layer(kind, n, sel, ref)
        composite()
        tonemap(0, 1.0)
    ti.sync()
    return (time.perf_counter() - t0) / reps * 1e3
