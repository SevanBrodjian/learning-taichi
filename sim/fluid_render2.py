"""Photoreal-leaning fluid renderer, v2 -- a refinement of ``sim/fluid_render.py``.

This is the follow-up pass that fixes the three problems the first renderer left on the table:

  1. WATER BEHAVIOR (a dynamics problem, not a rendering one). The v1 fluid used a soft, weakly
     compressible EOS (E=170) played back at ~4x slow motion, which reads as syrup. v2 stiffens the
     fluid toward incompressibility (higher E, with a correspondingly smaller dt for CFL stability),
     drops the ball from higher for a faster impact, and maps sim time to playback much closer to
     real time so the motion is fast, lively and splashy instead of oozing.

  2. INTERIOR HOLES (a surface-reconstruction problem). v1 took a single metaball density isocontour
     as both "is there liquid" and "where is the surface"; Poisson fluctuations in particle count dip
     the interior density below the isovalue, punching air holes inside a calm body. v2 SEPARATES the
     two: a FILLED interior mask (low threshold -> morphological closing -> fill only the small
     enclosed pockets, keeping genuinely large air cavities like a breaking-wave barrel) decides
     opacity, while the density gradient is used only for surface normals/shading. Optical thickness
     comes from a distance transform of the filled mask, so the body reads as a smooth solid volume
     with no speckle.

  3. PUSH REALISM. A real screen-space thickness (distance transform) drives clean Beer-Lambert
     absorption; refraction is driven by both the surface normal and the thickness gradient so the
     flat interior lenses the background instead of staying uniform; a richer structured background
     rewards that refraction; stylized floor caustics, chromatic edges, a soft contact shadow, softer
     foam and a higher render resolution push the look further.

All rendering is numpy/scipy, headless -> mp4. Non-differentiable and offline by design.

Usage:
    python sim/fluid_render2.py                 # full render of both scenes + comparisons
    python sim/fluid_render2.py --quick         # short low-res smoke test
    python sim/fluid_render2.py --probe SCENE FRAME
    python sim/fluid_render2.py --sweep         # tiny sim-parameter sweep, prints motion stats
"""
import argparse
import os

import numpy as np
import taichi as ti
from scipy.ndimage import (gaussian_filter, binary_closing, distance_transform_edt,
                           label, grey_closing)

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

# --------------------------------------------------------------------------- sim constants
dim = 2
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
p_rho = 1.0
gravity = 9.8
bound = 3
floor_y = bound * dx

MAX_P = 60000

x = ti.Vector.field(dim, float, MAX_P)
v = ti.Vector.field(dim, float, MAX_P)
C = ti.Matrix.field(dim, dim, float, MAX_P)
J = ti.field(float, MAX_P)
grid_v = ti.Vector.field(dim, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))
x_seed = ti.Vector.field(dim, float, MAX_P)


# --------------------------------------------------------------------------- MLS-MPM fluid step
@ti.func
def fluid_stress(p, dt, E, p_vol):
    # weakly-compressible pressure from the tracked volume ratio J (linearized Tait / EOS).
    s = -dt * 4.0 * E * p_vol * (J[p] - 1.0) * inv_dx * inv_dx
    return ti.Matrix([[s, 0.0], [0.0, s]])


@ti.kernel
def clear_grid():
    for i, j in ti.ndrange(n_grid, n_grid):
        grid_v[i, j] = ti.Vector.zero(float, dim)
        grid_m[i, j] = 0.0


@ti.kernel
def p2g(n: ti.i32, dt: ti.f32, E: ti.f32, p_vol: ti.f32, p_mass: ti.f32):
    for p in range(n):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = fluid_stress(p, dt, E, p_vol)
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
        if i > n_grid - bound and vx > 0:
            vx = 0.0
        grid_v[i, j] = ti.Vector([vx, vy])


@ti.kernel
def g2p(n: ti.i32, dt: ti.f32, flip: ti.f32):
    # APIC transfer (affine velocity field C). A small optional FLIP admixture (flip) re-injects
    # a touch of the particle's previous velocity to counter numerical damping and keep splashes
    # lively; flip=0 is pure APIC/MLS-MPM.
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
        # FLIP blend: v_new = v_pic + flip*(v_old - v_pic_of_old) is the standard form; here the
        # cheap proxy adds a fraction of the residual particle velocity for extra energy.
        v[p] = new_v + flip * (v[p] - new_v)
        x[p] = x[p] + dt * new_v
        x[p] = ti.math.clamp(x[p], floor_y, 1.0 - floor_y)
        J[p] = J[p] * (1.0 + dt * new_C.trace())
        C[p] = new_C


@ti.kernel
def init_state(n: ti.i32):
    for p in range(n):
        x[p] = x_seed[p]
        v[p] = ti.Vector.zero(float, dim)
        C[p] = ti.Matrix.zero(float, dim, dim)
        J[p] = 1.0


# --------------------------------------------------------------------------- seeding
def seed_disk(center, radius, n, rng):
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([center[0] + rad * np.cos(ang), center[1] + rad * np.sin(ang)], axis=1)


def seed_box(x0, x1, y0, y1, n, rng):
    return np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], axis=1)


def upload(pts):
    n = pts.shape[0]
    buf = np.zeros((MAX_P, dim), dtype=np.float32)
    buf[:n] = pts.astype(np.float32)
    x_seed.from_numpy(buf)
    return n


def run_scene(pts, E, dt, T, n_frames, fric=0.1, area=None, flip=0.0):
    """Roll the fluid to physical time T, capturing n_frames snapshots of (positions, velocities)."""
    n = upload(pts)
    if area is None:
        area = 1.0
    p_vol = area / n
    p_mass = p_vol * p_rho
    steps_per_frame = max(1, int(round((T / n_frames) / dt)))
    init_state(n)
    xs = np.zeros((n_frames, n, dim), dtype=np.float32)
    vs = np.zeros((n_frames, n, dim), dtype=np.float32)
    stable = True
    for f in range(n_frames):
        for _ in range(steps_per_frame):
            clear_grid()
            p2g(n, dt, E, p_vol, p_mass)
            grid_op(dt, fric)
            g2p(n, dt, flip)
        cx = x.to_numpy()[:n]
        cv = v.to_numpy()[:n]
        if not np.isfinite(cx).all():
            stable = False
            cx = np.nan_to_num(cx)
            cv = np.nan_to_num(cv)
        xs[f] = cx
        vs[f] = cv
    return xs, vs, stable


# =========================================================================== RENDERER
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def splat_field(pos, res, weights=None):
    """Histogram particles into a res x res grid (row 0 = top = high y)."""
    col = np.clip(pos[:, 0] * res, 0, res - 1e-3)
    row = np.clip((1.0 - pos[:, 1]) * res, 0, res - 1e-3)
    grid, _, _ = np.histogram2d(row, col, bins=res, range=[[0, res], [0, res]], weights=weights)
    return grid


def _value_noise(res, cells, rng, octaves=3):
    """Cheap tileable-ish value noise in [0,1] for surface ripple and background texture."""
    out = np.zeros((res, res), np.float32)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        c = cells * (2 ** o)
        base = rng.standard_normal((c + 1, c + 1)).astype(np.float32)
        # upsample with bilinear via zoom-free indexing
        ys = np.linspace(0, c, res, endpoint=False)
        xs = np.linspace(0, c, res, endpoint=False)
        y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
        fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
        a = base[np.ix_(y0, x0)]; b = base[np.ix_(y0, x0 + 1)]
        cc = base[np.ix_(y0 + 1, x0)]; d = base[np.ix_(y0 + 1, x0 + 1)]
        layer = a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + cc * fy * (1 - fx) + d * fy * fx
        out += amp * layer
        total += amp
        amp *= 0.5
    out /= total
    out = (out - out.min()) / (np.ptp(out) + 1e-9)
    return out


def build_background(res, tank, rng):
    """A structured studio backdrop that rewards refraction: cool-to-warm vertical gradient, several
    soft softbox glows, faint vertical light streaks and a low value-noise mottle so the flat body
    still displaces *something*, a horizon glow and a darker reflective floor slab."""
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    vv = yy / res
    uu = xx / res
    top = np.array([0.05, 0.09, 0.14])
    mid = np.array([0.09, 0.15, 0.21])
    bot = np.array([0.12, 0.17, 0.20])
    g = smoothstep(0.0, 0.6, vv)[..., None]
    bg = top * (1 - g) + mid * g
    g2 = smoothstep(0.5, 1.0, vv)[..., None]
    bg = bg * (1 - g2) + bot * g2
    # soft localized softbox glows (2D gaussians): bright structured things to reflect/refract.
    for cx, cyv, wu, wv, inten, tint in [
            (0.20, 0.30, 0.06, 0.10, 0.16, (0.80, 0.90, 1.0)),
            (0.80, 0.26, 0.08, 0.11, 0.12, (1.0, 0.93, 0.80)),
            (0.50, 0.16, 0.12, 0.06, 0.07, (0.90, 0.95, 1.0))]:
        glow = np.exp(-((uu - cx) ** 2) / (2 * wu ** 2) - ((vv - cyv) ** 2) / (2 * wv ** 2))
        bg = bg + glow[..., None] * inten * np.array(tint)
    # very faint, broken vertical detail: gives refraction something to bend without reading as
    # light shafts. Kept low-contrast and irregular on purpose.
    streak = 0.5 + 0.5 * np.cos(2 * np.pi * (uu * 4.0 + 0.35 * np.sin(vv * 5 + uu * 3)))
    streak = smoothstep(0.7, 1.0, streak) * smoothstep(0.1, 0.5, vv) * smoothstep(0.85, 0.5, vv)
    bg = bg + streak[..., None] * 0.016 * np.array([0.85, 0.92, 1.0])
    # low-frequency mottle so a flat lens still shifts visible detail
    mott = _value_noise(res, 6, rng, octaves=3)
    bg = bg + (mott - 0.5)[..., None] * 0.03 * np.array([0.7, 0.85, 1.0])
    # horizon glow just above the floor line
    fy = 1.0 - tank["floor"]
    glow = np.exp(-((vv - fy) ** 2) / (2 * 0.035 ** 2))
    bg = bg + glow[..., None] * np.array([0.07, 0.09, 0.11])
    # floor slab (kept for reflection reference; caustics are added later per-frame)
    floor_mask = smoothstep(fy - 0.004, fy + 0.004, vv)
    floor_col = np.array([0.045, 0.055, 0.075])
    bg = bg * (1 - floor_mask[..., None]) + floor_col * floor_mask[..., None]
    tank["floor_row"] = int(fy * res)
    return np.clip(bg, 0, 1).astype(np.float32)


def sample_bilinear(img, cols, rows):
    H, W = img.shape[:2]
    c = np.clip(cols, 0, W - 1.001)
    r = np.clip(rows, 0, H - 1.001)
    c0 = np.floor(c).astype(np.int32); r0 = np.floor(r).astype(np.int32)
    c1 = c0 + 1; r1 = r0 + 1
    fc = (c - c0)[..., None]; fr = (r - r0)[..., None]
    a = img[r0, c0]; b = img[r0, c1]; cc = img[r1, c0]; d = img[r1, c1]
    return (a * (1 - fc) * (1 - fr) + b * fc * (1 - fr) + cc * (1 - fc) * fr + d * fc * fr)


DEFAULT_LOOK = dict(
    res=1080,
    sigma_px=7.0,          # density blur (blob smoothness)
    iso_fill=0.28,         # LOW threshold for the filled interior mask (is there liquid here)
    iso_surf=0.55,         # higher threshold used only for surface/foam detail
    edge=0.09,             # feather half-width around the fill threshold for AA
    close_px=5,            # binary-closing radius: seals sub-particle-spacing pinholes
    fill_max_frac=0.010,   # enclosed air pockets <= this fraction of the frame get filled as liquid
    thick_char=55.0,       # px depth that maps to unit optical thickness (distance transform)
    thick_max=3.2,
    ripple_amp=0.22,       # subtle interior surface ripple that lets the flat body refract
    ripple_cells=9,
    normal_k=2.1,          # curvature constant: larger = flatter (more viewer-facing)
    normal_amp=3.0,        # in-plane normal strength from the mask gradient
    refract=60.0,          # background refraction offset (px) from the surface normal
    refract_lens=30.0,     # extra offset from the thickness gradient (interior lensing)
    chroma=0.35,           # chromatic dispersion fraction on the refraction offset
    absorb=1.30,           # Beer-Lambert absorption per unit thickness (lower = glassier/clearer)
    liquid=(0.02, 0.16, 0.30),   # deep water body color
    shallow=(0.20, 0.50, 0.60),  # thin-film color
    F0=0.02,               # water Fresnel reflectance at normal incidence
    rim=0.30,              # extra brightening of the grazing waterline edge
    light=(-0.55, 0.72),   # key light direction in image plane (x right, y UP)
    shininess=90.0,
    spec_gain=2.8,
    sheen=0.10,
    caustic_gain=0.45,     # stylized floor caustic brightness
    foam_speed=1.7,
    foam_gain=0.95,
    foam_thin=(0.30, 0.05),  # thickness range that reads as thin broken spray (droplet tips only)
    foam_thin_w=0.5,
    foam_tex=0.5,          # foam texture strength (noise breakup)
    contact=0.30,          # soft contact-shadow strength under the water footprint
    bloom=0.40,
    vignette=0.34,
    seed=7,
)


def _disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def fill_small_holes(mask, max_area):
    """Fill enclosed air pockets whose area <= max_area, keeping large genuine cavities and the
    border-connected outside air. This is what turns a holey isocontour into a solid body without
    also filling a breaking-wave barrel."""
    inv = ~mask
    lbl, n = label(inv)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    border = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
    fill = np.zeros(n + 1, bool)
    fill[1:] = sizes[1:] <= max_area
    fill[border] = False          # never fill anything connected to the outside air
    fill[0] = False
    return mask | fill[lbl]


def build_masks(pos, res, L):
    """Return the separated reconstruction:
      dens  -- normalized metaball density (Poisson-noisy; used only for surface detail)
      fill  -- feathered FILLED opacity mask in [0,1] (is there liquid here) -> compositing
      body  -- boolean solid interior
      thick -- smooth optical thickness from a distance transform (no speckle) -> absorption
    """
    raw = splat_field(pos, res)
    dens = gaussian_filter(raw, L["sigma_px"])
    ref = np.percentile(dens[dens > 1e-4], 80) if np.any(dens > 1e-4) else 1.0
    d = dens / (ref + 1e-9)
    # 1) generous low-threshold body, 2) grey-close to seal pinholes, 3) fill small enclosed pockets
    dc = grey_closing(d, footprint=_disk(3))
    body = dc > L["iso_fill"]
    body = binary_closing(body, structure=_disk(L["close_px"]), iterations=1)
    max_area = L["fill_max_frac"] * res * res
    body = fill_small_holes(body, max_area)
    # feathered opacity from a smoothed body (anti-aliased silhouette)
    bf = gaussian_filter(body.astype(np.float32), 2.0)
    fill = smoothstep(0.5 - L["edge"], 0.5 + L["edge"], bf)
    # smooth volumetric thickness: distance to nearest air inside the body
    dist = distance_transform_edt(body)
    thick = gaussian_filter(dist / L["thick_char"], 2.0)
    thick = np.clip(thick, 0.0, L["thick_max"])
    return d, fill, body, thick


def render_frame(pos, vel, tank, look=None, return_layers=False, naive=False, frame=0):
    """Render one frame. If naive=True, reproduce the v1 single-isocontour mask (holey) for the
    interior-fill comparison; otherwise use the filled-mask reconstruction."""
    L = dict(DEFAULT_LOOK)
    if look:
        L.update(look)
    res = L["res"]
    rng = np.random.default_rng(L["seed"])
    # cached noise (computed once per tank); ripple is rolled by frame index for subtle animation
    ripple_noise = tank.get("ripple_noise")
    foam_noise = tank.get("foam_noise")
    if ripple_noise is None:
        ripple_noise = _value_noise(res, L["ripple_cells"], rng, octaves=3)
    if foam_noise is None:
        foam_noise = _value_noise(res, 40, rng, octaves=3)
    sh = (frame * 5) % res
    ripple_noise = np.roll(ripple_noise, sh, axis=1)
    speed = np.linalg.norm(vel, axis=1)

    d, fill, body, thick = build_masks(pos, res, L)

    if naive:
        # v1 behavior: a single density isocontour is BOTH opacity and thickness -> interior holes.
        surf_mask = smoothstep(0.5 - 0.5, 0.5 + 0.5, d)  # iso=0.5, wide edge as in v1
        fill = surf_mask
        thick = np.clip(d, 0.0, 3.0)
        body = d > 0.5

    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)

    # --- surface normal: from the smooth body field + a subtle interior ripple so the flat slab
    # actually refracts. The ripple is masked to the interior and kept low amplitude.
    base_field = gaussian_filter(fill, 2.5) + 0.6 * np.clip(thick, 0, L["thick_max"]) / L["thick_max"]
    if not naive and L["ripple_amp"] > 0:
        ripple = ripple_noise - 0.5
        interior = smoothstep(0.2, 0.9, thick / L["thick_max"])
        base_field = base_field + L["ripple_amp"] * ripple * interior
    ds = gaussian_filter(base_field, 1.6)
    grow, gcol = np.gradient(ds)
    nx = -gcol * L["normal_amp"]
    ny = grow * L["normal_amp"]
    nz = np.full_like(nx, L["normal_k"])
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / nn, ny / nn, nz / nn

    # --- refraction of the background: normal-driven + thickness-gradient lensing, with chromatic
    # dispersion (R/G/B sampled at slightly different offsets) for subtle spectral edges.
    bg = tank["bg"]
    tgr, tgc = np.gradient(gaussian_filter(thick, 2.0))
    off_n = L["refract"] * np.clip(thick, 0, 1.6)
    lx_off = nx * off_n - tgc * L["refract_lens"]
    ly_off = ny * off_n + tgr * L["refract_lens"]   # note image-row sign handled below
    def _sample(scale):
        scol = xx + lx_off * scale
        srow = yy - ly_off * scale
        return sample_bilinear(bg, scol, srow)
    if L["chroma"] > 0 and not naive:
        r = _sample(1.0 + L["chroma"])[..., 0]
        g = _sample(1.0)[..., 1]
        b = _sample(1.0 - L["chroma"])[..., 2]
        refr = np.dstack([r, g, b])
    else:
        refr = _sample(1.0)

    # --- Beer-Lambert absorption / depth color -----------------------------------------------------
    liquid = np.array(L["liquid"]); shallow = np.array(L["shallow"])
    tt = np.clip(thick, 0, L["thick_max"])[..., None]
    transmit = np.exp(-L["absorb"] * tt)
    body_tint = shallow * transmit + liquid * (1 - transmit)
    col = refr * transmit + body_tint * (1 - transmit)

    # interior ambient occlusion for volume
    ao = 1.0 - 0.16 * smoothstep(0.6, L["thick_max"], thick)
    col = col * ao[..., None]

    # --- Fresnel reflection (environment) ----------------------------------------------------------
    cos_t = np.clip(nz, 0, 1)
    fres = L["F0"] + (1 - L["F0"]) * (1 - cos_t) ** 5
    rup = 2 * nz * ny
    sky = np.clip(0.55 + 0.45 * rup, 0, 1)[..., None]
    env = np.array([0.60, 0.74, 0.90]) * sky + np.array([0.10, 0.13, 0.17]) * (1 - sky)
    col = col * (1 - fres[..., None]) + env * fres[..., None]

    # explicit rim light at the grazing waterline
    rimg = L["rim"] * (1.0 - cos_t) ** 3
    col = col + rimg[..., None] * np.array([0.35, 0.58, 0.78])

    # --- Blinn-Phong specular (tight glint + broad wet sheen) --------------------------------------
    lx, ly = L["light"]; lz = 0.55
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    hx, hy, hz = lx, ly, lz + 1.0
    hn = np.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / hn, hy / hn, hz / hn
    ndh = np.clip(nx * hx + ny * hy + nz * hz, 0, 1)
    spec = L["spec_gain"] * ndh ** L["shininess"]
    sheen = L["sheen"] * ndh ** 8.0
    col = col + (spec + sheen)[..., None] * np.array([1.0, 1.0, 0.97])

    # --- foam / spray ------------------------------------------------------------------------------
    sp_sum = gaussian_filter(splat_field(pos, res, weights=speed), L["sigma_px"])
    avg_speed = sp_sum / (gaussian_filter(splat_field(pos, res), L["sigma_px"]) + 1e-6)
    motion = smoothstep(L["foam_speed"], L["foam_speed"] * 2.2, avg_speed)
    edgemag = np.hypot(*np.gradient(gaussian_filter(fill, 3.0)))
    band = smoothstep(0.004, 0.02, edgemag)
    ft0, ft1 = L["foam_thin"]
    thinfoam = fill * smoothstep(ft0, ft1, thick)     # only the genuinely thin broken fringe
    motionfoam = motion * band
    foam = np.clip(L["foam_gain"] * (0.9 * motionfoam + L["foam_thin_w"] * thinfoam), 0, 1)
    if L["foam_tex"] > 0 and not naive:
        foam = foam * (1.0 - L["foam_tex"] * (1.0 - foam_noise))   # break smooth white into aerated texture
    foam = gaussian_filter(foam, 1.8)
    foam_col = np.array([0.93, 0.97, 1.0])
    col = col * (1 - foam[..., None]) + foam_col * foam[..., None]

    # --- composite liquid over background ----------------------------------------------------------
    m = fill[..., None]
    img = bg * (1 - m) + col * m

    # --- stylized floor caustics: bright bands on the floor under the water, brightest where the
    # surface above focuses (large |horizontal surface slope|). Purely a look cue. --------------
    floor_row = tank.get("floor_row", int((1.0 - tank["floor"]) * res))
    if L["caustic_gain"] > 0 and not naive and floor_row < res - 4:
        col_liquid = fill.sum(axis=0)                       # how much water sits above each column
        surf_slope = np.abs(np.gradient(gaussian_filter(nx.mean(axis=0), 4.0)))
        focus = gaussian_filter((col_liquid / (col_liquid.max() + 1e-6)) * surf_slope, 6.0)
        focus = focus / (focus.max() + 1e-6)
        caustic = np.zeros((res, res), np.float32)
        band_h = int(0.05 * res)
        prof = np.exp(-((np.arange(res) - floor_row) ** 2) / (2 * (band_h * 0.5) ** 2))
        caustic = np.outer(prof, focus) * L["caustic_gain"]
        img = img + caustic[..., None] * np.array([0.75, 0.85, 1.0])

    # --- soft contact shadow: darken the floor just under the water footprint for grounding -------
    if L["contact"] > 0 and not naive and floor_row < res - 4:
        footprint = gaussian_filter((fill.sum(axis=0) > 3).astype(np.float32), 12.0)
        prof = np.exp(-((np.arange(res) - (floor_row + 6)) ** 2) / (2 * (0.02 * res) ** 2))
        shadow = np.outer(prof, footprint)
        img = img * (1 - L["contact"] * shadow[..., None])

    # --- floor reflection of the liquid (subtle) --------------------------------------------------
    fy_px = floor_row
    if fy_px < res - 2:
        strip = img[:fy_px]
        refl = strip[::-1]
        h = res - fy_px
        refl = refl[:h]
        fade = (1.0 - np.linspace(0, 1, refl.shape[0]))[:, None, None] * 0.20
        img[fy_px:fy_px + refl.shape[0]] = (
            img[fy_px:fy_px + refl.shape[0]] * (1 - fade) + refl * fade)

    # --- finish: bloom, vignette, tone map ---------------------------------------------------------
    bright = np.clip(img - 0.72, 0, None)
    bloom = gaussian_filter(bright, (res / 130.0, res / 130.0, 0))
    img = img + L["bloom"] * bloom
    cyc, cxc = res / 2, res / 2
    r2 = ((xx - cxc) / res) ** 2 + ((yy - cyc) / res) ** 2
    vig = 1.0 - L["vignette"] * smoothstep(0.12, 0.42, r2)
    img = img * vig[..., None]
    img = img / (img + 0.9)
    img = np.clip(img * 1.55, 0, 1) ** (1 / 1.15)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    if return_layers:
        return out, dict(dens=d, fill=fill, thick=thick, body=body,
                         normal=np.dstack([nx, ny, nz]), refr=refr, foam=foam)
    return out


def encode_mp4(path, frames, fps=30):
    import imageio
    even = [f[: f.shape[0] - f.shape[0] % 2, : f.shape[1] - f.shape[1] % 2] for f in frames]
    imageio.mimwrite(path, even, fps=fps, quality=9, macro_block_size=1, codec="libx264")


def save_png(path, img):
    import imageio
    imageio.imwrite(path, img)


# =========================================================================== SCENES
# Water behavior fix: stiffer (nearer incompressible) fluid E~=520 with a smaller dt for CFL, a
# higher drop for faster impact, a shallower pool for a bigger splash, and playback close to real
# time (fps chosen per scene so the clip is only mildly slowed) rather than ~4x slow motion.
def scene_balldrop(quick=False):
    rng = np.random.default_rng(0)
    wall = floor_y
    n_pool = 9000 if quick else 26000
    n_ball = 2400 if quick else 7000
    pool = seed_box(wall + 0.01, 1 - wall - 0.01, wall, 0.165, n_pool, rng)
    ball = seed_disk((0.5, 0.76), 0.10, n_ball, rng)
    pts = np.concatenate([pool, ball], axis=0)
    area = (1 - 2 * wall) * (0.165 - wall) + np.pi * 0.10 ** 2
    return dict(pts=pts, area=area, E=520.0, dt=7.0e-5, fric=0.06, flip=0.03,
                T=1.05, fps=60, hero_frame=None)


def scene_dambreak(quick=False):
    rng = np.random.default_rng(1)
    wall = floor_y
    n = 12000 if quick else 30000
    col = seed_box(wall, 0.30, wall, 0.80, n, rng)
    area = (0.30 - wall) * (0.80 - wall)
    return dict(pts=col, area=area, E=520.0, dt=7.0e-5, fric=0.04, flip=0.03,
                T=1.5, fps=60, hero_frame=None)


SCENES = {"balldrop": scene_balldrop, "dambreak": scene_dambreak}


def make_tank(res, look=None):
    L = dict(DEFAULT_LOOK)
    if look:
        L.update(look)
    tank = {"floor": floor_y}
    tank["bg"] = build_background(res, tank, np.random.default_rng(L["seed"]))
    rng = np.random.default_rng(L["seed"])
    tank["ripple_noise"] = _value_noise(res, L["ripple_cells"], rng, octaves=3)
    tank["foam_noise"] = _value_noise(res, 40, rng, octaves=3)
    return tank


def probe(scene_name, frame_idx, out_path, look=None, quick=True, naive=False, n_frames=None):
    cfg = SCENES[scene_name](quick=quick)
    L = dict(DEFAULT_LOOK)
    if look:
        L.update(look)
    if n_frames is None:
        n_frames = 70 if quick else 150
    xs, vs, stable = run_scene(cfg["pts"], cfg["E"], cfg["dt"], cfg["T"], n_frames,
                               fric=cfg["fric"], area=cfg["area"], flip=cfg["flip"])
    sp = np.linalg.norm(vs, axis=2)
    print(f"  {scene_name}: stable={stable} frames={n_frames} particles={xs.shape[1]} "
          f"peak_speed={sp.max():.2f} mean_speed={sp.mean():.3f}")
    tank = make_tank(L["res"], L)
    fi = min(frame_idx, n_frames - 1)
    img = render_frame(xs[fi], vs[fi], tank, L, naive=naive)
    save_png(out_path, img)
    print(f"  wrote {out_path}  (frame {fi}/{n_frames})")
    return xs, vs


def _montage(frames, idxs, out_path, scene):
    """Save a labeled contact sheet of candidate frames so a hero can be chosen by viewing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = len(idxs)
    fig, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 3.2), facecolor="#0a0e14")
    if cols == 1:
        axes = [axes]
    for ax, fi in zip(axes, idxs):
        ax.imshow(frames[fi])
        ax.set_title(f"{scene} f{fi}", color="#dfe6ee", fontsize=11, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#26313d")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.9, bottom=0.01, wspace=0.02)
    fig.savefig(out_path, dpi=80, facecolor="#0a0e14")
    plt.close(fig)


def render_scene_to_disk(scene_name, out_dir, hero_frame, n_frames=150, look=None,
                         montage_idxs=None):
    cfg = SCENES[scene_name](quick=False)
    L = dict(DEFAULT_LOOK)
    if look:
        L.update(look)
    xs, vs, stable = run_scene(cfg["pts"], cfg["E"], cfg["dt"], cfg["T"], n_frames,
                               fric=cfg["fric"], area=cfg["area"], flip=cfg["flip"])
    fps = cfg["fps"]
    sp = np.linalg.norm(vs, axis=2)
    print(f"  {scene_name}: stable={stable} particles={xs.shape[1]} frames={n_frames} "
          f"fps={fps} peak_speed={sp.max():.2f}")
    # persist the sim so any hero frame can be re-rendered later without re-simulating
    np.save(os.path.join(out_dir, f"_sim_{scene_name}_xs.npy"), xs)
    np.save(os.path.join(out_dir, f"_sim_{scene_name}_vs.npy"), vs)
    tank = make_tank(L["res"], L)
    import time
    t0 = time.time()
    frames = [render_frame(xs[f], vs[f], tank, L, frame=f) for f in range(n_frames)]
    print(f"  rendered {n_frames} frames in {time.time()-t0:.1f}s")
    encode_mp4(os.path.join(out_dir, f"{scene_name}.mp4"), frames, fps=fps)
    hf = min(hero_frame, n_frames - 1)
    save_png(os.path.join(out_dir, f"{scene_name}_hero.png"), frames[hf])
    if montage_idxs:
        _montage(frames, [min(i, n_frames - 1) for i in montage_idxs],
                 os.path.join(out_dir, f"_montage_{scene_name}.png"), scene_name)
    return xs, vs, stable, tank, L, frames


# --------------------------------------------------------------------------- comparison figures
def _resize(img, h, w):
    """Nearest/linear resize an (H,W,3) uint8 image to (h,w,3) using scipy zoom."""
    from scipy.ndimage import zoom
    zy, zx = h / img.shape[0], w / img.shape[1]
    out = zoom(img.astype(np.float32), (zy, zx, 1), order=1)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_before_after(old_png, new_img, scene, out_path):
    """Side-by-side: prior renderer's hero (left) vs this pass (right), same scene."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    old = imageio.imread(old_png)[..., :3]
    h = min(old.shape[0], new_img.shape[0])
    w = int(h)
    old_r = _resize(old, h, w)
    new_r = _resize(new_img, h, w)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.0), facecolor="#0a0e14")
    axes[0].imshow(old_r); axes[0].set_title("BEFORE  (v1 renderer)", color="#e6a15a", fontsize=15, pad=10)
    axes[1].imshow(new_r); axes[1].set_title("AFTER  (v2: water-like, no interior holes, clearer)",
                                             color="#7fd0ff", fontsize=15, pad=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#26313d")
    fig.suptitle(f"{scene}: before / after", color="#dfe6ee", fontsize=17, y=0.98)
    fig.subplots_adjust(left=0.006, right=0.994, top=0.9, bottom=0.01, wspace=0.03)
    fig.savefig(out_path, dpi=95, facecolor="#0a0e14")
    plt.close(fig)


def build_interior_breakdown(pos, vel, tank, L, out_path):
    """Teach the holes fix: naive single-isocontour (holey) vs filled-mask (solid), same particles,
    plus the mask/thickness panels that explain why."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    holey = render_frame(pos, vel, tank, L, naive=True)
    filled, layers = render_frame(pos, vel, tank, L, return_layers=True)
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.3), facecolor="#0a0e14")
    axes[0].imshow(holey)
    axes[0].set_title("Naive isocontour\n(interior holes)", color="#e6a15a", fontsize=13, pad=8)
    axes[1].imshow(np.clip(layers["dens"], 0, 2), cmap="magma", extent=[0, 1, 1, 0])
    axes[1].set_title("Metaball density\n(Poisson-noisy interior)", color="#dfe6ee", fontsize=13, pad=8)
    axes[2].imshow(layers["thick"], cmap="viridis", extent=[0, 1, 1, 0])
    axes[2].set_title("Filled mask -> distance thickness\n(smooth solid volume)",
                      color="#dfe6ee", fontsize=13, pad=8)
    axes[3].imshow(filled)
    axes[3].set_title("Filled reconstruction\n(no interior holes)", color="#7fd0ff", fontsize=13, pad=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#26313d")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.86, bottom=0.01, wspace=0.03)
    fig.savefig(out_path, dpi=100, facecolor="#0a0e14")
    plt.close(fig)


# Frame choices are validated against the iteration renders (same sim params, N=130). Kept as
# defaults; the montage lets them be re-picked after viewing, and --finalize re-renders from the
# saved sim without re-simulating.
HEROES = {"balldrop": 58, "dambreak": 118}
COMPARE = {"balldrop": 58, "dambreak": 118}   # frame used in the before/after (matched to old hero)
BREAKDOWN_FRAME = 18                            # thick collapsing column -> most dramatic holes
MONTAGE = {"balldrop": [40, 48, 55, 58, 64, 72, 90],
           "dambreak": [18, 30, 55, 100, 112, 118, 124]}


def run_all(out_dir, quick=False):
    os.makedirs(out_dir, exist_ok=True)
    nf = 60 if quick else 130          # both scenes: N=130 @ 60fps (balldrop ~2.0x, dambreak ~1.4x slow-mo)
    stats = {}
    xs_bd, vs_bd, st_bd, tank_bd, L_bd, fr_bd = render_scene_to_disk(
        "balldrop", out_dir, HEROES["balldrop"], nf, montage_idxs=MONTAGE["balldrop"])
    stats["balldrop"] = bool(st_bd)
    xs_db, vs_db, st_db, tank_db, L_db, fr_db = render_scene_to_disk(
        "dambreak", out_dir, HEROES["dambreak"], nf, montage_idxs=MONTAGE["dambreak"])
    stats["dambreak"] = bool(st_db)
    finalize(out_dir)
    print("stability:", stats)
    return stats


def finalize(out_dir, heroes=None, compare=None, breakdown_frame=None):
    """Rebuild heroes, before/after, and the interior-fill breakdown from the persisted sim data
    (no re-simulation), using chosen frame indices."""
    heroes = heroes or HEROES
    compare = compare or COMPARE
    breakdown_frame = BREAKDOWN_FRAME if breakdown_frame is None else breakdown_frame
    prior = os.path.join(os.path.dirname(out_dir), "non-differentiable-fluid-renderer")
    L = dict(DEFAULT_LOOK)
    tank = make_tank(L["res"], L)
    scenes = {}
    for sc in ("balldrop", "dambreak"):
        xs = np.load(os.path.join(out_dir, f"_sim_{sc}_xs.npy"))
        vs = np.load(os.path.join(out_dir, f"_sim_{sc}_vs.npy"))
        scenes[sc] = (xs, vs)
        hf = min(heroes[sc], xs.shape[0] - 1)
        save_png(os.path.join(out_dir, f"{sc}_hero.png"),
                 render_frame(xs[hf], vs[hf], tank, L, frame=hf))
        cf = min(compare[sc], xs.shape[0] - 1)
        old_png = os.path.join(prior, f"{sc}_hero.png")
        label = "Ball drop" if sc == "balldrop" else "Dam break"
        build_before_after(old_png, render_frame(xs[cf], vs[cf], tank, L, frame=cf),
                           label, os.path.join(out_dir, f"before_after_{sc}.png"))
    xs, vs = scenes["dambreak"]
    bf = min(breakdown_frame, xs.shape[0] - 1)
    build_interior_breakdown(xs[bf], vs[bf], tank, L,
                            os.path.join(out_dir, "interior_fill_breakdown.png"))
    print(f"finalized heroes={heroes} compare={compare} breakdown={breakdown_frame}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--probe", nargs=2, metavar=("SCENE", "FRAME"))
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--nframes", type=int, default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--hero", nargs=2, metavar=("BD", "DB"), type=int, default=None)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo, "runs", "realistic-rendering", "improve-basic-fluid-sim-realism")
    os.makedirs(out_dir, exist_ok=True)
    if args.sweep:
        # quick motion sanity: report peak/mean speed for a few stiffnesses (no rendering)
        for E, dt in [(170, 1.2e-4), (350, 9e-5), (520, 7e-5), (760, 6e-5)]:
            cfg = scene_balldrop(quick=True)
            xs, vs, st = run_scene(cfg["pts"], E, dt, cfg["T"], 70, fric=cfg["fric"],
                                   area=cfg["area"], flip=cfg["flip"])
            sp = np.linalg.norm(vs, axis=2)
            Jspread = "n/a"
            print(f"E={E:4.0f} dt={dt:.1e} stable={st} peak_speed={sp.max():6.2f} "
                  f"mean={sp.mean():.3f}")
    elif args.probe:
        sc, fr = args.probe[0], int(args.probe[1])
        tag = "_naive" if args.naive else ""
        probe(sc, fr, os.path.join(out_dir, f"_probe_{sc}_{fr}{tag}.png"),
              quick=args.quick, naive=args.naive, n_frames=args.nframes)
    elif args.finalize:
        heroes = None
        compare = None
        if args.hero:
            heroes = {"balldrop": args.hero[0], "dambreak": args.hero[1]}
            compare = dict(heroes)
        finalize(out_dir, heroes=heroes, compare=compare)
    else:
        run_all(out_dir, quick=args.quick)
