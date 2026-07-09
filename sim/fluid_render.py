"""Non-differentiable realistic fluid renderer for the 2D MLS-MPM fluid.

The physics is the easy part: a weakly compressible MPM fluid (the same constitutive law used in
``sim/material_showcase.py``, fluid path only) is rolled forward and its per-frame particle positions and
velocities are exported. The DELIVERABLE is the renderer that turns that cloud of dots into frames that read
as a real liquid seen side-on, like water in a glass tank.

Rendering pipeline (screen-space fluid, all numpy + scipy, headless -> mp4):
  1. Splat particles into a high-res density field with a Gaussian kernel (metaballs); the liquid is a
     density isocontour, smoothed for an anti-aliased surface.
  2. Build a screen-space surface normal from the density gradient: flat interior faces the viewer, the rim
     tilts to graze, which drives Fresnel and specular.
  3. Shade the body: Beer-Lambert depth/absorption color (thicker = deeper, more saturated), REFRACTION of a
     structured background offset by the surface normal (this sells realism most), Fresnel blend of an
     environment reflection vs the refracted body, Blinn-Phong specular glints, and a soft interior
     ambient-occlusion darkening.
  4. Foam / spray: bright whitewater where the fluid is thin/sparse (droplets, sheet edges) or moving fast
     (crests, splash crowns).
  5. Finish: bloom on the highlights, vignette, a subtle floor reflection, tone map, encode to mp4.

Two scenes: a fluid ball dropped into a shallow pool (splash crown + droplets) and a dam-break column
collapse (rolling wave into the far wall). Each is written as an mp4 plus a hero still, and a "how it's
built" breakdown figure shows particles -> density/surface -> shaded.

Usage:
    python sim/fluid_render.py            # full render of both scenes + breakdown
    python sim/fluid_render.py --quick    # short low-res smoke test
    python sim/fluid_render.py --probe SCENE FRAME   # render a single frame PNG for the look-loop
"""
import argparse
import datetime
import json
import os

import numpy as np
import taichi as ti
from scipy.ndimage import gaussian_filter, maximum_filter

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

MAX_P = 40000

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
def g2p(n: ti.i32, dt: ti.f32):
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
        v[p] = new_v
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


def run_scene(pts, E, dt, T, n_frames, fric=0.1, area=None):
    """Roll the fluid forward to physical time T, capturing n_frames snapshots of (positions, velocities)."""
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
            g2p(n, dt)
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
# All rendering is numpy/scipy. Domain is [0,1]^2 with y up; images are y-down, so y is flipped on splat.

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def splat_field(pos, res, weights=None):
    """Histogram particles into a res x res grid (image orientation: row 0 = top = high y)."""
    col = np.clip(pos[:, 0] * res, 0, res - 1e-3)
    row = np.clip((1.0 - pos[:, 1]) * res, 0, res - 1e-3)
    grid, _, _ = np.histogram2d(row, col, bins=res, range=[[0, res], [0, res]], weights=weights)
    return grid


def build_background(res, tank):
    """A pleasing studio background: cool-to-warm vertical gradient, soft light bars (so refraction is
    visible), a horizon glow and a darker floor slab. Returns an (res,res,3) float image in [0,1]."""
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    v = yy / res  # 0 top .. 1 bottom
    u = xx / res
    # vertical gradient: deep slate at top easing to a warmer muted teal lower down
    top = np.array([0.07, 0.11, 0.16])
    mid = np.array([0.10, 0.16, 0.21])
    bot = np.array([0.13, 0.17, 0.19])
    g = smoothstep(0.0, 0.6, v)[..., None]
    bg = top * (1 - g) + mid * g
    g2 = smoothstep(0.5, 1.0, v)[..., None]
    bg = bg * (1 - g2) + bot * g2
    # soft vertical light bars (studio softboxes) - these reveal refraction and give specular something to
    # catch. Placed off-center so bending is asymmetric and legible.
    # Soft localized softbox glows (2D gaussians) rather than full-height columns: they give the water a
    # bright, structured thing to reflect and refract without reading as light shafts.
    for cx, cyv, wu, wv, inten, tint in [
            (0.22, 0.34, 0.07, 0.11, 0.13, (0.82, 0.90, 1.0)),
            (0.78, 0.30, 0.09, 0.12, 0.10, (1.0, 0.94, 0.82))]:
        glow = np.exp(-((u - cx) ** 2) / (2 * wu ** 2) - ((v - cyv) ** 2) / (2 * wv ** 2))
        bg = bg + glow[..., None] * inten * np.array(tint)
    # horizon glow just above the floor line
    fy = 1.0 - tank["floor"]
    glow = np.exp(-((v - fy) ** 2) / (2 * 0.03 ** 2))
    bg = bg + glow[..., None] * np.array([0.06, 0.08, 0.10])
    # floor slab
    floor_mask = smoothstep(fy - 0.005, fy + 0.005, v)
    floor_col = np.array([0.05, 0.06, 0.08])
    bg = bg * (1 - floor_mask[..., None]) + floor_col * floor_mask[..., None]
    return np.clip(bg, 0, 1).astype(np.float32)


def sample_bilinear(img, cols, rows):
    """Sample img (H,W,3) at fractional (cols,rows) with clamped bilinear interpolation."""
    H, W = img.shape[:2]
    c = np.clip(cols, 0, W - 1.001)
    r = np.clip(rows, 0, H - 1.001)
    c0 = np.floor(c).astype(np.int32); r0 = np.floor(r).astype(np.int32)
    c1 = c0 + 1; r1 = r0 + 1
    fc = (c - c0)[..., None]; fr = (r - r0)[..., None]
    a = img[r0, c0]; b = img[r0, c1]; cc = img[r1, c0]; d = img[r1, c1]
    return (a * (1 - fc) * (1 - fr) + b * fc * (1 - fr) + cc * (1 - fc) * fr + d * fc * fr)


DEFAULT_LOOK = dict(
    res=1000,
    sigma_px=8.0,          # density blur (blob smoothness)
    iso=0.5,               # surface isovalue (fraction of typical interior density)
    edge=0.5,              # smoothstep half-width around iso for AA
    normal_k=2.0,          # curvature constant: larger = flatter (more viewer-facing)
    normal_amp=3.2,        # in-plane normal strength from gradient
    refract=58.0,          # background refraction offset in pixels at full thickness
    absorb=1.85,           # Beer-Lambert absorption per unit thickness (depth contrast)
    liquid=(0.02, 0.14, 0.27),   # deep water body color (dark blue)
    shallow=(0.14, 0.40, 0.52),  # thin-film color (cyan)
    F0=0.02,               # water Fresnel reflectance at normal incidence
    rim=0.35,              # extra brightening of the grazing waterline edge
    light=(-0.55, 0.72),   # key light direction in image plane (x right, y UP)
    shininess=80.0,
    spec_gain=2.6,
    sheen=0.12,            # broad wet-sheen specular (kept small: flat interior would wash out)
    foam_speed=1.5,        # speed above which motion foam appears
    foam_gain=1.35,
    bloom=0.42,
    vignette=0.35,
)


def render_frame(pos, vel, tank, look=None, return_layers=False):
    """Render one frame from particle positions (N,2 in [0,1]) and velocities (N,2). Returns uint8 RGB."""
    L = dict(DEFAULT_LOOK)
    if look:
        L.update(look)
    res = L["res"]
    speed = np.linalg.norm(vel, axis=1)

    # --- 1. density + thickness -------------------------------------------------------------------
    raw = splat_field(pos, res)
    dens = gaussian_filter(raw, L["sigma_px"])
    # normalize to a stable interior level using a high percentile of occupied cells
    ref = np.percentile(dens[dens > 1e-4], 80) if np.any(dens > 1e-4) else 1.0
    d = dens / (ref + 1e-9)
    surf = smoothstep(L["iso"] - L["edge"], L["iso"] + L["edge"], d)   # 0..1 anti-aliased liquid mask
    thick = np.clip(d, 0.0, 3.0)                                       # optical thickness proxy

    # --- 2. screen-space normal -------------------------------------------------------------------
    # gradient of the (smoothed) thickness; interior flat -> faces viewer, rim steep -> grazes.
    ds = gaussian_filter(d, 2.0)
    grow, gcol = np.gradient(ds)               # d/drow (down), d/dcol (right)
    nx = -gcol * L["normal_amp"]               # image x (right)
    ny = grow * L["normal_amp"]                # image y UP (row grows down, so flip sign)
    nz = np.full_like(nx, L["normal_k"])
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / nn, ny / nn, nz / nn

    # --- 3a. refraction of the background ----------------------------------------------------------
    bg = tank["bg"]
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    off = L["refract"] * np.clip(thick, 0, 1.5)
    scol = xx + nx * off
    srow = yy - ny * off                      # ny is up, image row is down
    refr = sample_bilinear(bg, scol, srow)

    # --- 3b. Beer-Lambert absorption / depth color -------------------------------------------------
    liquid = np.array(L["liquid"]); shallow = np.array(L["shallow"])
    tt = np.clip(thick, 0, 3.0)[..., None]
    transmit = np.exp(-L["absorb"] * tt)      # fraction of background surviving the liquid
    body_tint = shallow * transmit + liquid * (1 - transmit)   # thin->shallow, thick->deep
    body = refr * transmit + body_tint * (1 - transmit)

    # interior ambient occlusion: deep interior slightly darker for volume
    ao = 1.0 - 0.18 * smoothstep(0.7, 2.2, thick)
    body = body * ao[..., None]

    # --- 3c. Fresnel reflection (environment) ------------------------------------------------------
    cos_t = np.clip(nz, 0, 1)
    fres = L["F0"] + (1 - L["F0"]) * (1 - cos_t) ** 5
    # environment reflection: reflect the view (0,0,1) about N -> r = (2 nz nx, 2 nz ny, 2nz^2-1); sample a
    # sky gradient by the reflected up-component (bright sky up, darker down).
    rup = 2 * nz * ny                          # reflected ray up-component (image y up)
    sky = np.clip(0.55 + 0.45 * rup, 0, 1)[..., None]
    env = np.array([0.58, 0.72, 0.88]) * sky + np.array([0.10, 0.13, 0.17]) * (1 - sky)
    col = body * (1 - fres[..., None]) + env * fres[..., None]

    # explicit rim light: the bright waterline where the surface grazes the view. (1-nz) peaks at the
    # silhouette edge; this is the crisp bright edge that reads as a curved liquid meniscus.
    rimg = L["rim"] * (1.0 - cos_t) ** 3
    col = col + rimg[..., None] * np.array([0.35, 0.58, 0.78])

    # --- 3d. Blinn-Phong specular (a tight glint + a broad wet sheen) -------------------------------
    lx, ly = L["light"]; lz = 0.55
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    hx, hy, hz = lx + 0.0, ly + 0.0, lz + 1.0   # half vector with view (0,0,1)
    hn = np.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / hn, hy / hn, hz / hn
    ndh = np.clip(nx * hx + ny * hy + nz * hz, 0, 1)
    spec = L["spec_gain"] * ndh ** L["shininess"]        # tight bright glint
    sheen = L["sheen"] * ndh ** 8.0                       # broad soft sheen (wet look)
    col = col + (spec + sheen)[..., None] * np.array([1.0, 1.0, 0.97])

    # --- 4. foam / spray ---------------------------------------------------------------------------
    # Foam is aerated water: it appears where the surface is fast-moving (crests, crowns) or where the
    # liquid is genuinely thin and broken up (spray, droplet edges). It must NOT paint the whole body.
    sp_sum = gaussian_filter(splat_field(pos, res, weights=speed), L["sigma_px"])
    avg_speed = sp_sum / (dens + 1e-6)
    motion = smoothstep(L["foam_speed"], L["foam_speed"] * 2.2, avg_speed)   # only fast fluid
    # A "surface band": near the air interface, where |grad density| is large. Foam lives here, never in the
    # calm interior, which is what stops density noise from speckling the body with white.
    edgemag = np.hypot(*np.gradient(gaussian_filter(surf, 3.0)))
    band = smoothstep(0.004, 0.02, edgemag)
    # spray: the thin broken fringe (crown tips, sheet edges, droplets), where thickness is well below 1.
    # Small isolated droplets are thin everywhere, so whitening thin fluid makes them read as solid spray
    # rather than hollow refractive rings.
    thinfoam = surf * smoothstep(0.72, 0.2, thick)
    # motion foam: fast-moving fluid at the surface band (crests, crowns), not the deep calm interior.
    motionfoam = motion * band
    foam = np.clip(L["foam_gain"] * (0.9 * motionfoam + 0.85 * thinfoam), 0, 1)
    foam = gaussian_filter(foam, 2.2)
    foam_col = np.array([0.93, 0.97, 1.0])
    # foam is opaque white where strong, but let a little body tint bleed through the light foam
    col = col * (1 - foam[..., None]) + foam_col * foam[..., None]

    # --- composite liquid over background ----------------------------------------------------------
    m = surf[..., None]
    img = bg * (1 - m) + col * m

    # --- floor reflection of the liquid (subtle) ---------------------------------------------------
    fy_px = int((1.0 - tank["floor"]) * res)
    if fy_px < res - 2:
        strip = img[:fy_px]
        refl = strip[::-1]                       # mirror the scene about the floor line
        h = res - fy_px
        refl = refl[:h]
        fade = (1.0 - np.linspace(0, 1, refl.shape[0]))[:, None, None] * 0.22
        img[fy_px:fy_px + refl.shape[0]] = (
            img[fy_px:fy_px + refl.shape[0]] * (1 - fade) + refl * fade)

    # --- 5. finish: bloom, vignette, tone map ------------------------------------------------------
    bright = np.clip(img - 0.75, 0, None)
    bloom = gaussian_filter(bright, (L["res"] / 130.0, L["res"] / 130.0, 0))
    img = img + L["bloom"] * bloom
    # vignette
    cy, cx = res / 2, res / 2
    r2 = ((xx - cx) / res) ** 2 + ((yy - cy) / res) ** 2
    vig = 1.0 - L["vignette"] * smoothstep(0.12, 0.42, r2)
    img = img * vig[..., None]
    # filmic-ish tone map + gamma
    img = img / (img + 0.9)
    img = np.clip(img * 1.55, 0, 1) ** (1 / 1.15)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    if return_layers:
        return out, dict(dens=d, surf=surf, thick=thick, normal=np.dstack([nx, ny, nz]),
                         refr=refr, foam=foam)
    return out


def encode_mp4(path, frames, fps=30):
    import imageio
    even = [f[: f.shape[0] - f.shape[0] % 2, : f.shape[1] - f.shape[1] % 2] for f in frames]
    imageio.mimwrite(path, even, fps=fps, quality=9, macro_block_size=1, codec="libx264")


def save_png(path, img):
    import imageio
    imageio.imwrite(path, img)


# =========================================================================== SCENES
def scene_balldrop(quick=False):
    rng = np.random.default_rng(0)
    wall = floor_y
    # shallow pool spanning the tank, plus a ball released above it
    n_pool = 6000 if quick else 16000
    n_ball = 1800 if quick else 5000
    pool = seed_box(wall + 0.01, 1 - wall - 0.01, wall, 0.17, n_pool, rng)
    ball = seed_disk((0.5, 0.66), 0.085, n_ball, rng)
    pts = np.concatenate([pool, ball], axis=0)
    area = (1 - 2 * wall) * (0.17 - wall) + np.pi * 0.085 ** 2
    return dict(pts=pts, area=area, E=170.0, dt=1.2e-4, fric=0.08,
                T=1.15, hero=None)


def scene_dambreak(quick=False):
    rng = np.random.default_rng(1)
    wall = floor_y
    n = 8000 if quick else 24000
    col = seed_box(wall, 0.32, wall, 0.74, n, rng)
    area = (0.32 - wall) * (0.74 - wall)
    return dict(pts=col, area=area, E=170.0, dt=1.1e-4, fric=0.06,
                T=1.7, hero=None)


SCENES = {"balldrop": scene_balldrop, "dambreak": scene_dambreak}


def make_tank(res, look=None):
    L = dict(DEFAULT_LOOK)
    if look:
        L.update(look)
    tank = {"floor": floor_y}
    tank["bg"] = build_background(res, tank)
    return tank


def probe(scene_name, frame_idx, out_path, look=None, quick=True):
    """Render a single frame of a scene to a PNG for the render->look->improve loop."""
    cfg = SCENES[scene_name](quick=quick)
    L = dict(DEFAULT_LOOK)
    if look:
        L.update(look)
    n_frames = 60 if quick else 150
    xs, vs, stable = run_scene(cfg["pts"], cfg["E"], cfg["dt"], cfg["T"], n_frames,
                               fric=cfg["fric"], area=cfg["area"])
    print(f"  {scene_name}: stable={stable} frames={n_frames} particles={xs.shape[1]}")
    tank = make_tank(L["res"], L)
    fi = min(frame_idx, n_frames - 1)
    img = render_frame(xs[fi], vs[fi], tank, L)
    save_png(out_path, img)
    print(f"  wrote {out_path}  (frame {fi}/{n_frames})")
    return xs, vs


def render_breakdown(pos, vel, tank, look, path):
    """A 'how it's built' figure: raw particles -> density (metaballs) -> reconstructed surface (clay
    normal shading) -> final shaded render. Teaches the pipeline in one image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, layers = render_frame(pos, vel, tank, look, return_layers=True)
    d = layers["dens"]; nrm = layers["normal"]; surf = layers["surf"]
    res = look["res"]
    # clay shading of the reconstructed surface: Lambert from the normal field (no color, no refraction)
    lx, ly, lz = -0.45, 0.6, 0.66
    ln = np.sqrt(lx * lx + ly * ly + lz * lz); lx, ly, lz = lx / ln, ly / ln, lz / ln
    diff = np.clip(nrm[..., 0] * lx + nrm[..., 1] * ly + nrm[..., 2] * lz, 0, 1)
    clay = (0.18 + 0.82 * diff) * surf
    clay_rgb = np.dstack([clay * 0.7, clay * 0.8, clay])  # cool clay

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.2), facecolor="#0a0e14")
    titles = ["1. MPM particles", "2. Density field (metaballs)",
              "3. Reconstructed surface + normals", "4. Final shaded render"]
    # panel 1: raw particles
    col = pos[:, 0]; row = 1.0 - pos[:, 1]
    axes[0].scatter(col, row, s=1.2, c="#4db6ff", edgecolors="none", alpha=0.55)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(1, 0); axes[0].set_facecolor("#0a0e14")
    # panel 2: density
    axes[1].imshow(np.clip(d, 0, 2), cmap="mako" if "mako" in plt.colormaps() else "viridis",
                   extent=[0, 1, 1, 0])
    # panel 3: clay
    axes[2].imshow(np.clip(clay_rgb, 0, 1), extent=[0, 1, 1, 0])
    # panel 4: final
    axes[3].imshow(img, extent=[0, 1, 1, 0])
    for ax, t in zip(axes, titles):
        ax.set_title(t, color="#dfe6ee", fontsize=13, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#26313d")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.03)
    fig.savefig(path, dpi=100, facecolor="#0a0e14")
    plt.close(fig)


def render_scene_to_disk(scene_name, out_dir, hero_frame, n_frames=150, fps=30, look=None):
    """Full render: roll the scene, encode the mp4, save the hero still. Returns (xs, vs, stable)."""
    cfg = SCENES[scene_name](quick=False)
    L = dict(DEFAULT_LOOK)
    if look:
        L.update(look)
    xs, vs, stable = run_scene(cfg["pts"], cfg["E"], cfg["dt"], cfg["T"], n_frames,
                               fric=cfg["fric"], area=cfg["area"])
    print(f"  {scene_name}: stable={stable} particles={xs.shape[1]} frames={n_frames}")
    tank = make_tank(L["res"], L)
    import time
    t0 = time.time()
    frames = []
    for f in range(n_frames):
        frames.append(render_frame(xs[f], vs[f], tank, L))
    print(f"  rendered {n_frames} frames in {time.time()-t0:.1f}s")
    encode_mp4(os.path.join(out_dir, f"{scene_name}.mp4"), frames, fps=fps)
    hf = min(hero_frame, n_frames - 1)
    save_png(os.path.join(out_dir, f"{scene_name}_hero.png"), frames[hf])
    return xs, vs, stable, tank, L


def run_all(out_dir, n_frames=150, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    heroes = {"balldrop": 55, "dambreak": 115}
    stats = {}
    xs_bd, vs_bd, st_bd, tank_bd, L_bd = render_scene_to_disk("balldrop", out_dir, heroes["balldrop"],
                                                              n_frames, fps)
    stats["balldrop"] = st_bd
    xs_db, vs_db, st_db, tank_db, L_db = render_scene_to_disk("dambreak", out_dir, heroes["dambreak"],
                                                              n_frames, fps)
    stats["dambreak"] = st_db
    # breakdown from the dam-break wave-front frame (teaches the pipeline well)
    render_breakdown(xs_db[25], vs_db[25], tank_db, L_db,
                     os.path.join(out_dir, "breakdown.png"))
    print("stability:", stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--probe", nargs=2, metavar=("SCENE", "FRAME"))
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo, "runs", "realistic-rendering", "non-differentiable-fluid-renderer")
    if args.probe:
        sc, fr = args.probe[0], int(args.probe[1])
        os.makedirs(out_dir, exist_ok=True)
        probe(sc, fr, os.path.join(out_dir, f"_probe_{sc}_{fr}.png"), quick=args.quick)
    else:
        nf = 60 if args.quick else 150
        run_all(out_dir, n_frames=nf, fps=30)
