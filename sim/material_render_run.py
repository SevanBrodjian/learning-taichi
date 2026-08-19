"""Driver for the four-material rendering proposals: renders every clip, still and cost number.

The renderer itself (treatments, baseline port, scenes) lives in ``sim/material_render.py``. This file
only decides WHAT gets rendered and how it is laid out for review, because every claim in the task
needs its own artifact:

  * each material, current vs proposed, as VIDEO, on two different scenes;
  * two competing options per material, side by side with the current look;
  * all four together on ONE shared grid (distinctness is a property of the set);
  * the greyscale test — every material forced to the SAME albedo, output converted to luminance —
    which is the only honest way to check the complaint "they differ only in hue";
  * a measured per-frame render cost for every treatment, in the same harness as the current one.

Usage:
    python sim/material_render_run.py           # the full deliverable
    python sim/material_render_run.py --quick   # 30-frame clips, for a smoke test
    python sim/material_render_run.py --bench   # just the cost table
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sim.material_render import (ALTERNATE, DEMO_DENSITY, DEMO_ISO, DEMO_RADIUS, LABEL,  # noqa: E402
                                 MAT_ID, ORDER, PROPOSAL, RES, bench, blur, build_masks, clear_s,
                                 densb, distr, encode, fillm, label_panels, nrm, normals_from,
                                 physics, render_frame, ripple_into, run_fill, run_fourup,
                                 run_pool, run_solo, save_png, set_palette, tmpa, tmpb, tmpc,
                                 upload_frame, upload_static, velocities, _font)

RUN_DIR = os.path.join(_ROOT, "runs", "material-variants",
                       "propose-new-rendering-for-each-of-the-four-materials")

# What each treatment costs in PASSES over the frame. The demo's current renderer is two passes, so
# this table is the reasoned half of the cost claim and the measurement is the empirical half.
PASSES = {
    "current": "1 particle splat + 1 full-screen resolve",
    "water_glass": "1 splat + 1 speed splat + 4 separable blurs (8 passes) + 4 morphological disk "
                   "passes + 10 jump-flood passes + 1 shade",
    "water_film": "1 splat + 1 speed splat + 4 separable blurs (8 passes) + 2 morphological disk "
                  "passes + 10 jump-flood passes + 1 shade (no background sampling)",
    "rubber_tex": "1 splat + 1 material-coord splat + 4 separable blurs + 6 morphological disk "
                  "passes + 10 jump-flood passes + 1 shade",
    "rubber_flat": "1 splat + 4 separable blurs + 6 morphological disk passes + 10 jump-flood "
                   "passes + 1 shade",
    "snow_powder": "1 particle splat + 1 full-screen resolve -- exactly the current renderer's "
                   "shape; only the shading arithmetic differs",
    "snow_current": "1 particle splat + 1 full-screen resolve",
    "sand_grain": "1 splat + 1 grain-sprite pass (6 sprites/particle) + 3 separable blurs + 4 "
                  "morphological disk passes + 10 jump-flood passes + 1 shade",
    "sand_bare": "1 splat + 1 grain-sprite pass (6 sprites/particle) + 3 separable blurs + 4 "
                 "morphological disk passes + 10 jump-flood passes + 1 shade",
}
NICE = {"current": "current", "water_glass": "glass (A)", "water_film": "tinted film (B)",
        "rubber_tex": "border + printed grid (A)", "rubber_flat": "border, flat body (B)",
        "snow_powder": "powder (A)", "snow_current": "current, unchanged (B)",
        "sand_grain": "grains over a packed body (A)", "sand_bare": "loose grains (B)"}
KINDS = ["current", "water_glass", "water_film", "rubber_tex", "rubber_flat", "snow_powder",
         "sand_grain", "sand_bare"]


def downs(a, k=2):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k, a.shape[2]).mean((1, 3)).astype(np.uint8)


def tile2x2(frames, titles):
    from PIL import Image, ImageDraw
    q = [downs(f) for f in frames]
    h, w = q[0].shape[:2]
    out = Image.new("RGB", (2 * w + 3, 2 * h + 3), (10, 14, 20))
    d = ImageDraw.Draw(out)
    fnt = _font(max(12, int(0.055 * w)))
    for k, (p, t) in enumerate(zip(q, titles)):
        ox, oy = (k % 2) * (w + 3), (k // 2) * (h + 3)
        out.paste(Image.fromarray(p), (ox, oy))
        d.text((ox + 8, oy + 7), t, font=fnt, fill=(226, 236, 246),
               stroke_width=3, stroke_fill=(6, 9, 14))
    return np.asarray(out)


def clip(rollout, kinds, ref, grey=False):
    snaps, times, mats = rollout["snaps"], rollout["times"], rollout["mats"]
    vel = velocities(snaps, times)
    return [render_frame(snaps[i], mats, vel[i], snaps[0], kinds, ref, grey=grey)
            for i in range(snaps.shape[0])]


def main(quick=False):
    os.makedirs(RUN_DIR, exist_ok=True)
    upload_static()
    ref = DEMO_DENSITY / (RES * RES)
    nf = 30 if quick else 90

    def w_mp4(name, frames, fps=30):
        encode(os.path.join(RUN_DIR, name), frames, fps)
        print("  ->", name, len(frames), "frames", flush=True)

    def w_png(name, arr):
        save_png(os.path.join(RUN_DIR, name), arr)
        print("  ->", name, flush=True)

    stills, meta, greys = {}, {}, {}
    for scene_name, runner in (("fill", lambda m: run_fill(m, nf)),
                               ("slam", lambda m: run_solo("slam", m, nf))):
        for m in ORDER:
            print("[sim]", scene_name, m, flush=True)
            ro = runner(m)
            meta["%s/%s" % (scene_name, m)] = dict(n=int(ro["snaps"].shape[1]),
                                                   stable=ro["stable"], T=float(ro["times"][-1]))
            cur = clip(ro, {m: "current"}, ref)
            pro = clip(ro, {m: PROPOSAL[m]}, ref)
            alt = clip(ro, {m: ALTERNATE[m]}, ref)
            w_mp4("clip_%s_%s_current.mp4" % (scene_name, m), cur)
            w_mp4("clip_%s_%s_proposed.mp4" % (scene_name, m), pro)
            w_mp4("clip_%s_%s_alt.mp4" % (scene_name, m), alt)
            w_mp4("cmp_%s_%s.mp4" % (scene_name, m),
                  [label_panels([a, b], ["CURRENT  -  " + LABEL[m],
                                         "PROPOSED  -  " + LABEL[m] + "  -  " + NICE[PROPOSAL[m]]])
                   for a, b in zip(cur, pro)])
            w_mp4("opt_%s_%s.mp4" % (scene_name, m),
                  [label_panels([a, b, c], ["CURRENT", "OPTION A  " + NICE[PROPOSAL[m]],
                                            "OPTION B  " + NICE[ALTERNATE[m]]])
                   for a, b, c in zip(cur, pro, alt)])
            hero = int(0.62 * len(cur))
            if scene_name == "fill":
                w_png("still_%s_current.png" % m, cur[hero])
                w_png("still_%s_proposed.png" % m, pro[hero])
                w_png("still_%s_alt.png" % m, alt[hero])
                stills[m] = (cur[hero], pro[hero], alt[hero])
                greys[m] = (clip(ro, {m: "current"}, ref, grey=True),
                            clip(ro, {m: PROPOSAL[m]}, ref, grey=True))

    # ---------------------------------------------------------------- the greyscale test, tiled
    nfr = min(len(greys[m][0]) for m in ORDER)
    tc = [tile2x2([greys[m][0][i] for m in ORDER], [LABEL[m] for m in ORDER]) for i in range(nfr)]
    tp = [tile2x2([greys[m][1][i] for m in ORDER], [LABEL[m] for m in ORDER]) for i in range(nfr)]
    w_mp4("grey_tiles_current.mp4", tc)
    w_mp4("grey_tiles_proposed.mp4", tp)
    w_mp4("cmp_grey_tiles.mp4",
          [label_panels([a, b], ["GREYSCALE CURRENT  -  identical albedo for all four",
                                 "GREYSCALE PROPOSED  -  identical albedo for all four"])
           for a, b in zip(tc, tp)])
    hg = int(0.62 * nfr)
    w_png("grey_tiles_current.png", tc[hg])
    w_png("grey_tiles_proposed.png", tp[hg])
    del greys

    # ---------------------------------------------------------------- all four on ONE grid
    print("[sim] four_up", flush=True)
    fu = run_fourup(nf, T=1.05)
    meta["four_up"] = dict(n=int(fu["snaps"].shape[1]), stable=fu["stable"], dt=fu["dt"])
    cur_k = {m: "current" for m in ORDER}
    pro_k = dict(PROPOSAL)
    fc, fp = clip(fu, cur_k, ref), clip(fu, pro_k, ref)
    fgc, fgp = clip(fu, cur_k, ref, grey=True), clip(fu, pro_k, ref, grey=True)
    for nm, fr in (("clip_fourup_current.mp4", fc), ("clip_fourup_proposed.mp4", fp),
                   ("clip_fourup_grey_current.mp4", fgc), ("clip_fourup_grey_proposed.mp4", fgp)):
        w_mp4(nm, fr)
    w_mp4("cmp_fourup.mp4", [label_panels([a, b], ["CURRENT  -  one grid, four materials",
                                                   "PROPOSED  -  one grid, four materials"])
                             for a, b in zip(fc, fp)])
    w_mp4("cmp_fourup_grey.mp4",
          [label_panels([a, b], ["GREYSCALE CURRENT  -  water rubber snow sand, left to right",
                                 "GREYSCALE PROPOSED  -  water rubber snow sand, left to right"])
           for a, b in zip(fgc, fgp)])
    hf = int(0.55 * len(fc))
    w_png("still_fourup_current.png", fc[hf])
    w_png("still_fourup_proposed.png", fp[hf])
    w_png("still_fourup_grey_current.png", fgc[hf])
    w_png("still_fourup_grey_proposed.png", fgp[hf])

    # ---------------------------------------------------------------- canonical buoyancy scene
    print("[sim] pool_elastic", flush=True)
    po = run_pool("elastic", nf)
    meta["pool_elastic"] = dict(n=int(po["snaps"].shape[1]), stable=po["stable"], dt=po["dt"])
    pc, pp_ = clip(po, cur_k, ref), clip(po, pro_k, ref)
    w_mp4("clip_pool_current.mp4", pc)
    w_mp4("clip_pool_proposed.mp4", pp_)
    w_mp4("cmp_pool.mp4", [label_panels([a, b], ["CURRENT  -  rubber submerged in water",
                                                 "PROPOSED  -  rubber submerged in water"])
                           for a, b in zip(pc, pp_)])
    hp = int(0.55 * len(pc))
    w_png("still_pool_current.png", pc[hp])
    w_png("still_pool_proposed.png", pp_[hp])

    # ---------------------------------------------------------------- contact sheet
    rows = [label_panels(list(stills[m]), [LABEL[m] + "  CURRENT",
                                           LABEL[m] + "  A: " + NICE[PROPOSAL[m]],
                                           LABEL[m] + "  B: " + NICE[ALTERNATE[m]]])
            for m in ORDER]
    wmax = max(r.shape[1] for r in rows)
    w_png("contact_sheet.png",
          np.concatenate([np.pad(r, ((0, 8), (0, wmax - r.shape[1]), (0, 0))) for r in rows], 0))

    # ---------------------------------------------------------------- water pipeline breakdown
    print("[sim] breakdown", flush=True)
    ro = run_fill("fluid", nf)
    bi = int(0.45 * nf)
    vv = velocities(ro["snaps"], ro["times"])
    set_palette(False)
    n = upload_frame(ro["snaps"][bi], ro["mats"], vv[bi], ro["snaps"][0])
    build_masks(n, MAT_ID["fluid"], ref, 5.0, iso_fill=0.28, close_r=6)
    lay = [("particle density", np.clip(densb.to_numpy() * 0.9, 0, 1)),
           ("filled interior mask", fillm.to_numpy()),
           ("thickness (distance transform)", np.clip(distr.to_numpy() / 60.0, 0, 1))]
    blur(fillm, tmpc, 2.5, tmpa)
    ripple_into(0.22)
    blur(tmpb, tmpc, 1.6, tmpa)
    normals_from(tmpc, 3.0 * (RES / 1080.0), 2.1)
    # the in-plane normal components are amplified 5x for DISPLAY only: in the interior the
    # surface faces the viewer and the true xy tilt is near zero, which is the whole reason
    # Fresnel keeps the body clear and lights only the rim.
    nv = nrm.to_numpy()
    nvis = np.dstack([np.clip(nv[..., 0] * 5.0 + 0.5, 0, 1),
                      np.clip(nv[..., 1] * 5.0 + 0.5, 0, 1),
                      np.clip(nv[..., 2], 0, 1)])
    imgs = [np.dstack([a, a, a]) for _, a in lay] + [nvis]
    names = [t for t, _ in lay] + ["surface normal (xy tilt x5)"]
    u8 = [(np.clip(np.transpose(g, (1, 0, 2))[::-1], 0, 1) * 255).astype(np.uint8) for g in imgs]
    final = clip(ro, {"fluid": "water_glass"}, ref)[bi]
    w_png("breakdown_water.png",
          label_panels([downs(x) for x in u8 + [final]], names + ["shaded result"]))

    # ---------------------------------------------------------------- measured cost
    # The per-frame cost comes from sim/material_render_cost.py, which reads Taichi's kernel
    # profiler (CUDA events) instead of a wall clock. A wall clock here measures ~25-30 Python-side
    # kernel launches per frame and reports the same 3.3 ms at 360x360 and 1080x1080 -- a cost that
    # does not move with pixels is not measuring the render.
    cost_path = os.path.join(RUN_DIR, "render_cost.json")
    costs = {}
    if os.path.exists(cost_path):
        with open(cost_path) as fh:
            costs = json.load(fh)
    base = costs.get("456", {}).get("current", {}).get("gpu_ms")
    over = ({k: round(v["gpu_ms"] / base, 2) for k, v in costs["456"].items()} if base else {})

    metrics = dict(res=RES, particles_bench=16384, demo_density=DEMO_DENSITY, demo_iso=DEMO_ISO,
                   demo_radius=DEMO_RADIUS, physics_version=physics.VERSION,
                   render_cost_by_res=costs, over_current_at_456=over,
                   passes=PASSES, scenes=meta,
                   gpu="NVIDIA GeForce RTX 4090, Taichi 1.7.4 / CUDA (NOT WebGPU)")
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("wrote metrics.json", flush=True)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    if a.bench:
        upload_static()
        for k in KINDS:
            print("%-14s %6.2f ms" % (k, bench(k)))
    else:
        main(quick=a.quick)
