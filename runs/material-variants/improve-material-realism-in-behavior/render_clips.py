"""Render the deliverable clips. Every comparison puts OLD physics and NEW physics on the same scene,
same seed, side by side, because a single "new" clip is not evidence of a change.

Reads the snapshot archives written by diagnose.py / buoyancy.py out of the scratchpad and writes mp4s
into the run directory.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("TASK_SCRATCH", os.path.join(HERE, "_scratch"))

import render                                                  # noqa: E402
import common                                                  # noqa: E402
from sim.physics import core as pc                             # noqa: E402  (canonical diagnostics)

FLOOR = 3.0 / 128.0
T_OF = {"drop": 1.3, "column": 1.7, "heap": 1.6, "slam": 1.0, "dam": 1.4}
COL = {"fluid": "#4db6ff", "elastic": "#ff4d4d", "snow": "#f2f6fc", "sand": "#ffd24d"}
WATER = "#2f7fd0"

ap = argparse.ArgumentParser()
ap.add_argument("--which", default="all")
args = ap.parse_args()

before = np.load(os.path.join(SCRATCH, "snaps_before.npz"))
after = np.load(os.path.join(SCRATCH, "snaps_after.npz"))


def times_for(scene, nf):
    return np.arange(1, nf + 1) * (T_OF[scene] / nf)


def pair(scene, mat, out, ylim=(0.0, 0.72), sub=None, pt=2.0):
    a = before[f"{scene}/{mat}"].astype(np.float32)
    b = after[f"{scene}/{mat}"].astype(np.float32)
    panels = [("OLD physics", a, COL[mat]), ("NEW physics", b, COL[mat])]
    return render.render_panels(os.path.join(HERE, out), panels, times_for(scene, a.shape[0]),
                                FLOOR, ncols=2, ylim=ylim, panel_w=520, sub_fn=sub, pt=pt)


made = []
if args.which in ("all", "water"):
    # dam break: the front runs free across the floor before it reaches the far wall
    made.append(pair("dam", "fluid", "water_dam.mp4", ylim=(0.0, 0.62)))
    made.append(pair("drop", "fluid", "water_drop.mp4", ylim=(0.0, 0.62)))

if args.which in ("all", "rubber"):
    # A per-frame readout of the footprint the blob still covers, so the shrink is legible as it happens.
    # The clip is cut at t = 0.2 s and played at 8 fps on purpose. Everything the claim is about -- the
    # approach, the squash, the start of the rebound -- happens in that window, and canonical rubber is
    # bouncy enough that the rest of the roll is the blob leaving the top of the frame.
    a = before["slam/elastic"].astype(np.float32)
    b = after["slam/elastic"].astype(np.float32)
    ra = [common.retained_area(a), common.retained_area(b)]
    K = 12
    made.append(render.render_panels(
        os.path.join(HERE, "rubber_slam.mp4"),
        [("OLD  nu 0.20", a[:K], COL["elastic"]), ("NEW  nu 0.45", b[:K], COL["elastic"])],
        times_for("slam", a.shape[0])[:K], FLOOR, ncols=2, ylim=(0.0, 0.50), panel_w=560,
        sub_fn=lambda k, f: "area %.0f%%" % (100 * ra[k][f]), pt=2.4, fps=8))

if args.which in ("all", "regress"):
    for mat in ("snow", "sand"):
        for scene in ("drop", "heap"):
            a = before[f"{scene}/{mat}"].astype(np.float32)
            b = after[f"{scene}/{mat}"].astype(np.float32)
            made.append(render.render_panels(
                os.path.join(HERE, f"regress_{mat}_{scene}.mp4"),
                [("OLD physics", a, COL[mat]), ("NEW physics", b, COL[mat])],
                times_for(scene, a.shape[0]), FLOOR, ncols=2, ylim=(0.0, 0.45), panel_w=520,
                sub_fn=lambda k, f, A=a, B=b: "width %.3f" % pc.spread_width((A if k == 0 else B)[f]),
                pt=2.0))

if args.which in ("all", "pool"):
    buoy = np.load(os.path.join(SCRATCH, "buoy_after.npz"))
    meta = json.load(open(os.path.join(HERE, "buoyancy_after.json")))
    TP = meta["T"]

    def pool_panels(keys, labels, mats):
        panels = []
        for k, lab, m in zip(keys, labels, mats):
            sol = buoy[k + "/solid"].astype(np.float32)
            flu = buoy[k + "/fluid"].astype(np.float32)
            snaps = np.concatenate([flu, sol], axis=1)
            cols = [WATER] * flu.shape[1] + [COL.get(m, "#dfe6ee")] * sol.shape[1]
            panels.append((lab, snaps, cols))
        return panels

    nf = buoy["mat_snow/solid"].shape[0]
    tt = np.arange(1, nf + 1) * (TP / nf)
    panels = pool_panels(["mat_snow", "mat_elastic", "mat_sand"],
                         ["snow  rho 0.3", "rubber  rho 1.2", "sand  rho 1.6"],
                         ["snow", "elastic", "sand"])
    depth = [meta["runs"][k]["rest_depth_curve"] for k in ("mat_snow", "mat_elastic", "mat_sand")]
    made.append(render.render_panels(
        os.path.join(HERE, "buoyancy_three.mp4"), panels, tt, FLOOR, ncols=3, ylim=(0.0, 0.44),
        panel_w=560, pt=1.6,
        sub_fn=lambda k, f: "depth %+.3f" % depth[k][f]))

    keys = ["rho_0.3", "rho_0.6", "rho_1.0", "rho_1.6"]
    panels = pool_panels(keys, ["rho 0.3", "rho 0.6", "rho 1.0", "rho 1.6"], ["elastic"] * 4)
    depth2 = [meta["runs"][k]["rest_depth_curve"] for k in keys]
    made.append(render.render_panels(
        os.path.join(HERE, "density_ladder.mp4"), panels, tt, FLOOR, ncols=4, ylim=(0.0, 0.44),
        panel_w=400, pt=1.9,
        sub_fn=lambda k, f: "%+.3f" % depth2[k][f]))

if args.which in ("all", "single"):
    # Single-panel clips, one per side, so the task page can put OLD and NEW on the SAME frame and let
    # the reader flip between them. A side-by-side pair shows the difference; a flip makes the reader
    # perform the comparison, which is the stronger form.
    FLIPS = [("dam", "fluid", (0.0, 0.60), "spread_width"),
             ("slam", "elastic", (0.0, 0.50), "retained_area"),
             ("heap", "snow", (0.0, 0.40), "repose_angle"),
             ("heap", "sand", (0.0, 0.40), "repose_angle")]
    for scene, mat, ylim, readout in FLIPS:
        for tag, src in (("old", before), ("new", after)):
            s = src[f"{scene}/{mat}"].astype(np.float32)
            tt = times_for(scene, s.shape[0])
            fps = 30
            if readout == "retained_area":
                s, tt, fps = s[:12], tt[:12], 8   # the impact only; see the rubber clip above
                ra = common.retained_area(s)
                sub = lambda k, f, R=ra: "footprint %.0f%% of start" % (100 * R[f])
            elif readout == "repose_angle":
                sub = lambda k, f, S=s: "slope %.0f deg" % pc.repose_angle(S[f])
            else:
                sub = lambda k, f, S=s: "front %.2f" % common.front_position(S[f])
            p = render.render_panels(
                os.path.join(HERE, f"flip_{scene}_{mat}_{tag}.mp4"),
                [(f"{'OLD' if tag == 'old' else 'NEW'} physics", s, COL[mat])],
                tt, FLOOR, ncols=1, ylim=ylim, panel_w=640, fps=fps,
                sub_fn=sub, pt=2.2)
            made.append(p)

for m in made:
    print("wrote", os.path.basename(m), os.path.getsize(m) // 1024, "KB")
