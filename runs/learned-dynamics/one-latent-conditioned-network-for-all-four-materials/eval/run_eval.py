"""Evaluate the learned simulator: the golden signatures, trajectory error against the right band,
and side-by-side video against canonical.

THE PASS CONDITION IS NOT A LOSS. `sim/physics/signatures.py` already encodes what "behaves like this
material" means for this project, with thresholds chosen by the physics rather than by whatever the
network turned out to achieve. It is run unmodified against the learned simulator (see sigproxy.py).

THE ERROR BAND IS NOT ZERO. `traj_rmse` -- which, despite the name, is the mean per-particle
Euclidean distance, not an RMS (spec/registry/metrics.json) -- is reported against two references:
the 1e-7 initial-condition nudge band (how far canonical moves from a perturbation with no physical
meaning) and the ORACLE's own distance from canonical (the same scaffolding running the exact
analytic law, so the floor a perfect network would hit). Quoting a learned trajectory error against
zero would be meaningless: these are chaotic systems and even canonical does not reproduce itself.

    .venv/Scripts/python.exe runs/.../eval/run_eval.py --hidden 64 [--tag ""] [--quick]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RUN / "train"))

import learned_sim as LS            # noqa: E402
import netspec as NS                # noqa: E402
import sigproxy                     # noqa: E402
from sim.physics import core        # noqa: E402

MATS = ["fluid", "elastic", "snow", "sand"]
SCENES = ["drop", "heap", "column"]
N = 4000
NF = 30
NUDGE = 1e-7


def mean_dist(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def load_weights(hidden, tag=""):
    d = np.load(RUN / "train" / f"weights_h{hidden}{tag}.npz")
    nl = int(d["layers"])
    ps = [[d[f"W{i}"], d[f"b{i}"]] for i in range(nl + 1)]
    return ps, d["wgsl"]


def traj_table(hidden, l2, quick=False):
    rows = []
    nf = 8 if quick else NF
    for scn in SCENES:
        sc = core.scene(scn, N)
        rng = np.random.default_rng(21)
        for m in MATS:
            can, _, ok0 = core.simulate(m, sc["pts"], sc["area"], sc["T"], nf, v0=sc["v0"])
            nud, _, _ = core.simulate(m, sc["pts"] + rng.normal(0, NUDGE, sc["pts"].shape),
                                      sc["area"], sc["T"], nf, v0=sc["v0"])
            g = [{"material": m, "pts": sc["pts"], "area": sc["area"], "v0": sc["v0"]}]
            orc, _, _, ok1, _ = LS.rollout(g, sc["T"], nf, dt=core.MAT[m]["dt"], mode="oracle")
            nn, _, _, ok2, _ = LS.rollout(g, sc["T"], nf, dt=core.MAT[m]["dt"], mode="nn",
                                          hidden=hidden, l2=l2)
            rows.append({
                "scene": scn, "material": m,
                "traj_rmse_learned": mean_dist(can, nn),
                "traj_rmse_oracle": mean_dist(can, orc),
                "ic_nudge_band": mean_dist(can, nud),
                "learned_stable": bool(ok2), "oracle_stable": bool(ok1),
                "spread_canonical": core.spread_width(can[-1]),
                "spread_learned": core.spread_width(nn[-1]),
                "height_canonical": core.pile_height(can[-1]),
                "height_learned": core.pile_height(nn[-1]),
                "repose_canonical": core.repose_angle(can[-1]),
                "repose_learned": core.repose_angle(nn[-1]),
            })
            r = rows[-1]
            print(f"  {scn:7s} {m:8s} learned {r['traj_rmse_learned']:.4f}  "
                  f"oracle {r['traj_rmse_oracle']:.2e}  nudge {r['ic_nudge_band']:.2e}  "
                  f"spread {r['spread_learned']:.3f} vs {r['spread_canonical']:.3f}  "
                  f"stable={r['learned_stable']}", flush=True)
    return rows


def render_grid(hidden, l2, scene="heap", nf=90, out="learned_vs_canonical_heap.mp4"):
    """A 2 x 4 grid: canonical on top, learned below, one column per material, same scene and seed.

    Video and not final frames, because every claim here is about MOTION -- whether snow crumbles,
    whether sand yields to a slope, whether water runs flat. Two still frames can agree while the
    two runs got there completely differently.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    sc = core.scene(scene, N)
    data = {}
    for m in MATS:
        can, _, _ = core.simulate(m, sc["pts"], sc["area"], sc["T"], nf, v0=sc["v0"])
        g = [{"material": m, "pts": sc["pts"], "area": sc["area"], "v0": sc["v0"]}]
        nn, _, _, ok, _ = LS.rollout(g, sc["T"], nf, dt=core.MAT[m]["dt"], mode="nn",
                                     hidden=hidden, l2=l2)
        data[m] = (can, nn, ok)
        print(f"    rendered rollouts for {m} (learned stable={ok})", flush=True)

    frames = []
    for f in range(nf):
        fig, ax = plt.subplots(2, 4, figsize=(13.0, 6.8), dpi=100)
        fig.patch.set_facecolor("#0a0e14")
        for j, m in enumerate(MATS):
            can, nn, ok = data[m]
            col = core.MAT[m]["color"]
            for i, (snap, lab) in enumerate(((can, "canonical"), (nn, "learned"))):
                a = ax[i, j]
                a.set_facecolor("#0f141c")
                a.scatter(snap[f][:, 0], snap[f][:, 1], s=1.4, c=col, linewidths=0, alpha=0.9)
                a.axhline(core.floor_y, color="#33415a", lw=1.0)
                # the FULL domain, not a crop around where canonical ends up. A tighter window made the
                # learned snow panel look empty, because its particles had been thrown to the ceiling
                # and out of frame -- which reads as a broken render rather than as the result.
                a.set_xlim(0, 1); a.set_ylim(0, 1.0)
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_color("#2a3446")
                if i == 0:
                    a.set_title(m, color=col, fontsize=12, pad=6)
                a.text(0.02, 0.93, lab, transform=a.transAxes, color="#8fa3bf", fontsize=9)
                if i == 1 and not ok:
                    a.text(0.5, 0.5, "UNSTABLE", transform=a.transAxes, color="#ff8f8f",
                           fontsize=13, ha="center")
        fig.suptitle(f"one network, one weight set, four materials  --  scene '{scene}',  "
                     f"t = {sc['T'] * (f + 1) / nf:.2f}s", color="#dfe6ee", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)
    imageio.mimsave(RUN / out, frames, fps=20, quality=8, macro_block_size=1)
    print("  wrote", RUN / out, flush=True)

    # final-frame contact sheet, for anyone who will not press play
    fig, ax = plt.subplots(2, 4, figsize=(13.0, 6.8), dpi=110)
    fig.patch.set_facecolor("#0a0e14")
    for j, m in enumerate(MATS):
        can, nn, ok = data[m]
        col = core.MAT[m]["color"]
        for i, (snap, lab) in enumerate(((can, "canonical"), (nn, "learned"))):
            a = ax[i, j]
            a.set_facecolor("#0f141c")
            a.scatter(snap[-1][:, 0], snap[-1][:, 1], s=1.6, c=col, linewidths=0, alpha=0.9)
            a.axhline(core.floor_y, color="#33415a", lw=1.0)
            # the FULL domain, not a crop around where canonical ends up. A tighter window made the
            # learned snow panel look empty, because its particles had been thrown to the ceiling
            # and out of frame -- which reads as a broken render rather than as the result.
            a.set_xlim(0, 1); a.set_ylim(0, 1.0)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#2a3446")
            if i == 0:
                a.set_title(m, color=col, fontsize=12, pad=6)
            a.text(0.02, 0.93, lab + f"  repose {core.repose_angle(snap[-1]):.0f}deg  "
                   f"width {core.spread_width(snap[-1]):.2f}", transform=a.transAxes,
                   color="#8fa3bf", fontsize=8)
    fig.suptitle(f"final state, scene '{scene}' -- canonical (top) vs one shared network (bottom)",
                 color="#dfe6ee", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RUN / out.replace(".mp4", "_final.png"), facecolor=fig.get_facecolor())
    plt.close(fig)
    return {m: bool(data[m][2]) for m in MATS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ps, wg = load_weights(a.hidden, a.tag)
    H, L2 = LS.upload_weights(ps)
    print(f"loaded hidden={H} layers={'2' if L2 else '1'} tag='{a.tag}' "
          f"({sum(W.size + b.size for W, b in ps)} parameters)")

    print("--- golden signatures against the LEARNED simulator ---", flush=True)
    rows, summ = sigproxy.run(
        lambda *x, **k: LS.simulate(*x, mode="nn", **k),
        lambda *x, **k: LS.simulate_multi(*x, mode="nn", **k), label=f"learned h={H}")
    sigproxy.show(rows, f"the LEARNED simulator (one net, hidden {H})")

    print("--- trajectory error, against the oracle floor and the IC-nudge band ---", flush=True)
    traj = traj_table(H, L2, a.quick)

    vids = {}
    if not a.no_video:
        print("--- rendering learned vs canonical ---", flush=True)
        vids["heap"] = render_grid(H, L2, "heap", 30 if a.quick else 90,
                                   f"learned_vs_canonical_heap{a.tag}.mp4")
        vids["drop"] = render_grid(H, L2, "drop", 30 if a.quick else 90,
                                   f"learned_vs_canonical_drop{a.tag}.mp4")

    out = {"hidden": H, "layers": 2 if L2 else 1, "tag": a.tag,
           "physics_version": LS.VERSION,
           "signatures": rows, "signature_summary": summ,
           "trajectory": traj, "video_stable": vids}
    name = a.out or f"eval_h{H}{a.tag}.json"
    (HERE / name).write_text(json.dumps(out, indent=2))
    print("wrote", HERE / name)


if __name__ == "__main__":
    main()
