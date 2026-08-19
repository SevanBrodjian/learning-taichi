"""Does the learned grid update actually hold a rollout together?

A per-node velocity error of a couple of percent is applied 20,000 times per simulated second, so the
only meaningful accuracy test is a full rollout: canonical P2G and G2P, with `grid_op` swapped for the
MLP, run against the canonical reference from the same seed.

This is a HOST-side check (numpy MLP, canonical Taichi transfer kernels). It exists to answer "is the
learned operator stable and water-like at all" before any of it is ported to WGSL, and to produce the
learned-vs-ground-truth comparison frames.

    .venv/Scripts/python.exe .../train/rollout_check.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import sim.physics as phys                       # noqa: E402
from sim.physics import core as C                # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1]
D = RUN / "train"

FLUID = phys.MAT["fluid"]
DT = FLUID["dt"]
NG = C.n_grid
BOUND = C.bound
N_FRAMES = 60
SCENE, NPART, T = "drop", 6000, 1.0
WIDTHS = (8, 16, 32, 64)


def wall_feats():
    i = np.repeat(np.arange(NG), NG)
    j = np.tile(np.arange(NG), NG)
    return np.stack([(i < BOUND), (i > NG - BOUND), (j < BOUND), (j > NG - BOUND)], 1).astype(np.float32)


WALLS = wall_feats()


def load_net(h):
    z = np.load(D / "weights.npz")
    return [z[f"{h}_W1"], z[f"{h}_b1"], z[f"{h}_W2"], z[f"{h}_b2"], z[f"{h}_W3"], z[f"{h}_b3"]]


def net_apply(net, X):
    W1, b1, W2, b2, W3, b3 = net
    a = np.maximum(X @ W1 + b1, 0.0)
    a = np.maximum(a @ W2 + b2, 0.0)
    return a @ W3 + b3


def rollout(net, p_mass_known, n, pts, area, frames, T):
    """One rollout. `net=None` runs the canonical analytic grid_op."""
    npart = C._upload(pts, (0.0, 0.0), C.FLUID)
    p_vol = area / npart
    p_mass = p_vol * FLUID["rho"]
    C.init_state(npart)
    spf = max(1, int(round((T / frames) / DT)))
    snaps = np.zeros((frames, npart, 2), np.float32)
    stable = True
    Xbuf = np.zeros((NG * NG, 8), np.float32)
    Xbuf[:, 3:7] = WALLS
    Xbuf[:, 7] = FLUID["fric"]
    for f in range(frames):
        for _ in range(spf):
            C.clear_grid()
            C.p2g(C.FLUID, npart, DT, FLUID["E"], FLUID["nu"], 0.0, 0.0, p_vol, p_mass, FLUID["fric"])
            if net is None:
                C.grid_op(DT, FLUID["fric"], C.gravity)
            else:
                m = C.grid_m.to_numpy().reshape(-1)
                mom = C.grid_v.to_numpy().reshape(-1, 2)
                Xbuf[:, 0] = m / p_mass
                Xbuf[:, 1:3] = mom / p_mass
                v = net_apply(net, Xbuf).astype(np.float32)
                C.grid_v.from_numpy(v.reshape(NG, NG, 2))
            C.g2p(C.FLUID, npart, DT, FLUID["tc"], FLUID["ts"], FLUID["E"], FLUID["nu"], 0.0)
        cur = C.x.to_numpy()[:npart]
        if not np.isfinite(cur).all():
            stable = False
            cur = np.nan_to_num(cur)
        snaps[f] = cur
    return snaps, stable


def main():
    sc = C.scene(SCENE, NPART)
    pts, area = sc["pts"], sc["area"]
    print("ground truth ...", flush=True)
    gt, _ = rollout(None, None, NPART, pts, area, N_FRAMES, T)
    res = {"scene": SCENE, "n": int(gt.shape[1]), "T": T, "frames": N_FRAMES,
           "physics_version": phys.VERSION, "widths": {}}
    out = {"gt": gt}
    for h in WIDTHS:
        print(f"learned width {h} ...", flush=True)
        net = load_net(h)
        sn, stable = rollout(net, None, NPART, pts, area, N_FRAMES, T)
        d = np.linalg.norm(sn - gt, axis=2)
        res["widths"][str(h)] = {
            "stable": bool(stable),
            "traj_rmse": float(d.mean()),
            "traj_rmse_final": float(d[-1].mean()),
            "final_spread_width": float(C.spread_width(sn[-1])),
            "gt_final_spread_width": float(C.spread_width(gt[-1])),
            "final_pile_height": float(C.pile_height(sn[-1])),
            "gt_final_pile_height": float(C.pile_height(gt[-1])),
            "mean_speed_final": float(np.abs(np.diff(sn[-3:], axis=0)).mean()),
        }
        print("   ", json.dumps(res["widths"][str(h)]))
        out[f"h{h}"] = sn
    np.savez_compressed(D / "rollouts.npz", **out)
    (D / "rollout_stats.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", D / "rollouts.npz")


if __name__ == "__main__":
    main()
