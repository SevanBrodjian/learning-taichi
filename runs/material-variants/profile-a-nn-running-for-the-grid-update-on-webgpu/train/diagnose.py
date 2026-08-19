"""Why does the learned grid update detonate? Instrument the first few thousand substeps.

Tracks, for the learned rollout and the analytic reference from the same seed:
  * max |v| on the grid, and the error in the node velocity field
  * the error in the DIVERGENCE of the node velocity field, which is what G2P differentiates
  * the fluid's volume ratio J, which integrates that divergence and sets the pressure
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
NG, BOUND = C.n_grid, C.bound

i_ = np.repeat(np.arange(NG), NG)
j_ = np.tile(np.arange(NG), NG)
WALLS = np.stack([(i_ < BOUND), (i_ > NG - BOUND), (j_ < BOUND), (j_ > NG - BOUND)], 1).astype(np.float32)


def load_net(h):
    z = np.load(D / "weights.npz")
    return [z[f"{h}_W1"], z[f"{h}_b1"], z[f"{h}_W2"], z[f"{h}_b2"], z[f"{h}_W3"], z[f"{h}_b3"]]


def net_apply(net, X):
    W1, b1, W2, b2, W3, b3 = net
    a = np.maximum(X @ W1 + b1, 0.0)
    a = np.maximum(a @ W2 + b2, 0.0)
    return a @ W3 + b3


def div_err(v_pred, v_true, m):
    """Central-difference divergence of both node velocity fields, compared on OCCUPIED cells.

    G2P does not read the node velocity, it reads a weighted derivative of it (the affine matrix C),
    so the quantity that actually reaches a particle is a spatial derivative of whatever the grid
    update wrote. A pointwise error is therefore not the error that matters.
    """
    def dv(v):
        vv = v.reshape(NG, NG, 2)
        dx = (vv[2:, 1:-1, 0] - vv[:-2, 1:-1, 0]) * 0.5 * NG
        dy = (vv[1:-1, 2:, 1] - vv[1:-1, :-2, 1]) * 0.5 * NG
        return dx + dy
    a, b = dv(v_pred), dv(v_true)
    occ = (m.reshape(NG, NG)[1:-1, 1:-1] > 0)
    if occ.sum() == 0:
        return 0.0, 0.0
    return float(np.abs(a[occ] - b[occ]).mean()), float(np.abs(b[occ]).mean())


def main():
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    net = load_net(h)
    sc = C.scene("drop", 6000)
    npart = C._upload(sc["pts"], (0.0, 0.0), C.FLUID)
    p_vol = sc["area"] / npart
    p_mass = p_vol * FLUID["rho"]
    C.init_state(npart)
    X = np.zeros((NG * NG, 8), np.float32)
    X[:, 3:7] = WALLS
    X[:, 7] = FLUID["fric"]
    rows = []
    NS = 4000
    for s in range(NS):
        C.clear_grid()
        C.p2g(C.FLUID, npart, DT, FLUID["E"], FLUID["nu"], 0.0, 0.0, p_vol, p_mass, FLUID["fric"])
        m = C.grid_m.to_numpy().reshape(-1)
        mom = C.grid_v.to_numpy().reshape(-1, 2)
        X[:, 0] = m / p_mass
        X[:, 1:3] = mom / p_mass
        vp = net_apply(net, X).astype(np.float32)
        # what the canonical kernel would have produced from the SAME pre-state
        C.grid_op(DT, FLUID["fric"], C.gravity)
        vt = C.grid_v.to_numpy().reshape(-1, 2).copy()
        if s % 100 == 0:
            occ = m > 0
            e = np.linalg.norm(vp - vt, axis=1)
            de, dmag = div_err(vp, vt, m)
            Jv = C.J.to_numpy()[:npart]
            rows.append({
                "substep": s,
                "n_occupied": int(occ.sum()),
                "max_abs_v_pred": float(np.abs(vp).max()),
                "max_abs_v_true": float(np.abs(vt).max()),
                "v_mae_massw": float((X[:, 0] * e).sum() / max(X[:, 0].sum(), 1e-30)),
                "v_mean_true_massw": float((X[:, 0] * np.linalg.norm(vt, axis=1)).sum()
                                           / max(X[:, 0].sum(), 1e-30)),
                "div_mae": de, "div_mag_true": dmag,
                "J_min": float(Jv.min()), "J_max": float(Jv.max()),
                "n_nonfinite_x": int((~np.isfinite(C.x.to_numpy()[:npart])).sum()),
            })
            r = rows[-1]
            print(f"s={s:5d} occ={r['n_occupied']:5d} vmae_mw={r['v_mae_massw']:.3e} "
                  f"(|v|={r['v_mean_true_massw']:.3f})  div_err={de:.3e} vs |div|={dmag:.3e}  "
                  f"J=[{r['J_min']:.4f},{r['J_max']:.4f}]  maxv_pred={r['max_abs_v_pred']:.3f} "
                  f"maxv_true={r['max_abs_v_true']:.3f} nan_x={r['n_nonfinite_x']}", flush=True)
            if not np.isfinite(vp).all():
                print("  >>> network output non-finite")
                break
        C.grid_v.from_numpy(vp.reshape(NG, NG, 2))
        C.g2p(C.FLUID, npart, DT, FLUID["tc"], FLUID["ts"], FLUID["E"], FLUID["nu"], 0.0)
    (D / f"diagnose_h{h}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
