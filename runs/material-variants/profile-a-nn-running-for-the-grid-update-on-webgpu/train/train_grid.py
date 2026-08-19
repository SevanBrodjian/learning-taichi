"""Second training stage: fit the grid update on WHOLE grids, penalising the DERIVATIVE error.

Stage one (train_mlp.py) fits the operator cell by cell and reaches ~1.5% mass-weighted error on the
node velocity, which sounds fine and is useless: the rollout detonates after about 1800 substeps
because G2P gathers a spatial derivative of the node velocity field, and a pointwise error with no
spatial correlation produces a derivative error of the same order as the derivative itself.

So stage two adds a term on the first differences of the predicted field, evaluated on occupied cells,
alongside the pointwise term. Both terms are normalised by the variance of their own target so the
mixing weight LAM is a relative preference rather than a unit conversion.

    .venv/Scripts/python.exe .../train/train_grid.py [lam] [steps]
"""
import json
import pathlib
import sys
import time

import numpy as np

RUN = pathlib.Path(__file__).resolve().parents[1]
D = RUN / "train"
NG = 128
BOUND = 3
WIDTHS = (8, 16, 32, 64)
IN_SCALE = np.array([1 / 8.0, 1 / 8.0, 1 / 8.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.float32)
B = 6
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
LR0 = 1.2e-3
WCAP, WFLOOR = 10.0, 0.05

i_ = np.arange(NG)[:, None] * np.ones((1, NG))
j_ = np.ones((NG, 1)) * np.arange(NG)[None, :]
WALLS = np.stack([(i_ < BOUND), (i_ > NG - BOUND), (j_ < BOUND), (j_ > NG - BOUND)], -1).astype(np.float32)


def forward(P, X):
    z1 = X @ P["W1"] + P["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ P["W2"] + P["b2"]
    a2 = np.maximum(z2, 0.0)
    return a2 @ P["W3"] + P["b3"], (a1, a2, z1, z2)


def backward(P, X, cache, dy):
    a1, a2, z1, z2 = cache
    g = {"W3": a2.T @ dy, "b3": dy.sum(0)}
    dz2 = (dy @ P["W3"].T) * (z2 > 0)
    g["W2"] = a1.T @ dz2
    g["b2"] = dz2.sum(0)
    dz1 = (dz2 @ P["W2"].T) * (z1 > 0)
    g["W1"] = X.T @ dz1
    g["b1"] = dz1.sum(0)
    return g


def diffs(V):
    """First differences over two cells, in x and y, on the interior."""
    return (V[:, 2:, 1:-1, :] - V[:, :-2, 1:-1, :],
            V[:, 1:-1, 2:, :] - V[:, 1:-1, :-2, :])


def build_inputs(m, mom, fric):
    Bn = m.shape[0]
    X = np.empty((Bn, NG, NG, 8), np.float32)
    X[..., 0] = m
    X[..., 1:3] = mom
    X[..., 3:7] = WALLS[None]
    X[..., 7] = fric[:, None, None]
    return X


def evaluate(P, Xs, Vt, mh):
    pred, _ = forward(P, Xs.reshape(-1, 8))
    V = pred.reshape(Vt.shape)
    err = np.linalg.norm(V - Vt, axis=-1)
    w = mh
    dxp, dyp = diffs(V)
    dxt, dyt = diffs(Vt)
    occ = mh[:, 1:-1, 1:-1] > 0
    de = (np.abs(dxp - dxt)[occ].mean() + np.abs(dyp - dyt)[occ].mean()) * 0.5
    dm = (np.abs(dxt)[occ].mean() + np.abs(dyt)[occ].mean()) * 0.5
    return {
        "node_v_mae_massw": float((w * err).sum() / w.sum()),
        "node_v_mae": float(err.mean()),
        "node_v_mae_occupied": float(err[mh > 0].mean()),
        "gt_speed_massw": float((w * np.linalg.norm(Vt, axis=-1)).sum() / w.sum()),
        "grad_mae": float(de), "grad_mag_true": float(dm),
        "grad_rel": float(de / max(dm, 1e-12)),
    }


def main():
    z = np.load(D / "grids.npz")
    m_all, mom_all, v_all, f_all = z["m"], z["mom"], z["v"], z["fric"]
    n = m_all.shape[0]
    rng = np.random.default_rng(3)
    perm = rng.permutation(n)
    m_all, mom_all, v_all, f_all = m_all[perm], mom_all[perm], v_all[perm], f_all[perm]
    ntr = int(0.9 * n)
    print(f"{n} grids, {ntr} train")

    # target variances, so LAM is a relative weight rather than a unit conversion
    Vt_s = v_all[:40]
    var_p = float((Vt_s ** 2).mean())
    dxt, dyt = diffs(Vt_s)
    occ_s = m_all[:40, 1:-1, 1:-1] > 0
    var_g = float(((dxt ** 2)[occ_s].mean() + (dyt ** 2)[occ_s].mean()) * 0.5)
    print(f"target variance: pointwise {var_p:.4e}  gradient {var_g:.4e}  (ratio {var_p/var_g:.1f})")

    Xte = build_inputs(m_all[ntr:ntr + 60], mom_all[ntr:ntr + 60], f_all[ntr:ntr + 60]) * IN_SCALE
    Vte = v_all[ntr:ntr + 60]
    mte = m_all[ntr:ntr + 60]

    w0 = np.load(D / "weights.npz")
    out = {"lam": LAM, "steps": STEPS, "batch_grids": B, "widths": {},
           "in_scale": IN_SCALE.tolist(), "target_var": {"pointwise": var_p, "gradient": var_g}}
    newW = {}
    log = []
    for h in WIDTHS:
        P = {k: w0[f"{h}_{k}"].copy() for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        P["W1"] = (P["W1"] / IN_SCALE[:, None]).astype(np.float32)   # unfold the stage-1 export scale
        before = evaluate(P, Xte, Vte, mte)
        M = {k: np.zeros_like(v) for k, v in P.items()}
        V_ = {k: np.zeros_like(v) for k, v in P.items()}
        t0 = time.time()
        hist = []
        for s in range(STEPS):
            idx = rng.integers(0, ntr, B)
            mh, mom, fr, Vt = m_all[idx], mom_all[idx], f_all[idx], v_all[idx]
            X = (build_inputs(mh, mom, fr) * IN_SCALE).reshape(-1, 8)
            pred, cache = forward(P, X)
            Vp = pred.reshape(B, NG, NG, 2)

            wgt = (WFLOOR + np.minimum(mh, WCAP)).astype(np.float32)[..., None]
            r = Vp - Vt
            wsum = float(wgt.sum()) * 2.0
            Lp = float((wgt * r * r).sum() / wsum)
            dV = (2.0 * wgt * r / wsum) / var_p

            dxp, dyp = diffs(Vp)
            dxt_, dyt_ = diffs(Vt)
            occ = (mh[:, 1:-1, 1:-1] > 0)[..., None].astype(np.float32)
            gnorm = float(occ.sum()) * 4.0
            rx, ry = (dxp - dxt_) * occ, (dyp - dyt_) * occ
            Lg = float((rx * rx + ry * ry).sum() / gnorm)
            gx = (2.0 * rx / gnorm) * (LAM / var_g)
            gy = (2.0 * ry / gnorm) * (LAM / var_g)
            dV[:, 2:, 1:-1, :] += gx
            dV[:, :-2, 1:-1, :] -= gx
            dV[:, 1:-1, 2:, :] += gy
            dV[:, 1:-1, :-2, :] -= gy

            g = backward(P, X, cache, dV.reshape(-1, 2).astype(np.float32))
            lr = LR0 * (0.5 * (1 + np.cos(np.pi * s / STEPS)))
            for k in P:
                M[k] = 0.9 * M[k] + 0.1 * g[k]
                V_[k] = 0.999 * V_[k] + 0.001 * (g[k] * g[k])
                P[k] -= (lr * (M[k] / (1 - 0.9 ** (s + 1)))
                         / (np.sqrt(V_[k] / (1 - 0.999 ** (s + 1))) + 1e-8)).astype(np.float32)
            if s % 500 == 0 or s == STEPS - 1:
                hist.append({"step": s, "Lp": Lp, "Lg": Lg})
                line = (f"  h={h:3d} step {s:5d}  point {Lp/var_p:.4f}  grad {Lg/var_g:.4f}  lr {lr:.2e}")
                print(line, flush=True)
                log.append(line)
        after = evaluate(P, Xte, Vte, mte)
        Q = {k: v.copy() for k, v in P.items()}
        Q["W1"] = (P["W1"] * IN_SCALE[:, None]).astype(np.float32)
        newW.update({f"{h}_{k}": v for k, v in Q.items()})
        out["widths"][str(h)] = {
            "hidden": h, "before": before, "after": after, "history": hist,
            "params": int(8 * h + h + h * h + h + h * 2 + 2),
            "flops_per_cell": int(2 * (8 * h + h * h + h * 2)),
            "train_seconds": time.time() - t0,
        }
        line = (f"  h={h:3d}  grad rel err {before['grad_rel']:.3f} -> {after['grad_rel']:.3f}   "
                f"massw v MAE {before['node_v_mae_massw']:.3e} -> {after['node_v_mae_massw']:.3e}")
        print(line, flush=True)
        log.append(line)
    np.savez(D / "weights_grid.npz", **newW)
    (D / "train_grid_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (D / "train_grid_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("wrote", D / "weights_grid.npz")


if __name__ == "__main__":
    main()
