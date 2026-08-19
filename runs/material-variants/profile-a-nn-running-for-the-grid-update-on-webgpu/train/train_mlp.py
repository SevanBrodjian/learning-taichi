"""Fit a per-cell MLP to the CANONICAL grid update, at four hidden widths.

    input  (8) : [ m_hat, p_hat_x, p_hat_y, wallL, wallR, wallBottom, wallTop, friction ]
    output (2) : [ v_x, v_y ]        -- the node velocity the analytic grid_op would have produced

`m_hat` and `p_hat` are the node's mass and momentum in units of one particle mass. That is a
multiply by a single scalar that is uniform over the grid and known before the substep runs, so it is
folded into the first layer at export time and the shader does no preprocessing at all. Everything the
analytic kernel does -- the division by mass, gravity, the separating-boundary test, the Coulomb
friction cap -- is inside the network.

WHY THE LOSS IS MASS WEIGHTED, and why that is not a convenience.

G2P gathers node velocities with the quadratic B-spline weight w_{pi}, and P2G scattered exactly
w_{pi} * p_mass of mass into node i from particle p, so sum_p w_{pi} = m_hat_i. If node i's velocity is
wrong by dv_i, the total velocity error that node injects into the entire particle set is therefore

    sum_p w_{pi} |dv_i|  =  m_hat_i |dv_i| .

The mass-weighted error is the physically meaningful one; an unweighted average over cells is not,
because a node holding 1e-16 of a particle mass cannot be gathered from with weight greater than
1e-16. Both are reported.

    .venv/Scripts/python.exe .../train/train_mlp.py
"""
import json
import pathlib
import time

import numpy as np

RUN = pathlib.Path(__file__).resolve().parents[1]
D = RUN / "train"

WIDTHS = (8, 16, 32, 64)
IN_SCALE = np.array([1 / 8.0, 1 / 8.0, 1 / 8.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.float32)
N_TRAIN = 4_000_000
BATCH = 8192
STEPS = 14000
LR0 = 3e-3
SEED = 0
WCAP = 10.0            # mass weight is min(m_hat, WCAP); the floor keeps empty cells in the loss
WFLOOR = 0.05


def mlp_init(h, rng):
    def he(a, b):
        return (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
    return {"W1": he(8, h), "b1": np.zeros(h, np.float32),
            "W2": he(h, h), "b2": np.zeros(h, np.float32),
            "W3": (rng.standard_normal((h, 2)) * np.sqrt(1.0 / h)).astype(np.float32),
            "b3": np.zeros(2, np.float32)}


def forward(P, X):
    z1 = X @ P["W1"] + P["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ P["W2"] + P["b2"]
    a2 = np.maximum(z2, 0.0)
    y = a2 @ P["W3"] + P["b3"]
    return y, (a1, a2, z1, z2)


def backward(P, X, cache, dy):
    a1, a2, z1, z2 = cache
    g = {}
    g["W3"] = a2.T @ dy
    g["b3"] = dy.sum(0)
    da2 = dy @ P["W3"].T
    dz2 = da2 * (z2 > 0)
    g["W2"] = a1.T @ dz2
    g["b2"] = dz2.sum(0)
    da1 = dz2 @ P["W2"].T
    dz1 = da1 * (z1 > 0)
    g["W1"] = X.T @ dz1
    g["b1"] = dz1.sum(0)
    return g


def train_one(h, Xtr, Ytr, Wtr, Xte, Yte, mte, log):
    rng = np.random.default_rng(SEED + h)
    P = mlp_init(h, rng)
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    n = Xtr.shape[0]
    t0 = time.time()
    hist = []
    for s in range(STEPS):
        idx = rng.integers(0, n, BATCH)
        xb, yb, wb = Xtr[idx], Ytr[idx], Wtr[idx]
        pred, cache = forward(P, xb)
        r = pred - yb
        wsum = wb.sum()
        dy = (2.0 * wb[:, None] * r / wsum).astype(np.float32)
        loss = float((wb[:, None] * r * r).sum() / wsum)
        g = backward(P, xb, cache, dy)
        lr = LR0 * (0.5 * (1 + np.cos(np.pi * s / STEPS)))
        for k in P:
            M[k] = 0.9 * M[k] + 0.1 * g[k]
            V[k] = 0.999 * V[k] + 0.001 * (g[k] * g[k])
            mh = M[k] / (1 - 0.9 ** (s + 1))
            vh = V[k] / (1 - 0.999 ** (s + 1))
            P[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        if s % 1000 == 0 or s == STEPS - 1:
            hist.append({"step": s, "train_loss": loss})
            log(f"    h={h:3d} step {s:6d}  wmse={loss:.5e}  lr={lr:.2e}")
    # -------- evaluation on held-out cells --------
    pe, _ = forward(P, Xte)
    err = np.linalg.norm(pe - Yte, axis=1)
    mw = float((mte * err).sum() / mte.sum())            # the physically meaningful average
    occ = mte > 1e-3
    stats = {
        "hidden": h,
        "node_v_mae": float(err.mean()),
        "node_v_mae_massw": mw,
        "node_v_mae_occupied": float(err[occ].mean()),
        "node_v_p99": float(np.percentile(err, 99)),
        "gt_speed_mean_massw": float((mte * np.linalg.norm(Yte, axis=1)).sum() / mte.sum()),
        "params": int(8 * h + h + h * h + h + h * 2 + 2),
        "flops_per_cell": int(2 * (8 * h + h * h + h * 2)),
        "train_seconds": time.time() - t0,
        "history": hist,
    }
    stats["node_v_rel_massw"] = stats["node_v_mae_massw"] / stats["gt_speed_mean_massw"]
    return P, stats


def fold_scale(P):
    """Fold the fixed diagonal input scale into layer 1 so the shader feeds RAW (m_hat, p_hat, ...)."""
    Q = {k: v.copy() for k, v in P.items()}
    Q["W1"] = (P["W1"] * IN_SCALE[:, None]).astype(np.float32)
    return Q


def main():
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    z = np.load(D / "gridop_data.npz")
    Xtr, Ytr = z["Xtr"], z["Ytr"]
    Xte, Yte = z["Xte"], z["Yte"]
    rng = np.random.default_rng(7)
    if Xtr.shape[0] > N_TRAIN:
        k = rng.choice(Xtr.shape[0], N_TRAIN, replace=False)
        Xtr, Ytr = Xtr[k], Ytr[k]
    if Xte.shape[0] > 400_000:
        k = rng.choice(Xte.shape[0], 400_000, replace=False)
        Xte, Yte = Xte[k], Yte[k]
    mtr, mte = Xtr[:, 0].copy(), Xte[:, 0].copy()
    Wtr = (WFLOOR + np.minimum(mtr, WCAP)).astype(np.float32)
    Xtr = (Xtr * IN_SCALE).astype(np.float32)
    Xte_s = (Xte * IN_SCALE).astype(np.float32)
    log(f"train {Xtr.shape}  test {Xte_s.shape}")

    out = {"widths": {}, "in_scale": IN_SCALE.tolist(),
           "physics_version": str(z["physics_version"]),
           "dt": float(z["dt"]), "gravity": float(z["gravity"]),
           "inputs": ["m_hat", "p_hat_x", "p_hat_y", "wallL", "wallR", "wallB", "wallT", "friction"],
           "outputs": ["v_x", "v_y"], "activation": "relu", "layers": "8-h-h-2",
           "train": {"steps": STEPS, "batch": BATCH, "lr0": LR0, "n_train": int(Xtr.shape[0]),
                     "loss": "mass-weighted MSE, weight = 0.05 + min(m_hat, 10)"}}
    weights = {}
    for h in WIDTHS:
        log(f"  --- width {h} ---")
        P, st = train_one(h, Xtr, Ytr, Wtr, Xte_s, Yte, mte, log)
        Q = fold_scale(P)
        weights[str(h)] = {k: v.astype(np.float32) for k, v in Q.items()}
        out["widths"][str(h)] = st
        log(f"    h={h:3d}  mass-weighted node-velocity MAE = {st['node_v_mae_massw']:.4e} "
            f"({100 * st['node_v_rel_massw']:.2f}% of mean node speed)   "
            f"plain MAE = {st['node_v_mae']:.4e}")

    np.savez(D / "weights.npz", **{f"{h}_{k}": v for h, d in weights.items() for k, v in d.items()})
    (D / "train_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (D / "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print("wrote", D / "weights.npz")


if __name__ == "__main__":
    main()
