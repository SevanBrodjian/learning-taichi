"""Fit ONE latent-conditioned MLP to all four canonical constitutive laws, at several widths.

ONE weight set. The material enters only as z_m, four fixed well-separated codes (netspec.Z_CODES),
JITTERED every batch so the network learns a neighbourhood around each code rather than four point
lookups -- if it memorised four exact vectors, the "latent" would be an index in disguise and the
first thing a slightly-off code did would be to produce nonsense.

Plain Adam on a whitened MSE, in numpy. No autodiff library is needed because the model is a
one-or-two hidden layer MLP and the loss is a regression: the backward pass is four lines.

WHITENING IS FOLDED INTO THE WEIGHTS at export time (netspec.fold_normalisation). The seven outputs
span four orders of magnitude -- stress is O(10), the plastic corrections are O(1e-3) -- so training
without output whitening would fit the stress and ignore the plasticity entirely, which is precisely
the half of the seam that makes snow snow. Folding rather than shipping separate scale vectors means
the shader runs a bare MLP and there is no second place for a constant to be typed in differently.

    .venv/Scripts/python.exe runs/.../train/train_mlp.py --widths 8 16 32 64 128 --steps 12000
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import netspec as NS                # noqa: E402

MATS = ["fluid", "elastic", "snow", "sand"]
MAT_ID = {"fluid": 0, "elastic": 1, "snow": 2, "sand": 3}


def load(path, per_mat=500000, seed=0, drop_nuisance=False):
    d = np.load(path)
    rng = np.random.default_rng(seed)
    idx = []
    for m in range(4):
        w = np.flatnonzero(d["mat"] == m)
        idx.append(rng.choice(w, size=min(per_mat, w.size), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    S, C, V, Jp, M, Y = (d["S"][idx], d["C"][idx], d["V"][idx], d["Jp"][idx],
                         d["mat"][idx], d["Y"][idx])
    if drop_nuisance:
        C = np.zeros_like(C)
        V = np.zeros_like(V)
    return S, C, V, Jp, M, Y


def make_X(S, C, V, Jp, M, jitter, rng):
    z = NS.Z_CODES[M]
    if jitter > 0:
        z = z + rng.normal(0, jitter, z.shape).astype(np.float32)
    return np.concatenate([S, C, V, Jp[:, None], z], 1).astype(np.float32)


def whiten(X, in_mean, in_scale):
    """Plain affine whitening -- deliberately NO clipping.

    Clipping the tails would help the fit (the APIC affine C has heavy tails, and a particle grazing
    a wall can carry a velocity gradient a hundred spreads out). It is rejected anyway, because the
    whitening is FOLDED into the first layer at export time so the shader runs a bare MLP: a clip
    here and no clip there would mean the deployed network is not the network that was trained, on
    exactly the rare inputs where it matters most. The heavy tails are handled by a robust SCALE
    instead, which is a change of units and folds exactly."""
    return (X - in_mean) / in_scale


def loss_weights(Y, M, out_scale):
    """Per (material, output) weights for the MSE.

    Without them the fit is dominated by whichever material happens to have the largest stress, and
    that is not a hypothetical: rubber's stress standard deviation is 18 and snow's is 2.1, so a
    globally-whitened MSE can leave snow's stress predicted no better than its own mean -- which it
    did, at relative error 1.02 -- while the total loss looks respectable. Weighting by the inverse
    of each material's own spread makes the objective "be equally right about all four", which is
    the actual question.

    A CONSTANT output gets weight one, not infinity. Rubber's plastic correction is EXACTLY zero and
    so is its spread, and the obvious floor -- a twentieth of the global spread -- hands it a weight
    of 64. That is what the first attempt did, and the run is instructive: the five constant-target
    (material, output) pairs took 5.1x weight each while every genuinely hard one was pushed down to
    0.08, so the optimiser spent itself learning to emit zero and left snow's stress no better than
    its own mean. A target with no variance is trivially fit at any weight; it does not need help.
    The clip is also kept tight, at a factor of four in the squared weight, because the point is to
    stop one material dominating, not to invert the domination.
    """
    W = np.ones((4, Y.shape[1]), np.float32)
    for m in range(4):
        sel = M == m
        if not sel.any():
            continue
        sd = Y[sel].std(0)
        r = np.where(sd > 1e-3 * out_scale, out_scale / np.maximum(sd, 1e-30), 1.0)
        W[m] = np.clip(r, 0.5, 2.0) ** 2
    return W


def train(Xstat, S, C, V, Jp, M, Y, hidden, layers, steps, batch, lr, seed, jitter, log):
    rng = np.random.default_rng(seed)
    in_mean, in_scale, out_mean, out_scale = Xstat
    Yw = (Y - out_mean) / out_scale
    LW = loss_weights(Y, M, out_scale)
    LW = LW / LW.mean()
    ps = NS.init_params(hidden, layers, seed=seed)
    mom = [[np.zeros_like(W), np.zeros_like(b)] for W, b in ps]
    vel = [[np.zeros_like(W), np.zeros_like(b)] for W, b in ps]
    n = S.shape[0]
    b1, b2, eps = 0.9, 0.999, 1e-8
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        i = rng.integers(0, n, batch)
        Xb = whiten(make_X(S[i], C[i], V[i], Jp[i], M[i], jitter, rng), in_mean, in_scale)
        Yb = Yw[i]
        wb = LW[M[i]]
        # forward
        acts, h = [Xb], Xb
        for W, b in ps[:-1]:
            h = np.tanh(h @ W.T + b)
            acts.append(h)
        W, b = ps[-1]
        pred = h @ W.T + b
        err = pred - Yb
        loss = float((wb * err * err).mean())
        # backward
        g = 2.0 * wb * err / (batch * NS.N_OUT)
        grads = [None] * len(ps)
        grads[-1] = [g.T @ acts[-1], g.sum(0)]
        for li in range(len(ps) - 2, -1, -1):
            g = (g @ ps[li + 1][0]) * (1.0 - acts[li + 1] ** 2)
            grads[li] = [g.T @ acts[li], g.sum(0)]
        # adam
        lr_t = lr * min(1.0, step / 300) * (0.5 * (1 + np.cos(np.pi * min(1.0, step / steps))) * 0.9 + 0.1)
        for li in range(len(ps)):
            for q in range(2):
                mom[li][q] = b1 * mom[li][q] + (1 - b1) * grads[li][q]
                vel[li][q] = b2 * vel[li][q] + (1 - b2) * grads[li][q] ** 2
                mh = mom[li][q] / (1 - b1 ** step)
                vh = vel[li][q] / (1 - b2 ** step)
                ps[li][q] -= (lr_t * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        if step % max(1, steps // 12) == 0 or step == 1:
            hist.append({"step": step, "loss": loss})
            log(f"    h={hidden} L={layers} step {step:6d}/{steps}  whitened MSE {loss:.5f}  "
                f"({time.time() - t0:.0f}s)")
    return ps, hist


def evaluate(ps, Xstat, S, C, V, Jp, M, Y, rng, jitter=0.0):
    """Held-out one-step accuracy, per material and per output, in units of the target's own
    standard deviation. A relative error of 1.0 means the network is no better than predicting the
    mean, which is the number that matters -- an absolute MSE on a quantity whose scale differs by
    four orders of magnitude between outputs says nothing."""
    in_mean, in_scale, out_mean, out_scale = Xstat
    X = make_X(S, C, V, Jp, M, jitter, rng)
    pred = NS.forward(ps, whiten(X, in_mean, in_scale))[0] * out_scale + out_mean
    gsd = Y.std(0)

    def row(e, sd):
        # An output whose target is IDENTICALLY constant (rubber's plastic correction is exactly
        # zero) has no spread to normalise by, and dividing anyway produced 6e8 in the first run.
        # Those are scored against the GLOBAL spread instead, which is the honest question there:
        # is the error small compared with the size of the thing across the whole problem?
        out = {}
        for i in range(NS.N_OUT):
            rms = float(np.sqrt((e[:, i] ** 2).mean()))
            den = sd[i] if sd[i] > 1e-6 * max(gsd[i], 1e-12) else gsd[i]
            out[NS.OUT_NAMES[i]] = float(rms / (den + 1e-30))
            out["abs_" + NS.OUT_NAMES[i]] = rms
            out["sd_" + NS.OUT_NAMES[i]] = float(sd[i])
        return out

    res = {}
    for name, mid in MAT_ID.items():
        sel = M == mid
        if not sel.any():
            continue
        res[name] = row(pred[sel] - Y[sel], Y[sel].std(0))
    res["all"] = row(pred - Y, gsd)
    return res


def to_wgsl(ps):
    """Pack the folded weights into the shader's interleaved uniform layout.

    Per hidden unit k, six vec4: four holding W1's row (14 weights, the bias in slot 14, slot 15
    dead) and two holding that unit's column of W2 (7 outputs, slot 7 dead). Then two more vec4 for
    the output biases. That is the layout the shader's inner loop reads sequentially, and it is
    generated here rather than described anywhere twice.
    """
    W1, bb1 = ps[0]
    W2, bb2 = ps[-1]
    H = W1.shape[0]
    out = np.zeros(4 * (6 * H + 2), np.float32)
    for k in range(H):
        b = 24 * k
        out[b:b + NS.N_IN] = W1[k]
        out[b + 14] = bb1[k]
        out[b + 16:b + 16 + NS.N_OUT] = W2[:, k]
    out[24 * H: 24 * H + NS.N_OUT] = bb2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--per-mat", type=int, default=500000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--drop-nuisance", action="store_true",
                    help="zero the C and v inputs: the ablation that measures what they are worth")
    a = ap.parse_args()

    logf = open(HERE / f"train_log{a.tag}.txt", "w")
    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"loading {a.data} ({a.per_mat} per material, drop_nuisance={a.drop_nuisance})")
    S, C, V, Jp, M, Y = load(HERE / a.data, a.per_mat, drop_nuisance=a.drop_nuisance)
    ntr = int(0.9 * S.shape[0])
    log(f"  {S.shape[0]} samples -> {ntr} train / {S.shape[0] - ntr} held out")

    rng = np.random.default_rng(1)
    Xall = make_X(S[:ntr], C[:ntr], V[:ntr], Jp[:ntr], M[:ntr], 0.0, rng)
    in_mean = np.median(Xall, 0).astype(np.float32)
    # a ROBUST scale: half the 16-84 percentile span, i.e. one standard deviation for a Gaussian but
    # unmoved by the APIC affine's heavy tails, which are what a plain std would be measuring
    lo, hi = np.percentile(Xall, [16, 84], axis=0)
    in_scale = np.maximum((hi - lo) / 2.0, 1e-6).astype(np.float32)
    out_mean = Y[:ntr].mean(0).astype(np.float32)
    out_scale = (Y[:ntr].std(0) + 1e-12).astype(np.float32)
    Xstat = (in_mean, in_scale, out_mean, out_scale)
    del Xall

    results = {}
    for h in a.widths:
        ps, hist = train(Xstat, S[:ntr], C[:ntr], V[:ntr], Jp[:ntr], M[:ntr], Y[:ntr],
                         h, a.layers, a.steps, a.batch, a.lr, seed=h, jitter=NS.Z_JITTER, log=log)
        ev = evaluate(ps, Xstat, S[ntr:], C[ntr:], V[ntr:], Jp[ntr:], M[ntr:], Y[ntr:],
                      np.random.default_rng(2))
        ev_j = evaluate(ps, Xstat, S[ntr:], C[ntr:], V[ntr:], Jp[ntr:], M[ntr:], Y[ntr:],
                        np.random.default_rng(3), jitter=NS.Z_JITTER)
        folded = NS.fold_normalisation(ps, in_mean, in_scale, out_mean, out_scale)
        wg = to_wgsl(folded)
        np.savez(HERE / f"weights_h{h}{a.tag}.npz",
                 wgsl=wg, hidden=h, layers=a.layers,
                 **{f"W{i}": folded[i][0] for i in range(len(folded))},
                 **{f"b{i}": folded[i][1] for i in range(len(folded))})
        results[str(h)] = {"hidden": h, "layers": a.layers, "final_loss": hist[-1]["loss"],
                           "history": hist, "held_out": ev, "held_out_z_jittered": ev_j,
                           "n_params": int(sum(W.size + b.size for W, b in ps))}
        log(f"  h={h}: held-out relative error (1.0 = no better than the mean)")
        for m in MATS + ["all"]:
            r = ev[m]
            log("    " + m.ljust(8) + "  " + "  ".join(
                f"{k}={r[k]:.3f}" for k in NS.OUT_NAMES))
    (HERE / f"train_stats{a.tag}.json").write_text(json.dumps(
        {"widths": a.widths, "layers": a.layers, "steps": a.steps, "batch": a.batch,
         "z_jitter": NS.Z_JITTER, "z_sep": NS.Z_SEP, "drop_nuisance": a.drop_nuisance,
         "in_mean": in_mean.tolist(), "in_scale": in_scale.tolist(),
         "out_mean": out_mean.tolist(), "out_scale": out_scale.tolist(),
         "results": results}, indent=2))
    logf.close()


if __name__ == "__main__":
    main()
