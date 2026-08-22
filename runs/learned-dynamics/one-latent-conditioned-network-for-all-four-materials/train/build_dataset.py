"""Build the supervised dataset for the latent-conditioned constitutive net.

THIS IS NOT BACKPROP THROUGH A ROLLOUT, and that is the whole reason this task is tractable in one
night. The seam's target is an EXACT ANALYTIC FUNCTION of the per-particle state -- given the trial
stretch S and the plastic record Jp, canonical physics says precisely what the stress and the plastic
correction must be. So training is plain supervised regression on (state -> target) pairs, with no
gradient ever crossing a timestep. This project's documented failure mode (`core/03-failure-modes`)
is long-rollout differentiation; the seam was chosen so that it is not needed.

Because the label is exact for ANY input, the only thing the data distribution has to do is COVER the
states a rollout visits. Three mechanisms do that:

  1. ROLLOUTS. The oracle (analytic law, learned scaffolding -- proven equal to canonical by
     gate_oracle.py) is rolled on every canonical scene for every material, and the trial states it
     actually visits are captured mid-G2P. These are exactly in-distribution by construction.

  2. THE ISOTROPY SYMMETRY, as free augmentation. Every canonical constitutive model here is
     isotropic, so conjugating the material frame by any rotation Q maps a valid (input, target) pair
     to another valid one: S -> Q S Q^T sends tau -> Q tau Q^T and dS -> Q dS Q^T exactly. Rotating
     C and v the same way keeps the nuisance inputs consistent. This multiplies angular coverage at
     no simulation cost and bakes the symmetry into the fit instead of hoping the net discovers it.

  3. A JITTER SHELL. Each captured S is also perturbed slightly and RE-LABELLED. The label is
     analytic, so a perturbed state is not a noisy sample, it is a new exact sample -- and it puts
     training data just off the manifold the rollout visits, which is where a drifting learned sim
     ends up.

WHAT IS DELIBERATELY DECORRELATED. C and v are in the feature set because `one-nn-for-three-materials`
validated that 10-feature set and this task was told to reuse it. For THIS seam they carry no
information at all: the canonical target depends only on (S, Jp, material). Rather than pretend
otherwise, half the samples get their C and v shuffled across particles, which destroys the spurious
correlation the net would otherwise be free to exploit and then be wrong about off-distribution.
The ablation in train_mlp.py measures what those six inputs actually cost.

    .venv/Scripts/python.exe runs/.../train/build_dataset.py [--out data.npz]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import learned_sim as LS            # noqa: E402
import netspec as NS                # noqa: E402
from sim.physics import core        # noqa: E402

MATS = ["fluid", "elastic", "snow", "sand"]
SCENES = ["drop", "column", "heap", "dam", "slam", "two_blobs"]
N_PART = 3000
NF = 4


def capture_rollouts(mode="oracle", stride=53, hidden=None, l2=None):
    """Roll every material on every canonical scene and keep the trial states visited."""
    S, C, V, JP, M = [], [], [], [], []
    for scn in SCENES:
        sc = core.scene(scn, N_PART)
        for m in MATS:
            LS.reset_capture()
            g = [{"material": m, "pts": sc["pts"], "area": sc["area"], "v0": sc["v0"]}]
            _, _, _, ok, _ = LS.rollout(g, sc["T"], NF, dt=core.MAT[m]["dt"], mode=mode,
                                        capture=True, cap_stride=stride, hidden=hidden, l2=l2)
            s, c, v, jp, mm = LS.take_capture()
            good = np.isfinite(s).all(1) & np.isfinite(c).all(1) & np.isfinite(v).all(1) & np.isfinite(jp)
            S.append(s[good]); C.append(c[good]); V.append(v[good]); JP.append(jp[good]); M.append(mm[good])
            print(f"    {scn:10s} {m:8s} stable={ok} kept {int(good.sum()):7d}", flush=True)
    # the buoyancy scene: two materials, one grid, per-material density -- states no solo run visits
    for solid in ("snow", "elastic", "sand"):
        p = core.scene_pool(solid, 2400, T=1.2)
        LS.reset_capture()
        _, _, _, ok, _ = LS.rollout(p["groups"], p["T"], NF, mode=mode, capture=True,
                                    cap_stride=stride, hidden=hidden, l2=l2)
        s, c, v, jp, mm = LS.take_capture()
        good = np.isfinite(s).all(1) & np.isfinite(c).all(1) & np.isfinite(v).all(1) & np.isfinite(jp)
        S.append(s[good]); C.append(c[good]); V.append(v[good]); JP.append(jp[good]); M.append(mm[good])
        print(f"    pool       {solid:8s} stable={ok} kept {int(good.sum()):7d}", flush=True)
    return (np.concatenate(S), np.concatenate(C), np.concatenate(V),
            np.concatenate(JP), np.concatenate(M))


def rot_augment(S3, C4, V2, rng):
    """Conjugate the material frame by a random rotation. Exact for an isotropic law."""
    th = rng.uniform(0, 2 * np.pi, S3.shape[0]).astype(np.float32)
    c, s = np.cos(th), np.sin(th)
    # S' = Q S Q^T with Q = [[c,-s],[s,c]]
    a, b, d = S3[:, 0], S3[:, 1], S3[:, 2]
    S = np.stack([c * c * a - 2 * c * s * b + s * s * d,
                  c * s * (a - d) + (c * c - s * s) * b,
                  s * s * a + 2 * c * s * b + c * c * d], 1)
    C = C4.reshape(-1, 2, 2)
    Q = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)          # (N,2,2)
    Cn = np.einsum("nij,njk,nlk->nil", Q, C, Q).reshape(-1, 4)
    Vn = np.einsum("nij,nj->ni", Q, V2)
    return S.astype(np.float32), Cn.astype(np.float32), Vn.astype(np.float32)


def build(seed=0, jitter_copies=1, rot_copies=1):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    print("  capturing oracle rollouts...", flush=True)
    S3, C4, V2, JP, M = capture_rollouts()
    print(f"  captured {S3.shape[0]} raw states in {time.time() - t0:.0f}s", flush=True)

    parts = [(S3, C4, V2, JP, M)]
    for r in range(rot_copies):
        s, c, v = rot_augment(S3, C4, V2, rng)
        parts.append((s, c, v, JP, M))
    for j in range(jitter_copies):
        # a shell just OFF the visited manifold: perturb the stretch and re-label exactly
        sc = 1.0 + rng.normal(0, 0.004, (S3.shape[0], 1)).astype(np.float32)
        off = rng.normal(0, 0.003, S3.shape).astype(np.float32)
        s = S3 * sc + off
        s, c, v = rot_augment(s, C4, V2, rng)
        parts.append((s, c, v, JP * (1.0 + rng.normal(0, 0.01, JP.shape).astype(np.float32)), M))

    S3 = np.concatenate([p[0] for p in parts]).astype(np.float32)
    C4 = np.concatenate([p[1] for p in parts]).astype(np.float32)
    V2 = np.concatenate([p[2] for p in parts]).astype(np.float32)
    JP = np.concatenate([p[3] for p in parts]).astype(np.float32)
    M = np.concatenate([p[4] for p in parts]).astype(np.int32)

    # decorrelate the nuisance inputs on half the samples (see the module docstring)
    n = S3.shape[0]
    half = rng.permutation(n)[: n // 2]
    perm = rng.permutation(half)
    C4[half] = C4[perm]
    V2[half] = V2[perm]

    print(f"  labelling {n} states with the analytic teacher...", flush=True)
    Y = np.zeros((n, NS.N_OUT), np.float32)
    B = LS.CAP
    for i in range(0, n, B):
        j = min(i + B, n)
        Y[i:j] = LS.label(S3[i:j], JP[i:j], M[i:j])
    good = np.isfinite(Y).all(1)
    print(f"  labelled; {int((~good).sum())} non-finite rows dropped")
    return S3[good], C4[good], V2[good], JP[good], M[good], Y[good]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    S3, C4, V2, JP, M, Y = build(a.seed)
    np.savez_compressed(HERE / a.out, S=S3, C=C4, V=V2, Jp=JP, mat=M, Y=Y)
    stats = {"n": int(S3.shape[0]),
             "per_material": {m: int((M == core.MAT_ID[m]).sum()) for m in MATS},
             "physics_version": LS.VERSION,
             "target_std": {NS.OUT_NAMES[i]: float(Y[:, i].std()) for i in range(NS.N_OUT)},
             "target_std_per_material": {
                 m: {NS.OUT_NAMES[i]: float(Y[M == core.MAT_ID[m]][:, i].std())
                     for i in range(NS.N_OUT)} for m in MATS}}
    (HERE / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
