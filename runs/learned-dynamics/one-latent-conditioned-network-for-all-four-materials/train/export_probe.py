"""Export everything the WebGPU parity pass needs, and the host answers it will be scored against.

Two levels of parity, because either one alone can hide a bug:

  1. THE MLP ALONE. A few thousand real feature vectors, padded exactly as the shader packs them,
     with the host's forward pass saved beside them. This isolates the arithmetic: if the shader's
     dot products, tanh or weight layout are wrong, it shows here as a clean numerical disagreement
     with nowhere to hide.
  2. THE WHOLE LEARNED SIMULATOR. The same four-material initial condition rolled forward by the
     Taichi learned simulator and by the WGSL one. This catches everything level 1 cannot: a wrong
     feature ORDER, a wrong material code, the polar frame reconstructed differently, the plastic
     correction applied to the wrong tensor.

Trajectory agreement between the two is scored against the same chaotic reality as everywhere else --
they are different f32 orderings of the same arithmetic and will diverge. Level 1 is the one that has
to be tight.

    .venv/Scripts/python.exe runs/.../train/export_probe.py --hidden 64
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))

import learned_sim as LS            # noqa: E402
import netspec as NS                # noqa: E402
from sim.physics import core        # noqa: E402

MATS = ["fluid", "elastic", "snow", "sand"]
NF = 30
T = 0.9


def rng_box(x0, x1, y0, y1, n, seed):
    """The SAME xorshift32 stream the WGSL harness uses, so the two solvers get bit-identical
    initial positions. Reimplementing the seeding on each side and hoping they agree is how a parity
    check turns into a comparison of two different scenes."""
    s = np.uint32(seed if seed else 1)
    out = np.zeros((n, 2), np.float32)

    def rnd():
        nonlocal s
        s ^= np.uint32(s << np.uint32(13))
        s ^= np.uint32(s >> np.uint32(17))
        s ^= np.uint32(s << np.uint32(5))
        return float(s) / 4294967296.0

    for i in range(n):
        out[i, 0] = x0 + (x1 - x0) * rnd()
        out[i, 1] = y0 + (y1 - y0) * rnd()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--probe", type=int, default=4096)
    a = ap.parse_args()

    d = np.load(HERE / f"weights_h{a.hidden}{a.tag}.npz")
    nl = int(d["layers"])
    ps = [[d[f"W{i}"], d[f"b{i}"]] for i in range(nl + 1)]
    H, L2 = LS.upload_weights(ps)

    # ---- level 1: the MLP alone -------------------------------------------------------------
    ds = np.load(HERE / "data.npz")
    idx = np.random.default_rng(5).choice(ds["S"].shape[0], a.probe, replace=False)
    idx.sort()
    S, C, V, Jp, M = ds["S"][idx], ds["C"][idx], ds["V"][idx], ds["Jp"][idx], ds["mat"][idx]
    feats = np.concatenate([S, C, V, Jp[:, None], NS.Z_CODES[M]], 1).astype(np.float32)
    X = np.zeros((a.probe, NS.N_IN_PAD), np.float32)
    X[:, :NS.N_IN] = feats
    X[:, 14] = 1.0                       # the shader pins slot 14 so W1's 15th column is the bias
    host = NS.forward(ps, feats)[0].astype(np.float32)
    np.savez(HERE / "probe_inputs.npz", X=X, host=host, mat=M)
    print(f"  probe: {a.probe} samples, host output range "
          f"[{host.min():.3f}, {host.max():.3f}]")

    # ---- level 2: the whole learned simulator ------------------------------------------------
    q = a.n // 4
    edges = [(0.10, 0.31), (0.31, 0.50), (0.50, 0.71), (0.71, 0.90)]
    counts = [q, q, q, a.n - 3 * q]
    y0, y1 = core.floor_y, 0.30
    groups, off, meta = [], 0, []
    # ONE particle volume for the whole domain, because the WGSL port carries a single pVol uniform.
    # Matching it here rather than letting each group keep its own area/n is what makes the two
    # solvers the same experiment.
    box_area = sum((e[1] - e[0]) * (y1 - y0) for e in edges)
    pvol = box_area / a.n
    for i, m in enumerate(MATS):
        pts = rng_box(edges[i][0], edges[i][1], y0, y1, counts[i], i + 1)
        groups.append({"material": m, "pts": pts, "area": pvol * counts[i], "v0": (0.0, 0.0)})
        meta.append({"material": m, "n": int(counts[i])})
        off += counts[i]
    dt = core.shared_dt(MATS)
    snaps, _, mid, ok, _ = LS.rollout(groups, T, NF, dt=dt, mode="nn", hidden=H, l2=L2)
    pts = np.concatenate([g["pts"] for g in groups]).astype(np.float32)
    (RUN / "verify" / "ic_mixed4.f32").write_bytes(pts.tobytes())
    (RUN / "verify" / "mat_mixed4.i32").write_bytes(mid.astype(np.int32).tobytes())
    (RUN / "verify" / "host_nn_mixed4.f32").write_bytes(snaps.astype(np.float32).tobytes())
    # the analytic reference on the SAME initial condition, so the WGSL page can show both
    an, _, _, ok2, _ = LS.rollout(groups, T, NF, dt=dt, mode="oracle")
    (RUN / "verify" / "host_analytic_mixed4.f32").write_bytes(an.astype(np.float32).tobytes())
    meta_json = {"ic_file": "ic_mixed4.f32", "mat_file": "mat_mixed4.i32",
                 "host_nn_file": "host_nn_mixed4.f32",
                 "host_analytic_file": "host_analytic_mixed4.f32",
                 "n": int(a.n), "n_frames": NF, "T": T, "dt": dt, "pVol": float(pvol),
                 "groups": meta, "host_stable": bool(ok), "host_analytic_stable": bool(ok2)}
    (HERE / "wgsl_rollout.json").write_text(json.dumps(meta_json, indent=1))
    print(f"  host learned rollout stable={ok}  analytic stable={ok2}  dt={dt}  pVol={pvol:.3e}")
    print("  wrote", HERE / "wgsl_rollout.json")


if __name__ == "__main__":
    main()
