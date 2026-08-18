"""Score the WGSL 2x2 SVD against ti.svd on the exact same float32 inputs.

Four independent checks, because any one of them alone can be passed by a wrong decomposition:
  1. RECONSTRUCTION   ||U diag(s) V^T - A|| / ||A||   -- catches wrong factors
  2. ORTHOGONALITY    ||U^T U - I||, ||V^T V - I||    -- catches a U or V that is not a rotation
  3. ORDERING         s0 >= s1                        -- the plastic clamps assume it
  4. SINGULAR VALUES  |s_wgsl - s_taichi|             -- the ONLY thing the return maps consume,
     and the one an otherwise-consistent-but-different factorisation would still get right or wrong

Checks 1-3 prove the WGSL routine is *a* correct SVD. Check 4 proves it is *Taichi's* SVD, which is
what makes the ported snow clamp and Drucker-Prager return map land on canonical's answer rather
than a different-but-also-valid one.
"""
import json
import pathlib

import numpy as np

RUN = pathlib.Path(__file__).resolve().parents[1]
V = RUN / "verify"


def main():
    ref = np.load(V / "svd_ref.npz", allow_pickle=False)
    A, Ut, St, Vt, tags = ref["A"], ref["U"], ref["S"], ref["V"], ref["tags"]
    k = A.shape[0]
    raw = np.frombuffer((V / "out" / "svd_out.f32").read_bytes(), dtype=np.float32)
    assert raw.size == k * 12, "expected %d floats, got %d" % (k * 12, raw.size)
    Ug = raw[:k * 4].reshape(k, 2, 2)
    Sg = raw[k * 4:k * 8].reshape(k, 4)[:, :2]
    Vg = raw[k * 8:].reshape(k, 2, 2)

    Sd = np.zeros((k, 2, 2), dtype=np.float64)
    Sd[:, 0, 0] = Sg[:, 0]
    Sd[:, 1, 1] = Sg[:, 1]
    rec = Ug.astype(np.float64) @ Sd @ np.transpose(Vg.astype(np.float64), (0, 2, 1))
    scale = np.maximum(np.abs(A).max(axis=(1, 2)), 1e-6)
    rec_err = np.abs(rec - A).max(axis=(1, 2)) / scale

    I = np.eye(2)
    uo = np.abs(np.transpose(Ug, (0, 2, 1)) @ Ug - I).max(axis=(1, 2))
    vo = np.abs(np.transpose(Vg, (0, 2, 1)) @ Vg - I).max(axis=(1, 2))
    order = Sg[:, 0] - Sg[:, 1]

    st = np.stack([St[:, 0, 0], St[:, 1, 1]], -1)
    sv_abs = np.abs(Sg - st).max(axis=1)
    sv_rel = sv_abs / np.maximum(np.abs(st).max(axis=1), 1e-6)

    finite = np.isfinite(Ug).all(axis=(1, 2)) & np.isfinite(Sg).all(axis=1) & np.isfinite(Vg).all(axis=(1, 2))

    rows = []
    for t in dict.fromkeys(tags.tolist()):
        m = tags == t
        rows.append({
            "family": t, "n": int(m.sum()),
            "max_rel_reconstruction": float(rec_err[m].max()),
            "max_orthogonality_U": float(uo[m].max()),
            "max_orthogonality_V": float(vo[m].max()),
            "order_violations": int((order[m] < -1e-6).sum()),
            "max_abs_singular_vs_taichi": float(sv_abs[m].max()),
            "max_rel_singular_vs_taichi": float(sv_rel[m].max()),
            "non_finite": int((~finite[m]).sum()),
        })

    summary = {
        "n_matrices": int(k),
        "max_rel_reconstruction": float(rec_err.max()),
        "max_orthogonality": float(max(uo.max(), vo.max())),
        "order_violations": int((order < -1e-6).sum()),
        "max_abs_singular_vs_taichi": float(sv_abs.max()),
        "max_rel_singular_vs_taichi": float(sv_rel.max()),
        "non_finite": int((~finite).sum()),
        "by_family": rows,
    }
    # thresholds: f32 has ~1.2e-7 of relative resolution, and these are 2x2 products of 3-4 terms,
    # so a few ulps of accumulation is expected. 1e-4 is generous but still ~1000x tighter than any
    # error that would matter to a plastic return map.
    summary["pass"] = bool(
        summary["max_rel_reconstruction"] < 1e-4
        and summary["max_orthogonality"] < 1e-4
        and summary["order_violations"] == 0
        and summary["max_rel_singular_vs_taichi"] < 1e-4
        and summary["non_finite"] == 0)

    (V / "out" / "svd_score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        worst = int(np.argmax(rec_err))
        print("\nWORST reconstruction case (%s):" % tags[worst])
        print(" A =", A[worst].tolist())
        print(" wgsl U", Ug[worst].tolist(), "s", Sg[worst].tolist(), "V", Vg[worst].tolist())
        print(" taichi U", Ut[worst].tolist(), "s", st[worst].tolist(), "V", Vt[worst].tolist())
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
