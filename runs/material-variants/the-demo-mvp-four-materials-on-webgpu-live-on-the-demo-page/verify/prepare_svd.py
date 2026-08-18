"""Build the SVD unit-test set and its Taichi reference, BEFORE the WGSL SVD goes anywhere near
the simulation.

Why this file exists at all: a subtly wrong 2x2 SVD does not produce garbage, it produces
plausible-looking motion. Snow still crumbles, sand still slumps, and nothing on screen says the
singular values are wrong. So the SVD is proved in isolation, on matrices chosen to break it, and
the proof is a file on disk.

The matrix set is deliberately adversarial:
  * random well-conditioned matrices (the easy case, to establish a floor)
  * near-rotations (P ~ I: the Jacobi branch is skipped, |P01| < 1e-5)
  * near-singular (one singular value -> 0: adetB and 1/sqrt(adetB) blow up)
  * negative determinant (a REFLECTION: the polar decomposition has to take its second branch or
    it silently returns a rotation with the wrong handedness)
  * strongly anisotropic (condition number up to 1e4)
  * exact zeros and exact identities (the special-cased branch)
  * the deformation gradients an actual snow/sand rollout produces, sampled from a canonical run,
    so the test set contains the distribution the sim will really feed it

Writes:
    verify/svd_in.f32   -- (k,4) float32, row-major (m00,m01,m10,m11), what the browser fetches
    verify/svd_ref.npz  -- ti.svd's U, S, V on the SAME float32 inputs
"""
import pathlib
import sys

import numpy as np

RUN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN.parents[2]))

import taichi as ti                              # noqa: E402
import sim.physics as phys                       # noqa: E402
from sim.physics import core as C                # noqa: E402


def build_matrices(seed=20260817):
    rng = np.random.default_rng(seed)
    blocks = []

    def add(m, tag):
        blocks.append((tag, np.asarray(m, dtype=np.float32).reshape(-1, 2, 2)))

    add(rng.normal(size=(160, 2, 2)), "random")

    # near-rotations: R(theta) @ (I + tiny)
    th = rng.uniform(0, 2 * np.pi, 80)
    R = np.stack([np.stack([np.cos(th), -np.sin(th)], -1),
                  np.stack([np.sin(th), np.cos(th)], -1)], -2)
    add(R @ (np.eye(2) + rng.normal(scale=1e-7, size=(80, 2, 2))), "near_rotation")

    # near-singular: one singular value driven to ~0
    U = R[:60]
    s = np.stack([rng.uniform(0.5, 2.0, 60), 10.0 ** rng.uniform(-8, -3, 60)], -1)
    th2 = rng.uniform(0, 2 * np.pi, 60)
    V = np.stack([np.stack([np.cos(th2), -np.sin(th2)], -1),
                  np.stack([np.sin(th2), np.cos(th2)], -1)], -2)
    add(U @ (s[:, :, None] * np.transpose(V, (0, 2, 1))), "near_singular")

    # reflections: det < 0, which is the branch a naive closed-form polar rotation gets wrong
    m = rng.normal(size=(80, 2, 2))
    m[:, 1] *= -1.0
    det = m[:, 0, 0] * m[:, 1, 1] - m[:, 1, 0] * m[:, 0, 1]
    m[det > 0] = m[det > 0][:, ::-1]             # force the sign
    add(m, "reflection")

    # strongly anisotropic, condition number 1e1 .. 1e4
    cond = 10.0 ** rng.uniform(1, 4, 80)
    s2 = np.stack([np.ones(80), 1.0 / cond], -1)
    th3 = rng.uniform(0, 2 * np.pi, 80)
    V3 = np.stack([np.stack([np.cos(th3), -np.sin(th3)], -1),
                   np.stack([np.sin(th3), np.cos(th3)], -1)], -2)
    add(R[:80] @ (s2[:, :, None] * np.transpose(V3, (0, 2, 1))), "anisotropic")

    add(np.zeros((4, 2, 2)), "zero")
    add(np.tile(np.eye(2), (4, 1, 1)), "identity")
    add(np.tile(np.diag([1.0, -1.0]), (4, 1, 1)), "pure_reflection")
    # exactly the plastic clamp boundary snow lives on
    tc, ts = C.MAT["snow"]["tc"], C.MAT["snow"]["ts"]
    add(np.tile(np.diag([1.0 + ts, 1.0 - tc]), (4, 1, 1)), "snow_clamp_edge")

    # the real distribution: F from a canonical snow and a canonical sand rollout
    for mat in ("snow", "sand"):
        sc = C.scene("heap", n=800)
        C.simulate(mat, sc["pts"], sc["area"], 0.6, 4, v0=sc["v0"])
        Fs = C.F.to_numpy()[:800].astype(np.float32)
        add(Fs, "rollout_" + mat)

    tags, mats = [], []
    for tag, blk in blocks:
        tags += [tag] * blk.shape[0]
        mats.append(blk)
    return np.concatenate(mats, 0).astype(np.float32), np.array(tags)


def main():
    mats, tags = build_matrices()
    k = mats.shape[0]
    print("matrices:", k, {t: int((tags == t).sum()) for t in dict.fromkeys(tags)})

    A = ti.Matrix.field(2, 2, float, k)
    Uf = ti.Matrix.field(2, 2, float, k)
    Sf = ti.Matrix.field(2, 2, float, k)
    Vf = ti.Matrix.field(2, 2, float, k)

    @ti.kernel
    def go():
        for p in range(k):
            u, s, v = ti.svd(A[p])
            Uf[p] = u
            Sf[p] = s
            Vf[p] = v

    A.from_numpy(mats)
    go()
    U, S, V = Uf.to_numpy(), Sf.to_numpy(), Vf.to_numpy()

    (RUN / "verify" / "svd_in.f32").write_bytes(mats.reshape(-1).tobytes())
    np.savez(RUN / "verify" / "svd_ref.npz", A=mats, U=U, S=S, V=V, tags=tags,
             physics_version=phys.VERSION)
    rec = U @ S @ np.transpose(V, (0, 2, 1))
    print("taichi reconstruction max err:", float(np.abs(rec - mats).max()))
    print("wrote svd_in.f32 (%d matrices) + svd_ref.npz" % k)


if __name__ == "__main__":
    main()
