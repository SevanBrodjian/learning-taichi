"""ONE definition of the latent-conditioned constitutive network, imported by everything.

Data generation, training, the Taichi learned simulator, the WGSL exporter and the parity check all
read the layout from here, so there is exactly one answer to "what are the inputs, what are the
outputs, what is snow's code". A second copy of any of these numbers is how a port silently stops
being a port.

THE SEAM
--------
The network replaces the per-particle CONSTITUTIVE MODEL of canonical MLS-MPM: the stress that P2G
scatters, AND the plastic state update that G2P applies (snow's Stomakhin clamp, sand's Drucker-Prager
return map, the fluid's volumetric bookkeeping). B-spline P2G/G2P, the grid update and advection stay
analytic and are imported unchanged in spirit from `sim.physics.core`.

It is evaluated ONCE PER SUBSTEP, at the end of G2P, and its stress output is cached for the next
P2G. That is not an approximation: in MLS-MPM the stress P2G scatters at step n is a function of the
deformation gradient produced by G2P at step n-1, so computing it there and storing it is exactly the
same quantity, one kernel earlier.

THE POLAR FRAME
---------------
F is split as F = R S with R the polar rotation and S symmetric positive definite. The network sees
only S (rotation invariant) and predicts in that material frame; its stress is rotated back by R and
its corrected stretch is remounted as F' = R S'. This buys exact rotational equivariance for free and
is the same structuring `one-nn-for-three-materials` used.

It is also EXACTLY consistent with the analytic plastic maps. Writing F_trial = U diag(s) V^T, every
canonical return map replaces s by some s' and keeps U and V, so R' = U V^T = R is unchanged and
S' = V diag(s') V^T shares S's eigenvectors. Predicting the symmetric DELTA S' - S is therefore a
well-posed target for all four materials, and it is identically zero for elastic.

TWO LATENTS, KEPT SEPARATE
--------------------------
  z_m           IDENTITY. Four fixed, well-separated codes (the corners of a regular simplex in R^4,
                centred at the origin). One per material, shared by all its particles, never updated.
                JITTERED during training so the network learns a neighbourhood, not four lookups.
  (S, Jp)       HISTORY. Per particle, updated every substep, in the KNOWN parameterisation. The
                network predicts its update. A free learned latent state is deliberately out of
                scope: discovering one needs backprop through a long rollout.

The latent is a LABEL, not a physical axis. Four structurally unrelated materials have no ground
truth anywhere between their codes, so nothing here claims interpolation.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------- layout
Z_DIM = 4                      # material code dimensionality
N_FEAT = 10                    # S00,S01,S11, C00,C01,C10,C11, vx,vy, Jp
N_IN = N_FEAT + Z_DIM          # 14
N_IN_PAD = 16                  # padded to 4 x vec4 for the WGSL uniform-buffer packing
N_OUT = 7                      # tau00,tau01,tau11, dS00,dS01,dS11, dJp
N_OUT_PAD = 8

MATERIALS = ("fluid", "elastic", "snow", "sand")
MAT_ID = {"fluid": 0, "elastic": 1, "snow": 2, "sand": 3}

FEAT_NAMES = ["S00", "S01", "S11", "C00", "C01", "C10", "C11", "vx", "vy", "Jp"] + \
             [f"z{i}" for i in range(Z_DIM)]
OUT_NAMES = ["tau00", "tau01", "tau11", "dS00", "dS01", "dS11", "dJp"]


def simplex_codes(z_dim: int = Z_DIM, n: int = 4) -> np.ndarray:
    """`n` mutually equidistant unit-norm points in R^z_dim -- the corners of a regular simplex,
    centred at the origin.

    Well-separated and centred, both deliberately. Separation is what stops two materials sharing a
    neighbourhood in latent space; centring keeps the conditioning inputs zero-mean like every other
    feature, so no material's code dominates the first layer's scale.
    """
    v = np.eye(n, dtype=np.float64)
    v -= v.mean(axis=0, keepdims=True)                    # centre: now in an (n-1)-dim subspace
    v /= np.linalg.norm(v, axis=1, keepdims=True)         # unit norm
    out = np.zeros((n, z_dim), dtype=np.float32)
    out[:, :min(n, z_dim)] = v[:, :min(n, z_dim)]
    return out


Z_CODES = simplex_codes()                                  # (4, Z_DIM), row i = material id i
# The nearest-neighbour distance between codes. Jitter is quoted as a fraction of this so "how far
# apart the materials are" and "how big is the neighbourhood each one owns" are one number apart.
Z_SEP = float(np.min([np.linalg.norm(Z_CODES[i] - Z_CODES[j])
                      for i in range(4) for j in range(4) if i != j]))
Z_JITTER = 0.12 * Z_SEP


def polar(Fm: np.ndarray):
    """Batched 2x2 polar decomposition F = R S, R a rotation (or reflection if det F < 0), S
    symmetric. Returns (R, S) with shapes (...,2,2).

    Uses the same closed form the WGSL `polar_r` uses (and Taichi's `_polar_decompose2d`), rather
    than an SVD, so the host and the shader agree bit-for-bit up to f32 rounding.
    """
    a00, a01 = Fm[..., 0, 0], Fm[..., 0, 1]
    a10, a11 = Fm[..., 1, 0], Fm[..., 1, 1]
    det = a00 * a11 - a10 * a01
    b = np.stack([a00 + a11, a01 - a10, a10 - a01, a11 + a00], axis=-1)
    bneg = np.stack([a00 - a11, a01 + a10, a10 + a01, a11 - a00], axis=-1)
    b = np.where((det < 0)[..., None], bneg, b)
    adetB = np.abs(b[..., 0] * b[..., 3] - b[..., 2] * b[..., 1])
    k = 1.0 / np.maximum(np.sqrt(adetB), 1e-30)
    R = (b * k[..., None]).reshape(Fm.shape[:-2] + (2, 2))
    S = np.einsum("...ji,...jk->...ik", R, Fm)             # S = R^T F
    S = 0.5 * (S + np.swapaxes(S, -1, -2))                 # symmetrise away f32 asymmetry
    return R, S


def build_features(S: np.ndarray, C: np.ndarray, v: np.ndarray, Jp: np.ndarray,
                   z: np.ndarray) -> np.ndarray:
    """(N,2,2),(N,2,2),(N,2),(N,),(N,Z_DIM) -> (N, N_IN) in the frozen feature order."""
    return np.concatenate([
        S[:, 0, 0:1], S[:, 0, 1:2], S[:, 1, 1:2],
        C[:, 0, 0:1], C[:, 0, 1:2], C[:, 1, 0:1], C[:, 1, 1:2],
        v[:, 0:1], v[:, 1:2], Jp[:, None], z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------- the MLP itself
def init_params(hidden, layers=1, seed=0, n_in=N_IN, n_out=N_OUT):
    """He-ish init for a tanh MLP. `layers` is the number of HIDDEN layers."""
    rng = np.random.default_rng(seed)
    ps, d = [], n_in
    for _ in range(layers):
        ps.append([rng.normal(0, np.sqrt(1.0 / d), (hidden, d)).astype(np.float32),
                   np.zeros(hidden, np.float32)])
        d = hidden
    ps.append([rng.normal(0, np.sqrt(1.0 / d), (n_out, d)).astype(np.float32),
               np.zeros(n_out, np.float32)])
    return ps


def forward(ps, X):
    """X (N, n_in) -> (N, n_out). tanh hidden, linear head. Also returns the activations, which the
    backward pass needs."""
    acts = [X]
    h = X
    for W, b in ps[:-1]:
        h = np.tanh(h @ W.T + b)
        acts.append(h)
    W, b = ps[-1]
    return h @ W.T + b, acts


def fold_normalisation(ps, in_mean, in_scale, out_mean, out_scale):
    """Bake the input whitening and the output unwhitening into the first and last layers.

    The shader then runs a bare MLP with no normalisation step, and -- more importantly -- there is
    no second place for a scale to be typed in slightly differently. Returns a NEW parameter list.
    """
    ps = [[W.copy(), b.copy()] for W, b in ps]
    W1, b1 = ps[0]
    ps[0][1] = b1 - W1 @ (in_mean / in_scale).astype(np.float32)
    ps[0][0] = (W1 / in_scale[None, :]).astype(np.float32)
    Wl, bl = ps[-1]
    ps[-1][0] = (Wl * out_scale[:, None]).astype(np.float32)
    ps[-1][1] = (bl * out_scale + out_mean).astype(np.float32)
    return ps
