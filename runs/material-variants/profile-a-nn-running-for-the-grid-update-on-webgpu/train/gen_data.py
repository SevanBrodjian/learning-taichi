"""Harvest (node state -> node velocity) pairs from canonical water sims.

The SEAM being replaced is the whole grid update: `sim.physics.core.grid_op`. That kernel takes a
node's accumulated mass and momentum and produces its velocity, and it does FOUR things in one pass:

    1. divide the accumulated momentum by the node mass          v = p / m
    2. apply gravity                                             v.y -= dt * g
    3. clamp the normal component at a boundary, but only when the node is MOVING INTO the wall
    4. apply a Coulomb friction cap to the tangential component at that boundary

This script never re-implements any of that. It drives the canonical kernels (clear_grid, p2g,
grid_op, g2p) exactly as `sim.physics.simulate` does, snapshots the grid immediately BEFORE grid_op
(mass + momentum) and immediately AFTER it (velocity), and writes the pairs out. Ground truth is a
forward sim; nothing here needs gradients.

Two extra wrinkles, both deliberate:

  * FRICTION IS SWEPT. Canonical water has fric = 0, which makes the Coulomb cap the identity, so a
    network trained only on water would never see step 4 at all and the claim "the network does the
    whole grid update" would be hollow. Each captured pre-state is therefore re-run through the
    canonical `grid_op` at several friction coefficients, and the coefficient is fed to the network
    as an input. The STATES are water states; only the boundary law is swept.
  * SCALE. Mass and momentum are recorded in units of ONE PARTICLE MASS (m / p_mass), the same
    scene-independent normalisation the WebGPU port's fixed-point atomics already use. This is a
    multiply by a scalar that is uniform over the whole grid and known before the substep starts, so
    it folds into the first layer's weights. It is NOT the per-cell division the network has to learn.

    .venv/Scripts/python.exe runs/material-variants/profile-a-nn-running-for-the-grid-update-on-webgpu/train/gen_data.py
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import sim.physics as phys                       # noqa: E402
from sim.physics import core as C                # noqa: E402

RUN = pathlib.Path(__file__).resolve().parents[1]
OUT = RUN / "train"

FLUID = phys.MAT["fluid"]
DT = FLUID["dt"]
E = FLUID["E"]
RHO = FLUID["rho"]
NG = C.n_grid
BOUND = C.bound

# The friction coefficients the operator is fitted over. 0.0 is canonical water (the identity Coulomb
# cap); 0.5 is the canonical value for every other material; 0.25 fills the middle so the network sees
# the cap as a continuous dial rather than two isolated cases.
FRICS = (0.0, 0.25, 0.5)

# Scenes. Each is a canonical `sim.physics` scene run as fluid. `dam` and `column` are the ones that
# actually drive material into the side walls, which is where the boundary branch is exercised.
SCENES = [("drop", 9000, 1.2), ("dam", 9000, 1.4), ("column", 9000, 1.6)]

CAPTURE_EVERY = 25          # substeps between snapshots
EMPTY_KEEP = 0.03           # fraction of the (vast majority) empty cells to keep


def wall_flags():
    """The four boundary indicators, exactly the conditions grid_op branches on."""
    i = np.arange(NG)[:, None] * np.ones((1, NG))
    j = np.ones((NG, 1)) * np.arange(NG)[None, :]
    wl = (i < BOUND).astype(np.float32)
    wr = (i > NG - BOUND).astype(np.float32)
    wb = (j < BOUND).astype(np.float32)
    wt = (j > NG - BOUND).astype(np.float32)
    return wl, wr, wb, wt


WL, WR, WB, WT = wall_flags()


def targets_for(m, mom, fric):
    """Run the CANONICAL grid_op on a captured pre-state at a given friction coefficient.

    The kernel is called unchanged: the pre-state is written back into the canonical fields and
    grid_op is invoked. grid_fr is the mass-weighted friction P2G would have scattered, i.e. f*m.
    """
    C.grid_m.from_numpy(m.astype(np.float32))
    C.grid_v.from_numpy(mom.astype(np.float32))
    C.grid_fr.from_numpy((fric * m).astype(np.float32))
    C.grid_op(DT, fric, C.gravity)
    return C.grid_v.to_numpy().copy()


def run_scene(name, n, T, seed_off=0):
    sc = C.scene(name, n)
    pts = sc["pts"]
    area = sc["area"]
    npart = C._upload(pts, (0.0, 0.0), C.FLUID)
    p_vol = area / npart
    p_mass = p_vol * RHO
    C.init_state(npart)
    nsub = int(round(T / DT))
    rng = np.random.default_rng(1234 + seed_off)

    rows_in, rows_out = [], []
    for s in range(nsub):
        C.clear_grid()
        C.p2g(C.FLUID, npart, DT, E, FLUID["nu"], 0.0, 0.0, p_vol, p_mass, FLUID["fric"])
        if s % CAPTURE_EVERY == 0:
            m = C.grid_m.to_numpy().copy()
            mom = C.grid_v.to_numpy().copy()
            occ = m > 0.0
            keep = occ | (rng.random(occ.shape) < EMPTY_KEEP)
            idx = np.nonzero(keep)
            mh = (m[idx] / p_mass).astype(np.float32)
            ph = (mom[idx] / p_mass).astype(np.float32)
            feat_wall = np.stack([WL[idx], WR[idx], WB[idx], WT[idx]], 1).astype(np.float32)
            for f in FRICS:
                vt = targets_for(m, mom, f)[idx].astype(np.float32)
                X = np.concatenate(
                    [mh[:, None], ph, feat_wall, np.full((mh.size, 1), f, np.float32)], 1)
                rows_in.append(X)
                rows_out.append(vt)
            # restore the pre-state so the rollout continues on the canonical trajectory
            C.grid_m.from_numpy(m)
            C.grid_v.from_numpy(mom)
            C.grid_fr.from_numpy((FLUID["fric"] * m).astype(np.float32))
        C.grid_op(DT, FLUID["fric"], C.gravity)
        C.g2p(C.FLUID, npart, DT, FLUID["tc"], FLUID["ts"], E, FLUID["nu"], 0.0)
    X = np.concatenate(rows_in, 0)
    Y = np.concatenate(rows_out, 0)
    print(f"  {name}: {nsub} substeps, {X.shape[0]} samples, p_mass={p_mass:.3e}")
    return X, Y, p_mass


def main():
    Xs, Ys, pms = [], [], []
    for k, (name, n, T) in enumerate(SCENES):
        X, Y, pm = run_scene(name, n, T, seed_off=k)
        Xs.append(X)
        Ys.append(Y)
        pms.append(pm)
    X = np.concatenate(Xs, 0)
    Y = np.concatenate(Ys, 0)
    perm = np.random.default_rng(0).permutation(X.shape[0])
    X, Y = X[perm], Y[perm]
    ntr = int(0.9 * X.shape[0])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "gridop_data.npz",
                        Xtr=X[:ntr], Ytr=Y[:ntr], Xte=X[ntr:], Yte=Y[ntr:],
                        p_mass=np.array(pms, np.float64), dt=DT, gravity=C.gravity,
                        physics_version=phys.VERSION)
    occ = X[:, 0] > 0
    print("total", X.shape, "occupied frac", occ.mean())
    print("m_hat  occupied: min %.4g  p50 %.4g  p99 %.4g  max %.4g"
          % tuple(np.percentile(X[occ, 0], [0, 50, 99, 100])))
    print("|p_hat| occupied: p50 %.4g  p99 %.4g  max %.4g"
          % tuple(np.percentile(np.abs(X[occ, 1:3]).max(1), [50, 99, 100])))
    print("|v| target: p50 %.4g  p99 %.4g  max %.4g"
          % tuple(np.percentile(np.abs(Y).max(1), [50, 99, 100])))
    print("physics_version", phys.VERSION)


if __name__ == "__main__":
    main()
