"""Capture WHOLE 128x128 grid snapshots (pre-grid_op state + canonical post-grid_op velocity).

The cell-shuffled dataset in gen_data.py cannot express the constraint that actually matters. G2P does
not read a node velocity, it reads a weighted spatial DERIVATIVE of the node velocity field (the affine
matrix C, whose entries carry a 1/dx^2 factor). Fitting the operator cell by cell leaves the derivative
of the fitted field free, and a small pointwise error with no spatial correlation becomes a derivative
error of the same order as the signal. Training against that requires neighbours, so it requires whole
grids.

    .venv/Scripts/python.exe .../train/gen_grids.py
"""
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
NG = C.n_grid
FRICS = (0.0, 0.25, 0.5)
SCENES = [("drop", 6000, 1.1), ("dam", 9000, 1.3), ("column", 9000, 1.5)]
CAPTURE_EVERY = 60


def main():
    rng = np.random.default_rng(11)
    Ms, Ps, Vs, Fs = [], [], [], []
    for k, (name, n, T) in enumerate(SCENES):
        sc = C.scene(name, n)
        npart = C._upload(sc["pts"], (0.0, 0.0), C.FLUID)
        p_vol = sc["area"] / npart
        p_mass = p_vol * FLUID["rho"]
        C.init_state(npart)
        nsub = int(round(T / DT))
        got = 0
        for s in range(nsub):
            C.clear_grid()
            C.p2g(C.FLUID, npart, DT, FLUID["E"], FLUID["nu"], 0.0, 0.0, p_vol, p_mass, FLUID["fric"])
            if s % CAPTURE_EVERY == 0:
                m = C.grid_m.to_numpy().copy()
                mom = C.grid_v.to_numpy().copy()
                f = float(FRICS[rng.integers(0, len(FRICS))])
                C.grid_fr.from_numpy((f * m).astype(np.float32))
                C.grid_op(DT, f, C.gravity)
                Ms.append((m / p_mass).astype(np.float32))
                Ps.append((mom / p_mass).astype(np.float32))
                Vs.append(C.grid_v.to_numpy().astype(np.float32))
                Fs.append(f)
                got += 1
                # restore the pre-state and take the canonical step so the rollout stays on trajectory
                C.grid_m.from_numpy(m)
                C.grid_v.from_numpy(mom)
                C.grid_fr.from_numpy((FLUID["fric"] * m).astype(np.float32))
            C.grid_op(DT, FLUID["fric"], C.gravity)
            C.g2p(C.FLUID, npart, DT, FLUID["tc"], FLUID["ts"], FLUID["E"], FLUID["nu"], 0.0)
        print(f"  {name}: {got} snapshots  p_mass={p_mass:.3e}", flush=True)
    M = np.stack(Ms), np.stack(Ps), np.stack(Vs), np.array(Fs, np.float32)
    np.savez_compressed(D / "grids.npz", m=M[0], mom=M[1], v=M[2], fric=M[3],
                        physics_version=phys.VERSION)
    print("grids", M[0].shape, "->", D / "grids.npz")


if __name__ == "__main__":
    main()
