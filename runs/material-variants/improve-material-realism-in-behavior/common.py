"""Shared measurement helpers for the material-realism task.

Everything here is DIAGNOSTIC (shape / volume / fragmentation read off particle snapshots). No physics
is defined in this file -- the physics is `sim.physics`, imported unchanged.
"""
import numpy as np
from scipy import ndimage

RES = 160          # occupancy raster resolution used by every area / fragment measure


def occupancy(snap, res=RES):
    """Boolean occupancy raster of a particle cloud on the unit square."""
    xs = np.clip((snap[:, 0] * res).astype(int), 0, res - 1)
    ys = np.clip((snap[:, 1] * res).astype(int), 0, res - 1)
    occ = np.zeros((res, res), bool)
    occ[xs, ys] = True
    return occ


def occupied_area(snap, res=RES):
    """Area of the region the particles occupy, in domain units (1.0 = the whole unit square).

    Filled with a single binary closing so a one-cell sampling hole inside a dense body does not
    count as empty space; the closing is symmetric so it does not systematically inflate the outline.
    """
    occ = ndimage.binary_closing(occupancy(snap, res), np.ones((3, 3), bool))
    occ = ndimage.binary_fill_holes(occ)
    return float(occ.sum()) / (res * res)


def retained_area(snaps, res=RES):
    """occupied_area of every frame divided by the occupied area of frame 0. 1.0 = the body still
    covers exactly the region it started with; 0.7 = it has lost 30% of its footprint."""
    a0 = occupied_area(snaps[0], res)
    return np.array([occupied_area(s, res) / a0 for s in snaps])


def fragment_count(snap, res=RES, min_cells=6):
    """Number of connected components of the occupancy raster that are bigger than `min_cells`.

    1 = the body is in one piece. Small components are ignored so a couple of flung particles do not
    read as "the blob shattered"; the threshold is stated because the count is threshold-dependent."""
    occ = ndimage.binary_closing(occupancy(snap, res), np.ones((3, 3), bool))
    lab, k = ndimage.label(occ, structure=np.ones((3, 3), int))
    if k == 0:
        return 0
    sizes = ndimage.sum(occ, lab, range(1, k + 1))
    return int((sizes >= min_cells).sum())


def detached_fraction(snap, res=RES):
    """Fraction of particles NOT in the largest connected component. This is the quantitative form of
    "it broke apart": 0 = every particle is in one body, 0.3 = a third of the material has separated."""
    occ = ndimage.binary_closing(occupancy(snap, res), np.ones((3, 3), bool))
    lab, k = ndimage.label(occ, structure=np.ones((3, 3), int))
    if k <= 1:
        return 0.0
    sizes = ndimage.sum(occ, lab, range(1, k + 1))
    big = int(np.argmax(sizes)) + 1
    xs = np.clip((snap[:, 0] * res).astype(int), 0, res - 1)
    ys = np.clip((snap[:, 1] * res).astype(int), 0, res - 1)
    return float((lab[xs, ys] != big).mean())


def front_position(snap, q=99.0):
    """Right-hand front of a spreading body: the q-th percentile of particle x. Used for dam-break
    runout, where the interesting quantity is how far the leading edge travels, not the bulk width."""
    return float(np.percentile(snap[:, 0], q))


def centroid(snap):
    return snap.mean(axis=0)


def free_surface_y(snap, q=98.0):
    """Height of a body's free surface: the q-th percentile of particle y."""
    return float(np.percentile(snap[:, 1], q))


def summarize(snaps, core):
    """The standard bundle of shape diagnostics for one rollout."""
    last = snaps[-1]
    return {
        "spread_width": core.spread_width(last),
        "pile_height": core.pile_height(last),
        "repose_angle": core.repose_angle(last),
        "retained_area_final": float(retained_area(snaps)[-1]),
        "retained_area_min": float(retained_area(snaps).min()),
        "fragments_final": fragment_count(last),
        "detached_fraction_final": detached_fraction(last),
        "front_final": front_position(last),
    }
