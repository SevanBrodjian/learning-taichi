"""Canonical, frozen, tested physics for this project — the single source of ground truth.

Tasks IMPORT this and use it unchanged. Re-deriving the MPM step or a material's parameters inside a
task is a defect (CLAUDE.md -> "Canonical physics"). `VERSION` is a content hash of the physics source;
every run records the version it used, so two tasks are provably on the same ground truth or provably
not. Changing the physics is a deliberate, version-bumping, signature-test-gated commit.

Public API:
  simulate(material, pts, area, T, n_frames, ...) -> (snaps, times, stable)   # the forward ground truth
  simulate_multi(groups, T, n_frames, ...)                                    # several materials, ONE grid
      -> (snaps, times, mats, stable, dt)
  shared_dt(materials)                                                        # the dt a shared grid forces
  scene(name, n)                                                              # canonical initial conditions
  MAT / MAT_ID                                                                # frozen per-material params
  spread_width / pile_height / repose_angle / circularity                     # shape diagnostics
  core.*                                                                      # building-block ti.func's for
                                                                              #   learned-dynamics tasks to reuse
"""
from __future__ import annotations

import hashlib
import pathlib

from . import core
from .core import (MAT, MAT_ID, circularity, dp_alpha, pile_height, repose_angle, scene, seed_box,
                   seed_disk, shared_dt, simulate, simulate_multi, spread_width, surface_profile)


def _version() -> str:
    h = hashlib.sha256()
    d = pathlib.Path(__file__).parent
    for name in ("core.py", "signatures.py"):
        h.update((d / name).read_bytes())
    return "phys-" + h.hexdigest()[:12]


VERSION = _version()

__all__ = ["simulate", "scene", "MAT", "spread_width", "pile_height", "circularity",
           "seed_disk", "seed_box", "core", "VERSION"]
