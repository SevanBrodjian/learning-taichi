"""Canonical, frozen, tested physics for this project — the single source of ground truth.

Tasks IMPORT this and use it unchanged. Re-deriving the MPM step or a material's parameters inside a
task is a defect (CLAUDE.md -> "Canonical physics"). `VERSION` is a content hash of the physics source;
every run records the version it used, so two tasks are provably on the same ground truth or provably
not. Changing the physics is a deliberate, version-bumping, signature-test-gated commit.

Public API:
  simulate(material, pts, area, T, n_frames, ...) -> (snaps, times, stable)   # the forward ground truth
  scene(name, n)                                                              # canonical initial conditions
  MAT                                                                         # frozen per-material params
  spread_width / pile_height / circularity                                    # shape diagnostics
  core.*                                                                      # building-block ti.func's for
                                                                              #   learned-dynamics tasks to reuse
"""
from __future__ import annotations

import hashlib
import pathlib

from . import core
from .core import (MAT, circularity, pile_height, scene, seed_box, seed_disk,
                   simulate, spread_width)


def _version() -> str:
    h = hashlib.sha256()
    d = pathlib.Path(__file__).parent
    for name in ("core.py", "signatures.py"):
        h.update((d / name).read_bytes())
    return "phys-" + h.hexdigest()[:12]


VERSION = _version()

__all__ = ["simulate", "scene", "MAT", "spread_width", "pile_height", "circularity",
           "seed_disk", "seed_box", "core", "VERSION"]
