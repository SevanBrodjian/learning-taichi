# STATUS — main

**Last worker:** `improve-material-realism-in-behavior` (material-variants), finished.

The canonical physics changed for the first time since sand. `sim/physics/` now carries **per-material
density, Poisson ratio and boundary friction**, where all three used to be single globals shared by every
material. Rubber holds its volume through an impact, water is no longer squashy, and sand and rubber sink
in water while snow floats — with no buoyancy force anywhere in the code, because density enters as
particle mass and Archimedes falls out of the grid transfer.

**Physics version moved `phys-bebeaafbe73e` -> `phys-c518316a4a05`.** All 15 pre-existing golden
signatures stayed green and 9 new ones were added (24 total, all passing).

On disk, not committed:
- `sim/physics/core.py`, `sim/physics/signatures.py`, `sim/physics/__init__.py` — the change itself.
- `runs/material-variants/improve-material-realism-in-behavior/` — manifest, bespoke page, 17 clips,
  6 figures, the diagnosis sweeps, the ablations and the signature output.
- `reports/training/core/05-material-stiffness.md` — rewritten as "Material dials: E, nu, rho";
  `reports/training/prerequisites/04-math-toolkit.md` — new section on scaling and gauge freedom;
  `reports/training/index.json` — retitled that section.
- `spec/registry/materials.json` — regenerated (never hand-edited); `spec/registry/metrics.json` — four
  new entries (`rest_depth`, `submerged_fraction`, `volume_ratio`, `retained_area`);
  `harness/tools/sync_registry.py` — documents `rho`/`nu`/`fric` and records the new elastic drift.

**Two things the next session needs to know.**

1. **The shipped Demo is now STALE and was deliberately left that way.** The user asked for this change
   not to reach the demo yet. `harness/dashboard/src/components/mpm/params.js` is generated from
   `sim.physics` and still stamps `phys-bebeaafbe73e`: it carries the old `E` and `dt` for every
   material, one global `NU`, and no `rho` or `fric` at all, so it will not reproduce the new materials.
   Regenerating means running the demo run's `web/gen_params.py` then `web/sync_to_dashboard.py`, and the
   WGSL itself needs the per-material `nu`, `rho` and friction plumbed through before it would be right.
2. **Kernel signatures changed.** `core.p2g` takes `nu` and `fric`, `core.g2p` takes `nu`. The only
   caller outside the library is `runs/material-variants/interactive-simulation-of-one-material/verify/
   gpu_bench.py`, a completed run's benchmark script, which was left alone.

Also worth carrying forward: a pool of water seeded by uniform random sampling **compacts as it settles**
(free surface fell 25% over 2.2 s) because the weakly-compressible fluid's pressure comes from an advected
`J` rather than the actual particle packing. `core.seed_lattice` more than halves it and is what
`scene_pool` uses. This is not fixed, only measured and worked around.
