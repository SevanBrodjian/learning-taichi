# Canonical physics — what's in it, and how code gets promoted

`sim/physics/` is the **single, frozen, tested source of ground truth**. Every task imports it and uses
it unchanged. Re-deriving the MPM step or a material's parameters inside a task is a defect. This is the
mechanism that kills ground-truth drift (e.g. "snow quietly started behaving like elastic across tasks").

## What is canonical right now
- **MLS-MPM transfer skeleton** (P2G / grid update with Coulomb friction / G2P), `n_grid=128`.
- **fluid** — weakly-compressible pressure from `J`, plus an optional Newtonian viscosity `mu_visc (C+Cᵀ)`.
- **elastic** — corotated stress from the deformation gradient `F`.
- **snow** — elastic + Stomakhin plastic clamp + hardening. **Crumbles and holds an angle of repose**
  (asserted by the golden signatures).
- Frozen parameters live in `core.MAT`; scenes in `core.scene`; the forward roll is `core.simulate`.

## Not yet canonical (promote-later)
- **Surface tension** (the continuum-surface-force capillary term). A working implementation exists in
  `sim/fluid_surface_tension.py`; it needs a careful port + a passing golden signature before it becomes
  canonical. Do not add it to `core.py` without that test.

## Promotion criteria — how code becomes canonical physics
The default is **not** to promote: experiment-specific physics stays in the task's own code. Promote only
when all three gates are met, as a deliberate, reviewed commit:

1. **It is ground truth**, not experiment logic — a forward model or a material, the thing other tasks
   should measure against, not a task's optimizer / net / harness code.
2. **The golden signatures pass** (`python -m sim.physics.signatures`, or `sim/tests/test_signatures.py`).
   New canonical behavior adds a new signature asserting its qualitative truth.
3. **The version bumps.** `sim.physics.VERSION` is a content hash of the physics source; a promotion
   changes it. Every run records the `physics_version` it used, so two tasks are provably on the same
   ground truth or provably not.

Trigger to consider promotion: a second task needs the same physics, **or** a task establishes a new
canonical material/model future tasks should share. The orchestrator proposes it; the golden tests gate it.

## Ground truth is a *forward* sim
Ground truth never needs gradients. If you are generating observations to fit or evaluate a network, use
`core.simulate` (forward, cheap, stable). A task that must optimize *through* the physics builds its own
differentiable variant and says so in its contract — it does not make the canonical GT differentiable.

## The portable idea (carries to the next project)
"Freeze the ground truth." Every project has a data-generating process — a simulator, an environment, a
dataset. It lives in one versioned, tested module; tasks import it unchanged; a golden test suite encodes
the domain's qualitative invariants; runs are stamped with the version; forking is a defect. Porting means
swapping `sim/physics/` for the new domain's canonical model and refilling the golden tests.
