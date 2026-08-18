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
- **sand** — Hencky (log-strain) elasticity + Drucker-Prager return mapping (Klár et al. 2016), friction
  angle `phi`. **Cohesionless**: its shear strength is proportional to confining pressure, so unlike snow
  it cannot stand a vertical wall, and an over-steep heap relaxes to a finite angle of repose (asserted by
  the golden signatures).
- Frozen parameters live in `core.MAT`; scenes in `core.scene`; the forward roll is `core.simulate`.
- **Multi-material**: `core.simulate_multi(groups, ...)` runs several materials in ONE shared grid via a
  per-particle `mat_id` and a runtime branch. Each material still takes exactly its canonical path. The
  signatures assert that a single material pushed through this path lands where `simulate` lands, to within
  the simulator's own run-to-run noise. A shared grid forces a shared timestep: `core.shared_dt(materials)`
  is `min(dt)` over the materials present, and it is not optional.

### Known caveat on the plastic materials (measured, unresolved)
The settled shape of every non-elastic canonical material drifts with the **substep count**, not just with
physical time. On an over-steep heap, elastic gives an identical settled shape from `dt = 1.25e-5` to
`4e-4`, while snow relaxes from holding the seeded 60° slope at its canonical `dt` to about 17° at `dt/8`,
and sand from about 25° to about 14°. Any angle-of-repose number from this library must therefore be quoted
**with the timestep it was measured at**. The golden signatures are written as qualitative orderings that
hold across the tested range, precisely so they do not encode a `dt`-specific number.

## Not yet canonical (promote-later)
- **Surface tension** (the continuum-surface-force capillary term). A working implementation exists in
  `sim/fluid_surface_tension.py`; it needs a careful port + a passing golden signature before it becomes
  canonical. Do not add it to `core.py` without that test.
- **Material-to-material contact.** `simulate_multi` gives every node one velocity, so two different
  materials sharing a node exchange momentum as if the node held one blended material. That is enough for
  coexistence and is NOT a calibrated multi-phase contact model. Any claim about how sand and water
  interact at their interface needs a real contact treatment first.

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

## Two traps in this module — read before "fixing" either

**1. `Vt` is misnamed, and that is not a bug.** `ti.svd` returns `(U, S, V)` with $A = U S V^\top$, so
the third value is **V**, not $V^\top$. The plastic reconstructions therefore assemble `U @ S' @ Vt`
= $U \Sigma' V$, not $U \Sigma' V^\top$. That is a **gauge** choice: it right-multiplies $F$ by the
rotation $VV$, and an isotropic material cannot see a rotation of its reference configuration. Verified
numerically over 4000 random $F$ — Kirchhoff stress, $\det F$ and the singular values are invariant to
1e-13, and no golden signature moves. Note that `corotated_PFt` **does** transpose it
(`U @ Vt.transpose()`), which is correct *there* because a rotation is not gauge-free. The two uses look
inconsistent and both are right. Changing the convention would silently invalidate every state-for-state
port comparison for zero physical gain. Explained in full in `reports/training/core/16-svd-in-practice.md`.

**2. `VERSION` normalises line endings, deliberately.** It is a content hash of `core.py` +
`signatures.py`. Hashing *raw bytes* made the stamp depend on how git materialised the file: on Windows a
plain `git checkout` of an unchanged `core.py` rewrote LF to CRLF and moved the version, so byte-identical
physics stamped two different versions and a fresh worktree could disagree with `main` for no reason. A
version that changes when nothing changed converts "provably the same ground truth" into noise, which is
worse than having no version. Do not revert the normalisation.

## Ground truth is a *forward* sim
Ground truth never needs gradients. If you are generating observations to fit or evaluate a network, use
`core.simulate` (forward, cheap, stable). A task that must optimize *through* the physics builds its own
differentiable variant and says so in its contract — it does not make the canonical GT differentiable.

## The portable idea (carries to the next project)
"Freeze the ground truth." Every project has a data-generating process — a simulator, an environment, a
dataset. It lives in one versioned, tested module; tasks import it unchanged; a golden test suite encodes
the domain's qualitative invariants; runs are stamped with the version; forking is a defect. Porting means
swapping `sim/physics/` for the new domain's canonical model and refilling the golden tests.
