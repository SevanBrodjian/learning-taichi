# The Demo MVP — the target we are now building toward

**Set 2026-08-16.** This is the primary target. It will expand as we work; keep it current, and read it
before proposing or briefing anything demo-related. Aesthetic direction: `spec/aesthetic.md`. Page/plumbing
contract for the Demo tab: `coordination/rebuild-plan.md` → B5.

## What the MVP is

> One grid, one simulation running in real time. Four materials to choose from below it, which you can add
> into the grid. Each rendered differently enough to be visually distinct. Controls for simulation speed,
> reset, deleting material, and view type. Interactive: add and remove material, and drag it around the way
> the elastic port already demonstrates.

**The four materials:** `elastic` (stiff rubber — done), `water`, `sand`, `snow`.

**Where it ships:** the **Demo** tab, and then Sevan's portfolio site. The transplant contract holds — no
dependency on the dashboard, the data server, or the harness. WebGPU support is wanted on the portfolio
site.

## What already exists (2026-08-16)

`runs/material-variants/interactive-simulation-of-one-material/` — the beachhead. Single-threaded JS port of
the canonical MLS-MPM **elastic** step, verified exact against `sim.physics`
(`traj_rmse` 1.72e-4 vs a 6.63e-5 self-noise and a 2.21e-4 rounding-perturbation band), interactive on
mouse and touch, parameters generated from `sim.physics` by `web/gen_params.py`. Portable source in `web/`.

## The three hard facts that shape everything

**1. Substeps, not particles, are the budget.** An explicit solver costs
`substeps/frame x cost-per-substep`, and `substeps = (1/60)/dt` with `dt` pinned by CFL. From canonical
`MAT`:

| material | E | dt | substeps/frame at real time |
| --- | --- | --- | --- |
| fluid (water) | 180 | 1.2e-4 | **139** |
| elastic | 400 | 1.0e-4 | **167** |
| snow | 150 | 5.0e-5 | **333** |
| sand | — | — | **not in canonical physics yet** |

**2. A shared grid means ONE timestep, so the stiffest material present sets the cost for everything.**
Put snow in the scene and the whole scene pays 333 substeps/frame — roughly double elastic. At the measured
JS cost of 88 us/substep at 1000 particles, a snow-bearing scene is ~29 ms/frame, i.e. **~34 fps at 1000
particles, or ~550 particles at 60 fps.** This is the single biggest threat to the MVP and it must be
designed for, not discovered late.

**3. Sand is not canonical physics.** `sim/physics/` freezes exactly fluid, elastic and snow. Adding sand
means **promoting new ground truth** through `sim/physics/PROMOTION.md`: it is ground truth, the golden
signatures pass, the version bumps — and sand needs its own signature (it should pile at an angle of repose
and not spread like a fluid). It also needs a constitutive model choice (Drucker-Prager style plasticity is
the usual answer for granular MPM).

**Also not yet supported:** the canonical `simulate` runs **one material at a time**. A single grid holding
four materials needs a per-particle material id and a step that branches on it. That is an API change to
canonical physics, not just a demo feature.

## Why the RTX 4090 lost to one JavaScript thread — and what it means for the GPU

Measured, not inferred:

- an **empty** CUDA kernel launch costs **56.4 us**; one real grid-op kernel costs **84.0 us**
- a full substep costs **~345 us and is FLAT from 500 to 16384 particles** (379 us at 16384 — only 10%
  more than at 500)

So essentially the entire cost is **telling the GPU to start work**, repeated for every kernel of every
substep, from Python. The 4090 is idle. This is a measurement of **an API usage pattern**, not of the
device, and it is **not** evidence that a GPU demo is a dead end. The opposite: it says the fix is to stop
paying a launch per substep.

**Implication for WebGPU:** record many dispatches into **one command buffer and submit once per frame**,
rather than round-tripping per substep. Per-dispatch cost inside a recorded buffer is small compared to a
Python→CUDA launch, so the 333-substep problem becomes tractable.

### The crux risk for WebGPU: no native float atomics
WGSL provides `atomic<u32>` / `atomic<i32>` but **not** atomic float add. P2G is a scatter that
atomically accumulates **mass and momentum as floats**. Options:
1. **Fixed-point atomics** — scale to integers and `atomicAdd` on u32/i32. The standard trick. It changes
   the numerics, so it must be re-verified against canonical exactly as the f32/f64 question just was.
2. **Sort/bin particles per cell** and reduce without atomics — no precision change, more passes and more
   complexity.
3. Per-cell locking — generally too slow.

**Availability caveat:** the elastic task found **WebGPU unavailable in both browsers it tested** (Chromium
148 in Electron, Edge 151), so no browser GPU path has been measured at all yet. Confirming availability —
in the dashboard's Electron pane, in iPad Safari, and in desktop Chrome — is step zero, not an afterthought.
Plan for a **dual path: WebGPU when present, the existing JS port as fallback.**

## Working targets (revise as measurements land)

- **Now (JS, elastic only):** ~1150 particles at 60 fps real time.
- **Snow in scene (JS):** ~550 particles at 60 fps — the number to beat.
- **WebGPU hope:** enough headroom that substep count stops being the binding constraint and particle count
  becomes interesting again. Treat any specific figure as a conjecture until measured.
- **Acceptable fallback if real time is unaffordable:** a smooth 60 fps display at a stated fraction of real
  time. The existing page already exposes `sim speed x real time` honestly; slow motion that is *labelled*
  is far better than a stutter or a lie.

## Open questions for Sevan

- Is **labelled slow motion** acceptable for the MVP, or must it be true real time?
- Is **sand** worth promoting into canonical physics, or should the fourth material be something already
  frozen?
