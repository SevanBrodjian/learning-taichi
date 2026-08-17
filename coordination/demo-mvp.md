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
| sand | 300 | 1.0e-4 | **167** (canonical since `phys-bebeaafbe73e`) |

**2. A shared grid means ONE timestep, so the stiffest material present sets the cost for everything.**
Put snow in the scene and the whole scene pays 333 substeps/frame — roughly double elastic. At the measured
JS cost of 88 us/substep at 1000 particles, a snow-bearing scene is ~29 ms/frame, i.e. **~34 fps at 1000
particles, or ~550 particles at 60 fps.** This is the single biggest threat to the MVP and it must be
designed for, not discovered late.

**Sand does not make this worse.** Measured 2026-08-16: sand runs at dt=1e-4, i.e. 167 substeps/frame,
exactly what elastic costs. Adding sand to a scene that already contains snow costs **nothing**. Snow
remains the material to design around. (Sand alone takes a water-only scene from 139 to 167.)

**⚠ CORRECTION — why snow needs a small dt is UNKNOWN, and the hardening story was wrong.** This document
previously asserted that snow's dt=5e-5 comes from hardening making compacted snow ~3x stiffer than
elastic. The sand task measured it. The hardening is real and larger than that guess (compacted snow
reaches an effective stiffness of ~1858 at the 95th percentile against elastic's 400, and it is bimodal —
~44% of particles end up *softer* than nominal). **But snow's measured stability wall is 8x its canonical
timestep, and setting xi=0 does not move it.** Snow's dt was never set by stability. Do not repeat the
hardening explanation; the real reason is an open question.

**⚠ A plastic material's "strength" is not a converged quantity.** The settled slope of every plastic
material (snow, sand, and the fluid to a small degree) decays with **substep count, not physical time** —
canonical snow holds a 56 degree heap at its own dt and collapses to 19 degrees at dt/4, while at *equal
substep count* every timestep agrees to within 1.6 degrees. Elastic, which has no plastic projection, is
flat on both axes. So the fine-timestep run is the CORRUPTED one, not the converged one: a one-sided
return mapping appears to rectify transfer noise into permanent plastic strain, once per substep.

**Consequence for the MVP, and it is a correctness issue rather than a cost one:** a shared grid means a
shared dt, so putting snow in a scene forces sand to run at half its canonical timestep — twice the
substeps, therefore *more creep than canonical sand exhibits*. **Material behaviour would depend on what
else is in the scene.** Any four-material demo has to either fix the substep count per material, quote
behaviour at a stated dt and duration, or accept that mixed scenes are not quantitatively canonical.

**3. Sand is canonical physics as of 2026-08-16** (`phys-bebeaafbe73e`). Drucker-Prager elastoplasticity
(Klar et al. 2016) on a Hencky log-strain elastic law, `E=300, dt=1e-4, phi=50`. The structural reason it
differs from snow: sand is *cohesionless*, so its shear strength is proportional to confining pressure — a
**cone** in stress space — where snow's Stomakhin clamp is a fixed **box** (cohesion). That is why snow can
stand a vertical wall and sand cannot. Four new golden signatures; all pre-existing ones still green.

**Its measured angle of repose is ~26 degrees, and phi is NOT the repose angle** (phi=50 measures ~25).
That number is a signature that holds, but it is *not converged* — see the plastic-creep warning above.

**Multi-material is DONE.** Canonical physics now carries a per-particle material id and a runtime branch,
so one grid holds all four. A single material pushed through that path lands where canonical `simulate`
lands — at or below the effect of nudging its initial positions by one float32 rounding unit — and that
equivalence is now itself a golden signature for all four materials, not a one-off check.

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

**Availability: RESOLVED 2026-08-16 — WebGPU is present on every device that matters.** Confirmed on the
Windows desktop (nvidia/lovelace), the iPad, and the MacBook Air M4. The elastic task's claim that WebGPU
was unavailable was **wrong**, and so was the first capability probe: `navigator.gpu` is only exposed in a
**secure context**, so reaching the dashboard over plain HTTP from another device hid the API entirely and
read as "this device has no WebGPU". Fixed by fronting the dev server with a real certificate
(`tailscale serve --bg --https=443 http://localhost:5174` → `https://sevan-windows-home.tail9a3a96.ts.net`),
and the Demo tab now distinguishes *hidden* / *unsupported* / *supported* instead of collapsing all three
into "ABSENT".

**Consequence for the plan:** a JS fallback is **no longer clearly required**. Sevan is inclined to ship
WebGPU-only, or to keep the JS path behind a flag. Treat WebGPU as the primary target rather than an
experiment, and do not spend effort on fallback parity unless a device turns up that needs it. The portfolio
site will be HTTPS, so the production case was never in doubt.

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
