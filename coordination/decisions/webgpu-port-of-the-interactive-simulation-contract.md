<!-- auto_run_at: 1786930880 -->
# Contract — WebGPU port of the interactive simulation (deep)

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/webgpu-port-of-the-interactive-simulation.md`
**Runs SECOND**, after the sand task finishes — so its GPU timings are uncontended and it generates its
constants against the post-bump physics.

## The one idea
Stop paying a kernel launch per substep. The canonical Taichi/CUDA path costs **~345 µs/substep and is flat
from 500 to 16,384 particles**, because an empty launch alone costs **56.4 µs** and it pays one per kernel
per substep from Python — the 4090 is idle. WebGPU fixes exactly that by recording many dispatches into
**one command buffer submitted once per frame**.

## The crux, already pinned down
WGSL has **no atomic float add** — confirmed by compilation: `atomic<f32>` fails with *"'atomic' only
supports 'i32', 'u32' or 'vec2u'"*. P2G scatters mass and momentum as floats.
- Start with **fixed-point integer atomics**: mass → `atomic<u32>`, momentum → `atomic<i32>` (signed —
  momentum goes negative).
- **Bounded effort**: find a configuration good enough to prove the route, then stop. The deep precision
  study is a separate planned follow-up.
- A **CAS-loop float add** also compiles and is exact in f32 but slower under contention — a cheap
  head-to-head, reported either way.

## What it must prove
Fixed-point quantisation **changes the numerics**, so verification is the point, not a formality:
`traj_rmse` against canonical `sim.physics`, read against the simulator's **own self-noise band**, shown as
**motion, both sides together**. The JS port's marks to beat or match: **1.72e-4** against a **6.63e-5**
self-noise and a **2.21e-4** one-ULP perturbation band.

Then a **three-way comparison on the same machine and physics** — JS / Taichi-CUDA / WebGPU — with the
particle budget at 60 fps for each, timed with **`timestamp-query`** rather than `performance.now()` (which
is clamped to 100 µs in Chromium and already produced one wrong phase split).

## Settled, so it does not get re-litigated
WebGPU is **confirmed present** on the Windows desktop (nvidia/lovelace), the iPad, and the MacBook Air M4.
The earlier "unavailable" finding was wrong: `navigator.gpu` only exists in a **secure context**, so plain
HTTP over the LAN hid it. Consequence: **a JS fallback is no longer clearly required**, and the brief tells
the worker not to spend effort on fallback parity.

## What it will NOT do
- **Not** the atomics optimisation study — viability only.
- **Not** multi-material. Elastic alone; sand/water/snow are the other task.
- **Not** 3D, **not** differentiable, **not** a pretty renderer.
- **Not** wired into the Demo tab.
- **Not** JS-fallback parity work.
