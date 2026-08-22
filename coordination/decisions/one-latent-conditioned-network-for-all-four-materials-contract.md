<!-- auto_run_at: 1787382125 -->
# Contract — T-028 v2: one latent-conditioned network for four materials, ON WEBGPU

**Approve to run, or Reject with a note.** Auto-runs at the deadline so the night is used either way.
Full brief: `coordination/tasks/one-latent-conditioned-network-for-all-four-materials.md`

## What changed after your reject
> "Actually I DO want this task demonstrated on WebGPU, that's the whole point... exploring larger nets
> is still fine."

You were right and I mis-scoped it. Testing deployability in Taichi tests the wrong thing. Two changes:
- **WebGPU is now half the task**, not a follow-up.
- **The width sweep explores past the cliff** instead of treating 16 as a cap.

## The scheduling trick that de-risks the night
**Inference cost does not depend on the weight VALUES** — T-022's own finding. So the cost half runs
FIRST, with untrained nets, and does not wait for training to converge:

1. WGSL latent-conditioned inference + width sweep (fast, decisive, independent)
2. Train the real network in Taichi
3. Load trained weights into WGSL, parity-check, run the learned sim
4. If time: a small interactive page **in the run directory**, not the real Demo page

If training stalls you still get a complete cost answer, and vice versa. Neither half can sink the other.

## On width: the map, not the fence
T-022 measured 16 → 32 costing **11.9× for 4× the arithmetic**, but its own scan saw throughput recover
above ~48 — so the cliff looks like a *band*, plausibly register spilling, not a wall. Worth probing
directly. Levers it never tested and that could move the ceiling: **f16** (~2×?), weights in uniform or
workgroup storage, one dispatch per substep, batching substeps. Try what is cheap, report what each buys.

## The pass condition is the one you already own
The **golden signatures run against the LEARNED simulator** — fluid spreads, sand slumps to repose, snow
and elastic hold a slope, snow floats and sand sinks. A pass/fail table per material is the headline.
Three of four is a real result; none is also a real result.

## What it will NOT do
- **No free learned latent state.** That needs backprop through a long rollout — this project's documented
  failure mode. Natural follow-up, not tonight.
- **No claims of latent interpolation.** With four unrelated materials the code is a *label*, not an axis.
- **No physics changes**, and parameters imported rather than copied, so it cannot train against the
  drifted `xi = 3.0` snow.
- **It will not touch the real Demo page.**
- **No spin on a negative.** If the net cannot hold four materials, or cannot hit real time at any useful
  width, that gets said with the numbers.


**Resolution: APPROVED**
