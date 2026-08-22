<!-- auto_run_at: 1787381518 -->
# Contract — One latent-conditioned network for all four materials (deep, overnight)

**Approve to run, or Reject with a note.** Auto-runs at the deadline so it uses the night either way.
Full brief: `coordination/tasks/one-latent-conditioned-network-for-all-four-materials.md`

## Why this one, tonight
T-022 already bounded the **cost** question. What it cannot answer is **capacity**, and that is now the
real risk in your idea — so this is the highest-value thing to spend an unattended GPU on.

## The seam is NOT T-022's, and that matters
This replaces the **per-particle constitutive model** (stress + plastic state), keeping P2G/G2P and the
grid update analytic. T-022's structural accuracy failure was that gravity contributes `dt·g = 4.9e-4`
per substep while the net's own error was `2.7e-2`. Stress is O(E) — hundreds. That failure mode does
not transfer, which is exactly why this is worth running.

## Two latents, deliberately kept separate
`z_m` is **identity** — one fixed code per material, well-separated, jittered during training so the
network learns a neighbourhood rather than four point lookups. The **carried state** is **history**, per
particle, updated every substep. This task uses the known parameterisation (`S`, `Jp`) for the state; a
free learned latent needs backprop through a long rollout, which is this project's documented failure
mode, and is the natural follow-up rather than this run.

## The size ceiling is a hard constraint, not a preference
At 8,192 elements a width-16 per-element MLP costs 12.44 µs/substep against the analytic solver's 9.02 —
and **width 32 costs 50.33, i.e. 11.9× more for 4× the arithmetic.** Derated to a quarter GPU, width 16
sits right at the 60 fps edge and width 32 is dead. The brief targets **width ≤ 16** and requires the
width and parameter count to be reported, so a "success" at width 64 cannot be passed off as shippable.

## What it will NOT do
- **No WebGPU.** Taichi only, as you said.
- **No physics changes** — `sim/physics/` read-only, and parameters imported rather than copied, so it
  cannot train against the drifted `xi = 3.0` snow.
- **No claims of latent interpolation.** With four structurally unrelated materials the latent space is a
  *label*, not a physical axis; "halfway between snow and water" has no ground truth to check against.
- **No free learned state.** Deliberately out of scope tonight.
- **No spin on a negative.** If width 16 cannot hold four materials, that gets said with the evidence and
  the width it would actually take.

## The pass condition is the one you already own
The **golden signatures, run against the learned simulator** — fluid spreads, sand slumps to repose, snow
and elastic hold a slope, snow floats and sand sinks. A pass/fail table per material is the headline. A
network passing three of four is a real result; so is one passing none.


**Resolution: REJECTED** — Actually I DO want this task demonstrated on WebGPU, that's the whole point: we are targeting deployable real time systems, that's what we're testing. Also, even though real-time is the constraint, exploring larger nets is still fine -- maybe we'll fine a way to make it fmore efficient.
