# Worker brief: One latent-conditioned network for all four materials

## Effort tier: deep (overnight, 240 min)
**Persist.** This is a training task with a real chance of a negative result, and a negative result that
is cleanly established is a good outcome. Run everything to completion inside your turn.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `one-latent-conditioned-network-for-all-four-materials`. You are
**NOT the orchestrator**. Do not spawn further agents. Read this brief, do the task, write **all** results
to `runs/learned-dynamics/one-latent-conditioned-network-for-all-four-materials/`, add a training page,
and exit. **Do not commit.**

## Notifications + live status
```
python harness/tools/notify.py --kind started  --task one-latent-conditioned-network-for-all-four-materials "<one plain sentence>"
python harness/tools/notify.py --kind finished --task one-latent-conditioned-network-for-all-four-materials "<one plain sentence>"
python harness/tools/task_status.py --direction learned-dynamics --task one-latent-conditioned-network-for-all-four-materials --step "<a few words>"
```
Post status **often** on a long run — it is the only signal anyone has overnight.

## Objective
**Can ONE network with ONE shared weight set reproduce all four canonical materials — fluid, elastic,
snow, sand — with the material identified only by a compact latent code fed in as input?**

Taichi only. No WebGPU. The question is capacity and fidelity, not deployment.

## The seam — read this carefully, it is not the seam T-022 used
Replace the **per-particle constitutive model**. Keep the MPM scaffolding analytic.

- **Network input**: the position-free per-particle state — the polar stretch `S` of the deformation
  gradient, the APIC affine `C`, velocity, the plastic record `Jp` — plus the **material code `z_m`**.
  (`one-nn-for-three-materials` validated exactly this 10-feature set; reuse it rather than inventing one.)
- **Network output**: the **stress** the particle scatters to the grid, **and its plastic state update**
  (the new `S`/`Jp`), so snow's clamp and sand's return mapping are learned rather than applied
  analytically afterwards.
- **Analytic and untouched**: B-spline P2G/G2P, the grid update (mass-normalise, gravity, walls,
  friction), advection. That scaffolding is the frozen ground truth and keeps momentum conservation free.

**Why not the grid update:** T-022 replaced it and found the accuracy failure was structural — gravity
contributes `dt·g = 4.9e-4` per substep while the network's own error was `2.7e-2`, 56× larger. Stress is
O(E), i.e. hundreds. That failure mode does not transfer to this seam, which is the reason this task is
worth running.

## Two latents, and do not conflate them
This is the design mistake to avoid:

| | role | scope | changes over time? |
| --- | --- | --- | --- |
| **`z_m`** material code | identity — "I am snow" | one per material class, shared by all its particles | **fixed** |
| **carried state** | history — "I have been compacted" | per particle | **updated every substep** |

- **`z_m`**: 4–8 dims. **Assign them well-separated** (e.g. simplex corners), and **jitter them during
  training** so the network learns a neighbourhood around each code rather than four point lookups —
  that is what makes the mapping a function instead of a table.
- **Carried state**: use the **known parameterisation** (`S`, `Jp`) and have the network predict its
  update. **Do not attempt a free learned latent state in this task.** Discovering one requires
  backprop through a long rollout, which is this project's documented failure mode
  (`reports/training/core/03-failure-modes.md`). A free latent is the natural follow-up, not this run.

**Honest framing for the write-up:** with four structurally unrelated materials, the latent space is a
*labelling* device, not a physical axis. "Halfway between snow and water" has no ground truth to check
against. Jitter buys robustness, **not** meaningful interpolation — do not claim interpolation you cannot
verify.

## Training data
Ground truth from **canonical `sim.physics`** (forward, cheap, stable — it needs no gradients).

- **Each material alone**, across the canonical scenes (drop, column, heap) — this is where per-material
  fidelity is learned.
- **Mixed scenes where materials interact**, including the new `scene_pool` buoyancy setup. These matter
  because a particle's neighbourhood then contains *other* materials, and the grid quantities it gathers
  are blends. A network trained only on pure scenes may fail exactly at interfaces.
- Report the split and the scene list. If you use noise injection on inputs to make the network robust to
  its own drift, say so — it is a standard fix for rollout divergence and worth doing.

## The evaluation that matters
**Run the golden signatures against the LEARNED simulator.** `sim/physics/signatures.py` already encodes
what "behaves like this material" means: fluid spreads flat, sand slumps to a repose angle, snow and
elastic hold a seeded slope, rubber holds its volume, the density ordering floats snow and sinks sand.

That is a far better test than an RMSE, and it is already written. Report **which signatures the learned
sim passes and which it fails** — a network that passes three of four materials is a real, publishable
result, and so is one that passes none.

Also report `traj_rmse` against canonical per material, **read against the self-noise band** (re-run
canonical twice; nudge ICs by ~1e-7), not against zero. Read `spec/registry/metrics.json` before quoting
it — it is a mean per-particle distance, not an RMS.

## Size constraint — keep it deployable
T-022 measured a hard cliff: at 8,192 elements a width-16 per-element MLP costs 12.44 µs/substep against
the analytic solver's 9.02, and **width 32 costs 50.33 — 11.9× more for 4× the arithmetic.** Derated to a
quarter of this GPU, width 16 sits right at the 60 fps edge and width 32 is dead.

So: **target width ≤ 16, and report parameter count and width for whatever you train.** If a bigger
network is needed to reach fidelity, that is a legitimate finding — say so explicitly and state what
width it took, because that is the number that decides whether this can ever ship. Do not silently train
a width-64 net and report success.

## Do NOT
- **Do not modify `sim/physics/`.** It is read-only ground truth at `phys-c518316a4a05`.
- **Do not inherit the known drift.** `sim/one_nn_materials.py` and `sim/learned_materials.py` both run
  snow at `xi = 3.0` where canonical is `10.0`. **Import parameters from `sim.physics`** — training
  against a fourth different snow would invalidate the comparison.
- **Do not touch the Demo page.**
- A differentiable variant of the *step* is fine and expected; it must import the canonical *parameters*
  and *constitutive laws* and state any deviation in the findings (CLAUDE.md → Canonical physics).

## Evidence discipline (non-negotiable)
- **One setup is one setup.** Scope every claim to the scenes and materials actually trained and tested.
- Expect a **capacity tradeoff** and report it plainly: one shared net across three materials previously
  landed 1–2% off canonical against ~0.1% for separate per-material nets. Four structurally different
  materials should be harder. That tradeoff is the finding, not a failure.
- Keep **observed / hypothesised / would-test** separate. `hypothesis` and `limitations` required.
- **A negative result is a good outcome if it is cleanly established.** If a width-16 net cannot hold four
  materials, say so with the evidence and state what width it would take.

## Visualization standard (graded)
- **Learned vs canonical, side by side, as video**, per material, on the same scene and seed. Ground truth
  is mandatory.
- The **signature pass/fail table** as the headline — it is the clearest statement of what was achieved.
- A fidelity-vs-width plot if you sweep width, with the width-16 deployability line drawn on it.
- **Open every figure and watch every clip before writing a finding.**

## Output contract
`runs/learned-dynamics/one-latent-conditioned-network-for-all-four-materials/manifest.json` (schema v2)
plus media: `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]`,
`training_refs[]`, `physics_version`.
- **Two layers: `summary` (shown) + `full_report` (expander).** The summary must state plainly whether one
  network held four materials, and at what size.
- **Write the manifest LAST**; every media `src` must resolve.

## Training textbook contribution (required)
One short standalone page (`spec/style_training_report.md`). The natural subject is **conditioning a
single network on a material code** — why descriptor conditioning works where weight-blending was
degenerate, what the latent space is and is not (a label, not a physical axis, when the materials are
unrelated), and why the carried state has to be separate from the identity code.
`core/learned-materials` and `core/differentiable-materials` exist — **extend rather than duplicate**.
Every `[[link]]` must resolve.

## Definition of done
- One network, one weight set, four materials, identity carried only by `z_m`.
- **Golden signatures run against the learned simulator**, with a pass/fail table per material.
- `traj_rmse` per material against canonical, read against the self-noise band.
- Width and parameter count reported, against T-022's width-16 deployability ceiling.
- Finished within your turn; every figure viewed; every media `src` resolves.
- Manifest carries scoped findings, honest `hypothesis` and `limitations`.
- Training page renders (KaTeX), reads standalone, every `[[link]]` resolves.

## Known failures to avoid
- **Do not conflate the material code with the carried state.** One is fixed identity, the other is
  per-particle history. Both are needed.
- **Do not train against drifted snow.** Import `xi` from `sim.physics`.
- **Do not claim latent interpolation** between unrelated materials — there is no ground truth for it.
- **Do not quote `traj_rmse` without reading its registry entry.**
- Do not end your turn on a background training job — run it to completion.
