# Worker brief: T-028 — One latent-conditioned network for all four materials, on WebGPU

## THIS BRIEF WAS REJECTED ONCE AND REGENERATED. Sevan's note is the instruction:
> **"Actually I DO want this task demonstrated on WebGPU, that's the whole point: we are targeting
> deployable real time systems, that's what we're testing. Also, even though real-time is the constraint,
> exploring larger nets is still fine — maybe we'll find a way to make it more efficient."**

The previous version scoped this to Taichi only and treated width 16 as a hard ceiling. Both were wrong.
**Deployability is the thesis, not a follow-up**, and the width sweep is an exploration, not a cap.

## Effort tier: deep (overnight, 300 min)
**Persist.** Two halves, a real chance of a negative result, and a negative result cleanly established is
a good outcome. Run everything to completion inside your turn.

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
Post status **often** — on an overnight run it is the only signal anyone has.

## Objective
**Can ONE network with ONE shared weight set reproduce all four canonical materials — fluid, elastic,
snow, sand — with the material identified only by a latent code, AND run in real time on WebGPU?**

Two questions, and they are independent: **capacity** (can it learn them) and **cost** (can it ship).
Answer both. Either one being "no" is a real result worth reporting cleanly.

## Order of work — the cost half does NOT wait for the training half
This is the scheduling insight that de-risks the night, and it comes from T-022: **inference cost does
not depend on the weight VALUES.** So:

1. **Cost first, with untrained nets.** Port latent-conditioned per-particle MLP inference to WGSL and
   sweep width immediately. This is fast, decisive, and independent of whether training succeeds.
2. **Then train** the real network in Taichi against canonical ground truth.
3. **Then close the loop**: load the trained weights into the WGSL path, verify parity against the host
   implementation, and confirm the learned sim runs.
4. **If time remains**: a small interactive page in the run directory showing it running. **Not** on the
   real Demo page.

If training does not converge, you still owe a complete cost answer — and vice versa. Do not let one half
sink the other.

## The seam
Replace the **per-particle constitutive model**. Keep the MPM scaffolding analytic.

- **Input**: the position-free per-particle state — polar stretch `S` of the deformation gradient, the
  APIC affine `C`, velocity, the plastic record `Jp` — plus the **material code `z_m`**.
  (`one-nn-for-three-materials` validated exactly this 10-feature set; reuse it.)
- **Output**: the **stress** scattered to the grid, **and** the plastic state update (new `S`/`Jp`), so
  snow's clamp and sand's return mapping are learned rather than applied analytically afterwards.
- **Analytic and untouched**: B-spline P2G/G2P, the grid update, advection.

**Why this seam and not T-022's:** T-022 replaced the grid update and hit a structural accuracy wall —
gravity contributes `dt·g = 4.9e-4` per substep while the network's own error was `2.7e-2`, 56× larger.
Stress is O(E), i.e. hundreds. That failure mode does not transfer here, which is why this is worth
running. It also means the analytic thing being replaced is **expensive** (SVD, polar, Drucker-Prager),
where T-022 replaced a nearly-free kernel — so the cost *ratio* should look very different.

## Two latents, kept separate
| | role | scope | changes? |
| --- | --- | --- | --- |
| **`z_m`** material code | identity — "I am snow" | one per material, shared by its particles | **fixed** |
| **carried state** | history — "I have been compacted" | per particle | **updated every substep** |

- **`z_m`**: 4–8 dims, assigned **well-separated** (simplex corners), and **jittered during training** so
  the network learns a neighbourhood around each code rather than four point lookups.
- **Carried state**: use the **known parameterisation** (`S`, `Jp`) and predict its update. **A free
  learned latent state is out of scope** — discovering one needs backprop through a long rollout, this
  project's documented failure mode (`core/03-failure-modes`). It is the natural follow-up, not tonight.

**Do not claim latent interpolation.** With four structurally unrelated materials the latent space is a
*label*, not a physical axis; "halfway between snow and water" has no ground truth to check against.

## The width sweep — explore, do not cap
Sevan explicitly wants larger networks explored. T-022's numbers are the **map, not the fence**:

| width | whole solver, n=8192 | vs 60 fps at quarter GPU |
| --- | --- | --- |
| analytic | 9.02 µs/substep | fits |
| 8 | 10.06 | fits |
| 16 | 12.44 | right at the edge |
| 32 | 50.33 | 4× over |
| 64 | 58.76 | 4.7× over |

The 16→32 step costs **11.9× for 4× the arithmetic**, and T-022's own width scan saw throughput recover
above width ~48 — so the cliff looks like a *band*, plausibly register spilling, **not** a wall. That is
worth probing directly: sweep widths through and past the band and report the real curve.

**Levers T-022 explicitly never tested, and any of them could move the ceiling:**
- **f16** — could plausibly ~2× throughput.
- **Weights in uniform or workgroup storage** rather than a storage buffer.
- **One dispatch per substep** vs the current structure.
- **Batching several substeps** before returning to the host.

Try what is cheap. Report what each buys. **A width that misses real time is still a valid data point** —
label it, do not hide it, and do not silently ship a width-64 "success" as though it were deployable.

## Evaluation — the pass condition you already own
**Run the golden signatures against the LEARNED simulator.** `sim/physics/signatures.py` already encodes
what "behaves like this material" means: fluid spreads flat, sand slumps to repose, snow and elastic hold
a seeded slope, rubber holds its volume, density floats snow and sinks sand.

Report **which signatures the learned sim passes and which it fails, per material**. Three of four passing
is a real result; none passing is also a real result. That table is the headline.

Also report `traj_rmse` per material against canonical, read against the **self-noise band** (re-run
canonical twice; nudge ICs by ~1e-7), never against zero. Read its registry entry before quoting it — it
is a mean per-particle distance, not an RMS.

## Do NOT
- **Do not modify `sim/physics/`** — read-only ground truth at `phys-c518316a4a05`.
- **Do not inherit the known drift.** `sim/one_nn_materials.py` and `sim/learned_materials.py` run snow at
  `xi = 3.0` where canonical is `10.0`. **Import parameters from `sim.physics`** or the comparison is void.
- **Do not touch the real Demo page** or `harness/dashboard/src/components/mpm/`.
- A differentiable variant of the *step* is expected; it must import canonical *parameters* and
  *constitutive laws* and declare any deviation.

## Evidence discipline (non-negotiable)
- One setup is one setup. Scope claims to the scenes, materials and hardware actually tested.
- **Expect a capacity tradeoff and report it plainly.** One shared net across three *related* materials
  previously landed 1–2% off canonical against ~0.1% for separate nets. Four unrelated ones should be
  harder.
- Every timing is one GPU, one browser, one scene. Report raw **and** derated to a quarter GPU.
- Keep **observed / hypothesised / would-test** separate. `hypothesis` and `limitations` required.

## Visualization standard (graded)
- **Learned vs canonical, side by side, as video**, per material, same scene and seed. Ground truth
  mandatory.
- **The signature pass/fail table** as the headline.
- **A cost-vs-width curve** with the 60 fps line drawn on it, at full and quarter GPU, and the analytic
  solver marked as the baseline it replaces.
- **Open every figure and watch every clip before writing a finding.**

## Output contract
`runs/learned-dynamics/one-latent-conditioned-network-for-all-four-materials/manifest.json` (schema v2)
plus media: `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]`,
`training_refs[]`, `physics_version`.
- **Two layers: `summary` (shown) + `full_report` (expander).** The summary must answer both questions
  plainly: did one network hold four materials, and at what width does it still run in real time.
- **Write the manifest LAST**; every media `src` must resolve.

## Training textbook contribution (required)
One short standalone page (`spec/style_training_report.md`). Natural subject: **conditioning one network
on a material code** — why descriptor conditioning works where weight-blending was degenerate, what the
latent space is and is not when the materials are unrelated, and why identity must be separate from
carried state. `core/learned-materials` and `core/differentiable-materials` exist — **extend rather than
duplicate**. Every `[[link]]` must resolve.

## Definition of done
- One network, one weight set, four materials, identity carried only by `z_m`.
- **Running in WGSL**, with trained weights, parity-checked against the host implementation.
- **Golden signatures run against the learned simulator**, pass/fail per material.
- **Cost-vs-width curve measured on WebGPU** with `timestamp-query`, past the cliff band, with whatever
  efficiency levers you tried and what each bought.
- Finished within your turn; every figure viewed; every media `src` resolves.
- Manifest carries scoped findings, honest `hypothesis` and `limitations`.

## Known failures to avoid
- **The 8-storage-buffer ceiling.** Weight buffers make this likelier; exceeding it silently invalidates
  the bind group and drops every dispatch — a beautiful flat curve over all-zero data. Assert non-zero
  motion before believing any timing.
- **`performance.now()` is clamped to 100 µs in Chromium** — use `timestamp-query`. And T-022 found raw
  timestamps quantised to 32,768 ns; if your numbers are exact multiples of that, you are reading the
  quantum, not a cost. Price it a second way to confirm.
- **A cost that does not move with resolution or width** means you are timing the clock.
- **`navigator.gpu` needs a secure context** — serve the harness from `http://localhost`.
- Do not conflate the material code with the carried state. Do not train against drifted snow.
- Do not end your turn on a background training job.
