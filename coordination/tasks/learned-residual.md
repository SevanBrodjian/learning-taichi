# Worker brief: Learn a residual correction to the grid update

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `learned-residual`. You are **NOT the orchestrator**. Do not spawn
further agents. Read this brief, do the task, write **all** results to disk under
`runs/learned-dynamics/learned-residual/`, extend the training textbook, and exit. **Do not commit** — the
orchestrator reviews and commits your work. Fire the two pings below.

## Notifications (exactly two)
At the start:
```
python harness/tools/notify.py --kind started --task learned-residual "Starting the learned grid-velocity residual trained through the rollout."
```
When your results are on disk:
```
python harness/tools/notify.py --kind finished --task learned-residual "Done; the hybrid learned-residual experiment and a training page are on disk."
```
Use `--kind blocked` instead of `finished` if you hit a hard stop. **One sentence of human status, never a
metrics dump or a technical report.**

## Objective
Embed a **small learned network** inside the otherwise-explicit MLS-MPM step — a residual correction added
to the grid velocity after the grid op — and train **its weights** by backpropagating through the full
differentiable rollout. The question: **can gradients that have passed through hundreds of physics steps
actually train a network embedded in the simulator**, and on a concrete supervised target, does the hybrid
do something the unmodified simulator cannot? This is the minimal first hybrid (explicit physics + learned
component), the first concrete step toward learned dynamics.

## Background — what already exists
- `sim/diffmpm.py` is the differentiable rollout; the per-step grid update is `grid_op(f)` producing
  `grid_v_out[f]`. The natural injection point is **right after `grid_op`**:
  `grid_v_out[f] += residual_net(features)` before `g2p` reads it.
- The optimizer pattern (manual Adam over a parameter, finite-diff/grad checks) is in `diffmpm.py` and
  `sim/optimizer_compare.py`. Here the parameters are the **network weights**, not `v0`.
- Mass stabilization (`vel = grid_v_in / max(m, MASS_EPS)`) must stay **ON** (see
  `reports/training/core/03-failure-modes.md`) so the fixed overflow cannot confound training.

## What to implement
Write a new self-contained script `sim/learned_residual.py` (seed from `sim/diffmpm.py`). Keep the MLS-MPM
step intact and add a **small MLP residual on the grid velocity**, implemented as differentiable Taichi
fields/kernels so it lives on the **same autodiff tape** as the physics (do not bolt on a separate PyTorch
graph — the whole point is gradients flowing through the simulator into the weights):

- **Per-node input features**: keep it small and physically meaningful — e.g. the post-grid-op velocity
  `grid_v_out`, the grid mass `grid_m` (or a normalized/`log1p` version), and optionally normalized node
  coordinates. 3–5 input features.
- **Network**: one or two hidden layers (width ~16–32), `tanh` activation, 2 outputs (the Δv added to the
  node velocity). Weights/biases are `needs_grad=True` Taichi fields. **Scale the residual** by a small
  factor (e.g. `0.1`) or `tanh`-bound it so it starts as a gentle correction and cannot blow the sim up.
- Apply per **active** node (`grid_m > MASS_EPS`); leave empty nodes untouched.
- **Initialize weights small** (so the hybrid starts ≈ the pure simulator) with a fixed seed.

**Gradient-flow probe FIRST (mandatory, anti-degeneracy).** Before training, confirm the loss gradient
w.r.t. the **network weights** is finite and non-zero, and that a few optimizer steps strictly decrease the
loss and the weights move off their init. A finite-difference check on one or two weights vs the autodiff
grad is the gold standard — do it on a short horizon. If the grad is zero/NaN, fix it (feature scaling,
residual scaling, horizon) before training. **Never trust a flat-loss run.**

## Experiments / deliverables — pick a target the residual can plausibly help with
The cleanest framing is a **system-identification / correction** task where the residual has a real job:
1. **Define a target trajectory the bare simulator misses.** Recommended (cheap, honest): generate target
   center-of-mass (or full particle) trajectories from the **same simulator run with a perturbed physical
   constant** the residual is *not* given — e.g. a different gravity, or an added linear drag
   `v *= (1−k)` — so a "true" dynamics exists that differs from the model's. The residual must learn to
   **close that model-mismatch gap** purely from the supervised trajectory, with `v0` held fixed. This
   makes the win unambiguous and avoids over-parameterized cheating.
   - Hold-out honesty: if feasible, fit on one initial condition / `v0` and **evaluate on a held-out
     `v0` or target** to show the residual learned a transferable correction, not just memorized one
     trajectory. If a held-out test is not feasible in the time budget, say so explicitly in limitations.
2. **Baselines to make the result meaningful** (this is what turns "it trained" into evidence):
   - **bare simulator** (no residual) loss against the target — the gap to close;
   - **trained hybrid** loss — how much of the gap the learned residual closed;
   - report both, and the fraction of the gap closed.
3. Log the training loss curve, the weight-gradient norm over training, and whether anything went
   non-finite.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- This is **one network architecture, one mismatch task, one (or few) condition(s)**. The claim is
  bounded: "on this task, gradients through ~N physics steps trained a small grid-velocity residual that
  closed X% of the model-mismatch gap" — **not** "learned residuals work" or "hybrid sims are better".
- Separate observation (the loss curves, the gap closed), hypothesis (why it trained — the residual sits
  near the end of the per-step Jacobian product so its gradient is less attenuated than `v0`'s; what
  limits it), and what-would-test-it (more architectures, more mismatch types, held-out generalization,
  longer horizons).
- The manifest MUST carry honest `hypothesis` and `limitations`. If you could not run a held-out test,
  the `limitations` must say the result may be partly memorization.

## Visualization standard (graded, not optional)
- **A video comparing three rollouts**: target (ground truth), bare simulator, trained hybrid — each
  overlaying the **center-of-mass trajectory** so the viewer can *see the bare sim miss and the hybrid
  recover*. Reuse the headless matplotlib/Agg renderer from `diffmpm.py` (no `ti.GUI`).
- **A loss-curve plot** (training loss vs iteration, log-y, labeled) and a **weight grad-norm plot**.
- A **table**: bare-sim loss, hybrid loss, gap-closed %, network size (params), horizon, mismatch used,
  and (if run) held-out loss.
- Labeled axes, readable fonts (renders small on iPad).

## Training textbook contribution (required)
Add **one short, standalone page** under `reports/training/` (a new `core/` page) in the objective
textbook voice (`spec/style_training_report.md`): impersonal, standalone, no reference to "this run".
Teach the **hybrid idea**: where a learned component can sit inside an explicit differentiable step, why
the same autodiff tape that carries `v0`'s gradient can also carry the network weights' gradients, and the
intuition for **why a residual near the end of the rollout is easier to train than a parameter at the
start** (Jacobian-product attenuation — link `[[differentiating-the-rollout]]`,
`[[failure-modes]]`, `[[mls-mpm-forward]]`). Embed the comparison figure. Register the page in
`reports/training/index.json`.

## Output contract
Write `runs/learned-dynamics/learned-residual/manifest.json` (**schema_version "2"** — see
`runs/README.md`) plus media: `task_id` `learned-residual`, `direction` `learned-dynamics`, `objective`,
scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (the 3-way comparison video, the loss +
grad plots, the table), and `training_refs[]` including the new page id. Leave everything on disk; **do
not commit**.

## Paths & params
- Run dir: `runs/learned-dynamics/learned-residual/`
- Code: `sim/learned_residual.py` (new)
- Start from: `n_grid=64`, `n_particles=4096`, `E=400`, `dt=2e-4`, horizon ~256–400 steps (shorter is
  fine and keeps the tape memory bounded), MLP width ~16–32, residual scale ~0.1, Adam on weights, fixed
  seed. Record what you actually used.

## Definition of done
- The weight-gradient is verified finite/non-zero (probe + a finite-difference spot check) before
  training; no flat-loss run is reported.
- The result is **framed against the bare-simulator baseline** (gap-to-close and fraction closed), not as
  a bare "loss went down".
- Manifest schema-v2 with honest `hypothesis` + `limitations` (including the memorization caveat if no
  held-out test); 3-way comparison video overlays COM trajectories; loss/grad plots and table present;
  math render-checks in KaTeX.
- One standalone training page added and registered in `index.json`.

## Known failures to avoid
- **Implement the network on the Taichi autodiff tape**, not as a detached PyTorch model — the experiment
  is meaningless if the weight gradients do not actually flow through the physics.
- **Verify the weight gradient before training** (probe + FD check). A zero gradient usually means the
  residual is detached, mis-scaled to nothing, or the horizon is so long the gradient vanished — fix the
  cause, don't report a flat run.
- Keep the residual **small/bounded** at init so the hybrid starts near the pure simulator and training is
  stable; an unbounded residual will blow the sim to NaN.
- Make the target a **genuine model mismatch** the residual is not handed directly (perturbed gravity /
  added drag), so a falling loss is real learning, not the network trivially reproducing a constant it was
  given. Beware over-parameterized memorization — note it in limitations if you cannot test held-out.
- Do **not** spawn a long background command then end the turn waiting on it; run to completion in the
  foreground before the `finished` ping. Keep mass stabilization ON.
