# Task brief — Long-rollout gradient pathologies

> Orchestrator → worker contract. Read `CLAUDE.md` (esp. the Autonomy charter) and all of `spec/`
> first. Branch: `claude/long-rollout-grads`. Direction id: `long-rollout-pathologies`.

## Context (what is already established)
The DiffMPM baseline (`sim/diffmpm.py`) backpropagates through a 512-step MLS-MPM rollout to optimize a
shared initial velocity. It converges (loss $0.144 \to 5\times10^{-5}$) and then, reproducibly, the
**backward pass returns a NaN around iter 67 while the forward loss stays finite** (~$4.6\times10^{-4}$).
A guard catches it and stops. Three leading hypotheses are written up in
`reports/training/core/03-failure-modes.md`:
1. out-of-domain particles hitting the non-smooth wall clamps (contact),
2. division by near-zero grid mass (backward $\propto 1/m_i^2$),
3. long-rollout amplification (a long product of per-step Jacobians).

This task owns hypothesis 3 and the general question of how gradient health scales with rollout length.

## Objective
Explain the NaN mechanistically and demonstrate at least one mitigation that lets optimization continue
cleanly past where it currently dies, while characterizing how gradient quality degrades with horizon.

## Experiments (start here, adapt as you learn)
1. **Instrument the backward pass.** Log per-step gradient norms (and max abs) flowing back through the
   rollout, plus per-step grid-mass minima. Localize the first step where values explode or go non-finite.
   This is the single highest-value experiment; do it first.
2. **Horizon sweep.** Vary `max_steps` (e.g. 128 / 256 / 512 / 1024) at a fixed task. Plot final-iter
   gradient norm and the iteration at which NaN appears (if any) vs. horizon.
3. **Precision.** Try `ti.init(default_fp=ti.f64)`. Does the NaN vanish, or just move later? That
   distinguishes a true singularity from a near-overflow.
4. **Gradient hygiene.** Add gradient clipping and a non-finite-skip, confirm the optimization is
   otherwise healthy (loss keeps dropping past iter 67).
5. **Checkpointing.** Sketch or implement gradient checkpointing so longer horizons fit in memory;
   report the compute/memory trade-off you actually measure.
6. **Isolate contact.** Re-run with a target/horizon that keeps particles off the walls vs. one that
   drives them into walls, to separate hypothesis 3 from hypothesis 1.

Keep the baseline `sim/diffmpm.py` working. Prefer flags or a sibling module over destructive edits.

## Deliverables (the dashboard contract)
- Code under `sim/` (flags or a variant module). Use the main `.venv`.
- One `runs/claude/long-rollout-grads/<run-id>/` per notable config: `manifest.json` (schema_version "1",
  with `training_refs`), `metrics.json`, and `video.mp4` for at least the headline result.
- **Teach it.** Substantially expand `reports/training/core/03-failure-modes.md` with the mechanism you
  find and the fix that worked. Add prerequisite sections as needed and err toward more (e.g. exploding/
  vanishing gradients in unrolled/recurrent systems, the adjoint method, matrix conditioning). Derive the
  math, link with `[[wiki-links]]`. KaTeX rule: multiline `$$` must be three-line (open and close on their
  own lines).
- Update `coordination/directions.json` status as you go; keep `agents/claude/long-rollout-grads/STATUS.md`
  current and append to `LOG.md`.
- ntfy: `progress` pings freely; `gate` only at a real decision, written to `coordination/decisions/`,
  and never block on it (see the Autonomy charter).

## Definition of done
- A clear, evidence-backed statement of which hypothesis(es) drive the NaN, with the instrumentation plot
  that shows it.
- At least one mitigation demonstrated end-to-end (a loss curve that continues cleanly past the old
  failure point, or a documented principled reason it cannot).
- `03-failure-modes.md` expanded enough that the user can explain the pathology and its fix at a
  whiteboard.
- Results visible on the master dashboard (runs + transcluded textbook sections).
