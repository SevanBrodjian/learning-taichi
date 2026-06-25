# STATUS — long-rollout-pathologies / softened-wall

**Worker task `softened-wall` — COMPLETE, on disk for review (NOT committed).**

## What was done
Replaced the hard wall clamp (`if vel<0: vel=0`, a non-smooth kink) with a smooth `smoothstep` ramp
that gates the inward-normal velocity to zero across a boundary band of width `ramp_cells`. Compared
hard vs two soft widths (r=3, r=6) on a task that drives the blob into the LEFT wall
(target [0.06, 0.5], 400 steps, fixed seed, Adam lr 0.15). **Mass stabilisation ON in every
condition** so the already-fixed near-zero-mass overflow cannot confound the comparison.

## Headline (scoped to this one contact scenario)
- Soft walls improve autodiff-vs-finite-difference gradient agreement, reproducibly: hard
  mean rel-err **3.42e-2** vs soft **~2.2–2.3e-2** (mean of 3 repeats; the gap between the two soft
  widths is within GPU run-to-run noise).
- Final loss (stable across runs): hard **3.37e-3**, soft r=3 **1.90e-3** (best — tight band),
  soft r=6 **4.72e-3** (worst — wide band damps particles before true contact → physics distortion,
  ended with 0% particles in the band).
- Iter-to-iter gradient-direction cosine indistinguishable (~0.994–0.997): the benefit is in gradient
  ACCURACY, not path stability, on this task.
- Framing held: contact was NOT the NaN cause (mass overflow was); this is strictly "does smoothing
  contact help gradient quality", and the answer here is "a little, with a narrow band".

## Files written (uncommitted)
- `sim/softened_wall.py` (new, unique sibling script)
- `runs/long-rollout-pathologies/softened-wall/{manifest.json, metrics.json, loss_compare.png,
  gradnorm_compare.png, video_hard.mp4, video_soft_r3.mp4, video_soft_r6.mp4}`
- `agents/long-rollout-pathologies/{STATUS.md, LOG.md}`

## Contamination note
Another worker shared the GPU (~2 GB / ~30% util baseline observed). The FD-sweep metric is not
bitwise reproducible (p2g atomic-add non-determinism), so the hard-vs-soft claim rests on 3 repeats,
not one. Final loss was stable to ~1%.
