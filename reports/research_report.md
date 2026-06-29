# Differentiable simulation toward structured generative worlds

> **Scratchpad, not the paper yet.** Right now this file is a short, living note of the research
> directions worth pulling on and the threads that are turning into real results. It is kept to one page,
> added to conservatively as evidence lands, and pruned freely. The eventual end target is a conservative,
> shippable technical report (see `spec/style_research_report.md`); it graduates toward that form only at
> the user's direction, at deliberate moments, never gradually on its own. Depth lives in the training
> textbook (`reports/training/`); this is the index of what is worth saying about the work as a whole.

## The throughline
The question underneath everything here is whether explicit, differentiable physics is a usable substrate
for *authorable* dynamics, the structure half of structured generative worlds. A controllable world model
needs dynamics whose sensitivities can be trusted and steered, and MLS-MPM made differentiable is a small,
honest testbed for finding out where that trust holds and where it breaks.

## Threads with real evidence
- **Control by backprop works on the clean case.** Throwing a blob to a target by optimizing an initial
  velocity through a 500-step differentiable rollout converges cleanly. The gradient through the physics
  is real and usable, which is the precondition for everything else.
- **The optimizer story is landscape-dependent, not universal.** Across several control tasks, L-BFGS wins
  only when the loss landscape is smooth; on contact-rich or higher-dimensional controls a first-order
  method (Adam) wins, and the tidy "all optimizers reach the same basin" picture breaks. This is the kind
  of scoped, falsifiable result the project is built to produce.
- **Long-rollout gradients fail by overflow, not by singularity, and the fix is cheap.** Differentiating
  through a long rollout produces NaN gradients from a near-zero grid-mass division amplified by the
  product of per-step Jacobians. It is a float32 overflow (float64 removes it), and a mass floor on the
  grid division removes it surgically. The gradient health of long rollouts is plausibly *the* central
  difficulty of the whole approach.

## Directions worth pulling next
- **Material variants** (elastic, fluid, snow) under one control task: the most visually striking axis and
  a direct test of how constitutive choice changes gradient behavior and controllability.
- **Learned dynamics**: a small learned residual on the grid update, trained through the differentiable
  rollout. The first concrete step from a hand-written simulator toward a hybrid learned-and-structured
  world, which is where the broader vision points.
- **Open gradient questions**: whether smooth contact measurably improves gradient quality, and whether
  per-step Jacobian norms localize where a long rollout becomes ill-conditioned.
