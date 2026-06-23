# Researcher profile

> The "who is this for" calibration. Everything in `reports/` and the dashboard is written for this
> reader. Co-authored — refine the `TODO(you)` lines.

## Who
Sevan — researcher in **generative interactive simulations and world models**.

## The broader vision
**Structured generative worlds**: worlds with the unconstrained generative potential of diffusion
models, but that maintain *persistent commitments*, *coherence*, and *editability* for authoring —
serving expression, education, entertainment, and exploration.

This project advances one axis of that vision: **structured, explicit dynamics** that provide
persistence, consistency, and efficient processing while exposing editable components — all while
remaining differentiable, learned, and general.

## What I want from THIS project
*(This will evolve as we go as well)*
- Working fluency with **Taichi** and efficient, GPU-aware design.
- A real understanding of **how gradients flow through physical simulation**, what the **failure
  modes** are, and how to get past them.
- A calibrated sense of **what's easily achievable today, what's achievable with work, and what's
  open** in differentiable simulation.
- An impressive, explainable demo + reusable concepts for future projects.

## Background to calibrate explanations against
In general I have quite solid intuition with autodiff, gradient learning, deep learning, neural networks, etc. and am capable in mathematics. However, I actually find I'm surprisingly untrained in a lot of the technicals and details, and I really want to learn. So intuition is deep but also jagged, rigor is relatively low and want to get to deep.
- Comfort with **deep learning / autodiff** in general: **medium to high (not uniform)**.
- Comfort with **linear algebra, calculus, numerical methods**: assumed **medium to high, want to learn more**.
- Prior exposure to **physical simulation / MPM / continuum mechanics**: assumed **low–moderate** —
  explain sim-specific machinery (P2G/G2P, constitutive models, MLS-MPM) from the ground up.
- Prior **Taichi** experience: assumed **low** — introduce Taichi idioms (fields, kernels, the autodiff
  tape) as they appear.
- GPU programming depth (memory hierarchy, kernel design): assumed **low-moderate**.

## How the output gets used
Self-integration into my mental model, presentations, and discussion with others — so I must be able to
explain any automated work *in depth* myself. Demos ship to a React + Django personal website.
