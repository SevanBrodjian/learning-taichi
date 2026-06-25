# Task brief — Training textbook: build the foundation

> You are a WORKER agent, not the orchestrator. Do not spawn agents. Read `CLAUDE.md` and all of
> `spec/` first (especially `spec/style_training_report.md` and `spec/researcher_profile.md`). Write all
> output to the MAIN checkout at `C:/Users/Owner/Projects/learning-taichi/` using absolute paths. When
> done, write everything to disk and STOP; the orchestrator reviews and commits. Direction:
> `differentiable-control`. Task id: `training-foundations`.

## The problem (the user's words)
The training textbook currently reads "less like a textbook and more like a homework page from halfway
through a course." A reader hits a line like "older transfers store only a particle velocity and lose
the local rotation and shear of the velocity field" without ever having been told what a *transfer* is,
why we store particle velocities, or the physical intuition behind any of it. It needs to **start from
the beginning**.

## Objective
Rebuild the front of the textbook so a reader with strong ML / autodiff intuition but **low physical-
simulation background** (see `spec/researcher_profile.md`) can build genuine understanding from zero.

1. Add the missing **Motivation & Background** group (its own folder of sections) — per the spec this is
   required and is currently the biggest gap. Cover: why differentiable physics at all; where MPM sits
   versus FEM / SPH / Eulerian methods and why the particle+grid hybrid exists; and the honest through-
   line to the user's structured-generative-worlds vision.
2. Rewrite and expand the **Prerequisites** (and the opening of **Core/01**) so every concept is built
   up before it is used: what a particle↔grid *transfer* is and why it exists, why a particle carries a
   velocity (and only then why APIC adds an affine matrix), what the background grid is *for*, and the
   physical meaning of mass, momentum, the volume ratio $J$, and stress — all before the equations.

## How to write it (read the spec; the essentials)
- First principles, **physical intuition before math**. Overexplain rather than underexplain; it is
  easy to skim what you know, hard to fill a silent gap.
- Tie ideas back to the world-models vision frequently and concretely.
- Voice: confident, clear, concise, precise. Avoid em dashes and semicolons, avoid colon-led lists,
  avoid LLM "we ask whether"-style phrasing. Dry wit is welcome, nothing cringe.
- It is a **teaching** document. No "things to try next", no research planning (that lives in
  `coordination/directions/`).
- **KaTeX rules (strict):** multiline display math MUST be the three-line `$$` form (open and close on
  their own lines); single-line `$$f(x)$$` is fine; never use `\*`; brace every multi-character sub or
  superscript. Use `[[wiki-links]]` (section ids) to cross-reference.

## Deliverables (files, on MAIN)
- `reports/training/motivation/` with sections (e.g. `01-why-differentiable-physics.md`,
  `02-where-mpm-sits.md`) — the new group.
- Rewritten/expanded `reports/training/prerequisites/01-mpm-in-context.md` and `02-math-toolkit.md` that
  genuinely start from zero (transfers, why store velocity, the grid's role, the physical meaning of
  each quantity).
- Update `reports/training/index.json` to add the `motivation` group **first** (before prerequisites),
  listing the new sections, plus any new prerequisite sections you add.
- Existing core sections must keep working; you may lightly improve their openings for continuity, but
  the focus is the foundation.

## Definition of done
A reader who knows ML but not simulation can read Motivation → Prerequisites → Core/01 and understand
what MPM is doing and why, with physical intuition, before being shown a single equation. No unexplained
jargon on first use. Return a short summary of the files you created and changed.
