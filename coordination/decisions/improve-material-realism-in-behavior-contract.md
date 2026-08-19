<!-- auto_run_at: 1787102668 -->
# Contract — Improve material realism in behavior (deep) · RUNS FIRST

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/improve-material-realism-in-behavior.md`

**Runs FIRST of the two queued tasks**, because it changes the canonical materials and bumps
`physics_version`. Doing it first means the rendering task draws the *final* materials instead of ones
that are about to change under it.

## What it changes
Canonical `sim/physics/` — deliberately. Four complaints:
- **Water too mushy and sticky** → should flow freely.
- **Rubber compresses far too much** (a blob ends up occupying much less area) and **breaks too easily**.
- **Snow and sand are already good** → the task must show it did not move them.
- **No density at all** → sand and rubber should sink, snow should float.

## The one structural finding up front
**There is no density parameter.** `p_rho = 1.0` is a single global constant, so every material has
identical density today and nothing *can* sink or float. Buoyancy is therefore a new per-material `rho`
threaded through the transfer — and the brief requires it to **emerge from the mass ratio**, not be bolted
on as an explicit buoyancy force. Related lead: `NU = 0.2` is one global Poisson ratio shared by all
solids, which is very compressible; real rubber is ≈0.5. That is the prime suspect for the rubber
complaint, but moving it globally would also move snow and sand, so it likely becomes per-material.

## What gates it
The three promotion gates in `sim/physics/PROMOTION.md`: it must be genuine ground truth, **every existing
golden signature stays green**, and the version bumps. New canonical behaviour must add **new** signatures —
sand sinks, rubber sinks, snow floats, plus whatever the water and rubber fixes actually assert. Signatures
must state *qualitative truths*, never pin a number just measured.

## What it will NOT do
- **It will not touch the Demo page.** You said not to incorporate it yet. Bumping the physics does make
  the demo's generated `params.js` stale — the task will **say so and name the two scripts that fix it**,
  and stop there.
- **No rendering work.** That is the other queued task.
- **No retro-editing of prior runs' version stamps.** They are on the old ground truth; that is what
  stamping is for.
- **No "realistic" claims dressed up as measurements.** Aesthetic judgement stays labelled as judgement;
  what gets measured is spread, retained area, settled slope, sink depth.

## The headline visual
Each solid dropped into a pool of water, sink/float ordering annotated — plus old-physics vs new-physics on
the same scene and seed, as video, for every change. The old behaviour is the mandatory baseline.
