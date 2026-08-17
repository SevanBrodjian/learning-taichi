<!-- auto_run_at: 1786930580 -->
# Contract — Sand as a fourth canonical material, + four materials in one grid (deep)

**Approve to run, or Reject with a note.** Full brief:
`coordination/tasks/sand-as-a-fourth-canonical-material-and-four-materials-in-one-grid.md`
**Runs FIRST.** The WebGPU task is held until this finishes — see the scheduling note below.

## What it changes
**Frozen ground truth.** This is a promotion event under `sim/physics/PROMOTION.md`, not a task-local
material: sand enters `sim/physics/`, gets **its own golden signature** (it must pile and hold an angle of
repose, not spread like a fluid), every existing signature stays green, and **the version bumps**.

## The number that matters most
**Sand's stable `dt`.** A shared grid means ONE timestep, so the stiffest material present sets the cost of
the whole scene. Today: water 139 substeps/frame, elastic 167, **snow 333** — and snow's cost is *not* from
its stiffness (E=150, the lowest) but from hardening making compacted snow ~3× stiffer than elastic. If sand
hardens or dilates, its `dt` could be smaller still. The brief demands this be measured and reported as a
headline, with a plain statement of whether **sand or snow** now binds the Demo's frame budget.

## Also delivered
- **Per-particle material id** and a branching step, so one grid holds all four at once. Material
  interactions may be weird — explicitly fine; the goal is coexistence for the first time.
- Proof the refactor changed nothing: a single-material run through the new path must match canonical
  `simulate` **to self-noise**, with the number shown.
- Particle-rendered video of each material alone and all four together — **elastic red, sand yellow, water
  blue, snow white**.

## ⚠️ Scheduling — worth knowing
Serialized, not parallel. The WebGPU task's headline deliverable is *timings*, and a concurrent Taichi
sweep would corrupt them (`CLAUDE.md` carries this scar). Running sand first also means its version bump
lands **before** the port generates its constants, so that coupling disappears instead of being managed.

## What it will NOT do
- **Not** change fluid / elastic / snow parameters. Sand is added alongside them.
- **Not** physically correct multi-material contact — coexistence only.
- **Not** rendering polish. Dots and colour; the visual pass is a later task.
- **Not** interactive, **not** in the browser, **not** differentiable. Forward Taichi only.
- **Not** wired into the Demo tab.
