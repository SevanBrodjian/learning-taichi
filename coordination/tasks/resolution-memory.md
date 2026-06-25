# Task brief — Gradient memory and throughput vs resolution

> WORKER agent for `resolution-memory` in `long-rollout-pathologies`. You are NOT the orchestrator. No
> spawning, no dashboard, and **do NOT git commit** (leave on disk for review). Read `CLAUDE.md`
> (especially **Evidence discipline**), `spec/`, `coordination/shared_memory/working-with-sevan.md`,
> `runs/README.md`.
>
> Write ONLY to `runs/long-rollout-pathologies/resolution-memory/` and a uniquely-named sibling script
> `sim/resolution_memory.py`. Do NOT edit any shared file. Deps (taichi, numpy, matplotlib) are in the
> venv; only install if missing and report it. venv python:
> `C:/Users/Owner/Projects/learning-taichi/.venv/Scripts/python.exe`. Absolute paths, on MAIN.
> IMPORTANT: other GPU workers may be running in parallel tonight, which will pollute timing and memory
> numbers. Detect/note this; if a measurement looks contaminated, say so, and prefer reporting the
> *scaling shape* over absolute one-off numbers. The orchestrator can re-run you solo if needed.

## Objective
Map the practical envelope of what is differentiable, at what cost. Vary **grid resolution** (e.g. 32, 64,
128) and **particle count** (e.g. 1k, 4k, 16k) for the differentiable 512-step rollout, and measure:
(a) forward+backward **wall-clock per iteration** (throughput), and (b) the **memory of the stored
backward tape** — the time-indexed fields dominate: roughly `steps x (grid^2 + particles) x
(value+grad)`. Report measured GPU memory if you can get it (Taichi memory stats / `nvidia-smi` /
torch.cuda if importable); otherwise give a principled estimate from field sizes and clearly label it an
estimate.

## What to produce
- A sweep over a sensible grid x particle matrix. Do NOT hard-OOM the GPU blindly: back off and **record
  where it OOMs** — that boundary is itself a key result.
- Plots: time-per-iter vs resolution, and memory vs resolution (log axes as needed).
- A `table`: grid | particles | tape memory (MB, measured or est.) | fwd+bwd time/iter | OOM?

## Honesty (mandatory)
Scope to this GPU and this sim. Clearly separate measured numbers from estimates, and flag any numbers
likely polluted by concurrent GPU load. `hypothesis` (how memory and time scale, and what that implies for
the max horizon/resolution you can differentiate) + `limitations` required.

## Deliverables (schema v2, `runs/long-rollout-pathologies/resolution-memory/`)
`manifest.json` (status `active`): objective, scoped findings, hypothesis, limitations; results: the two
plots (PNG `image`) + the table. Video optional. `training_refs`: ["differentiating-the-rollout"]. Leave
on disk; do not commit.
