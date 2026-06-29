# Worker brief: <task-title>

> The orchestrator copies this file to `coordination/tasks/<task-id>.md` and fills every angle bracket
> before spawning a worker. It is the full executable contract; the proposed task's one-line `note` was
> only the seed. Keep it concrete enough that a cold subagent can execute without further questions.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `<task-id>`. You are **NOT the orchestrator**. Do not spawn further
agents. Read this brief, do the task, write **all** results to disk under
`runs/<direction-id>/<task-id>/`, extend the training textbook, and exit. **Do not commit** — the
orchestrator reviews and commits your work. Fire the two pings below.

## Notifications (exactly two)
At the start:
```
python harness/tools/notify.py --kind started --task <task-id> "<one plain sentence: what you're starting>"
```
When your results are on disk:
```
python harness/tools/notify.py --kind finished --task <task-id> "<one plain sentence: what's ready to review>"
```
Use `--kind blocked` instead of `finished` if you hit a hard stop. **One sentence of human status, never a
metrics dump or a technical report.**

## Objective
<The precise question this task answers. One or two sentences.>

## Experiments / deliverables
<Concrete steps. What to vary, what to hold fixed, what to measure. If a claim of generality is wanted,
this MUST span several conditions/tasks — see Evidence discipline.>

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope every claim to exactly what was tested. One task supports a **hypothesis**, not a general truth.
- To claim generality, test generality (several tasks/conditions). Otherwise label the broad version a
  conjecture and say so.
- The manifest must carry an honest `hypothesis` (why the result holds) and `limitations` (what was not
  tested) field. Prefer "on this task, X" over "X is true".

## Visualization standard (this is graded, not optional)
Results are how the work becomes understandable, so make the visuals *informative*, not decorative.
- **Show the quantity the objective is actually about.** If the task optimizes the center of mass to a
  target, the video must overlay the center-of-mass trajectory and the target, not just show the blob
  moving. The viewer should be able to see *how close it got*, not infer it.
- **Prefer clear, simple, informative demos over dense technical ones.** A frame with the grid drawn and a
  heatmap of mass or velocity teaches more than a raw particle splat. Annotate axes, targets, and key
  quantities.
- Every `plot` result needs labeled axes and readable fonts (the dashboard renders small on iPad).
- Good visuals here are reusable: the best of them belong in the training textbook too (see below).

## Training textbook contribution (required)
End the run by adding **at least one short, standalone training page** under `reports/training/` in the
objective textbook voice (`spec/style_training_report.md`): impersonal, no first or second person, no
reference to this brief or "this run", and readable cold by someone who never saw the task. Prefer **one
or two short new pages over one long one**, link prerequisites with `[[anchors]]`, and embed an
informative figure or short video where a picture beats prose. Teach the *understanding* the task
produced, not a log of what was done.

## Output contract
Write `runs/<direction-id>/<task-id>/manifest.json` (schema v2 — see `runs/README.md`) plus its media,
with: `objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (video / image /
plot / table), and `training_refs[]` pointing at the page(s) you added. Leave everything on disk; do not
commit.

## Paths & params
- Run dir: `runs/<direction-id>/<task-id>/`
- Code: `sim/<...>`
- Key params: <horizon, resolution, seeds, optimizer, etc.>

## Definition of done
<The bar. What the artifact must contain for the orchestrator to accept it. Render-check the math.>

## Known failures to avoid
<Anything learned from prior attempts, e.g. verify gradients actually move the control before launching a
long sweep; do not spawn a long background command then end the turn waiting on it.>
