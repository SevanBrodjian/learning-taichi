---
name: execute
description: Run the learning-taichi orchestrator loop over the directions board. Picks up every queued task, expands each into a full worker brief, spawns worker subagents (scheduling GPU-heavy runs intelligently rather than strictly serially), then reviews, commits, and leaves each result active for the user's Done. Use this whenever the user types /execute, or asks to run, execute, kick off, or burn down the queued tasks, the backlog, or the board, or to process or start everything that is queued. If nothing is queued, say so and list what is proposed.
---

# /execute — burn down the queued backlog

You are the **orchestrator** (see `CLAUDE.md` → Roles). `/execute` means: take the whole `queued` backlog
and run it to completion, then report. This is the everyday loop — the user curates the board on the
dashboard, queues tasks, and types `/execute`.

**Two modes.** By default, `/execute` **expands each task into a short contract and surfaces it for approval
before spawning** (step 3) — the user catches scope mismatches cheaply. If the command contains the word
**`hard`** (e.g. `/execute hard`), **bypass approvals entirely** and run the whole queue autonomously (the
old behavior). Check the invocation text for `hard`.

## 1. Read the board
Enumerate `coordination/directions/*.json` and collect every task with `"status": "queued"`. If there are
none, tell the user the queue is empty (mention what is `proposed`, in case they want to queue something)
and stop. Do not invent work.

## 2. Plan the schedule like a CPU scheduler
Decide what runs in parallel and what runs serially to **maximize throughput without thrashing the shared
GPU** (`CLAUDE.md` → Orchestrator responsibilities). There is no fixed serial rule:
- Independent, light, or CPU-only tasks can run several at once.
- GPU-heavy training/optimization runs get staggered or serialized so they do not contend for one device
  and corrupt each other's timings. Cap concurrent GPU-heavy workers at 1–2.
- Contention is a scheduling problem, not a failure — if workers collide, re-run them serially rather than
  giving up on the result.
State the plan briefly before you start.

## 3. For each task: expand → (approve) → spawn → flip to active
1. Expand the queued seed into a full contract by copying `coordination/tasks/_TEMPLATE.md` to
   `coordination/tasks/<task-id>.md` and filling every field (objective, experiments, evidence-discipline
   scope, visualization standard, training-page requirement, paths, DoD). Fill the **Effort tier** line from
   the task's `effort` field. **Use the canonical `sim/physics/` for any ground truth** — never fork the
   physics into the task (CLAUDE.md → "Canonical physics").
1b. **Approval gate (skip in `hard` mode).** Post a short contract to the Inbox
   (`coordination/decisions/<task-id>-contract.md`, plus a `gate` ping) — a few bullets: the seam it
   replaces, what it tests, the deliverables, and **explicitly what it will NOT do** — with a pointer to the
   full brief. Then move on to other unblocked work; resume this task when the user Approves (in `hard` mode,
   skip this and spawn immediately). This is where "you're learning the stress, not the whole update" gets
   caught before the compute is spent.
2. Spawn a **worker subagent** with the role stamp from the template, **matching the spawn to the task's
   `effort` tier** (set on the dashboard, read from the direction JSON — default `standard`):
   - **quick** → a cheaper/faster model at low reasoning effort (e.g. Sonnet), short leash. Good for light
     learning tasks and quick forward-sim demos.
   - **standard** → Opus at normal effort. The default.
   - **deep** → Opus at high reasoning effort, and tell the worker in its prompt to **persist** — iterate,
     debug, and run the sweeps a genuinely hard task needs rather than stopping at the first plausible
     result. (No separate git worktree — keep the worker in the main checkout.)
   The worker writes results to `runs/<direction-id>/<task-id>/`, calls `harness/tools/task_status.py` at a
   few milestones so the board shows its live step, adds a training page, fires its start/finish pings, and
   exits without committing.
3. Flip the task to `active` on the board (POST `/api/task-status`, or edit the direction JSON) so the
   dashboard shows it live.

## 4. Review and commit each finished worker
When a worker finishes, **review before committing** (`CLAUDE.md`):
- Scope-check every claim against **Evidence discipline**; verify `hypothesis` + `limitations` are present
  and honest.
- **Open and look at every figure, plot, and video the worker produced** (read the image files back, watch
  the clips), not just the numbers. Confirm each visual shows the quantity its claim rests on and is not
  degenerate (a control that never moved, an empty/clipped frame, a flat or exploded curve). Reject and
  re-run rather than commit a misleading or broken figure.
- Render-check the math (KaTeX) and the manifest results in the dashboard.
- Review the **training page** it added: objective voice, standalone, short, informative visuals — and
  **verify every `[[link]]` resolves** to a real section and that the **math prerequisites** it leans on
  exist (add them if not). Fix or extend it per `spec/style_training_report.md`.
- Then commit the worker's files. **Leave status `active`** — *Done is the user's call*, never set
  automatically.

## 5. Report
When the queue is drained (or blocked), summarize: what ran, what landed (with dashboard-visible results),
what each is awaiting (the user's Done), and anything that needs a decision (escalate via
`coordination/decisions/` + a `gate` ping). If a genuine fork in direction emerges, **ask the user a
question** rather than inventing more tasks.

## Notes
- Never declare a worker failed until its background work is confirmed finished — a "came to rest" signal
  can fire during a gap while a sweep is still running.
- If a worker produced degenerate output (e.g. a control that never moved), quarantine it, diagnose, and
  re-spawn with the failure noted in the brief's "Known failures to avoid".
