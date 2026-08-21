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

## 1. Read the board — including WHY something is queued
Enumerate `coordination/directions/*.json` and collect every task with `"status": "queued"`. If there are
none, tell the user the queue is empty (mention what is `proposed`, in case they want to queue something)
and stop. Do not invent work.

**Read `rework_history` and `notes` on every queued task, not just `note`.** A task can be queued for two
completely different reasons and they demand different responses:

- **Never run before** → expand the seed `note` into a brief as normal.
- **SENT BACK by the user** → `rework_history` has an entry, and its latest note is *the actual
  instruction*. The seed note is stale context; the rework note is what must change. It goes at the TOP
  of the regenerated brief, verbatim, as the objective — and the brief says explicitly what was wrong
  with the previous attempt so the worker does not repeat it.

**A task that has already run and been sent back must NEVER be re-run as if it were new.** Check
`runs/<direction>/<task>/` before spawning: if a manifest is already there, this is a rework, and the
brief must say what to keep as well as what to change. Re-running a completed task from its seed note
throws away everything that was right about it.

This is a scar, not a hypothetical: T-027 was sent back with a specific note about the water rendering,
and the orchestrator read only `note`, saw a task it had already completed, and nearly re-ran the whole
thing. The user's reason for sending it back was invisible because it was never read.

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
   (`coordination/decisions/<task-id>-contract.md`) — a few bullets: the seam it replaces, what it tests, the
   deliverables, and **explicitly what it will NOT do** — with a pointer to the full brief. **Include a
   machine-readable auto-run deadline** as an HTML comment near the top: `<!-- auto_run_at: <unix ts now+600> -->`
   (10 minutes out; the dashboard shows a live countdown, and the task auto-runs at the deadline if the user
   has not acted). Fire a `gate` ping.
   Then **set up the wake instead of blocking**: launch, in the **background** (never a foreground wait —
   those time out), `python harness/tools/await_contract.py --id <task-id>-contract --deadline <ts>`. It
   exits when the user Approves (0), Rejects (1), or the deadline passes (2), which re-invokes you — then
   **spawn** on approve or timeout (run it as-is), or **send the task back to the queue** with the note on
   reject. Meanwhile do any other unblocked work. This is where "you're learning the stress, not the whole
   update" gets caught before the compute is spent, and it auto-runs so the user need not come back.
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
3. Flip the task to `active` on the board (POST `/api/task-status`) so the dashboard shows it live,
   **and set its review state to `running`** (POST `/api/task-review`). That badge is how Sevan can
   tell a live worker from a finished-but-unreviewed result, and it is what stops him sending a task
   back mid-run — the server refuses that while the state is `running`.
4. **Arm the adaptive check-in.** Note the spawn time and the task's `budget_minutes` (a soft expectation
   from the effort tier, tunable on the dashboard). Launch, in the **background**,
   `python harness/tools/watch_worker.py --direction <d> --task <t> --budget <min> --started <unix_ts>`.
   It wakes you every ~20 min (or at the budget); on its exit act per the verdict: **HEALTHY (0)** → re-arm
   another watch and keep going; **STALE (1)** → the worker went silent / ended its turn on a background job
   → intervene (nudge it to converge, or take over its run); **OVER_BUDGET (2)** → converge (review the
   on-disk result / take over — if a manifest exists it likely has a complete result). This is the fix for
   "a deep worker ran for hours and nobody noticed" — budgets are a soft check, not a hard cap.

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
- **Re-derive the WHOLE task graph — mandatory, every time.** A citation made when the task was proposed is
  a guess about a run that had not happened yet. Now that the results exist, re-read what the task actually
  turned out to be and:
  - re-point and re-type **its own** edges, overriding the user's citation where the result says otherwise
    (a proposed "follow-up" is very often really a **`re-does`**, and **`refutes`** is only ever knowable
    after the fact);
  - **re-check every other task's edges** for connections this result exposes and kinds this evidence
    changes.

  Apply by editing the `GRAPH` table in `harness/tools/rebuild_graph.py` and re-running it (idempotent),
  then say in the commit which edges changed and why. Appending only is how the graph decayed last time.
- **Set the review state.** `awaiting-review` the moment the worker finishes, and **`reviewed` only
  after you have actually done the review above and committed.** Sevan judges results off that badge:
  `unreviewed` means the figures have not been opened and the claims have not been scope-checked, so
  he should not send it back yet for something you would have caught. Never stamp `reviewed` early.
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
