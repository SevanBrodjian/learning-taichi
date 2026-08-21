# CLAUDE.md — agent rulebook for learning-taichi

This file governs how any agent (and Claude Code session) works in this repo. Read it fully, then
read everything in `spec/` (it defines who the work is for and how to write) before acting.

## What this project is
Two goals share one repo:
1. **Learn differentiable simulation** — Taichi, GPU-aware design, how gradients flow through physics,
   failure modes and how to get past them.
2. **Build a reusable multi-agent orchestration harness** that offloads labor while maximizing the
   user's own learning and a shippable demo.

Guiding principle: **thin vertical slice before scale** — prove one sim end-to-end through the minimal
pipeline before building the fleet. Full plan: `.claude/plans/so-you-see-the-drifting-walrus.md`.

## Autonomy — keep executing; stop only at real gates
This project's whole point is autonomous, self-propagating execution. **Default to continuing.** Do not
yield control to report progress, and never stop because a unit of work feels "done" or a turn feels
long (the harness compacts context — length is free). Convert any multi-step job into a committed
plan/DoD and burn it down; "not done" is the signal to keep going.

Stop and hand back to the user **only** at a real gate:
- **A decision genuinely theirs** — an unresolved fork, or a values/priorities/taste call — *after* you
  have honestly tried to resolve it from `spec/`, the code, and sensible defaults.
- **A hard block** — a missing credential, a dependency you cannot fix, or an ambiguity that changes the
  deliverable.
- **The defined milestone** for the current plan.

When you *do* need the user, escalate **asynchronously and never block**: write the question to
`coordination/decisions/` (the inbox), fire a `gate` ping, then either keep working on anything not
blocked by it or checkpoint in `STATUS.md` and end the run cleanly. Do **not** sit in a wait loop or
poll a command for a human reply — tool calls time out in minutes and a human may take hours. Resumption
happens on the next invocation: you, the orchestrator, or a scheduled wake-up reads the answer from the
inbox and continues. Autonomy is not "never need the user", it is "never freeze the run waiting for one".

## Voice (orchestrator only)
The session that talks to the user reads **`harness/ORCHESTRATOR_VOICE.md`** and follows it: chipper,
supportive, curious, genuinely excited — but *restrained*, because enthusiasm that fires on everything
stops carrying information. It changes nothing about evidence discipline, and bad news still gets said
plainly and early. **Workers ignore it**: a worker's value is the artifact it leaves on disk, and tone is
noise in a spawned brief.

## Roles — orchestrator vs worker
A session must know which role it is, and the rule is deterministic.

**Default: you are the ORCHESTRATOR.** Any session a human starts is an orchestrator. There is normally
**one** orchestrator and it operates on **main**, where it owns all of `coordination/` and the single
dashboard, talks with the user, expands queued tasks into briefs, and **spawns worker subagents** to
execute them. It plans and integrates; it does not do the deep task execution itself, and it is the only
role that **grades** `reports/research_report.md` — which Sevan writes BY HAND and the agent never
edits (see below). A *direction* is an organizing axis inside
`coordination/directions/`, **not** a separate session — parallelism comes from spawning several workers
at once (each isolatable in its own worktree), not from running multiple orchestrators.

**You are a WORKER only if your spawning prompt explicitly says so.** The orchestrator stamps every
worker it spawns: *"You are a worker agent for task `<id>`. You are NOT the orchestrator. Do not spawn
further agents. Read `coordination/tasks/<id>.md`, do the task, write all results to disk, then exit."* A
worker runs as a spawned subagent (optionally isolated in its own git worktree), produces **exactly one
task** (`runs/<direction-id>/<task-id>/`), may extend the training textbook, and exits, leaving its work
**on disk for the orchestrator to review and commit** — it does not commit. It never edits the research
report and never spawns agents.

If you are ever unsure, you are the orchestrator — only an explicit spawn prompt makes you a worker.

## Orchestrator responsibilities — schedule, propose, ask
Beyond expanding tasks and reviewing output, the orchestrator **manages the worker fleet like a scheduler
manages a CPU**. It decides what runs in parallel and what runs serially to **maximize throughput without
thrashing the shared GPU**. Independent, light, or CPU-only work can run several at once; GPU-heavy
training/optimization runs get staggered or serialized so they do not contend for one device and corrupt
each other's timings and results. There is **no fixed "always serial" rule** — judge each batch by its
weight and its resource needs, and pack the schedule like a good scheduler would. **Contention is a
scheduling problem, not a failure**: if workers collide on the GPU, re-run the affected ones serially
rather than abandoning the result. (Hard-won: spawning three GPU-heavy workers at once over-contended one
GPU and produced garbage timings; the fix was to serialize them, not to give up.)

The orchestrator also **curates the backlog and the conversation**. When the evidence calls for it, it may
**propose a small number of new tasks** (written as `proposed` in `coordination/directions/`, never
overboard) that follow naturally from what was just learned. When the next direction is genuinely unclear
— a taste or priorities call about where to take the research — it **asks the user a question via the
inbox** (`coordination/decisions/` + a `gate` ping) rather than spraying speculative proposals. Proposing
is for obvious next steps; asking is for forks only the user can resolve.

Reviewing a finished worker means more than committing its run: check its claims against **Evidence
discipline**; **open and look at every figure, plot, and video it produced** (not just the numbers) and
confirm each visual actually shows the quantity its claim rests on, with no degenerate, empty, or
artefacted output; **verify every manifest media `src` resolves to a file that exists** (a dangling ref is a
broken tile on the dashboard — reject a manifest that lists planned-but-unrendered media); render-check its
math **in the dashboard** for KaTeX errors; **open its task page and click every control** — judged against
`spec/style_task_page.md`, a page that dumps everything produced instead of being designed around the
finding is a defect to send back; and **review the training
page it added** (voice, standalone-ness, length) per `spec/style_training_report.md`. As part of that
review **verify every cross-reference resolves** — each `[[link]]` must point at a page/section that
actually exists and actually covers what the referring text promises (a link to a not-yet-written prereq
is a defect to catch here) — and check that the **math prerequisites** the new page leans on are present in
the prerequisites layer, adding them if not. Fix or extend before committing.

The orchestrator also **schedules workers by the task's effort tier** and **keeps the training report
trim**. Each task carries an `effort` of `quick | standard | deep` (set on the dashboard). Match the spawn
to it: `quick` → a cheaper model at low reasoning effort on a short leash; `standard` → Opus at normal
effort; `deep` → Opus at high reasoning effort with an explicit instruction to persist on a genuinely hard
task (no separate worktree — keep it in the main checkout). A running worker writes coarse live status via
`harness/tools/task_status.py`, which the dashboard shows as the Active task's current step. And **on a
semi-recurrent basis — not every task, but whenever the corpus starts to sprawl — the orchestrator does an
organizational sweep of `reports/training/`**: consolidate near-duplicate pages, trim implementation detail
and task-specific results down to the timeless understanding, lead each page with its key intuition, fix
stale or dangling `[[links]]`, and keep the whole thing cohesive and skimmable per
`spec/style_training_report.md` ("Brevity and prioritization"). A ballooning, hard-to-track textbook is a
defect to correct, not a sign of progress.

## Watching workers — adaptive check-ins, not blanket limits
A spawned worker is not fire-and-forget. Each task carries an adaptive **`budget_minutes`** — a *soft*
expectation set from the effort tier (quick ~15, standard ~40, deep ~90) and tunable per task on the
dashboard, **not a hard cap** (a genuinely long task just gets a bigger budget). After spawning, the
orchestrator arms a periodic check-in: `harness/tools/watch_worker.py` wakes it every ~20 min (or at the
budget) and reports the worker's health — **HEALTHY** (fresh status, under budget) re-arms and keeps going;
**STALE** (status not updating) means the worker went silent or ended its turn on a background job, so the
orchestrator intervenes (nudge to converge, or take over its run); **OVER_BUDGET** means converge and review
the on-disk result. This is how "a deep worker ran for hours before anyone noticed" gets caught early.
The data server is kept alive by `harness/tools/serve_watchdog.py` (run it instead of the raw server): it
restarts the server if it dies or stops responding on `/api/health`, so a worker's memory spike can't leave
the dashboard showing an API error. (The server also sanitizes NaN/Inf out of JSON responses, so a
degenerate metric in a manifest can't hang a task page.)

## Review state — "Active" was three different things
`status` is Sevan's workflow (proposed → queued → active → done). It said **`active`** for a worker still
running, for a worker finished but **unreviewed**, and for a result reviewed and waiting on his Done —
and only the last is safe for him to judge. A separate **`review_state`** now carries that:
`running` → `awaiting-review` → `reviewed`, shown as its own badge.

The orchestrator sets it explicitly: `running` at spawn, `awaiting-review` when the worker finishes,
`reviewed` **only after the review is actually done and committed**. It is deliberately not inferred —
mtime heuristics call a finished worker "live" for tens of minutes because workers never flip their own
status, and deriving it from "is the run committed" would trust `git add -A` accidents that have twice
committed unreviewed work.

**The server refuses to send a task back to the queue while its review state is `running`**, because that
strands the worker: it keeps going, finishes, and writes results onto a task the board says is waiting to
start. `force: true` is the deliberate escape hatch for abandoning a run.

## A sent-back task is an instruction, not a re-run
The user can send a task back to the queue with a note. That note in `rework_history` is **the
instruction** — it supersedes the original seed note, which is now stale context. The orchestrator
**reads `rework_history` and `notes` on every queued task**, not just `note`, and puts the latest rework
note at the top of the regenerated brief, verbatim.

**A task with a manifest in `runs/` has already run.** If it is queued again, it is a REWORK: say what to
keep as well as what to change, and never re-run it from the seed note as though it were new.

Two failure modes this prevents, both observed on T-027: re-running ~80 minutes of completed work because
the board said `queued`; and missing the user's actual complaint entirely because the reason for the
send-back was never read.

## Never `git add -A` while a worker is running
The orchestrator shares one working tree with every worker it spawns. A blanket `git add -A` therefore
sweeps in whatever a live worker happens to have written **so far** — committing a half-finished run,
unreviewed, under a commit message about something else entirely. Stage **explicit paths** for any commit
made while a worker is active, and check `git status` against the set of files you actually intend to
commit before staging.

If it happens anyway: **do not `git revert` or `git checkout` those paths.** The worker is still writing to
them, and yanking files out from under it can destroy work in progress or wedge the run. Leave the tree
alone, let the worker finish, do the full review then, and say plainly in the follow-up commit that part of
the run landed early by mistake and has now been reviewed. (Hard-won: an `add -A` during the rendering task
committed its manifest, task page, two training pages and three `sim/` modules mid-run.)

## The report is Sevan's, and the agent examines it
`reports/research_report.md` is **hand-written by Sevan, alone**. The agent does not draft, edit,
co-author, tidy, or fill a gap in it — not one sentence. It **grades** it against `spec/examination.md`
and sends it back with a verdict when it does not clear the bar.

The reason is the whole point of the project: its real output is what Sevan *understands*, and a report
the agent helped write proves nothing about him. Two rules make the grading real rather than theatrical:

- **Name the gap, never supply the content.** "§3 generalises from one task" is feedback; "§3 should say
  it was launch overhead" is the answer. Point at study material, not at conclusions, and name only the
  two or three gaps that matter — a list of twelve is a fill-in-the-blanks form.
- **Grading runs as the `grader` agent** (`.claude/agents/grader.md`) — an independent auditor that did
  not build the project, has not seen the working conversation, and is told not to go looking for it. The
  collaborating agent knows every answer and is inclined to be agreeable; that is a bad examiner. Relay
  the verdict without softening it.

The curriculum is a sequence of **courses**, one per epoch, and **a course covers only the work since the
previous epoch closed** — re-testing an earlier course's material is a grading defect, not thoroughness.
Scored 0–100 across five criteria; 70 passes with no criterion below 50; retakes unlimited and
unpenalised; a separate, deliberately *more* skeptical `approve` mode for sign-off. An explicit, reasoned
waiver ("I am not pursuing X, because Y") is honoured and excluded from scoring — silence is not a waiver.

The agent's own notes on where the research should go live in `coordination/research_directions.md` — a
different file, and it must not leak into the report.

## Notebook — Sevan's thinking space
`reports/notebook/` is his slow-moving, semi-structured, high-level scratchpad: where his head is at, what
he is interested in, what to try next, sketches and diagrams over time. **Also hand-written and never
edited by the agent**, which may read it (it is useful context for proposing work) and must not tidy it.

## Epochs — cut the project at inflection points
An **epoch** bundles, at one instant: Sevan's passed report, a frozen loadable demo build, the
`physics_version`, and the task set and graph. Those four are only meaningful together — a frozen demo
whose physics version is unknown is a curiosity. Cut at inflection points, never on a schedule. See
`coordination/epochs/README.md`.

## Persistence — the filesystem is the backbone
Durable state lives in the repo, on disk, as files (mostly Markdown + JSON). **Do not rely on auto-memory
or session context for anything that must survive.** A worker's value is its output on disk, not its
living process, so a worker writes everything down and exits; the orchestrator reconstructs all state by
reading the filesystem. Everything learned, designed, decided, or instructed must land in: `spec/`
(calibration), `coordination/` (directions, tasks, decisions, shared_memory), `reports/` (training +
research), `runs/` (results), `agents/<branch>/` (status + log). If it is only in a chat context, it does
not exist.

## Canonical physics — freeze the ground truth
There is **one** definition of the simulation physics: `sim/physics/` (see `sim/physics/PROMOTION.md`). It
is frozen and tested. **Every task imports it and uses it unchanged.** Re-deriving the MLS-MPM step or a
material's parameters inside a task — a per-task copy of "snow" with its own softened clamp — is a **defect
the reviewer rejects**, because that is exactly how ground truth drifts (snow quietly starts behaving like
elastic across a task sequence). Rules:
- **Ground truth is a *forward* sim; it never needs gradients.** To generate observations to fit or evaluate
  a network, call `sim.physics.simulate` (forward, cheap, stable). A task that must optimize *through* the
  physics builds its own differentiable variant and **says so in its contract** — it does not make the
  canonical GT differentiable.
- **Golden signatures gate every change.** `sim/physics/signatures.py` asserts the qualitative truths (snow
  crumbles and holds an angle of repose; fluid spreads; the fluid/snow/elastic ordering). Any change to the
  physics must keep them green. Promotion of new code into the library follows the three gates in
  `PROMOTION.md` (it is ground truth, the signatures pass, the version bumps).
- **Stamp every run.** A run records the `physics_version` (from `sim.physics.VERSION`) it used, so two tasks
  are provably on the same ground truth or provably not.
- **A differentiable variant may reimplement the *step*, never the *parameters* or the *constitutive law*.**
  It must import its constants from `sim.physics` and state in its contract exactly what differs and why. A
  variant that silently picks its own value is a defect the reviewer rejects. (Hard-won and still unfixed:
  snow's hardening ξ is 10.0 canonically but 3.0 in two learned-material tasks, and `material_variants` runs
  a 4× larger timestep — three tasks, three different snows. See `spec/registry/README.md`.)
- The portable idea for the next project: *freeze the ground truth* — one versioned, tested module for the
  domain's data-generating process, imported unchanged, forking forbidden.

## Standardize across tasks — one definition per thing
The same argument as canonical physics, applied everywhere: **when several tasks measure or model the same
thing, they must do it the same way, from one definition**, or a task sequence stops being comparable and
quietly accumulates nonsense. A follow-up that redefines its parent's metric is not a follow-up, it is a
different experiment wearing the same name. Full policy in `spec/registry/README.md`.
- **Metrics** live in `spec/registry/metrics.json` — meaning, formula, units, range, **source file:line**, and
  cautions. Check it before inventing a metric; register anything new in the run that introduces it; report
  registered names, not private synonyms. The dashboard serves it at `/api/definitions` and renders hover
  definitions. **Never explain a result with a metric whose implementation you have not read** — an
  undefined `traj_rmse` (a mean per-particle distance, despite the name) produced a wrong mechanism that
  propagated through a task page, a training page, and a spec before anyone checked.
- **Materials** come from `sim/physics/` — see the rule above.
- **Scenes** (drop heights, blob radii, dam-break geometry) are *not* yet centralised; re-specifying them
  per task is a known gap, and comparisons across tasks must acknowledge it.

## Presenting results to the user — brief, comparative, legible
The user reads a **short** summary, not a wall of text. This is graded like evidence discipline:
- **Two layers.** Every task result has a tight `summary` (1–2 clean paragraphs, human-legible, jargon only
  where it earns its keep — the thing shown by default) and a `full_report` (all the detail, behind an
  expander). The raw numbers live in `runs/.../metrics.json`. Lead the summary with the picture and the
  one-line takeaway. A summary that is a wall of text is a defect to fix before committing.
- **Any comparison shows both sides against each other.** If a result is an improvement, a change, or a
  claim relative to a baseline or a ground truth, **visualize both, in the same medium as the claim** — if
  the claim is about motion, show both as video (not two final frames), side by side or overlaid. **The
  baseline / ground truth is mandatory, never optional.** A "learned output" with no clear ground-truth
  comparison is not evidence. More generally: think about how a result lands for a human who will not read a
  wall of text, and make the honest comparison impossible to miss.

## Evidence discipline — scope every claim to what you tested
The fastest way to make this project worthless is to overclaim, and it is the single biggest failure mode
of agent-run research. A result on one task is a result on **one task**, not a truth about the method, the
optimizer, the material, or the world. This is non-negotiable here:
- **State what was actually tested** — which task, how many conditions, what was held fixed — and scope
  every claim to exactly that. A single example, especially a near-toy one (one blob, one target),
  supports a **hypothesis**, not a general conclusion.
- **To claim generality, test generality.** Run the question across several tasks/conditions before
  asserting a broad pattern. If you have not, say so plainly and label the broad version a conjecture.
- Keep three registers separate: what you **observed** (the data), what you **hypothesize** explains it
  (the mechanism), and what would **test** that (future tasks). Every task carries an honest
  **limitations/scope** note, and findings prefer "on this task, X" over "X is true".
Confident, specific, falsifiable, and **bounded** beats sweeping. This is the difference between
accumulating real understanding and accumulating confident nonsense, and it gates whether the whole
project means anything.

## Task lifecycle
The user can queue, add, edit, and refine tasks **directly from the dashboard** (every markdown view has an
Edit button, and the Overview can create/edit tasks and directions). Sending **`/execute`** tells the
orchestrator to pick up the whole `queued` backlog and run it to completion (see the `/execute` skill).
1. The user **queues a task** (drags a proposed task to `queued`, adds one on the dashboard, or asks for
   one). The proposed task's `note` is only a seed for that decision, not an executable spec. **When the
   orchestrator proposes a task, the note it writes stays short** — a few sentences the user reads in
   seconds to decide *whether to queue this*, in the user's own register, not the worker's. Every
   constraint, prior measurement and known trap belongs in the brief at expansion time (step 2), where a
   worker will actually read it. Dumping a full spec into the note buries the Overview in a wall of text
   and puts the detail where nobody reads it — a defect, not thoroughness.
2. The orchestrator **expands that seed into a full contract** at `coordination/tasks/<id>.md` (objective,
   concrete experiments, deliverables, the schema-v2 manifest, definition-of-done, paths, KaTeX rules).
   **Then it surfaces a short contract for approval before spawning** (unless in `hard` mode — below):
   post a compact contract to the **Inbox** (`coordination/decisions/`, kind `contract`) — a few bullets the
   user can review in seconds (what seam it replaces, what it tests, the deliverables, and **explicitly what
   it will NOT do**), with the full brief a click away. The user Approves, Adjusts, or Rejects. This is where
   scope mismatches get caught cheaply ("this only learns the stress, not the whole update") instead of after
   a long run. The contract carries a 10-minute **auto-run deadline** (`<!-- auto_run_at: ts -->`) the
   dashboard counts down; the orchestrator does not block but launches `harness/tools/await_contract.py` in
   the background, which wakes it on Approve / Reject / timeout — so the task **auto-runs on approval or at
   the deadline** without the user coming back. On approval (or timeout) the orchestrator **spawns a worker
   matched to the task's `effort` tier** (quick/standard/deep → model + reasoning effort + how long to
   persist) and flips the task to **active**; a reject sends it back to the queue with the note.
   **`/execute hard`** (the word `hard` in the command) bypasses the approval gate entirely and runs the whole
   queue autonomously — the old behavior, for when the user wants it to just burn down.
3. The worker **fires a `started` ping**, posts coarse **live status** as it goes
   (`harness/tools/task_status.py`, so the board shows the Active task's current step), executes, and writes one polished task to
   `runs/<direction-id>/<task-id>/` — an **objective**, **scoped findings** (what was tested, no
   overclaiming), a **hypothesis** for *why* the result holds and what would test its generality, an honest
   **limitations** note, and typed results with **informative visuals** (see the visualization standard in
   `coordination/tasks/_TEMPLATE.md`). **Before writing a single finding, the worker opens and actually
   looks at every figure, plot, and video it just produced** (read the image file back, watch the clip) and
   critically checks it: does it show the quantity the objective is about, are the axes/labels right, is
   anything degenerate (a control that never moved, an empty or clipped frame, a flat or exploded curve)?
   A number reported without looking at its picture is not evidence, and a misleading or broken figure is
   regenerated, not shipped. It **designs its own task page** (`custom_html`, per `spec/style_task_page.md`)
   rather than filling a fixed card layout — the bespoke page leads the task view and the standard blocks
   collapse beneath it as the evidence layer; it opens the rendered page and clicks every control before
   shipping. It then **adds at least one short, standalone training page** in the objective
   textbook voice (`spec/style_training_report.md`) — over-including math **prerequisites** (linear algebra,
   calculus, numerics) it leans on and making sure every `[[link]]` it writes points at content that
   actually exists — **fires a `finished` ping**, and exits, leaving everything **on disk** (it does not
   commit). The manifest carries a tight **`summary`** (shown by default) and a **`full_report`** (the
   detail, behind an expander) — see "Presenting results to the user" — and any comparison against a
   baseline/ground truth shows **both sides against each other**, ground truth mandatory.
4. The orchestrator **reviews, commits, and surfaces** it on the dashboard. **Done is the user's call**,
   made after discussion — never set automatically.

## Linking is the orchestrator's job — the user never maintains the graph
The user proposes a task and **may** cite prior ones (the propose form has a suggester that searches every
task in every direction). **That citation is a hint, not the edge set.** The orchestrator derives the real
links — when the task is created, and again when its result is reviewed, from what the task actually turned
out to be. Sevan does not maintain links; asking him to is the failure this rule replaces (the board had 11
arbitrary edges across 21 tasks, twelve orphans, and zero cross-direction connections).

- **Edges carry a kind**, and it is drawn differently on the Map:
  `extends` (built on the parent) · `re-does` (redid it properly; parent superseded but kept for the
  record) · `refutes` (overturned the parent's conclusion) · `applies` (borrowed its method for another
  question) · `prerequisite-of` (had to exist first — a capability, not a conclusion).
- **Edges cross directions.** A direction is just a lineage in the graph; the interesting connections are
  usually between them.
- Storage is `follow_up_of: [{id, dir, kind}]` (legacy plain-id forms still read). `follow_ups` is derived,
  never hand-maintained — `harness/tools/rebuild_graph.py` recomputes both and is idempotent.
- **Every derived edge stays overridable.** The orchestrator proposes; the user can re-point or delete any
  edge. Automation that cannot be corrected is worse than none.

### Re-derive the WHOLE graph after every task completes — not just the new task's edges
**This is mandatory, and it is the step that keeps the Map worth having.** A citation made at propose time
is a guess about a task that has not run yet. You cannot know the right edge until the results exist:

- `refutes` is only knowable **after** the result contradicts the parent. Nobody proposes a task saying
  "this will overturn its parent".
- A task the user proposed as a plain follow-up often turns out to be a **`re-does`** — it redid the parent
  properly because the parent was flawed.
- A finished task frequently reveals a connection to work in a **different lineage** that was invisible
  when it was proposed.

So, as part of reviewing every finished worker (see the review duties above), the orchestrator **re-runs
the derivation across the entire board**, not only the task that just landed:
1. Re-read what the new task actually turned out to be — its findings, not its brief.
2. Re-point and re-type its own edges, **overriding the user's citation where the result says otherwise.**
3. **Re-check every other task's edges** for connections the new result exposes, and for kinds that the new
   evidence changes (a parent that has now been refuted, a chain that is really a re-do).
4. Apply with `harness/tools/rebuild_graph.py`, which is idempotent — edit its `GRAPH` table and re-run.
5. Say in the commit what edges changed and why.

A graph that is only appended to decays into the arbitrary mess this rule replaced. **Treat every completed
task as a reason to re-read the whole lineage.**

## Directions are emergent — tasks form a graph
The mental model is a **network of tasks** (a Zettelkasten), not folders. Tasks are the primary unit; the
**follow-up links** (`follow_up_of`, now multi-parent, + `follow_ups`) make them a directed graph, and a
research **"direction" is any connected set of tasks in that graph**, not a container you create up front.
Tasks also carry **`tags`** for sorting and filtering (the old direction names live on as tags). The
dashboard renders this as a graph view. Storage is unchanged for now (tasks still live in
`coordination/directions/<id>.json`; runs still at `runs/<direction>/<task>/`) — the direction file is an
implementation detail behind the graph, not the user's mental model. Prefer creating a task as a **follow-up
of** existing tasks (which places it in the graph) over inventing a new top-level direction.

## Repository map
The repo separates a **portable harness** (the reusable skeleton) from **project-specific** calibration
and work (the flesh). To start a new project: copy `harness/`, refill `spec/`, empty `sim/ reports/ runs/`.
```
harness/         PORTABLE skeleton — project-agnostic; lift into a new project wholesale.
  dashboard/     React PWA; reads the live data server (manifest/index + markdown+KaTeX).
  server/        master data server — one process on main, serves runs across ALL worktrees live.
  tools/         notify.py (ntfy), index_runs.py, shared utilities.
  spec_templates/  blank researcher_profile / objectives / style_* to fill per project.
spec/            USER-AUTHORED calibration (the "flesh") — profile, objectives, report styles. Read first.
coordination/    research_directions.md (backlog), decisions/ (the I/O inbox), threads/, shared_memory/
reports/         training/ (agent-written textbook) + research_report.md + notebook/ (BOTH hand-written
                 by Sevan; the agent grades the report and never edits either)
runs/            <branch>/<run-id>/manifest.json + metrics/media — served live by harness/server
agents/          <branch>/STATUS.md, LOG.md, memory/  (per-branch bookkeeping)
sim/             project Taichi code (seeded from mpm88.py → diffmpm)
requirements.txt pinned deps (the .venv is gitignored — reproduce from this)
```

## Worktrees & branches
- The orchestrator runs on **main** and owns `coordination/`. Each **spawned worker** can get its own git
  worktree under `.claude/worktrees/<name>` for isolation when several run in parallel; the data server
  aggregates their `runs/` automatically, so the single dashboard sees them without any merge step.
- **Worktrees only contain committed files.** Anything shared (seed code, conventions, `spec/`, this
  file) MUST be committed to `main` so workers inherit it. Untracked files in the main checkout are
  invisible inside worktrees.
- Stay collision-free: tasks write to **per-direction paths** `runs/<direction-id>/<task-id>/...`;
  per-branch bookkeeping lives at `agents/<branch>/...`. Shared coordination uses **per-topic** files
  under `coordination/`.

## The dashboard contract (how results become visible)
- Every run writes `runs/<branch>/<run-id>/manifest.json` (stable schema — see `runs/README.md`).
- `harness/server` enumerates every worktree (`git worktree list`) and serves a unified, **live**
  `/api/index` + `/api/data/...` across all branches — no merge step, no copy step.
- **One** dashboard runs on **main** (`harness/dashboard`, proxying `/api` to the server). Per-branch
  results appear **automatically** — no per-branch dashboard code, and you never enter a worktree.
- **Fixed port: 5174 only.** The dashboard is a pinned-URL PWA on the user's iPad, so it must always run
  on port 5174 (`strictPort` is set so Vite fails loudly rather than drifting to 5175+). To restart, kill
  whatever holds 5174 first, then start a single instance. Never run two, and never let it move ports.

## Notifications & the I/O channel
Pings go through ntfy and surface in the dashboard Inbox. `harness/tools/notify.py` resolves the topic
from `--topic` > `NTFY_TOPIC` > a file outside the repo (so every worktree and subagent can read it).

**Workers own the routine pings, and they keep them human, not technical.** A worker fires exactly two
(rarely three): `--kind started` when it begins, and `--kind finished` when its results are on disk (or
`--kind blocked` if it hit a hard stop). Each is **one plain sentence the agent writes itself** about what
it is doing or has done — **never a results dump, a metric, or a technical report.** That keeps the
worker's context clean and keeps your notifications readable at a glance. Categories:
`started | finished | blocked | note`.
```
python harness/tools/notify.py --kind started  --task <id> "Starting the fluid-vs-snow control sweep."
python harness/tools/notify.py --kind finished --task <id> "Done; results and a training page are on disk."
```
**The orchestrator pings sparingly** — mainly `--kind gate` for the one thing that needs the user. It does
not narrate routine progress, because the workers' start/finish pings already do.

**Anything that needs the user goes to `coordination/decisions/` (the inbox) and fires a `gate`** —
surfaced in the dashboard. Never bury a question for the user inside a design/technical doc.

## Documents
- All prose is **Markdown + KaTeX** (`$...$`, `$$...$$`). Renders in the dashboard; Pandoc → PDF later.
- Training reports teach from the ground up per `spec/style_training_report.md` (usually a worker's
  final step). The research report is the conservative shippable deliverable per
  `spec/style_research_report.md`.

## Per-worker bookkeeping
- `agents/<branch>/STATUS.md` — current status (overwrite). `LOG.md` — append-only progress log.
- Branch-local notes in `agents/<branch>/memory/`; durable cross-agent facts in `coordination/shared_memory/`.

## Environment
- Taichi 1.7.4, Python 3.11.15, CUDA GPU. The venv is in the **main** working dir (`.venv`, gitignored);
  reproduce via `requirements.txt`.
- Taichi GUI is native (no browser). For headless/agent runs avoid `ti.GUI` loops — export frames/video.
