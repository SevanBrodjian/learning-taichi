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

## Roles — orchestrator vs worker
A session must know which role it is, and the rule is deterministic.

**Default: you are the ORCHESTRATOR.** Any session a human starts is an orchestrator. There is normally
**one** orchestrator and it operates on **main**, where it owns all of `coordination/` and the single
dashboard, talks with the user, expands queued tasks into briefs, and **spawns worker subagents** to
execute them. It plans and integrates; it does not do the deep task execution itself, and it is the only
role that touches `reports/research_report.md`. A *direction* is an organizing axis inside
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
math **in the dashboard** for KaTeX errors; and **review the training
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
- The portable idea for the next project: *freeze the ground truth* — one versioned, tested module for the
  domain's data-generating process, imported unchanged, forking forbidden.

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
   one). The proposed task's `note` is only a seed for that decision, not an executable spec.
2. The orchestrator **expands that seed into a full contract** at `coordination/tasks/<id>.md` (objective,
   concrete experiments, deliverables, the schema-v2 manifest, definition-of-done, paths, KaTeX rules).
   **Then it surfaces a short contract for approval before spawning** (unless in `hard` mode — below):
   post a compact contract to the **Inbox** (`coordination/decisions/`, kind `contract`) — a few bullets the
   user can review in seconds (what seam it replaces, what it tests, the deliverables, and **explicitly what
   it will NOT do**), with the full brief a click away. The user Approves, Adjusts, or Rejects. This is where
   scope mismatches get caught cheaply ("this only learns the stress, not the whole update") instead of after
   a long run. On approval the orchestrator **spawns a worker matched to the task's `effort` tier**
   (quick/standard/deep → model + reasoning effort + how long to persist) and flips the task to **active**.
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
   regenerated, not shipped. It then **adds at least one short, standalone training page** in the objective
   textbook voice (`spec/style_training_report.md`) — over-including math **prerequisites** (linear algebra,
   calculus, numerics) it leans on and making sure every `[[link]]` it writes points at content that
   actually exists — **fires a `finished` ping**, and exits, leaving everything **on disk** (it does not
   commit). The manifest carries a tight **`summary`** (shown by default) and a **`full_report`** (the
   detail, behind an expander) — see "Presenting results to the user" — and any comparison against a
   baseline/ground truth shows **both sides against each other**, ground truth mandatory.
4. The orchestrator **reviews, commits, and surfaces** it on the dashboard. **Done is the user's call**,
   made after discussion — never set automatically.

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
reports/         training/ (multi-file ground-up textbook) + research_report.md (shippable)
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
