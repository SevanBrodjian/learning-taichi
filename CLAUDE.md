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

## Persistence — the filesystem is the backbone
Durable state lives in the repo, on disk, as files (mostly Markdown + JSON). **Do not rely on auto-memory
or session context for anything that must survive.** A worker's value is its output on disk, not its
living process, so a worker writes everything down and exits; the orchestrator reconstructs all state by
reading the filesystem. Everything learned, designed, decided, or instructed must land in: `spec/`
(calibration), `coordination/` (directions, tasks, decisions, shared_memory), `reports/` (training +
research), `runs/` (results), `agents/<branch>/` (status + log). If it is only in a chat context, it does
not exist.

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
1. The user **queues a task** (drags a proposed task to `queued`, or asks for one). The proposed task's
   `note` is only a seed for that decision, not an executable spec.
2. The orchestrator **expands that seed into a full contract** at `coordination/tasks/<id>.md` (objective,
   concrete experiments, deliverables, the schema-v2 manifest, definition-of-done, paths, KaTeX rules),
   **spawns a worker**, and flips the task to **active** (not user-undoable).
3. The worker executes and writes one polished task to `runs/<direction-id>/<task-id>/` — an **objective**,
   **scoped findings** (what was tested, no overclaiming), a **hypothesis** for *why* the result holds and
   what would test its generality, an honest **limitations** note, and typed results — extends the training
   textbook if warranted, and exits, leaving its work **on disk** (it does not commit).
4. The orchestrator **reviews, commits, and surfaces** it on the dashboard. **Done is the user's call**,
   made after discussion — never set automatically.

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
- `python harness/tools/notify.py --level progress|gate "<msg>"` (topic via `NTFY_TOPIC` env var, or a
  file outside the repo so every worktree sees it — see `harness/tools/notify.py`).
- `progress` = non-blocking FYI (run started, loss updates, report drafted) — emit freely.
- `gate` = needs the user (milestone decision, divergence, hard block) — use sparingly.
- Policy: run mostly autonomously, ping often, block rarely.
- **Anything that needs the user goes to `coordination/decisions/` (the inbox) and fires a `gate`** —
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
