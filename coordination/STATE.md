# STATE — orchestrator working memory

> At-a-glance state, reconstructable from the repo but kept here for convenience. Read `CLAUDE.md`,
> `spec/`, and `coordination/` for the full picture. _Updated 2026-06-28._

## Where we are
- The **harness is built and live**: portable `harness/` (dashboard, data server, tools), a Direction->Task
  data model, schema-v2 task manifests, one live dashboard on **main**, served as a fixed-port (5174) iPad PWA.
- **Orchestrator model (decided):** ONE orchestrator on `main` owns `coordination/` + the dashboard;
  parallelism comes from spawning several worker subagents (each worktree-isolatable), not from multiple
  orchestrators. Reconciled in `CLAUDE.md` -> Roles.
- **Filesystem is canonical** (not auto-memory). Collaboration norms in
  `coordination/shared_memory/working-with-sevan.md`.

## Directions (`coordination/directions/`)
- **differentiable-control** (active): `throw-to-target` (done), `optimizer-comparison` (active, awaiting the user's Done).
- **long-rollout-pathologies** (active): `nan-root-cause` (done) + proposed follow-ups (softened wall, checkpointing, jacobian norms, resolution memory).
- **material-variants**, **learned-dynamics** (proposed).

## Orchestrator->worker runs so far
1. **training-foundations** (worker, opus): rebuilt the textbook front (motivation group + from-zero prerequisites). Reviewed + committed.
2. **optimizer-comparison** (worker, opus): SGD/Adam/L-BFGS on the throw task, mass-stabilized. Reviewed + committed. L-BFGS wins on speed; all reach the same v0*. **Caveat (evidence discipline): this is ONE near-toy task** — speed/landscape claims are a hypothesis, not a general truth. The `1e-13` is the f32 loss floor; v0* differs from the baseline because the stabilized sim is a different (regularized) system.

## Standing corrections from Sevan
- **Evidence discipline (critical):** never overclaim from one task; scope to the evidence; test generality before asserting; separate observation/hypothesis/test. Codified in `CLAUDE.md`.
- **Tasks need a hypothesis/why section** (added to schema-v2 + the dashboard) that also seeds follow-up tasks.
- A worker should ideally test a question across **several tasks/conditions**, not one toy example.

## In flight (bigger test — launched 2026-06-25 overnight, 3 parallel workers)
- **optimizer-comparison** (re-run ACROSS several tasks, for generality — fixes the earlier overclaim),
  **softened-wall**, **resolution-memory**. All three `active`; workers write to disk and do NOT commit;
  the orchestrator reviews + commits each on completion. The GPU is shared by all three, so
  resolution-memory's timing/memory numbers may need a solo re-run.
- On each completion: scope-check claims (Evidence discipline), verify `hypothesis`+`limitations` present,
  render-check, commit the worker's files, leave status `active` for the user's Done call. Do NOT auto-mark
  done.
- **optimizer-comparison FAILED (2026-06-25 night):** worker came to rest with broken output — degenerate
  optimization (control stuck at [0,0], loss flat) and an incomplete manifest (1 of N tasks). Quarantined
  to `runs/differentiable-control/optimizer-comparison/manifest.failed.json` (task shows in-progress, no
  result). Cannot resume a subagent in bypass mode, so **RE-SPAWN it fresh once the GPU frees** (after
  softened-wall + resolution-memory finish); the brief now has a "Known failure to avoid" section
  (verify gradients flow + scope to ~3 tasks). Root cause: a control-parameterization bug + 3-way GPU
  contention. Lesson: spawning 3 GPU-heavy workers at once over-contends a single GPU.
- **UPDATE (later that night):** softened-wall + resolution-memory both finished, were reviewed +
  committed (both `active`, awaiting the user's Done; both honest with hypothesis+limitations). The
  optimizer task was **re-spawned SOLO** on the now-free GPU with bug-avoidance baked into the prompt
  (verify gradients move the control before sweeping; scope to ~3 tasks). It is IN PROGRESS — do NOT
  re-spawn it again; await its completion, then review + commit (leave active for the user's Done).
- **UPDATE 2 — the solo re-run ALSO stalled (no output written).** Same pattern: the worker backgrounded
  a long sweep and yielded to "wait", stranding itself (subagents do NOT reliably resume from their own
  background children, and I cannot resume a subagent in bypass mode). The committed single-task result
  remains in place (task `active`, reopened, shows the rework banner). **DO NOT auto-re-spawn — it fails
  the same way twice.** The optimizer multi-task comparison is OPEN for the user: re-brief it to run the
  sweep **inline** (no backgrounded long command, no yield-to-wait), or the orchestrator runs it directly.
- **Harness lessons (write into CLAUDE.md / the worker brief template next):** (1) cap concurrent
  GPU-heavy workers at 1-2, queue the rest. (2) Workers must run long sweeps **inline / in chunks** and
  must NOT spawn a long background command then end their turn to wait for it — that strands them.
- **CORRECTION (morning):** "UPDATE 2" above was WRONG. The solo optimizer re-run did NOT stall — it
  SUCCEEDED. I checked its run dir during a gap while its background sweep was still finishing, saw only
  old files, and prematurely declared it dead. It then completed a real 3-task comparison (throw-far /
  into-wall / split-field) and is committed (`1d4153e`): **L-BFGS wins only 1/3** (the smooth throw); Adam
  wins on into-wall (contact) and split-field (higher-dim), and "same basin" breaks on both — correctly
  overturning the original single-task overclaim. Status `active`, awaiting the user's Done. RETRACT the
  "workers must run sweeps inline" lesson (it was based on my error; the backgrounded sweep finished
  fine). The real lesson: **do not declare a worker failed until its background work is confirmed done** —
  a `came to rest` notification can fire during a gap. The 3-way-GPU-contention cap still stands.
- Optional later: cold-start orchestrator test (a fresh session orchestrates from the filesystem alone).

## Harness overhaul (2026-06-28) — 14-point feedback pass, all landed on `main`
Sevan reviewed the harness end-to-end and requested structural changes; all implemented this session and
committed to `main` (this work was done from a worktree session but edited `main`'s tree directly, since
the server, dashboard, and the next orchestrator all live on `main`). The stale worktree branch
`claude/elegant-bassi-cb7174` is behind `main` and is being retired — do not merge it.
- **Orchestrator-on-main is now concrete.** All edits land on `main`; the next session starts there as the
  orchestrator. Workers still get worktrees.
- **CLAUDE.md:** added "Orchestrator responsibilities — schedule, propose, ask" (scheduler-style worker
  concurrency, *not* strictly serial; propose tasks sparingly; ask via inbox on real forks). Rewrote
  Notifications so **workers** own start/finish pings.
- **Notifications:** `notify.py` now has `--kind started|finished|blocked|note|gate` (+ `--task`); workers
  fire two human, one-sentence pings; orchestrator pings sparingly. Legacy `--level` still works.
- **Specs:** training spec gained hard objective-voice rules (no first person, avoid second person, no
  transient/brief refs, standalone), an "explain every symbol/why" rule, "many short pages" granularity,
  and a visuals section. Research spec reframed: end target = technical paper, **current use = a ≤1-page
  evolving directions scratchpad** (graduates only at the user's direction).
- **Content:** rewrote `core/01-mls-mpm-forward.md` (added $V_p$, $A_p$ intuition, affine/offset roles,
  explicit $m_i$ accumulation) and `core/03-failure-modes.md` (de-logbooked into textbook voice). Fixed
  first-person/brief offenders in `prerequisites/01-mpm-in-context.md`. Reframed `reports/research_report.md`
  into the scratchpad form.
- **`/execute` skill** (`.claude/skills/execute/SKILL.md`): orchestrator burns down the whole queued
  backlog, scheduling concurrency intelligently, reviewing + committing each, leaving status `active`.
- **Worker brief template** `coordination/tasks/_TEMPLATE.md`: role stamp, two pings, evidence discipline,
  **visualization standard** (show the optimized quantity, e.g. plot center-of-mass vs target; grid+heatmap
  demos), required training-page contribution.
- **Dashboard authoring/editing:** server gained `POST /api/file` (edit any displayed `.md`),
  `/api/task-create`, `/api/task-edit`, `/api/direction-create`, and `mtime` on training sections. Frontend:
  `DocEditor` (Edit→raw-markdown textarea→Save) on every doc; Overview `+ Task`/`+ Direction` + task edit;
  training **"New" tag** (per-device localStorage read-tracking); **persist-place** in `App.jsx` so the iPad
  PWA resume keeps section/filter/open-task.
- **Verified:** `py_compile` clean; `npm run build` clean. Data server restarted so the new endpoints are
  live; Vite on 5174 HMR-reloads the frontend.

## Pending follow-ups (orchestrator todo)
- **Training "you"→impersonal sweep.** The explicit first-person/brief offenders are fixed, but generic
  second-person "you" is still pervasive across `reports/training/**` (motivation + prerequisites + core).
  Do a consistency pass to the new impersonal voice. Tracked here rather than as a board "direction"
  (it is orchestrator housekeeping, not a research task).
- Two queued tasks await `/execute`: `material-variants/fluid-vs-snow`, `learned-dynamics/learned-residual`.
- Fold the softened-wall finding into `core/03-failure-modes.md` open-questions if not already covered.
