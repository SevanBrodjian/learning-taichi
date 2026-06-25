# STATE — orchestrator working memory

> At-a-glance state, reconstructable from the repo but kept here for convenience. Read `CLAUDE.md`,
> `spec/`, and `coordination/` for the full picture. _Updated 2026-06-25._

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
- Optional later: cold-start orchestrator test (a fresh session orchestrates from the filesystem alone).
