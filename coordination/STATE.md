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

## Next (NOT started — await the user's go)
- **Bigger test: two worker subagents on bigger tasks, in parallel.** Make the evidence/hypothesis/table
  changes first, report, then launch.
- Optional later: cold-start orchestrator test (a fresh session orchestrates from the filesystem alone).
