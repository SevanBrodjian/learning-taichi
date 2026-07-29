# Rebuild plan — three tracks, one hour at a time

**Written 2026-07-28.** The controlled burn, sequenced. Background and the *why* behind all of this is in
`coordination/shared_memory/what-is-failing-and-why.md` — read that first if you are picking this up cold.

## How to use this

Sevan's stated plan: **one chunk per session**, then spend the rest of that session reading the training
report. Tracks are ordered A → B → C but the *tracks* are independent — do them in whatever order appeals.
Chunks **within** a track are ordered and depend on each other.

**Do not start new research tasks until all three tracks are done** and the material feels solid. That is a
deliberate decision, not a backlog accident.

Each chunk is scoped to roughly an hour of focused work. When one is done, tick it here and commit.

- [x] **Track 0 — Training report** (done 2026-07-28: 25 pages/3,499 lines → 20/3,019; anthology → curriculum;
      the spec rule that caused the sprawl inverted). Ongoing: Sevan reads it; further trims as he finds
      gaps. Gaps he hits while reading are the highest-signal input this project has — capture them.

---

## Track A — Generative UI for task pages

**The problem.** Task pages "do not convey enough information visually and clearly, and it took a lot of
pushing from me to even get to this point." Today a task fills a fixed card schema and may attach one
`custom_html` block — too restricted. The goal is for each task to **design the presentation of its own
result**: bespoke code and visualization chosen for what that specific result needs, with real reasoning
behind the choice.

**Why it is first.** It is the most direct attack on the distrust problem. A result you can see properly is
a result you can check.

### A1 — Write the presentation standard *(1h, no code)*
Decide the rules before building the machinery, or workers will fill freedom with noise.
- Audit 3–4 existing task pages on the dashboard. Write down specifically what fails to land and why.
- Draft `spec/style_task_page.md`: what earns a figure vs a table vs a video vs an interactive; the
  mandatory both-sides comparison (ground truth is never optional); "if the claim is about motion, the
  evidence is motion"; when an interactive beats a static image; how much text before the first visual.
- Include 2–3 worked *examples* and at least one **anti-example** (a page that reports a number with no
  picture of the quantity).
- **Done when:** the spec exists and you could hand it to a worker cold.

### A2 — Build the runtime *(1h)*
- Extend the manifest schema + dashboard so a task can ship a **self-contained bespoke page** (its own
  HTML/JS/CSS), not just a card in a fixed layout.
- Sandbox, CSP, sizing, and a graceful fallback when the bespoke page fails to render — a broken custom page
  must never blank the task.
- **Done when:** one existing run renders through the new path end-to-end on the dashboard.

### A3 — Build the exemplar *(1h)*
- Rebuild **one** existing task page as a genuinely bespoke page, to set the bar. Suggested:
  `material-variants/train-one-nn-to-mimic-viscosity-and-st` — it has a 5×5 grid, a held-out corner, videos,
  and an honest partial result, so it exercises everything.
- **Done when:** the page makes the result legible at a glance, and it is linked from `spec/style_task_page.md`
  as *the* reference.

### A4 — Wire it into the worker contract *(1h)*
- Update `coordination/tasks/_TEMPLATE.md`, the `/execute` skill, and `CLAUDE.md` so every worker **designs
  its presentation** and self-reviews it against the standard.
- Add the presentation check to the orchestrator's review duties.
- **Done when:** a fresh worker would produce an A3-quality page without being pushed.

---

## Track B — Task graph, tags, and the dashboard

**The problem.** `directions` are a container that no longer matches the mental model. Directions are now
**lineages** — a direction is any connected set of tasks in the follow-up graph. Tags replace them. The
existing links were made "very messily" and several tasks overlap or duplicate.

### B1 — Design the migration *(1h, no code)*
- Read all 22 tasks. Produce a written plan: which tasks **merge**, which are duplicates, which are really
  extensions of another.
- Choose **just a few** brief, meaningful tags — reuse the old direction names where they genuinely fit
  (`materials`, `gradients`, `rendering`, `learned-dynamics` are the obvious candidates). Resist inventing
  a taxonomy; the test is whether a tag would ever be used to *filter*.
- Draft the intended follow-up graph (parents per task) on paper before touching JSON.
- **Done when:** a written migration plan exists that someone could execute mechanically.

### B2 — Migrate the storage *(1h)*
- Make tags + graph links canonical; retire `directions` as an organizing container.
- Keep `runs/<direction>/<task>/` paths **stable** — they are referenced by every manifest and every
  training page's media URLs. Renaming run paths is a separate, much bigger job; do not fold it in here.
- Migration script + a verification pass (every task reachable, no dangling parent ids, every run still
  resolves).
- **Done when:** the server serves the new structure and the existing dashboard still loads.

### B3 — Apply the merges and rebuild the graph *(1h)*
- Execute B1's plan: merge the overlapping tasks, rewrite `follow_up_of` / `follow_ups` as a deliberate DAG,
  apply tags.
- **Done when:** every task has considered parents and tags, and the graph reads as a real research lineage.

### B4 — Dashboard: graph view, tag chips, filtering *(1h)*
- Improve the Map/graph view so the lineage is actually legible (layout by lineage, not a hairball).
- Tag chips on task cards; filter and sort the Overview by tag; remove the direction-based UI.
- **Done when:** you can find any task by tag or by walking its lineage.

---

## Track C — The research report

**The problem.** `reports/research_report.md` is "basically empty and hardly updated." Nobody has decided
what it actually *is*, so nothing ever gets added with confidence.

### C1 — Define what it is *(1h, mostly thinking)*
The real question, and it is a taste call only Sevan can make. Options to decide between:
- a **shippable paper-like artifact** (conservative, defensible, for an outside reader), or
- a **findings ledger** (every scoped claim the project has earned, with its evidence and limits), or
- a **narrative of the research arc** (what was asked, what was found, what it means).

Then: who writes it (orchestrator only, per `CLAUDE.md`), when, and what bar a finding must clear to enter.
- Rewrite `spec/style_research_report.md` to match the decision.
- **Done when:** the definition is written and a section skeleton exists.

### C2 — Backfill it from the completed work *(1h)*
- Walk the ~18 done runs. Pull each defensible finding into the structure from C1, **scoped per evidence
  discipline** — "on this task, X", never "X is true". Several existing findings are already correctly
  scoped in their manifests and can be lifted nearly as-is.
- Flag honestly where the project has a hypothesis rather than a result.
- **Done when:** the report reflects where the project genuinely is today.

### C3 — Make it stay current *(1h)*
- Decide and encode the update mechanism so it stops going stale: a step in the orchestrator's review, a
  checklist item, or an explicit "does this change the research report?" gate at task close.
- Update `CLAUDE.md`.
- **Done when:** there is a rule that would have prevented the current emptiness.

---

## Backlog state while this is underway

Pruned 2026-07-28 so the board is clean to return to. **Two proposed tasks kept**, both about the gradient
itself and both named as open in the rewritten training report:

- **`jacobian-norms`** — estimate the per-step Jacobian norm by finite differences to find exactly where the
  rollout gradient becomes ill-conditioned. Measures what `core/03-failure-modes.md` currently only asserts.
- **`checkpointing-long-horizon`** — recompute instead of store, to reach 1024+ steps.
  `core/02-differentiating-the-rollout.md` explicitly calls this a queued direction, not yet a result.

**Two discarded** — `shape-match-materials` (material-variants is saturated at nine tasks) and
`residual-hard-mismatch` (legitimate, not urgent). Neither reasoning is lost: both notes were derived from a
completed task's **limitations** section, which is still on disk in `runs/material-variants/fluid-vs-snow/`
and `runs/learned-dynamics/learned-residual/`. Re-proposing either is a one-liner if it becomes interesting
again.

Nothing is `queued`, so `/execute` is a no-op until you deliberately queue something.

## Ordering note

A and B are independent. C benefits from being last (the report is easier to write once the task graph is
clean, and A/B will themselves generate findings worth recording). If motivation matters more than
efficiency, do the track you most want to see working.
