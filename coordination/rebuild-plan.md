# Rebuild plan — three tracks, ~1 hour of Sevan's time each

**Written 2026-07-28, re-scoped the same evening.** Background and the *why* is in
`coordination/shared_memory/what-is-failing-and-why.md` — read that first if picking this up cold.

## Budget reality

~4 weeks left at ~5 h/week ≈ **20 hours total**. The first draft of this plan spent 11 of them on harness
rebuild, which is over half the remaining project spent not learning anything. That was a mis-scope.

**The correction, and the thing to hold on to: Sevan's scarce resource is *attention*, not wall-clock.**
The orchestrator can execute; what it cannot do is make the taste calls. So each track below is split into
**decide** (his, minutes), **execute** (the orchestrator's, unbounded), and **review** (his, minutes). The
hour per track is *his* hour. Execution happens around it.

**Priority if time runs short:** A and C are worth it. **B is the least valuable** — merging and tagging
tasks is mostly cosmetic and can be pure execution with a 5-minute approval. Cut B first, not last.

- [x] **Track 0 — Training report** (2026-07-28: 25 pages/3,499 lines → 20/3,019; anthology → curriculum;
      the spec rule that caused the sprawl inverted). Ongoing: Sevan reads it. See "The reading loop" below —
      the mechanism matters more than the remaining trims.

---

## Track A — Generative UI for task pages *(~1h of his time)*

**What is actually broken.** Less than assumed. `custom_html` is *already* a sandboxed iframe with
`allow-scripts` + `srcDoc` (`harness/dashboard/src/components/TaskView.jsx:353`), so a task can already ship
arbitrary self-contained HTML/JS/CSS. Two real limits:
1. It renders as **one block at the bottom** of the fixed card stack — a footnote, not the presentation.
2. **Nothing instructs workers to use it** as the primary way to convey the result.

So this is a promotion-and-instruction job, not a runtime build.

- **Decide (20 min, his):** the presentation rules. What earns a figure vs a video vs an interactive; how
  much text before the first visual; whether the bespoke page should *replace* or *lead* the standard cards.
  The one non-negotiable already in `CLAUDE.md`: any comparison shows **both sides**, ground truth mandatory.
- **Execute:** promote `custom_html` to the primary slot with the cards as supporting detail; write the rules
  into `spec/style_task_page.md` + the task template + `/execute`; rebuild **one** existing task page as the
  exemplar (`material-variants/train-one-nn-to-mimic-viscosity-and-st` — 5×5 grid, held-out corner, videos,
  honest partial, so it exercises everything).
- **Review (40 min, his):** does the exemplar make the result legible at a glance? Does it beat the current
  page badly enough to be worth the standard? If not, the rules are wrong, not the runtime.

**Fold in here:** the training-report reading loop below. Same surface, same session.

---

## Track B — Task graph, tags, merges *(~30 min of his time — cut this first)*

Directions become **tags**; a "direction" is just a lineage in the follow-up graph. Links were made messily
and several tasks overlap.

- **Decide (5 min, his):** approve or edit a proposed tag list. Candidates, reusing old direction names:
  `materials`, `gradients`, `rendering`, `learned-dynamics`. Four tags, multiple per task. The test for a
  tag is whether it would ever actually be used to *filter*.
- **Execute:** propose the merge list from the 20 tasks, migrate storage so tags + graph links are canonical,
  rebuild `follow_up_of` / `follow_ups` as a deliberate DAG, apply tags, add tag chips + filtering to the
  Overview. **Keep `runs/<direction>/<task>/` paths stable** — every manifest and every training-page media
  URL points at them; renaming is a separate, much bigger job.
- **Review (25 min, his):** does the graph read as a real research lineage?

The existing Map view already exists; improving its layout is optional polish, not part of the hour.

---

## Track C — The research report *(~1h of his time)*

`reports/research_report.md` is "basically empty and hardly updated" because nobody decided what it **is**.
This is the track most likely to matter in four weeks, since it is the shippable deliverable.

- **Decide (30 min, his — the real work of this track):** what is it?
  - a **shippable paper-like artifact** (conservative, defensible, for an outside reader), or
  - a **findings ledger** (every scoped claim the project earned, with evidence and limits), or
  - a **narrative of the research arc** (what was asked, what was found, what it means).

  Then: what bar must a finding clear to enter, and when does it get written.
- **Execute:** rewrite `spec/style_research_report.md` to match; backfill from the ~18 done runs, scoped per
  evidence discipline ("on this task, X", never "X is true"); encode an update rule in `CLAUDE.md` so it
  stops going stale.
- **Review (30 min, his):** does it honestly represent where the project is?

---

## The reading loop — the point-of-confusion problem

Sevan's objection, which is correct: *"Fixing the training report after I've already read the bad version
isn't very satisfying because then I just have to re-read, again."*

**A note that becomes a fix he must re-read is a debt, not a fix.** The loop has to resolve confusion at the
moment it happens. Design, in priority order:

1. **Answer at the point of confusion.** Marking a spot should produce a **targeted explanation to him**, not
   just a ticket. That unblocks the read immediately and means there is nothing to re-read — the page fix is
   a *byproduct*, not the payoff.
2. **Anchor the note.** Select text → "you lost me here" / "explain this", written to disk with the paragraph
   anchor and his words. Specific beats general by a mile: *"I didn't know what $C_p$ was by the time it got
   used"* is fixable in minutes; *"chapter 4 was confusing"* is not.
3. **Make re-reading a delta, never a re-read.** When a page changes materially, flag **only the changed
   sections** as "updated since you read this". He should never be asked to re-read a page he has read.

**Available tonight with zero infrastructure:** he is reading while the orchestrator is live, so pasting
confusions straight into chat gets an immediate answer. The dashboard mechanism just makes that durable,
anchored, and usable when nobody is live.

---

## Backlog state while this is underway

Pruned 2026-07-28. **Two proposed tasks kept**, both about the gradient itself and both named as open in the
rewritten training report:

- **`jacobian-norms`** — estimate the per-step Jacobian norm by finite differences to find where the rollout
  gradient becomes ill-conditioned. Measures what `core/03-failure-modes.md` currently only asserts.
- **`checkpointing-long-horizon`** — recompute instead of store, to reach 1024+ steps.
  `core/02-differentiating-the-rollout.md` explicitly calls this a queued direction, not yet a result.

**Two discarded** — `shape-match-materials` (material-variants is saturated at nine tasks) and
`residual-hard-mismatch` (legitimate, not urgent). Neither reasoning is lost: both notes came from a
completed task's **limitations** section, still on disk under `runs/material-variants/fluid-vs-snow/` and
`runs/learned-dynamics/learned-residual/`. Re-proposing either is a one-liner.

Nothing is `queued`, so `/execute` is a no-op until something is deliberately queued.
