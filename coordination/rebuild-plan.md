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

**Priority — revised 2026-08-01.** B *was* the one to cut when it was only a data migration. It is now the
front-end pass and carries **B5, the Demo page**, which is the only artifact intended for an audience
beyond Sevan. So: **B and C are the ones that matter**, and within B the Demo is the point. Track A is
done and locked.

- [x] **Track 0 — Training report** (2026-07-28: 25 pages/3,499 lines → 20/3,019; anthology → curriculum;
      the spec rule that caused the sprawl inverted). Ongoing: Sevan reads it. See "The reading loop" below —
      the mechanism matters more than the remaining trims.

---

## Track A — Generative UI for task pages *(~1h of his time)* — **EXECUTED 2026-08-01, awaiting review**

**Decision taken:** bespoke page *leads*, evidence collapses beneath it (option 1).
**Shipped:** `spec/style_task_page.md`; TaskView restructured + self-sizing frame; exemplar rebuilt at
`runs/material-variants/train-one-nn-to-mimic-viscosity-and-st/bespoke_page.html`; wired into
`_TEMPLATE.md` and `CLAUDE.md` (both worker and reviewer sides). Commit `128531a`.
**Open:** Sevan's review of the exemplar — does it beat the old page enough to justify the standard?

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

## Track B — Layout, aesthetic, graph/tags, and the Demo page *(re-scoped 2026-08-01 — no longer the one to cut)*

Originally a cosmetic data migration. Sevan expanded it into the **whole front-end pass**, including the
project's flagship deliverable. Design direction for all of it: **`spec/aesthetic.md`** (read it first).

### B1 — Layout: tabs, not a global sidebar  ✅ DONE 2026-08-01
The sidebar is dead weight — **only the Tasks page uses it**; everywhere else it sits as a blank spacer.
- Replace it with a **browser-style tab strip across the top**. Each page then owns the full width beneath.
- The **Tasks page keeps its own dedicated sidebar**, scoped to itself.
- **Scrolling discipline:** the app shell never scrolls. The tab strip is always fixed. Each page scrolls
  its own content independently, *and only where scrolling makes sense* — the Map, for one, should not
  scroll normally.

### B2 — Aesthetic pass on the dashboard *(mild)*  ✅ DONE 2026-08-03
Currently *"leans a little too sterilized AI design."* Apply `spec/aesthetic.md` at **low intensity** —
character and intent, not decoration. It must stay a working tool readable on an iPad.

### B3 — The Map page: full redesign  ✅ DONE 2026-08-03 (crash fix UNCONFIRMED — see below)
*"Clunky and not very useful."* Aesthetic **and** functional redesign, not a restyle. Sevan's specifics
(2026-08-03), captured verbatim enough to act on later:

**Defects**
- **It crashes the whole PWA when scrolling on that page, sometimes.** A hard bug, not a polish item —
  investigate before redesigning on top of it.
- Tasks *"aren't cleanly separated"*; the layout is *"too blocky"*; it *"feels fragile"*.
- **Lines overlap blocks** — edges route through nodes instead of around them.

**What it is missing (the real problem)**
- *"It doesn't feel like I understand what the use case is when I open it."* The page has no stated job.
- It should **show the lineage of the work like a good story** — the arc, not a hairball.
- It should **surface connections he would have missed**, and **promote ideas** rather than only record them.
- **Edge semantics:** consider *different kinds of arrows*, and *labels on arrows* — a follow-up that
  refutes its parent is not the same edge as one that extends it, and the graph currently cannot say so.
- Tagging *"doesn't add anything on that page"* yet — **correctly blocked on B4**, which is why B4 goes first.

### B4 — Tagging and task organization  ✅ DONE 2026-08-03
The tagging/organization system *"needs fixing"*. Directions stop being containers and become **tags**;
a direction is just a lineage in the follow-up graph.
**DECIDED 2026-08-03 — execute against this, do not re-ask.**

- **Tags: five** (was four), multi-tagged per task. `demo` added 2026-08-16.
  | tag | what it means | tasks |
  | --- | --- | --- |
  | `gradients` | gradient health / optimizing through the rollout | 8 (incl. both proposed) |
  | `materials` | constitutive models and material physics | 9 |
  | `learned` | a network replaces part of the physics | 6 |
  | `rendering` | the visual pipeline | 4 |
  | `demo` | work aimed at the shippable interactive Demo (see `coordination/demo-mvp.md`) | 3 |

  The learned-material chain carries **both** `learned` and `materials`. Rejected: a separate `control`
  tag (only 2 tasks) and a second `kind` axis (two taxonomies to keep current).
- **Merging: conservative.** Merge only genuine duplicates/superseded pairs — realistically just
  `generalize-one-nn-across-viscosity-and-surface-tension` into `train-one-nn-to-mimic-viscosity-and-st`,
  which told the same story twice. Everything else stays a distinct task and is organized by tags + graph
  edges. **Rationale: merging tasks means touching `runs/<direction>/<task>/` paths, which every manifest
  and every training-page media URL depends on.** Keep that risk at zero.

### B4b — Linking becomes the orchestrator's job, not Sevan's  ✅ DONE 2026-08-03

> *"I'd rather not have to worry about it... I propose a new task, maybe I can reference previous tasks
> (explicitly, with a prompt suggester or something of that nature), but then it's left up to the AI (you)
> to slot that task into the graph so that I build a task network naturally."*

The current edges are **arbitrary, with missed connections** — they record whatever was typed at creation
time, which is why the graph does not read as a lineage. Redesign:

1. **The user proposes a task and may optionally cite prior tasks.** A **suggester** in the create/propose
   UI surfaces likely-related existing tasks as he types, so citing one is a click, never a lookup. Citing
   is a *hint*, not the final edge set.
2. **The orchestrator places the task in the graph.** On creation (and on review), it derives the real
   `follow_up_of` edges from what the task actually is — the seed text, the objective, and once it has run,
   its findings. Sevan never maintains links.
3. **Edges carry a kind.** Pairs with B3's ask for different arrow styles and labels:
   `extends` · `re-does` (supersedes a flawed run) · `refutes` · `applies` (borrows a method) ·
   `prerequisite-of`. A follow-up that overturned its parent must not look identical to one that built on it.
4. **Re-derive the whole existing graph once**, not just new tasks — that is what fixes "arbitrary and
   missing connections" across the 21 tasks already on the board.
5. **Every derived edge is reviewable and overridable.** The orchestrator proposes; Sevan can delete or
   re-point any edge from the Map. Automation that cannot be corrected is worse than none.

**Sequencing:** this makes B4 the substantive half of the track and confirms B4 → B3. The Map cannot be
designed to "show lineage like a good story" until the edges *are* a lineage and carry kinds to draw.
- **Execute:** merge overlapping/duplicate tasks, rebuild `follow_up_of` / `follow_ups` as a deliberate
  DAG, apply tags, add tag chips + filtering/sorting.
- **Keep `runs/<direction>/<task>/` paths stable** — every manifest and every training-page media URL
  points at them. Renaming them is a separate, much bigger job; do not fold it in.

### B5 — The **Demo** page (new tab) — the flagship deliverable  ✅ PLACEHOLDER SHIPPED 2026-08-01
The reason this track matters. *"This project needs to result in something I can put on my website —
genuinely immersive and impressive and worthwhile. Right now we have no means to make one."*

- A **new top-level tab**, full-page, at **full aesthetic strength**.
- **Transplantable**: it must lift onto his personal website **separately from the dashboard**, with no
  dependency on the data server, the task registry, or the harness.
- **Standalone-legible**: written for a visitor who knows nothing about MPM or this project. **No jargon.**
  The dashboard may be offered as an optional thing to poke at; the Demo is the thing people come for.
- **Now:** a deliberately-designed *"no demo exists yet"* placeholder. It should read as a highly polished
  demo that happens to be empty — never as an unfinished page.
- High-res delivery (vector/procedural, sharp at any DPI); intentionally low-poly or low-sample components
  are welcome — see the resolution note in `spec/aesthetic.md`.

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


---

## Track B — closed 2026-08-03

All five items shipped. Numbers, so the before/after is checkable:

| | before | after |
| --- | --- | --- |
| edges | 11 | **24** |
| cross-direction edges | 0 | **5** |
| orphan tasks | 12 | **0** (3 genuine roots) |
| edges crossing a node body | 5 | **0** (measured by path sampling) |
| tags | 5 direction names | **4 real tags** filtering 9/10/6/4 |

**Left open, deliberately:**
- **The PWA scroll crash is fixed-but-unconfirmed.** Three real faults were found and repaired (a
  null-deref inside a setState updater, an unguarded `setPointerCapture`, and a passive wheel handler
  that both failed to `preventDefault` and queued a re-render per tick). None could be reproduced here —
  it happens on Sevan's iPad. If it recurs, that is genuine new information, not a regression.
- **No one has seen the Map rendered.** Verification was geometric and behavioural (node/edge overlap
  sampling, event storms, filter counts), not visual.
- **Edge kinds are derived once, by hand, in `rebuild_graph.py`.** New tasks get `extends` as a
  placeholder until the orchestrator reviews them. Deriving kinds automatically from task content is a
  future improvement, not a shipped one.

**Next:** Track C (needs Sevan's decision on what the research report *is*), and the two minimal research
tasks — the only remaining items that produce new learning rather than better tooling.
