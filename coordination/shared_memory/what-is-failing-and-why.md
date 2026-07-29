# What is failing in this project, and why — Sevan's assessment, 2026-07-28

Durable record of the user's own critique, written down because it must shape how every future task and
worker behaves. Read this alongside `working-with-sevan.md`.

## The verdict, in his words (paraphrased closely)

> "So far this project has mostly been a failure. I have not been able to keep up with the actual details
> of the implementation, the harness is perpetually incomplete, the training report feels like it only
> half-teaches me and leaves me more confused afterward than before. I don't feel deep enough grasp to even
> understand what I'm asking half the time, and I distrust all the outputs because I can't be sure that
> it's really implementing what I wanted or expected."

Counted as successes by him: **the dashboard** ("functional and more useful than I even expected"), the
**queue/execute lifecycle** ("smooth and does work"), and **per-task pages**.

## The root diagnosis

**The project optimized for producing results, not for transferring understanding to Sevan.** Goal #1 in
`CLAUDE.md` is his learning, but essentially every mechanism built — evidence discipline, contracts,
manifests, honest-partial reporting — makes outputs **honest**. Almost nothing makes them **legible to a
non-expert who is trying to become an expert.** Honest and teaching are different targets.

The distrust is *downstream* of the comprehension gap, not a separate problem. If he cannot read the code,
and the textbook does not teach him to read *this repo's* code, then every result is a claim he must take
on faith. **Fixing legibility fixes trust. Adding more verification rituals does not.**

## What this means for how work is done here

- **A result nobody can independently check is worth little to him**, however rigorously it was produced.
  Prefer showing the check (a finite-difference gradient verification, a ground-truth overlay) over
  asserting the conclusion.
- **Training pages must teach him to read the actual code in `sim/`**, not generic MPM theory. Cite real
  files and real functions he can open.
- **Do not add harness features when the harness is already the most finished part.** The stopping-rule
  problem is real: every session adds instead of declaring done. New harness work needs a reason beyond
  "it would be nice".
- **Task pages are the weak point.** They "do not convey enough information visually and clearly, and it
  took a lot of pushing from me to even get to this point."

## The steer: generative UI for task pages

His stated direction, not yet built. Today a task can supply a `custom_html` card, which he considers "too
restricted". What he wants:

> "Why not give each task the ability to generate custom code and visualization specific to that task, in
> order to deliver information in an optimal way with actual reasoning and intelligence going into it?"

So: a worker should **design the presentation of its own result** — bespoke code and visualization chosen
for what that particular result needs — rather than filling in a fixed card schema. Treat the manifest
schema as a floor, not a ceiling.

## On burning it down

He raised a **controlled burn** ("sometimes you need a controlled burn... I spend forever building the
house, just to come and knock it down"). The position taken on 2026-07-28, which he accepted: **burn
narrowly, not broadly.** Keep the dashboard, the server, the task lifecycle, the frozen physics library and
its signatures — those are green and expensive. Burn only what is actually rotten:

1. **The training report** — done 2026-07-28. It was an anthology (one page per task, by 18 different
   workers, four of them narrating one evolving question). Rewritten as a curriculum; the spec rule that
   caused it (`Granularity — prefer many short pages`) was inverted. See
   `spec/style_training_report.md`.
2. **The `directions` abstraction** — he has called it: directions become **tags**, tasks form a **graph**
   via follow-up links, and a "direction" is just a lineage in that graph. **Not yet implemented.**

## Still outstanding as of 2026-07-28

Asked for, not yet done — all deferred for time, none rejected:

- **Task graph + tags migration.** Undo `directions` as a container; merge duplicated/overlapping tasks;
  rebuild the follow-up links deliberately ("I did it very messily"); pick **just a few** brief, meaningful
  tags (reusing old direction names where they fit) and allow multiple per task.
- **The research report** (`reports/research_report.md`) — "basically empty and hardly updated".
- **Generative UI for task pages** — the steer described above.
