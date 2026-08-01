# Style: the task page

How a finished task presents itself on the dashboard. This governs the **bespoke page** every task
authors (`custom_html` in the manifest) and how it relates to the standard blocks.

The reason this spec exists, in the user's words: task pages *"do not convey enough information visually
and clearly, and it took a lot of pushing from me to even get to this point."* A task that produces a real
result and presents it badly has not finished.

## The shape of a task page

```
  Title
  Objective
  Summary            <- the tight, human-legible anchor (1-2 paragraphs)
  ─────────────────
  YOUR PAGE          <- the main event: whatever this result actually needs
  ─────────────────
  ▸ Evidence & detail  <- raw results grid, full findings, hypothesis, limitations
  From the textbook
```

The bespoke page **leads**. The standard blocks still exist underneath, collapsed, so the raw material is
always one click away — that floor is deliberate, and it means a bespoke page never has to re-present
everything. Show the *designed view*; let the expander carry the *complete record*.

A task with no bespoke page falls back to the old layout with its results grid shown directly. That path
still works, but it is the exception now, not the default.

## The core rule

> **Design the page around what this result needs a reader to understand. Do not fill in a schema.**

Before writing any HTML, answer in one sentence: *what is the single thing a reader must walk away
knowing?* Then build the page so that thing is unmissable, and everything else supports it.

That sentence is usually **not** "here are my outputs". It is a claim, a failure, a tradeoff, or a trap.

## Rules that have teeth

**1. Lead with the honest verdict.** One or two sentences at the top, in plain language, including the
part that did not work. A reader must not have to infer the result from a figure.

**2. Both sides, always. Ground truth is mandatory, never optional.** Any comparison — improvement,
change, learned-vs-true — shows both against each other. A learned output with no ground truth beside it
is not evidence.

**3. Match the medium to the claim.** A claim about *motion* is evidenced by *video*, not two final
frames. A claim about a *distribution* is evidenced by the spread, not the mean. If the claim is about
shape, show shape.

**4. Find the flip.** The strongest pages let a reader **switch between two states and see the finding
happen** — two metrics on the same cells, before/after, learned/truth, trained/held-out. A toggle that
changes the story is worth more than two static figures side by side, because the reader performs the
comparison rather than being told about it. *This is the highest-leverage move available; reach for it
first.*

**5. Beware the scalar that lies — and show what catches it.** If a summary number can look acceptable
while the result is actually wrong, that gap **is** the finding. Put the honest metric next to the
flattering one. (The exemplar exists for this reason: a held-out corner with a survivable-looking
trajectory RMSE of 0.24 whose shape was completely wrong, because a vertical spike and a settled blob
share a centre of mass.)

**6. A number reported without a picture of the quantity is not evidence.** If a claim rests on a value,
the page shows the thing that value measures.

**7. Interactive only when interaction earns it.** Sweeps, grids, parameter families, anything where the
reader wants to ask "what about *that* cell" — interact. A single comparison does not need a widget; a
clear static figure beats a fussy one.

**8. Scope on the page, not just in the manifest.** The limitation that matters most belongs *visible*,
near the claim it bounds — not only in the collapsed Limitations block.

## Technical constraints

The page renders in a **sandboxed iframe** (`sandbox="allow-scripts"`, no same-origin). Therefore:

- **Fully self-contained.** No CDNs, no external fonts, no `fetch`, no network. Inline your data as a JS
  literal, inline all CSS and JS.
- **Media by absolute path**: `/api/data/learning-taichi/runs/<direction>/<task>/<file>`. Relative paths do
  not resolve. Verify every one actually loads.
- **Height is automatic.** The frame is injected with a reporter that posts its content height up, so the
  page sizes itself. Do not build your own internal scroll box.
- **Prefer drawing from data over embedding images.** Building a grid or chart procedurally from numbers
  inlined out of `metrics.json` is smaller, sharper, and interactive. Base64 sprites are a fallback. (The
  exemplar went from 159 KB of embedded PNGs to 12 KB of data-driven HTML and gained a working toggle.)
- **Emit it standalone too**, as `bespoke_page.html` in the run directory, so it can be opened and reviewed
  outside the dashboard.
- Dark theme. Background `#0a0e14`, text `#dfe6ee`, muted `#7f8ea3`, accent `#6fd3ee`. Avoid red/green as
  the sole encoding.

## Before you ship it

- **Open the page and look at it.** Not the JSON — the rendered page. Click every control. This is not
  optional and it is the check most often skipped.
- Every media URL resolves (a dangling `src` is a broken page, and the reviewer rejects the manifest).
- The verdict at the top matches what the visuals actually show, including the failures.
- Someone who has not read the brief could look at this page and correctly state the result.

## The exemplar and the anti-example

**Exemplar:** `runs/material-variants/train-one-nn-to-mimic-viscosity-and-st/bespoke_page.html`. One
interactive 5×5 grid with a metric toggle; flipping it redraws the same 25 runs from "one bad corner"
(RMSE) to "the whole high-surface-tension region is broken" (shape error). The trap is called out
explicitly with the held-out video beside it. Built by `harness/tools/build_exemplar_page.py`.

**Anti-example:** the same task's previous page — a summary paragraph, then a grid montage PNG, then an
RMSE heatmap PNG, then a roundness heatmap PNG several scrolls apart, then an "Interactive" block at the
bottom. Every fact was present. The finding was invisible, because nobody would ever put the two heatmaps
side by side and notice they disagreed.

**The difference is not effort or polish. It is that one page was designed around the finding and the
other was a dump of everything produced.**
