# Style: the task page

How a finished task presents itself on the dashboard. This governs the **bespoke page** every task
authors (`custom_html` in the manifest) and how it relates to the standard blocks.

The reason this spec exists, in the user's words: task pages *"do not convey enough information visually
and clearly, and it took a lot of pushing from me to even get to this point."* A task that produces a real
result and presents it badly has not finished.

## The shape of a task page

```
  Title              [🗒 notes]   <- the user's own passive margin notes, rolled up
  TL;DR              <- ONE sentence. The punchline. Required.
  Objective
  Summary            <- the tight, human-legible anchor (1-2 paragraphs)
  ─────────────────
  YOUR PAGE          <- the main event: whatever this result actually needs
  ─────────────────
  ▸ Evidence & detail  <- raw results grid, full findings, hypothesis, limitations
  From the textbook
```

**`tldr` is a required manifest field.** One sentence, no jargon, stating what happened *including the
part that failed*. It exists so a reader scanning many tasks can triage without opening anything. "Trained
a network on three corners of a viscosity/surface-tension square; exact where trained, not physical
anywhere else" is a TL;DR. "Investigated conditioned material networks" is not — it says nothing.

The bespoke page **leads**. The standard blocks still exist underneath, collapsed, so the raw material is
always one click away — that floor is deliberate, and it means a bespoke page never has to re-present
everything. Show the *designed view*; let the expander carry the *complete record*.

A task with no bespoke page falls back to the old layout with its results grid shown directly. That path
still works, but it is the exception now, not the default.

## The core rule

> **Design the page around what this result needs a reader to understand. Do not fill in a schema.**

## Present findings, not conclusions

The reader is the researcher. **Your job is to make the evidence maximally legible so they can do the
reasoning — not to do the reasoning for them.** In the user's words: *"Feed me findings and results with as
clear of a presentation as possible, so that I am able to focus my efforts on the reasoning and
conclusions."*

In practice:

- **Show what happened; let the reader judge what it means.** "The high-viscosity column throws vertical
  jets instead of settling" is a finding. "The model fails to generalize because the capillary net is
  out of distribution" is a conclusion — it belongs in `hypothesis`, clearly labelled as a hypothesis, not
  asserted on the page as though it were observed.
- **Keep the three registers separate and visibly so:** what was *observed*, what is *hypothesised* to
  explain it, what would *test* that. This is the same discipline as `CLAUDE.md` → Evidence discipline,
  applied to layout.
- **Never let an interpretation replace the artifact it came from.** If you say the drop under-rounds,
  the drop is on the page, next to ground truth.
- **A verdict line is still required** (rule 1) — that is a *summary of what happened*, stated plainly,
  not an interpretation of why. Say "edge-exact at the trained corners, not physical elsewhere", not
  "this validates the conditioning approach".

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

**5. Show where your metrics disagree — and never let one scalar certify a result.** If two measures of
the same run rank it differently, that disagreement **is** a finding: put both on the page and mark the
cases where they diverge. A distance-style scalar is useful to *rank* and useless to *certify*, because its
scale usually is not interpretable — "0.12" tells a reader nothing without a reference. Prefer a metric
read directly against ground truth (which is interpretable) and show it beside the one that is not.

> **This rule earned its place by catching an error in this very spec.** Its first draft asserted the
> exemplar's held-out corner had a "deceptively low" distance metric because *a spike and a blob share a
> centre of mass*. That was inherited from the worker's manifest and was **false**: `traj_rmse` is a mean
> per-particle distance, not a centre-of-mass distance, and the held-out corner scored 0.246 against
> 0.012–0.031 at the trained corners — it screamed. The real finding, verified from the data, is that the
> two metrics correlate only moderately (Spearman $\rho \approx 0.55$) and that **two interior cells** pass
> the distance check while being badly wrong in shape. The wrong mechanism had already reached a task page,
> a training-report page, and this spec before anyone read the implementation. **Check the metric's
> definition before you explain a result with it** — see `spec/registry/README.md`.

**6. A number reported without a picture of the quantity is not evidence.** If a claim rests on a value,
the page shows the thing that value measures.

**6b. If it is selectable, selecting it must show it.** A grid, sweep or list the reader can click has to
reveal *that item's own evidence* — its frames, its clip, its numbers — not merely a readout. This has a
consequence for the run, not just the page: **export per-item media**, not only an aggregate montage. (The
exemplar can only show per-cell *stills* because the run exported one whole-grid video and no per-cell
clips; the stills had to be recovered from a previous page's embedded base64.)

**6c. Answer the obvious questions on the page.** For a task involving a learned component, "what is the
architecture and where does it sit?" is not a side detail a reader should have to dig for. Put a **compact,
high-level** version up front — a diagram of the pipeline marking which pieces are learned and which are
fixed, plus layer shapes — and leave the full specification to the Evidence layer. The same goes for any
metric the page reports: **define it, or link its definition.** Every metric label on the exemplar carries
its canonical definition from `spec/registry/metrics.json` on hover.

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

## Metrics come from the registry

`spec/registry/metrics.json` is the canonical list. **Check it before inventing a metric**, use registered names,
and register anything new in the same run that introduces it — with a real `source` file:line. The dashboard
renders these as hover definitions, and a bespoke page should do the same for every metric it displays.
Rationale and the full policy (including material/scene standardization) live in `spec/registry/README.md`.

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

## Verify the EMBEDDED page, not just the standalone file
`custom_html` is delivered to the dashboard through an iframe **`srcdoc` attribute**, and the HTML parser
**entity-decodes that attribute before the script inside it is parsed**. So a page can render perfectly
when you open `bespoke_page.html` from disk and be **completely blank** in the dashboard — opening the
standalone file proves nothing about the embedded copy. Anything that survives decoding differently
(`&`, `<`, `>`, quotes, and especially `&&`, `<`, `>` inside script) is a candidate.

**Open the task page IN THE DASHBOARD and confirm the content is actually there** before shipping. A
build step that `node --check`s the *decoded* script is the reliable version of this check.
