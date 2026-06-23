# reports/ — what the agents write for you

Two distinct kinds of document (both Markdown + KaTeX so they render in the dashboard):

- **`training/<topic>.md`** — ground-up teaching docs written *for you*, so you can fully explain any
  automated work (presentations, discussion, reuse). Style governed by
  [`spec/style_training_report.md`](../spec/style_training_report.md). Usually a worker's final step.

- **`research_report.md`** — the conservative, slow-growing **shippable deliverable**. Captures core
  principles and results for external communication — *not* a running log. Style governed by
  [`spec/style_research_report.md`](../spec/style_research_report.md). Created in Phase 2.

Polished PDF export (e.g. of the research report) is done later via Pandoc.
