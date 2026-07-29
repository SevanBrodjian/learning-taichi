# Style: training reports

> The long-prompt for how `reports/training/<topic>.md` is written. An agent should be able to follow
> this to produce a report you'd actually learn from. Co-authored — refine `TODO(you)` lines.

## Audience & goal
Written for **me** (see `researcher_profile.md`): Background in ML, generative modeling, and simulation with mixed depth. After
reading, I should be able to **explain the topic at a whiteboard**, give a presentation, and reuse the
concepts. Teach from first principles, but never condescend or pad. In general, it's preferred that you overexplain rather than underexplain. It's easier to skim past things I know than to have to go back and search to fill gaps.

## Voice: objective and standalone (hard rules)
The training report is a **textbook, not a logbook.** It records what is understood and why it works, in an
impersonal, timeless voice, and every page must read as if it had always been part of the book. These are
hard rules, not preferences, and reviewing them is part of the orchestrator's job before any page is
committed:
- **No first person, ever; avoid second person.** Never "I", "we", "me", "us", "our", "let's" — a textbook
  does not say "we then divide by the mass", it says "the grid update divides by the mass". Strongly prefer
  impersonal phrasing over addressing the reader as "you"; a little generic "you" is a tolerable teaching
  device, but never personal ("your work", "the outcome you want here"). The target voice is fully
  impersonal. (This spec itself writes "me/my" to mean *you, the reader being taught* — that is calibration
  for the agent and must **never** leak into the report's own voice.)
- **No transient or meta references.** Never mention the task brief, "this run", the session, the
  experiment "we just did", a TODO, or "the question the brief left unexplained". The reader has none of
  that context. Refer to a result by the property it teaches — "differentiating through a 500-step rollout
  drives grid masses toward zero, so the $1/m_i$ term blows up" — not by the logistics of the run that
  produced it.
- **Standalone.** Each page stands on its own and links to prerequisites for anything it assumes. A reader
  opening it cold must never hit a dangling reference to context that lived only in a chat or a task file.
- **The vision connection stays, stated objectively.** Tie ideas to structured generative worlds and
  controllable simulation often, but as a property of the subject ("this is exactly the sensitivity a
  controllable world model has to expose"), never as "your work" or "what we want".

## Structure (default; adapt as needed)
*This structure works well as a loose commitment. I like the idea of separating out background knowledge / pre-requisites from the core teaching material. That way we can accumulate both at the same time, and I can easily reference the pre-reqs (which should include a lot of the math) to skim what I know and dive into what I want to learn. That way I'm not doing this constantly while going through the core training document. We can lean into that idea even more with this report. Hyperlinking and organizing into sections is a much better idea than following a chronology specific to the order in which things were written. It should be much closer to something like a textbook, or smaller training document like that.*
1. **Motivation** — what problem this solves and why it matters to the world-models vision.
2. **Intuition** — the mental model before the math.
3. **The math** — derived, not asserted; KaTeX (`$...$`, `$$...$$`); define all notation.
4. **Implementation mapping (brief, and only when it teaches)** — tie the math to the *idea* in code and
   the Taichi idiom that matters. Keep this light: name the file, not line ranges (they go stale and are
   task-specific), and skip it entirely when it would just be a code tour. Exact hyperparameters and
   one-run numbers belong in the run manifest, not here.
5. **Failure modes & fixes** — what broke, *why* (mechanistically), and what resolved it. This section
   is mandatory and often the most valuable. Teach the mechanism, not the debugging chronology.
6. **What's open** — honest limits, what's hard, what's an active research question.

**Top-level organization (the textbook groups).** The report is organized like a textbook into three
groups, each a folder of sections with stable anchors:
1. **Motivation & Background** — *currently the weakest part and the most important to fix.* Why this whole
   line of work matters, where it sits in the field, and the conceptual backstory a newcomer needs before
   any math. This is its **own group**, not a one-line preamble at the top of a core section.
2. **Prerequisites** — the standing math and machinery (see "Prerequisites" below), skim-friendly.
3. **Core** — the actual method, built up section by section. The numbered shape above (intuition → math →
   implementation mapping → failure modes → what's open) is the per-section template *within* Core.

**This is a teaching document, not a research log or a planning doc.** It records what is *understood* and
*why it works*, with honest open questions. It must **never** contain a worker's TODO list, a "things to
try next" list, or research planning — those belong in `coordination/directions/`. "What's open" means a
genuine intellectual limit that teaches something, not a backlog of experiments to run.

## Voice & rigor
- First-principles and concrete; prefer a worked example over abstraction.
- **Honest about hand-waving** — flag where an argument is informal.
- Use small diagrams/figures where a picture beats prose (reference images the run exported).
- Depth over breadth; no filler, no marketing tone.
- Use confident, clear, concise, precise writing style. Avoid complicated sentence structure, avoid em dashes and semicolons, avoid colons unless clearly beneficial.
- Avoid generic phrasing and LLM language. In particular, avoid any phrasing along the lines of "we ask a simpler question" or "our method asks whether" or all these variants of "asking". LLMs dramatically overuse this. Additionally, avoid overuse of colons. LLM writing really overuses colons followed by lists or claims. Just rework writing to avoid them.
- Tie ideas back to my core objectively strongly and frequently. If you are struggling to do this, then the connection probably isn't very clear, and I'm probably going to lose interest. Even if it's a negative connection, like "this idea does this, which is common and considered standard, but it probably will get overwritten by approaches that do this in your work".
- Write in an engaging format, not dense and cold technicality. It can be lightly humorous, although I prefer dry humor. Don't be cringy. If something is genuinely interesting to you, or you think there's a real connection to my core research, don't hold back. Let the enthusiasm shine through.

## Brevity and prioritization (the report must stay trackable)
The textbook is a corpus a person actually reads and keeps in their head, and it had started to balloon into
something hard to track. So the governing constraint is now **cohesion and concision**, not volume:
- **Lead with the key idea and its "why", then stop climbing.** Each page gets the main intuition across
  fast and offers deeper insight only where it earns its place. A reader should grasp the core of a page in
  the first screen; anything past that is genuinely additive depth, not padding.
- **Prioritize ruthlessly.** Most of the value is in a few ideas per page. Cut throat-clearing, restated
  context, and near-duplicate explanations that another page already owns.
- **Grow the corpus slowly and keep it organized.** Prefer **revising an existing page** to adding another
  one (see "Granularity" — this is now a hard default, not a preference). When a new result bears on a page's
  topic, rewrite that page to state the current understanding rather than bolting on a section or starting a
  sequel.
- **Implementation details and task-specific results have a *limited* presence.** Exact hyperparameters,
  code line ranges, this-run loss numbers, and one-off decisions live in the run manifest, not the textbook.
  A page teaches the timeless understanding; the run carries the evidence.

This reframes, but does not cancel, the "over-explain rather than under-explain" preference above: over-explain
the **math and the mechanism** (never leave a symbol undefined or a "why" unanswered), because those are cheap
to skim and expensive to reconstruct. Do **not** over-explain by logging implementation steps or restating
context. Depth in service of understanding, yes; length for its own sake, no.

## Math
Mathematical rigor is preferred but used precisely. In general I struggle to feel engaged with the math unless I understand why I'm learning it. Also, don't hesitate to explain the principles needed before showing equations. I don't just want to learn what an equation for this system looks like, I want to learn the underlying mathematics and physics that explains where it came from and how someone could have arrived at that idea.

## Explain every symbol and every "why" (skim beats lost)
When a formula appears, **define every symbol the moment it shows up**, and explain the intuition for
**each term**, not just the equation as a whole. A reader should never have to stop and wonder what a
letter means or why a factor is there. For every nontrivial expression, answer the obvious questions a
careful reader asks: *what is this quantity, where did it come from, why is it multiplied/divided here,
and why does this term make sense?* If a quantity is accumulated or defined elsewhere (for example a node
mass built up in an earlier sum), say so explicitly rather than leaving it implied. This is in deliberate
tension with conciseness, and the resolution is the reader's experience: **it is far cheaper to skim past
a sentence you already know than to get lost and have to reconstruct a gap.** When in doubt, explain it.
The forward-step core page is the bar to clear for this level of detail.

## Show what a parameter does, not just what it is
When a page introduces a **material or model parameter** (Young's modulus $E$, the Poisson ratio, the
timestep $\Delta t$, the grid resolution, a residual scale, a learning rate), define the symbol **and** show
its **effect** — how changing it changes the behavior. A definition without an effect ("$E$ is the
stiffness") has not really taught the parameter; the reader should come away able to complete "turn $E$ up
and ___" for the physics, the numerics, and, wherever it applies, the gradient/optimization landscape that
is the spine of this project. Prefer a **worked example or a small figure**: a plot of the response at two
or three parameter values, a before/after pair, or a short derivation of the scaling. A figure derived
directly from the model equations is legitimate and clearly honest so long as it is labeled as such rather
than dressed up as a measured sweep; a figure backed by a real run is better still. The bar to clear is the
`material-stiffness` core page, which follows one scalar into stress, wave speed, timestep stability, and
usable learning rate.

## Granularity — one topic, one page, revised in place
**Default to editing an existing page. Adding a new page is the exception and must be justified.**

This rule was previously the opposite ("default to adding a new small page"), and eighteen workers followed
it faithfully. The result was not a textbook, it was an **anthology**: twenty-five pages, each documenting
the task that produced it, four of them narrating one evolving question and each opening by summarizing the
previous three. It read as more confusing after than before, which is the exact failure this section now
exists to prevent.

The corpus is a **curriculum a person reads in order, once** — not a log of what was learned when.

- **One topic owns exactly one page, for the life of the project.** If a later task revises, deepens, or
  overturns what a page says, **rewrite that page** so it states the current best understanding, with the
  superseded idea kept only where the *reason it was wrong* is itself instructive. **Never add a sequel
  page.** "Page X ended on a prediction; this page tests it" is the anthology smell — if you catch yourself
  opening that way, you are editing the wrong file.
- **A new page is warranted only when the topic fits no existing page** — a genuinely new concept, not a new
  *result* about an existing concept. Before creating one, name the existing page it would otherwise extend
  and say why that does not work.
- **Write for a reader who has not read any task page.** A page may assume the pages *before it in the
  index*, never a run, a task id, or "the precursor". No chronology, no task numbers, no "this pass".
- **Corpus budget: the core spine should stay in the neighborhood of a dozen or so pages.** If adding one
  would push it well past that, the right move is to consolidate first and then add. A ballooning textbook
  is a defect to correct, not a sign of progress.
- **Splitting is still correct when a page genuinely holds two unrelated ideas** — but a long page telling
  *one* coherent story is fine and usually better than three short ones that each re-establish context.

Connect pages with stable anchors and hyperlinks ([[like-this]]), and make every link resolve to content
that actually covers what the referring sentence promises.

## Visuals — clear, simple, informative
A good picture often teaches faster than a paragraph, and training pages **may and should embed images,
diagrams, and short videos** (markdown image syntax; reference media a run exported under
`runs/<direction-id>/<task-id>/`). Prefer **clear, simple, informative demos over dense technical ones**:
an animation that shows the grid with a heatmap of mass or velocity as a step computes teaches the P2G/G2P
transfer better than any prose. Diagrams of data flow, annotated frames, and small before/after pairs all
count. Aim for the figure a good lecturer would draw, not an exhaustive technical plot, except where the
technical detail is the point.

## Prerequisites (err on the side of more)
Bias toward over-including prerequisites. When an implementation leans on a piece of mathematics, teach
that mathematics from the ground up in the prerequisites layer, even if it feels basic. If a step uses an
SVD, a change of basis, a quadratic form, a Jacobian, or a numerical-integration scheme, add or extend a
prerequisite section that derives it and builds intuition, then link to it from the core. A standing goal
of this project is to strengthen my mathematical foundations across linear algebra, calculus, and
numerical methods, so depth in the prerequisites is a feature rather than padding. The prerequisites are
skim-friendly by design, so over-coverage costs me nothing and fills gaps I might not know I have.

**Linear algebra is the running foundation and deserves the most generous coverage** — matrices as linear
maps, the determinant and trace and what they mean geometrically, the transpose and inverse-transpose,
symmetric and orthogonal matrices, eigen-decomposition, and especially the SVD and polar decomposition that
the solid constitutive models depend on. Give these their own standalone prerequisite pages rather than a
cramped aside.

**Never point a cross-reference at content that does not exist.** A `[[link]]` to a prerequisite is a
promise that the prerequisite is there and covers what the referring text says it does. If the core needs a
piece of math, write or extend that prerequisite page *first*, then link to it. A link to a not-yet-written
(or wrong) section is a review-stage defect, not a placeholder — it is exactly the kind of dangling
reference that must be caught before a page is committed. Before any page is done, confirm every `[[link]]`
in it resolves to a real section that genuinely covers the promised material.

## Notation conventions
> TODO(you): any standing preferences (e.g. matrices bold capital, particle index $p$, grid node $i$)?
> Default: state notation locally per report.

**KaTeX-safe math (hard rules — these already bit us twice).** The dashboard renders math with KaTeX,
which is stricter than full LaTeX. One bad token makes KaTeX render the *entire* expression as red
plaintext (the "red paragraph" failure), so these are not optional:
- Never write `\*`. It is undefined in KaTeX. For a star/superscript-star use `x^{*}` (literal) or `x^{\ast}`.
- Escape a literal dollar amount in prose as `\$` so `remark-math` does not open a stray math span.
- Brace every multi-character sub/superscript: `x_{T,p}`, not `x_T,p`.
- Stick to the KaTeX support table; do not assume a LaTeX-only macro exists. `\lVert \rVert`,
  `\operatorname{...}`, `\tfrac`, `\mathbf`, `\mathrel{...}` are safe and used here.
- **Multiline display math must use fenced `$$`.** `remark-math` v6 parses block math like a fenced
  code block: the closing `$$` must be on its own line. Writing `$$...content...$$` where `$$` ends
  mid-line is not closed — the parser swallows everything to end-of-document and KaTeX renders it all
  red. Always use the three-line form:
  ```
  $$
  ...equation...
  $$
  ```
  Single-line `$$f(x)$$` (open and close on the same line) is fine. This bit `01-mls-mpm-forward.md`.
- Render-check every new math-bearing report in the dashboard before calling it done — KaTeX errors
  are silent in the source and only visible when rendered.

## Before a page is done (self-review checklist)
Run this pass on every new or edited page before handing it off. Each item here has bitten a real page.
- **If you created a new page, justify it.** Name the existing page this material would otherwise have
  extended, and why extending it was wrong. If you cannot, you should have edited that page instead. A page
  that opens by summarizing another page is a sequel and must be merged into its parent (see "Granularity").
- **Render-check the math** in the dashboard (or headlessly through KaTeX). One bad token turns a whole
  expression into a red paragraph, and it is invisible in the source.
- **Look at every figure and video you embedded.** Open the actual file. Confirm it shows the quantity the
  text claims, has readable axes and labels, and has no degenerate or clipped output. A caption is written
  as plain prose (it renders as visible caption text, so no `$math$` inside it).
- **Every `[[link]]` resolves** to a real section that covers what the referring sentence promises. No
  dangling or wrong cross-references.
- **The prerequisites it leans on exist.** If the page uses a piece of math, the prerequisite page for it
  is written and linked, not merely promised.
- **Voice is impersonal and standalone** — no "I/we/this run/the brief", readable cold, per the hard rules
  above.
