# Style: training reports

> The long-prompt for how `reports/training/<topic>.md` is written. An agent should be able to follow
> this to produce a report you'd actually learn from. Co-authored — refine `TODO(you)` lines.

## Audience & goal
Written for **me** (see `researcher_profile.md`): Background in ML, generative modeling, and simulation with mixed depth. After
reading, I should be able to **explain the topic at a whiteboard**, give a presentation, and reuse the
concepts. Teach from first principles, but never condescend or pad. In general, it's preferred that you overexplain rather than underexplain. It's easier to skim past things I know than to have to go back and search to fill gaps.

## Structure (default; adapt as needed)
*This structure works well as a loose commitment. I like the idea of separating out background knowledge / pre-requisites from the core teaching material. That way we can accumulate both at the same time, and I can easily reference the pre-reqs (which should include a lot of the math) to skim what I know and dive into what I want to learn. That way I'm not doing this constantly while going through the core training document. We can lean into that idea even more with this report. Hyperlinking and organizing into sections is a much better idea than following a chronology specific to the order in which things were written. It should be much closer to something like a textbook, or smaller training document like that.*
1. **Motivation** — what problem this solves and why it matters to the world-models vision.
2. **Intuition** — the mental model before the math.
3. **The math** — derived, not asserted; KaTeX (`$...$`, `$$...$$`); define all notation.
4. **Implementation mapping** — tie the math to the *actual code* (reference `sim/...` files and line
   ranges); explain the Taichi idioms used.
5. **Failure modes & fixes** — what broke, *why* (mechanistically), and what resolved it. This section
   is mandatory and often the most valuable.
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

## Length & math
The length should be as long as needed. There's really no limit, and in some sense longer is better. The key is to grow it appropriately and slowly but steadily. If every task is adding 3 pages that may be overloading to try and keep up. But if many tasks accumulate steadily into a 20 page training document and I kept up along the way that's an excellent outcome in my eyes.

Mathematical rigor is preferred but used precisely. In general I struggle to feel engaged with the math unless I understand why I'm learning it. Also, don't hesitate to explain the principles needed before showing equations. I don't just want to learn what an equation for this system looks like, I want to learn the underlying mathematics and physics that explains where it came from and how someone could have arrived at that idea.

## Prerequisites (err on the side of more)
Bias toward over-including prerequisites. When an implementation leans on a piece of mathematics, teach
that mathematics from the ground up in the prerequisites layer, even if it feels basic. If a step uses an
SVD, a change of basis, a quadratic form, a Jacobian, or a numerical-integration scheme, add or extend a
prerequisite section that derives it and builds intuition, then link to it from the core. A standing goal
of this project is to strengthen my mathematical foundations across linear algebra, calculus, and
numerical methods, so depth in the prerequisites is a feature rather than padding. The prerequisites are
skim-friendly by design, so over-coverage costs me nothing and fills gaps I might not know I have.

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
