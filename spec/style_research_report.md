# Style: research report

> The long-prompt for `reports/research_report.md` — the conservative, shippable deliverable. Distinct
> from training reports (which teach me) and from running logs (which track work). Co-authored.

## Audience & purpose
External readers — portfolio/website visitors and peers. Communicates the **core principles and
results** worth shipping: what was built, what it shows, and why it's interesting. Evolves **slowly and
conservatively** — it captures durable conclusions, not day-to-day progress.

## Structure (default)
1. **Problem & framing** — the question, situated in the structured-generative-worlds vision loosely (the paper shouldn't fixate on this).
2. **Approach** — the method at a level a peer can follow; link to training reports for depth.
3. **Results** — figures, key plots, the demo; concrete and honest.
4. **Significance** — what it demonstrates; what's genuinely impressive.
5. **Limitations & future work** — candid.

## Voice & framing
The tone should be primarily academic. For most projects it can be something like one or two steps down from a proper conference submission in terms of formality. Meaning we can take more liberties with informal writing, explaining things and assuming less reader knowledge (possibly non-technical), and even being more lighthearded and non-serious. In general I dislike how cold and uncaring most academic papers are, like the universe they're written about. Since this is personal work unlikely to be submitted anywhere, our reports can be more engaging, self-aware, humorous (I prefer dry, wry humor). Nothing over the top and cringy.

Use confident, clear, concise, precise writing style. Avoid complicated sentence structure, avoid em dashes and semicolons, avoid colons unless clearly beneficial.
The first sentence of each paragraph should introduce the point or claim of that paragraph. The subsequent sentences should then argue, support, justify, elaborate, or discuss that point.
Do not use uniform paragraph lengths. The paragraphs should vary naturally in their lengths to accommodate the content of each paragraph and natural break-points.
Avoid starting a paragraph with a sentence that refers back to a point or content from a preceding paragraph, like "this shows" or "We build off this idea..."
Avoid generic phrasing and LLM language. In particular, avoid any phrasing along the lines of "we ask a simpler question" or "our method asks whether" or all these variants of "asking". LLMs dramatically overuse this. Additionally, avoid overuse of colons. LLM writing really overuses colons followed by lists or claims. Just rework writing to avoid them.

Here is a sample of my writing style from an abstract I wrote for a CVPR research paper:

"Sonar is often the only modality suitable for high-resolution imaging underwater due to light attenuation and turbidity. Forward-looking imaging sonar provides measurements over range and horizontal angle but collapses vertical structure into a flat image, creating ambiguities that make 3D recovery challenging. A common use case for imaging sonar is underwater terrain mapping (bathymetry), yet current methods require many views, expensive multi-sensor setups, or significant training data, which limits use and adaptability to new environments. 

We present a training-free method that recovers bathymetry from a single sonar image in under 30 seconds via differentiable rendering, conditioned on a known seafloor tilt. To our knowledge, this is the first differentiable rendering approach for single-view height recovery in sonar. Our method implements differentiable sonar ray tracing and optimizes an explicit height field to reproduce the target image. On synthetic datasets, our approach outperforms a supervised CNN under distribution shift and remains close on rough terrain, while the CNN wins in-distribution. By modeling physically grounded priors of the sonar process, our method adapts across sensor configurations and environments without training data."

## Length & format
- Tight; every figure and claim earns its place. Markdown + KaTeX; Pandoc → PDF for a polished version.
- We should target a PDF release as well. Goal is roughly 4 pages for smaller projects, 8 pages for larger projects. Not significantly longer than that unless it's a very deep and evolving project.

## What this is NOT
- Not a changelog or running log (those live in `agents/<branch>/LOG.md`).
- Not a tutorial (that's `reports/training/`).
