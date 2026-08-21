# Epoch 1 — the first demo. Exam prompt.

> This is what your report has to cover. It is **not** an outline to fill in — how you structure it is
> part of what is being marked. Graded against `spec/examination.md` by an agent that has not seen our
> working conversation.

## The question
**What did you actually learn building a differentiable-simulation project, and what does the evidence in
this repo actually support?**

Twenty-seven tasks ran. The report is not a tour of them. It is an argument about what is now known,
what is merely suggested, and what turned out to be wrong.

## What it must engage with
Not section headings — things the report has to have a defensible position on.

1. **The demo.** Four materials, one grid, real time, in a browser. What makes that possible, and what
   is it actually costing? Be specific about where the frame goes.
2. **At least two results that overturned something**, including what was believed before and what the
   evidence was. This project has several. Finding them is part of the exam.
3. **At least one thing that failed, or did not work, or remains unknown** — stated as such, not as a
   footnote. A report where everything worked is not a report about this project.
4. **The learned-dynamics arc.** Networks were put inside the solver repeatedly. What is the honest
   conclusion about when that is worth doing?
5. **Where the ground truth lives and why that mattered.** This one is about method, not physics.

## What will sink it fastest
From the rubric, in the order they are most likely to bite:
- **Generalising from one task.** The single most common failure. If a result came from one scene on one
  GPU, the report has to say so.
- **Quoting a metric whose definition you have not read.** Automatic revise. The registry is
  `spec/registry/metrics.json` and at least one metric in this project does not mean what its name
  suggests.
- **Describing without explaining.** A correct account of what happened, with no mechanism, is a revise.
- **Restating a training page.** The textbook is the material, not the answer.

## What you can ask me for
Which section is weakest. Whether a claim is scoped correctly. What a metric actually means (you will get
the registry entry to read). What to study.

**Not**: what you are missing, whether a claim is right when you have not tried to support it, or
anything whose answer is the report. I will decline those in one sentence and point you at what to read —
that is the deal, not obstruction.

## Where it goes
`reports/research_report.md`. Length is yours to judge; the rubric says nothing about it, and a tight
report that clears all five criteria beats a long one that does not.

## Epoch 1 closes when this passes
Not before. The frozen demo build, the physics version and the task graph get cut at the same moment.
