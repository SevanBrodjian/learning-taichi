# Course 1 — Differentiable simulation, and getting it into a browser

> The assessment for Epoch 1. Graded by the **`grader`** agent, an independent auditor that has not seen
> the working conversation. Rules: `spec/examination.md`. Assessment artifact:
> `reports/research_report.md`, hand-written by Sevan alone.

## Scope
**This is the first course, so its scope is the whole project to date: T-001 … T-027**, from
`throw-to-target` (23 Jun 2026) to `incorporate-improved-materials-on-real-demo-page-and-improve-polish`
(20 Aug 2026). Twenty-seven tasks, plus the training textbook and the registries as they stand at the cut.

Future courses will **not** look like this. Each subsequent course covers only the work since the previous
epoch closed, and re-examining an earlier course's material is a defect in the grading. This one is broad
only because it is the first.

## The question
**What did you actually learn building this, and what does the evidence in the repo actually support?**

The report is not a tour of twenty-seven tasks. It is an argument about what is now known, what is merely
suggested, and what turned out to be wrong. How you structure it is part of what is marked — this list is
not an outline to fill in.

## What it has to have a defensible position on
1. **The demo.** Four materials, one grid, real time, in a browser. What makes that possible, and what
   does it cost? Be specific about where the frame actually goes.
2. **At least two results that overturned something** — what was believed before, and what the evidence
   was. This project has several. Finding them is part of the exam.
3. **At least one thing that failed, or did not work, or is still unknown**, stated as such rather than as
   a footnote. A report in which everything worked is not a report about this project.
4. **The learned-dynamics arc.** Networks were put inside the solver repeatedly, across several tasks.
   What is the honest conclusion about when that is worth doing?
5. **Where the ground truth lives, and why that mattered.** This one is about method, not physics.

## What will sink it fastest
In the order they are most likely to bite:
- **Generalising from one task.** The most common failure by far. One scene on one GPU is one scene on
  one GPU, and the report has to say so.
- **Quoting a metric whose registered definition you have not read.** At least one metric here does not
  mean what its name suggests. The registry is `spec/registry/metrics.json`.
- **Describing without explaining.** A correct account with no mechanism scores low.
- **Restating a training page.** The textbook is the material, not the answer.

## Grading
Five criteria, each 0–100, plus an overall: evidence discipline, mechanism, command of the evidence,
honest negatives, own understanding. **70 overall passes, provided no criterion is below 50.**

**Retakes are unlimited and unpenalised.** Aim for a pass on what you do not care about and high scores
on what you do — that is your call, not the grader's. When you are satisfied, ask for **`approve`**; that
mode is deliberately more skeptical, not less.

**Honest waivers count.** An explicit, reasoned "I am not pursuing X, because Y" is excluded from scoring
rather than marked down. Silence is not a waiver, and waivers cannot hollow out the course.

## What you can ask me (the collaborating agent) for
Which section is weakest. Whether a claim is scoped correctly. What a metric actually means — you will get
the registry entry to read, not a paraphrase. What to study.

**Not**: what you are missing, whether an unsupported claim is right, or anything whose answer is the
report. I will decline in one sentence and point you at what to read. That is the deal, not obstruction.

## Epoch 1 closes when this is approved
Not before. The frozen demo build, the physics version and the task graph are cut at the same moment.
