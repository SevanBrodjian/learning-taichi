# Examination — how the research report gets graded

> The research report is **written by Sevan, by hand, alone.** This file defines what it must clear and
> how the agent grades it. It is deliberately strict: a bar that everything passes is not a bar.

## Why this exists
The project's real output is not the demo and not the runs — it is what Sevan actually understands. A
report he wrote himself is the only artifact that can demonstrate that. Anything the agent co-writes
proves nothing about him, so **the agent never drafts, edits, ghost-writes, "tidies", or fills a gap in
`reports/research_report.md`. Not one sentence.** If it needs work, it goes back to him with a verdict.

## The one rule that makes or breaks this
**Name the gap. Never supply the content.**

Grading is worthless if the feedback contains the answer, because then the next draft is the agent's work
wearing his name. The line is precise:

| Allowed | Not allowed |
| --- | --- |
| "§3 generalises from a single task without saying so." | "§3 should say the 318× speedup was launch overhead." |
| "The claim in §2 has no evidence behind it in this repo." | "The evidence for §2 is in the dispatch-floor measurement." |
| "You have not distinguished what you observed from what you inferred anywhere in §4." | "In §4 the observation is X and the inference is Y." |
| "Re-read `core/14-real-time-cost` before revising §5." | "As `core/14` explains, the flat curve measured the API." |

Pointing at **study material is teaching**; pointing at **conclusions is answering**. A section reference
is fine. A summary of what that section says is not.

Corollary: **do not enumerate everything missing.** A list of twelve gaps is a table of contents he can
fill in mechanically. Name the **two or three that matter most**, and let him find the rest — finding them
is the skill being tested.

## The grader must not be the collaborator
The agent that grades has spent days building this with him, knows every answer, and is structurally
inclined to be agreeable. That is a bad examiner. So:

**Grading runs in a fresh subagent that has NOT seen the working conversation.** It receives only:
- this rubric,
- the repo (training textbook, `runs/`, `spec/registry/`),
- the submission.

It has no chat history to leak and no accumulated rapport to protect. The orchestrator relays the verdict
without softening it, and **does not overturn a FAIL to be kind**. If the orchestrator disagrees with the
grader it says so openly, with reasons, and Sevan decides.

## The rubric
Five criteria. **A report passes only if every one is `met`.** Anything else is `revise`.

1. **Evidence discipline.** Every claim is scoped to what was actually tested. Single-task results are
   labelled as such. Observed / hypothesised / untested are visibly separated. *This is the project's
   central skill and the most common failure.*
2. **Mechanism, not narrative.** Explains *why* a result holds, not just what happened. A correct
   description with no mechanism is `revise`.
3. **Command of the evidence.** Cites the actual runs and numbers, and gets them right. Uses registered
   metric names with their real meanings — quoting a metric whose definition he has not read is an
   automatic `revise`.
4. **Honest negatives.** Includes what failed, what was refuted, and what remains unknown. A report where
   everything worked is not a report about this project.
5. **Own voice, own understanding.** Reads as a person who did the work. Passages that restate a training
   page without engaging with it are `revise` — and the grader says which passage, not what is missing
   from it.

## Verdict format
The grader returns exactly:
- **`PASS`** or **`REVISE`**
- The five criteria, each `met` / `not met`, one sentence each — naming the gap, never the content.
- **At most three** things to address, ordered by importance.
- For each, **what to study** (a training page, a run, a registry entry) — a pointer, not a précis.

No praise padding. No consolation. A `REVISE` says so in the first line.

## What Sevan can ask for, and what he cannot
- **Can**: "which section is weakest?", "is this scoped correctly?", "does this metric mean what I think?"
  (the answer is the registry entry to read), "point me at what to study."
- **Cannot, and the agent should decline plainly**: "what am I missing?", "write this bit", "is this
  right?" about a claim he has not attempted to support, or any request whose answer is the report.

Declining is not obstruction — it is the whole point. Say so in one sentence and offer the study pointer
instead.

## Cadence
One report per **epoch**, cut at an inflection point (see `coordination/epochs/README.md`), not on a
schedule. The epoch does not close until its report passes.
