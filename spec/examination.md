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

## The grader is a separate agent type, with no project context
Grading runs as the **`grader`** subagent (`.claude/agents/grader.md`). It is defined as an independent
third-party auditor: it did not build the project, has not seen the working conversation, and is
instructed not to go looking for it. It reads the course scope, the repo and the submission.

This is structural, not decorative. The collaborating agent knows every answer and is inclined to be
agreeable — a bad examiner. The orchestrator **relays the grader's verdict without softening it** and
does not overturn a `RESUBMIT` out of kindness. If it disagrees it says so openly, with reasons, and
Sevan decides.

## Courses — one per epoch, scoped to that epoch ONLY
The curriculum is a sequence of **courses**, one per epoch. **A course covers only the work since the
previous epoch closed.**

This scoping is the point, not an implementation detail: without it the grader drifts into re-testing
material from three reports ago, which is both discouraging and a waste of a real assessment. Earlier
courses remain legitimate *background* — Sevan may rely on them freely and is never penalised for not
re-explaining them. Each epoch directory carries a `course.md` stating scope explicitly: which tasks,
which training pages, which findings are in play.

## Grading — a percentage, with unlimited retakes
Five criteria, each scored **0–100**, plus an overall.

1. **Evidence discipline** — claims scoped to what was tested; single-task results labelled; observed /
   hypothesised / untested separated. *The central skill, and the most common failure.*
2. **Mechanism** — explains *why*, not just *what*.
3. **Command of the evidence** — real runs and numbers, correct, with registered metric names used in
   their registered meanings.
4. **Honest negatives** — what failed, what was refuted, what is still unknown.
5. **Own understanding** — reads as someone who did the work and thought about it.

**70 overall passes, provided no criterion is below 50.**

**Retakes are unlimited and unpenalised.** Sevan resubmits as often as he likes; each attempt is graded
fresh. The aim is a *pass* on material he does not care about and *high scores* on the material he does —
which is his call, not the grader's.

When he is satisfied he asks for **`approve`**, a deliberately separate mode. It is not a rubber stamp:
the grader re-examines and is instructed to be *more* skeptical, because a report polished over several
retakes can acquire fluency without acquiring understanding.

## Honest waivers
He may explicitly decide he does not need to master something. **An explicit, reasoned waiver is
honoured** — the topic is excluded from scoring rather than marked failed. Two limits:
- **Silence is not a waiver.** An unmentioned gap is a gap; a waiver is a stated "I am not pursuing X,
  because Y."
- **Waivers cannot hollow out the course.** If they cover so much that nothing remains to demonstrate
  understanding, the grader returns `INSUFFICIENT COVERAGE` instead of a score.

A waiver costs nothing on the floor but caps the ceiling: no high score for material not attempted.

## What Sevan can ask the collaborating agent for, and what he cannot
- **Can**: "which section is weakest?", "is this scoped correctly?", "what does this metric actually
  mean?" (answer: the registry entry to read), "point me at what to study."
- **Cannot, and it should decline plainly**: "what am I missing?", "write this bit", "is this claim
  right?" about something he has not attempted to support — anything whose answer is the report.

Declining is the mechanism working, not obstruction. One sentence, then the study pointer.

## Cadence
One report per course, cut at an inflection point (`coordination/epochs/README.md`), never on a schedule.
The epoch does not close until its report is approved.
