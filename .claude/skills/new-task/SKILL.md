---
name: new-task
description: Create a new task inside an existing research direction on the learning-taichi board. Use whenever the user types /new-task, or asks to create/add a task, experiment, or piece of work to the project or to a direction. If the user does not say which direction, infer the best-fit one from the direction summaries and say which you picked; if it is genuinely ambiguous or nothing fits, ask. Writes the task into coordination/directions/<dir>.json and commits it.
---

# /new-task — add a task to a direction

You are the **orchestrator** on `main` (see `CLAUDE.md` → Roles). This skill adds one task to an existing
direction. A task's `note` is a **seed** for a future worker brief, not the full contract — a sentence or
two is right; `/execute` later expands a queued task into the full brief.

## 1. Get the task's title and note
- **title** — a short imperative name for the work.
- **note** — one or two sentences seeding what it should accomplish and why. If the user gave only a title,
  draft a sensible note and show it.

## 2. Choose the direction (infer, or ask)
Read every `coordination/directions/*.json` (each has `name`, `summary`, and existing `tasks`).
- If the user **named** a direction, use it (match by name/id; if that direction does not exist, offer to
  create it with `/new-direction`).
- If the user did **not** name one, **infer the best fit** by matching the task's topic against the
  directions' names and summaries. If one direction is a clear fit, use it and **state which you chose and
  why in one line**.
- If it is **genuinely ambiguous** (two plausible fits) or **nothing fits well**, ask the user which
  direction — list the best one or two candidates, and offer `/new-direction` if none fit. Do not force a
  bad fit.

## 3. Pick the status
Default to **`proposed`** — it lands on the board as a proposal the user can review and queue, and it does
**not** auto-run on the next `/execute`. Use **`queued`** only if the user signals they want it run
(“queue it”, “and run it”, “let’s do X now”). Always state which status you set.

## 4. Write the task into the direction file
Slugify the title for the id (lowercase, non-alphanumeric runs → `-`, strip ends). If that id already
exists in the direction, suffix `-2`, `-3`, … until unique. Append to the direction's `tasks` array:
```json
{ "id": "<slug>", "title": "<title>", "status": "<proposed|queued>", "note": "<note>" }
```
**If this task is explicitly a follow-up/extension of an existing completed task**, also link it both ways:
put `"follow_up_of": "<parent-id>"` on the new task, and append the new id to that parent task's
`"follow_ups"` array (create the array if absent). This matches the dashboard's "Propose follow-up".

## 5. Commit it
Commit just the one direction file (coordination state is committed; workers only see committed files):
```
git add coordination/directions/<dir>.json && git commit -q -m "board: add task <dir>/<id>"
```
The data server serves directions live, so it appears on the dashboard within a few seconds; no restart.

## 6. Report
One or two sentences: what task was added, to which direction (and why, if you inferred it), at what
status, and the next step (queue it / run `/execute` when ready).
