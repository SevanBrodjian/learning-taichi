---
name: new-direction
description: Create a new research direction (an organizing axis) on the learning-taichi board. Use whenever the user types /new-direction, or asks to create/add/start a new direction, research axis, research thread, or a new area of investigation for the project. Writes coordination/directions/<id>.json and commits it so it appears on the dashboard. If the name or intent is unclear, ask one concise question before creating.
---

# /new-direction — add a research direction

You are the **orchestrator** on `main` (see `CLAUDE.md` → Roles). A *direction* is an organizing axis in
`coordination/directions/`, each a JSON file joined onto the dashboard board. This skill creates one.

## 1. Gather the two fields
- **name** — the human title of the axis (e.g. "Material variants", "Differentiable control").
- **summary** — one or two sentences stating the organizing question the axis pursues.

Take them from what the user said. If the **name is missing or the intent is genuinely unclear** (you
cannot tell what axis they mean), ask **one** short question and wait. Do not invent an axis. If the user
gave enough to write a sensible summary, write it yourself and show it to them rather than asking.

## 2. Write the direction file
Slugify the name for the id: lowercase, replace each run of non-alphanumeric characters with `-`, strip
leading/trailing `-` (e.g. "Material variants" → `material-variants`). If
`coordination/directions/<id>.json` already exists, pick a distinct id or tell the user it exists — never
overwrite one.

Write exactly this shape (a new direction starts `proposed` with no tasks):
```json
{
  "id": "<slug>",
  "name": "<name>",
  "status": "proposed",
  "summary": "<summary>",
  "tasks": []
}
```
If the user described one or more concrete tasks in the same breath, you may seed them into `tasks` as
`proposed` entries (id = slug of the task title, plus `title` and a `note`), but keep the direction itself
`proposed`. Otherwise leave `tasks` empty and point them at `/new-task`.

## 3. Commit it
Commit just this one file (the harness treats coordination state as committed, and worktrees/workers only
see committed files):
```
git add coordination/directions/<id>.json && git commit -q -m "board: add direction <id>"
```
The data server reads the directions live, so the new axis shows up on the dashboard within a few seconds;
no server restart is needed.

## 4. Report
Tell the user the direction is created (name, id, status `proposed`) and the natural next step: add tasks
to it with `/new-task`, or drag/queue work once tasks exist. Keep it to a sentence or two.
