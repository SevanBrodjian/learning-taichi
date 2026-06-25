# Working with Sevan (durable collaboration notes)

Canonical, in-repo notes on how to collaborate with the user. Every session, orchestrator or worker,
should read this. It complements `spec/` (which calibrates *what* to write) and `CLAUDE.md` (the
operating rules). These facts live here, in the repo, not in any session or auto-memory.

## Who
Sevan — researcher in generative interactive simulations and world models. Full profile and the
"structured generative worlds" vision are in `spec/researcher_profile.md`. Read that first.

## How he works
- **Co-author, do not ask-and-wait.** Bring a strong draft or a clear opinion, not a blank menu. He
  refines from a real proposal.
- **Blunt honesty over flattery.** State where you disagree and why. He explicitly asks for pushback and
  values it. Do not pad or hedge.
- **Mostly autonomous, ping often, block rarely.** Keep executing; escalate asynchronously (see the
  Autonomy charter in `CLAUDE.md`). He will steer.
- **Teach from the ground up.** He will not ship or present anything he cannot explain in depth. The
  training textbook is a first-class deliverable. Start from the beginning, physical intuition before
  math, define every term before using it.
- **Bold over safe.** He dislikes boring, safe bets and is fine with an ambitious miss. Reach for the
  big version of an idea.
- **The filesystem is the backbone.** Everything learned, decided, or designed must land in the repo.
  Do not rely on auto-memory or session context (see `CLAUDE.md` -> Persistence).
- **The demo and the UX matter.** The end demo must make even a non-technical person *feel* why it
  matters. He gives detailed, specific UI feedback and notices rough edges. Take them seriously.
- **Hates overclaiming (gating).** A result on one task is one data point, not a general truth. Scope
  claims to the evidence, separate observation from hypothesis, and test generality across several tasks
  before asserting a pattern. He flags this as make-or-break for the project (see `CLAUDE.md` -> Evidence
  discipline). He also wants each task to include a *why/hypothesis* discussion that seeds new tasks.

## Practical
- The dashboard is a fixed-URL PWA on his iPad, port 5174 only. **Touch matters**: HTML5 drag-and-drop
  does not work on iPad, so any board interaction must also have a tap-friendly path.
- Stay on his Claude subscription: run workers as local subagents or worktree sessions, not cloud agents.
