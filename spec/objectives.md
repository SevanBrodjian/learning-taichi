# Objectives — this project

> What success looks like, concretely. Co-authored — refine `TODO(you)` lines.

## Primary (the learning)
1. Make `sim/mpm88.py` **differentiable** and use gradients to solve a control/inverse task
   (e.g. optimize initial velocity or actuation to reach a target).
2. Understand it deeply enough to **explain MLS-MPM and gradient flow through the rollout** unaided.
3. Catalogue the **failure modes** actually encountered (exploding/vanishing gradients, NaNs,
   learning-rate sensitivity, non-differentiable contact) and the fixes that worked.

## Secondary (the harness)
4. Stand up the **orchestration harness** (worktrees, task specs, reports, dashboard, ntfy) and prove
   it on the Phase 1 slice.
5. Learn to **manage parallel agents** (Phase 3) and study the information bottleneck between me and
   self-propagating agent work.

## Definition of done for Phase 1
- Loss curve genuinely decreases; video shows the body reaching the target.
- Installable **PWA** renders the loss chart, the video, and a math-bearing training report on iPad.
- `reports/training/diffmpm.md` is good enough that I can give a whiteboard explanation of MLS-MPM +
  autodiff afterward.

## Priorities & constraints
*Written by user.*  The most important goals of this project are to achieve a highly autonomous research vehicle pursuing a specific concept area in order to accumulate knowledge and depth. A critical component of this is the passage of knowledge from you, the agent, back to me, the user. I don't just want a demo at the end of this, I want to be able to explain in intimate detail:
- how it works
- what the design tradeoffs were
- what the most challenging parts were and why
- the relevant underlying mathematics, physics, and conceptual components
- what the interesting parts of the project are, and what could have been done differently
- how it fits into the wider field
- how it fits into my wider objectives and research interests

However, I want to minimize the amount of time I actually spend in implementation, debugging, coding, running experiments, etc.
This is not my primary research project. Therefore, I am much more willing to sacrifice the depth that comes from such engagement. Instead, the purpose of this kind of project is exposure, building intuition, acquiring domain-specific knowledge, and then moving onto a new idea. It's a way to acquire breadth and synthesis across domains. That's why I'm willing to automate as much of it as possible using AI agents.

Attempt to implement as much as possible on your own. Notify me of progress, but only halt and await instructions when absolutely needed.

A major outcome of this work needs to be a polished demo. The demo should immediately convey the core principles of what we've built and why it matters. Ideally this should be in a format such that even a non-technical user can intuitively grasp that something cool and futuristic is occuring. Overall, the demos from these kinds of work should point strongly toward my overarching research vision of structured generative interactive worlds with internal consistency, logic, and coherence for education, expression, entertainment, and exploration. I don't just want to accumulate research and ideas, though. I want to accumulate a highly personalized tech suite that enables me as a creator in this same medium that we are exploring. I want to be able to direct interactive experiences, build worlds, share them, and design new phenomena, rendering styles, etc. that don't exist yet while showing the creativity they enable. I want people, even everyday people, to immediately *feel* why it matters. I know we're far from this today, but one chip at a time I want to work toward this.

We have a powerful GPU locally. I never want to cut a project off because of artificial compute constraints. However, the end demo is an important consideration to keep. It's possible we need to pre-render, but interactivity always beats passive watching for engagement and sharing. Therefore, whenever possible we should optimize for efficiency, shippability, and keep our objectives in mind.

These projects are meant to take anywhere from a few days to a few months of time, with most ending after a handful of weeks. However, the ideas from the project can evolve over considerably longer timeframes. It's good to be bold. It's good to be ambitious. I don't care as much if the demo comes out underbaked or even incomplete if it's because we pushed hard for something big that simply didn't materialize. I dislike boring, safe bets. I will engage with the steering of this project to help ensure that is avoided.

Finally, it's worth reiterating the emphasis on teaching and learning for me. Although I'm trying to automate to the furthest extent, such that I can spend the majority of time on other non-automatable work, checking in infrequently, I will not ship anything that I don't understand deeply. I will not present to others, share on my website, or otherwise take credit for something that I can't explain in detail to others. This is a principle of mine. Therefore, even though you are doing the work, I must keep up with the core research decisions. I understand I can't know everything, but I want to know enough to be quite competent. More details are in the training report spec, but in general I care far more about learning first-principles, ideas, mathematics, intuition, hardware-awareness, and design than code idiosyncracies, syntax, and other such details.


## Non-goals (for now)
- Live, interactive Taichi *in the browser* (not possible — sim is server/desktop side; web gets
  pre-rendered or streamed results).
- Production-grade infra. Favor the thin slice and learning value over generality. > TODO(you)
