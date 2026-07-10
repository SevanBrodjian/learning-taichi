# Orchestrator log — main (append-only)

## 2026-07-07 → 07-09 — big multi-batch session (dashboard + specs + skills + a full learned-materials arc)

**Dashboard/harness engineering.** Added task delete (confirm), reject-to-queue-with-note, propose-follow-up
(bidirectional links), a phone dropdown menu (<600px; iPad/laptop untouched), large-monitor scaling, nav
badges for new training / unread notifications. Fixed the training-video refresh (shared `VideoPlayer` +
`React.memo`'d `MarkdownReport`) and dashboard typing lag (pause poll while a field is focused + isolated
Overview form components). Server: new `/api/task-delete` and `/api/task-follow-up`; surfaced
`rework_history` / follow-up graph. **Bug fixed:** `_git_commit` committed the whole index (no pathspec) →
scoped to its path.

**Specs + skills.** Hardened CLAUDE.md / `spec/style_training_report.md` / task template / `/execute`:
workers must view their own figures, over-include math prereqs, resolve every `[[link]]`, show a parameter's
effect, write the manifest last referencing only existing media, and finish in-turn. Added `/new-direction`
and `/new-task` skills. Added prereq pages `linear-algebra` and `svd-polar` (resolved a dangling SVD ref)
and core `material-stiffness` (E worked example).

**Research (material-variants).** Forward fluid/elastic/snow showcase → differentiable versions
(FD-verified; the prior "bad gradients" were a CFL-unstable forward, not a differentiation defect) →
viscosity sweep (oil→honey) → learn a NN per **viscosity** and interpolate weights (sags below the linear
ideal; coordinate-mismatch ruled out) → learn a NN per **material** and interpolate weights (**degenerate
interior**; first run overclaimed a "smooth morph" and was sent back; fixed the α=0 endpoint bug —
state-rule held at the wrong endpoint) → **one net conditioned on a 2-param descriptor** (beats
weight-blending, but a real ~1–2% edge-fidelity tradeoff; abrupt solidity axis; ill-posed fluid+plasticity
quadrant degenerates; two rework rounds to strip overclaims).

**Research (realistic-rendering).** Non-differentiable photoreal-aspiring renderer → realism follow-up
(stiffer/less-damped water; filled-interior mask + distance-transform to kill Poisson holes) →
**GPU-accelerated renderer** (Taichi: atomic-scatter splat, separable blur, jump-flooding EDT, on-device;
**130–265×**, visual parity) → long diverse showcase (6× 11–14 s clips + per-particle-dye color mixing).

**Process notes:** caught and sent back two worker overclaims at review (viewing the media is what caught
them). Workers repeatedly ended their turn on long background GPU jobs (a runtime turn-budget cutoff) —
resumed them via `SendMessage(agentId)`; the reliable fix was constraining a resume to narrative-only. Ran
the GPU-renderer benchmark alone to avoid GPU-timing contention. Durable ops lessons recorded in
`coordination/shared_memory/orchestration-lessons.md`.
