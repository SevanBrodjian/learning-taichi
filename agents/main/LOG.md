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

## worker: interactive-simulation-of-one-material (material-variants)
Ported the canonical elastic MLS-MPM step to single-threaded JavaScript so it runs interactively in a
browser, verified it against `sim.physics`, and measured what interactive rates cost.
- **Port is exact.** traj_rmse 1.7e-4 (launched-disk, 2.5 s) against canonical, versus 0.7e-4 self-noise
  and 2.2e-4 for a 1e-7 initial-condition nudge, so the divergence is chaos, not bias. Verified on two
  scenes; drop scene 1.5e-5 vs 1.0e-5 self-noise.
- **Parameters generated, never retyped** (`web/gen_params.py` imports `sim.physics` -> `web/params.js`,
  physics-version stamped). Constitutive law unchanged.
- **Three forced changes:** `ti.svd` deleted (2D elastic needs only the closed-form polar rotation, which
  is what Taichi's own svd2d is built on); dense grid loop rewritten sparse (exact, bit-identical over
  3340 substeps); f64 arithmetic with f32 storage.
- **The cost finding:** dt=1e-4 forces 167 substeps/frame, capping the machine at ~1154 particles at
  60 fps. Taichi's dense grid sweep alone is 15.6 ms/frame of empty cells. Canonical CUDA is *flat* at
  ~345 us/substep from 500 to 16384 particles (launch-bound), so one JS thread beats an RTX 4090 below
  ~4300 particles. Raising dt 1.5x costs 3 orders of magnitude of accuracy while still looking stable.
- **Learned grid update priced, not trained:** smallest useful MLP is 242x the analytic update in JS
  (13x over the frame budget); on GPU it is free relative to analytic but only because both hide under a
  56 us kernel launch. Conjecture recorded: only a coarse-time (once-per-frame) learned update could win.
- Two engine gotchas worth remembering: `performance.now()` is clamped to 100 us in Chromium, so
  per-substep phase profiling is garbage and phases must be priced by differencing whole loops; and the
  dashboard renders every manifest prose field as **plain text**, so markdown tables belong in
  `results[]` as `type: "table"`, never in `full_report`.
- Textbook: new core page `real-time-cost`; revised `svd-polar` (added the 2D closed-form polar rotation
  and corrected the claim that the SVD is the numerically stable route to R) and `material-stiffness`
  (the CFL wall is now measured; accuracy dies before stability does).

## 2026-08-16 — worker: sand as a fourth canonical material, four materials in one grid
- **Promotion.** `sand` is canonical: Drucker-Prager return mapping (Klár et al. 2016) on a Hencky
  log-strain elastic law, `E=300, dt=1e-4, phi=50`. Chosen because sand is *cohesionless* (shear strength
  ∝ confining pressure = a cone), where snow's Stomakhin clamp is a fixed box = cohesion. Log strain makes
  the volume/shape split of the yield condition exact, so the cone projection is closed-form.
  `phys-1dc280eb52c9` → `phys-bebeaafbe73e`.
- **New canonical scene `heap`** (over-steep 60° triangle released from rest) — the honest angle-of-repose
  test. A collapsing column measures runout, not repose.
- **4 new signatures for sand + 4 asserting the multi-material path == canonical. All 14 pass.**
- **`simulate_multi`**: per-particle `mat_id`, runtime branch in P2G/G2P, `shared_dt = min(dt)`.
- **Sand costs 167 substeps/frame — the same as elastic. SNOW (333) STILL BINDS THE DEMO.**
- **Two standing claims failed, both worth remembering:**
  1. *Snow's dt is not set by hardening, and not by stability at all.* Snow's measured stability wall is
     **8× its canonical dt**; `xi=0` does not move it. The hardening IS real (and bimodal across 4 decades:
     ~44% of particles end up *softer* than nominal, ~50% stiffer than elastic) but it is not what pins dt.
  2. *An angle of repose from this solver is not converged.* The settled slope of every **plastic** material
     decays with **substep count, not physical time** — snow's spread across dt is 37.1° at equal time and
     **1.6° at equal substep count**; elastic (no plastic projection) is 0.0° on both. So the FINE run is the
     corrupted one. Hypothesised mechanism: a one-sided return mapping rectifies transfer noise into
     permanent plastic strain, once per substep. Canonical snow's cohesion decays over a long rollout.
- **Two methodology lessons (both caught false alarms in this run):** single-sample comparisons against a
  single self-noise number are coin flips on chaotic scenes — the frozen-material check and the refactor
  check each produced a false FAIL before being made distributional (N repeats, within- vs across-code
  distributions, plus a one-ulp rounding-perturbation bracket for the refactor).
- **Re-confirmed the gotcha already in this log:** manifest prose fields render as PLAIN TEXT. Markdown was
  written first and had to be rewritten as flat prose with CAPS emphasis; all tables moved to `results[]`.
- Textbook: rewrote `material-showcase` (now four materials, leads with "a material is defined by which
  deformations it refuses to remember"); extended `constitutive-models` (the DP cone + return mapping),
  `real-time-cost` (one grid = one dt; and the substep-creep result), `linear-algebra` (deviatoric split),
  `svd-polar` (log/Hencky strain). Registered `repose_angle`, `shape_drift`, `substeps_per_frame`,
  `dt_stable_max`, `dt_faithful_max`.
