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

## the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page
- Ported a Taichi-exact 2x2 SVD to WGSL (polar incl. the det<0 reflection branch + one Jacobi rotation);
  unit-verified against `ti.svd` on 2,076 matrices in 11 adversarial families before wiring it in.
- Extended the elastic-only WebGPU engine to four materials WITHOUT adding storage buffers: material id
  and Jp packed into `vel` widened to vec4, fluid's J into `Fm.x`. Still 7 of the guaranteed 8.
- Verified per material against canonical on the angle-of-repose heap, and on a mixed four-material
  scene against `simulate_multi`; all judged against canonical's own self-noise band.
- Shipped to the Demo tab; drove the real dashboard over CDP and clicked every control there.
- Caught two defects that every numerical check passed: a CSS `[hidden]` override that drew the
  no-WebGPU overlay over a working sim, and a fixed 1/60 s per frame loop (2.2x too fast at 133 Hz).

## improve-material-realism-in-behavior
- Changed FROZEN ground truth (`sim/physics/`) for the first time since sand: per-material `rho`, `nu`
  and `fric` in `MAT`, all three previously single globals. Version `phys-bebeaafbe73e` ->
  `phys-c518316a4a05`.
- Diagnosed before tuning. Rubber's "compresses too much" is the global `NU = 0.2`: on a hard floor
  impact the body held 89% of its area at peak with its worst particle at det F = 0.18. Water's "mushy"
  is the weak-compressibility stiffness: a water particle reached J = 0.51. Both measured on the old
  physics with a parameter sweep, before anything was edited.
- Buoyancy needed no buoyancy force. Density enters as `p_mass = p_vol * rho`; the grid divides the
  fluid's pressure impulse by node mass while gravity is applied to velocity, so `a = -g(1 - rf/rs)`
  falls out. Snow floats, rubber and sand sink, and the SAME blob at four densities orders monotonically.
- The reason snow and sand did not move is a gauge symmetry, not luck: a lone material is exactly
  invariant under `(rho, E) -> (k rho, k E)` because the momentum balance only contains `E/rho`. Each
  material's density was introduced with its stiffness scaled to match. Now a golden signature.
- 9 new signatures (buoyancy x5, the density gauge x3, volume retention x2, per-material friction);
  all 15 pre-existing ones stayed green.
- Two honest negatives. "Rubber breaks too easily" could not be reproduced on any scene tried. And
  water's WALL clinging got worse, not better -- and the ablation blamed the friction change, not the
  stiffness, which was the opposite of the obvious guess. Frictionless water climbs a wall as freely as
  it slides along a floor.
- Found and halved a pre-existing artifact while building the pool scene: a randomly seeded pool
  compacts as it settles (free surface -25% over 2.2 s) because the fluid's pressure comes from an
  ADVECTED J, not from the actual packing. Lattice seeding (`seed_lattice`) cuts it to -12%. With the
  random pool the falling surface hid the rubber blob sinking entirely.
- Stability was checked on velocity and deformation, NOT positions: `simulate` clamps positions into the
  domain, so a diverging run still returns finite in-range x and a check on x alone passes on a material
  that has already exploded.

## 2026-08-19 — worker: propose-new-rendering-for-each-of-the-four-materials

Proposed a distinct visual treatment per canonical material (water / rubber / snow / sand), rendered
against the demo's current shader on two scenes, plus the four together on one grid and the canonical
buoyancy pool. Demo page and `sim/physics/` untouched (`git status` clean for both).

- **Greyscale test is the deliverable.** Same neutral albedo for all four, luminance output. Current:
  four visually identical mushy blobs. Proposed: four different shape languages. Everything else on the
  page supports that one comparison.
- **Baseline is a line-by-line port** of `mpm4.js` `fs_splat`/`fs_resolve` with the demo's own constants
  (radius 0.017 of the domain, iso 2.6), and every scene is seeded at the demo's areal density
  (28,294/unit area, ~1.7 particles per grid cell) so the iso threshold still means what it means there.
  Could NOT pixel-diff against the live WebGPU demo: this environment refuses to composite a frame from
  the demo canvas. Fidelity rests on code correspondence.
- **A wall clock was the wrong instrument and nearly produced a wrong conclusion.** Timed on the host,
  every screen-space treatment reported ~3.3 ms — the same at 360^2 and 1080^2, a 9x change in pixels.
  Re-measured with Taichi's kernel profiler (`sim/material_render_cost.py`, which wraps `ti.init` to turn
  the profiler on without touching `sim/physics`): 0.31–1.12 ms of actual device time. The gap is
  Python-side kernel launches and matches the `dispatch_floor_us` result from the WebGPU-port task.
  Registered `render_gpu_ms` and `render_wall_ms` in `spec/registry/metrics.json`, with the caution.
- **PLATFORM DEFECT, affects every task page in the repo, not just this one.** Media referenced by
  `/api/data/...` inside `custom_html` does NOT load in the dashboard: the task page runs in
  `sandbox="allow-scripts"`, i.e. an opaque origin, and in this browser that document is refused every
  subresource from the data server. Isolated it: the identical srcdoc iframe WITHOUT the sandbox
  attribute loads the same URL fine (`plain/root: ok 723` vs `sandbox/root: ERR`). `improve-material-
  realism-in-behavior` has 14 such refs and would be equally blank. Mitigated here by inlining ~140 KB of
  base64 stills that take over on a video `error` event, plus a banner explaining it. **The harness fix
  belongs to someone else** — either add `allow-same-origin` to the iframe, or inline media.
- **Display math must have `$$` on its own line.** `$$content
content$$` makes the dashboard's renderer
  swallow the rest of the document into one math run; two KaTeX errors were caught only by opening the
  rendered pages. Fixed; both new pages now render with 0 errors.
- Honest negatives kept on the page: snow's treatment is a near-no-op and reads by elimination; the
  water reconstruction loses the airborne spray the splat keeps.

## 2026-08-20 — T-027 REWORK: the water reconstruction

Sent back with one note: the water still looked like the old water. It did. The first run ported
T-020's water **shading** (all of it, faithfully) and not the **reconstruction** the shading reads,
so `th` was still the local splat sum and `nrm` still its four-tap gradient. A splat accumulation is
lumpy at the particle spacing, so Beer-Lambert on it is lumpy and a cos^70 specular on its gradient
is thousands of highlights. The shading was correct and was lighting the wrong surface. That failure
is invisible to every check short of putting the two images side by side.

Ported `sim/material_render.py:build_masks` to WGSL as eleven half-resolution render passes between
the existing splat and the existing resolve: separable Gaussian (the horizontal pass also does the
2x downsample; its sigma derived so splat-width + blur = T-020's smoothing of a point histogram),
threshold at 0.24 of full packing (computed by the host from particle density and splat radius, not
a per-frame percentile), jump flood, seeds -> distance with a 3x3 box. Opacity, normal and optical
thickness all come off that distance field now.

Three platform constraints shaped it. Per-pass arguments go through one uniform at a **dynamic
offset**, because `queue.writeBuffer` between two `beginRenderPass` calls is queue-ordered and would
apply to every pass in the submission. Seeds live in **rg16float** — f16 is exact on integers to
2048, and the 16-bit float formats are filterable where 32-bit float is not, which the resolve needs
for the bilinear upsample. The resolve went **premultiplied**, so `1 - alpha` carries
`exp(-absorb*t)` and the water transmits without ever sampling the background; the other three
materials are bit-identical under that change.

Two things that were not in the plan. T-020's **tone curve** had to come along, applied to water
only: T-020 tonemaps its whole frame and the demo does not, and a Beer-Lambert radiance written
straight to an 8-bit non-sRGB swapchain gave a near-black pool on the first build. And the foam's
motion gate reads the **grid velocity** buffer (already bound to the fragment stage for the grid
view) instead of T-020's mass-weighted speed splat, which would have needed a second render target
on the heaviest pass. Verified firing: 237 whitewater pixels at the splash peak against 109 before.

**Measurement trap worth remembering.** The first cost numbers came back as exact multiples of
32,768 ns. That is Chromium quantising `timestamp-query`, not a cost.
`--disable-dawn-features=timestamp_quantization` shrank the quantum but did not remove it (16-33 us
residual), which is still the size of the thing being measured. Priced it twice instead: as a
difference against a matched control in the same run, and as the slope of running the chain K times
in one timed region. They agree. Chain costs 0.028 ms at 480^2 and 0.052 ms at 1080^2 — sub-linear,
because a fixed ~0.020 ms of twelve-render-pass setup dominates at these sizes. Frame goes 7.06 ->
7.11 ms of a 16.67 ms budget. Written into the `render_gpu_ms` registry entry so it is not
rediscovered.

Landed within 3/255 of T-020's own render on interior colour, untuned. Layout re-measured
byte-for-byte identical at all five viewports; snow, sand, rubber and `sim/physics/` untouched.

Honest consequence, found by driving the real page rather than by reasoning: option B makes shallow
water nearly invisible. That is what the treatment is, not a defect in the port, and it is the
user's call whether to accept it. In section 4 of the task page and in limitations.
