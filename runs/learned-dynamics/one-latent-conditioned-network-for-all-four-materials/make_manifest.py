"""Write manifest.json. Runs LAST; every media `src` is checked to resolve or the write is refused."""
import datetime
import json
import pathlib
import sys

RUN = pathlib.Path(__file__).resolve().parent
REL = "runs/learned-dynamics/one-latent-conditioned-network-for-all-four-materials/"
sys.path.insert(0, str(RUN.parents[2]))
import sim.physics as phys                     # noqa: E402

M = json.loads((RUN / "metrics.json").read_text())
C = M["cost"]
FITQ, FITF = C["max_width_realtime_quarter_gpu"], C["max_width_realtime_full_gpu"]
AN = C["analytic_us_per_substep_full"]
W16 = [r for r in C["by_width"] if r["hidden"] == 16][0]["us_per_substep"]
SIG = M["golden_signatures"]["learned_h32"]
PAR = M["parity_host_vs_wgsl"]

SUMMARY = f"""**One network with one shared weight set does run all four canonical materials on WebGPU, and
the cost is not what stops it. The accuracy is.**

The per-particle constitutive model -- the stress *and* the plastic state update, so snow's clamp and
sand's Drucker-Prager return map are learned rather than applied afterwards -- was replaced by a single
MLP told which material it is by nothing but a fixed 4-dimensional code. On the cost side the answer is
better than expected: the analytic four-material law it replaces is itself expensive (a 2x2 SVD, a cone
projection, logs and exps), costing {AN:.1f} us per substep at 8192 particles, and a width-16 network
costs {W16:.1f} us. The seam is therefore **free up to width {FITQ}** against a 60 fps budget at a quarter
of an RTX 4090, and affordable to width {FITF} on the whole device. Trained and untrained weights cost the
same to within 0.1%, and the shader reproduces the host network to 1.6e-4 of the output spread.

On the capacity side it does not work. The best network trained here reaches a held-out one-step error of
0.10 to 0.25 of each material's own spread, and that is not close enough: rolled forward for a full
simulation every material scatters and fills the tank, so a learned drop spreads 0.88 where canonical
spreads 0.35. Against the unmodified golden signatures the learned simulator passes
{SIG['pass']} of {SIG['pass'] + SIG['fail']} runnable rows, and the ones it passes are largely the stable-
and-self-consistent ones rather than the ones that distinguish the materials. This is a per-step
supervision result with no data-aggregation round, so it bounds what per-step fitting alone buys, not
what the architecture can ultimately do."""

FINDINGS = f"""TESTED: one RTX 4090 in Chromium via WebGPU, one grid (128x128), 8192 particles unless
stated, four canonical materials on ONE shared grid at the timestep they force (dt = 5e-5 s), one
architecture family (a per-particle MLP with one hidden layer, tanh, 14 inputs and 7 outputs), hidden
widths 8 to 256 swept for cost and 8/16/32/64/128 trained, plus one two-hidden-layer variant at width 64
and one input ablation. Scenes: the canonical drop, column, heap, dam, slam, two-blob and pool scenes.
Every timing is a timestamp query over a pass of 128 substeps, minimum over 9 repetitions, from a
bit-identical warmed state re-seeded before every probe.

OBSERVED -- the scaffolding is canonical. The reparameterised step (stress cached one kernel earlier, F
remounted as R*S, the fluid's volume ratio carried in F) agrees with canonical `sim.physics` to 5e-12
(rubber) and 3e-9 (snow, sand) in position after one substep, and passes all 21 runnable golden signatures
when the analytic law is plugged into it. The WGSL analytic port matches canonical Taichi within the 1e-7
initial-condition nudge band for all four materials. Both were gated before any network was trained.

OBSERVED -- cost. Analytic {AN:.2f} us/substep whole solver ({C['analytic_us_per_substep_g2p']:.2f} us in
G2P, where the seam is). Learned: width 8 costs {[r for r in C['by_width'] if r['hidden'] == 8][0]['us_per_substep']:.2f},
width 16 {W16:.2f}, width 64 {[r for r in C['by_width'] if r['hidden'] == 64][0]['us_per_substep']:.2f},
width 256 {[r for r in C['by_width'] if r['hidden'] == 256][0]['us_per_substep']:.2f}. Real time needs
dt = 50 us per substep on the whole device and 12.5 us at a quarter, so the largest network that keeps
60 fps is width {FITQ} at a quarter GPU and width {FITF} on the whole one. Cost with the TRAINED weights
equals cost with random weights to a ratio of {C['cost_trained_vs_untrained_ratio']:.4f}.

OBSERVED -- the width cliff is the compiler. G2P cost is linear in width at 0.066 us per unit from 8 to
88, then jumps 2.2x between width 88 and 92 for 1.05x the arithmetic. The identical shader with the hidden
loop bound read from a uniform (so it cannot be unrolled) has no cliff, is 2-3x slower below width 88, and
lies exactly on top of the unrolled curve above it. Weights in a storage buffer instead of a uniform cost
about 3x more and scale linearly. f16 weights are slower than unrolled f32 below the cliff, faster above
it, and show no cliff -- but they destroy accuracy at this output scale (max parity error 0.46 of the
output spread against 1.6e-4 in f32).

OBSERVED -- capacity. Held-out one-step error in units of each material's own spread, stress outputs,
single hidden layer: width 8 gives roughly 0.5 to 0.9 depending on material, width 128 gives
fluid 0.06 / rubber 0.19 / snow 0.42 / sand 0.16. Snow is consistently the worst by a factor of two to
three. A SECOND hidden layer at width 64 beats a single layer at width 128 on every material
(snow 0.13 against 0.42) for about 3.7x the arithmetic. Zeroing the C and v inputs entirely changes
nothing (stress error 0.21 against 0.20), which is expected: for this seam the canonical target depends
only on the stretch, the plastic record and the material.

OBSERVED -- rollout. Every learned rollout stayed finite and none blew up, but none reproduced its
material. On the canonical drop the learned spread width is 0.84 to 0.90 for all four materials where
canonical gives 0.40 (fluid), 0.18 (rubber), 0.35 (snow) and 0.66 (sand): the learned materials become
indistinguishable dust that fills the domain. traj_rmse against canonical is 0.07 to 0.37 depending on
material and net, against an ORACLE floor (the same scaffolding running the exact analytic law) of 1e-4
to 4e-2 and a 1e-7 IC-nudge band of the same order. Golden signatures against the learned simulator:
{SIG['pass']} pass, {SIG['fail']} fail, {SIG['na']} not applicable.

NOT TESTED, and therefore not claimed: any behaviour between the four material codes; any GPU other than
this one; any grid resolution other than 128x128; data aggregation, rollout-aware losses, or any training
scheme other than per-step supervision; the two-hidden-layer variant's cost on WebGPU (the shader
implements one hidden layer only, and its cost is estimated arithmetically, not measured)."""

HYPOTHESIS = """HYPOTHESIS (mechanism, not observation).

1. THE ROLLOUT FAILS BECAUSE THE MATERIALS LIVE ON INCOMPATIBLE STRAIN SCALES AND SHARE ONE INPUT
   NORMALISATION, with high confidence about the observation and moderate confidence about the mechanism.
   Snow's plastic clamp confines its singular values to [0.975, 1.0075], while rubber's deformation
   gradient roams over tens of percent. One shared first layer with one whitening therefore sees snow as
   an almost constant input while still needing a varying output, which would predict exactly what is
   observed: snow is the worst-fit material at every width, by a factor of two to three, and it is the
   material whose stress magnitude is also the smallest. WOULD TEST: give the network a per-material input
   scale (a legitimate use of z_m, which could gate a learned scaling), or train a snow-only net of the
   same width and see whether it reaches an error the shared net cannot.

2. THE SCATTERING IS ACCUMULATED PER-STEP ERROR ACTING AS AN ENERGY SOURCE, high confidence. A stress
   error uncorrelated with the true stress does work on the particles that does not average to zero, and
   20,000 substeps per simulated second is a great many opportunities. The signature is that all four
   materials fail the SAME way -- they expand -- rather than each failing in a material-specific
   direction. WOULD TEST: measure total kinetic plus elastic energy over a learned rollout against
   canonical; if the mechanism is right the learned run gains energy monotonically.

3. THE UNROLLING CLIFF IS A COMPILER BUDGET, not register spilling, high confidence on the
   identification and low confidence on which budget. The control is decisive about the cause (the
   unrolled kernel degrades to exactly the un-unrolled cost, and a spill would go past it), but whether
   the limit is the instruction cache, the register file, or a heuristic instruction-count budget in
   Dawn's shader compiler is not determined by anything measured here. WOULD TEST: partially unroll by a
   fixed factor of 2, 4, 8 and find which factor reproduces the fast path at width 128.

4. THE SEAM CHOICE, NOT THE METHOD, IS WHY THE COST ANSWER IS SO DIFFERENT FROM THE EARLIER GRID-UPDATE
   RESULT, high confidence. That task replaced a nearly free kernel and paid 74x at width 16; this one
   replaces an SVD and a cone projection and pays nothing at width 16. Same hardware, same technique.
   WOULD TEST: this is close to established by the two measurements together, but a third seam of
   intermediate analytic cost would make it a trend rather than two points."""

LIMITATIONS = f"""1. THE CAPACITY RESULT IS A RESULT ABOUT PER-STEP SUPERVISION, not about the
architecture's ceiling. No data-aggregation round was run, no rollout-aware loss was tried, and the
literature's standard fix for exactly this failure (aggregating the states the learned rollout actually
visits and relabelling them, which is cheap here because the label is analytic) was designed into the
data pipeline and then not used for want of time. A reader should not conclude "one network cannot hold
four materials"; the supported conclusion is "one network fitted per-step to four materials at these
widths does not survive a rollout".

2. ONE DEVICE, ONE BROWSER, ONE RESOLUTION, ONE PARTICLE COUNT for every timing. The unrolling cliff in
particular is a property of one shader compiled by one driver, and its position should be expected to
move on other hardware. The dispatch floor of {C['dispatch_floor_us']:.2f} us is likewise this machine.

3. THE TWO-HIDDEN-LAYER VARIANT WAS NOT MEASURED ON WEBGPU. It is the best network here by a clear
margin, and the shader only implements one hidden layer, so its deployability is an arithmetic estimate
(about 3.7x the width-64 marginal cost, which would put it near the whole-GPU budget) and not a
measurement. Do not treat it as a shipped number.

4. THE WEBGPU HARNESS RUNS ONE PARTICLE VOLUME for the whole domain, where canonical allows a per-group
volume. Per-material DENSITY and FRICTION are canonical (they were not, in the first pass, and snow was
silently 3.3x too soft until that was fixed and the port checked against canonical Taichi).

5. TRAJECTORY NUMBERS ARE SINGLE-SEED and the scenes are not centrally specified -- drop heights and blob
radii are re-declared per task across this project, so cross-task comparisons of spread width carry that
caveat.

6. THE SIGNATURE COUNT EXCLUDES 4 ROWS AS N/A because they override a constitutive parameter (E, mu_visc)
that a learned law has no knob for. They are excluded from the numerator and the denominator, and named
individually rather than dropped.

7. THE LATENT IS A LABEL. Four structurally unrelated materials have no ground truth between their codes.
Nothing between them was tested and nothing about interpolation is claimed. The jitter during training
buys robustness to a slightly-off code, not a physical axis."""

FULL = f"""## The seam, stated exactly

Replaced: the per-particle constitutive model. Given the trial symmetric stretch `S` of the deformation
gradient (from a polar decomposition of `F`), the APIC affine `C`, the velocity `v`, the plastic record
`Jp`, and the material code `z_m`, one MLP outputs the material-frame stress (3 symmetric components),
the plastic correction to the stretch (3 components) and the change in `Jp`. `F` is remounted as
`R (S + dS)` and the stress is rotated back as `R tau R^T`.

Untouched and analytic: the B-spline P2G and G2P, the grid update, gravity, the separating walls with
Coulomb friction, advection, per-material density and volume.

The network is evaluated ONCE per substep, fused into the bottom of G2P, with its stress cached for the
next P2G. That is exact rather than approximate -- the stress P2G scatters at step n is a function of the
`F` that G2P produced at step n-1 -- and it keeps the dispatch count at three, which matters because an
empty dispatch on this device costs {C['dispatch_floor_us']:.2f} us.

## Three declared deviations of the STEP (the law and every parameter are canonical)

1. Stress is cached one kernel earlier rather than recomputed at the top of P2G.
2. `F` is remounted as `R S'` where canonical stores `U diag(s') V`. The two differ by a rotation on the
   right of `F`, which is unobservable for an isotropic model.
3. The fluid carries its volume ratio inside `F` (isotropised every substep) instead of in a separate
   scalar `J`, so the network has ONE state representation to learn for all four materials.

All three were gated before training: `train/gate_oracle.py` runs the analytic law through the
reparameterised step and checks it against canonical at the STEP level (mean position difference after one
substep: rubber 4.7e-12, snow 3.0e-9, sand 3.3e-9, fluid 9.5e-8) and against the golden signatures
(21 pass, 0 fail, 4 N/A).

## Two latents, kept apart

`z_m` is IDENTITY: four fixed codes at the corners of a regular simplex in R^4, separation
{M['capacity']['z_sep']:.3f}, jittered by {M['capacity']['z_jitter']:.3f} every training batch so the
network learns a neighbourhood rather than four point lookups. Never updated during a rollout.

The carried state is HISTORY: `S` and `Jp`, per particle, updated every substep in the known
parameterisation, with the network predicting the update. A free learned latent state was out of scope
because discovering one requires backprop through a long rollout.

## Training

Supervised regression, not backprop through time. The target is an exact analytic function of the state,
so any input distribution gives correct labels and the only job of the data is coverage. 32.4M labelled
states came from oracle rollouts on every canonical scene and material, augmented by the isotropy
symmetry (conjugating the material frame by a random rotation maps a valid pair to another valid pair
exactly) and by a jitter shell of perturbed-and-relabelled states just off the visited manifold. Half the
samples have their C and v shuffled across particles, because for this seam those six inputs carry no
information and the ablation confirms it.

Two loss details mattered. Per-material output weighting, because rubber's stress spread is 9x snow's and
a globally-whitened MSE leaves snow no better than its own mean. And a robust (16-84 percentile) input
scale rather than a standard deviation, because the APIC affine has heavy tails.

## What each efficiency lever bought

| lever | effect |
| --- | --- |
| fusing into G2P instead of a fourth dispatch | avoids {C['dispatch_floor_us']:.2f} us/substep of pure launch latency |
| weights in a uniform buffer, not a storage buffer | about 3x, and removes the linear-in-width memory term |
| letting the compiler unroll the hidden loop | 2-3x below width 88, nothing above it |
| f16 weights | slower below width 88, ~1.4x faster above it, and unusable for accuracy here |
| batching substeps into one submit | 61 us -> 23 us per substep of wall-clock host cost, a factor of 2.6 |

## Parity, host against shader

The MLP alone, on 4096 real feature vectors: max absolute disagreement
{PAR['mlp']['max_abs']:.2e} against an output spread of {PAR['mlp']['out_sd']:.2f}, i.e.
{PAR['mlp']['max_rel_to_sd']:.1e} relative, in f32. In f16 the same comparison gives
{PAR['mlp_f16']['max_abs']:.2f}, which is {PAR['mlp_f16']['max_rel_to_sd']:.2f} of the output spread --
f16 is a cost lever, not an accuracy-neutral one.

The whole learned simulator, same initial condition, host against shader: {PAR['rollout']['frame1']:.1e}
mean per-particle distance after the first frame, growing to {PAR['rollout']['traj_rmse_host_vs_wgsl']:.3f}
over 30 frames. The growth is the expected chaotic divergence of two f32 orderings of the same
arithmetic; the first-frame number is the one that certifies the port.

## Traps this run paid for

* The WebGPU port ran every material at one density and one friction coefficient, which made snow 3.3x
  too soft (E/rho = 45 instead of 150) and its drop spread 0.28 against canonical's 0.46. Fixed by
  scattering per-material mass and mass-weighted friction into a widened momentum buffer -- deliberately
  not a ninth storage buffer, which would have silently invalidated the bind group.
* The first cost sweep seeded the scene once and let every probe advance it, so early repetitions timed a
  falling blob and late ones a settled puddle. The tell was a control: P2G+grid is byte-identical across
  variants and drifted 56%. Re-seeding to a fixed warmed state before every probe brought it to
  {C['control_pg_drift_pct']:.1f}%.
* The first per-material loss weighting gave rubber's exactly-zero plastic target a weight of 64, so the
  optimiser learned to emit zero and left snow's stress at 1.02 relative error. A constant target needs
  weight one, not infinity.
"""

RESULTS = [
    {"type": "video", "src": REL + "learned_vs_canonical_heap.mp4",
     "caption": "The claim is about motion, so both sides are video, same scene and same seed. Top "
                "row canonical, bottom row the one shared network (width 64, two hidden layers -- the "
                "most accurate net trained here). The angle-of-repose scene separates all four "
                "materials: canonical fluid runs flat, rubber and snow keep the seeded slope, sand "
                "yields to a finite one. The learned versions all do the same thing as each other, "
                "which is the failure."},
    {"type": "video", "src": REL + "learned_vs_canonical_drop.mp4",
     "caption": "A dropped disk, canonical above and learned below. Canonical spread widths are 0.40 "
                "(fluid), 0.18 (rubber), 0.35 (snow), 0.66 (sand); the learned ones are 0.84 to 0.90 "
                "for all four."},
    {"type": "image", "src": REL + "fig/cost_vs_width.png",
     "caption": "The cost answer. Whole-solver microseconds per substep against hidden width, with "
                "both real-time lines drawn and the analytic four-material solver marked as the "
                "baseline it replaces. The analytic law sits essentially ON the quarter-GPU line, "
                "which is why widths 8 and 16 are free."},
    {"type": "image", "src": REL + "fig/unroll_cliff.png",
     "caption": "The control that identifies the cliff. Blue is the shipped shader (loop bound a "
                "literal, so the compiler may unroll); yellow is the identical shader with the bound "
                "read from a uniform, so it may not. Below width 88 unrolling is worth 2-3x; above it "
                "the two curves coincide."},
    {"type": "image", "src": REL + "learned_vs_canonical_heap_final.png",
     "caption": "Final states of the repose scene with each panel's measured repose angle and spread "
                "width, for a reader who will not press play."},
    {"type": "image", "src": REL + "fig/capacity_vs_cost.png",
     "caption": "Held-out one-step error against width, per material, in units of each material's own "
                "spread, with the real-time width cutoffs shaded. Snow is the worst-fit material at "
                "every width. The error is still falling at the largest width measured."},
    {"type": "image", "src": REL + "fig/traj_error.png",
     "caption": "Trajectory error read against the two references that make it interpretable: the "
                "ORACLE floor (the same scaffolding running the exact analytic law) and the 1e-7 "
                "initial-condition nudge band. Against zero the learned numbers would mean nothing."},
    {"type": "json", "src": REL + "metrics.json",
     "caption": "Every number on this page, including the full width sweep, the per-material one-step "
                "errors at five widths, the golden signature rows, and the host-vs-shader parity."},
]


def main():
    html = (RUN / "bespoke_page.html").read_text(encoding="utf-8")
    ok = True
    for r in RESULTS:
        p = RUN.parents[2] / r["src"]
        if not p.exists():
            print("MISSING MEDIA:", r["src"])
            ok = False
    if not ok:
        raise SystemExit("refusing to write a manifest with a dangling media src")
    man = {
        "schema_version": "2",
        "task_id": "one-latent-conditioned-network-for-all-four-materials",
        "direction": "learned-dynamics",
        "title": "One latent-conditioned network for all four materials, on WebGPU",
        "tldr": ("One network with one weight set runs fluid, rubber, snow and sand on WebGPU and is "
                 f"free up to width {FITQ} because the analytic law it replaces is expensive -- but at "
                 "every width tried it fits each step well enough and still turns all four materials "
                 "into the same scattering dust over a full rollout."),
        "status": "active",
        "created": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "physics_version": phys.VERSION,
        "objective": (
            "Two independent questions about one design. CAPACITY: can a single network with a single "
            "shared weight set reproduce all four canonical materials -- fluid, elastic, snow, sand -- "
            "with the material identified only by a fixed latent code? COST: can that network run in "
            "real time on WebGPU? The seam is the per-particle constitutive model, meaning the stress "
            "AND the plastic state update, so snow's clamp and sand's Drucker-Prager return mapping are "
            "learned rather than applied analytically afterwards. P2G, G2P, the grid update and "
            "advection stay analytic. The pass condition is the project's own golden signatures, run "
            "unmodified against the learned simulator."),
        "summary": SUMMARY,
        "findings": FINDINGS,
        "hypothesis": HYPOTHESIS,
        "limitations": LIMITATIONS,
        "results": RESULTS,
        "custom_html": html,
        "training_refs": ["networks-inside-a-kernel", "learned-materials", "real-time-cost",
                          "constitutive-models", "svd-polar", "mls-mpm-forward",
                          "differentiable-materials"],
        "metrics_used": ["us_per_substep", "dispatch_floor_us", "traj_rmse", "self_noise",
                         "spread_width", "pile_height", "repose_angle", "onestep_rel_err",
                         "golden_signatures_passed", "physics_version", "substeps_per_frame",
                         "rest_depth", "submerged_fraction"],
        "device": M["setup"],
        "code": {
            "learned_simulator": REL + "train/learned_sim.py",
            "network_spec": REL + "train/netspec.py",
            "canonical_gate": REL + "train/gate_oracle.py",
            "dataset": REL + "train/build_dataset.py",
            "trainer": REL + "train/train_mlp.py",
            "webgpu_engine": REL + "web/mpm4nn-webgpu.js",
            "params_generator": REL + "web/gen_params.py",
            "harness": REL + "verify/harness.html",
            "harness_driver": REL + "verify/drive.py",
            "signature_proxy": REL + "eval/sigproxy.py",
            "evaluation": REL + "eval/run_eval.py",
            "figures": REL + "fig/make_figs.py",
        },
        "full_report": FULL,
    }
    (RUN / "manifest.json").write_text(json.dumps(man, indent=1))
    print("wrote manifest.json", len(json.dumps(man)), "bytes; all", len(RESULTS), "media resolve")


if __name__ == "__main__":
    main()
