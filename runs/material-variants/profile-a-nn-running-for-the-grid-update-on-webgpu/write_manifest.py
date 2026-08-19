"""Write manifest.json LAST, and assert that every media src it names actually exists on disk."""
import json
import pathlib
import sys
import datetime

RUN = pathlib.Path(__file__).resolve().parent
REL = "runs/material-variants/profile-a-nn-running-for-the-grid-update-on-webgpu"
ROOT = RUN.parents[2]


def main():
    m = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    html = (RUN / "bespoke_page.html").read_text(encoding="utf-8")
    SPF = m["substeps_per_frame"]
    bud, budq = m["budget_us_per_substep"], m["budget_us_per_substep_quarter_gpu"]

    def gm(h, key="nn"):
        v = [r["grid_us"][key] for r in m["sweep"] if r["hidden"] == h]
        return sum(v) / len(v)

    ana = sum(r["grid_us"]["analytic"] for r in m["sweep"]) / len(m["sweep"])
    g8, g16, g32, g64 = gm(8), gm(16), gm(32), gm(64)

    # The largest particle count each variant still fits, interpolated on the measured curve exactly as
    # spec/registry/metrics.json defines particle_budget_60fps. Hand-writing these was wrong once.
    ns = sorted({r["n"] for r in m["sweep"]})

    def crossing(key, h, lim):
        pts = [(n, next(r[key] for r in m["sweep"] if r["n"] == n and r["hidden"] == h)) for n in ns]
        if pts[0][1] > lim:
            return None
        if pts[-1][1] <= lim:
            return ("beyond", pts[-1][0])
        for (n0, v0), (n1, v1) in zip(pts, pts[1:]):
            if v1 > lim >= v0:
                return ("at", int(round(n0 + (n1 - n0) * (lim - v0) / (v1 - v0))))
        return None

    def budget_cell(key, h):
        q = crossing(key, h, budq)
        if q is None:
            f = crossing(key, h, bud)
            if f is None:
                return "no, at any particle count (and none on the whole GPU either)"
            k, v = f
            return ("no; on the WHOLE GPU it fits beyond %s particles" % f"{v:,}" if k == "beyond"
                    else "no; on the WHOLE GPU it fits to ~%s particles" % f"{v:,}")
        k, v = q
        return ("yes, beyond %s particles" % f"{v:,}") if k == "beyond" else ("yes, to ~%s particles" % f"{v:,}")

    results = [
        {"type": "video", "src": f"{REL}/learned_vs_truth.mp4",
         "caption": "The claim is about motion, so both sides are video. Same seed, same analytic P2G "
                    "and G2P, 20,000 substeps per simulated second; only the grid update differs. Left "
                    "is the canonical grid update, right is the best-trained learned one (width 64, "
                    "derivative loss). The canonical drop falls, hits the floor and splashes. The "
                    "learned one hovers, frays and drifts sideways, because gravity's contribution to "
                    "one substep is 56 times smaller than the network's own fitting error."},
        {"type": "image", "src": f"{REL}/cost_vs_budget.png",
         "caption": "The headline. Whole-solver microseconds per substep against particle count, with "
                    "the analytic grid update it replaces as the solid baseline and the 60 fps budget "
                    "drawn twice: on the whole GPU and derated to a quarter of it. Every learned curve "
                    "is flat in particle count because the grid has 16,384 cells regardless."},
        {"type": "image", "src": f"{REL}/flat_curve.png",
         "caption": "Why the flat curve does not bend. Left: the grid kernel alone, dense and compacted "
                    "to occupied cells, at every particle count. Right: measured speedup against the "
                    "reduction in workgroups dispatched. Issuing 36x less work takes the same time, "
                    "which is the signature of a latency-bound kernel."},
        {"type": "image", "src": f"{REL}/gravity_below_noise.png",
         "caption": "The accuracy mechanism as a measurement. Left: mean particle height over time; the "
                    "canonical drop falls and settles, two learned rollouts leave the frame upward, and "
                    "the best descends at a constant crawl rather than accelerating. Right: each "
                    "network's velocity error against gravity's whole per-substep contribution."},
        {"type": "image", "src": f"{REL}/width_cliff.png",
         "caption": "Cost swept finely over width with untrained networks (cost does not depend on the "
                    "weight values). Two independent passes. Throughput sits near 3,500 GFLOP/s at "
                    "widths 4-20 and 48-128 and collapses to about 1,000 in between, so width 48 costs "
                    "less in absolute terms than width 40 despite 44% more arithmetic."},
        {"type": "image", "src": f"{REL}/clip_contact_sheet.png",
         "caption": "The headline clip as stills, for reading rather than watching. Top row canonical, "
                    "bottom row learned at width 64."},
        {"type": "image", "src": f"{REL}/final_frames.png",
         "caption": "Final frames at t = 1.0 s. The two cell-wise-loss rollouts look like empty panels "
                    "because 82% and 88% of their particles are stacked on the domain boundary where "
                    "G2P's position clamp holds them; the percentage is printed under each panel so "
                    "the degenerate state is legible rather than looking like a broken plot."},
        {"type": "video", "src": f"{REL}/analytic_port_check.mp4",
         "caption": "The baseline is sound before any of it is believed. The analytic WGSL fluid against "
                    f"canonical Taichi from the same seed: mean per-particle distance "
                    f"{m['analytic_port_check']['traj_rmse_vs_canonical']:.5f} against the reference's "
                    f"own run-to-run noise of {m['canonical_self_noise']:.5f}."},
        {"type": "image", "src": f"{REL}/accuracy.png",
         "caption": "What the cost buys. Left: 60 fps frames a learned rollout tracks the canonical one. "
                    "Right: the node velocity error against the error in its spatial derivative, which "
                    "is the quantity G2P actually gathers."},
        {"type": "table",
         "columns": ["grid update", "us per substep", "vs analytic", "fits 60 fps at a quarter GPU?"],
         "rows": [
             ["analytic (the baseline)", f"{ana:.3f}", "1x", budget_cell("full_us_analytic", 8)],
             ["learned, width 8", f"{g8:.2f}", f"{g8/ana:.0f}x", budget_cell("full_us_nn", 8)],
             ["learned, width 16", f"{g16:.2f}", f"{g16/ana:.0f}x", budget_cell("full_us_nn", 16)],
             ["learned, width 32", f"{g32:.1f}", f"{g32/ana:.0f}x", budget_cell("full_us_nn", 32)],
             ["learned, width 64", f"{g64:.1f}", f"{g64/ana:.0f}x", budget_cell("full_us_nn", 64)],
         ],
         "caption": "The grid-update kernel's own cost, isolated by differencing against a null kernel "
                    "with identical memory traffic and no physics. RTX 4090, Chromium, 128x128 grid, "
                    f"canonical water. The budget is {bud:.1f} us per substep on the whole device and "
                    f"{budq:.1f} us at a quarter of it, because {SPF} substeps must fit in 16.7 ms."},
    ]

    tldr = (
        "A network can replace the whole grid update on WebGPU, but it costs 30 to 1000 times the "
        "analytic kernel it replaces, gets no cheaper with fewer particles because it is latency-bound, "
        "and the only size fast enough for real time keeps water looking like water for about a tenth "
        "of a second."
    )

    summary = (
        "**Verdict: not viable at any useful size on this hardware.** The whole grid update - the "
        "division by node mass, gravity, the separating-wall test and the Coulomb friction cap - was "
        "replaced by a per-cell MLP running in WGSL, with P2G and G2P left analytic and nothing applied "
        "to the network's output afterwards. Measured on one RTX 4090 in Chromium at a 128x128 grid, "
        f"the grid kernel's own cost goes from {ana:.3f} us per substep analytic to {g16:.1f} us at "
        f"hidden width 16 ({g16/ana:.0f}x) and {g64:.0f} us at width 64 ({g64/ana:.0f}x). Assuming a "
        f"quarter of this GPU, which caps a substep at {budq:.1f} us for the whole solver, the largest "
        "network that still holds 60 fps is width 16, at about 8,300 particles against the analytic "
        "solver's own 14,400 at the same budget - and a width-16 network "
        "tracks the canonical rollout for 5 frames out of 60 before the water stops being water.\n\n"
        "Two things make this worse than a simple speed problem. First, the cost is **latency-bound**: "
        "dispatching the same kernel over 7 workgroups instead of 256, a 36-fold cut in work issued, "
        "changes the time by one percent, so the usual sparsity levers do nothing and the cost is flat "
        "in particle count. Second, the accuracy failure is structural rather than a training-effort "
        "problem: gravity contributes 4.9e-4 of velocity per substep while the best network's own error "
        "is 2.7e-2, so the term the simulation is driven by sits 56x below the network's noise floor "
        "and the learned fluid visibly hovers instead of falling."
    )

    full_report = f"""## What was replaced, exactly

The seam is the **whole** grid update. P2G and G2P are analytic and unchanged. The kernel that became a
network reads a node's accumulated mass and momentum and writes its velocity, and in the canonical
`sim.physics.core.grid_op` that pass does four things at once:

1. divide the accumulated momentum by the node mass, `v = p / m`
2. apply gravity, `v.y -= dt * g`
3. zero the normal component at a boundary, but only when the node is moving *into* the wall
4. cap the tangential component by Coulomb friction

**All four are inside the network.** No gravity is added to its output afterwards; no wall clamp is applied
afterwards. The vector it emits is the node velocity G2P gathers. The one piece of bookkeeping the learned
kernel keeps is zeroing the atomic accumulators, which `grid_op` does only because it is their sole reader.

Per cell the network takes 8 inputs and emits 2:

    in  : node mass, node momentum x, node momentum y,
          four wall flags (left, right, floor, ceiling), friction coefficient
    out : node velocity x, node velocity y

Mass and momentum are expressed in units of one particle mass. That scale is a single scalar, uniform over
the whole grid and known before the substep starts, so it is folded into the first layer's weights at export
and the shader does no preprocessing. It is a constant fold, not the per-cell division the network has to
learn. The same normalisation the fixed-point atomics already use.

Architecture: a two-hidden-layer ReLU MLP, `8 -> h -> h -> 2`, at h = 8, 16, 32, 64. Trained supervised
against the canonical kernel on canonical water (physics `{m['physics_version']}`), on grid states harvested
from three canonical scenes (drop, dam break, collapsing column) by driving the canonical Taichi kernels and
snapshotting the grid immediately before and after `grid_op`.

**Friction was swept even though water does not need it.** Canonical water has `fric = 0`, which makes the
Coulomb cap the identity, so a network trained only on water would never see step 4 and the claim "the
network does the whole grid update" would be hollow. Each captured pre-state was therefore re-run through the
canonical `grid_op` at three friction coefficients and the coefficient fed as an input. The grid STATES are
all water states; only the boundary law is swept. That is a real limitation and it is listed below.

## Two training stages, and why the second one exists

Stage one fits the operator cell by cell with a mass-weighted loss. The weighting is exact rather than a
convenience: P2G scattered `w_pi * p_mass` into node i from particle p and the weights over a node sum to its
mass in particle masses, so `m_hat_i * |dv_i|` is precisely the total velocity error node i injects into the
whole particle set. It reaches 2.7% mass-weighted error at width 64, which sounds fine and is useless - the
rollout detonates after about 1,800 substeps.

The diagnosis is that **G2P does not read the node velocity.** It reads the affine matrix C, whose entries
carry a 1/dx^2 factor, so what reaches a particle is a weighted spatial derivative of whatever the grid update
wrote. Fitting cell by cell leaves that derivative free, and a small pointwise error with no spatial
correlation produces a derivative error of the same order as the derivative itself. Instrumenting the first
few thousand substeps showed the error in the divergence of the fitted field running at 7 against a true
divergence of 9.

Stage two therefore trains on whole grids with a term on the first differences of the predicted field. It
helps the rollout (survival roughly doubles) and barely moves the metric: relative derivative error goes from
0.41 to 0.38 at width 64. Both weight sets are shipped and both are selectable in the live demo.

## The measurement protocol, and the two ways it was wrong first

Every GPU timing is a `timestamp-query` over a compute pass of 200 substeps - several hundred dispatches, so
the pass is milliseconds long. The clock's own quantum was measured rather than assumed and came out at
{m.get('timestamp_quantum_ns', 32)} ns, far below any interval here. `performance.now()` is clamped to 100 us
in Chromium and is not used for anything short.

The estimator is the **minimum over 11 repetitions**, not the mean or median. This desktop has a dozen other
GPU clients (compositor, chat apps, vendor overlays) and contention can only add time.

Two protocol errors produced confident nonsense before they were found, and both are worth recording:

- **Variants must be interleaved.** Measuring all repetitions of A then all of B lets clock drift under
  sustained load masquerade as a difference between them. The first pass produced a *negative* differenced
  cost for the analytic grid kernel and a sparse kernel twice as slow as the dense one it strictly does less
  work than.
- **Anything that advances the simulation must be re-seeded between variants.** The full-solver phase includes
  G2P, so it moves particles, and the learned kernels move them wrongly. Sharing one state, the analytic
  kernel's turn came round on a scene the network had already piled into a few cells, and it measured seven
  times its true cost from P2G atomic contention alone. Re-seeding before every timed measurement fixed it.
  The phase without G2P cannot move anything, which is why those numbers were stable from the start.

The grid kernel's own cost is isolated by differencing against a **null** grid kernel that performs identical
memory traffic - load the three atomic accumulators, zero them, write the node vec4 - and no physics. That
removes the per-dispatch floor and the memory traffic from the number and leaves the arithmetic. The analytic
result ({ana:.3f} us) sits at the resolution floor of that differencing and should be read as an upper bound.

## Verification, before any timing was believed

- **The analytic WGSL baseline against canonical Taichi**: mean per-particle distance
  {m['analytic_port_check']['traj_rmse_vs_canonical']:.5f} over a 1 s rollout, against the reference's own
  run-to-run noise of {m['canonical_self_noise']:.5f}. Final spread width
  {m['analytic_port_check']['final_spread_width']:.4f} against {m['analytic_port_check']['ref_final_spread_width']:.4f}.
- **Non-zero motion asserted**: particles moved {m['analytic_port_check']['moved_by_frame1']:.5f} domain
  lengths by frame 1. A dropped dispatch from an over-limit bind group produces a beautiful flat curve over
  all-zero data, and that has happened in this project before.
- **The WGSL MLP against the same weights on the host**: largest disagreement over all 16,384 cells was
  {max(r['max_abs_diff'] for r in m['inference_verification']):.1e} absolute, relative
  {max(r['max_rel_diff'] for r in m['inference_verification']):.1e} - float32 rounding. The trick for
  recovering the exact per-cell input is the null kernel, which writes the raw node momentum into the display
  buffer; a second pass over the same particle state then runs the MLP over a bit-identical grid, because
  fixed-point accumulation is integer addition and reproduces exactly whatever order it happens in.
- **Dense against empty-cell-skipping**: identical to the last bit on every cell any particle can gather from.
- **Storage buffers**: 7 per stage against the guaranteed limit of 8. The fluid state is packed so velocity
  and the volume ratio J share one vec4, which is what leaves room for the weights buffer.
- **GPU errors raised during the whole run**: none.

## The numbers

Grid-update kernel alone, microseconds per substep, averaged over all five particle counts (each is flat in
particle count to within a few percent):

| kernel | us/substep | multiple of analytic |
|---|---|---|
| analytic | {ana:.3f} | 1x |
| learned, width 8 | {g8:.2f} | {g8/ana:.0f}x |
| learned, width 16 | {g16:.2f} | {g16/ana:.0f}x |
| learned, width 32 | {g32:.1f} | {g32/ana:.0f}x |
| learned, width 64 | {g64:.1f} | {g64/ana:.0f}x |

Per-dispatch floor: {m['dispatch_floor_us']:.2f} us, corroborating the 1.11 us measured by the earlier
analytic port on the same device. Substeps per 60 fps frame at water's canonical dt: {SPF}, so the whole
solver may cost {bud:.1f} us per substep, or {budq:.1f} us at a quarter of the device.

Largest learned width that still holds 60 fps, by particle count:

| particles | analytic us/substep | whole GPU | a quarter of it |
|---|---|---|---|
""" + "\n".join(
        "| {n:,} | {a:.1f} | {f} | {q} |".format(
            n=r["n"], a=r["analytic_full_us"],
            f=("width %d" % r["max_width_nn_full_gpu"]) if r["max_width_nn_full_gpu"] else "none",
            q=("width %d" % r["max_width_nn_quarter_gpu"]) if r["max_width_nn_quarter_gpu"] else "none")
        for r in m["verdict"]["rows"]) + f"""

## Why compaction does not rescue it

A dense grid update evaluates the network once per cell per substep and the cell count is fixed by the
resolution, so the cost cannot depend on the particle count. The obvious repair is to run it only on cells
that hold material, and that repair is **exact rather than approximate**: G2P gathers from precisely the cells
P2G scattered into, so a cell with zero mass cannot be read by any particle and whatever is written there is
unobservable. Two versions were measured.

- **Empty cells exit early** (measured, shipped as a kernel variant): saves a few percent. It keeps all 256
  workgroups and merely makes some of them return early.
- **The dispatch is compacted** (measured as a cost proxy: the same kernel over the number of workgroups an
  occupied-cell list would need, which prices the dispatch but not the list-building pass): at 512 particles
  that is 7 workgroups instead of 256, a 36-fold cut in the work issued, and it changes the time by
  {min(r['speedup'] for r in m['compaction']):.2f}x to {max(r['speedup'] for r in m['compaction']):.2f}x.

Issuing 36 times less work in the same elapsed time has one explanation: the kernel is **latency-bound**.
16,384 cells is 256 workgroups of 64, which does not come close to occupying a 4090, so the elapsed time is
set by how long one thread takes to walk its own network - a serial chain of dependent multiply-accumulates
each waiting on a weight load - and adding threads is free until the machine fills. It does not fill.

The consequence is that the usual sparsity lever does nothing here, and the only thing that shortens the time
is a shorter dependency chain per cell, which means a smaller network.

## Cost is not proportional to arithmetic either

Cost does not depend on what is in the weight buffer, so the width axis was swept at fine spacing with
untrained networks. Two independent passes agree. Achieved throughput is near 3,500 GFLOP/s at widths 4-20
and again at 48-128, and drops to roughly 1,000 across 24-40, so **width 48 costs less in absolute terms than
width 40** despite 44% more arithmetic. Observed, reproducible, and not explained here. The hypothesis is that
the two hidden activation vectors are function-scope arrays that the shader compiler keeps in registers only
up to some size and spills to scratch memory in the middle band; what would test it is reading the generated
ISA, or holding the hidden state in workgroup memory and seeing whether the band disappears.

## What the cost buys

| trained with | width | 60 fps frames tracking truth | seconds of fluid | how it ended |
|---|---|---|---|---|
""" + "\n".join(
        "| {t} | {h} | {f:.0f} | {s:.3f} | {r} |".format(
            t=("derivative loss" if s["net"].startswith("deriv") else "cell-wise loss"),
            h=s["hidden"], f=s["frames_60fps_tracked"], s=s["sim_seconds_tracked"],
            r=s["reason"] or "-")
        for s in m["survival"]) + """

"Tracking" means the mean per-particle distance from the canonical rollout stayed under 0.05 of a domain
length, about a fifth of the drop's own diameter. The best result is 8 frames out of 60.

The mechanism is quantitative and is the sharpest number in the run. Gravity's entire contribution to one
grid update is `dt * g = 4.9e-4` of velocity, about a thousandth of a typical node speed; it only becomes a
falling drop because it is applied 20,000 times a second. The trained networks' own mass-weighted velocity
errors are 1.4e-1 (width 8) to 2.7e-2 (width 64), i.e. **289x down to 56x gravity's per-substep effect**. The
term the whole simulation is driven by sits far below the network's noise floor, and the visible consequence
is exactly what that predicts: the learned fluid does not fall. More training does not touch this - at width
64 the network would have to become 56 times more accurate before gravity was visible to it at all.
"""

    man = {
        "schema_version": "2",
        "task_id": "profile-a-nn-running-for-the-grid-update-on-webgpu",
        "direction": "material-variants",
        "title": "Profile a NN running for the grid update on WebGPU",
        "tldr": tldr,
        "status": "active",
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "physics_version": m["physics_version"],
        "objective": (
            "Answer one question with measurements: is it viable to replace the ENTIRE analytic grid "
            "update with a learned network on WebGPU, and at what network size and particle count? The "
            "seam is stated exactly - P2G and G2P stay analytic, and the whole grid update becomes a "
            "network, gravity and the boundary/friction handling included, with nothing analytic applied "
            "to its output afterwards. This is a profiling task: the network's accuracy is secondary and "
            "its cost is the deliverable, so 'not viable' is an acceptable answer if that is what the "
            "numbers say."),
        "summary": summary,
        "findings": (
            "TESTED: one GPU (RTX 4090), one browser (Chromium), one grid (128x128), one material "
            "(canonical water at dt=5e-5), one architecture family (two-hidden-layer ReLU MLP evaluated "
            "per cell), hidden widths 8/16/32/64 trained and 4-128 swept for cost, crossed with particle "
            "counts 512 to 32,768. Every timing is a GPU timestamp-query over a 200-substep pass, minimum "
            "of 11 repetitions, with the variants interleaved and the scene re-seeded before any "
            "measurement that advances the simulation.\n\n"
            f"COST. The grid kernel's own cost is {ana:.3f} us/substep analytic against {g8:.2f} at width "
            f"8, {g16:.2f} at 16, {g32:.0f} at 32 and {g64:.0f} at 64 - a factor of {g8/ana:.0f}x to "
            f"{g64/ana:.0f}x. It is FLAT in particle count at every width, as it must be: the cell count "
            "does not depend on the particle count. At a quarter of this GPU the largest width that holds "
            "a 60 fps frame is 16, at about 8,000 particles; on the whole device width 32 fits only below "
            "2,048 particles and width 64 fits nowhere.\n\n"
            "WHY IT DOES NOT GET CHEAPER. Skipping cells with no mass is exact (G2P gathers only from "
            "cells P2G scattered into) and saves a few percent. Compacting the dispatch to the workgroups "
            "an occupied-cell list would need - 7 instead of 256 at 512 particles, a 36-fold cut in work "
            "issued - changes the time by 1-2%. The kernel is latency-bound: 16,384 cells is 256 "
            "workgroups, which does not occupy this device, so the time is set by one thread's serial "
            "walk through the network rather than by how many threads there are.\n\n"
            "COST IS NOT PROPORTIONAL TO ARITHMETIC. Swept finely with untrained networks over two "
            "independent passes, achieved throughput is ~3,500 GFLOP/s at widths 4-20 and 48-128 and "
            "~1,000 across 24-40, so width 48 costs LESS in absolute terms than width 40 despite 44% more "
            "arithmetic.\n\n"
            "ACCURACY (secondary, but it bounds the verdict). The best-trained network tracks the "
            "canonical rollout for 5-8 frames out of 60 before the water stops being water. Gravity "
            "contributes 4.9e-4 of velocity per substep and the width-64 network's own mass-weighted "
            "error is 2.7e-2, i.e. 56x larger, so the learned fluid measurably does not fall. Separately, "
            "the relative error in the first difference of the fitted velocity field - which is what G2P "
            "gathers, since C carries a 1/dx^2 factor - is 41% at width 64 with a cell-wise loss and 38% "
            "when trained against the derivative directly, against 2.7% in the velocity itself.\n\n"
            "NOT CLAIMED: nothing here bears on a learned FRAME-to-frame operator, on a different "
            "architecture family, on a finer grid, or on a smaller device."),
        "hypothesis": (
            "HYPOTHESIS (mechanism, not observation). Three, in decreasing confidence.\n\n"
            "1. LATENCY, NOT ARITHMETIC, SETS THE COST, because a 128x128 grid is too small a batch for "
            "this device. The direct evidence is that a 36-fold reduction in dispatched work changed the "
            "time by 1%. The mechanism would be a serial chain of dependent multiply-accumulates, each "
            "waiting on a weight load that cannot be hidden because there are not enough resident warps. "
            "What would test it: the same kernel on a much finer grid, where the workgroup count rises "
            "enough to fill the machine - if this is right the achieved throughput should rise and "
            "compaction should start to pay. A smaller GPU should show the same effect for the opposite "
            "reason.\n\n"
            "2. THE ACCURACY FAILURE IS A SIGNAL-TO-NOISE PROBLEM, NOT A CAPACITY PROBLEM. The grid "
            "update's OUTPUT is dominated by v = p/m, a change of variables that carries no physics; the "
            "physics is a perturbation on it three orders of magnitude smaller. A loss written on the "
            "output therefore spends its capacity on the change of variables. The prediction this makes "
            "is specific and was borne out: the fluid should fail to accelerate downward, and it does. "
            "What would test it more sharply: train on the RESIDUAL against p/m computed analytically, "
            "which would be a partial replacement rather than a whole one but should recover gravity; or "
            "train a frame-to-frame operator, where gravity contributes S*dt*g and is no longer buried.\n\n"
            "3. THE NON-MONOTONIC WIDTH CURVE IS REGISTER SPILLING. The two hidden vectors are declared "
            "as function-scope arrays with dynamic indices, and the shader compiler likely promotes them "
            "to registers up to some size, spills to scratch memory in the 24-40 band, and changes "
            "strategy again above it. What would test it: read the generated ISA, or hold the hidden "
            "state in workgroup memory and see whether the band disappears."),
        "limitations": (
            "1. ONE DEVICE, ONE BROWSER, ONE RESOLUTION. Every number is an RTX 4090 in Chromium at "
            "n_grid=128. The latency-bound conclusion is SPECIFICALLY a claim about 16,384 cells failing "
            "to occupy a very large GPU and would not survive a much finer grid or a much smaller device. "
            "A finer grid raises the analytic baseline too, so the RATIO is what would need re-measuring, "
            "not either number alone.\n"
            "2. ONE ARCHITECTURE FAMILY. A two-hidden-layer ReLU MLP evaluated per cell. Nothing here "
            "tests a convolutional operator over the grid, a network with a different activation, weights "
            "held in a uniform or workgroup buffer, or half precision - all of which could move the cost "
            "substantially and none of which were measured.\n"
            "3. FRICTION IS FITTED ON STATES IT WOULD NOT VISIT. The friction coefficient is an input and "
            "was swept over {0, 0.25, 0.5}, but every grid state in the dataset came from canonical water, "
            "which is frictionless at the boundary. The Coulomb branch is therefore fitted correctly as a "
            "function but on an off-distribution input set. A frictional material was not run.\n"
            "4. THE COMPACTION RESULT IS A COST PROXY. It prices the dispatch (the same kernel over the "
            "workgroup count an occupied-cell list would need), not the list-building pass, and the cells "
            "it happens to touch are not the occupied ones. Since it shows no saving even with the "
            "list-building cost excluded, the conclusion is if anything understated - but it is not a "
            "working compaction implementation.\n"
            "5. ACCURACY WAS NOT PUSHED HARD. Two training stages, a numpy MLP, no hyperparameter search, "
            "no on-policy data aggregation, no architecture search. The claim is not 'this cannot be "
            "fitted better'; it is that the specific quantitative obstacle measured here - gravity being "
            "56x below the fitting error - is not the kind that a better optimiser removes.\n"
            "6. THE ANALYTIC GRID COST IS AN UPPER BOUND. At 0.03-0.09 us/substep it sits at the "
            "resolution floor of the differencing against the null kernel; the true value may be lower, "
            "which would make the ratios larger, not smaller.\n"
            "7. SINGLE SCENE FAMILY FOR ACCURACY. Survival is measured on one drop scene at 4,096 "
            "particles. Different scenes would give different survival times."),
        "results": results,
        "custom_html": html,
        "training_refs": ["real-time-cost", "hybrid-learned-residual", "learned-materials",
                          "mls-mpm-forward", "fixed-point-atomics"],
        "metrics_used": ["grid_update_us", "us_per_substep", "frame_ms", "substeps_per_frame",
                         "dispatch_floor_us", "particle_budget_60fps", "traj_rmse", "self_noise",
                         "node_velocity_mae_massw", "node_velocity_grad_rel_err",
                         "frames_tracked_60fps", "spread_width", "physics_version"],
        "device": m["device"],
        "code": {
            "engine": f"{REL}/web/mpm-nn-webgpu.js",
            "params_generator": f"{REL}/web/gen_params.py",
            "data": f"{REL}/train/gen_data.py, {REL}/train/gen_grids.py",
            "training": f"{REL}/train/train_mlp.py, {REL}/train/train_grid.py",
            "harness": f"{REL}/verify/harness.html",
            "scoring": f"{REL}/verify/score.py",
            "figures": f"{REL}/verify/plots.py, {REL}/verify/render.py",
            "page": f"{REL}/build_page.py",
        },
        "full_report": full_report,
    }

    missing = []
    for r in man["results"]:
        if "src" in r:
            if not (ROOT / r["src"]).exists():
                missing.append(r["src"])
    if missing:
        print("MISSING MEDIA:", *missing, sep="\n  ")
        return 1
    (RUN / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("wrote manifest.json; every media src resolves (%d results)" % len(man["results"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
