# Worker brief: GPU-accelerate the fluid renderer

> Direction: `realistic-rendering`. Task id: `gpu-accelerate-fluid-renderer`.
> The screen-space fluid renderer `sim/fluid_render2.py` is pure numpy/scipy on the **CPU**: ~0.5-2 s per
> 1080² frame, so a multi-scene showcase took ~25 minutes while a 4090 sat idle. The physics is trivial on
> the GPU; the **renderer** is the whole bottleneck. Port it to the GPU with Taichi so it renders orders of
> magnitude faster **while preserving the visual quality**, and benchmark the speedup. This is also a
> first-class **GPU-aware-design / Taichi** learning task (a core project objective), so teach it well.

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `gpu-accelerate-fluid-renderer`. You are **NOT the orchestrator**. Do not
spawn further agents. Read this brief, do the task, write **all** results to disk under
`runs/realistic-rendering/gpu-accelerate-fluid-renderer/`, add a training page, and exit. **Do not commit.**
Fire the two pings.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task gpu-accelerate-fluid-renderer "<one plain sentence>"
python harness/tools/notify.py --kind finished --task gpu-accelerate-fluid-renderer "<one plain sentence>"
```
`--kind blocked` on a hard stop. One human sentence, never a metrics dump.

## Objective
Reimplement the `sim/fluid_render2.py` rendering pipeline as a **GPU (Taichi) renderer** that produces
visually equivalent (or better) frames far faster, and **benchmark it against the CPU version**. The result
is a drop-in fast renderer (`sim/fluid_render_gpu.py`) that the showcase re-run
(`more-realistic-basic-fluid-sims`) will use. Do **not** change the *look*; change only *where and how fast*
it computes.

## Why it is slow now, and the target
Per 1080² frame the CPU renderer does several full-image passes — a Gaussian metaball splat/blur, a
`distance_transform_edt` for thickness, per-pixel background refraction sampling, Fresnel, specular, foam,
bloom, compositing — single-threaded in numpy/scipy. That is ~0.5-2 s/frame; thousands of frames = tens of
minutes. On a 4090, splatting ~25k particles and shading a 1080² image is a **sub-millisecond-to-a-few-
millisecond** operation. **Target: interactive rates — aim for ≤ ~5 ms per 1080² frame (≥ ~200 fps), so a
few-hundred-frame scene renders in ~1-3 s of GPU compute and the whole multi-scene showcase renders in well
under a minute** (video *encoding* is separate and I/O-bound — see below). Push as fast as you reasonably can
and **report the actual measured throughput**; do not settle for "a few minutes".

## What to build
- **`sim/fluid_render_gpu.py`** — the same pipeline in Taichi kernels, everything on device, reading back
  **only the final RGB frame** (no per-stage host↔device roundtrips, which would kill the speedup):
  - Particle → **density field** by atomic scatter into a Taichi grid field (the metaball splat).
  - A **Gaussian blur** as a Taichi kernel (separable, two passes) rather than scipy.
  - **Surface normals** from the density gradient (as in `fluid_render2.py`).
  - **Preserve the filled-interior / no-holes property** — this is non-negotiable, it is the fix the prior
    task shipped. Find a GPU-friendly equivalent of the CPU `distance_transform_edt` + filled mask: e.g. a
    **jump-flooding** distance transform kernel, or a bounded iterative erosion/closing on the mask, or an
    equivalent smooth thickness proxy that yields the same solid, speckle-free body. A calm body must render
    as a filled volume; genuine cavities (splash craters, a wave barrel) must survive.
  - **Beer-Lambert depth color, background refraction, Fresnel, specular, foam, bloom, tone map** — same
    formulas as `fluid_render2.py`, in kernels.
- **Benchmark (a required deliverable).** On the **same** particle data, time the CPU renderer
  (`fluid_render2.py`) vs the GPU renderer per 1080² frame, and report a table: ms/frame and fps for each,
  the speedup, and the total for a representative scene (e.g. 300 frames). **Run the benchmark on a clean
  GPU with no other heavy job** running (contention corrupts timings — see CLAUDE.md). Warm up the kernels
  once before timing (exclude Taichi's first-call JIT compile from the per-frame number, but report it).
- **Visual parity.** Render the *same* scene/frame with both renderers and produce a **side-by-side still**
  (CPU vs GPU) plus a short GPU clip, so equivalence is visible. Any differences must be improvements or
  imperceptible, never regressions (especially no return of interior holes).
- Note on **encoding**: writing frames to mp4 via ffmpeg is separate from rendering and is I/O/codec-bound;
  if it dominates end-to-end time, mention it and, if easy, stream frames straight to the encoder rather than
  through PNGs. The graded speed metric is the **render** throughput.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- The benchmark is a **timing** result: run it uncontended, warm up first, average over enough frames, and
  report the setup (resolution, particle count, GPU). Scope the speedup claim to that setup.
- Be honest if a stage could not be fully GPU'd or if visual parity required an approximation; show the
  side-by-side so the reader can judge. Manifest carries honest `hypothesis`/`limitations`.

## Visualization standard (graded)
- A **CPU-vs-GPU side-by-side still** (visual parity) and a **benchmark table** (ms/frame, fps, speedup).
- A short GPU-rendered clip proving it looks right in motion, with no interior holes.
- **View every still/clip and check the numbers before writing findings.** If the GPU output regressed
  visually (holes, wrong color, missing foam), that is a bug to fix, not to ship.

## Training textbook contribution (required)
Add **one short, standalone** page (suggested `reports/training/core/15-gpu-rendering.md`, id `gpu-rendering`)
in the impersonal textbook voice: how a screen-space fluid renderer maps onto the GPU with Taichi — atomic
scatter for the splat, separable blur kernels, a jump-flooding (or equivalent) distance transform, keeping
everything on device and reading back only the final image — and the broader lesson that the **render, not
the physics, was the bottleneck**, with the measured speedup. This squarely serves the project's GPU-aware-
design / Taichi learning goal. Tie to `[[fluid-rendering]]`, `[[fluid-realism]]`, `[[mpm-in-context]]` (all
exist — every `[[link]]` must resolve). Embed the viewed side-by-side or benchmark figure; captions plain
prose (no `$math$`). Render-check the KaTeX. **Do NOT edit `reports/training/index.json`** — leave it
untouched; the orchestrator registers your page. In your final message, give the page **id, title, filename**.

## Output contract
Write `runs/realistic-rendering/gpu-accelerate-fluid-renderer/manifest.json` (schema v2 — copy the shape from
`runs/realistic-rendering/improve-basic-fluid-sim-realism/manifest.json`) plus media, with `objective`,
`findings` (the port + the measured speedup), `hypothesis`/`limitations`, typed `results[]` (the CPU-vs-GPU
still, the benchmark `table`, a GPU clip), and `training_refs[]`. Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/realistic-rendering/gpu-accelerate-fluid-renderer/`
- New code: `sim/fluid_render_gpu.py`. Reference `sim/fluid_render2.py` for the pipeline (import/copy for the
  CPU-vs-GPU benchmark), and drive the fluid forward from `sim/material_showcase.py`. **Do not mutate shared
  files in place.** Taichi is `ti.init(arch=ti.gpu)`.

## Definition of done
- `sim/fluid_render_gpu.py` renders the pipeline on the GPU with **visual parity** to `fluid_render2.py`
  (metaball surface, **no interior holes**, depth color, refraction/Fresnel/specular, foam), shown by a
  side-by-side still.
- A **clean benchmark table** showing a large speedup and a per-frame render time in the **low single-digit
  milliseconds** at 1080² (or an honest account if a stage capped it), run without GPU contention.
- Training page renders (KaTeX), reads standalone, **every `[[link]]` resolves**, embeds a viewed figure;
  `index.json` left untouched (report the page id/title/file). Manifest complete schema-v2.

## Known failures to avoid
- **Do not regress the visuals** — especially do not bring back interior holes; the filled-interior property
  must survive the GPU port (find a GPU distance/fill equivalent). Verify by viewing the side-by-side.
- **Do not benchmark under GPU contention** — the orchestrator is running this task alone for exactly this
  reason; still, warm up kernels and average, and exclude first-call JIT from the per-frame figure.
- Avoid per-stage host↔device copies; keep the whole frame on device and read back once. A renderer that
  shuttles arrays to numpy each stage will not be fast.
- Headless only (no `ti.GUI`). Do not spawn a long run then end the turn without viewing outputs and
  confirming the files and the benchmark numbers exist.
