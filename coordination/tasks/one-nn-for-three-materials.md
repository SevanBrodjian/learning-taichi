# Worker brief: One conditioned NN for three materials, and a 2-parameter material grid

> Direction: `material-variants`. Task id: `one-nn-for-three-materials`.
> Follow-up to `train-material-replicating-nns-and-interpolate`, and the **honest alternative that task's
> training page predicted**: it showed that blending the *weights* of three separate per-material nets gives
> a degenerate interior (the chord leaves the manifold of valid constitutive laws). The fix it argued for is
> to **condition ONE network on a material descriptor** so the whole continuum is trained on real physics.
> This task builds exactly that, with a **two-parameter** descriptor, and asks whether interpolating the two
> parameters gives the smooth physical morph that weight-blending could not. **The user flagged this as a
> tough task — work hard and carefully.**

## Your role (paste verbatim into the spawn prompt)
You are a **worker agent** for task `one-nn-for-three-materials`. You are **NOT the orchestrator**. Do not
spawn further agents. Read this brief, do the task, write **all** results to disk under
`runs/material-variants/one-nn-for-three-materials/`, add a training page, and exit. **Do not commit.** Fire
the two pings, and **finish the whole task within your turn** — do not launch a long training/render in the
background and end.

## Notifications (exactly two)
```
python harness/tools/notify.py --kind started  --task one-nn-for-three-materials "<one plain sentence>"
python harness/tools/notify.py --kind finished --task one-nn-for-three-materials "<one plain sentence>"
```
`--kind blocked` on a hard stop. One human sentence, never a metrics dump.

## Objective
Train **one** network with **one shared weight set** to reproduce all three materials — weakly-compressible
fluid, corotated elastic, Stomakhin snow — where the material is selected by a **small material descriptor
$m$ of exactly TWO scalar parameters** fed as extra inputs. Same weights for every material; only $m$
changes. Then, instead of interpolating weights, **interpolate the two parameters** and characterize the
result. Visualize the outcome as a **2-D square grid** with one parameter on each axis. The hard requirement:
at every **edge/corner where $m$ equals a trained material's setting, the conditioned rollout must reproduce
that material's simulation exactly** (verified, to simulator-noise level), just like the endpoint-parity fix
in the precursor task.

## Design (this is the crux; get it right)
- **Two parameters, and what they control.** Pick exactly two scalars $m = (m_1, m_2)$ that span
  fluid→elastic→snow, and let them condition **both** the stress network **and** the unified state kernel
  from `train-material-replicating-nns-and-interpolate` (the isotropization that keeps $F$ volumetric for a
  fluid, and the plastic-clamp yield band for snow — both are state rules that live outside the weights and
  must move with $m$). A natural choice (you may refine it): $m_1$ = solidity (0 = fluid, $F$ kept
  volumetric and no shear stress; 1 = solid, full $F$ with shear stress), $m_2$ = plasticity (0 = elastic, no
  clamp; 1 = snow, clamp fires). Then the three trained materials sit at fixed points — e.g. fluid
  $(0,\cdot)$, elastic $(1,0)$, snow $(1,1)$ — and the grid sweeps the square between them. State your mapping
  explicitly and justify it.
- **One conditioned net.** A single MLP $g_\theta(\text{features}, m)$ whose position-free features are the
  same as the precursor (the polar stretch $S$ of $F$, the APIC affine matrix $C_p$, velocity, plastic
  record) **plus the two-parameter $m$**, outputting the material-frame stress. Train it **jointly** on all
  three materials at once (each training sample tagged with its material's $m$), on the same kind of varied,
  signature-exercising scenes + mirror augmentation the precursor used (a soft drop, a hard impact that fires
  snow's clamp, a column slump, a slab, a lateral throw). One shared feature standardization and output
  scale. Reuse the substrate/data machinery in `sim/learned_materials.py` (import/copy — do not mutate it).
- **Edge-exactness is mandatory.** At $m = $ each trained material's value, the conditioned net + the
  $m$-driven state kernel must reproduce that material's own rollout to trajectory RMSE at the level of the
  simulator's run-to-run (GPU atomic) noise — the same bar the precursor's endpoint-parity check hit. Verify
  and report it per material. If an edge does not reproduce, the conditioning or the state-rule schedule is
  wrong and must be fixed before any interior is read (this is exactly the class of bug that was sent back
  last time).

## Experiments / deliverables
1. **Replicate:** the one conditioned net, at each material's $m$, reproduces that material on its scenes
   (diagnostic + overlay clip vs the true simulator). Report the fit per material.
2. **Generalize:** the conditioned net at a fixed material $m$ transfers to a held-out scene (as in the
   precursor). Report the gap.
3. **Edge-exactness:** the parity check above, explicitly, at the three trained $m$-points.
4. **Two-parameter interpolation + the 2-D grid (the headline):** sweep $m$ across the unit square on a grid
   (e.g. 5×5), run the conditioned sim at each grid point, and **characterize the morph honestly** — is it a
   smooth, physical progression of materials (the outcome the precursor predicted conditioning would give),
   or does it break somewhere? Note the untrained corner (e.g. isotropic + plastic) may be ill-defined; report
   whatever it does. **This is the direct test of the precursor's prediction** that conditioning a single net
   beats blending separate nets, so compare the two outcomes explicitly.
5. **Visualization:** a **square grid** with $m_1$ on one axis and $m_2$ on the other, each cell a
   representative frame (a montage image) **and** a grid-sweep video, with the trained materials marked at
   their cells. An **interactive HTML** version is a strong bonus — the schema-v2 manifest has a `custom_html`
   field that the dashboard renders in a sandboxed iframe (scripts allowed), so a self-contained grid widget
   (pick/hover a cell, see the material) would render live; do it if you can, but the static montage + video
   are required.

## Evidence discipline (non-negotiable — see CLAUDE.md)
- Scope to this architecture, these two parameters, these materials and scenes. One conditioned net on three
  materials is a **demonstration**, not a general law.
- Keep separate: replicates each material, generalizes, edges are exact, and what the interior grid does. A
  smooth morph is a real positive result **only if the edges are verified exact and the interior is actually
  physical** (viewed, finite, settling into plausible materials) — do not overclaim a morph from a diagnostic
  alone. If it does not fully work, say where and why.
- Manifest carries honest `hypothesis` and `limitations`.

## Visualization standard (graded)
- The 2-D grid (montage + video), the edge-parity overlays, the per-material replication and generalization
  clips. Same-scale, labeled panels; the grid axes clearly labeled with the two parameters.
- **View every clip/plot/grid-cell before writing findings.** A grid cell that blew up, froze, or scattered
  is a bug to diagnose, not a result. Regenerate anything degenerate.

## Training textbook contribution (required)
Add **one short, standalone** page (suggested `reports/training/core/16-conditioned-material-net.md`, id
`conditioned-material-net`) in the impersonal textbook voice: conditioning one network on a small material
descriptor instead of blending separate nets, why interpolating an **input** parameter can give a physical
morph where interpolating **weights** could not, and the 2-D material space that results. Tie to
`[[learned-material-interpolation]]` (the precursor + its prediction), `[[learned-viscosity-interpolation]]`,
`[[material-showcase]]`, `[[constitutive-models]]`, `[[svd-polar]]`, `[[differentiable-materials]]` (all
exist — every `[[link]]` must resolve). Embed a viewed figure (the grid or an edge-parity panel); captions
plain prose (no `$math$`). Render-check the KaTeX. **Do NOT edit `reports/training/index.json`** — leave it
untouched; the orchestrator registers your page. In your final message, give the page **id, title, filename**.

## Output contract
Write `runs/material-variants/one-nn-for-three-materials/manifest.json` (schema v2 — copy the shape from
`runs/material-variants/train-material-replicating-nns-and-interpolate/manifest.json`) plus media, with
`objective`, scoped `findings`, `hypothesis`, `limitations`, typed `results[]` (replicate/generalize clips,
edge-parity image, the grid montage image + grid video, a table; optional `custom_html` for the interactive
grid), and `training_refs[]`. **Write the manifest LAST, after every media file it references exists, and
reference only files that exist** (no dangling media). Leave everything on disk; do not commit.

## Paths & params
- Run dir: `runs/material-variants/one-nn-for-three-materials/`
- New code: `sim/one_nn_materials.py`. Reuse the true physics, the unified state kernel, the features, and
  the multi-scene data from `sim/learned_materials.py` and `sim/material_showcase.py` (import/copy — **do not
  mutate shared files**). Keep per-material `dt` stable (snow/elastic need a smaller `dt` than fluid — CFL).

## Definition of done
- **One network, one weight set, all three materials** selected by a **two-scalar** $m$; each material
  replicated, and generalization shown for at least one.
- **Edges exact:** at each trained $m$ the conditioned rollout reproduces that material to simulator-noise
  level, verified and reported.
- A **2-D parameter grid** visualization (montage + video, axes = the two parameters, trained materials
  marked) with an honest characterization of the interior morph, explicitly compared to the weight-blend
  degeneracy from the precursor.
- Every clip/grid-cell **viewed**; nothing degenerate ships. Training page renders (KaTeX), reads standalone,
  **every `[[link]]` resolves**; `index.json` untouched (report the page id/title/file). Manifest complete
  schema-v2, written last, referencing only existing media.

## Known failures to avoid
- **The edge-exactness is the load-bearing requirement** and the exact thing that was buggy last time — at a
  trained $m$ the model must reproduce that material identically (verify it before trusting the interior). A
  wrong state-rule schedule or a net that ignores $m$ will fail this.
- **Do not overclaim a smooth morph.** View every grid cell; a diagnostic that varies smoothly is not proof
  the cells are physical materials. The honest result is whatever the viewed grid actually shows.
- Keep the conditioned rollout stable (a bad $m$ can blow up); the untrained corner may be degenerate — report
  it, do not hide it.
- **Finish within your turn** (do not spawn a long background job and end), and **write the manifest last,
  referencing only media that exists** (both bit prior workers). Headless only (no `ti.GUI`).
