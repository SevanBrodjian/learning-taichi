# The registry — standardize across tasks wherever possible

```
spec/registry/
  README.md       this file — the policy
  metrics.json    HAND-AUTHORED. Metrics live across many task files; each entry cites its own file:line.
  materials.json  GENERATED from sim.physics by harness/tools/sync_registry.py — never edit by hand.
```

Everything here is served merged at **`/api/definitions`**, and the dashboard renders any registered term
as a click/hover definition. Splitting by kind keeps it legible and makes the whole directory liftable
into the next project alongside `harness/`.

Adding a new kind of standard (scenes, scenarios, seeds) means adding a file here, not growing one.

**The principle:** when several tasks measure or model the same thing, they must measure or model it the
**same way**, from **one** definition, or the task sequence stops being comparable and quietly accumulates
nonsense. A follow-up that redefines its parent's metric or material is not a follow-up — it is a different
experiment wearing the same name.

This applies at three levels, in decreasing order of how well the repo currently enforces it:

| level | canonical source | enforcement today |
| --- | --- | --- |
| **Physics / ground truth** | `sim/physics/` (`PROMOTION.md`, `signatures.py`) | Strong for forward GT. **Leaks for differentiable variants — see below.** |
| **Metrics** | `spec/registry/metrics.json` | New. The registry is the source of truth. |
| **Scenes / scenarios** | not yet centralised | Open. Drop scenes, dam breaks, and blob seeds are re-specified per task. |

## Metrics — the registry

`spec/registry/metrics.json` holds every registered metric: what it means, the actual formula, units, expected
range, the **source file and line**, and any cautions. The dashboard reads it to show definitions on hover.

Rules:

1. **Before inventing a metric, check the registry.** If it exists, use it unchanged, by name.
2. **Every entry cites real code.** If you cannot point at the function that computes it, it does not go in.
   A definition written from memory is how the drift starts.
3. **A new metric gets registered in the same run that introduces it.** Not later.
4. **Report the registered name**, not a private synonym, so results are comparable across tasks.
5. **If a metric's name is misleading, say so in `caution`** rather than silently renaming it — old runs
   already used the name and their numbers must stay interpretable.

### Why this exists — a real failure it would have caught

`traj_rmse` is called "trajectory RMSE" everywhere in this project. It is **not** a root-mean-square, and it
is **not** a centre-of-mass distance. It is the mean per-particle Euclidean distance
(`sim/one_nn_whole_fluid.py:619`).

Because nobody had written that down, a worker's manifest explained a result with *"a spike and a blob share
a centre of mass, so the distance reads deceptively small"* — a statement that is simply not true of this
metric. That explanation was then copied into a training-report page and into a task page's headline framing
before anyone checked the implementation. The actual data says the two metrics correlate only moderately
(Spearman $\rho \approx 0.55$), that the held-out corner scores 0.246 against 0.012–0.031 at the trained
corners (so nothing is hidden there), and that the genuine discordance is **two interior cells** whose
distance looks fine while their shape is badly wrong.

**One undefined metric produced a wrong mechanism, propagated through three artifacts.** That is the cost.

## Materials — one snow, and the leak

`sim/physics/core.py` is explicit: *"there is exactly ONE fluid, ONE elastic, ONE snow."* Parameters are
frozen in `MAT`, changing them is a version-bumping, signature-gated event, and `signatures.py` asserts the
qualitative truths (snow crumbles and holds an angle of repose; the fluid/snow/elastic ordering).

**That holds for forward ground truth. It leaks for differentiable variants**, because `CLAUDE.md` permits a
task that must optimize *through* the physics to build its own differentiable copy. Nothing currently forces
that copy to keep the canonical constants, and it has drifted:

| | ξ (hardening) | dt | plastic clamp |
| --- | --- | --- | --- |
| **canonical** `sim/physics/core.py` | **10.0** | 5e-5 | 2.5e-2 / 7.5e-3 |
| `sim/one_nn_materials.py` | **3.0** | 5e-5 | matches |
| `sim/learned_materials.py` | **3.0** | 5e-5 | matches |
| `sim/material_variants.py` | 10.0 | **2e-4** | — |

ξ sets how much compacted snow stiffens, via $h=\exp(\xi(1-J_p))$. **3.0 versus 10.0 is a large physical
difference**, and it means "snow" in the learned-material tasks is a softer material than "snow" in the
canonical library and the showcase. This is precisely the drift `CLAUDE.md` warns about ("snow quietly
starts behaving like elastic across a task sequence"), and it happened.

**The rule going forward:** a differentiable variant may reimplement the *step* (it must, to carry
gradients). It may **not** reimplement the *parameters* or the *constitutive law*. Import the constants from
`sim.physics`, and state in the task contract exactly what differs and why. A variant that silently picks
its own ξ is a defect the reviewer rejects.

**Status: this is a known, unfixed defect.** The affected results are not invalidated — each task is
internally consistent — but cross-task comparisons involving snow are not on common ground, and any claim
comparing those tasks' snow must say so. Fixing it means re-running the affected tasks, which is GPU work
that has not been scheduled.

## Scenes — still open

Drop heights, blob radii, dam-break geometry and floor friction are re-specified per task today. The same
argument applies: a "drop test" that differs between two tasks makes their numbers incomparable. Not yet
centralised; the natural home is `sim/physics/` alongside the materials.
