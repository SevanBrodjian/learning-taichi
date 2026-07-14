# Learning the whole material, not just the stress

[[conditioned-fluid]] learned a liquid's **stress** with a conditioned network and left the rest of the
material analytic: the surface-tension force was a hand-written formula and the volume update was the
textbook continuity rule. That is a hybrid, not a learned material. This page takes the harder and more
honest position. Inside a fixed MPM solver, replace the **entire** per-particle material with networks and
keep only the transfer machinery. The question that matters for a controllable world model is not "can a
network fit a stress" (it can), it is "can a network **be** the material" so that everything constitutive
becomes a differentiable, conditionable function instead of frozen code.

## Where a material actually lives in MPM

An MLS-MPM step (see [[mls-mpm-forward]]) touches the material in exactly three places, and nowhere else:

1. **The stress**, at particle to grid. Each particle contributes momentum $p_{\text{mass}}\,v_p + (\text{stress}
   + p_{\text{mass}} C_p)\,(x_i - x_p)$ to nearby nodes. The material enters only through the stress
   $\sigma_p$, a function of the particle's state.
2. **The carried-state update**, at grid to particle. After the grid solve the particle reads back a new
   affine matrix $C_{\text{new}}$ and advances its own memory of deformation. For a liquid that memory is
   the single volume ratio $J$, updated by continuity $J \leftarrow J\,(1 + \Delta t\,\operatorname{tr} C_{\text{new}})$.
   For a solid it is the deformation gradient $F$. Either way this is where the material **remembers**.
3. **Surface tension**, a capillary force applied on the grid at the interface. Unlike the other two it is
   **non-local**: its strength is set by the interface curvature, which no single particle can see.

Everything else in the step, the B-spline weights, the scatter and gather, the mass-normalise, gravity, the
Coulomb floor, advection, is **transfer and boundary scaffolding**. It is the same for water, sand, or
rubber. So "learn the whole material" has a precise meaning. Learn the three pieces above; import the rest
unchanged. If any of the three is still an analytic equation in the rollout, the material is not learned, it
is decorated.

## Three learned pieces, one descriptor

The material is a small set of networks, all selected by one two-scalar descriptor $m = (m_{\text{visc}},
m_{\text{st}})$ fed as extra inputs (the [[conditioned-material-net]] protocol: interpolate the input, not
the weights).

- **Stress.** A per-particle network maps the local state $(J, C_p, v_p)$ and $m$ to the world stress
  $\sigma_p$. Viscosity is a **local** law, the viscous stress $\mu\,(C_p + C_p^\top)$ is a pointwise
  function of the particle's own affine matrix (see [[viscosity]]), so the local inputs suffice and
  $m_{\text{visc}}$ becomes a viscosity dial.
- **Surface tension**, folded into the same particle-to-grid scatter as a **per-particle body force**. The
  particle reads a $5\times5$ patch of the smoothed grid density $\phi$ around it, plus $m_{\text{st}}$, and
  outputs a capillary force. Handing the particle that density window is what resolves the non-locality: the
  curvature $\kappa = -\nabla\cdot(\nabla\phi/\lVert\nabla\phi\rVert)$ is a second difference of $\phi$
  reaching two cells out (see [[surface-tension]] and [[vector-calculus]]), so a $5\times5$ patch is exactly
  the support the force depends on. Scattered as momentum $p_{\text{mass}}\,\Delta t\,f_{\text{cap}}/\rho$
  and then mass-normalised on the grid, a per-particle force reproduces the same velocity increment
  $\Delta t\,f/\rho$ the grid-level capillary force would apply, so no separate surface-tension pass is
  needed.
- **State evolution.** A second small network reads $J$, the post-solve affine $C_{\text{new}}$, and $m$,
  and outputs the volume-rate $\dot J$. The rollout advances $J \leftarrow J + \Delta t\,\dot J$, which
  replaces the analytic continuity rule outright. The integrator still multiplies a rate by $\Delta t$, but
  that is numerics, not material physics; the material's volumetric response $\dot J$ is the net's output.

For a liquid the state law is almost trivial, $\dot J = J\,\operatorname{tr} C_{\text{new}}$, and a network
learns it to a few percent without effort. The point of learning it anyway is generality. For a solid the
carried state is the deformation gradient and its update is where plasticity, hardening, and every
history-dependent effect live. A method that can only learn the stress is stuck with hand-written state
rules; a method that learns the state update too can, in principle, learn materials with memory. That is the
version a controllable simulator needs.

## Verifying the scaffolding is really the ground truth

A subtle failure mode of "keep the transfer fixed" is quietly forking it. The canonical physics is frozen
and versioned for exactly this reason. The learned step here reuses the canonical stress building block and
gather, and splits the grid update only so a ground-truth surface-tension pass can sit between gravity and
the boundary. To prove the split changed nothing, the custom "true" step with surface tension off is run
against the canonical `simulate` and matches it to a trajectory RMSE around $10^{-6}$, i.e. GPU-noise. Only
then is the scaffolding trustworthy as the fixed skeleton the material is dropped into. Freezing the ground
truth and checking against it is not bureaucracy, it is what keeps a long chain of learned-material
experiments provably on the same physics.

## The edge is exact; the interior and the unseen corner are the test

![Grid of the whole learned material (cyan) over the ground-truth liquid (grey) at each descriptor cell,
viscosity increasing left to right and surface tension increasing bottom to top, with three trained corners
starred and the top-right corner held out.](/api/data/learning-taichi/runs/material-variants/train-one-nn-to-mimic-viscosity-and-st/grid_overlay_montage.png)

Trained at three corners of the descriptor square, the whole learned material is **edge-exact** there: the
stress, the capillary force, and the volume rate each reproduce their analytic targets, and the full rollout
tracks the true liquid. That establishes the pieces are individually learnable inside the fixed scaffolding.
The real questions are the **interior** cells (pure descriptor interpolation) and the **held-out** corner,
where the thick-fluid stress and the strong capillary force must **compose** in a regime seen on neither
training axis. Fidelity there is read from the overlay **video** against the grey ground truth, cell by
cell, not from a trajectory-RMSE number, because a spike and a blob can share a center of mass and so score a
deceptively small distance. Judging a learned simulator by a single distance-to-truth scalar is the trap;
the shape and the motion are the evidence.

## Why the training signal is the crux

All the pieces are fit by **per-step supervised regression**: at states drawn from ground-truth rollouts,
match the analytic stress, capillary force, and volume rate. This makes each piece **locally** accurate,
which is why the trained corners look excellent. It does not, by itself, make the **rollout** stable,
because it never sees the rollout. A learned force that is right at every point the ground truth visits can
still be wrong at the slightly-off states the learned rollout drifts into, and those errors **accumulate**
over hundreds of integration steps into a jet or a blow-up. This is **covariate shift**: the training
distribution (the ground truth's own trajectory) is not the test distribution (the learned rollout's
trajectory).

Two cheap countermeasures attack the shift without the cost of differentiating through the rollout. Adding
**input noise** during training forces each piece to stay sane on perturbed states. **DAgger** (dataset
aggregation) rolls the current learned material out, collects the off-distribution states it actually
visits, relabels them with the analytic targets, and retrains, so the training distribution grows to cover
the drift. The honest catch is that aggregation is only as good as the rollout it samples: if the current
material is bad, its visited states are garbage and folding them in makes the fit worse, which is exactly
what happens when the base fit is weak. The robust fix is **rollout-based model selection**, keep the
aggregation round whose learned rollout best matches ground truth, so aggregation can only help.

Neither of these is the real answer, and it is worth being clear about that. The real answer is training
**through** the rollout under a trajectory loss (the differentiable-simulation route of
[[differentiating-the-rollout]]), which puts error accumulation directly in the objective and buys
long-horizon stability at the price of a much harder gradient. Per-step supervision plus DAgger is the
pragmatic middle: it learns the whole material cleanly at the corners and covers mild drift, but it treats
long-horizon stability as a property it hopes for rather than one it optimizes. For a controllable world
model, where a learned material has to stay stable under a controller pushing it into unfamiliar states,
that gap is the whole game, and closing it is what [[differentiating-the-rollout]] is for.

## What's open

Learning the state update is demonstrated on the easy case, a liquid whose only carried state is a scalar
volume with a linear evolution law. The hard and interesting case is a solid, where the state is the
deformation gradient and the update carries plasticity and hardening; whether a network learns that update
as cleanly is untested and is the natural next step. The surface-tension force is learned but trained against
the analytic continuum surface force, so it inherits that reference's diffuse-interface approximation and is
tied to the grid resolution and smoothing count the density patch was built at. The per-particle capillary
force is not explicitly mean-subtracted, so a small net momentum can leak on a strongly asymmetric
interface. And composition across the two descriptor axes at the held-out corner is precisely where a
per-step objective is weakest, which is the recurring lesson: fitting instantaneous material laws is cheap
and local, keeping the resulting simulator faithful over a long rollout is a global property that a per-step
loss does not buy.
