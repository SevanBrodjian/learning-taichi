# Hybrid simulation: a learned residual inside the differentiable step

Everything so far has optimized a *physical* quantity through the simulator. The throw task in
[[differentiating-the-rollout]] tuned one initial velocity $v_0$; the material study in
[[constitutive-models]] sat parameters inside a fixed stress law. In every case the simulator was a
fixed function and the gradient steered a handful of its inputs. This page takes the step that opens the
door to learned dynamics. It puts a small **neural network inside the simulation step** and trains its
weights with the same gradient that used to train $v_0$. The simulator stops being a fixed function and
becomes a function with learnable parts.

This is the first **hybrid**: explicit physics plus a learned component, on one tape. It matters for
controllable world models because a pure simulator can only ever be as right as its hand-written laws,
and a pure neural simulator throws away the physics that is already known to be exactly right. A hybrid
keeps the known physics and learns only the part that is missing or wrong. The question this page settles
is whether that is even trainable, because the network's weights sit behind hundreds of chained physics
steps, exactly the place [[failure-modes]] warned that gradients go to die.

## Where the learned part goes, and why there

The [[mls-mpm-forward]] step has a natural seam. After particle-to-grid scatter and the grid update
`grid_op` produce a post-update grid velocity $\mathbf v^{\text{out}}_i$ at each node $i$, the
grid-to-particle gather reads that velocity back to the particles. Inserting a correction **right
between those two**, on the grid velocity, is the least invasive possible change. The physics that
builds $\mathbf v^{\text{out}}_i$ is untouched, and the physics that consumes it is untouched. Only the
value handed across the seam is adjusted,

$$
\mathbf v^{\text{out}}_i \;\leftarrow\; \mathbf v^{\text{out}}_i \;+\; \alpha\,\tanh\!\big(g_\theta(\mathbf f_i)\big),
$$

where $g_\theta$ is a small multilayer perceptron with weights $\theta$, $\mathbf f_i$ is a short vector
of features read at node $i$, $\alpha$ is a small fixed scale, and the outer $\tanh$ bounds the
correction so it can never inject more than $\alpha$ of velocity per component. The grid is the right
place for this because it is where the field is dense and structured. Every active node carries a
velocity and a mass, the same data the physics already uses, so the network sees a clean physical state
rather than a particle soup.

Two design choices keep this honest and stable, and both are load-bearing.

**The correction is a residual that starts at zero.** This is the residual-network trick imported into a
simulator. The output layer of $g_\theta$ is initialized near zero, so at the start of training
$g_\theta(\mathbf f_i)\approx 0$ and the whole hybrid is byte-for-byte the bare simulator. Training then
*grows* the correction from nothing rather than having to first cancel a large random perturbation. The
identity (here, the unmodified physics) is the natural starting point, and a small bounded residual is a
gentle departure from it. An unbounded or large-at-init residual does the opposite. It blows the
velocity field up and NaNs the rollout before a single useful gradient arrives.

**The features are chosen to decide what the network is allowed to learn.** The MLP is a universal-ish
approximator, so whatever it is *given*, it will fit. Feeding it absolute node coordinates lets it
memorize where a particular trajectory went; withholding them and feeding only velocity and a log-mass
scalar forces it to learn a *velocity-space law* instead, a rule of the form "given this local velocity
and density, output this correction." A velocity-space law is the kind of thing that transfers to a new
initial condition, because the physics it is correcting is itself a velocity-space law. The feature set
is not a detail. It is the lever that decides whether the learned part generalizes or overfits.

## Why the same tape carries the weight gradients

Nothing about the autodiff changes. Reverse-mode differentiation does not care whether a value came from
a physical constant or a network weight; it propagates a sensitivity backward through whatever primitive
operations produced the loss. In [[differentiating-the-rollout]] the chain ran

$$
\frac{\partial \mathcal L}{\partial v_0}
= \frac{\partial \mathcal L}{\partial s_T}\, J_T J_{T-1}\cdots J_1\, \frac{\partial s_1}{\partial v_0},
$$

where $s_t$ is the full simulator state at step $t$, $J_t = \partial \phi(s_{t-1})/\partial s_{t-1}$ is
the per-step Jacobian of the update map $\phi$, and the last factor injects $v_0$ at the very first step.
The network weights enter the identical machinery, only at a different place. The residual at step $f$
makes the update map at that step depend on $\theta$, so the loss sensitivity to $\theta$ accumulates a
contribution from **every** step the residual fired,

$$
\frac{\partial \mathcal L}{\partial \theta}
= \sum_{f} \Big(\frac{\partial \mathcal L}{\partial s_T}\, J_T \cdots J_{f+1}\Big)\,
  \frac{\partial \phi_f}{\partial \theta},
$$

read term by term. The sum over $f$ is there because the *same* weights $\theta$ are reused at every
step, so each step contributes one gradient term and they add, exactly the weight-sharing sum a
recurrent network produces. Inside each term, $\partial \phi_f/\partial\theta$ is how the step-$f$ update
moves when the weights move, computed locally through the MLP and the grid-to-particle gather. The
bracket $\partial\mathcal L/\partial s_T \cdot J_T\cdots J_{f+1}$ is how a nudge to the state at step $f$
propagates forward to the loss, a product of only the Jacobians **downstream** of step $f$. That bracket
is where the interesting asymmetry lives.

## Why a late residual is easier to train than an early parameter

Compare the two gradients factor by factor. The initial velocity $v_0$ enters once, at step one, and its
sensitivity rides the **entire** product $J_T J_{T-1}\cdots J_1$, all $T$ factors. The [[failure-modes]]
page made the general point about such products. When the typical per-step amplification sits below one
the product shrinks geometrically in the number of factors, and the gradient that reaches $v_0$ is the
most attenuated gradient in the whole system. It is the deepest thing in the rollout, buried behind
every step.

The residual is different in two ways that both help. First, its contribution at step $f$ rides only
$J_T\cdots J_{f+1}$, the factors **after** $f$. For a residual that fires near the end of the rollout
that is a handful of factors, not hundreds, so its gradient is barely attenuated at all. Second, the
residual fires at *every* step, so its total gradient is a sum over $f$ of terms with attenuation
ranging from severe (early steps, long downstream product) to negligible (late steps, short downstream
product). The late terms dominate the sum precisely because they are the least attenuated, so the
network always receives a strong, low-variance signal from the tail of the rollout even when the head of
the rollout contributes almost nothing. A parameter at the start gets only the single worst-attenuated
path; a residual reused throughout gets a whole spectrum of paths and is carried by the best of them.

This is the mechanistic reason a network embedded near the consumption end of each step is trainable
through a horizon that would vanish a gradient to the initial condition. It is the same structural
insight that makes residual and skip connections work in deep networks, that a short path to the loss
keeps a gradient alive, transplanted into a physics rollout. For a controllable world model the
implication is direct. The controls that are *easiest* to learn are the ones that act late and often,
close to where their effect is read out, and the hardest are the ones that must author the entire future
from the first instant.

## What this buys, stated narrowly

A residual of this kind can absorb a **model mismatch**: a discrepancy between the simulator's physics
and the dynamics the data actually obeyed. The clean test is to supervise the hybrid against
trajectories generated by the same simulator with one extra physical effect the hybrid is never told
about, for instance a linear drag $\mathbf v \leftarrow \mathbf v\,(1-k)$ applied to the grid velocity.
The bare simulator, lacking the drag, drifts past the target center of mass; the residual's job is to
discover the missing effect from the supervised trajectory alone, with the initial velocity held fixed
so it cannot cheat by re-aiming.

![Three rollouts at one initial velocity, each overlaying its running center of mass (yellow trail) with
the target center-of-mass endpoint marked by the red cross. Left: the drag target (truth). Middle: the
bare simulator, whose blob drifts past the target because it is missing the drag. Right: the trained
hybrid, whose learned grid-velocity residual reproduces the drag and lands the center of mass on the
target.](/api/data/learning-taichi/runs/learned-dynamics/learned-residual/comparison.mp4)

The figure shows the bare simulator overshooting and the hybrid recovering the truth, which is the
visible form of the gradient having reached the weights through the rollout. The honest scope is narrow.
A linear drag is a smooth, near-linear, velocity-only correction, the friendliest possible target for a
velocity-fed residual. The result that gradients through a few hundred physics steps can train such a
network, and that a position-free residual transfers a fair fraction of its correction to an unseen
initial velocity rather than memorizing one path, is real and worth having, but it is a result about
*this* mismatch and *this* architecture. It says nothing yet about non-smooth, strongly nonlinear, or
genuinely unknown dynamics, where both the capacity of the network and the smoothness of the gradient
become live questions rather than settled ones.

## What is open

The residual here is bounded to a small per-component magnitude, adequate only because the missing physics
is weak; a more violent mismatch would need a larger bound (rerisking the blow-up) or a smarter stable
parameterization. Whether the late-residual trainability survives a genuinely hard mismatch (a non-smooth
contact law, a state-dependent drag), and whether the velocity-space generalization holds across blob shape,
resolution, and mismatch type, are what separate a promising demonstration from a method.
