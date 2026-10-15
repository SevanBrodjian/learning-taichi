# Differentiating the rollout

**The key idea:** a simulation that runs for $T$ steps is just a very long composition of simple
differentiable functions. Nothing about it is special. If you can differentiate one step, you can
differentiate a thousand — the chain rule does not care that the function happens to be physics. What
*is* special is the bookkeeping (you must not throw away anything the chain rule will need) and the
trustworthiness (a number can come back that is finite and still meaningless).

This page is the hinge of the whole project. Everything before it is forward simulation; everything
after it optimizes *through* that simulation.

## A simulation is a function

Start by collapsing the thing. The forward step from [[mls-mpm-forward]] takes a state $S$ and returns
the next state:

$$S_{s+1} = f(S_s;\ \theta),$$

where $S_s$ bundles every particle and grid quantity at step $s$, and $\theta$ is whatever you are
allowed to change — an initial velocity, a material stiffness, the weights of a network. Run it $T$
times and score the result with a loss:

$$\mathcal L = \ell\big(\underbrace{f(f(\cdots f}_{T \text{ times}}(S_0;\theta)\cdots;\theta);\theta)\big).$$

That is the entire object. It looks intimidating written out, but it is structurally identical to a
deep neural network with $T$ layers that happen to share parameters — and we already know how to
differentiate those.

### The concrete first task

The throw task makes it real. Optimize one shared initial velocity $v_0$ so the blob's center of mass
after $T$ steps lands on a target $x^{*}$:

$$\mathcal L = \lVert \bar{x}_T - x^{*}\rVert^2,\qquad \bar{x}_T = \frac1N\sum_p x_{T,p}.$$

Nothing about the physics changes. The simulator is the same simulator. Only $v_0$ is free, and the
gradient $\partial \mathcal L / \partial v_0$ has to travel back through all $T$ steps to reach it.

## Why *reverse* mode

There are two ways to apply the chain rule, and the choice is not stylistic — it decides whether the
computation is affordable.

**Forward mode** propagates derivatives alongside the simulation: start with $\partial S_0/\partial\theta$
and push it forward step by step. Cost scales with the number of *inputs* you differentiate with respect to.

**Reverse mode** runs the simulation forward, then walks backward from the loss:

$$
\frac{\partial \mathcal L}{\partial S_s}
= \frac{\partial \mathcal L}{\partial S_{s+1}}\cdot\frac{\partial S_{s+1}}{\partial S_s}.
$$

Cost scales with the number of *outputs*. You have one scalar loss and potentially millions of
parameters (every weight of a network), so reverse mode wins by orders of magnitude. This is the same
reason backpropagation, not forward differentiation, trains neural networks.

The price is memory, and that price is the rest of this page.

## The three changes versus the forward code

Reverse mode needs to replay the forward computation backwards, which means **every intermediate value
the backward pass will read must still exist when it gets there.** Ordinary simulation code violates
this constantly — it overwrites state in place, because why would you keep step 400 around once you are
on step 401? Three changes fix that.

**1. Time-indexed state.** Each state field gains a leading time axis, so step $s$ *reads* slice $s$ and
*writes* slice $s+1$, and nothing is ever overwritten. In `sim/diffmpm.py` the particle fields `x, v, C, J`
are shaped `(steps, n_particles)` and the grid fields `grid_v_in, grid_m, grid_v_out` are shaped
`(steps, n_grid, n_grid)` — all allocated with `needs_grad=True`, which tells Taichi to carry a shadow
gradient buffer of the same shape ([sim/diffmpm.py:41](sim/diffmpm.py:41)–[48](sim/diffmpm.py:48)).

This is the single biggest structural difference between `sim/mpm88.py` (the 88-line forward seed) and
`sim/diffmpm.py`. The physics is the same. The indexing is what changed.

**2. Split the grid velocity in two.** The raw P2G result and the post-gravity, post-wall velocity live
in *separate* fields, `grid_v_in` and `grid_v_out`. Writing both into one field would be an in-place
update — read a value, modify it, write it back to the same address — and the tape cannot cleanly
attribute a gradient to a slot whose value it no longer knows. Splitting makes the data flow a clean
directed graph.

**3. Wrap the rollout in a tape.** The full forward loop plus the loss runs inside `with ti.ad.Tape(loss):`
([sim/diffmpm.py:174](sim/diffmpm.py:174)). Taichi records the forward graph, then replays it in reverse
to populate `v0.grad`. An optimizer step then nudges $v_0$ downhill, and you repeat.

Read [sim/diffmpm.py:167](sim/diffmpm.py:167) (`optimize`) end to end once. It is short, and it is the
entire idea in one function: tape, read gradient, step, repeat.

## The memory tension, stated plainly

Storing every intermediate costs memory scaling like

$$\text{memory} \;\sim\; \underbrace{T}_{\text{steps}} \times \underbrace{(\text{grid} + \text{particles})}_{\text{state size}} \times \underbrace{2}_{\text{value} + \text{gradient}}.$$

For a short horizon at low resolution this is free. It is also the **first hard wall** you hit when you
scale either the horizon or the resolution, and it arrives faster than people expect because the factor
of two is not optional — every field carries a gradient twin.

The standard escape is **checkpointing**: keep only a sparse set of saved states, and during the backward
pass recompute the missing stretches on demand from the nearest checkpoint. You trade compute for memory,
typically recovering a $\sqrt{T}$-ish memory profile for roughly one extra forward pass. That tradeoff is
a queued research direction here, not yet a result — see the limitations in the resolution/memory task
rather than assuming a number.

## Verifying a gradient — do not take it on faith

**This is the most useful habit on this page.** Autodiff returns a number whether or not that number
means anything. A tape can be technically correct and still hand you a gradient that is useless for
optimization, and it can be silently wrong if the code has an in-place write the framework did not catch.
So check it against something that cannot lie: the definition of a derivative.

Perturb one parameter component by a small $\varepsilon$, rerun the *forward* simulation twice, and
compare the central difference against the autodiff value:

$$
\frac{\partial \mathcal L}{\partial \theta_i} \;\approx\;
\frac{\mathcal L(\theta + \varepsilon e_i) - \mathcal L(\theta - \varepsilon e_i)}{2\varepsilon}.
$$

If those agree to a few significant figures, the gradient is real. If they diverge, something is wrong
and no amount of pretty loss curves will fix it. Several tasks in this project run exactly this check
before reporting anything, and it is the reason a claim like "the residual gradient is correct through a
320-step rollout" is evidence rather than a hope.

Two practical notes. Pick $\varepsilon$ carefully — too large and you measure curvature, too small and
floating-point cancellation eats the signal; somewhere near $10^{-4}$ relative to the parameter scale is
a reasonable start. And check a *few* components, not one, because a single lucky agreement can hide a
systematic error elsewhere.

## Finite is not the same as meaningful

A gradient can pass the check above and still be a bad optimization signal. The mass division in the
grid update has a backward sensitivity scaling like $1/m_i^2$, so a node barely grazed by one particle
can amplify a gradient enormously. The wall condition is a hard branch, non-smooth exactly where contact
happens. Over hundreds of steps these compound, and the result is the exploding, NaN-producing, or simply
uninformative gradients that [[failure-modes]] is entirely about.

Hold the distinction: **existence** (autodiff returns a number), **correctness** (it matches finite
differences), and **usefulness** (it points somewhere worth going) are three separate properties, and
this project has hit failures of all three.

## Why this is the interesting capability

A learned policy can imitate demonstrations. A gradient through the *actual dynamics* tells you the exact
direction in parameter space that improves the outcome, grounded in real physics rather than a surrogate
that may be confidently wrong off-distribution. That is what makes differentiable simulation worth the
memory bill: you are not approximating the world's response, you are querying it.

The catch is that the gradient is only as trustworthy as the smoothness of the step — which is exactly
where [[failure-modes]] begins.

---

**Code for this page:** `sim/diffmpm.py` (time-indexed rollout, tape, optimize loop);
`sim/mpm88.py` for the forward original to diff against.
**Next:** [[failure-modes]] — what breaks, and why it breaks where it does.
