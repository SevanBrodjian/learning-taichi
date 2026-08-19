# What a real-time step costs

> Assumes [[mls-mpm-forward]] (the P2G, grid, G2P skeleton) and [[material-stiffness]] (why stiffness sets
> the timestep). Everything here is about the *cost* of that step rather than its content, because the
> moment a simulation has to answer a pointer in real time, cost is the physics constraint it is under.

## The key idea

A simulation that runs at 60 fps is not a cleverer simulation. It is one whose **substeps per frame** fit
inside 16.7 ms. For an explicit solver that number is not a design choice, it is dictated by the material:

$$
S \;=\; \frac{1/60}{\Delta t}, \qquad \text{frame time} \;=\; S \cdot (\text{cost of one substep}) \;+\; \text{draw}.
$$

Here $S$ is the substep count per displayed frame, $\Delta t$ is the solver timestep, and the $1/60$ is the
wall-clock seconds a 60 fps frame is allowed. The stability condition in [[material-stiffness]] pins
$\Delta t \lesssim \Delta x / c$ with $c \propto \sqrt{E/\rho}$ the elastic wave speed, so a stiff material
forces a small $\Delta t$, a small $\Delta t$ forces a large $S$, and a large $S$ multiplies every
microsecond spent inside a substep by a hundred or more.

For a rubber-stiff 2D solid on a $128 \times 128$ grid this works out to $S = 167$: **one hundred and
sixty-seven full P2G, grid, G2P passes for every frame a person sees**. The entire performance question
collapses to one line: a substep may cost about 100 microseconds, and no more.

That inverts the intuition a graphics-shaped mind arrives with. The particle count is a slider. The
timestep is a wall.

## One grid means one timestep, so the stiffest material bills everyone

The budget equation is written per *simulation*, not per material, and that is not a simplification. A
single grid carries one velocity field, and one substep advances all of it at once. Two materials
sharing that grid are therefore advanced together, at a single $\Delta t$, and the only safe choice is
the smallest one any of them needs:

$$
\Delta t_{\text{scene}} \;=\; \min_{m \,\in\, \text{materials present}} \Delta t_m .
$$

The consequence is sharper than it first sounds. Substep count is not the average of the materials in a
scene, it is the **maximum**, so adding one small object made of a demanding material multiplies the cost
of everything else in the scene. A pool of water sitting comfortably at 139 substeps per frame more than
doubles to 333 the moment a single snowball is dropped into it, and every one of those extra substeps is
also spent on all the water. There is no version of this where the water pays its own cheaper bill.

That makes **material choice a budget decision rather than an aesthetic one**, which is an unusual
constraint to design a scene under. The mitigations are all uncomfortable. Sub-cycling the stiff material
on its own finer clock means the two clocks have to exchange momentum at some coarser rate, and getting
that exchange stable is a harder problem than the one it solves. Softening the expensive material to buy
a larger timestep changes what the material *is*, which is exactly the ground-truth drift a frozen
physics library exists to prevent. Accepting the cost, and being honest about the frame rate or the
particle count that results, is often the least bad option.

The wider point for a controllable world model is that a simulation whose cost is set by its most
demanding constituent scales badly in a way a learned model does not. A network's cost is a property of
the network. An explicit solver's cost is a property of *whatever happens to be in the scene*, which
means adding one new material can silently halve the frame rate of content that was fine yesterday.

## The timestep is not a performance knob

The tempting move is obvious and wrong. Doubling $\Delta t$ halves $S$ and doubles the frame rate for free.
It is worth seeing exactly how badly this fails, because the failure is not where intuition puts it.

![Accuracy against speedup as the timestep is raised on an elastic disk. Buying a 1.6x speedup costs about 320x
(two and a half orders of magnitude) of trajectory accuracy, long before the solver becomes unstable.](/api/data/learning-taichi/runs/material-variants/interactive-simulation-of-one-material/dt_tradeoff.png)

Instability is the *last* thing to arrive, not the first. At 1.5 times the canonical timestep an elastic
disk is still perfectly stable, still looks like rubber, and has already moved about 320 times further
from the reference trajectory than the exact port does (three orders of magnitude arrives past 2.5x). At 2 to 3 times it rolls a visibly different distance and settles in a
different place. Only past 4 times, where the wave crosses more than a cell per step, does it actually blow
up to non-finite values in the manner [[failure-modes]] describes.

The practical rule is that **a stable-looking simulation at an inflated timestep is the dangerous case**.
Divergence announces itself. Silent trajectory error does not. Anything that consumes a rollout downstream,
a learned model fit to it, a controller optimised through it, a comparison between two materials, is reading
a different physical system than the one it thinks it is.

## And refining the timestep does not save a plastic material

The natural repair for the previous section is to shrink $\Delta t$ until the answer stops changing. For
an elastic solid that works exactly as advertised: released as an over-steep pile, it settles into an
identical shape across a thirty-fold range of timestep, to the last decimal place the diagnostics report.

For anything with a **plastic projection** it does not work, and the way it fails is worth understanding
because the naive reading of the data is backwards. Halve the timestep and a settled pile of snow or sand
sits noticeably lower. Halve it again and it sags further. There is no timestep at which the answer stops
moving, so "refine until converged" never terminates.

The diagnostic that resolves it is to ask **which axis the runs collapse on**. Plot the pile's slope
against physical time and the curves for different timesteps fan apart. Plot the same runs against the
*cumulative number of substeps taken* and they land on top of each other. For snow the spread across
timesteps falls from tens of degrees at equal time to a degree or two at equal substep count. Elastic,
the control, is flat on both axes. A process that depends on how many times a loop ran rather than on how
much time passed is not physics.

The mechanism is a **ratchet**, and it is worth spelling out because the ingredients are generic. Every
substep, the particle-to-grid and grid-to-particle round trip returns a velocity gradient that is right to
within a small quadrature error, so the trial deformation each particle computes carries a little noise.
An elastic material stores that noise as elastic strain and gives it back, so it averages out. A plastic
return mapping is **one-sided**: it can move a state from outside the admissible set to the boundary, and
it can never move one back out. Symmetric noise fed through a one-way valve becomes a drift. The drift
accrues once per projection, the projection happens once per substep, and so the total accumulated
artificial yielding is proportional to the substep count. Halving $\Delta t$ doubles it.

That reading also settles which run to trust, and it is the opposite of the usual answer. **The coarse run
is the clean one.** The finer run has not resolved anything the coarse one missed, it has simply spent
more substeps ratcheting. Reported this way round, a material's apparent strength always has to be quoted
with the timestep and the physical duration it was measured at, because "how strong is this snow" has no
answer without them.

This is a caution and not a solved problem. It is measured on one scene family at one grid resolution with
a handful of timesteps per material, and the ratchet story is the mechanism the evidence is most consistent
with rather than one that has been isolated. The tests that would isolate it are a sweep across grid
resolutions, since transfer noise should scale with the cell size, and a return mapping that is not applied
once per substep, which should remove the substep dependence outright if the mechanism is right.

The consequence for a learned world model is uncomfortable and interesting. Any dataset of plastic material
behaviour generated by an explicit solver carries the solver's substep count baked into its labels, so a
network fit to short rollouts and a network fit to long ones are being taught different materials. That is
a data-generation bug that looks exactly like a modelling result, and nothing in a loss curve would reveal
it.

## Where a substep's time goes, and why that depends on the machine

A substep has two kinds of work with different scaling. P2G and G2P are **per particle**, each touching a
$3 \times 3$ stencil. The grid update is **per cell**, and the number of cells is fixed by the resolution
whether or not any material is there.

On a GPU the cell loop is free, in the sense that all cells run at once, so the natural implementation
sweeps the entire grid every substep. Transplant that same loop to one CPU thread and it becomes the
dominant cost. A blob of material occupying a few percent of the domain lights up roughly 760 of 16 384
cells; the other 95 percent of the sweep is arithmetic on zeros. Measured on one thread, the full sweep
costs about 93 microseconds per substep on its own, which at $S = 167$ is 15.6 ms per frame. **The empty
cells alone consume the entire 60 fps budget before a single particle is touched.**

The fix is to make the grid loop sparse. P2G already knows which cells it scattered into, so recording that
list lets the grid update and the clear walk only those cells. The important property is that this is
**exact rather than approximate**: every node a particle gathers from in G2P is a node it scattered to in
P2G, so no cell outside the touched set can influence any particle. Sparse and dense implementations agree
bit for bit.

The lesson generalises past this one loop. **The right data structure is a property of the execution model,
not of the physics.** A dense array is correct on a GPU and wasteful on a thread; a sparse list is the
reverse at high occupancy. Porting a solver between execution models means re-deriving those choices, not
translating them.

## A flat cost curve measures the API, not the device

The most useful measurement to make on any implementation is **cost against problem size**, because its
*shape* identifies what is being paid for. If halving the particle count does not halve the time, the
particles were never the bill.

Driving a GPU one kernel at a time from a scripting language produces exactly that signature: a substep
cost that is **flat** across a thirty-fold range of particle count. Flat means the arithmetic finished long
before the next instruction to start work arrived. The floor is easy to measure directly, and it should
always be measured directly: time a kernel that does *nothing*. If an empty launch costs tens of
microseconds and a substep issues four of them $S$ times a frame, then several tens of milliseconds per
frame are spent on nothing at all, and no particle count changes that.

The seductive misreading is "small problems do not want a GPU". That conclusion does not follow, and it is
wrong. The fixed cost being measured is per **submission**, not per unit of work, so it disappears if the
work is submitted differently. Modern graphics APIs let a program **record many dispatches into one command
buffer and submit it once**, with the ordering and memory-visibility guarantees between consecutive
dispatches that a $\text{P2G} \to \text{grid} \to \text{G2P}$ chain needs. All $3S$ dispatches of a frame
then go across the boundary as a single object.

The difference this makes is not incremental. On one machine an empty compute dispatch inside a recorded
buffer measured **about 1 microsecond against about 56 microseconds** for an empty kernel launched
individually from a scripting language, a factor of fifty in the floor, with the identical device and the
identical arithmetic. A substep dominated by launch overhead at 345 microseconds became a substep of about
7 microseconds, and the cost curve went from flat to properly proportional to particle count. Cost then
scales with the P2G scatter, which is the thing that ought to dominate.

So the honest version of the lesson is sharper and more useful than the original one:

- **A flat cost curve is a diagnosis, not a verdict.** It says the bottleneck is issuing work, and issuing
  work is the part you can restructure.
- **Latency-bound and throughput-bound are properties of how work is submitted**, not just of how much of
  it there is. Interactive simulation has a small state, a hard sequential dependency, and hundreds of
  steps per frame, all of which make per-submission cost the thing to attack first.
- **Batching does not touch $S$.** The substep count is still set by $\Delta t$, and the frame still costs
  $S$ times something. Removing the launch overhead changes what that something is; it does not repeal the
  budget equation.

The comparison worth internalising is that on the same machine and the same physics, one CPU thread, a
GPU driven one launch at a time, and the same GPU driven with one submission per frame differ by more than
two orders of magnitude in the particle count they sustain at 60 fps &mdash; and the middle of those three
is the *slowest* at every size below a few thousand particles. Which is to say: the implementation strategy
outweighed the choice of hardware.

## What survives a port, and what was never physics

Moving a solver to a new execution model is a useful exercise precisely because it forces the question of
what is load-bearing.

The **parameters** and the **constitutive law** are the physics. A port that re-picks a stiffness or softens
a clamp has silently created a different material and every comparison against it is meaningless. The safe
construction is mechanical: generate the constants from the canonical source rather than retyping them, and
stamp the result with a version.

Almost everything else is an artifact of the original framework. Dense grid loops are a GPU habit. Float32
is a GPU habit; a port doing float64 arithmetic is *more* accurate per operation, not less. Nondeterminism
is a GPU habit, arising from atomic scatter order, and a single-threaded port is deterministic where its
reference is not. And the SVD in the elastic stress is a habit too: in two dimensions the corotated model
needs only the polar rotation, which has the closed form derived in [[svd-polar]], so an entire matrix
factorisation drops out of the inner loop for free.

## Verifying a port against a reference that is not deterministic

A port cannot be checked by asking for identical output, because the reference does not produce identical
output twice. GPU atomics accumulate in a different order run to run, so two runs of the same code on the
same input already disagree. Chaotic dynamics then amplifies that disagreement exponentially, and a
contact-rich rollout is chaotic.

The correct standard is a **noise floor built from the reference itself**. Run the reference twice on
identical input and measure how far apart it lands. Run it again with the initial positions nudged by one
float32 rounding unit, roughly $10^{-7}$, and measure that too. Those two numbers bracket the disagreement
that carries no information. If the port's divergence from the reference sits inside that band and grows at
the same rate, there is nothing left to detect, and the residual is chaos rather than bias.

The signature to look for is the *shape* of the divergence curve, not its endpoint. Bias shows up
immediately and grows linearly. Chaos starts at rounding scale and grows exponentially until it saturates.
A port that is genuinely wrong in one term looks nothing like the reference's own self-noise on a log axis,
even when a single summary number happens to be small.

## What a learned operator inside the substep actually costs

The budget equation prices a learned step before any training happens, and the price is worth working out
carefully because the naive arithmetic makes the idea look affordable and the measurement does not.

Take the whole grid update as the thing to be replaced. It reads a node's accumulated mass and momentum and
writes its velocity, and on a $128 \times 128$ grid that is $16\,384$ evaluations per substep. At water's
timestep the solver runs $1/\Delta t = 20\,000$ substeps per simulated second, so a dense learned grid update
is evaluated

$$
16\,384 \times 20\,000 \;\approx\; 3.3 \times 10^{8} \ \text{times per simulated second.}
$$

A two-hidden-layer perceptron of width $h$ with a handful of inputs costs roughly $2h^2$ floating-point
operations per evaluation once $h$ is large enough for the square term to dominate. At $h = 16$ that is about
830 FLOP, so $2.7 \times 10^{11}$ FLOP/s, and at $h = 64$ about $9.5 \times 10^3$ FLOP, so $3 \times 10^{12}$
FLOP/s. Against a large GPU's tens of teraFLOP/s of fp32 those are single-digit and low-double-digit
percentages of *peak*. On paper the idea is not absurd, which is exactly why it has to be measured.

**Peak is not what a tiny per-cell network gets, and the reason is not arithmetic.** Measured on one RTX 4090
in a browser, with P2G and G2P left analytic and only the grid update swapped, the grid kernel's own cost per
substep came out at roughly

| hidden width | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| grid update, microseconds per substep | 1.3 | 3.4 | 41 | 48 |

against an analytic grid update measured at **under 0.1 microseconds** on the same device. That is a factor of
about 30 at the smallest width tested and about 1000 at width 64. The learned operator is not somewhat more
expensive than the formula it replaces; it is three orders of magnitude more expensive.

![Whole-solver cost per substep against particle count, with the analytic grid update as the baseline and the 60 fps budget drawn as a line, both at full device throughput and derated to a quarter of it. The learned curves are flat because the cell count does not depend on the particle count.](/api/data/learning-taichi/runs/material-variants/profile-a-nn-running-for-the-grid-update-on-webgpu/cost_vs_budget.png)

### The cost does not fall when the problem gets smaller

Three properties of that measurement are more instructive than the numbers themselves, and all three are
consequences of one fact.

**The cost is flat in particle count.** A dense grid update evaluates the network once per cell, and the cell
count is a property of the resolution, not of the scene. Sixty-four times more particles cost the analytic
solver four times more (its bill is P2G's scatter) and cost the learned grid update nothing extra at all.
A flat curve here is the *expected* result rather than a diagnosis of the kind described earlier on this page.

**Skipping empty cells is exact and does not help.** G2P gathers from precisely the cells P2G scattered into,
so a cell with zero mass cannot be read by any particle and whatever is written there is unobservable.
Skipping the network on those cells is therefore not an approximation. On a scene where only 2.4% of the grid
held material, it saved a few percent.

**Compacting the dispatch does not help either.** Running the identical kernel over the number of workgroups
an occupied-cell list would need — seven instead of 256, a thirty-six-fold cut in the work issued — changed
the time by one or two percent.

Issuing thirty-six times less work for the same elapsed time has only one explanation: **the kernel is
latency-bound, not throughput-bound.** With 64 cells per workgroup, $16\,384$ cells is 256 workgroups, and a
GPU with more than a hundred multiprocessors is nowhere near occupied by that. The elapsed time is set by how
long *one* thread takes to walk its own network — a serial chain of dependent multiply-accumulates, each
waiting on a weight load — and adding threads is free until the machine fills up. On this device it does not
fill up.

The practical consequence inverts the usual instinct. Sparsity, culling and level-of-detail all attack the
number of evaluations, and here the number of evaluations is not the bill. **The only lever that shortens the
time is a shorter dependency chain per cell, which means a smaller network.** A corollary worth keeping: this
conclusion is scoped to a small grid on a large GPU, and a much finer grid would push the same kernel into the
throughput-bound regime where the usual instincts return.

A second measured surprise, in the same vein: cost is not even monotonic in width. Sweeping the width finely
with untrained networks — cost does not depend on what is in the weight buffer, so this needs no training —
shows achieved throughput around 3,500 GFLOP/s at widths 4 to 20, collapsing to about 1,000 across widths 24
to 40, and recovering above 48. Width 48 costs *less in absolute terms* than width 40 despite 44% more
arithmetic. The likely mechanism is that the hidden activation vectors stop fitting in registers and spill to
scratch memory over part of the range, which is a statement about a shader compiler rather than about
networks, and it is a hypothesis rather than something isolated here. The transferable lesson is the one that
survives either way: **for a small fused network, cost is governed by where its working set lives, not by how
many operations it contains.**

### Fitting the output is not fitting the physics

The cost is only half the story, and the accuracy half is the more interesting one because it is not a
training-effort problem.

Almost everything the grid update *outputs* is the division $\mathbf{v}_i = \mathbf{p}_i / m_i$, which carries
no physics at all — it is a change of variables from momentum to velocity. The physics is a small perturbation
on top of it. Gravity is one line, $v_{i,y} \mathrel{-}= \Delta t\, g$, and at water's timestep that is

$$
\Delta t \cdot g \;=\; 5 \times 10^{-5} \times 9.8 \;\approx\; 4.9 \times 10^{-4}
$$

of velocity per substep, roughly a thousandth of a typical node speed. It only becomes a falling drop because
it is applied twenty thousand times a second.

Now put that next to what a fitted network achieves. Trained per cell against the canonical kernel on water,
the mass-weighted error in the node velocity came out at $1.4 \times 10^{-1}$ at width 8 and
$2.7 \times 10^{-2}$ at width 64 — that is, **56 to 289 times gravity's entire per-substep contribution**. The
term the whole simulation is driven by sits far below the network's own noise floor. The visible consequence
is exactly what that predicts: the learned fluid does not fall. It hovers, frays, and drifts, while the
analytic reference from the same seed falls, splashes and settles.

![Left: mean particle height against simulated time for the canonical solver, its analytic port, and three learned grid updates from the same seed. The canonical drop falls, hits the floor and settles; two learned rollouts leave the frame upward and the best one descends at a small constant rate instead of accelerating. Right: the mass-weighted node velocity error of each trained width, against the single green line marking gravity's whole contribution to one substep.](/api/data/learning-taichi/runs/material-variants/profile-a-nn-running-for-the-grid-update-on-webgpu/gravity_below_noise.png)

This is not fixed by training harder. At width 64 the network would have to become fifty-six times more
accurate before gravity was even *visible* to it, and the errors it does make are not the sign-consistent kind
that would average out.

There is a second, independent version of the same trap. G2P does not read the node velocity; it reads the
affine matrix $\mathbf{C}_p$, whose entries carry a $1/\Delta x^2$ factor, so what reaches a particle is a
weighted spatial **derivative** of whatever the grid update wrote. Fitting an operator cell by cell leaves the
derivative of the fit completely free, and small pointwise errors with no spatial correlation produce a
derivative error of the same order as the derivative itself. Measured on the fitted fields, the relative error
in the first difference of the velocity field ran from 87% at width 8 to 41% at width 64 — against 14% and
2.7% in the velocity itself. Adding a term on the derivative directly to the training objective moved it to
77% and 38% and no further.

The general principle is worth stating on its own, because it applies to any learned operator placed inside a
loop:

> **An operator applied $N$ times per unit of simulated time must be fitted to an accuracy finer than the
> increment it contributes per application, and in whatever functional the consumer actually reads.** A loss
> on the operator's output, at an error scale set by the output's magnitude, guarantees neither.

Both failures are the same shape. The magnitude of the output is dominated by a term that carries no physics,
so a loss written on that magnitude spends its capacity in the wrong place.

### So what is the version of this idea that could work

The honest framing is that a learned per-substep update is optimising the wrong term twice over. It pays $S$
evaluations per frame for a kernel that has to beat a division and a branch, and it is asked to resolve a
perturbation three orders of magnitude below the thing it is fitted to.

The version that could pay for itself maps the state at $t$ directly to the state one *frame* later, taking
one evaluation per frame instead of hundreds, and absorbs the stability constraint that forces the small
timestep into learned weights rather than obeying it. Crucially it also changes the accuracy arithmetic:
across a whole frame gravity contributes $S \Delta t\, g$, which is hundreds of times larger than the
per-substep increment, so it is no longer buried under a plausible fitting error. That is a genuinely
different proposition from the residual-inside-the-step models of [[hybrid-learned-residual]] and the material
networks of [[learned-materials]], and it remains a conjecture. What would settle it is training on
coarse-time transitions and measuring both the trajectory error against a canonical rollout and the cost per
frame.

## What's open

The substep-count argument is exact, but the constant in "a substep may cost 100 microseconds" is a statement
about one machine, one language, and one problem size, and it moves with all three. The batched-submission
result above says the same thing more sharply: that constant moved by a factor of fifty without a line of
arithmetic changing.

The learned-operator numbers carry the same caveat and one more. They are one grid resolution on one very
large GPU, and the latency-bound finding is specifically a claim about $16\,384$ cells failing to occupy that
device. A grid four times finer in each direction would put the same kernel in a regime where compaction and
sparsity start to matter again, and the analytic baseline would rise too, so the ratio between them is the
quantity that would need re-measuring rather than either number alone. Whether a coarse-time learned model can
hold a rollout together at one evaluation per frame is untested and is the interesting question, because a
positive answer would decouple interactive simulation from the CFL condition entirely, and a negative one
would say that explicit stability is a floor no amount of learning removes. Also untested is whether the
sparse-grid rewrite still pays at high occupancy, where the material fills most of the domain and the
touched-cell list stops being a small fraction of the grid.
