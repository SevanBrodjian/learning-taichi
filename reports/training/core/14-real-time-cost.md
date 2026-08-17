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

## Small problems do not want a GPU

The other surprise from measuring rather than assuming: at these sizes a GPU is not fast.

![Cost of one MLS-MPM substep against particle count for three implementations, with the 60 fps real-time
line marked. The GPU curve is flat because it is launch-bound, and one JavaScript thread is cheaper below
about four thousand particles.](/api/data/learning-taichi/runs/material-variants/interactive-simulation-of-one-material/substep_budget.png)

The GPU curve is **flat** from five hundred particles to sixteen thousand. Flat means the arithmetic is not
what is being paid for. A substep launches a handful of kernels, each launch carries a fixed cost, and at
16 384 cells the kernels finish long before the next launch can be issued. An empty kernel that touches
nothing already costs tens of microseconds. Below roughly four thousand particles a single scalar CPU thread
is genuinely cheaper than a large discrete GPU running the same physics.

This is the difference between the **throughput regime** and the **latency regime**. Batch work, where a
million particles are simulated offline and only the total matters, lives in the first and the GPU wins by
orders of magnitude. Interactive work, where a small state must be advanced many times per frame with a
sequential dependency between every step, lives in the second, where fixed per-launch costs are multiplied
by $S$ and dominate. A structured, controllable world model that must respond to input is a latency
problem, and reasoning about it with throughput benchmarks gives the wrong answer.

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

## What this implies for learned dynamics

The budget equation prices a learned step before any training happens. A network placed inside the substep
is evaluated $S$ times per frame, so its cost is multiplied by 167, and it must therefore be *cheaper* than
the analytic update it replaces. It never is. Even a very small multilayer perceptron evaluated per active
cell costs a couple of orders of magnitude more than the handful of arithmetic operations in the analytic
grid update, which is a division by mass, a gravity term, and a boundary branch. On a GPU the two can look
equally cheap, but only because both are hidden under launch overhead, which is a statement about the
launch and not about the model.

So the honest framing is that a learned per-substep update is optimising the wrong term. The term that
matters is $S$, and $S$ is set by $\Delta t$. The version of the idea that could pay for itself is a model
that maps the state at $t$ directly to the state one *frame* later, taking one evaluation per frame instead
of 167, and absorbing the stability constraint that forces the small timestep into learned weights instead
of obeying it. That is a genuinely different proposition from the residual-inside-the-step models of
[[hybrid-learned-residual]] and the material networks of [[learned-materials]], and it is a conjecture here
rather than a result. What would settle it is training on coarse-time transitions and measuring both the
trajectory error against a canonical rollout and the cost per frame.

## What's open

The substep-count argument is exact, but the constant in "a substep may cost 100 microseconds" is a
statement about one machine, one language, and one problem size, and it moves with all three. Whether a
coarse-time learned model can hold a rollout together at one evaluation per frame is untested and is the
interesting question, because a positive answer would decouple interactive simulation from the CFL condition
entirely, and a negative one would say that explicit stability is a floor no amount of learning removes.
Also untested is whether the sparse-grid rewrite still pays at high occupancy, where the material fills most
of the domain and the touched-cell list stops being a small fraction of the grid.
