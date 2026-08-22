# Putting a network inside a simulation kernel

> Assumes [[real-time-cost]] (the substep budget), [[mls-mpm-forward]] (the P2G, grid, G2P skeleton) and
> [[constitutive-models]] (what a constitutive model computes). [[learned-materials]] covers *whether* a
> network can replace a constitutive law. This page covers what it costs when it does, which turns out to
> be a mostly separate question with mostly separate answers.

## The key idea

A small network evaluated per particle is not "running a neural network on the GPU". It is a few hundred
fused multiply-adds and a handful of transcendentals, inserted into a kernel that was already running. At
that size the cost is decided almost entirely by three things that have nothing to do with the network
itself:

1. whether the network gets its **own dispatch** or is **fused** into an existing one,
2. where the **weights** live in the memory hierarchy,
3. whether the shader compiler **unrolls** the loop over hidden units.

Get all three right and a width-16 network costs the same as the analytic constitutive model it replaces.
Get the third one wrong by a single hidden unit and the cost jumps by a factor of two.

That last sentence is the surprising part, and it is the reason this page exists. The natural mental model
of "cost grows with width" predicts a smooth curve. The measured curve is smooth, then discontinuous, then
smooth again on a different line, and the discontinuity is a property of the compiler rather than of the
arithmetic.

## The budget, in one line

Real time means one second of simulated time per second of wall clock. An explicit solver takes
$1/\Delta t$ substeps to advance one second, where $\Delta t$ is the timestep, so the whole solver gets

$$
T_{\text{substep}} \;=\; \Delta t
$$

seconds of wall clock per substep. That is a strikingly clean result and worth pausing on. The frame rate
does not appear. The particle count does not appear. Real time costs exactly the timestep, per substep, and
the timestep is fixed by the stiffest material in the scene ([[material-stiffness]]).

For four canonical materials sharing one grid, the shared timestep is $\Delta t = 5 \times 10^{-5}$ s, so
the budget is **50 microseconds per substep on the whole device**. A page that also has to render, composite
and share a machine with a browser cannot assume the whole device; a quarter is a defensible planning share,
giving **12.5 microseconds per substep**. Everything below is measured against those two numbers.

## Fuse the network, never dispatch it

The obvious implementation gives the network its own compute pass: gather the inputs, dispatch an
"inference" kernel, write the outputs, carry on. This is the expensive way, and the reason is latency, not
arithmetic.

An **empty** compute dispatch over 128 workgroups, measured on an RTX 4090 through WebGPU with a
timestamp query, costs about **0.86 microseconds**. It does nothing. Three dispatches per substep, which is
what P2G, grid, G2P already need, therefore spend roughly 2.6 microseconds before any work happens, and a
fourth dispatch for inference adds another 0.86 microseconds. Against a 12.5 microsecond quarter-device
budget that is seven percent of everything, paid for the privilege of launching.

Worse, that floor is **flat in the amount of work issued**. A kernel launched over 7 workgroups costs
essentially what the same kernel costs over 256. A dispatch-per-substep design is therefore latency-bound
at small problem sizes, and all the usual optimisations — skipping empty regions, shrinking the grid,
reducing particle counts — buy nothing at all, because they reduce work that was not the cost.

Fusing removes the term entirely. If the seam being replaced is the per-particle constitutive model, the
network can be evaluated at the bottom of G2P and its stress written to a buffer for the next substep's P2G
to read. The dispatch count stays at three. Nothing is added to the launch bill, and what is measured
afterwards is arithmetic rather than overhead.

The one subtlety fusing introduces: the stress that P2G scatters at step $n$ must be the stress of the
deformation gradient that G2P produced at step $n-1$. That is already true of the analytic solver, so
computing it one kernel earlier and caching it is exact, not an approximation. It does need a **priming**
pass before the first substep, because the first P2G has no previous G2P to have filled the cache. For an
analytic law the primed value is exactly zero at $F = I$ and the pass can be skipped; a network does not
output exactly zero, so for a learned seam the priming pass is mandatory.

## Where the weights live is worth a factor of three

Every particle in a warp reads the *same* weight at the same moment. That is the access pattern a constant
bank exists for, and using one instead of a general read-write buffer is close to free to arrange.

Measured cost of the G2P kernel alone, same shader, same weights, only the storage class changed:

| hidden width | weights in a uniform buffer | weights in a storage buffer |
| --- | --- | --- |
| 16 | 2.9 µs | 8.4 µs |
| 32 | 4.3 µs | 13.7 µs |
| 64 | 6.4 µs | 25.3 µs |
| 128 | 22.5 µs | 50.4 µs |

Roughly a factor of three below the unrolling cliff, and the storage-buffer curve is a clean straight line
in width where the uniform one is not. The straight line is the tell: a storage buffer read is a memory
transaction per hidden unit per particle, and memory transactions add up linearly. The uniform path is
being served from a cache that broadcasts.

There is a catch worth knowing about before designing around it. WebGPU guarantees only **8 storage buffers
per shader stage**, and a WGSL uniform array must be an array of 16-byte-aligned elements, so weights in a
uniform buffer have to be packed as `vec4`. Both constraints push in the same direction, which is
convenient, because exceeding the storage-buffer limit does not produce an error. It produces an invalid
bind group, every dispatch silently becomes a no-op, and the resulting cost curve is beautifully flat over
trajectories of pure zeros. Assert that particles actually moved before believing any timing.

## The unrolling cliff

Here is the measured cost of the G2P kernel as the hidden width grows, with the weights in a uniform buffer
and the loop bound written as a literal:

| width | 8 | 16 | 32 | 64 | 80 | 88 | **92** | 96 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| µs | 2.60 | 2.93 | 4.34 | 6.36 | 7.34 | 7.86 | **17.08** | 20.07 | 22.51 | 43.94 |

From 8 to 88 the curve is a straight line, and a good one: 0.066 microseconds per unit of width, which is
exactly what "each hidden unit is a fixed number of multiply-adds" predicts. Between 88 and 92 the cost
jumps by a factor of 2.2 for 1.05 times the arithmetic. After that it is a straight line again, on a
different, steeper line.

The mechanism is identifiable with one control. Compile the *identical* shader with the loop bound read
from a uniform instead of written as a literal, so the compiler cannot know the trip count and cannot
unroll:

| width | 16 | 32 | 64 | 88 | 96 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| literal bound (µs) | 2.93 | 4.34 | 6.36 | 7.86 | 20.07 | 22.51 | 43.94 |
| uniform bound (µs) | 4.55 | 7.79 | 13.88 | 18.18 | 19.62 | 25.34 | 50.18 |

Below the cliff the unrolled version is two to three times faster. Above it, the two curves lie on top of
each other. The unrolled kernel does not degrade into something worse than the rolled one; it degrades into
*exactly* the rolled one. The compiler was fully unrolling the hidden loop, that was worth a factor of two
to three, and somewhere between 88 and 92 hidden units it stopped.

Why unrolling is worth so much here is worth spelling out, because it is not obvious that it should be. The
loop body is a dot product of a weight row against a fixed input vector, a `tanh`, and an accumulate. Unrolled,
the compiler can see every weight index as a compile-time constant, schedule all the loads far ahead of the
uses, interleave independent chains from different hidden units, and delete the loop counter arithmetic and
the branch. Rolled, each iteration is a dependent chain with a branch at the end and an address computation
at the start, and the latency of the `tanh` cannot be hidden behind anything.

Two practical consequences. First, a cost curve measured at a few widely spaced widths can miss the cliff
entirely and report a smooth trend that does not exist. Second, and more useful, **the cliff is a property
of the shader, not of the network**: the same width sits on either side of it depending on how the kernel is
written, so a cliff found at some width is a fact about the compilation, and moving it is a legitimate
optimisation rather than a law of the hardware.

Half-precision weights behave differently again. They are slower than unrolled `f32` below the cliff (a
width-64 `f16` network costs 9.3 µs against 6.4), and faster above it (31.5 µs against 43.9 at width 256),
and they show no cliff at all. The natural reading is that the conversion and the non-native `tanh` cost
real instructions, while the halved weight traffic pays off only once the working set stops fitting.

## The comparison that decides everything: what is being replaced

None of the numbers above mean anything without the baseline. The interesting question is never "is a
network cheap" but "is it cheaper than the thing it removes".

For a per-particle constitutive model over four materials, the analytic code is not cheap. Snow's plastic
clamp and sand's Drucker-Prager return map both need the **singular values** of the deformation gradient, so
both need a real SVD ([[svd-polar]]), which in 2D is a polar decomposition followed by a Jacobi rotation.
Sand's return map then needs logarithms of the singular values, a projection onto a cone, and exponentials
to get back. Snow's hardening is another exponential. Measured on the same hardware in the same kernel, all
of that costs **2.73 µs** per substep at 8192 particles.

A width-8 network costs 2.60 µs. A width-16 network costs 2.93 µs. The learned seam is, at those sizes,
**free** — not because the network is fast but because the analytic law is slow.

Contrast this with replacing a seam that is nearly free. The MPM grid update is a division by node mass, an
addition of gravity, and a boundary test, and a network replacing it was measured at 74 times the analytic
cost at width 16. Same hardware, same technique, opposite conclusion, entirely because of what sat on the
other side of the trade. **Choose the seam by what the analytic version costs, not by what feels most
learnable.**

## Batching substeps

Command encoding is a real cost on the host side, and at $1/\Delta t = 20{,}000$ substeps per simulated
second it is not a rounding error. Measured wall-clock cost per substep, varying how many substeps go into
one submitted command buffer:

| substeps per submit | 1 | 4 | 16 | 64 | 256 |
| --- | --- | --- | --- | --- | --- |
| µs per substep (wall) | 61 | 35 | 33 | 29 | 23 |

A factor of 2.6 for a change that is purely about how the work is packaged. WebGPU orders dispatches inside
a compute pass and makes each one's writes visible to the next, which is exactly the P2G → grid → G2P
dependency, so an entire frame of substeps can go into a single pass with no synchronisation and no
readback. The remaining gap between the wall-clock number and the GPU-timed one (23 µs versus 12 µs) is
host-side encoding that batching cannot remove, only amortise.

## Failure modes

**The flat curve over zeros.** Exceeding the storage-buffer limit, or any other bind-group validation
failure, makes every dispatch a silent no-op in WebGPU. The simulation "runs" at the speed of doing nothing
and the cost curve looks wonderful. Assert non-zero particle motion, and assert that the trajectory is
finite, before believing a single timing.

**Timing the clock instead of the kernel.** `performance.now()` is clamped to about 100 microseconds in
Chromium, so nothing short can be measured with it; timestamp queries are the only option, and their own
quantum must be measured rather than assumed (it has been observed anywhere from 32 ns to 32,768 ns on
different machines). The diagnostic that catches both problems: a cost that does not move with particle
count or with width is not a cost.

**Letting the scene evolve underneath the sweep.** A benchmark that seeds a scene once and then measures
every variant against it is measuring a *different physical state* for each variant, because each timed pass
advances the simulation. Over a full sweep this can amount to seconds of simulated time, so the early
repetitions time a falling blob and the late ones time a settled puddle. Taking the minimum over
repetitions does not save it. The tell is a **control**: time a kernel that is byte-identical across all
variants — the P2G and grid passes, if only the G2P differs — and check that it does not move. It drifted
56% before re-seeding to a fixed state before every probe, and 0.3% after.

**Comparing against a baseline that is not the real physics.** A port that runs every material at one
density and one friction coefficient is not the canonical solver, and both its accuracy and its cost are
answers to a different question. Snow at $\rho = 1$ instead of $0.3$ has $E/\rho$ three times too small and
is simply a softer material wearing snow's name.

## What's open

The activation has not been isolated. A `tanh` per hidden unit per particle is 500,000 transcendentals per
substep at width 64, and on NVIDIA hardware those issue at a fraction of the multiply-add rate. Whether the
0.066 µs-per-width slope is dominated by the multiply-adds or by the activation is untested, and the answer
would decide whether a cheaper activation is worth the accuracy it costs.

Where the unrolling threshold actually comes from is also unresolved. It is bracketed sharply between 88 and
92 hidden units for one specific kernel, but whether the limiting resource is the instruction cache, the
register file, or a heuristic budget inside the compiler is not determined by the measurement, and the
threshold should be expected to move with the shader, the driver and the vendor.
