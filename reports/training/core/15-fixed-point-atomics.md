# Scattering floats when the machine has no atomic float add

> Assumes [[mls-mpm-forward]] (P2G is a scatter with contention) and the floating-point section of
> [[math-toolkit]] (relative precision, ULP, and why summation order changes an answer). The cost side of
> the same port is [[real-time-cost]].

## The key idea

Particle-to-grid transfer is a **scatter**: every particle adds a weighted share of its mass and momentum to
each of nine surrounding nodes, and neighbouring particles hit the same node at the same time. Running it in
parallel therefore needs an atomic read-modify-write, and the quantities being accumulated are floats.

Some parallel languages provide an atomic float add. Some do not &mdash; notably WGSL, whose atomics are
`i32` and `u32` only. When it is missing, the standard repair is to **accumulate in fixed point**: multiply
by a scale, round to an integer, and use an integer atomic add. The scale converts back on the way out.

$$
\texttt{node\_int} \;\mathrel{+}=\; \operatorname{round}\!\big(\text{contribution} \times \sigma\big),
\qquad
\text{node value} \;=\; \frac{\texttt{node\_int}}{\sigma}.
$$

That single constant $\sigma$ is the whole design, and choosing it badly changes the physics silently. The
useful way to think about it is that **32 bits have to be split between resolution and range**, and the two
requirements pull in opposite directions.

## Express the scale in units of the physics, not in absolute numbers

The first practical move is to stop thinking of $\sigma$ as a number of quanta per kilogram. Pick a natural
unit from the problem &mdash; for MPM, one **particle mass** $m_p$ &mdash; and write

$$
\sigma \;=\; \frac{2^{k}}{m_p},
$$

so an accumulator value of $2^{k}$ means "one particle's worth of mass sits on this node". The scale is now
scene-independent: it does not have to be re-tuned when the domain, the density, or the particle count
changes, and both requirements become statements in the same readable unit.

- **Resolution.** The quantum is $2^{-k}$ particle masses. Make it too coarse and every one of the nine
  contributions per particle is rounded away.
- **Range.** A `u32` holds $2^{32}$ quanta, so it saturates at $2^{\,32-k}$ particle masses on a single
  node. Make $k$ too large and a heavily loaded node overruns the accumulator.

Two further details matter and are easy to miss. **Momentum needs a signed type**, because momentum goes
negative while mass never does; the mass accumulator can use the full `u32` range, the momentum one gets
`i32` and one bit less. And the conversion must **round**, not truncate: truncation always moves a value
toward zero, which over hundreds of contributions per node per substep is not noise but a systematic drag
on the momentum field.

## How much resolution the physics actually needs

Resolution is not a matter of taste, and the honest way to settle it is a sweep read against the
simulator's own noise floor (the construction in [[real-time-cost]]: re-run the reference, and re-run it
with the initial state nudged by one ULP; anything inside that band is chaos rather than bias).

Measured on a 2D elastic solid, the answer has two parts that a single test would have got wrong.

On a gentle scene &mdash; a disk released from rest, settling &mdash; a coarse scale of $2^{20}$ quanta per
particle mass is already close to the noise band. On a **contact-heavy** scene &mdash; the same disk
launched sideways so it bounces and rolls, spending the whole rollout in the friction branch &mdash; the
same $2^{20}$ lands roughly eighty times outside the band, and the divergence is visible as a displaced
body, not just a number. Around $2^{22}$ the error drops into the band, and by $2^{24}$ it is
indistinguishable from an exact float accumulation.

Two lessons generalise past this material:

1. **Quantisation error is amplified by contact and by chaos.** A scene that settles quietly forgives a
   coarse accumulator; a scene that keeps hitting things does not. Validating a quantised solver on the
   gentle case only is how a coarse scale gets shipped.
2. **The requirement is roughly "one ULP of a typical node value".** A loaded node carries on the order of
   ten particle masses, so $2^{-24}$ particle masses of resolution is about $10^{-8}$ relative &mdash; right
   at `float32`'s own precision. Fixed point has to *match the float it replaced*, which is exactly the
   bar one would set from first principles, and it consumes about 24 of the available 32 bits to do it.

## The other half of the budget, and it fails silently

Whatever resolution buys, range pays for. With 24 bits below one particle mass, only 8 remain above it: the
accumulator saturates at $2^{8} = 256$ particle masses on one node. Whether that is generous or fatal
depends on how densely the material is sampled, which is measurable and turns out to be simple &mdash; the
heaviest node carries roughly **twice the particles-per-cell**, in particle-mass units, across a wide range
of densities. So $k = 24$ is comfortable up to something like a hundred particles per cell and is a bug
past it.

The reason this deserves its own paragraph is the failure mode. An integer atomic add that exceeds its type
does not raise, does not produce a NaN, and does not stop: it **wraps**. A node that should hold 300
particle masses reports a small number instead, the grid update divides momentum by that wrong mass, and
the material is launched apart. On a plot of positions it looks like an explosion; in the numbers it looks
like an ordinary instability of the kind [[failure-modes]] catalogues, and nothing points at the
accumulator. The tell is that it depends on **density rather than timestep** &mdash; refining $\Delta t$
does not help, and thinning the particles does.

Hence the design rule, both halves together:

$$
2^{-k} \lesssim \text{one ULP of a loaded node}
\qquad\text{and}\qquad
2^{\,32-k} \;>\; \text{a safety factor} \times 2 \times \frac{N_{\text{particles}}}{\text{cells occupied}} .
$$

If those two cannot both be satisfied, 32 bits are genuinely not enough and the answer is a different
mechanism, not a different $k$.

## The exact alternative, and what it costs

There is a way to get an exact float add out of integer atomics: **compare-and-swap**. Read the word,
reinterpret its bits as a float, add, reinterpret back, and atomically swap it in only if the word has not
changed meanwhile; retry if it has.

```
old = load(node)
loop:
    new = bits_of(float_of(old) + contribution)
    (old, ok) = compare_exchange(node, old, new)
    if ok: break
```

This quantises nothing. Its accuracy is exactly that of a float addition, so it is the right **control** for
a quantisation study: run it beside the fixed-point version on the same initial condition and any remaining
difference from the reference belongs to something other than the scale.

It is worth internalising what that control shows. An exact-float port still does **not** reproduce the
reference bit for bit, and still sits at the edge of the noise band rather than at zero, because the
accumulation *order* differs and float addition is not associative ([[math-toolkit]]). "Exact arithmetic"
and "identical answer" are different claims, and only the first is achievable in parallel.

The price is contention. Every colliding thread retries, so cost grows with how many particles land on the
same node. Measured against fixed point on the same scenes it ran **roughly two to four times slower** on
the P2G phase. That is a real cost but not a prohibitive one, which is the interesting part: on a machine
where the frame budget is not tight, taking the exact path and spending the extra time is a defensible
choice, and it removes the entire resolution-versus-range problem along with the possibility of a silent
wrap.

## What to take away

The transferable shape of this is not about MPM at all. When a parallel primitive is missing and the
standard workaround is a fixed-point encoding:

- express the scale in a **unit of the problem** so it does not need re-tuning per scene;
- treat the word width as a **budget split between resolution and range**, and measure both requirements
  rather than guessing them;
- validate on the **hardest** dynamics available, not the calmest, because quantisation error is amplified
  by contact;
- keep an **exact but slower** implementation available as a control, so numerical questions can be
  answered by differencing rather than by argument;
- and remember that the range failure is **silent**, which makes it the one worth engineering a check
  around rather than trusting.
