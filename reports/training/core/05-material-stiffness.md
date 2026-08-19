# The three material dials: stiffness $E$, incompressibility $\nu$, and density $\rho$

Every constitutive model in [[constitutive-models]] is steered by three scalars, and each one owns a
different question.

- $E$, **Young's modulus**, sets *how hard* the material pushes back.
- $\nu$, the **Poisson ratio**, sets *what it pushes back against*: shape change only, or volume change too.
- $\rho$, the **density**, sets *how heavy* it is, and it turns out to be invisible until a second material
  shows up.

Those three are worth separating because they fail in different ways. Getting $E$ wrong makes a material
too floppy or too expensive. Getting $\nu$ wrong makes a solid quietly shrink when it is hit. And $\rho$
carries a trap that is genuinely surprising the first time it is met: for a single material simulated on
its own, changing the density changes *nothing at all*.

## $E$ enters every stress law linearly

The weakly compressible fluid uses a pressure stiffness $k = 4E$, so its stress is

$$
\sigma^{\text{fluid}} = -4E\,(J - 1)\,I,
$$

and the elastic Lamé parameters are $\mu = E / [2(1+\nu)]$ and $\lambda = E\nu / [(1+\nu)(1-2\nu)]$, both
**proportional to $E$** at a fixed Poisson ratio $\nu$. So doubling $E$ doubles the restoring stress the
material develops at the same deformation, in every model. Everything below is a consequence of that one
linear fact, chased into its physical and numerical corners.

![Two consequences of the stiffness dial, both read straight off the model equations rather than measured.
Left, the fluid restoring stress against the volume ratio J. The line always passes through rest at J equal
to 1 but gets steeper as E grows, so a stiffer material answers the same compression with a larger push.
Right, the free wobble of a springy blob. Its oscillation frequency scales as the square root of E, so the
4E material (red) completes exactly twice as many cycles as the E material (green) in the same
time.](/api/data/learning-taichi/reports/training/assets/e-stiffness.png)

## Statics: a stiffer blob sags less

Rest a blob under gravity and it compresses until its internal stress balances its own weight. The stress
at a small volumetric strain $J - 1$ is about $E\,(J-1)$ in magnitude, and the weight it must support per
unit area is about $\rho g L$, where $\rho$ is the material density, $g$ gravity, and $L$ the height of the
column of material pressing down. Setting the push equal to the load,

$$
E\,\lvert J - 1 \rvert \;\sim\; \rho\, g\, L
\qquad\Longrightarrow\qquad
\lvert J - 1 \rvert \;\sim\; \frac{\rho\, g\, L}{E}.
$$

The equilibrium compression is **inversely proportional to $E$**. A ten-times stiffer material settles to
one-tenth the squash under the same gravity. This is the visual difference between a slab of gel that
noticeably flattens under its own weight and a block of hard rubber that holds its shape, and it is set
entirely by where $E$ lands.

## Dynamics: a stiffer blob wobbles faster

Deformation stores elastic energy, and stored energy released against inertia is an oscillation. The
restoring force per unit displacement scales with $E$ and the inertia scales with the mass, so a blob
behaves like a mass on a spring whose stiffness rises with $E$. A spring of stiffness $k$ and mass $m$
oscillates at $\omega = \sqrt{k/m}$, which gives the scaling

$$
\omega \;\propto\; \sqrt{E}.
$$

The same $\sqrt{E}$ governs the speed of elastic waves inside the material, $c \propto \sqrt{E/\rho}$, since
a wave is just this restoring-versus-inertia balance travelling through the bulk. The right panel of the
figure shows the bare consequence. Quadruple $E$ and the wobble frequency doubles. A soft material sloshes
slowly and looks liquid, a stiff one rings at a high frequency and looks rigid, and the crossover is a
square-root in one parameter.

## Numerics: a stiffer material forces a smaller timestep

The wave speed is not just a visual property, it sets what the explicit solver is allowed to do. An
explicit timestep can only be trusted if information moves less than about one grid cell per step, the
**CFL condition**, which caps the timestep at roughly

$$
\Delta t \;\lesssim\; \frac{\Delta x}{c} \;\propto\; \frac{\Delta x}{\sqrt{E}}.
$$

The largest stable $\Delta t$ falls as **$1/\sqrt{E}$**. Raise the stiffness and keep the old timestep and
the fastest elastic wave now outruns a grid cell per step, the update over-corrects, and the energy grows
without bound, the exact overflow-to-NaN blow-up that [[failure-modes]] dissects. So $E$ cannot be turned
up for free. A stiffer material is also a more expensive one, because holding it stable demands more, and
finer, steps to cover the same simulated time.

Approaching that limit from the other side is more instructive than crossing it. Raising $\Delta t$ on a
stiff elastic solid destroys the trajectory **long before** it destroys stability: at 1.5 times a
CFL-number-0.27 timestep the material is still stable, still looks like rubber, and has already drifted
three orders of magnitude from the reference trajectory. Blow-up only arrives past four times. The
dangerous regime is therefore the stable-looking one, and the cost of staying inside it, measured in
substeps per displayed frame, is what [[real-time-cost]] works out.

## Gradients: a stiffer material needs a smaller learning rate

The reason $E$ belongs in the core rather than a physics footnote is what it does to the loss landscape a
controller descends. A control loss reaches its parameter through the stress, and the stress is proportional
to $E$, so the sensitivity of the forces to the state, which is the curvature the optimizer feels, also
scales with $E$. Larger curvature means a smaller step before the optimizer overshoots, so the largest
usable learning rate scales roughly as

$$
\eta_{\max} \;\propto\; \frac{1}{E}.
$$

This is the same lesson [[constitutive-models]] reaches by comparing fluid to elastic, now traced to its
single cause. The elastic solid needed a much smaller learning rate than the fluid not because it is a
different kind of object but because its stress law is stiffer, and stiffness is curvature is a smaller
stable step. A shared optimizer setting does not survive a change of $E$ any more than it survives a change
of material, because at fixed physics they are the same change. For a controllable, differentiable world
model this is the practical face of the whole difficulty. The dial that makes a material feel solid and
interesting is the same dial that sharpens its loss surface and shrinks the step by which it can be steered.

## The second dial: $\nu$ decides whether the material can change volume

$E$ answers "how hard", and it answers nothing about *which kind* of deformation is being resisted. That is
the Poisson ratio's job.

Any small deformation splits into two independent pieces, exactly the split [[linear-algebra]] builds as the
trace-and-deviator decomposition: a change of **volume** (the material gets bigger or smaller) and a change
of **shape** at constant volume (it gets squashed one way and stretched another). The two Lamé parameters
are the price of each. $\mu$ resists shape change and $\lambda$ resists volume change, and both are built
from $E$ and $\nu$:

$$
\mu = \frac{E}{2(1+\nu)}, \qquad \lambda = \frac{E\,\nu}{(1+\nu)(1-2\nu)}.
$$

The informative object is their ratio, because that is what says which of the two the material cares about
more:

$$
\frac{\lambda}{\mu} = \frac{2\nu}{1-2\nu}.
$$

Read the denominator. As $\nu \to \tfrac{1}{2}$ it goes to zero and $\lambda/\mu$ diverges. That divergence
*is* the definition of incompressible: a material that would rather do anything than change its volume.
Concrete values make the range vivid. At $\nu = 0.2$ the ratio is $0.67$, so volume is barely defended at
all. At $\nu = 0.45$ it is $9$. At $\nu = 0.49$ it is $49$. Real rubber sits at about $0.4995$; water, if
one insists on a Poisson ratio for it, is essentially $0.5$.

**What that buys, physically.** Under a static load the volumetric strain is set by the bulk modulus, which
in two dimensions is $K = \lambda + \mu$; a bigger $\lambda$ means a smaller squash. The more instructive
case is impact, where the deformation is paid for out of kinetic energy rather than weight. A body arriving
at speed $v$ carries kinetic energy density $\tfrac{1}{2}\rho v^{2}$, and if it stops by compressing, that
energy goes into the volumetric spring $\tfrac{1}{2} K \varepsilon^{2}$, where $\varepsilon$ is the
fractional volume change. Equating them,

$$
\varepsilon \;\sim\; v\,\sqrt{\frac{\rho}{K}}.
$$

So the peak squash on impact falls as $1/\sqrt{K}$, and $K$ climbs steeply with $\nu$. This is the
mechanism behind a failure that looks bizarre until it is named: a blob of "rubber" thrown hard at the
floor comes back **smaller than it left**. Nothing removed material, and a purely elastic model has no
plasticity to lose volume to permanently, but a solid at $\nu = 0.2$ genuinely occupies a much smaller area
at the moment of impact, and while it is in that state it is what the eye sees. Turning $\nu$ up is the
whole fix, and the effect is large: on a hard floor impact, raising a blob's Poisson ratio from $0.2$ to
$0.45$ takes the *body's* peak area loss from about one part in nine to under two percent, and lifts the
first-percentile particle from about a quarter of its rest volume to about two thirds.

![Volume ratio det(F) of an elastic blob through a hard floor impact, at two Poisson ratios. The solid
curve is the body average, which is literally the area the blob occupies, and the dashed curve is the
first-percentile particle, the crushed tail. At the low Poisson ratio the body loses about a tenth of its
area at the moment of impact and its crushed tail falls to roughly a quarter of rest volume; at the high
Poisson ratio both stay near
one.](/api/data/learning-taichi/runs/material-variants/improve-material-realism-in-behavior/rubber_volume.png)

**What it costs, numerically.** The fastest wave in a solid is the pressure wave, at
$c_p = \sqrt{(\lambda + 2\mu)/\rho}$, and $\lambda$ is what is growing. Substituting the Lamé formulas,

$$
c_p \;=\; \sqrt{\frac{E}{\rho}}\,\sqrt{\frac{1-\nu}{(1+\nu)(1-2\nu)}},
$$

which diverges as $\nu \to \tfrac{1}{2}$ because of that same $1-2\nu$ in the denominator.
The CFL cap $\Delta t \lesssim \Delta x / c_p$ therefore collapses as the material becomes incompressible.
There is no free lunch here, and it is why production solvers stop pushing $\nu$ and switch to an implicit
solve or a pressure projection once they want true incompressibility. An explicit scheme can afford
"nearly incompressible" and cannot afford "incompressible". A useful way to hold the whole page in one
thought: $E$ and $\nu$ both buy stiffness and both charge for it in timestep, but $E$ charges for stiffness
in *every* mode while $\nu$ charges only for the volumetric one.

## The third dial: $\rho$ is invisible until a second material shows up

Now the surprise. Write the momentum balance a continuum obeys,

$$
\rho\,\frac{D\mathbf{v}}{Dt} \;=\; \nabla \cdot \boldsymbol{\sigma} \;+\; \rho\,\mathbf{g},
$$

where $\mathbf{v}$ is velocity, $\boldsymbol{\sigma}$ the stress, $\mathbf{g}$ gravity, and
$D/Dt$ the material derivative (the rate of change following a moving chunk of material). Every stress law
on this page is **linear in $E$**, so write $\boldsymbol{\sigma} = E\,\hat{\boldsymbol{\sigma}}$ with
$\hat{\boldsymbol{\sigma}}$ depending only on the deformation. Divide through by $\rho$:

$$
\frac{D\mathbf{v}}{Dt} \;=\; \frac{E}{\rho}\,\nabla\cdot\hat{\boldsymbol{\sigma}} \;+\; \mathbf{g}.
$$

The density has vanished except inside the ratio $E/\rho$. This is exactly the gauge freedom
[[math-toolkit]] describes for a mass on a spring, and the consequence is the same: the map
$(\rho, E) \to (\alpha\rho, \alpha E)$ leaves every trajectory identical. **A single material cannot see
its own density.** Doubling the density of a lone blob and doubling its stiffness produces motion that is
not merely similar but the same, frame for frame.

It survives the discretisation too, which is worth checking rather than assuming, because it is easy to
believe a symmetry of the continuum equations dies in the transfer. In MLS-MPM a particle carries mass
$m_p = V_p\,\rho$ and scatters to node $i$ the quantity
$w_{ip}\,(m_p \mathbf{v}_p + (\mathbf{P} + m_p \mathbf{C}_p)\,\mathbf{d}_{ip})$, where $w_{ip}$ is the
interpolation weight, $\mathbf{C}_p$ the APIC velocity-gradient matrix, $\mathbf{d}_{ip}$ the offset from
particle to node, and $\mathbf{P} \propto V_p E$ the stress term. The grid then divides by the node mass
$m_i = \sum_p w_{ip} m_p$, which is itself proportional to $\rho$. So in the node velocity the stress term
appears as $V_p E / m_p = E/\rho$, and the two inertial terms appear as $m_p/m_p$, free of $\rho$ entirely.
Gravity is applied to the node **velocity**, not the momentum, so it too is mass-blind. Every place density
could have entered, it either cancels or arrives paired with $E$.

That has a practical edge. Introducing per-material density into a frozen simulator sounds like a change
that must perturb everything, and it is not, so long as each material's $E/\rho$ is held fixed while its
$\rho$ moves. Snow at $(\rho, E) = (1, 150)$ and snow at $(0.3, 45)$ are the *same* snow, verifiable by
running both and checking the particle positions agree to the simulator's own run-to-run noise. What was
gained is not new behaviour for snow. It is that snow can now be compared against water.

### Buoyancy is an output, not a force

Break the gauge by putting two materials on one grid and the ratio of their densities becomes physical
immediately, with nothing added to the code. Follow a node holding a heavy solid, surrounded by fluid.

1. The fluid around it is in hydrostatic balance, so its pressure rises with depth and its divergence
   delivers an **upward impulse** to nearby nodes proportional to the displaced volume and to
   $\rho_{\text{fluid}}$. This is the buoyant force, though nothing in the code calls it that; it is just
   what a pressure gradient does.
2. That impulse is converted to a velocity by dividing by the node mass, which is proportional to
   $\rho_{\text{solid}}$. Heavy node, small velocity change.
3. Gravity is added to the velocity directly, so it contributes $-g$ regardless of mass.

Adding steps 2 and 3, the solid's net acceleration is

$$
a \;=\; -g\left(1 - \frac{\rho_{\text{fluid}}}{\rho_{\text{solid}}}\right),
$$

which is Archimedes' principle, derived rather than imposed. Denser than the fluid and the bracket is
positive, so the body sinks. Lighter and it is negative, so the body rises until it breaks the surface,
at which point the displaced volume stops growing and it settles floating. Equal and the body hangs
wherever it was released. **The moment an explicit buoyancy term appears in an MPM code, something has gone
wrong**, because it is already there in the mass ratio and adding it again double-counts.

![Three solids released at rest, fully submerged, in the same pool of water. Snow at 0.3 times water's
density rises and rides the surface; rubber at 1.2 and sand at 1.6 both sink to the floor. No buoyancy
force exists in the solver; the only difference between the three runs is particle
mass.](/api/data/learning-taichi/runs/material-variants/improve-material-realism-in-behavior/buoyancy_three.png)

A caution that matters for anything built on top of this. A shared grid gives every material at a node one
velocity, so two materials meeting at an interface exchange momentum as though the node held a single
blended substance. That is enough to make bulk buoyancy come out right, because the interior of a blob many
cells wide is the dominant term, but it is not a calibrated multi-phase contact model, and a claim about
what happens *at* the interface needs one. The practical symptom is an artificial drag: a body only
slightly denser than the fluid drifts down far more slowly than a real one would, because the water it has
to push past is locked to its own velocity wherever the two share a node.

For a controllable world model this is the pleasant kind of structure. Density is not a behaviour that had
to be authored; it is one number per material, and floating, sinking and hovering are consequences. The
same argument says the reverse, too, and it is the sharper half: a system asked to *learn* a material from
video of a single object in isolation can never recover $\rho$ and $E$ separately, only their ratio. The
data does not contain the answer. Only footage of two materials interacting does.

## What's open

The relations above are the scalings the model equations force, read off the stress law and the
mass-spring and CFL arguments, not constants measured from a sweep. They fix the exponents, that
equilibrium compression goes as $1/E$, frequency and wave speed as $\sqrt{E}$, stable timestep as
$1/\sqrt{E}$, and usable learning rate as about $1/E$, but the proportionality constants depend on
resolution, Poisson ratio, and blob geometry. The timestep exponent now has one measured point against it,
a single stiffness whose CFL wall sits where the argument says, but a sweep over $E$ would be needed to
confirm the exponent rather than the location. Checking that the $1/E$ step-size scaling holds across a real
control task, rather than only in the local-curvature argument, is still entirely untested.

Two open questions belong to the other two dials. The impact-strain estimate $\varepsilon \sim v\sqrt{\rho/K}$
is an energy argument with no measured exponent behind it; the direction is confirmed but the power of $v$
is not. And the floating body's submerged fraction should equal the density ratio exactly, which it does for
a stiff blob and does not for a soft one that spreads into a raft and pushes a bump of water up around
itself. The size of that error is presumably a function of how far the body deforms, which is a statement
about the coupling between $\nu$, $E$ and the interface rather than about buoyancy, and nothing here pins
it down.
