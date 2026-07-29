# Stiffness: what Young's modulus $E$ actually does

The constitutive models in [[constitutive-models]] all carry a single stiffness dial, **Young's modulus**
$E$. It is the number that says how hard the material pushes back per unit of deformation, and turning it
is the most direct way to make the same blob behave like water, jelly, or hard rubber. This page isolates
that one parameter and follows it through four places it shows up, because $E$ is a clean case where a
single scalar reaches into the physics, the numerics, and the gradient all at once, and seeing how is worth
more than memorizing where it sits in a formula.

$E$ enters every stress law linearly. The weakly compressible fluid uses a pressure stiffness $k = 4E$, so
its stress is

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

## What's open

The relations above are the scalings the model equations force, read off the stress law and the
mass-spring and CFL arguments, not constants measured from a sweep. They fix the exponents, that
equilibrium compression goes as $1/E$, frequency and wave speed as $\sqrt{E}$, stable timestep as
$1/\sqrt{E}$, and usable learning rate as about $1/E$, but the proportionality constants depend on
resolution, Poisson ratio, and blob geometry. Pinning those constants, and checking that the $1/E$ step-size
scaling holds across a real control task rather than only in the local-curvature argument, is a concrete
measurement a stiffness sweep would settle.
