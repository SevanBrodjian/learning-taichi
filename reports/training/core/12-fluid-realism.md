# Making it read as water: dynamics, interior fill, and the realism gap

The [[fluid-rendering]] page builds a screen-space renderer that turns an MPM particle cloud into a frame
that reads as liquid, and it ends on an honest admission: the result gets most of the way to a double-take
and then stops short. Two of the things holding it back are not subtle once they are named. The liquid does
not quite move like water, and a calm body of it shows air holes inside a region that should be solid. The
first is a simulation problem, the second is a reconstruction problem, and neither is fixed by adding more
optical polish on top. This page is about both, and about the smaller realism cues that close some of the
remaining distance once they are dealt with.

The two defects are textbook examples of a model looking wrong for reasons a newcomer would misattribute.
The sluggish motion is not a rendering problem and no shader fixes it; the interior holes are not a physics
problem and no solver tuning removes them. Diagnosing which layer a visual defect lives in before reaching
for a fix is the whole skill, and it is exactly what a controllable world model demands when its output looks
off and the cause could be the dynamics, the reconstruction, or the shading.

## Why weakly compressible MPM water looks sluggish

A weakly compressible MPM fluid carries a pressure that pushes back on changes in volume. Each particle
tracks a volume ratio $J$, the factor by which the material around it has expanded or compressed relative to
its rest state, so $J = 1$ is undisturbed, $J < 1$ is compressed, and $J > 1$ is expanded. The pressure is a
stiffness times the deviation from unit volume,

$$
p = -E\,(J - 1),
$$

where $E$ is the bulk stiffness, the same modulus followed through stress, wave speed, and timestep in
[[material-stiffness]]. This is the whole constitutive law of the fluid: no shear resistance, only a
restoring push against compression. The value of $E$ is what decides whether the result looks like water,
and there are three separate ways the motion can end up looking like syrup instead.

**Compressibility.** Water is very nearly incompressible. Its real bulk modulus is enormous, so in a real
splash the volume barely changes and the fluid responds instantly and stiffly, snapping into sheets and
jets. A low $E$ makes the simulated fluid squishy: it visibly compresses under impact, stores the blow as a
slow spongy rebound, and oozes rather than snaps. Turning $E$ up moves the fluid toward incompressibility
and the motion gets crisper and more water-like. The cost is stability, and it is not free. The pressure
wave in the fluid travels at a sound speed

$$
c = \sqrt{\frac{E}{\rho}},
$$

with $\rho$ the density, and an explicit MPM step stays stable only while a wave cannot cross more than a
grid cell in one step, the Courant-Friedrichs-Lewy condition,

$$
\Delta t \;\lesssim\; \frac{\Delta x}{c} \;=\; \Delta x\,\sqrt{\frac{\rho}{E}}.
$$

Every symbol here earns its place. $\Delta x$ is the grid spacing, $c$ is how fast information moves, and the
ratio is simply the time a signal needs to traverse one cell. The consequence is the scaling worth
memorizing: the stable timestep falls like $\Delta t \sim 1/\sqrt{E}$. Making the fluid four times stiffer,
and therefore markedly more water-like, forces the timestep down by a factor of two and doubles the number
of substeps for the same clip. Stiffer water is more expensive water, and pushing $E$ without shrinking
$\Delta t$ blows the sim up into non-finite noise that renders as static. This is the same stiffness-cost
tradeoff [[material-stiffness]] follows in the elastic setting, seen from the fluid side.

**Numerical damping.** The particle-to-grid and grid-to-particle transfers that move momentum each step are
slightly lossy. A pure particle-in-cell transfer, which throws away each particle's old velocity and
overwrites it with the interpolated grid velocity, is the most dissipative and quietly bleeds energy out of
the flow, so splashes lose their liveliness and settle too fast. The affine transfer used here keeps a local
velocity gradient per particle and is far less lossy, but a small residual remains. Re-injecting a fraction
of each particle's previous velocity, the fluid-implicit-particle correction, returns some of that lost
energy and keeps crests and spray alive. This is the opposite end of the same axis that [[viscosity]]
controls deliberately: viscosity is resistance to the rate of shear added on purpose, while numerical
damping is an unwanted resistance smuggled in by the transfer, and both make a fluid look thicker and slower
than intended. A fluid meant to look like water wants that resistance as low as the scheme allows.

**Time scale and playback.** Even a perfectly tuned splash looks like syrup if it is played too slowly. A
clip spans some amount of physical time $T$, and it is shown as $N$ frames at some frame rate, so the video
lasts $N / f$ seconds. When $N/f$ is much larger than $T$, the motion is in slow motion, and slow-motion
water reads as thick and gelatinous no matter how correct the dynamics are. A clip that covers roughly a
second of physical time but is stretched across five seconds of video is playing at a fifth of real speed,
which is enough to make lively water look like oozing gel. The fix is to map simulated time back to playback
much closer to real time, keeping only a mild slow-motion factor for legibility rather than a heavy one. The
brief warning worth heeding is to not paper over sluggish dynamics by simply speeding up the video: if the
underlying motion is gloopy because the fluid is too soft or too damped, faster playback just gives fast
gloop. Fix the dynamics first, then set the playback to taste.

Put together, water-like motion comes from a stiffer and more nearly incompressible fluid, transfers kept as
undamped as the scheme allows, and a sim-to-playback mapping near real time. None of it touches the renderer.

## Why a naive metaball isocontour punches interior holes

The second defect lives one layer up, in the surface reconstruction. [[fluid-rendering]] builds a density
field $D$ by splatting particles and blurring, then declares the liquid to be the region where $D$ exceeds a
single isovalue $c$. That one threshold is asked to do two jobs at once: decide **where there is liquid at
all**, which sets the opacity, and, through its gradient, decide **which way the surface faces**, which sets
the shading. Those two jobs have conflicting needs, and forcing one threshold to serve both is what carves
holes.

The trouble is that the particles are a discrete, randomly placed sample of a body that is supposed to be
solid. The number of particles landing in any small patch fluctuates the way independent random counts do,
so even deep inside a calm body the blurred density is not flat. It ripples, and wherever a ripple dips below
the isovalue the region is declared to be air. The result is a body speckled with small interior holes,
worst exactly where it should look most solid, in a still pool or the thick base of a wave. Raising the
isovalue makes it worse by carving more, and lowering it thins the whole silhouette and starts swallowing
thin sheets and droplets. There is no single threshold that both fills the interior and keeps the fine
surface, because the interior wants a generous low threshold and the surface detail wants a strict one.

The fix is to stop using one field for both jobs. Reconstruction splits into two products from the same
density field.

**A filled interior mask answers "is there liquid here".** Threshold the density at a deliberately low
level so the whole body is captured with margin, then repair the discreteness. A morphological closing seals
pinholes smaller than the particle spacing, and an enclosed-pocket fill closes any remaining interior gap
whose area is below a set size. The size cap is the important part: it fills the small Poisson holes that
should not exist while leaving genuinely large air cavities alone, so a splash crater or the air tube inside
a breaking wave survives as real air while the speckle disappears. This mask, feathered for an anti-aliased
edge, is what decides opacity. Air now appears only where there is really air.

**The density gradient still answers "where is the surface".** Nothing about the shading normal changes; it
is still lifted from the density field the same way [[fluid-rendering]] describes, because the fine slope
information near the boundary is exactly what makes the surface read as curved and glassy. The point is only
that this detailed, noisy field no longer gets a vote on opacity, so its interior ripples can no longer punch
through the body.

There is a bonus once the filled mask exists. Optical thickness, which drives the depth color and the
refraction, no longer has to be read off the noisy density. Instead it comes from a distance transform of
the filled mask, the distance from each interior pixel to the nearest air. That is zero at the edge and grows
smoothly toward the core, which is exactly what a thickness should do, and it is speckle-free by
construction. The body reads as a smooth solid volume that darkens with depth, with no mottling, because the
quantity driving the color is now a clean geometric distance rather than a random particle count. The
before-and-after below shows the same scene under the old single-isocontour reconstruction and the new
filled-mask one; the interior mottling on the left is entirely a reconstruction artifact, and it is simply
gone on the right.

![Left, the prior renderer: the liquid body is speckled with dark interior holes wherever the particle
density randomly dipped below one isovalue, worst in the calm thick regions that should look most solid.
Right, this pass: a filled interior mask decides opacity while the density gradient is kept only for the
surface normals, so the body reads as a continuous solid volume with air only where there is genuinely air.
The water is also clearer and more depth-graded, and the motion behind it comes from a stiffer, less damped,
nearer-to-real-time fluid.](/api/data/learning-taichi/runs/realistic-rendering/improve-basic-fluid-sim-realism/before_after_dambreak.png)

## The smaller realism cues, and the honest gap

With the motion water-like and the body solid, the remaining gains come from the optical cues the earlier
page named as weak. Three are worth calling out because each targets a specific tell.

The clean distance-transform thickness feeds the Beer-Lambert absorption directly, so the depth color is
smooth and the crossover from pale shallow water to deep saturated blue no longer rides on density noise.
Lowering the absorption a little makes the body glassier, which lets more of the background show through and
turns the water from an inky slab into something that reads as clear and wet.

Refraction was the strongest realism lever in [[fluid-rendering]] and also its most uneven, because a flat
interior has a nearly uniform normal and so bends the background by almost nothing, leaving refraction
visible only at curved rims. Adding an offset driven by the gradient of the thickness, not just the surface
normal, gives the flat interior a gentle lens: a mound of water displaces the background outward the way a
real thick slab would, so the bending is visible across the body and not only at its edges. A structured
studio background with soft light sources gives that lens something worth bending, and a small chromatic
split of the refraction offset across the color channels adds the faint spectral fringe of real dispersion.

The rest is finishing. Stylized bright bands on the floor beneath the water stand in for caustics, a soft
darkening under the footprint grounds the body with a contact shadow, and the foam is textured and gated so
it marks torn, fast, thin water without chalking the calm surface. Every one of these is judged by eye, and
none of them would survive a gradient, which is the same non-differentiable bargain [[fluid-rendering]]
already made and [[material-showcase]] sets against the trainable core of the project.

The honest register is unchanged from [[fluid-rendering]]: still a stylized 2D side view, not light
transport, with a screen-space thickness, painted caustics, and colors chosen by eye. What the two fixes buy
is not photorealism but the removal of two tells louder than any missing light-transport subtlety, moving the
result from obviously synthetic to convincing-at-a-glance. The lesson is the diagnosis, not the fluid: when a
generated world looks wrong, the cause lives in a definite layer, and naming it before fixing separates a
real correction from optical spackle over a dynamics bug.
