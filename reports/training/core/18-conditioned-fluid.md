# Conditioning across two fluid axes: a local stress the net learns, a non-local force it needs help to learn

[[conditioned-material-net]] built one network conditioned on a material descriptor and interpolated the
descriptor, not the weights, to slide between materials. This page pushes that idea onto a liquid with two
knobs, **viscosity** and **surface tension**, and runs into the single most important distinction in learning
constitutive physics: **local versus non-local**. Viscosity is a local law and a per-particle network learns
it effortlessly. Surface tension is not, and the same kind of network **structurally cannot** learn it. The
fix is not a bigger network. It is giving a second network the right *inputs*.

## The one idea: a particle sees itself, not its neighbourhood

An MLS-MPM stress network is a per-particle function. It maps a particle's own local state, its volume ratio
$J$, its APIC affine matrix $C_p$, its velocity $v_p$, to a stress. That is exactly the right shape for
**viscosity**, because the Newtonian viscous stress is genuinely a pointwise function of that state:
$$
\sigma_{\text{visc}} = \mu\,\big(C_p + C_p^{\top}\big).
$$
Here $\mu$ is the viscous coefficient (how thick the fluid is), and $C_p + C_p^{\top}$ is the symmetric part
of the local velocity gradient, the **rate of shear** at that particle (see [[viscosity]]). Everything the
law needs is in the particle. Feed the descriptor scalar $m_{\text{visc}}$ alongside the state and the
network learns $\mu(m_{\text{visc}})$, a viscosity dial, cleanly.

**Surface tension is a different animal.** It is a capillary force concentrated at the *interface* that pulls
a blob toward the shape of least surface area (see [[surface-tension]]). Its strength at a point is set by the
**curvature** of the interface there, and curvature is not something a single particle can know. Written on
the grid density field $\phi$ (a smoothed indicator, near $1$ inside the fluid and $0$ outside), the capillary
force is
$$
f = \sigma_{st}\,\kappa\,\nabla\phi, \qquad \kappa = -\nabla\cdot n, \qquad n = \frac{\nabla\phi}{\lVert\nabla\phi\rVert}.
$$
$\sigma_{st}$ is the surface-tension strength, $\nabla\phi$ is the density gradient (nonzero only in the thin
band at the surface, so the force lives only there), $n$ is the unit surface normal, and $\kappa = -\nabla\cdot n$
is the curvature, the amount the surface bends per unit length. [[vector-calculus]] derives this
curvature-as-a-divergence-of-the-normal identity in full. The point for learning is what $\kappa$ **depends
on**: it is a *second derivative of the density field*, computed across several grid cells. A particle's own
$(J, C_p, v_p)$ carries none of that. Ask a per-particle stress net to produce surface tension and it has no
input that even correlates with the interface curvature. It cannot represent the force, no matter how large
it is.

This is why a first, tempting design fails: learn the viscous stress with the conditioned net and leave
surface tension as the analytic formula. That does produce a working fluid, but the surface tension is never
learned, it is bolted on. The honest version has to hand a network the interface signal.

## Two networks, split by locality

The working design keeps **one shared descriptor** $m = (m_{\text{visc}}, m_{\text{st}})$ but uses **two
learned networks**, split exactly along the local/non-local line:

- **A per-particle stress network (viscosity).** It predicts the full fluid stress, the weakly-compressible
  pressure $E(J-1)$ plus the viscous term, from $(J, C_p, v_p)$ and the descriptor. This is the
  [[conditioned-material-net]] protocol, and viscosity fits it because the law is local.

- **A capillary-force network (surface tension).** For each grid node it reads a **$5\times5$ patch of the
  smoothed density field** $\phi$ around that node, plus the surface-tension strength, and outputs the
  capillary force at the node. It is trained supervised against the analytic force above. Crucially it is
  **not** given $\kappa$: handing it the curvature would reduce it to echoing $f = \sigma_{st}\,\kappa\,\nabla\phi$.
  It gets the raw density neighbourhood and must **infer the curvature itself**. The $5\times5$ window is not
  arbitrary. The analytic curvature stencil at a node reaches two cells out (a central difference of the
  normal, which is itself a central difference of $\phi$), so a $5\times5$ patch is exactly the support the
  force depends on. Give the network that support and the non-local force becomes a learnable, low-order
  function of its inputs.

The learned rollout at any descriptor is then (per-particle stress net) plus (capillary net), with **no
analytic surface tension anywhere in it**. The analytic continuum surface force survives only as the
supervised target the capillary net trains against and as the ground truth the result is checked against.

![The learned capillary force plotted against the analytic surface-tension force it was trained to reproduce.
On the left, at the trained strength, both force components lie on the identity line. On the right, at an
untrained intermediate strength, the network still matches half the analytic force, showing it learned the
correct linear-in-strength capillary law from only the two endpoint strengths it
saw.](/api/data/learning-taichi/runs/material-variants/generalize-one-nn-across-viscosity-and-surface-tension/capillary_fit.png)

The capillary net fits well for a reason worth internalizing: $\kappa\,\nabla\phi$ is a smooth, low-order
function of a smoothed field (finite differences of $\phi$), well inside a small MLP's reach once the MLP can
see the patch. And it generalizes in strength because the true force is exactly **linear** in $\sigma_{st}$,
so two endpoint strengths pin the line and every intermediate strength is a correct interpolation.

## The surface-tension schedule has to be gentle

Surface tension rounds a blob **fast** and then **saturates**: past a modest strength the droplet is already
as round as a disk gets and turning the knob higher does nothing visible. If the descriptor maps its upper
range into that saturated tail, every high-$m_{\text{st}}$ row of a sweep looks identical and the axis carries
no information. The fix is to **calibrate first** on a cheap gravity-off blob, sweep the strength at low
values, watch roundness climb and flatten, then pick a **low** $\sigma_{\max}$ near the top of the visible
transition together with a gentle schedule
$$
\sigma_{st}(m_{\text{st}}) = \sigma_{\max}\,m_{\text{st}}^{\,p}, \qquad p > 1,
$$
which keeps most of the descriptor range down where roundness is still changing. The exponent $p$ front-loads
the strength toward the top of the axis so the rows land at evenly spaced roundness instead of bunching. The
general rule: a descriptor axis is only useful over the range where its physical effect is not saturated, and
finding that range is a calibration done before building the grid, not after.

## Where it works, and where composing two learned laws breaks

![A five by five grid of the learned fluid dropping the same disk. Horizontal axis is viscosity increasing
left to right; vertical axis is surface tension increasing bottom to top. Three corners are trained and
marked; the top-right corner, high viscosity and high surface tension, is held out. Up the low-viscosity
column the drop rounds gradually into a more cohesive droplet, but the second viscosity column sprays
particles upward in a fountain, and the high-viscosity cells stretch into tall vertical spikes rather than
settled droplets, worst at the held-out
corner.](/api/data/learning-taichi/runs/material-variants/generalize-one-nn-across-viscosity-and-surface-tension/grid_montage.png)

Two things genuinely work. The three trained corners are edge-exact: the full learned rollout overlays the
true fluid there. And the viscosity dial tracks its linear ideal across the interior, the concrete
confirmation of [[learned-material-interpolation]]'s prediction that conditioning one network on the parameter
beats blending two networks' weights.

But the grid does **not** stay physical, and the honest lesson is in the failures. One viscosity column is
degenerate, spraying particles upward instead of settling, a localized stability blow-up. More telling, the
**held-out corner fails**: at high viscosity and high surface tension the two networks must combine in a
regime neither saw, and they do not compose into a droplet, they jet into a tall narrow spike. Its trajectory
RMSE against the true fluid is low, which is exactly the trap, because a spike and a compact blob share a
center of mass, so a distance-to-truth number reads fine while the shape is entirely wrong. Reading the
picture, not the number, is what catches it.

Why does composition break? The capillary force was learned only at low viscosity, where the interface stays
smooth and its density patches fall in the trained distribution. Drive it with the stiff high-viscosity stress
and the interface visits geometries the capillary net never saw, and the two forces reinforce a vertical jet
rather than cancelling into a settled droplet. The decoupled-axes hope, that a force learned on one axis
transfers unchanged across the other, breaks down exactly where the two axes interact most strongly.

Underneath both failures is the **training objective**. Both networks are fit by per-step supervised
regression onto instantaneous force targets, with no rollout in the loss. That yields a locally accurate force
law, which is why the trained corners and the capillary fit look excellent, but nothing in training penalizes
the **accumulation** of small per-step errors over hundreds of integration steps. A force that is right
pointwise can still compound into a spike or a blow-up once integrated forward. This is the sharp contrast
with training *through* the rollout under a trajectory loss (the differentiable-simulation route of
[[differentiating-the-rollout]]), which does see error accumulation and pays for the harder gradient with
long-horizon stability. Learning instantaneous forces is cheap and local; keeping a learned simulator stable
over a long rollout is a global property a per-step loss does not buy.

## What's open

The capillary net is trained against the analytic continuum surface force and is only as good as that
reference. It inherits the diffuse-interface approximation (a smoothed band a few cells wide), and because the
curvature the patch encodes depends on the grid resolution and the number of smoothing passes, the network is
tied to the resolution it trained at. A resolution sweep is untested. The held-out corner shares the trained
strength, so it tests transfer to an unseen *viscosity* combination, not extrapolation to a stronger, unseen
surface tension. And the whole result is one 2D material family with linear schedules and supervised fits; a
nonlinear parameter, or an axis that genuinely enters the learned stress, would be the harder test.
