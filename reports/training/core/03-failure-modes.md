# Failure modes

The teaching value of differentiable simulation lives here, in what breaks and why. This section grows as runs surface new pathologies. It opens with the first one we actually hit.

## Failure mode 1 — NaN gradient after convergence
On the first DiffMPM run the optimization converged cleanly, with loss falling from $0.144$ to about $6\times10^{-5}$ by iteration 29. Then at iteration 32 the **backward pass produced a NaN while the forward loss stayed finite** at about $4.6\times10^{-4}$. A guard in the optimizer caught the non-finite gradient and stopped. The forward staying finite while the backward blows up is the tell that the problem is in the sensitivities, not the state.

Three mechanisms are the leading suspects, and they are not exclusive.

- **Out-of-domain particles hitting the wall branch.** Once $v_0$ is large, particles can reach the boundary band. The wall condition zeroes the inward normal velocity with a hard branch, and the base-index floor is non-smooth. Gradients through a branch like that are ill-defined right at the kink, so a particle sitting on the boundary can inject a meaningless sensitivity. This is the contact-differentiability story, seeded from `sim/mpm88.py` lines 49 to 59 and 77.
- **Division by near-zero grid mass.** The grid step divides momentum by node mass, and its backward sensitivity scales like $1/m_i^2$. A node barely grazed by one particle has a tiny mass, so it can amplify a gradient by a huge factor even when the forward velocity looks ordinary.
- **Long-rollout amplification.** Five hundred chained steps form a long product of Jacobians. A single ill-conditioned step can be magnified into an overflow by everything downstream of it, the simulation analogue of an exploding gradient in a deep recurrent net.

## What to try next
Each suspect suggests a concrete probe, and these are the experiments that turn a crash into understanding.
- Gradient clipping or a non-finite skip, to confirm the optimization is otherwise healthy.
- A softened wall condition, replacing the hard zero with a smooth ramp, to test the contact hypothesis directly.
- Higher precision with `default_fp=ti.f64`, to see whether the NaN is a near-overflow rather than a true singularity.
- A shorter horizon, to isolate long-rollout amplification from the other two.

These map onto the **long-rollout gradient pathologies** and **contact and boundary differentiability** directions in `coordination/research_directions.md`. Getting past these cleanly is precisely the skill the project is after, since a world you can author by gradient is only as good as the regions where those gradients can be trusted.
