# MPM in context

> Prerequisite, skim-level. Enough background to make the core sections land. Depth grows here over time (see the training-report spec).

## What the Material Point Method is
The Material Point Method (MPM) is a hybrid simulation scheme. The material lives on a cloud of **particles** that carry mass, velocity, and deformation, while the equations of motion are solved on a temporary **background grid** that is thrown away and rebuilt every step. Each step moves information particle to grid, updates the grid, then moves it back grid to particle. Particles never connect to each other through a fixed mesh, which is exactly why MPM handles enormous deformation, splitting, and merging without the tangling that breaks mesh methods.

## Why this hybrid, and where it sits
Pure particle methods like SPH make neighbor queries expensive and pressure noisy. Pure grid (Eulerian) methods advect fields across cells and smear sharp features. Mesh methods like FEM are accurate for modest deformation but fail when the mesh tangles. MPM keeps the strengths of both halves. Particles give a clean Lagrangian record of history and material, and the grid gives a cheap regular structure for computing forces and resolving contact. That regular grid is also what makes MPM a natural fit for a GPU, since the heavy work is structured stencil math over a dense array.

## Why a *differentiable* MPM matters here
A forward simulator answers one question, which is what happens next. A differentiable simulator also answers the inverse question, which is what input would produce a wanted outcome, because gradients of an outcome with respect to the inputs flow back through the whole rollout. That inversion is the bridge to the broader goal of [[structured generative worlds]]. A world model that is both generative and *controllable* needs dynamics you can author against, and gradients are the most direct handle for authoring. Learning where those gradients are trustworthy and where they break is the real prize, and it is the spine of the [[core]] sections.

## The vocabulary you will see
- **P2G / G2P** are the particle-to-grid and grid-to-particle transfers that bracket every step.
- **APIC** is the transfer rule that also carries a small affine velocity field per particle, which kills the energy loss and noise of older transfers. The affine matrix is defined in the [[math-toolkit]].
- **Constitutive model** is the stress-strain law that turns deformation into force. The first runs use a simple weakly compressible elastic model, and material variants are a queued research direction.
