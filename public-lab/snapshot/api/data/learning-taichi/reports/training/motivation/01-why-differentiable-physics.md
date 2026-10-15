# Why differentiable physics

> Motivation, read first. No equations here, only the shape of the problem and why it is worth your
> time. Everything later in the textbook is in service of the idea on this page.

## Start with what a simulator is

A physics simulator is a function that answers one question. Given the state of the world right now,
what is the state of the world a tiny moment later. State means everything you would need to redraw the
scene and keep going. Where each piece of material is, how fast it is moving, how stretched or squeezed
it is. Call that bundle of numbers $s$. The simulator is a step function that takes $s$ and returns the
next $s$. Run it in a loop and you get motion. A blob of jelly falls, hits the floor, wobbles, settles.

That loop is the whole game. There is no separate "physics engine" doing something mysterious. There is
one step function applied over and over, each application reading the last state and writing the next
one. The art is in making that one step both physically faithful and cheap enough to run thousands of
times per second of simulated time.

If you come from machine learning, you already know this shape. A simulator stepped $T$ times is
structurally identical to a recurrent network unrolled $T$ times, or a residual network with $T$ layers
that all share weights. Same state threaded through the same function again and again. The difference is
only that the step function here is handwritten physics rather than a learned layer. Hold onto that
analogy. It is the bridge that lets your autodiff intuition carry over almost entirely.

## The forward question and the inverse question

Run the simulator forward and you answer the **forward question**. Given these starting conditions,
what happens. Drop the blob from here with this velocity, and watch where it lands. This is what every
game engine and every visual-effects pipeline does. It is genuinely useful and also genuinely limited,
because in most problems you actually care about you do not know the right starting conditions. You know
the outcome you want.

The **inverse question** flips it. Given the outcome I want, what starting conditions produce it. What
initial velocity makes the blob land exactly on that target. What material stiffness makes the cloth
drape the way the costume designer asked for. What sequence of pushes folds the sheet into the shape on
the page. The inverse question is the one that shows up whenever you are trying to *control*, *design*,
or *fit* a physical system rather than just watch it.

A plain forward simulator cannot answer the inverse question directly. The honest brute-force option is
to guess starting conditions, run the simulation, see how wrong the outcome is, adjust the guess, and
repeat. With a handful of knobs you can sometimes search by hand. With thousands of knobs, which is the
realistic case, blind search is hopeless. You need a signal that tells you not just *that* you were
wrong but *which way to move every knob to be less wrong*. That signal is a gradient.

## What "differentiable" buys you

A **differentiable** simulator is one where you can compute the gradient of an outcome with respect to
the inputs. Concretely, you define a single number that measures how bad the outcome is. Call it the
loss. Distance from the blob's final position to the target, say. A differentiable simulator can tell
you the derivative of that loss with respect to every input knob at once. The derivative with respect
to the initial velocity. The derivative with respect to the stiffness. With respect to every one of a
thousand control inputs.

That derivative is a direction in knob-space. It points the way that *increases* the loss, so you step
the opposite way and the outcome gets a little better. Repeat, and you are doing gradient descent on a
physical system. This is exactly the loop that trains neural networks, pointed at simulation parameters
instead of weights. The reason it works at scale is the same reason backprop works at scale. One
backward pass computes the gradient with respect to *all* inputs at roughly the cost of one forward
pass, no matter how many inputs there are. Blind search costs one forward pass per knob per step.
Gradients collapse that to one pass total.

The mechanism that makes this possible is **reverse-mode automatic differentiation**, the same engine as
backpropagation. The prerequisites cover it properly in [[math-toolkit]]. For now the only
claim you need to accept is that if every operation inside the step function is differentiable, then the
whole unrolled rollout is differentiable, and the gradient of the final loss with respect to the very
first input can be computed by walking the computation backward once.

## Why this is the right thing to learn, not a detour

It would be fair to ask why bother with the messy physical step function at all. You could train a
neural network to imitate the dynamics from data and differentiate *that*. People do, and it has its
place. The trouble is that an imitator only knows what it was shown. Push it outside the training
distribution and it confabulates, often confidently and often in ways that violate conservation of
mass or momentum, because nothing in a generic network forces those laws to hold.

A gradient through the *actual* physics does not have that failure mode. It is grounded in the real
dynamics, so the direction it gives you is the true direction in control space that improves the
outcome, not a surrogate that happens to look right on the training set. This is the entire reason this
project starts with explicit, handwritten dynamics rather than a learned black box. You want to
understand where the trustworthy gradient comes from before you start approximating it.

There is a cost, and it is the honest center of this whole study. A handwritten physics step is not
perfectly smooth. Real materials hit walls, stick, separate, and snap, and those events are kinks or
jumps in the step function. The chain rule does not care that a kink is physically reasonable. It
happily multiplies a meaningless slope at a kink into a meaningless gradient, or amplifies a tiny
quantity in a denominator into an overflow. Learning exactly where the gradient is trustworthy and
where it lies to you is the real prize here, and it is the spine of the core sections, especially
[[failure-modes]].

## The through-line to structured generative worlds

This is where it connects to the larger vision and why it is worth your attention specifically. The
goal you are building toward is **structured generative worlds**. Worlds with the open-ended generative
power of diffusion models, but that hold *persistent commitments*, stay coherent over time, and remain
*editable* so an author can shape them on purpose. A generative model that can dream any frame but
cannot keep a promise from one frame to the next is a slot machine, not a world.

Explicit differentiable dynamics are one concrete way to get the structure half of that. They give you
state that persists by construction, conservation laws that hold because they are baked into the step,
and components you can read and edit because they mean something physical. And because the dynamics are
differentiable, they are *authorable*. Authoring a world means steering its evolution toward what you
intend, and a gradient through the dynamics is the most direct handle for that steering you can ask for.
You change what you want the future to be, and the gradient tells you how to change the present to get
there.

That is the bet this textbook is built around. Not that explicit physics is the final form of a
generative world, but that understanding how to push gradients through real dynamics, and where that
breaks, is foundational to ever marrying generative freedom with structural commitment. The Material
Point Method is the specific simulator chosen to learn this on, and the next section explains what it is
and why it is a good choice for the job. See [[where-mpm-sits]].
