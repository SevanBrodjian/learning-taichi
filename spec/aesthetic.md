# Aesthetic — the look and feel

Authored from Sevan's own description, 2026-08-01. This is a **passion project that will live on his
personal website**, not a corporate research dashboard. Design accordingly.

## The core of it

> Cryptic, emotional, graphics/simulation heavy. Retro themes — Frutiger Aero era — but delivered with the
> polish and fluidity of 2026. **The kind of thing a company would never ship**, because it may be too
> off-putting, even borderline antagonistic to users.

Touchstones he named: **iPad Baby, Suburban Basketball, Crypt Underworld, Cruelty Squad.**

And, in deliberate tension with that:

> **Industrial simplicity.** The site should always have a strong feeling of *"it just works."* Think
> **Factorio** or **RimWorld** — every button feels solid and reliable, the UI informative and
> straightforward.

Holding both at once is the whole assignment. The chrome is **solid, legible, trustworthy**; the
atmosphere around it is **dark, strange, and personal.** Never sacrifice the first for the second — an
unusable page is not edgy, it is broken.

## The 2000s, actually delivered

> Use the effects that the 2000s era was *trying* to use, but implemented with the smoothness and
> capability of 2026 technology.

Gloss, bloom, glass, refraction, scanlines, lens flare, chrome, skeuomorphic depth, aquatic gradients —
the whole Frutiger Aero vocabulary. The 2000s wanted these and shipped them at 12 fps in a JPEG. Ship them
at 120 fps, GPU-composited, resolution-independent, responding to input in real time. **The joke is that
it is genuinely fast.**

## Resolution — the deliberate contradiction

He stated both, and resolved it himself:

> The overall **content** should be delivered at **high res** — vector, sharp at any DPI. But if
> components, backgrounds, or certain aesthetics are intentionally **low-sample or low-poly**, that is
> perfectly acceptable. Imagine a **low-poly character with weirdly fluid animations.**

So: **crisp delivery of deliberately coarse things.** A 64-sample field rendered at device resolution and
animated at full framerate. Never accidental blur, never upscaled raster. Low *sample count* is a
choice; low *fidelity* is a bug.

## Emotional register

> Dark, Halloweeny, hidden-secrets, small details and easter eggs — with a weird kind of **resolute
> optimism in darkness**, and promotion of **self-expression**.

Reward attention. Things that only appear on hover, after a delay, at certain values, or on the third
visit. Nothing critical hidden behind a secret; everything *delightful* is fair game.

## Where to apply it, and how hard

| surface | intensity |
| --- | --- |
| **Demo page** | **Full strength.** This is the flagship artifact for his website. Go hard. |
| **Dashboard chrome** (tabs, task pages, training report) | **Mild.** A touch-up, not a rebuild. It has to stay a working tool he reads on an iPad. |

His own guardrail, quoted so it is not lost:

> *"That's a very strong aesthetic I've just described, so don't worry about overfitting to it. Probably
> tone it down against what I just described a bit, especially for the overall dashboard design."*

The current dashboard's failure mode is that it *"leans a little too sterilized AI design."* The fix is
character and intent, not decoration for its own sake. If an effect makes the tool harder to use, it is
wrong. If a page looks like every other AI-generated dashboard, that is also wrong.

## Practical rules

- **Vector and procedural first.** SVG, canvas, WebGL, CSS. Anything that must be sharp on a Retina iPad
  and at 4K without shipping five raster sizes.
- **Motion is real, not decorative.** Physics-driven or simulation-driven where possible — this is a
  differentiable-simulation project, so the motion on screen should plausibly *be* simulation.
- **Every control feels mechanical.** Immediate feedback, obvious affordance, no ambiguity about state.
  Factorio's buttons, not a startup's.
- **Performance is part of the aesthetic.** Jank breaks the entire premise. Prefer GPU-composited
  transforms, `requestAnimationFrame`, and honest frame budgets. Degrade gracefully on the iPad.
- **Respect `prefers-reduced-motion`** — an ambient background may idle, but nothing essential should
  require motion to be understood.
- **Self-contained.** No CDNs, no external fonts, no network calls (the same constraint bespoke task
  pages already live under).

## The Demo page specifically

Its own deliverable, described in `coordination/rebuild-plan.md` (Track B). Two hard constraints:

1. **Transplantable.** It must lift onto his personal website **separately from the dashboard** — a
   self-contained bundle with no dependency on the data server, the task registry, or the harness.
2. **Standalone-legible.** A visitor who knows nothing about MPM or this project should find it immersive
   and worth their time. The dashboard is for him; the Demo is for everyone else. **No jargon.**

Current state: a deliberately-designed **"no demo exists yet"** placeholder. It should look like a highly
polished demo that happens to be empty — not like an unfinished page.
