# Epochs

An **epoch** is a cut across the whole project at an inflection point. It exists so that "what did this
look like in August" is a question with an answer you can open, not a git archaeology exercise.

## What an epoch is NOT
Not a folder that tasks get swept into when they pile up. Archiving for its own sake adds ceremony and
buys nothing — the task graph, the `T-nnn` refs and the `created` dates already give ordering and history,
and git already has every byte.

An epoch is worth cutting only when something is **true at that moment that will stop being true**.

## What an epoch bundles
Cut at an inflection point (the first demo, a direction changing, a result that resets the plan), an
epoch captures four things that otherwise drift apart:

1. **The report.** Sevan's hand-written `reports/research_report.md`, as it stood, having **passed**
   (`spec/examination.md`). The epoch does not close until it passes.
2. **A frozen, loadable demo.** The demo is deliberately self-contained — the transplant contract means
   `components/mpm/` imports nothing from the harness — so freezing it is a copy, not a build system.
   That constraint was maintained for portability and this is the second thing it buys.
3. **The physics version.** `phys-<hash>`, so the frozen demo's behaviour is reproducible and provably
   the same ground truth the runs used.
4. **The task set and graph** at that instant: which tasks existed, their refs, their edges.

The point is that these four are only meaningful *together*. A frozen demo whose physics version is
unknown is a curiosity; a report whose evidence has since been superseded is misleading. Cutting them as
one is what makes an epoch a citable state of the world.

## Layout
```
coordination/epochs/<n>-<slug>/
  epoch.json        # cut date, physics_version, task refs + edges, demo version, report verdict
  report.md         # frozen copy of Sevan's report as it passed
  verdict.md        # the grader's verdict on it
demo-versions/<n>-<slug>/   # the frozen, loadable demo build (self-contained)
```

The live files keep moving. The epoch copy does not.

## Cadence
Cut **at inflection points, never on a schedule**. Two or three a year is plausible; monthly would be
ceremony. If it is not obvious that something has shifted, it has not.
