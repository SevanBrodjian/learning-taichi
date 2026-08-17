"""B4 — apply the canonical tags and RE-DERIVE the whole task graph.

Why this exists: the edges on the board were whatever happened to be typed at creation time. The result
was 11 links across 21 tasks, twelve of which were orphans, and no cross-direction connections at all —
so the Map could not read as a lineage. Per rebuild-plan B4b, linking is the orchestrator's job now:
the user proposes and may cite, the orchestrator decides the real edges.

This script encodes that derivation ONCE, from the actual content of the 21 tasks. It is re-runnable and
idempotent: it rewrites `tags` and `follow_up_of` and recomputes every `follow_ups` from scratch.

Edge kinds (see EDGE_KINDS in harness/server/app.py):
  extends         built on top of the parent's result
  re-does         redid the parent properly; the parent's result is superseded but kept for the record
  refutes         overturned the parent's conclusion
  applies         borrowed the parent's method/machinery for a different question
  prerequisite-of the parent had to exist first (capability, not conclusion)
"""
import json, os, collections

DIR = os.path.join("coordination", "directions")

# ── tags ────────────────────────────────────────────────────────────────────────────────────────────
# Exactly four, multi-tagged. Decided 2026-08-03; do not expand without a decision.
TAGS = {
    "throw-to-target":                                   ["gradients"],
    "optimizer-comparison":                              ["gradients"],
    "learned-residual":                                  ["learned", "gradients"],
    "nan-root-cause":                                    ["gradients"],
    "softened-wall":                                     ["gradients"],
    "resolution-memory":                                 ["gradients"],
    "checkpointing-long-horizon":                        ["gradients"],
    "jacobian-norms":                                    ["gradients"],
    "implement-nondifferentiable-material-variants":     ["materials"],
    "fluids-snow-and-solids-as-differentiable-simulations": ["materials", "gradients"],
    "varying-liquid-viscosity":                          ["materials"],
    "implement-liquids-across-viscosity-and-surface-tension": ["materials"],
    "train-and-interpolate-nns-to-mimic-viscous-liquids": ["learned", "materials"],
    "train-material-replicating-nns-and-interpolate":     ["learned", "materials"],
    "one-nn-for-three-materials":                        ["learned", "materials"],
    "generalize-one-nn-across-viscosity-and-surface-tension": ["learned", "materials"],
    "train-one-nn-to-mimic-viscosity-and-st":            ["learned", "materials"],
    "non-differentiable-fluid-renderer":                 ["rendering"],
    "improve-basic-fluid-sim-realism":                   ["rendering", "materials"],
    "gpu-accelerate-fluid-renderer":                     ["rendering"],
    "more-realistic-basic-fluid-sims":                   ["rendering"],
    "interactive-simulation-of-one-material":            ["materials"],
}

# ── the derived graph ───────────────────────────────────────────────────────────────────────────────
# child -> [(parent_id, kind, why)].  `why` is documentation for the reviewer, not stored.
GRAPH = {
    # gradients spine: one inverse problem, then everything learned from making it work
    "optimizer-comparison": [
        ("throw-to-target", "extends", "same rollout + loss, swaps the optimizer"),
    ],
    "nan-root-cause": [
        ("throw-to-target", "extends", "diagnoses the failure that task hit at longer horizons"),
    ],
    "softened-wall": [
        ("nan-root-cause", "extends", "fixes the non-smooth contact that root-cause identified"),
    ],
    "resolution-memory": [
        ("throw-to-target", "extends", "scales the same rollout; measures the memory/throughput wall"),
    ],
    "checkpointing-long-horizon": [
        ("resolution-memory", "extends", "attacks the memory wall that task measured"),
    ],
    "jacobian-norms": [
        ("nan-root-cause", "extends", "measures the per-step amplification root-cause argued for"),
        ("softened-wall", "applies", "needs the smoothed contact to isolate amplification from the branch"),
    ],

    # materials spine
    "fluids-snow-and-solids-as-differentiable-simulations": [
        ("implement-nondifferentiable-material-variants", "extends", "same three materials, now differentiable"),
    ],
    "varying-liquid-viscosity": [
        ("fluids-snow-and-solids-as-differentiable-simulations", "extends", "adds a viscosity axis to the fluid"),
    ],
    "implement-liquids-across-viscosity-and-surface-tension": [
        ("varying-liquid-viscosity", "extends", "adds surface tension as a second liquid axis"),
    ],

    # learned-material arc — the project's main story
    "learned-residual": [
        ("throw-to-target", "applies", "reuses the differentiable rollout + tape to train weights inside it"),
    ],
    "train-and-interpolate-nns-to-mimic-viscous-liquids": [
        ("varying-liquid-viscosity", "extends", "learns the viscosity family that task generated"),
        ("learned-residual", "applies", "same idea of a network inside the solver, now for the stress"),
    ],
    "train-material-replicating-nns-and-interpolate": [
        ("train-and-interpolate-nns-to-mimic-viscous-liquids", "extends", "same weight-blend, structurally different materials"),
        ("fluids-snow-and-solids-as-differentiable-simulations", "applies", "borrows its three constitutive laws as targets"),
    ],
    "one-nn-for-three-materials": [
        ("train-material-replicating-nns-and-interpolate", "refutes",
         "weight-blending was shown degenerate; this replaces it with descriptor conditioning"),
    ],
    "generalize-one-nn-across-viscosity-and-surface-tension": [
        ("one-nn-for-three-materials", "extends", "same conditioning protocol on a two-axis liquid"),
        ("implement-liquids-across-viscosity-and-surface-tension", "applies", "fits the ground-truth liquids that task built"),
    ],
    "train-one-nn-to-mimic-viscosity-and-st": [
        ("generalize-one-nn-across-viscosity-and-surface-tension", "re-does",
         "the redo: learns the WHOLE material, not just the stress. Supersedes it."),
    ],

    # rendering spine, plus the cross-links that were missing entirely
    "improve-basic-fluid-sim-realism": [
        ("non-differentiable-fluid-renderer", "extends", "fixes the two tells that renderer left"),
        ("varying-liquid-viscosity", "applies", "its sluggishness diagnosis is the viscosity/damping axis"),
    ],
    "gpu-accelerate-fluid-renderer": [
        ("improve-basic-fluid-sim-realism", "extends", "ports that exact pipeline to the GPU, parity-checked"),
    ],
    "interactive-simulation-of-one-material": [
        ("implement-nondifferentiable-material-variants", "applies",
         "ports that forward elastic material out of Taichi into the browser; forward sim, no gradients"),
        ("gpu-accelerate-fluid-renderer", "applies",
         "same method - profile where the frame time ACTUALLY goes before optimising - but it lands on the "
         "opposite answer: there the render was the whole cost and the physics free; here draw is 0.15ms and "
         "the 167 substeps are the wall"),
    ],

    "more-realistic-basic-fluid-sims": [
        ("gpu-accelerate-fluid-renderer", "applies", "only affordable because the GPU port made it ~130x faster"),
        ("improve-basic-fluid-sim-realism", "extends", "showcases the realism work at length"),
    ],
}

# ── apply ───────────────────────────────────────────────────────────────────────────────────────────
files = {}
where = {}                                   # task id -> direction id
for fn in sorted(os.listdir(DIR)):
    if not fn.endswith(".json"):
        continue
    p = os.path.join(DIR, fn)
    d = json.load(open(p, encoding="utf-8"), object_pairs_hook=collections.OrderedDict)
    files[p] = d
    for t in d.get("tasks", []):
        where[t["id"]] = d.get("id", fn[:-5])

known = set(where)
unknown_tag = [k for k in TAGS if k not in known]
unknown_edge = [c for c in GRAPH if c not in known] + \
               [p for v in GRAPH.values() for (p, _, _) in v if p not in known]
if unknown_tag or unknown_edge:
    raise SystemExit("unknown task ids -> tags:%s edges:%s" % (unknown_tag, sorted(set(unknown_edge))))
untagged = sorted(known - set(TAGS))
if untagged:
    raise SystemExit("these tasks would end up untagged: %s" % untagged)

# 1. tags + 2. parents (normalized, direction-qualified, typed); 3. wipe follow_ups for recompute
children = collections.defaultdict(list)
for p, d in files.items():
    for t in d.get("tasks", []):
        tid = t["id"]
        t["tags"] = TAGS[tid]
        parents = GRAPH.get(tid, [])
        if parents:
            t["follow_up_of"] = [{"id": pid, "dir": where[pid], "kind": kind} for pid, kind, _ in parents]
            for pid, _, _ in parents:
                children[pid].append(tid)
        else:
            t.pop("follow_up_of", None)
        t["follow_ups"] = []

# 4. recompute follow_ups from the parent table so both directions always agree
for p, d in files.items():
    for t in d.get("tasks", []):
        kids = children.get(t["id"], [])
        if kids:
            t["follow_ups"] = [{"id": c, "dir": where[c]} for c in kids]
        else:
            t.pop("follow_ups", None)

for p, d in files.items():
    open(p, "w", encoding="utf-8").write(json.dumps(d, indent=2, ensure_ascii=False) + "\n")

n_edges = sum(len(v) for v in GRAPH.values())
cross = sum(1 for c, v in GRAPH.items() for (pid, _, _) in v if where[pid] != where[c])
kinds = collections.Counter(k for v in GRAPH.values() for (_, k, _) in v)
roots = sorted(t for t in known if t not in GRAPH)
print("tasks           %d" % len(known))
print("edges           %d  (was 11)" % n_edges)
print("cross-direction %d  (was 0)" % cross)
print("kinds           %s" % dict(kinds))
print("roots           %s" % roots)
tagc = collections.Counter(x for v in TAGS.values() for x in v)
print("tags            %s" % dict(tagc))
