"""Regenerate spec/registry/materials.json from the frozen canonical physics.

The registry must never be able to disagree with the code it documents, so the material entries are
GENERATED from `sim.physics`, not hand-written. Run this after any (version-bumping, signature-gated)
change to the canonical materials.

Metrics are hand-authored in spec/registry/metrics.json because they live across many task files and have
no single importable source; each entry cites its own file:line instead.
"""
import json, os, sys

sys.path.insert(0, os.getcwd())
from sim.physics import core as pc          # noqa: E402
from sim.physics import VERSION             # noqa: E402

# Known drift: per-task differentiable variants that reimplemented parameters instead of importing them.
# Recorded here so the registry states the truth rather than an aspiration. See spec/registry/README.md.
DRIFT = {
    "snow": [
        {"where": "sim/one_nn_materials.py:71", "param": "xi", "canonical": 10.0, "actual": 3.0},
        {"where": "sim/learned_materials.py:92", "param": "xi", "canonical": 10.0, "actual": 3.0},
        {"where": "sim/material_variants.py:50", "param": "dt", "canonical": 5.0e-5, "actual": 2.0e-4},
    ],
}

MEANING = {
    "E":  "Young's modulus / bulk stiffness. Sets how hard stress pushes back, and via c=sqrt(E/rho) the "
          "CFL-stable timestep (dt ~ 1/sqrt(E)).",
    "dt": "Stable explicit timestep for this material at the canonical resolution.",
    "xi": "Snow hardening coefficient in h = exp(xi*(1-Jp)): how much compacted snow stiffens. 0 for "
          "materials with no plastic record.",
    "tc": "Plastic clamp, compression side: singular values of F are clamped below 1-tc. 0 = no clamp.",
    "ts": "Plastic clamp, stretch side: singular values of F are clamped above 1+ts. 0 = no clamp.",
    "color": "Canonical render colour, so a material looks the same across every task's figures.",
}

out = {
    "_generated_by": "harness/tools/sync_registry.py -- DO NOT EDIT BY HAND",
    "_source": "sim/physics/core.py (MAT)",
    "physics_version": VERSION,
    "_param_meaning": MEANING,
    "_rule": "A task's differentiable variant may reimplement the STEP; it may not reimplement these "
             "PARAMETERS or the constitutive law. Import them from sim.physics and declare any deviation "
             "in the task contract.",
    "friction": getattr(pc, "FRICTION", None),
    "materials": {},
}

for name, params in pc.MAT.items():
    entry = dict(params)
    entry["signatures"] = "asserted in sim/physics/signatures.py"
    if name in DRIFT:
        entry["_known_drift"] = DRIFT[name]
    out["materials"][name] = entry

dst = os.path.join("spec", "registry", "materials.json")
open(dst, "w", encoding="utf-8").write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print("wrote %s at %s" % (dst, VERSION))
for n, e in out["materials"].items():
    flag = "  <-- %d known drift(s)" % len(e["_known_drift"]) if "_known_drift" in e else ""
    print("  %-8s E=%-6s dt=%-8s xi=%-5s tc=%-8s ts=%s%s"
          % (n, e["E"], e["dt"], e["xi"], e["tc"], e["ts"], flag))
