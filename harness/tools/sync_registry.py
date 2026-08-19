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
    # Introduced when elastic got its own Poisson ratio. Every per-task differentiable variant still
    # hardcodes the old single global NU = 0.2, so "elastic" in those files is a much more compressible
    # solid than canonical rubber. Their E values are also the pre-density numbers, but that half is
    # harmless: only E/rho is observable for a material simulated alone, and E/rho did not change for
    # snow or sand. nu is a genuine difference.
    "elastic": [
        {"where": "sim/learned_materials.py:88", "param": "nu", "canonical": 0.45, "actual": 0.2},
        {"where": "sim/one_nn_materials.py:69", "param": "nu", "canonical": 0.45, "actual": 0.2},
        {"where": "sim/material_variants.py:52", "param": "nu", "canonical": 0.45, "actual": 0.2},
        {"where": "sim/material_showcase.py:43", "param": "nu", "canonical": 0.45, "actual": 0.2},
        {"where": "sim/material_diff.py:48", "param": "nu", "canonical": 0.45, "actual": 0.2},
    ],
}

MEANING = {
    "E":  "Young's modulus / bulk stiffness. Sets how hard stress pushes back, and via c=sqrt(E/rho) the "
          "CFL-stable timestep (dt ~ sqrt(rho/E)). Only the RATIO E/rho is observable for a material "
          "simulated on its own -- see 'rho'.",
    "rho": "Density, with water as the unit. What makes sand (1.6) and rubber (1.2) sink and snow (0.3) "
           "float when materials share a grid. No buoyancy force exists in the code: mass is p_vol*rho, "
           "the grid divides scattered momentum by it, and gravity is applied to velocity, so a heavy "
           "node feels less of the surrounding fluid's upward push. A LONE material cannot see its own "
           "density -- (rho,E) -> (k rho, k E) is an exact symmetry -- which is why introducing density "
           "did not move snow or sand: their E/rho is unchanged (snow 150, sand 300, as before).",
    "nu": "Poisson ratio: resistance to changing VOLUME rather than shape. la = E nu/((1+nu)(1-2nu)) "
          "diverges as nu -> 1/2, which is what 'incompressible' means numerically and why rubber "
          "(nu=0.45) costs a smaller timestep than the granular materials (nu=0.20).",
    "fric": "Coulomb friction coefficient at the floor and the side walls. Water is 0 (it does not grip a "
            "smooth boundary); snow, sand and rubber are 0.5. Scattered to the grid mass-weighted, so a "
            "node shared by two materials gets the friction of what is actually on it.",
    "dt": "Stable explicit timestep for this material at the canonical resolution.",
    "xi": "Snow hardening coefficient in h = exp(xi*(1-Jp)): how much compacted snow stiffens. 0 for "
          "materials with no plastic record.",
    "tc": "Plastic clamp, compression side: singular values of F are clamped below 1-tc. 0 = no clamp.",
    "ts": "Plastic clamp, stretch side: singular values of F are clamped above 1+ts. 0 = no clamp.",
    "phi": "Drucker-Prager internal friction angle, degrees: the slope of the granular yield cone via "
           "alpha = sqrt(2/3)*2 sin(phi)/(3-sin(phi)). 0 for materials with no pressure-dependent yield. "
           "NOT the angle of repose -- canonical sand runs phi=50 and MEASURES about 25 (see the "
           "repose_angle metric).",
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
    print("  %-8s E=%-6s dt=%-8s xi=%-5s tc=%-8s ts=%-8s phi=%s%s"
          % (n, e["E"], e["dt"], e["xi"], e["tc"], e["ts"], e.get("phi", 0.0), flag))
