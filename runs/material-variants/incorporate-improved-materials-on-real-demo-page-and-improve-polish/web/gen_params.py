"""Generate web/params.js DIRECTLY from sim.physics, for the four-material WebGPU demo.

Canonical-physics rule (CLAUDE.md): a port may reimplement the *step*, never the *parameters* or the
*constitutive law*. So no constant is typed by hand in the WGSL or the JS -- this script imports
`sim.physics` and emits the constants file, stamped with the physics version hash.

WHAT CHANGED AT phys-c518316a4a05 (and why this generator had to change with it)
--------------------------------------------------------------------------------
The previous generator emitted ONE global Poisson ratio and no density at all, because canonical had
only one of each. Canonical now carries THREE quantities per material that used to be global, and all
three are load-bearing for a scene that mixes materials:

  * `rho`  -- density. A particle's mass is p_vol*rho, so a heavy material is genuinely heavier on the
              shared grid. This is the ONLY thing that makes sand and rubber sink and snow float:
              there is no buoyancy force in canonical physics and there must not be one in the port.
              Emitted per material; the WGSL computes the particle mass from it.
  * `nu`   -- Poisson ratio, 0.45 for rubber and 0.20 for the rest. It feeds BOTH Lame parameters, so
              `mu` and `la` below are computed with the material's OWN nu, not the module-level NU.
              (Emitting mu/la with the global NU is exactly the bug this comment exists to prevent:
              rubber would look right in the JSON and be four times too compressible in the shader.)
  * `fric` -- Coulomb friction at the floor and the walls, 0 for water and 0.5 for the rest.
              Canonical scatters a MASS-WEIGHTED friction to the grid, so a node shared by two
              materials gets the friction of whatever is actually sitting on it; the WGSL mirrors
              that with a second fixed-point accumulator rather than a per-node material id.

Beyond that it emits, as before: the full MAT table for all four canonical materials including the
plastic parameters the WebGPU G2P needs (snow's `xi`/`tc`/`ts`, sand's Drucker-Prager cone slope
`alpha` computed with `sim.physics.core.dp_alpha` rather than re-derived from `phi` in JS), `mat_id`
matching `sim.physics.core.MAT_ID` exactly, and the per-material `dt` so the demo can compute the
SHARED timestep the same way canonical `shared_dt` does: min(dt) over the materials present.

  * the two FIXED-POINT SCALES the WebGPU P2G needs. WGSL has no atomic float add, so the scatter
    accumulates into `atomic<u32>` (mass, and mass*friction) and `atomic<i32>` (momentum). The scales
    are expressed in units of ONE REFERENCE PARTICLE MASS -- p_vol * rho_ref with rho_ref = 1, i.e.
    water. It has to be a fixed reference rather than "the particle's own mass" now that mass is
    per-material, or two materials would scatter into the same accumulator on different scales.

        mass_int      = round(mass_float      * (2^kM / p_mass_ref))
        momentum_int  = round(momentum_float  * (2^kV / p_mass_ref))

Run from the repo root:
    .venv/Scripts/python.exe runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/web/gen_params.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import sim.physics as phys                      # noqa: E402
from sim.physics import core as C               # noqa: E402

mats = {}
for name, cfg in C.MAT.items():
    # Per-material Poisson ratio. `.get(..., C.NU)` is the same fallback canonical's own simulate()
    # uses; every canonical material overrides it, so the fallback should never fire.
    nu = cfg.get("nu", C.NU)
    mats[name] = {
        "id": C.MAT_ID[name],
        "E": cfg["E"],
        "rho": cfg.get("rho", C.p_rho),
        "nu": nu,
        "fric": cfg.get("fric", C.FRICTION),
        "dt": cfg["dt"],
        "xi": cfg["xi"],
        "tc": cfg["tc"],
        "ts": cfg["ts"],
        "phi": cfg["phi"],
        # the cone slope itself, NOT phi, so JS never re-derives it
        "alpha": C.dp_alpha(cfg["phi"]),
        # Lame parameters from the material's OWN nu (see the module docstring).
        "mu": cfg["E"] / (2.0 * (1.0 + nu)),
        "la": cfg["E"] * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        "color": cfg["color"],
    }

params = {
    # --- world constants, verbatim from sim/physics/core.py ---
    "dim": C.dim,
    "n_grid": C.n_grid,
    "dx": C.dx,
    "inv_dx": C.inv_dx,
    "p_rho": C.p_rho,
    "gravity": C.gravity,
    "bound": C.bound,
    "floor_y": C.floor_y,
    "NU": C.NU,
    "FRICTION": C.FRICTION,
    "MAX_P": C.MAX_P,
    # --- every canonical material, verbatim from MAT ---
    "materials": mats,
    "mat_order": ["fluid", "elastic", "snow", "sand"],
    "mat_id": C.MAT_ID,
    # --- fixed-point atomics: exponents SET BY MEASUREMENT in the WebGPU port task ---
    # 2^20 is the obvious first guess and it is wrong: on a contact-heavy scene it lands 79x outside
    # canonical's own noise band. 2^24 lands inside it, matching the exact-f32 CAS path. The cost is
    # RANGE: at 2^24 the u32 mass accumulator saturates at 2^(32-24) = 256 REFERENCE particle masses
    # on one node, and overrun WRAPS SILENTLY. Per-material density eats into that headroom, since
    # the heaviest material (sand, rho 1.6) reaches the ceiling at 256/1.6 = 160 of its own
    # particles on one node. `rho_max` is emitted so the demo can state that ceiling rather than
    # assume it.
    "kM": 24,
    "kV": 22,
    "rho_max": max(m["rho"] for m in mats.values()),
    # --- provenance ---
    "physics_version": phys.VERSION,
    "source": "sim.physics (MAT, MAT_ID, dp_alpha + sim.physics.core world constants)",
}

body = json.dumps(params, indent=2)
out = f"""// GENERATED FILE -- do not edit by hand.
// Emitted by runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/web/gen_params.py,
// which imports sim.physics and writes the whole MAT table plus the frozen world constants verbatim.
// physics_version: {phys.VERSION}
var MPM_PARAMS = {body};
if (typeof module === 'object' && module.exports) {{ module.exports = MPM_PARAMS; }}
if (typeof window !== 'undefined') {{ window.MPM_PARAMS = MPM_PARAMS; }}
"""
dst = pathlib.Path(__file__).with_name("params.js")
dst.write_text(out, encoding="utf-8")
print("wrote", dst)
print("physics_version:", phys.VERSION)
print(json.dumps(params, indent=2))
