"""Generate web/params.js DIRECTLY from sim.physics, for the four-material WebGPU demo.

Canonical-physics rule (CLAUDE.md): a port may reimplement the *step*, never the *parameters* or the
*constitutive law*. So no constant is typed by hand in the WGSL or the JS -- this script imports
`sim.physics` and emits the constants file, stamped with the physics version hash.

What it emits beyond the previous (elastic-only) generator:

  * the FULL MAT table for all four canonical materials, including the plastic parameters the WebGPU
    G2P now needs: snow's `xi` / `tc` / `ts`, and sand's Drucker-Prager cone slope `alpha`, computed
    with `sim.physics.core.dp_alpha` rather than re-derived from `phi` in JS.
  * `mat_id`, matching `sim.physics.core.MAT_ID` exactly, because the WGSL branches on the integer.
  * the per-material `dt`, so the demo can compute the SHARED timestep the same way canonical
    `shared_dt` does: min(dt) over the materials actually present.
  * the two FIXED-POINT SCALES the WebGPU P2G needs. WGSL has no atomic float add, so the scatter
    accumulates into `atomic<u32>` (mass) and `atomic<i32>` (momentum). The scales are expressed in
    units of ONE PARTICLE MASS:

        mass_int      = round(mass_float      * (2^kM / p_mass))
        momentum_int  = round(momentum_float  * (2^kV / p_mass))

    so an integer of 2^kM means "one particle's worth of mass at this node".

Run from the repo root:
    .venv/Scripts/python.exe runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/web/gen_params.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import sim.physics as phys                      # noqa: E402
from sim.physics import core as C               # noqa: E402

mats = {}
for name, cfg in C.MAT.items():
    mats[name] = {
        "id": C.MAT_ID[name],
        "E": cfg["E"],
        "dt": cfg["dt"],
        "xi": cfg["xi"],
        "tc": cfg["tc"],
        "ts": cfg["ts"],
        "phi": cfg["phi"],
        # the cone slope itself, NOT phi, so JS never re-derives it
        "alpha": C.dp_alpha(cfg["phi"]),
        "mu": cfg["E"] / (2.0 * (1.0 + C.NU)),
        "la": cfg["E"] * C.NU / ((1.0 + C.NU) * (1.0 - 2.0 * C.NU)),
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
    # --- fixed-point atomics: exponents SET BY MEASUREMENT in the previous WebGPU task ---
    # 2^20 is the obvious first guess and it is wrong: on a contact-heavy scene it lands 79x outside
    # canonical's own noise band. 2^24 lands inside it, matching the exact-f32 CAS path. The cost is
    # RANGE: at 2^24 the u32 mass accumulator saturates at 2^(32-24) = 256 particle masses on one
    # node, and overrun WRAPS SILENTLY. This demo lets a user pile material up by hand, so the
    # headroom is re-measured here under deliberate piling rather than assumed from a fixed scene.
    "kM": 24,
    "kV": 22,
    # --- provenance ---
    "physics_version": phys.VERSION,
    "source": "sim.physics (MAT, MAT_ID, dp_alpha + sim.physics.core world constants)",
}

body = json.dumps(params, indent=2)
out = f"""// GENERATED FILE -- do not edit by hand.
// Emitted by runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/web/gen_params.py,
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
