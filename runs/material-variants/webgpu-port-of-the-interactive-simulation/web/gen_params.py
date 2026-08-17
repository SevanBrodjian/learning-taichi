"""Generate web/params.js DIRECTLY from sim.physics, for the WebGPU port.

Canonical-physics rule: a port may reimplement the *step*, never the *parameters*. So no constant is
typed by hand in the WGSL or the JS -- this script imports `sim.physics` and emits the constants file,
stamped with the physics version hash.

It also emits the two FIXED-POINT SCALES the WebGPU P2G needs. WGSL has no atomic float add, so the
scatter accumulates into `atomic<u32>` (mass) and `atomic<i32>` (momentum). The scales are expressed in
units of ONE PARTICLE MASS rather than as raw absolute numbers, because that is the only way to pick
them without knowing the scene:

    mass_int      = round(mass_float      * (2^kM / p_mass))
    momentum_int  = round(momentum_float  * (2^kV / p_mass))

so an integer of 2^kM means "one particle's worth of mass at this node" and an integer of 2^kV means
"one particle mass times one unit of velocity". p_mass = area/n * p_rho is known at seed time, so the
scale is computed per scene at run time from kM/kV, which are the only free knobs.

Run from the repo root:
    .venv/Scripts/python.exe runs/material-variants/webgpu-port-of-the-interactive-simulation/web/gen_params.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import sim.physics as phys                      # noqa: E402
from sim.physics import core as C               # noqa: E402

ELASTIC = phys.MAT["elastic"]

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
    # --- the elastic material, verbatim from MAT["elastic"] ---
    "E": ELASTIC["E"],
    "dt": ELASTIC["dt"],
    "color": ELASTIC["color"],
    # --- fixed-point atomics: default exponents, SET BY MEASUREMENT (verify/score.py) ---
    # 2^20 is the obvious first guess and it is wrong: on a contact-heavy scene it lands 79x
    # outside canonical's own noise band. 2^24 lands inside it, matching the exact-f32 CAS path.
    # The cost is range: at 2^24 the u32 mass accumulator saturates at 2^(32-24) = 256 particle
    # masses on one node, and the heaviest node runs at roughly 2x the particles-per-cell, so this
    # default is good to about 120 particles per cell. Denser than that needs a smaller kM (and
    # loses accuracy) or more than 32 bits.
    "kM": 24,        # mass     quanta per particle mass       -> u32 saturates at 2^(32-24) = 256 pm
    "kV": 22,        # momentum quanta per particle-mass*vel   -> i32 saturates at 2^(31-22) = 512 pm*v
    # --- provenance ---
    "physics_version": phys.VERSION,
    "source": "sim.physics (MAT['elastic'] + sim.physics.core world constants)",
}

body = json.dumps(params, indent=2)
out = f"""// GENERATED FILE -- do not edit by hand.
// Emitted by runs/material-variants/webgpu-port-of-the-interactive-simulation/web/gen_params.py,
// which imports sim.physics and writes MAT["elastic"] plus the frozen world constants verbatim.
// physics_version: {phys.VERSION}
var MPM_PARAMS = {body};
if (typeof module === 'object' && module.exports) {{ module.exports = MPM_PARAMS; }}
"""
dst = pathlib.Path(__file__).with_name("params.js")
dst.write_text(out, encoding="utf-8")
print("wrote", dst)
print("physics_version:", phys.VERSION)
print(json.dumps(params, indent=2))
