"""Generate web/params.js DIRECTLY from sim.physics.

The canonical-physics rule says a port may reimplement the *step* but must not invent the
*parameters*. So the parameters are not typed by hand anywhere in the JS: this script imports
`sim.physics` and emits the constants file, stamped with the physics version hash. If the canonical
module ever changes, re-running this regenerates the file and the stamp changes with it.

Run from the repo root:
    .venv/Scripts/python.exe runs/material-variants/interactive-simulation-of-one-material/web/gen_params.py
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
    # --- provenance ---
    "physics_version": phys.VERSION,
    "source": "sim.physics (MAT['elastic'] + sim.physics.core world constants)",
}

body = json.dumps(params, indent=2)
out = f"""// GENERATED FILE -- do not edit by hand.
// Emitted by runs/material-variants/interactive-simulation-of-one-material/web/gen_params.py,
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
