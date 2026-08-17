// GENERATED FILE -- do not edit by hand.
// Emitted by runs/material-variants/webgpu-port-of-the-interactive-simulation/web/gen_params.py,
// which imports sim.physics and writes MAT["elastic"] plus the frozen world constants verbatim.
// physics_version: phys-bebeaafbe73e
var MPM_PARAMS = {
  "dim": 2,
  "n_grid": 128,
  "dx": 0.0078125,
  "inv_dx": 128.0,
  "p_rho": 1.0,
  "gravity": 9.8,
  "bound": 3,
  "floor_y": 0.0234375,
  "NU": 0.2,
  "FRICTION": 0.5,
  "E": 400.0,
  "dt": 0.0001,
  "color": "#ff9d5c",
  "kM": 24,
  "kV": 22,
  "physics_version": "phys-bebeaafbe73e",
  "source": "sim.physics (MAT['elastic'] + sim.physics.core world constants)"
};
if (typeof module === 'object' && module.exports) { module.exports = MPM_PARAMS; }
