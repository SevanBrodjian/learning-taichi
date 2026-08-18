// GENERATED FILE -- do not edit by hand.
// Emitted by runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/web/gen_params.py,
// which imports sim.physics and writes the whole MAT table plus the frozen world constants verbatim.
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
  "MAX_P": 16384,
  "materials": {
    "fluid": {
      "id": 0,
      "E": 180.0,
      "dt": 0.00012,
      "xi": 0.0,
      "tc": 0.0,
      "ts": 0.0,
      "phi": 0.0,
      "alpha": 0.0,
      "mu": 75.0,
      "la": 50.0,
      "color": "#4db6ff"
    },
    "elastic": {
      "id": 1,
      "E": 400.0,
      "dt": 0.0001,
      "xi": 0.0,
      "tc": 0.0,
      "ts": 0.0,
      "phi": 0.0,
      "alpha": 0.0,
      "mu": 166.66666666666669,
      "la": 111.11111111111111,
      "color": "#ff9d5c"
    },
    "snow": {
      "id": 2,
      "E": 150.0,
      "dt": 5e-05,
      "xi": 10.0,
      "tc": 0.025,
      "ts": 0.0075,
      "phi": 0.0,
      "alpha": 0.0,
      "mu": 62.5,
      "la": 41.66666666666667,
      "color": "#e6ecff"
    },
    "sand": {
      "id": 3,
      "E": 300.0,
      "dt": 0.0001,
      "xi": 0.0,
      "tc": 0.0,
      "ts": 0.0,
      "phi": 50.0,
      "alpha": 0.5599687663604146,
      "mu": 125.0,
      "la": 83.33333333333334,
      "color": "#ffd24d"
    }
  },
  "mat_order": [
    "fluid",
    "elastic",
    "snow",
    "sand"
  ],
  "mat_id": {
    "fluid": 0,
    "elastic": 1,
    "snow": 2,
    "sand": 3
  },
  "kM": 24,
  "kV": 22,
  "physics_version": "phys-bebeaafbe73e",
  "source": "sim.physics (MAT, MAT_ID, dp_alpha + sim.physics.core world constants)"
};
if (typeof module === 'object' && module.exports) { module.exports = MPM_PARAMS; }
if (typeof window !== 'undefined') { window.MPM_PARAMS = MPM_PARAMS; }
