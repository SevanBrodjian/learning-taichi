# regenerable intermediates

`bench.json` and `range.json` are the raw browser measurements this task's `metrics.json` was built
from; they are kept because they are small and are the primary record.

Everything large was deleted after scoring and is reproducible:

* `traj_<scene>_<variant>.f32` (14 x 2.4 MB) -- WebGPU trajectories. Regenerate by serving the run dir
  (`verify/serve.py`) and loading `verify/harness.html` from `http://localhost:<port>`.
* `over_*.f32` -- the deliberate-overflow end states, from `verify/harness2.html`.
* `../gt_*.npy` -- canonical ground truth. Regenerate with
  `verify/prepare.py --force-gt` (it caches, so the flag is required).

Order to reproduce the whole task: `web/gen_params.py` -> `verify/prepare.py` ->
`verify/serve.py` + `harness.html` + `harness2.html` in a browser -> `verify/baselines.py` ->
`verify/score.py` -> `verify/render.py` -> `web/build_page.py` -> `write_manifest.py`.
