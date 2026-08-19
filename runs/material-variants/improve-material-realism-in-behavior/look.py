"""Contact sheets straight from saved snapshots, so a rollout can actually be LOOKED at.

Not a deliverable -- this is the "open the figure before writing a finding" step. Writes into the
scratchpad, not the run directory.
"""
import argparse, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("TASK_SCRATCH", os.path.join(HERE, "_scratch"))

BG = "#0a0e14"
COL = {"fluid": "#4db6ff", "elastic": "#ff6a4d", "snow": "#f2f6fc", "sand": "#ffd24d"}

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="before")
ap.add_argument("--keys", required=True, help="comma list of scene/material")
ap.add_argument("--nsel", type=int, default=8)
ap.add_argument("--out", required=True)
ap.add_argument("--ymax", type=float, default=0.75)
ap.add_argument("--f0", type=int, default=0)
ap.add_argument("--f1", type=int, default=-1)
ap.add_argument("--pt", type=float, default=0.8)
args = ap.parse_args()

tags = args.tags.split(",")
keys = args.keys.split(",")
data = {t: np.load(os.path.join(SCRATCH, f"snaps_{t}.npz")) for t in tags}
T_OF = {"drop": 1.3, "column": 1.7, "heap": 1.6, "slam": 1.0, "dam": 1.4}

rows = [(t, k) for t in tags for k in keys]
nsel = args.nsel
ph = max(1.1, 2.0 * args.ymax)
fig, axes = plt.subplots(len(rows), nsel, figsize=(2.0 * nsel, ph * len(rows) + 0.5),
                         facecolor=BG)
axes = np.atleast_2d(axes)
for r, (t, k) in enumerate(rows):
    snaps = data[t][k].astype(np.float32)
    nfr = snaps.shape[0]
    times = np.linspace(1, nfr, nfr) * (T_OF[k.split("/")[0]] / nfr)
    f1 = nfr - 1 if args.f1 < 0 else args.f1
    idx = np.linspace(args.f0, f1, nsel).round().astype(int)
    mat = k.split("/")[1]
    for c, i in enumerate(idx):
        ax = axes[r, c]
        ax.set_facecolor(BG)
        ax.scatter(snaps[i][:, 0], snaps[i][:, 1], s=args.pt, c=COL.get(mat, "#dfe6ee"), linewidths=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, args.ymax)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a3444")
        if r == 0:
            ax.set_title(f"t={times[i]:.2f}", color="#7f8ea3", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"{t}\n{k}", color="#dfe6ee", fontsize=8)
fig.tight_layout()
fig.savefig(args.out, facecolor=BG, dpi=95)
print("wrote", args.out)
