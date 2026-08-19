"""Contact sheets of the buoyancy runs (water + one solid, two colours)."""
import argparse, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("TASK_SCRATCH", os.path.join(HERE, "_scratch"))
BG = "#0a0e14"
WATER = "#2f7fd0"
SOLID = {"snow": "#f2f6fc", "elastic": "#ff6a4d", "sand": "#ffd24d"}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="after")
ap.add_argument("--keys", required=True)
ap.add_argument("--mats", required=True, help="solid material per key, comma list")
ap.add_argument("--nsel", type=int, default=7)
ap.add_argument("--ymax", type=float, default=0.45)
ap.add_argument("--T", type=float, default=2.2)
ap.add_argument("--out", required=True)
args = ap.parse_args()

d = np.load(os.path.join(SCRATCH, f"buoy_{args.tag}.npz"))
keys = args.keys.split(",")
mats = args.mats.split(",")
fig, axes = plt.subplots(len(keys), args.nsel,
                         figsize=(2.3 * args.nsel, 2.3 * args.ymax / 1.0 * len(keys) + 0.6),
                         facecolor=BG, squeeze=False)
for r, (k, m) in enumerate(zip(keys, mats)):
    sol = d[k + "/solid"].astype(np.float32)
    flu = d[k + "/fluid"].astype(np.float32)
    nf = sol.shape[0]
    idx = np.linspace(0, nf - 1, args.nsel).round().astype(int)
    for c, i in enumerate(idx):
        ax = axes[r, c]
        ax.set_facecolor(BG)
        ax.scatter(flu[i][:, 0], flu[i][:, 1], s=0.6, c=WATER, linewidths=0)
        ax.scatter(sol[i][:, 0], sol[i][:, 1], s=1.6, c=SOLID.get(m, "#dfe6ee"), linewidths=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, args.ymax)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a3444")
        if r == 0:
            ax.set_title("t=%.2f" % (args.T * (i + 1) / nf), color="#7f8ea3", fontsize=8)
        if c == 0:
            ax.set_ylabel(k, color="#dfe6ee", fontsize=9)
fig.tight_layout()
fig.savefig(args.out, facecolor=BG, dpi=95)
print("wrote", args.out)
