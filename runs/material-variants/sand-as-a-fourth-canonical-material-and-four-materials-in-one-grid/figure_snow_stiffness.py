"""Redraw the snow-hardening figure honestly: the distribution is BIMODAL and spans four decades.

A linear axis clipped at 3000 hid both facts, and quoting only the median described neither lobe. Snow
after impact splits into an expanded surface population that the same hardening law makes very SOFT
(Jp > 1 gives exp(xi(1-Jp)) < 1) and a compacted interior that becomes far stiffer than any other
canonical material. The stiff tail is the one that matters for a timestep, since stability is set by the
stiffest particle present, not the typical one.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

import render as R                                  # noqa: E402
import sim.physics as P                             # noqa: E402
from sim.physics import core as pc                  # noqa: E402

N = 4000
sc = P.scene("drop", N)
P.simulate("snow", sc["pts"], sc["area"], 0.9, 4)
jp = pc.Jp.to_numpy()[:N]
eff = P.MAT["snow"]["E"] * np.exp(P.MAT["snow"]["xi"] * (1.0 - jp))
eff = np.clip(eff, 1e-1, None)

stats = {"E_eff_median": float(np.median(eff)), "E_eff_p95": float(np.percentile(eff, 95)),
         "E_eff_p99": float(np.percentile(eff, 99)), "E_eff_max": float(eff.max()),
         "frac_softer_than_nominal": float((eff < P.MAT["snow"]["E"]).mean()),
         "frac_stiffer_than_elastic": float((eff > P.MAT["elastic"]["E"]).mean()),
         "Jp_median": float(np.median(jp)), "Jp_gt_1_frac": float((jp > 1).mean())}
print(json.dumps(stats, indent=1))

fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=140, facecolor=R.BG)
ax.set_facecolor(R.BG)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#2a3444")
ax.tick_params(colors=R.MUTED, labelsize=9)
ax.grid(alpha=0.13, color="#3a4658", lw=0.7)
ax.set_axisbelow(True)
bins = np.logspace(-1, np.log10(max(eff.max(), 1e4)), 70)
ax.hist(eff, bins=bins, color=R.DEMO_COLOR["snow"], alpha=0.85)
ax.set_xscale("log")
top = ax.get_ylim()[1]
for m, lab in (("fluid", "water"), ("sand", "sand"), ("elastic", "elastic")):
    ax.axvline(P.MAT[m]["E"], color=R.DEMO_COLOR[m], lw=2.0)
    ax.text(P.MAT[m]["E"] * 1.04, top * 0.97, f"{lab} E={P.MAT[m]['E']:.0f}",
            color=R.DEMO_COLOR[m], fontsize=9, rotation=90, va="top")
ax.axvline(P.MAT["snow"]["E"], color="#8899aa", lw=2.0, ls="--")
ax.text(P.MAT["snow"]["E"] * 0.94, top * 0.97, "snow nominal E=150", color="#8899aa", fontsize=9,
        rotation=90, va="top", ha="right")
ax.annotate("", xy=(1.5, top * 0.42), xytext=(90, top * 0.42),
            arrowprops=dict(arrowstyle="->", color="#8fa0b3", lw=1.2))
ax.text(9, top * 0.46, "expanded snow:\nthe same law SOFTENS it", color="#8fa0b3", fontsize=8.6,
        ha="center")
ax.annotate("", xy=(4200, top * 0.42), xytext=(700, top * 0.42),
            arrowprops=dict(arrowstyle="->", color="#8fa0b3", lw=1.2))
ax.text(1900, top * 0.46, "compacted snow:\nstiffer than anything else", color="#8fa0b3",
        fontsize=8.6, ha="center")
ax.set_xlabel("effective stiffness  E · exp(ξ(1−Jp))  after impact   (log scale)", color=R.INK,
              fontsize=10)
ax.set_ylabel("particles", color=R.INK, fontsize=10)
ax.set_title(f"Snow's hardening is real and bimodal: 95th pct {stats['E_eff_p95']:.0f} against "
             f"elastic's 400,\nbut it is NOT what pins snow's timestep", color=R.INK, fontsize=11.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "snow_stiffness.png"), facecolor=R.BG)
plt.close(fig)
json.dump(stats, open(os.path.join(D, "snow_stiffness.json"), "w"), indent=1)
print("wrote snow_stiffness.png")
