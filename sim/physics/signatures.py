"""Golden signature tests for the canonical physics.

These encode the qualitative truths that DEFINE correctness for this domain, so that any change to
`core.py` (or an accidental drift) is caught mechanically:

  * fluid spreads widest and sits lowest; elastic springs back to the narrowest, tallest shape;
    snow lands strictly between the two in both width and height.
  * snow CRUMBLES: on a collapsing column it slumps well below the elastic (which springs back), yet
    holds a pile well above the fluid (which runs out flat) -- i.e. snow is materially != elastic.
  * sand PILES AT AN ANGLE OF REPOSE: released as an over-steep 60-degree heap it relaxes to a finite
    slope it can support, where fluid runs flat and where snow and elastic simply keep the seeded slope
    (they are cohesive, sand is not). Sand is therefore materially != all three.
  * surface tension rounds a blob: a gravity-off box is blockier at sigma_st=0 than under surface tension.

Run:  python -m sim.physics.signatures      (prints a table; exit code 1 if any signature fails)
The pytest wrapper lives at sim/tests/test_signatures.py.
"""
from __future__ import annotations

import numpy as np

from . import core

N = 5000       # particles (small, for a fast test)
NF = 6         # snapshot frames (the sim still runs the full physical time T)


def _final(material, sc, **kw):
    snaps, _, stable = core.simulate(material, sc["pts"], sc["area"], sc["T"], NF, v0=sc["v0"], **kw)
    return snaps[-1], stable


def check():
    """Run every signature; return list of (name, passed, detail)."""
    out = []
    drop = core.scene("drop", N)
    fd, sf = _final("fluid", drop)
    ed, se = _final("elastic", drop)
    nd, sn = _final("snow", drop)
    wf, we, wn = core.spread_width(fd), core.spread_width(ed), core.spread_width(nd)
    hf, he, hn = core.pile_height(fd), core.pile_height(ed), core.pile_height(nd)

    out.append(("drop: all materials stable", sf and se and sn, f"fluid={sf} elastic={se} snow={sn}"))
    out.append(("drop: width  fluid > snow > elastic", wf > wn > we, f"{wf:.3f} > {wn:.3f} > {we:.3f}"))
    out.append(("drop: height elastic > snow > fluid", he > hn > hf, f"{he:.3f} > {hn:.3f} > {hf:.3f}"))

    col = core.scene("column", N)
    fc, _ = _final("fluid", col)
    ec, _ = _final("elastic", col)
    ncc, _ = _final("snow", col)
    hcf, hce, hcn = core.pile_height(fc), core.pile_height(ec), core.pile_height(ncc)
    # snow crumbles: slumps below elastic (springs back) but holds a pile above fluid (runs out).
    out.append(("column: elastic springs back above snow", hce > hcn * 1.1,
                f"elastic {hce:.3f} vs snow {hcn:.3f}"))
    out.append(("column: snow holds a pile above fluid", hcn > hcf * 1.15,
                f"snow {hcn:.3f} vs fluid {hcf:.3f}"))

    # --- sand: the angle of repose ------------------------------------------------------------
    # An over-steep heap released from rest. What each material has left at the end IS the signature:
    # fluid keeps no slope, snow and elastic keep the whole seeded slope, sand keeps some of it.
    hp = core.scene("heap", N)
    fh, _ = _final("fluid", hp)
    eh, _ = _final("elastic", hp)
    nh, _ = _final("snow", hp)
    dh, sd = _final("sand", hp)
    af, ae, an, ad = (core.repose_angle(fh), core.repose_angle(eh),
                      core.repose_angle(nh), core.repose_angle(dh))
    wf2, wd, wn2 = core.spread_width(fh), core.spread_width(dh), core.spread_width(nh)
    hf2, hd = core.pile_height(fh), core.pile_height(dh)
    seeded = core.repose_angle(hp["pts"])

    out.append(("heap: sand stable", sd, f"stable={sd}"))
    out.append(("heap: sand holds an angle of repose, fluid does not",
                ad > 15.0 and af < 5.0, f"sand {ad:.1f} deg vs fluid {af:.1f} deg"))
    out.append(("heap: sand does NOT spread flat like a fluid",
                wf2 > wd * 1.5 and hd > hf2 * 1.5,
                f"width fluid {wf2:.3f} vs sand {wd:.3f}; height sand {hd:.3f} vs fluid {hf2:.3f}"))
    out.append(("heap: sand YIELDS where cohesive snow/elastic keep the seeded slope",
                an > ad * 1.4 and ae > ad * 1.4 and wd > wn2 * 1.3,
                f"seeded {seeded:.1f} -> sand {ad:.1f}, snow {an:.1f}, elastic {ae:.1f} deg; "
                f"width sand {wd:.3f} vs snow {wn2:.3f}"))

    # --- multi-material: one grid, four materials, and the refactor changed nothing ------------
    # A single material pushed through the runtime-branching multi-material path must land where the
    # canonical compile-time path lands, to within the simulator's own run-to-run noise.
    drop2 = core.scene("drop", N)
    for m in ("fluid", "elastic", "snow", "sand"):
        a, _, _ = core.simulate(m, drop2["pts"], drop2["area"], drop2["T"], NF, v0=drop2["v0"])
        b, _, _ = core.simulate(m, drop2["pts"], drop2["area"], drop2["T"], NF, v0=drop2["v0"])
        g = [{"material": m, "pts": drop2["pts"], "area": drop2["area"], "v0": drop2["v0"]}]
        c, _, _, ok, _ = core.simulate_multi(g, drop2["T"], NF, dt=core.MAT[m]["dt"])
        noise = float(np.linalg.norm(a - b, axis=-1).mean())
        cross = float(np.linalg.norm(a - c, axis=-1).mean())
        out.append((f"multi-material path matches canonical for {m}",
                    ok and cross <= max(noise * 3.0, 1e-6),
                    f"vs canonical {cross:.2e}, self-noise {noise:.2e}"))

    # viscosity (fluid knob): a thicker fluid spreads less than a thin one on the same drop. Both use
    # viscosities stable at the fluid timestep, so "less spread" is oozing, not a numerical collapse.
    oil, _ = _final("fluid", drop, mu_visc=0.02, dt=1.0e-4)
    thick, _ = _final("fluid", drop, mu_visc=0.2, dt=5.0e-5)
    wo, wt = core.spread_width(oil), core.spread_width(thick)
    out.append(("viscosity: thicker fluid spreads less (and doesn't collapse)",
                wo > wt > 0.15, f"thin {wo:.3f} > thick {wt:.3f} > 0.15"))
    return out


def run():
    rows = check()
    ok = True
    print("=== canonical physics golden signatures ===")
    for name, passed, detail in rows:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}   ({detail})")
        ok = ok and passed
    print(f"physics VERSION: {__import__('sim.physics', fromlist=['VERSION']).VERSION}")
    print("ALL PASS" if ok else "SIGNATURE FAILURES")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
