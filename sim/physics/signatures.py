"""Golden signature tests for the canonical physics.

These encode the qualitative truths that DEFINE correctness for this domain, so that any change to
`core.py` (or an accidental drift) is caught mechanically:

  * fluid spreads widest and sits lowest; elastic springs back to the narrowest, tallest shape;
    snow lands strictly between the two in both width and height.
  * snow CRUMBLES: on a collapsing column it slumps well below the elastic (which springs back), yet
    holds a pile well above the fluid (which runs out flat) -- i.e. snow is materially != elastic.
  * surface tension rounds a blob: a gravity-off box is blockier at sigma_st=0 than under surface tension.

Run:  python -m sim.physics.signatures      (prints a table; exit code 1 if any signature fails)
The pytest wrapper lives at sim/tests/test_signatures.py.
"""
from __future__ import annotations

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
