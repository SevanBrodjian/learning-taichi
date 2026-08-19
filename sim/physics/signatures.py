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
  * DENSITY DECIDES WHAT FLOATS. Released at rest inside a pool of water, snow rises to the surface and
    rides on it, rubber and sand fall to the bottom, and the SAME blob at four different densities
    settles monotonically deeper as it gets heavier. No buoyancy force exists anywhere in core.py, so
    this is the mass ratio in the transfer showing through.
  * A LONE MATERIAL CANNOT SEE ITS OWN DENSITY. (rho, E) -> (k rho, k E) leaves a single material's
    motion unchanged, because the momentum balance only ever contains E/rho. This is a gauge symmetry
    of the model, and it is what let per-material density be introduced without moving snow or sand.
  * RUBBER AND WATER HOLD THEIR VOLUME. Through a hard floor impact an elastic blob keeps its area and
    no part of it is crushed to a fraction of its size; the weakly-compressible fluid likewise stays
    close to J = 1. Both are statements about the VOLUMETRIC response, which is what the Poisson ratio
    governs for the solid and what E governs for the fluid.
  * FRICTION IS A MATERIAL PROPERTY. The same fluid released as a dam runs measurably further when it
    is given water's zero friction than when it is given a granular material's.

Run:  python -m sim.physics.signatures      (prints a table; exit code 1 if any signature fails)
The pytest wrapper lives at sim/tests/test_signatures.py.
"""
from __future__ import annotations

import numpy as np

from . import core

N = 5000       # particles (small, for a fast test)
NF = 6         # snapshot frames (the sim still runs the full physical time T)
NP = 4000      # particles for the pool scenes (water + one solid share this budget)


def _final(material, sc, **kw):
    snaps, _, stable = core.simulate(material, sc["pts"], sc["area"], sc["T"], NF, v0=sc["v0"], **kw)
    return snaps[-1], stable


def _volume_floor(material, sc, times, **kw):
    """Worst volume ratio the material reaches over a set of sampled times.

    det(F) is the model's own volume ratio for a solid (J for the fluid), so `initial_area * mean(det F)`
    is literally the area the body occupies. Sampling several times is necessary because the squash is a
    transient: a settled blob looks fine long after it was crushed on impact.

    Returns (worst mean over the samples, worst 1st-percentile particle over the samples)."""
    wm, wp = 1e9, 1e9
    for T in times:
        core.simulate(material, sc["pts"], sc["area"], float(T), 1, v0=sc["v0"], **kw)
        d = core.J.to_numpy()[:N] if material == "fluid" else np.linalg.det(core.F.to_numpy()[:N])
        wm = min(wm, float(d.mean()))
        wp = min(wp, float(np.percentile(d, 1)))
    return wm, wp


def _pool(solid, rho=None, T=1.6):
    """Release `solid` at rest, fully submerged, in a pool of water. Returns
    (stable, how far the body MOVED in absolute height, change in its depth below the free surface,
    final submerged fraction). Negative height movement is sinking; negative depth change is rising
    relative to the water around it.

    Both measures are reported because each has a weakness and they fail differently. The absolute
    height is clean but says nothing about where the water is. The depth below the surface is the
    physically meaningful one, but the pool's free surface itself creeps downward over a long roll
    (the weakly-compressible fluid takes its pressure from an advected J, not from the actual particle
    packing, so a settling pack is not pushed back on), which eats into the measured sinking of a body
    that is only slightly denser than water.

    Reference values are read off the SEED positions, not off the first captured frame. A captured
    frame already sits one frame-interval into the roll, by which time a heavy blob has begun to fall,
    and comparing against it hides most of the effect."""
    p = core.scene_pool(solid, NP, T=T, rho=rho)
    water0, solid0 = p["groups"][0]["pts"], p["groups"][1]["pts"]
    d0 = core.rest_depth(solid0, water0)
    y0 = float(solid0[:, 1].mean())
    sn, _, mid, ok, _ = core.simulate_multi(p["groups"], p["T"], 3)
    sel = mid == core.MAT_ID[solid]
    d1 = core.rest_depth(sn[-1][sel], sn[-1][~sel])
    y1 = float(sn[-1][sel][:, 1].mean())
    return ok, y1 - y0, d1 - d0, core.submerged_fraction(sn[-1][sel], sn[-1][~sel])


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

    # --- density: what floats is decided by the mass ratio, not by a buoyancy term -------------
    # Each solid starts at rest, fully submerged, in a pool of water. Nothing in core.py applies an
    # upward force to anything; the only quantity that differs between these runs is particle mass.
    ok_s, y_s, d_s, sub_s = _pool("snow")
    ok_e, y_e, d_e, sub_e = _pool("elastic")
    ok_d, y_d, d_d, sub_d = _pool("sand")
    out.append(("pool: snow FLOATS -- it rises to the surface and breaks it",
                ok_s and y_s > 0.008 and d_s < -0.03 and sub_s < 0.70,
                f"rose {y_s:+.3f}, depth change {d_s:+.3f}, submerged {sub_s:.2f}"))
    out.append(("pool: rubber SINKS -- it goes down and stays fully under",
                ok_e and y_e < -0.02 and sub_e > 0.95,
                f"moved {y_e:+.3f}, depth change {d_e:+.3f}, submerged {sub_e:.2f}"))
    out.append(("pool: sand SINKS, and faster than rubber does",
                ok_d and y_d < -0.04 and y_d < y_e and sub_d > 0.95,
                f"moved {y_d:+.3f} vs rubber {y_e:+.3f}, submerged {sub_d:.2f}"))
    out.append(("pool: resting depth is ordered by density (sand > rubber > snow)",
                d_d > d_e > d_s,
                f"sand {d_d:+.3f} > rubber {d_e:+.3f} > snow {d_s:+.3f}  "
                f"(rho {core.MAT['sand']['rho']} / {core.MAT['elastic']['rho']} / "
                f"{core.MAT['snow']['rho']})"))
    # The control that makes it a result rather than a coincidence: ONE material, one stiffness, one
    # scene, three densities. If the outcome follows rho it is buoyancy and not a quirk of a
    # constitutive model.
    _, yl, dl, bl = _pool("elastic", rho=0.3)
    _, yn, dn, bn = _pool("elastic", rho=1.0)
    _, yh, dh, bh = _pool("elastic", rho=1.6)
    out.append(("pool: the SAME blob orders by DENSITY alone (0.3 floats, 1.0 hangs, 1.6 sinks)",
                yl > 0.008 and yh < -0.04 and yl > yn > yh and bl < 0.70 and bh > 0.95,
                f"rho 0.3 -> {yl:+.3f} ({100 * bl:.0f}% under),  1.0 -> {yn:+.3f} ({100 * bn:.0f}%),  "
                f"1.6 -> {yh:+.3f} ({100 * bh:.0f}%)"))

    # --- density is a gauge for a LONE material: (rho, E) -> (k rho, k E) changes nothing --------
    # The momentum balance rho Dv/Dt = div(sigma) + rho g has sigma proportional to E, so only E/rho is
    # physical for a single material. This is why giving snow rho = 0.3 and sand rho = 1.6 did not move
    # either of them, and it is the guard against a future density edit quietly changing a material.
    for m in ("snow", "sand", "elastic"):
        a, _, _ = core.simulate(m, drop2["pts"], drop2["area"], drop2["T"], NF, v0=drop2["v0"])
        b, _, _ = core.simulate(m, drop2["pts"], drop2["area"], drop2["T"], NF, v0=drop2["v0"])
        k = core.MAT[m]["rho"]
        c, _, ok = core.simulate(m, drop2["pts"], drop2["area"], drop2["T"], NF, v0=drop2["v0"],
                                 rho=1.0, E=core.MAT[m]["E"] / k)
        noise = float(np.linalg.norm(a - b, axis=-1).mean())
        cross = float(np.linalg.norm(a - c, axis=-1).mean())
        # The absolute floor matters here. The elastic path is almost bit-deterministic, so its
        # run-to-run noise is ~3e-6 domain units and three times nothing is still nothing. 2e-4 is a
        # fortieth of a grid cell and a twentieth of the particle spacing: unmoved by any physical
        # standard, while still catching a real change, which would be orders of magnitude larger.
        out.append((f"density rescaling leaves a lone material unmoved ({m})",
                    ok and cross <= max(noise * 5.0, 2e-4),
                    f"rho {k} vs rescaled to 1.0: {cross:.2e}, self-noise {noise:.2e}"))

    # --- volume: rubber is nearly incompressible, and so is water -------------------------------
    slam = core.scene("slam", N)
    em, ep = _volume_floor("elastic", slam, (0.09, 0.12, 0.16, 0.22, 0.30))
    out.append(("slam: rubber holds its VOLUME through a hard impact",
                em > 0.95 and ep > 0.45,
                f"worst mean det(F) {em:.3f} (>0.95), worst 1st-pct particle {ep:.3f} (>0.45)"))
    fm, fp = _volume_floor("fluid", drop, (0.28, 0.34, 0.42, 0.55, 0.80))
    out.append(("drop: water is nearly incompressible",
                fm > 0.97 and fp > 0.85,
                f"worst mean J {fm:.3f} (>0.97), worst 1st-pct particle {fp:.3f} (>0.85)"))

    # --- friction is a per-material property, not one number for the whole world ----------------
    # Same fluid, same dam, same seed; only the friction coefficient it carries differs. Measured while
    # the front is still running FREE across the floor. Waiting for the settled state would measure the
    # reflection off the far wall instead, which reverses the ordering and says nothing about the floor.
    # Measured as the distance the front has travelled from the dam face at a fixed early time, while
    # it is still running FREE across the floor. Two traps this avoids: the settled width reverses the
    # ordering, because by then it is measuring the reflection off the far wall; and a late-time front
    # speed understates the effect, because Coulomb friction only bites while a node is still moving
    # downward, which is the first part of the collapse.
    dam = core.scene("dam", N)
    x_dam = 0.22

    def _run(**kw):
        sn, _, _ = core.simulate("fluid", dam["pts"], dam["area"], 0.20, 1, v0=dam["v0"], **kw)
        return float(np.percentile(sn[-1][:, 0], 99)) - x_dam

    rs, rg = _run(), _run(fric=0.5)
    out.append(("dam break: frictionless water outruns the same water given a granular friction",
                rs > rg * 1.06,
                f"front has travelled {rs:.3f} vs {rg:.3f} at t = 0.20 s"))
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
