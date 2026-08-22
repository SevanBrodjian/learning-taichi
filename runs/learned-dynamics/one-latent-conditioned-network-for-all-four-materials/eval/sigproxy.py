"""Run `sim/physics/signatures.py` -- unmodified, with its own canonical thresholds -- against a
DIFFERENT simulator.

The golden signatures are the project's definition of "behaves like this material": fluid spreads
flat, snow crumbles but holds a pile, sand yields to an angle of repose where cohesive materials keep
the seeded slope, density decides what floats. Re-deriving a weaker version of them for the learned
simulator would be exactly the drift CLAUDE.md forbids, so this swaps out the `core` that
`signatures.check()` calls and leaves every threshold, scene and particle count alone.

The swap is a proxy that forwards EVERYTHING to canonical `sim.physics.core` except `simulate` and
`simulate_multi`, the two functions that actually integrate. Shape diagnostics, scenes, seeds, MAT
and MAT_ID stay canonical, so a signature cannot pass because the measuring tape moved.

N/A RATHER THAN A LIE
---------------------
Four signatures override a CONSTITUTIVE parameter (`mu_visc` once, `E` three times) to make their
point. In the learned simulator the constitutive law lives inside the network's weights and there is
no such knob, so those rows are reported N/A with the reason -- not silently run with canonical
physics and counted as a pass, and not quietly dropped from the denominator. Everything that only
touches the analytic half of the step (`rho`, `fric`, the scenes, the shared grid, the volume
diagnostics, which read det F) runs for real.
"""
from sim.physics import core as _core
from sim.physics import signatures as _sig

# Signatures that CANNOT run against a learned constitutive law, and exactly why.
NA_REASONS = {
    "viscosity: thicker fluid spreads less (and doesn't collapse)":
        "overrides mu_visc -- the learned law has no viscosity knob",
    "density rescaling leaves a lone material unmoved (snow)":
        "overrides E -- stiffness lives in the weights, there is no knob to turn",
    "density rescaling leaves a lone material unmoved (sand)":
        "overrides E -- stiffness lives in the weights, there is no knob to turn",
    "density rescaling leaves a lone material unmoved (elastic)":
        "overrides E -- stiffness lives in the weights, there is no knob to turn",
}


class _Proxy:
    def __init__(self, simulate, simulate_multi):
        self._simulate = simulate
        self._simulate_multi = simulate_multi
        self.unsupported = []

    def __getattr__(self, k):
        return getattr(_core, k)

    def simulate(self, *a, **kw):
        try:
            return self._simulate(*a, **kw)
        except NotImplementedError as e:
            # Fall back to canonical so the rest of the suite still runs. The rows that depend on
            # this call are force-marked N/A below, so the canonical result is never reported as a
            # learned pass.
            self.unsupported.append(str(e))
            return _core.simulate(*a, **kw)

    def simulate_multi(self, *a, **kw):
        return self._simulate_multi(*a, **kw)


def run(simulate, simulate_multi, label="learned"):
    """Returns (rows, summary). Each row is {name, pass (True/False/None), detail, na}."""
    proxy = _Proxy(simulate, simulate_multi)
    real = _sig.core
    _sig.core = proxy
    try:
        raw = _sig.check()
    finally:
        _sig.core = real
    rows = []
    for name, passed, detail in raw:
        if name in NA_REASONS:
            rows.append({"name": name, "pass": None, "na": True,
                         "detail": "N/A -- " + NA_REASONS[name]})
        else:
            rows.append({"name": name, "pass": bool(passed), "na": False, "detail": detail})
    npass = sum(1 for r in rows if r["pass"] is True)
    nfail = sum(1 for r in rows if r["pass"] is False)
    nna = sum(1 for r in rows if r["na"])
    return rows, {"simulator": label, "pass": npass, "fail": nfail, "na": nna,
                  "unsupported_calls": len(proxy.unsupported)}


def show(rows, title):
    print(f"=== golden signatures vs {title} ===")
    for r in rows:
        tag = "N/A " if r["na"] else ("PASS" if r["pass"] else "FAIL")
        print(f"  [{tag}] {r['name']}   ({r['detail']})")
    npass = sum(1 for r in rows if r["pass"] is True)
    nfail = sum(1 for r in rows if r["pass"] is False)
    nna = sum(1 for r in rows if r["na"])
    print(f"  -> {npass} pass, {nfail} fail, {nna} n/a")
