"""Golden signature test for the canonical physics (sim/physics).

Runnable two ways:
  python -m pytest sim/tests/test_signatures.py     (if pytest is installed)
  python sim/tests/test_signatures.py               (standalone; exit 1 on failure)

Any change to sim/physics/core.py that breaks a qualitative invariant (snow stops crumbling, fluid
stops spreading, the fluid/snow/elastic ordering flips) fails here. Physics is only promoted / changed
when this stays green (CLAUDE.md -> "Canonical physics").
"""
from sim.physics.signatures import check


def test_signatures():
    rows = check()
    failed = [(name, detail) for name, ok, detail in rows if not ok]
    assert not failed, f"canonical physics signature failures: {failed}"


if __name__ == "__main__":
    import sys
    from sim.physics.signatures import run
    sys.exit(0 if run() else 1)
