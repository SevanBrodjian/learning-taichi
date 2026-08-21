import { useEffect, useRef } from "react";
import MPMDemo4 from "./mpm/demo4.js";
import DEMO_CSS from "./mpm/demo4.css.js";

/**
 * The Demo page — the flagship artifact (rebuild-plan Track B5, coordination/demo-mvp.md).
 *
 * A real MLS-MPM simulation of all four canonical materials — water, rubber, snow and sand — on ONE
 * shared grid, stepped on the GPU through WebGPU, in real time, that you can pour into, drag around
 * and carve up.
 *
 * TRANSPLANT CONTRACT: this file imports NOTHING from the harness — no api.js, no shared components,
 * no app CSS. The only imports are the demo's own bundle in ./mpm/, which is itself framework-free
 * and network-free. To lift the whole thing onto a personal site, copy ./mpm/ and either render
 * <DemoView /> or drop the four plain files (params.js, mpm4-webgpu.js, demo4.js, demo4.css) into a
 * page with four <script>/<link> tags — that standalone page exists and is the same code:
 *   runs/material-variants/incorporate-improved-materials-on-real-demo-page-and-improve-polish/web/demo.html
 * Do not add a harness import here without breaking that promise on purpose.
 *
 * SOURCE OF TRUTH is that run's web/ directory. ./mpm/ is generated from it by web/sync_to_dashboard.py;
 * edit the run's copy and re-run the sync, or the Demo tab and the transplantable page will drift.
 *
 * PHYSICS: every constant comes from sim.physics (physics_version phys-c518316a4a05) via a generated
 * params.js — nothing is retyped into JS. The step is reimplemented in WGSL, the parameters and the
 * constitutive laws are not. Each material carries its OWN density, Poisson ratio and wall friction, so
 * the sink/float ordering (snow floats, rubber and sand sink) is an OUTPUT of the mass ratio and not of
 * any buoyancy term — there is none. Verified against canonical `simulate_multi` on buoyancy and
 * four-material scenes; every one lands inside or beside canonical's own self-noise band.
 *
 * RENDERING: the material view accumulates PER-MATERIAL weight in the four channels of one rgba16float
 * target, so a single resolve pass can shade water, snow, sand and rubber differently without a second
 * pass or a material-id buffer. Sand adds one pass of six hashed grains per particle. `draw({treatment:
 * 'mvp'})` selects the previously shipped single treatment, which exists so a before/after comparison can
 * change one thing at a time.
 *
 * LAYOUT: the field is sized `min(100%, 100cqh)` and is square at every viewport. It must stay square —
 * the canvas backing store is square and the domain is the unit square, so a non-square frame does not
 * crop the simulation, it stretches it.
 *
 * AESTHETIC: spec/aesthetic.md at FULL strength (the dashboard chrome gets the mild version).
 * Frutiger-era gloss — aqueous sheen, refracted iso-surface, scanlines, glass — around controls that
 * are solid, mechanical and informative. The joke is that it is genuinely fast: a 2000s-looking
 * surface running a real 20,000-substep-per-simulated-second solver at 60+ fps.
 *
 * EASTER EGGS live in demo4.js and are documented there. Per spec, none of them announce themselves
 * and none of them address the reader.
 */
export default function DemoView() {
  const hostRef = useRef(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let handle = null;
    let dead = false;
    // mount() is synchronous; the GPU device arrives on handle.ready. A tab switch before the
    // device resolves must still tear the loop down, hence the `dead` flag.
    try {
      handle = MPMDemo4.mount(host, {});
      if (handle && handle.ready && handle.ready.catch) handle.ready.catch(() => {});
    } catch (e) {
      host.textContent = "The demo failed to start: " + String((e && e.message) || e);
    }
    return () => {
      dead = true;
      if (handle && handle.stop) {
        try { handle.stop(); } catch (e) { /* already torn down */ }
      }
      void dead;
    };
  }, []);

  return (
    <>
      <style>{DEMO_CSS}</style>
      <div ref={hostRef} style={{ position: "absolute", inset: 0 }} />
    </>
  );
}
