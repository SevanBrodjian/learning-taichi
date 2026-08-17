import { useEffect, useRef, useState } from "react";

/**
 * The Demo page — the flagship artifact (rebuild-plan Track B5).
 *
 * TRANSPLANT CONTRACT: this file imports NOTHING from the harness (no api.js, no shared components, no
 * app CSS). Its styles are injected inline below. To lift it onto a personal site, copy this file and
 * render <DemoView />, or copy the canvas logic straight into a plain <script>. Do not add a harness
 * import here without breaking that promise on purpose.
 *
 * AESTHETIC: see spec/aesthetic.md, at FULL strength here (the dashboard gets the mild version).
 * - Low sample count, high resolution: ~2600 particles drawn as hard little squares, but the canvas is
 *   sized to devicePixelRatio so nothing is ever blurry. Coarse thing, crisp delivery.
 * - Frutiger-era vocabulary (aqueous gradients, bloom, glass, scanlines) run at 60fps instead of the
 *   12fps slideshow the 2000s actually shipped. The joke is that it is fast.
 * - It is a flow field, not decoration: this is a simulation project, so the motion is simulated.
 *
 * EASTER EGGS — per spec/aesthetic.md, the bar is that you cannot tell whether it is an egg, a bug, or
 * normal functionality. Nothing addresses the user. There are three, none announced:
 *   1. The field REMEMBERS where the pointer dwelt. Wells persist and decay over ~40s, so the flow near
 *      somewhere you have been behaves differently than it did. Reads as state that shouldn't exist.
 *   2. One particle ignores the field entirely and drifts on its own heading forever. Reads as stuck.
 *   3. The STATUS readout very rarely resolves to something other than UNDER CONSTRUCTION for a single
 *      half-second tick, then goes back. Reads as a glitch.
 * A previous build shipped a timed "you stayed. that counts for something." — legible, warm, and exactly
 * the wrong register. Removed.
 */

const CSS = `
.demo-root{position:absolute;inset:0;overflow:hidden;background:#05070b;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#cfe6ef;isolation:isolate}
.demo-canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
/* aqueous sheen — the one unapologetically Frutiger element */
.demo-sheen{position:absolute;inset:-20%;pointer-events:none;mix-blend-mode:screen;opacity:.5;
  background:
    radial-gradient(38% 30% at 22% 18%, rgba(60,220,255,.28), transparent 70%),
    radial-gradient(34% 26% at 82% 76%, rgba(150,60,255,.24), transparent 70%),
    radial-gradient(50% 40% at 50% 110%, rgba(20,255,180,.14), transparent 70%);
  animation:demo-drift 24s ease-in-out infinite alternate}
@keyframes demo-drift{from{transform:translate3d(-2%,-1%,0) scale(1)}to{transform:translate3d(2%,1.5%,0) scale(1.06)}}
.demo-scan{position:absolute;inset:0;pointer-events:none;opacity:.16;
  background:repeating-linear-gradient(180deg,rgba(255,255,255,.10) 0 1px,transparent 1px 3px)}
.demo-vig{position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(72% 62% at 50% 46%,transparent 40%,rgba(0,0,0,.82) 100%)}
.demo-plate{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);pointer-events:none;
  text-align:center;padding:0 24px;width:min(760px,92vw)}
.demo-kicker{font-size:11px;letter-spacing:.42em;color:#5f8ea0;margin:0 0 18px;text-transform:uppercase}
.demo-title{font-size:clamp(30px,7.2vw,68px);line-height:.98;margin:0;letter-spacing:-.02em;font-weight:700;
  color:#eaf8ff;text-shadow:0 0 26px rgba(70,220,255,.35),0 0 3px rgba(255,255,255,.5)}
.demo-title .dim{color:#3d5a68;text-shadow:none}
.demo-rule{height:1px;margin:22px auto;width:min(420px,70%);
  background:linear-gradient(90deg,transparent,rgba(110,220,255,.55),transparent)}
.demo-sub{font-size:13px;line-height:1.75;color:#8fb3c2;margin:0 auto;max-width:56ch}
.demo-sub b{color:#cfe6ef;font-weight:600}
/* industrial readout: solid, informative, "it just works" */
.demo-hud{position:absolute;left:18px;bottom:16px;pointer-events:none;font-size:10.5px;letter-spacing:.13em;
  color:#4d7686;display:flex;gap:18px;flex-wrap:wrap}
.demo-hud b{color:#7fd8f0;font-weight:600}
.demo-badge{position:absolute;right:18px;bottom:16px;font-size:10.5px;letter-spacing:.16em;color:#3d5a68;
  border:1px solid #16303a;padding:5px 10px;border-radius:3px;background:rgba(8,16,22,.6);pointer-events:none}
@media (prefers-reduced-motion: reduce){.demo-sheen{animation:none}}
`;

// Plausible-looking system states. None of them explain themselves; all of them imply the page is a
// surface over something that is still running.
const OTHER_STATUS = ["RESOLVING", "LISTENING", "SOLVER IDLE", "AWAITING GEOMETRY", "STEP 0 OF ——"];

export default function DemoView() {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const [fps, setFps] = useState(0);
  // Egg 3: the status readout is *usually* UNDER CONSTRUCTION.
  const [status, setStatus] = useState("UNDER CONSTRUCTION");

  // WebGPU capability probe, visible on whatever device is holding the dashboard -- the iPad and phone
  // cannot easily be inspected any other way, and their support decides whether a JS fallback is needed.
  //
  // IT MUST DISTINGUISH "unsupported" FROM "hidden". navigator.gpu is only exposed in a SECURE CONTEXT.
  // localhost counts as secure; http://<lan-ip>:5174 does not. So every device reaching this dashboard
  // over the LAN sees no navigator.gpu at all, regardless of whether its browser supports WebGPU. A probe
  // that just reports "ABSENT" there is actively misleading -- it blames the device for a transport
  // problem. (This is a local-dev issue only: a portfolio site on HTTPS is a secure context.)
  const [gpu, setGpu] = useState("probing…");
  useEffect(() => {
    let alive = true;
    const set = (v) => { if (alive) setGpu(v); };
    (async () => {
      try {
        if (!navigator.gpu) {
          set(window.isSecureContext
            ? "UNSUPPORTED (secure ctx, no API)"
            : `HIDDEN — needs HTTPS (origin ${location.protocol}//${location.hostname})`);
          return;
        }
        const a = await navigator.gpu.requestAdapter();
        if (!a) { set("NO ADAPTER (API present)"); return; }
        const d = await a.requestDevice().catch(() => null);
        const who = (a.info && (a.info.vendor || a.info.architecture))
          ? `${a.info.vendor || "?"}/${a.info.architecture || "?"}` : "adapter";
        set(d ? `YES · ${who}` : `ADAPTER ONLY · ${who}`);
      } catch (e) { set("ERROR: " + String(e).slice(0, 40)); }
    })();
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    const cv = canvasRef.current;
    const wrap = wrapRef.current;
    if (!cv || !wrap) return;
    const ctx = cv.getContext("2d", { alpha: false });
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let W = 0, H = 0, dpr = 1;
    const N = 2600;                       // low sample count, deliberately
    const px = new Float32Array(N), py = new Float32Array(N);
    const vx = new Float32Array(N), vy = new Float32Array(N);
    const seed = new Float32Array(N);
    const pointer = { x: -1e6, y: -1e6, on: false };

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2.5);   // crisp on Retina, capped for the iPad
      const r = wrap.getBoundingClientRect();
      W = Math.max(1, Math.floor(r.width));
      H = Math.max(1, Math.floor(r.height));
      cv.width = Math.floor(W * dpr);
      cv.height = Math.floor(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#05070b";
      ctx.fillRect(0, 0, W, H);
    }

    function scatter() {
      for (let i = 0; i < N; i++) {
        px[i] = Math.random() * W;
        py[i] = Math.random() * H;
        vx[i] = 0; vy[i] = 0;
        seed[i] = Math.random();
      }
    }

    resize(); scatter();
    const onResize = () => { resize(); scatter(); };
    window.addEventListener("resize", onResize);

    const move = (e) => {
      const r = cv.getBoundingClientRect();
      const p = e.touches ? e.touches[0] : e;
      pointer.x = p.clientX - r.left; pointer.y = p.clientY - r.top; pointer.on = true;
    };
    const leave = () => { pointer.on = false; pointer.x = pointer.y = -1e6; };
    wrap.addEventListener("pointermove", move);
    wrap.addEventListener("pointerleave", leave);
    wrap.addEventListener("touchmove", move, { passive: true });

    // Egg 1: the field remembers. Dwelling deposits a well that persists and decays over ~40s, so the
    // flow somewhere you have been is not the flow you left. Bounded ring, so it cannot grow unbounded.
    const WELLS = 10;
    const wx = new Float32Array(WELLS), wy = new Float32Array(WELLS), wAge = new Float32Array(WELLS);
    let wNext = 0, dwell = 0;
    const WELL_LIFE = 40;

    // A cheap curl-ish field. Not MPM — but it is a real velocity field being integrated, which is the
    // point: the motion on a simulation project's front page should actually be simulated.
    function field(x, y, t) {
      const a = Math.sin(x * 0.0042 + t * 0.30) + Math.cos(y * 0.0037 - t * 0.22);
      const b = Math.cos(x * 0.0031 - t * 0.19) + Math.sin(y * 0.0048 + t * 0.27);
      let fx = b * 26, fy = -a * 26;
      for (let k = 0; k < WELLS; k++) {
        if (wAge[k] <= 0) continue;
        const dx = x - wx[k], dy = y - wy[k];
        const d2 = dx * dx + dy * dy;
        if (d2 > 90000) continue;
        const w = (wAge[k] / WELL_LIFE) * 5200 / (d2 + 2200);
        fx += -dy * w; fy += dx * w;          // a slow residual swirl where attention was paid
      }
      return [fx, fy];
    }

    let raf = 0, last = performance.now(), acc = 0, frames = 0, t0 = performance.now();
    function frame(now) {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      const t = (now - t0) / 1000;

      // Trails: fade instead of clear. Cheap, and it reads as long-exposure rather than smear.
      ctx.fillStyle = "rgba(5,7,11,0.17)";
      ctx.fillRect(0, 0, W, H);

      // age the wells; dwelling in one spot deposits a new one
      for (let k = 0; k < WELLS; k++) if (wAge[k] > 0) wAge[k] -= dt;
      if (pointer.on) {
        dwell += dt;
        if (dwell > 0.9) {
          dwell = 0;
          wx[wNext] = pointer.x; wy[wNext] = pointer.y; wAge[wNext] = WELL_LIFE;
          wNext = (wNext + 1) % WELLS;
        }
      } else dwell = 0;

      for (let i = 0; i < N; i++) {
        // Egg 2: one particle does not participate. It holds its own heading and wraps forever.
        if (i === 0) {
          px[i] += 7.5 * dt; py[i] += 2.5 * dt;
          if (px[i] > W + 4) px[i] = -4;
          if (py[i] > H + 4) py[i] = -4;
          ctx.fillStyle = "rgba(150,220,240,0.5)";
          ctx.fillRect(px[i], py[i], 1.2, 1.2);
          continue;
        }
        const [fx, fy] = field(px[i], py[i], reduced ? 0 : t);
        vx[i] += (fx - vx[i]) * 1.6 * dt;
        vy[i] += (fy - vy[i]) * 1.6 * dt;

        if (pointer.on) {
          const dx = px[i] - pointer.x, dy = py[i] - pointer.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 34000 && d2 > 0.5) {
            const inv = 1 / Math.sqrt(d2);
            const push = 2300 / (d2 + 900);
            vx[i] += dx * inv * push; vy[i] += dy * inv * push;
          }
        }

        px[i] += vx[i] * dt; py[i] += vy[i] * dt;
        if (px[i] < -4) px[i] = W + 4; else if (px[i] > W + 4) px[i] = -4;
        if (py[i] < -4) py[i] = H + 4; else if (py[i] > H + 4) py[i] = -4;

        const s = seed[i];
        const speed = Math.min(1, Math.hypot(vx[i], vy[i]) / 90);
        // teal -> violet by speed, with a rare warm particle (a small secret)
        if (s > 0.985) ctx.fillStyle = "rgba(255,168,92,0.95)";
        else ctx.fillStyle = `rgba(${(60 + speed * 130) | 0},${(210 - speed * 90) | 0},${(235 - speed * 20) | 0},${0.28 + speed * 0.6})`;
        const sz = s > 0.985 ? 2.4 : (s > 0.7 ? 1.8 : 1.2);   // hard squares: low-poly, not soft blur
        ctx.fillRect(px[i], py[i], sz, sz);
      }

      frames++; acc += dt;
      if (acc >= 0.5) {
        setFps(Math.round(frames / acc)); frames = 0; acc = 0;
        // Egg 3: ~1 in 220 ticks (~ every 2 min) the status resolves to something else for one beat.
        if (Math.random() < 0.0045) {
          setStatus(OTHER_STATUS[(Math.random() * OTHER_STATUS.length) | 0]);
          setTimeout(() => setStatus("UNDER CONSTRUCTION"), 520);
        }
      }
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      wrap.removeEventListener("pointermove", move);
      wrap.removeEventListener("pointerleave", leave);
      wrap.removeEventListener("touchmove", move);
    };
  }, []);

  return (
    <div className="demo-root" ref={wrapRef}>
      <style>{CSS}</style>
      <canvas className="demo-canvas" ref={canvasRef} />
      <div className="demo-sheen" />
      <div className="demo-scan" />
      <div className="demo-vig" />

      <div className="demo-plate">
        <p className="demo-kicker">differentiable matter</p>
        <h1 className="demo-title">NO DEMO<br /><span className="dim">EXISTS YET</span></h1>
        <div className="demo-rule" />
        <p className="demo-sub">
          The flagship is unbuilt. What moves behind this text is a placeholder field — real, integrated
          every frame, and <b>not the demo</b>. When there is something worth showing, it lives here.
        </p>
      </div>

      <div className="demo-hud">
        <span>PARTICLES <b>2600</b></span>
        <span>INTEGRATOR <b>SEMI-IMPLICIT</b></span>
        <span>FPS <b>{fps || "--"}</b></span>
        <span>STATUS <b>{status}</b></span>
        <span>WEBGPU <b>{gpu}</b></span>
      </div>
      <div className="demo-badge">v0 · PLACEHOLDER</div>
    </div>
  );
}
