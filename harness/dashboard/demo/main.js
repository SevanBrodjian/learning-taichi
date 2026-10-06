// Standalone entry for the WebGPU MPM demo: the same bundle the dashboard's Demo tab mounts
// (src/components/mpm/, see DemoView.jsx), on a page of its own that fills its frame, so it can be
// embedded with an <iframe>. No dashboard, no data server, no network. The demo's own fallback panels
// (no WebGPU / insecure context / no GPU device) are part of demo4.js and show here unchanged.
//
// Build: `npm run build:demo` -> dist-demo/index.html, a single self-contained file.
import MPMDemo4 from "../src/components/mpm/demo4.js";
import DEMO_CSS from "../src/components/mpm/demo4.css.js";

const style = document.createElement("style");
style.textContent = DEMO_CSS;
document.head.appendChild(style);

const host = document.getElementById("demo");
let handle = null;
try {
  handle = MPMDemo4.mount(host, {});
  if (handle && handle.ready && handle.ready.catch) handle.ready.catch(() => {});
} catch (e) {
  host.textContent = "The demo failed to start: " + String((e && e.message) || e);
}
window.addEventListener("pagehide", () => {
  if (handle && handle.stop) {
    try { handle.stop(); } catch (e) { /* already torn down */ }
  }
});
