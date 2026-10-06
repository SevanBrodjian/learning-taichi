// Public, read-only, static mode (`npm run build:public`).
//
// The same dashboard, built to browse a frozen snapshot written by harness/tools/export_public.py instead
// of talking to the live data server. In this mode:
//   - every GET resolves to a static file under `snapshot/` (relative to the page, so the bundle works
//     under any subpath),
//   - every write helper refuses locally without touching the network, and the controls that would call
//     them are not rendered,
//   - Notebook / Inbox are not in the navigation, and nothing polls.
// The flag is replaced at build time, so in the normal build every `PUBLIC &&` branch is dead code and
// the live dashboard behaves exactly as before.
import { useEffect, useState } from "react";

// `vite build --mode public` (npm run build:public) or VITE_PUBLIC=1. Both are replaced at build time.
export const PUBLIC = import.meta.env.MODE === "public" || import.meta.env.VITE_PUBLIC === "1";

// Relative on purpose: resolved against the page, so `/lab/x/index.html` reads `/lab/x/snapshot/...`.
export const SNAPSHOT_ROOT = "snapshot/";

// Written by the exporter next to the API files: the snapshot date and which opt-in views it carries.
let _meta = null;
let _metaPromise = null;
export function loadSnapshotMeta() {
  if (!PUBLIC) return Promise.resolve(null);
  if (_meta) return Promise.resolve(_meta);
  if (!_metaPromise) {
    _metaPromise = fetch(`${SNAPSHOT_ROOT}meta.json`)
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({}))
      .then((m) => (_meta = m || {}));
  }
  return _metaPromise;
}

export function useSnapshotMeta() {
  const [meta, setMeta] = useState(_meta);
  useEffect(() => {
    if (!PUBLIC || _meta) return;
    let alive = true;
    loadSnapshotMeta().then((m) => alive && setMeta(m));
    return () => { alive = false; };
  }, []);
  return meta;
}

// "6 Oct 2026" from "2026-10-06" without letting the viewer's timezone shift the day.
export function formatSnapshotDate(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso || "");
  if (!m) return "";
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}
