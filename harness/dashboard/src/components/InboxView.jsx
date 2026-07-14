import { useEffect, useState } from "react";
import { fetchDecisions, resolveDecision } from "../api.js";
import DocView from "./DocView.jsx";

// The I/O channel. Two clean parts: Decisions (things that need you, from coordination/decisions/ —
// including task CONTRACTS you Approve/Reject before a run spawns) and a live ntfy notification feed.
const DISMISS_KEY = "lt_dismissed_notifs";
const loadDismissed = () => {
  try { return new Set(JSON.parse(localStorage.getItem(DISMISS_KEY) || "[]")); } catch { return new Set(); }
};

function timeAgo(sec) {
  if (!sec) return "";
  const d = Date.now() / 1000 - sec;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

export default function InboxView({ notifData }) {
  const [decisions, setDecisions] = useState([]);
  const [activeId, setActiveId] = useState(undefined);
  const [dismissed, setDismissed] = useState(loadDismissed);
  const [rejecting, setRejecting] = useState(false);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(Date.now());
  const notifs = notifData;

  // Tick every second so a contract's "auto-runs in M:SS" countdown updates live.
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(id); }, []);

  const load = () =>
    fetchDecisions()
      .then((d) => {
        const it = d.decisions || [];
        setDecisions(it);
        setActiveId((cur) => (it.find((x) => x.id === cur) ? cur : it[0]?.id ?? null));
      })
      .catch(() => { setDecisions([]); setActiveId(null); });
  useEffect(() => { load(); }, []);

  const active = decisions.find((d) => d.id === activeId);

  const resolve = async (resolution) => {
    if (!active) return;
    setBusy(true);
    try {
      await resolveDecision(active.id, resolution, note.trim());
      setRejecting(false); setNote("");
      await load();
    } finally { setBusy(false); }
  };

  const dismiss = (id) => {
    const next = new Set(dismissed);
    next.add(id);
    setDismissed(next);
    localStorage.setItem(DISMISS_KEY, JSON.stringify([...next]));
  };
  const visible = (notifs?.notifications || []).filter((n) => !dismissed.has(n.id));

  return (
    <div className="inbox">
      <section className="inbox-block">
        <h2>Decisions</h2>
        {decisions.length === 0 ? (
          <div className="muted">Nothing needs you right now. Decisions an agent escalates show up here.</div>
        ) : (
          <div className="doc-layout">
            <nav className="toc">
              <div className="toc-group-title">Open</div>
              {decisions.map((it) => (
                <button key={it.id} className={`toc-item ${it.id === activeId ? "active" : ""}`}
                        onClick={() => { setActiveId(it.id); setRejecting(false); setNote(""); }}>
                  <span className="toc-item-title">{it.title}</span>
                  {it.kind === "contract" && !it.resolved && <span className="new-tag">contract</span>}
                  {it.resolved && <span className="done-tag">done</span>}
                </button>
              ))}
            </nav>
            <article className="doc-body">
              {active && active.kind === "contract" && !active.resolved && (
                <div className="contract-actions">
                  <span className="contract-label">Run this task?</span>
                  {active.auto_run_at && (() => {
                    const rem = Math.round(active.auto_run_at - now / 1000);
                    if (rem <= 0) return <span className="contract-countdown go">auto-running…</span>;
                    const m = Math.floor(rem / 60), s = rem % 60;
                    return <span className="contract-countdown">auto-runs in {m}:{String(s).padStart(2, "0")}</span>;
                  })()}
                  {rejecting ? (
                    <div className="contract-reject">
                      <textarea className="reopen-input" rows={2} value={note} placeholder="What to change (sent back to the queue)"
                                onChange={(e) => setNote(e.target.value)} />
                      <div className="reopen-actions">
                        <button className="act-btn" onClick={() => { setRejecting(false); setNote(""); }}>Cancel</button>
                        <button className="act-btn danger" disabled={busy} onClick={() => resolve("reject")}>Send back</button>
                      </div>
                    </div>
                  ) : (
                    <div className="contract-btns">
                      <button className="act-btn primary" disabled={busy} onClick={() => resolve("approve")}>Approve &amp; run</button>
                      <button className="act-btn" disabled={busy} onClick={() => setRejecting(true)}>Reject</button>
                    </div>
                  )}
                </div>
              )}
              <DocView url={active?.url} />
            </article>
          </div>
        )}
      </section>

      <section className="inbox-block">
        <h2>Notifications</h2>
        {notifs == null ? (
          <div className="muted">Loading…</div>
        ) : notifs.configured === false ? (
          <div className="muted">No ntfy topic configured.</div>
        ) : visible.length === 0 ? (
          <div className="muted">No recent notifications.</div>
        ) : (
          <div className="notif-feed">
            {visible.map((n) => (
              <div key={n.id} className={`notif p${n.priority}`}>
                <div className="notif-row">
                  <span className="notif-title">{n.title || "learning-taichi"}</span>
                  <span className="notif-time">{timeAgo(n.time)}</span>
                  <button className="notif-x" onClick={() => dismiss(n.id)} aria-label="dismiss">×</button>
                </div>
                {n.message && <div className="notif-msg">{n.message}</div>}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
