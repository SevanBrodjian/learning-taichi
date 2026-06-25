import { useEffect, useState } from "react";
import { fetchDecisions, fetchNotifications } from "../api.js";
import DocView from "./DocView.jsx";

// The I/O channel. Two clean parts: Decisions (things that need you, from coordination/decisions/) and
// a Notifications feed pulled live from ntfy so you do not have to switch apps.
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

export default function InboxView() {
  const [decisions, setDecisions] = useState([]);
  const [activeUrl, setActiveUrl] = useState(undefined);
  const [notifs, setNotifs] = useState(null);
  const [dismissed, setDismissed] = useState(loadDismissed);

  useEffect(() => {
    fetchDecisions()
      .then((d) => { const it = d.decisions || []; setDecisions(it); setActiveUrl(it[0]?.url ?? null); })
      .catch(() => { setDecisions([]); setActiveUrl(null); });
    fetchNotifications().then(setNotifs).catch(() => setNotifs({ notifications: [] }));
  }, []);

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
                <button key={it.id} className={`toc-item ${it.url === activeUrl ? "active" : ""}`} onClick={() => setActiveUrl(it.url)}>
                  {it.title}
                </button>
              ))}
            </nav>
            <article className="doc-body"><DocView url={activeUrl} /></article>
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
