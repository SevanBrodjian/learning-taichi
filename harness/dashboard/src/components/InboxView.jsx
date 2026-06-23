import { useEffect, useState } from "react";
import { fetchDecisions } from "../api.js";
import DocView from "./DocView.jsx";

// The I/O channel: decisions that need the user. Populated from coordination/decisions/*.md and
// announced via an ntfy `gate`. Empty is the healthy default.
export default function InboxView() {
  const [items, setItems] = useState(null);
  const [activeUrl, setActiveUrl] = useState(undefined);

  useEffect(() => {
    fetchDecisions()
      .then((d) => {
        const list = d.decisions || [];
        setItems(list);
        setActiveUrl(list[0]?.url ?? null);
      })
      .catch(() => {
        setItems([]);
        setActiveUrl(null);
      });
  }, []);

  if (items === null) return <div className="muted pad">Loading…</div>;
  if (items.length === 0) {
    return (
      <div className="muted pad">
        Inbox empty. Decisions that need you show up here, and ping your iPad via ntfy.
      </div>
    );
  }
  return (
    <div className="doc-layout">
      <nav className="toc">
        <div className="toc-group-title">Open decisions</div>
        {items.map((it) => (
          <button
            key={it.id}
            className={`toc-item ${it.url === activeUrl ? "active" : ""}`}
            onClick={() => setActiveUrl(it.url)}
          >
            {it.title}
          </button>
        ))}
      </nav>
      <article className="doc-body">
        <DocView url={activeUrl} />
      </article>
    </div>
  );
}
