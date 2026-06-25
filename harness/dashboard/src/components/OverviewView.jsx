import { useState } from "react";
import { setTaskStatus } from "../api.js";

// The live pipeline. Done is a collapsed section below (not a crowding column). Cards are tasks,
// filterable by direction. Proposed and queued cards can be dragged between those two columns
// (the user's call). A proposed/queued card with no artifact opens a detail modal; a card with a
// result navigates to its task.
const COLUMNS = [
  { id: "proposed", label: "Proposed" },
  { id: "queued", label: "Queued" },
  { id: "active", label: "Active" },
];
const DRAGGABLE = new Set(["proposed", "queued"]);

export default function OverviewView({ overview, onOpenTask, onChange }) {
  const [filter, setFilter] = useState("all");
  const [showDone, setShowDone] = useState(false);
  const [modal, setModal] = useState(null);
  const [drag, setDrag] = useState(null);
  const [over, setOver] = useState(null);

  if (!overview) return <div className="muted pad">Loading…</div>;
  const dirs = overview.directions || [];

  const tasks = dirs
    .filter((d) => filter === "all" || d.id === filter)
    .flatMap((d) => d.tasks.map((t) => ({ ...t, direction: d.id, directionName: d.name })));

  const done = tasks.filter((t) => t.status === "done");
  const columns = COLUMNS.map((c) => ({ ...c, items: tasks.filter((t) => t.status === c.id) }));

  const openCard = (t) => (t.has_artifact ? onOpenTask(t) : setModal(t));

  const drop = (colId) => {
    if (drag && DRAGGABLE.has(colId) && drag.status !== colId) {
      setTaskStatus(drag.direction, drag.id, colId).then(() => onChange && onChange());
    }
    setDrag(null);
    setOver(null);
  };

  // Tap-friendly path (drag-and-drop does not work on touch / iPad): move via the modal.
  const move = (t, status) => {
    setTaskStatus(t.direction, t.id, status).then(() => onChange && onChange());
    setModal(null);
  };

  return (
    <div className="overview">
      <div className="dir-filter">
        <button className={`chip ${filter === "all" ? "active" : ""}`} onClick={() => setFilter("all")}>All</button>
        {dirs.map((d) => (
          <button key={d.id} className={`chip ${filter === d.id ? "active" : ""}`} title={d.summary} onClick={() => setFilter(d.id)}>
            {d.name}
          </button>
        ))}
      </div>

      <div className="board3">
        {columns.map((c) => (
          <div
            className={`board-col col-${c.id} ${over === c.id && DRAGGABLE.has(c.id) ? "drop-ok" : ""}`}
            key={c.id}
            onDragOver={(e) => { if (drag && DRAGGABLE.has(c.id)) { e.preventDefault(); setOver(c.id); } }}
            onDragLeave={() => setOver((o) => (o === c.id ? null : o))}
            onDrop={() => drop(c.id)}
          >
            <div className="board-col-head">{c.label} <span className="board-count">{c.items.length}</span></div>
            {c.items.map((t) => (
              <button
                key={t.direction + "/" + t.id}
                className={`task-card ${t.has_artifact ? "has" : ""} ${DRAGGABLE.has(t.status) ? "drag" : ""}`}
                draggable={DRAGGABLE.has(t.status)}
                onDragStart={() => setDrag({ direction: t.direction, id: t.id, status: t.status })}
                onDragEnd={() => { setDrag(null); setOver(null); }}
                onClick={() => openCard(t)}
              >
                <div className="task-card-title">{t.title}</div>
                <div className="task-card-dir">{t.directionName}</div>
                <div className="task-card-open">{t.has_artifact ? "view result →" : "details →"}</div>
              </button>
            ))}
            {c.items.length === 0 && <div className="muted dir-empty">none</div>}
          </div>
        ))}
      </div>

      <div className="done-section">
        <button className="done-toggle" onClick={() => setShowDone((s) => !s)}>
          {showDone ? "▾" : "▸"} Done ({done.length})
        </button>
        {showDone && (
          <div className="done-list">
            {done.map((t) => (
              <button key={t.direction + "/" + t.id} className="done-item" onClick={() => onOpenTask(t)}>
                <span className="task-card-title">{t.title}</span>
                <span className="task-card-dir">{t.directionName}</span>
              </button>
            ))}
            {done.length === 0 && <div className="muted dir-empty">nothing done yet</div>}
          </div>
        )}
      </div>

      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setModal(null)} aria-label="close">×</button>
            <div className="modal-dir">{modal.directionName}</div>
            <h3>{modal.title}</h3>
            <p className="modal-note">{modal.note || "No description yet."}</p>
            <div className="modal-status">Status: {modal.status}</div>
            <div className="modal-actions">
              {modal.status === "proposed" && (
                <button className="act-btn primary" onClick={() => move(modal, "queued")}>Queue</button>
              )}
              {modal.status === "queued" && (
                <button className="act-btn" onClick={() => move(modal, "proposed")}>Move to Proposed</button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
