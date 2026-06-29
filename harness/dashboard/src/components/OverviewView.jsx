import { useState } from "react";
import { setTaskStatus, createTask, editTask, createDirection } from "../api.js";

// The live pipeline. Done is a collapsed section below (not a crowding column). Cards are tasks,
// filterable by direction. Proposed and queued cards can be dragged between those two columns
// (the user's call). A proposed/queued card with no artifact opens a detail modal; a card with a
// result navigates to its task. Tasks and directions can also be authored here (+ Task / + Direction).
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
  const [author, setAuthor] = useState(null); // { mode: "task" | "direction" }
  const [editing, setEditing] = useState(false); // editing the open task modal
  const [form, setForm] = useState({});
  const [busy, setBusy] = useState(false);

  if (!overview) return <div className="muted pad">Loading…</div>;
  const dirs = overview.directions || [];

  const tasks = dirs
    .filter((d) => filter === "all" || d.id === filter)
    .flatMap((d) => d.tasks.map((t) => ({ ...t, direction: d.id, directionName: d.name })));

  const done = tasks.filter((t) => t.status === "done");
  const columns = COLUMNS.map((c) => ({ ...c, items: tasks.filter((t) => t.status === c.id) }));

  const openCard = (t) => (t.has_artifact ? onOpenTask(t) : setModal(t));
  const refresh = () => onChange && onChange();

  const drop = (colId) => {
    if (drag && DRAGGABLE.has(colId) && drag.status !== colId) {
      setTaskStatus(drag.direction, drag.id, colId).then(refresh);
    }
    setDrag(null);
    setOver(null);
  };

  // Tap-friendly path (drag-and-drop does not work on touch / iPad): move via the modal.
  const move = (t, status) => {
    setTaskStatus(t.direction, t.id, status).then(refresh);
    setModal(null);
  };

  const openAuthor = (mode) => {
    const dir = filter !== "all" ? filter : dirs[0]?.id || "";
    setForm({ direction: dir, title: "", note: "", name: "", summary: "" });
    setAuthor({ mode });
  };

  const submitAuthor = async () => {
    setBusy(true);
    try {
      if (author.mode === "task") {
        if (!form.direction || !form.title.trim()) return;
        await createTask(form.direction, form.title.trim(), form.note.trim(), "queued");
      } else {
        if (!form.name.trim()) return;
        await createDirection(form.name.trim(), form.summary.trim());
      }
      setAuthor(null);
      refresh();
    } finally {
      setBusy(false);
    }
  };

  const startEdit = (t) => {
    setForm({ title: t.title, note: t.note || "" });
    setEditing(true);
  };
  const saveEdit = async () => {
    setBusy(true);
    try {
      await editTask(modal.direction, modal.id, form.title.trim(), form.note);
      setEditing(false);
      setModal({ ...modal, title: form.title.trim(), note: form.note });
      refresh();
    } finally {
      setBusy(false);
    }
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
        <span className="dir-filter-spacer" />
        <button className="chip add" onClick={() => openAuthor("task")} disabled={dirs.length === 0}>+ Task</button>
        <button className="chip add" onClick={() => openAuthor("direction")}>+ Direction</button>
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
        <div className="modal-backdrop" onClick={() => { setModal(null); setEditing(false); }}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => { setModal(null); setEditing(false); }} aria-label="close">×</button>
            <div className="modal-dir">{modal.directionName}</div>
            {editing ? (
              <div className="author-form">
                <label>Title</label>
                <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
                <label>Note (the seed for the worker brief)</label>
                <textarea rows={5} value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} />
                <div className="modal-actions">
                  <button className="act-btn" onClick={() => setEditing(false)}>Cancel</button>
                  <button className="act-btn primary" onClick={saveEdit} disabled={busy || !form.title.trim()}>Save</button>
                </div>
              </div>
            ) : (
              <>
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
                  <button className="act-btn" onClick={() => startEdit(modal)}>Edit</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {author && (
        <div className="modal-backdrop" onClick={() => setAuthor(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setAuthor(null)} aria-label="close">×</button>
            {author.mode === "task" ? (
              <>
                <h3>New task</h3>
                <div className="author-form">
                  <label>Direction</label>
                  <select value={form.direction} onChange={(e) => setForm({ ...form, direction: e.target.value })}>
                    {dirs.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                  </select>
                  <label>Title</label>
                  <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} placeholder="What the task should accomplish" />
                  <label>Note (seed for the worker brief)</label>
                  <textarea rows={5} value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} placeholder="A sentence or two; the orchestrator expands this into a full brief." />
                  <div className="modal-actions">
                    <button className="act-btn" onClick={() => setAuthor(null)}>Cancel</button>
                    <button className="act-btn primary" onClick={submitAuthor} disabled={busy || !form.direction || !form.title.trim()}>Add to Queued</button>
                  </div>
                </div>
              </>
            ) : (
              <>
                <h3>New direction</h3>
                <div className="author-form">
                  <label>Name</label>
                  <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="e.g. Material variants" />
                  <label>Summary</label>
                  <textarea rows={4} value={form.summary} onChange={(e) => setForm({ ...form, summary: e.target.value })} placeholder="The organizing question for this research axis." />
                  <div className="modal-actions">
                    <button className="act-btn" onClick={() => setAuthor(null)}>Cancel</button>
                    <button className="act-btn primary" onClick={submitAuthor} disabled={busy || !form.name.trim()}>Create</button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
