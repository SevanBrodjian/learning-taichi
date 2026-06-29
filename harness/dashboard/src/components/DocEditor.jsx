import { useEffect, useRef, useState } from "react";
import { saveFile } from "../api.js";

// Wraps any rendered markdown doc with an Edit affordance. In view mode it floats an "Edit" button
// over `children` (the rendered MarkdownReport); in edit mode it swaps to a raw-markdown textarea with
// Save / Cancel and writes straight back to the file on disk. `onSaved(text)` lets the parent refresh
// its rendered body (and, for training, its read/New state).
export default function DocEditor({ url, body, onSaved, children }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(body ?? "");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);
  const areaRef = useRef(null);

  useEffect(() => {
    setDraft(body ?? "");
    setEditing(false);
    setErr(null);
  }, [url]); // a new doc resets the editor

  useEffect(() => {
    if (editing && areaRef.current) areaRef.current.focus();
  }, [editing]);

  const canEdit = url && /\.md(\?|$)/.test(url) && body != null;

  async function save() {
    setSaving(true);
    setErr(null);
    try {
      const r = await saveFile(url, draft);
      if (!r || !r.ok) throw new Error((r && r.error) || "save failed");
      setEditing(false);
      onSaved?.(draft);
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setSaving(false);
    }
  }

  if (editing) {
    return (
      <div className="doc-editor">
        <div className="doc-editor-bar">
          <span className="doc-editor-label">Editing markdown source</span>
          <div className="doc-editor-actions">
            {err && <span className="error-inline">{err}</span>}
            <button className="btn ghost" onClick={() => { setEditing(false); setDraft(body ?? ""); }}>
              Cancel
            </button>
            <button className="btn primary" onClick={save} disabled={saving}>
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </div>
        <textarea
          ref={areaRef}
          className="doc-editor-area"
          value={draft}
          spellCheck={false}
          onChange={(e) => setDraft(e.target.value)}
        />
      </div>
    );
  }

  return (
    <div className="doc-editable">
      {canEdit && (
        <button className="doc-edit-btn" onClick={() => { setDraft(body ?? ""); setEditing(true); }}>
          Edit
        </button>
      )}
      {children}
    </div>
  );
}
