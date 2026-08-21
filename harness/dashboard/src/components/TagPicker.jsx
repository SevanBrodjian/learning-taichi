import { useState } from "react";
import { createTag } from "../api.js";

/**
 * The one tag picker. Tags used to be a hard-coded array copied into three components, which meant a new
 * tag needed a code edit — so this reads the live registry (served as the union of coordination/tags.json
 * and every tag actually in use) and can CREATE one inline.
 *
 * `options` is [{name, color}], `value` is the selected names, `onChange` gets the next array.
 */
export default function TagPicker({ options = [], value = [], onChange, onTagCreated }) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const toggle = (name) =>
    onChange(value.includes(name) ? value.filter((x) => x !== name) : [...value, name]);

  const submit = async () => {
    const name = draft.trim().toLowerCase().replace(/\s+/g, "-");
    if (!name) return;
    setBusy(true); setErr("");
    try {
      const r = await createTag(name);
      if (r && r.ok) {
        if (!value.includes(r.id)) onChange([...value, r.id]);
        onTagCreated && onTagCreated();
        setDraft(""); setAdding(false);
      } else {
        setErr((r && r.error) || "could not create that tag");
      }
    } finally { setBusy(false); }
  };

  return (
    <div className="tagpick">
      {options.map((t) => (
        <button key={t.name} type="button"
                className={`tagpick-opt ${value.includes(t.name) ? "on" : ""}`}
                style={{ "--tc": t.color || "#7f8ea3" }}
                onClick={() => toggle(t.name)}>{t.name}</button>
      ))}
      {adding ? (
        <span className="tagpick-new">
          <input autoFocus value={draft} placeholder="new-tag" disabled={busy}
                 onChange={(e) => setDraft(e.target.value)}
                 onKeyDown={(e) => {
                   if (e.key === "Enter") { e.preventDefault(); submit(); }
                   if (e.key === "Escape") { setAdding(false); setDraft(""); setErr(""); }
                 }} />
          <button type="button" className="tagpick-ok" disabled={busy} onClick={submit}>add</button>
          <button type="button" className="tagpick-x"
                  onClick={() => { setAdding(false); setDraft(""); setErr(""); }}>×</button>
        </span>
      ) : (
        <button type="button" className="tagpick-add" onClick={() => setAdding(true)}>+ new tag</button>
      )}
      {err && <span className="tagpick-err">{err}</span>}
    </div>
  );
}
