import { useCallback, useEffect, useRef, useState } from "react";
import { beaconSave, saveFile } from "../api.js";

// Wraps any rendered markdown doc with an Edit affordance. In view mode it floats an "Edit" button
// over `children` (the rendered MarkdownReport); in edit mode it swaps to a raw-markdown textarea with
// Save / Cancel and writes straight back to the file on disk. `onSaved(text)` lets the parent refresh
// its rendered body (and, for training, its read/New state).
//
// The Notebook (a writing surface rather than a document that gets corrected) turns on three optional
// behaviours. They are options on THIS editor rather than a second editor, so there is one place where
// dashboard markdown gets written back:
//   autosave       save ~1.5 s after typing stops, on leaving edit mode, and on the tab being hidden.
//                  Autosaves do not commit; the deliberate save on exit does (see write_file).
//   backupKey      mirror every keystroke into localStorage, so a crash/kill between autosaves still
//                  leaves the text recoverable. Losing the writing is the unacceptable failure.
//   onUploadImage  paste or drop an image into the textarea -> uploaded, reference inserted at cursor.
const AUTOSAVE_MS = 1500;

const EXT_FOR = {
  "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp",
  "image/avif": ".avif", "image/heic": ".heic", "image/heif": ".heic", "image/bmp": ".bmp",
};

// FileReader gives "data:<type>;base64,<payload>"; the server wants just the payload.
const toBase64 = (file) =>
  new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result).split(",")[1] || "");
    fr.onerror = () => reject(new Error("could not read the image"));
    fr.readAsDataURL(file);
  });

const clockOf = (ts) =>
  new Date(ts).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

export default function DocEditor({
  url, body, onSaved, children,
  autosave = false, backupKey = null, onUploadImage = null,
  openLabel = "Edit", editLabel = "Editing markdown source", variant = "",
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(body ?? "");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState(null);
  const [state, setState] = useState("clean");   // clean | dirty | saving | saved | error
  const [savedAt, setSavedAt] = useState(null);
  const [backup, setBackup] = useState(null);    // recovered localStorage draft, if it differs
  const [dropping, setDropping] = useState(false);
  const [uploading, setUploading] = useState(false);
  const areaRef = useRef(null);
  const timerRef = useRef(null);
  const draftRef = useRef(draft);                // latest text, readable from unload handlers
  const onDiskRef = useRef(body ?? "");          // last text we know reached the file
  const caretRef = useRef(null);

  useEffect(() => {
    setDraft(body ?? "");
    draftRef.current = body ?? "";
    onDiskRef.current = body ?? "";
    setEditing(false);
    setErr(null);
    setState("clean");
  }, [url]); // a new doc resets the editor

  const canEdit = url && /\.md(\?|$)/.test(url) && body != null;

  // ---- crash backup -------------------------------------------------------------------------------
  const readBackup = useCallback(() => {
    if (!backupKey) return null;
    try {
      const b = JSON.parse(localStorage.getItem(backupKey) || "null");
      return b && b.url === url && typeof b.text === "string" ? b : null;
    } catch { return null; }
  }, [backupKey, url]);

  const clearBackup = useCallback(() => {
    if (backupKey) { try { localStorage.removeItem(backupKey); } catch { /* private mode */ } }
    setBackup(null);
  }, [backupKey]);

  // Checked ONCE per document, the moment its text arrives — that is the recovery case (a reload after
  // a crash, a kill, or the PWA being evicted). Re-checking on every keystroke would pop the banner
  // during normal typing, since the backup is legitimately ahead of the file between autosaves.
  const checkedRef = useRef(null);
  useEffect(() => {
    if (!backupKey || body == null || checkedRef.current === url) return;
    checkedRef.current = url;
    const b = readBackup();
    setBackup(b && b.text !== body ? b : null);
    if (b && b.text === body) clearBackup();
  }, [backupKey, body, url, readBackup, clearBackup]);

  // ---- saving -------------------------------------------------------------------------------------
  const doSave = useCallback(async (commit) => {
    const text = draftRef.current;
    if (text === onDiskRef.current && !commit) return true;
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    setSaving(true);
    setState("saving");
    setErr(null);
    try {
      const r = await saveFile(url, text, commit);
      if (!r || !r.ok) throw new Error((r && r.error) || "save failed");
      onDiskRef.current = text;
      setState("saved");
      setSavedAt(Date.now());
      clearBackup();
      onSaved?.(text);
      return true;
    } catch (e) {
      setErr(String(e.message || e));
      setState("error");
      return false;
    } finally {
      setSaving(false);
    }
  }, [url, onSaved, clearBackup]);

  const change = (text) => {
    setDraft(text);
    draftRef.current = text;
    if (!autosave) return;
    setState(text === onDiskRef.current ? "saved" : "dirty");
    if (backupKey) {
      try { localStorage.setItem(backupKey, JSON.stringify({ url, text, ts: Date.now() })); }
      catch { /* quota / private mode: the autosave below is the real protection */ }
    }
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => doSave(false), AUTOSAVE_MS);
  };

  // The iPad PWA reloads from scratch when it comes back from the background, so "hidden" is the last
  // moment a pending edit can be written. Try a real save, and leave a beacon in case we are unloading.
  useEffect(() => {
    if (!autosave) return;
    const flush = () => {
      if (draftRef.current === onDiskRef.current) return;
      if (document.visibilityState === "hidden") beaconSave(url, draftRef.current);
      doSave(false);
    };
    const onHide = () => { if (document.visibilityState === "hidden") flush(); };
    const onUnload = () => {
      if (draftRef.current !== onDiskRef.current) beaconSave(url, draftRef.current);
    };
    document.addEventListener("visibilitychange", onHide);
    window.addEventListener("pagehide", onUnload);
    window.addEventListener("beforeunload", onUnload);
    return () => {
      document.removeEventListener("visibilitychange", onHide);
      window.removeEventListener("pagehide", onUnload);
      window.removeEventListener("beforeunload", onUnload);
      if (timerRef.current) clearTimeout(timerRef.current);
      if (draftRef.current !== onDiskRef.current) beaconSave(url, draftRef.current);
    };
  }, [autosave, url, doSave]);

  useEffect(() => {
    if (editing && areaRef.current) areaRef.current.focus();
  }, [editing]);

  // Restore the caret after an image reference is spliced in, so typing continues where it left off.
  useEffect(() => {
    if (caretRef.current == null || !areaRef.current) return;
    const pos = caretRef.current;
    caretRef.current = null;
    areaRef.current.focus();
    areaRef.current.setSelectionRange(pos, pos);
  }, [draft]);

  // ---- images -------------------------------------------------------------------------------------
  const insertAtCursor = (snippet) => {
    const el = areaRef.current;
    const text = draftRef.current;
    const at = el ? el.selectionStart : text.length;
    const end = el ? el.selectionEnd : text.length;
    const before = text.slice(0, at);
    const after = text.slice(end);
    // Keep the reference on its own line without inventing blank lines the writer did not ask for.
    const lead = before === "" || before.endsWith("\n") ? "" : "\n";
    const block = `${lead}${snippet}\n`;
    change(before + block + after);
    caretRef.current = before.length + block.length;
  };

  const takeFiles = async (files) => {
    if (!onUploadImage || !files || !files.length) return false;
    const images = [...files].filter((f) => f.type && f.type.startsWith("image/"));
    if (!images.length) return false;
    setUploading(true);
    setErr(null);
    try {
      for (const f of images) {
        const ext = EXT_FOR[f.type] || ".png";
        const name = /\.[a-z0-9]+$/i.test(f.name || "") ? f.name : `pasted${ext}`;
        const b64 = await toBase64(f);
        const r = await onUploadImage(name, b64);
        if (!r || !r.ok) throw new Error((r && r.error) || "upload failed");
        insertAtCursor(r.markdown);
      }
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setUploading(false);
    }
    return true;
  };

  const onPaste = (e) => {
    if (!onUploadImage) return;
    const files = e.clipboardData?.files;
    if (files && files.length && [...files].some((f) => f.type?.startsWith("image/"))) {
      e.preventDefault();
      takeFiles(files);
    }
  };

  const onDrop = (e) => {
    if (!onUploadImage) return;
    const files = e.dataTransfer?.files;
    if (files && files.length) { e.preventDefault(); setDropping(false); takeFiles(files); }
  };

  const onKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      doSave(true);
    }
  };

  // ---- save-state readout (the "is my writing safe" signal) ---------------------------------------
  const statusEl = !autosave ? null : (
    <span className={`save-state save-${state}`}>
      {state === "saving" && "Saving…"}
      {state === "dirty" && "Unsaved…"}
      {state === "error" && "Not saved"}
      {state === "saved" && `Saved ${savedAt ? clockOf(savedAt) : ""}`}
      {state === "clean" && "Saved"}
    </span>
  );

  const backupBanner = backup && (
    <div className="doc-backup">
      <span>
        An unsaved draft from {clockOf(backup.ts)} is still in this browser and differs from the file.
      </span>
      <span className="doc-backup-actions">
        <button
          className="btn"
          onClick={() => { setDraft(backup.text); draftRef.current = backup.text; setState("dirty"); setEditing(true); setBackup(null); }}
        >
          Restore it
        </button>
        <button className="btn ghost" onClick={clearBackup}>Discard</button>
      </span>
    </div>
  );

  if (editing) {
    return (
      <div className={`doc-editor ${variant}`}>
        {backupBanner}
        <div className="doc-editor-bar">
          <span className="doc-editor-label">{editLabel}</span>
          <div className="doc-editor-actions">
            {uploading && <span className="muted small">Uploading image…</span>}
            {err && <span className="error-inline">{err}</span>}
            {statusEl}
            {!autosave && (
              <button className="btn ghost" onClick={() => { setEditing(false); setDraft(body ?? ""); }}>
                Cancel
              </button>
            )}
            <button
              className="btn primary"
              onClick={async () => { const ok = await doSave(true); if (ok) setEditing(false); }}
              disabled={saving}
            >
              {saving ? "Saving…" : autosave ? "Done" : "Save"}
            </button>
          </div>
        </div>
        <textarea
          ref={areaRef}
          className={`doc-editor-area${dropping ? " dropping" : ""}`}
          value={draft}
          spellCheck={autosave ? true : false}
          onChange={(e) => change(e.target.value)}
          onPaste={onPaste}
          onKeyDown={onKeyDown}
          onDragOver={onUploadImage ? (e) => { e.preventDefault(); setDropping(true); } : undefined}
          onDragLeave={onUploadImage ? () => setDropping(false) : undefined}
          onDrop={onUploadImage ? onDrop : undefined}
        />
        {onUploadImage && (
          <div className="doc-editor-hint">
            Paste or drop an image to file it in <code>media/</code> and reference it here.
            <span className="sep">·</span> ⌘/Ctrl+S saves now.
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`doc-editable ${variant}`}>
      {backupBanner}
      {canEdit && (
        <div className="doc-edit-affordance">
          {statusEl}
          <button className="doc-edit-btn" onClick={() => { setDraft(body ?? ""); draftRef.current = body ?? ""; setEditing(true); }}>
            {openLabel}
          </button>
        </div>
      )}
      {children}
    </div>
  );
}
