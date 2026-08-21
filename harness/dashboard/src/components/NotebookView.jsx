import { useEffect, useState } from "react";
import { fetchNotebook, fetchText, uploadImage } from "../api.js";
import MarkdownReport from "./MarkdownReport.jsx";
import DocEditor from "./DocEditor.jsx";

/**
 * The Notebook — Sevan's thinking space (reports/notebook/README.md).
 *
 * It is one living markdown file, `reports/notebook/current.md`, hand-written by him. The board records
 * what was DONE and the textbook records what is KNOWN; neither records what he is THINKING, which is
 * the part that decides what gets built next and the part that does not survive anywhere else.
 *
 * Three things this page is for, in order:
 *  1. NEVER LOSE THE WRITING. It autosaves ~1.5 s after typing stops, on leaving edit mode, and when the
 *     tab is hidden (the iPad PWA reloads from scratch when it is resumed, so "hidden" is the last
 *     chance to write). Every keystroke is also mirrored into localStorage, so a kill between autosaves
 *     is still recoverable, and the recovered draft is offered back on the next load.
 *  2. GET A SKETCH IN WITHOUT LEAVING. Paste or drop an image into the editor; it is filed in
 *     reports/notebook/media/ and referenced at the cursor. /api/file is markdown-only, so the bytes go
 *     through /api/upload, base64 over JSON, written with write_bytes.
 *  3. BE COMFORTABLE TO WRITE IN. One column at a readable measure, the same type, size, leading and
 *     position in both states — switching between reading and writing moves the text as little as
 *     possible, so it does not feel like changing tools.
 *
 * The agent reads this document and never edits it (that rule lives in the README, and nothing here
 * writes to the file except the person typing in it).
 */
export default function NotebookView() {
  const [info, setInfo] = useState(null);
  const [body, setBody] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    fetchNotebook()
      .then((d) => {
        if (!alive) return;
        setInfo(d);
        return fetchText(d.url)
          .then((t) => alive && setBody(t))
          .catch(() => alive && setBody("")); // not created yet: an empty page is still writable
      })
      .catch((e) => alive && setError(String(e.message || e)));
    return () => { alive = false; };
  }, []);

  const upload = (filename, dataB64) =>
    uploadImage(info.rid, info.media_dir, filename, dataB64);

  if (error) return <div className="error">{error}</div>;
  if (!info || body == null) return <div className="muted pad">Loading…</div>;

  return (
    <div className="notebook">
      <header className="nb-head">
        <h1>Notebook</h1>
        <p className="nb-sub">
          Yours. Overwrite anything, delete anything — git keeps the history.
          <span className="sep">·</span>
          <code>{info.path}</code>
        </p>
      </header>
      <div className="nb-sheet">
        <DocEditor
          url={info.url}
          body={body}
          onSaved={setBody}
          autosave
          backupKey="lt_notebook_backup"
          onUploadImage={upload}
          openLabel="Write"
          editLabel="Writing"
          variant="nb"
        >
          {body.trim()
            ? <MarkdownReport markdown={body} baseUrl={info.base_url} />
            : <p className="nb-empty">Nothing written yet.</p>}
        </DocEditor>
      </div>
    </div>
  );
}
