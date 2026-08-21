import { useEffect, useState } from "react";
import { fetchText } from "../api.js";
import MarkdownReport from "./MarkdownReport.jsx";
import DocEditor from "./DocEditor.jsx";

// Generic single-markdown panel (Directions, Reports, a single inbox item). Editable in place —
// except where the document is a frozen copy (an epoch's report), which is `readOnly` by definition:
// the whole point of the copy is that it does not move.
export default function DocView({ url, empty, readOnly = false }) {
  const [body, setBody] = useState(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    if (!url) {
      setMissing(true);
      return;
    }
    let alive = true;
    setBody(null);
    setMissing(false);
    fetchText(url)
      .then((t) => alive && setBody(t))
      .catch(() => alive && setMissing(true));
    return () => {
      alive = false;
    };
  }, [url]);

  if (missing) return <div className="muted pad">{empty || "Nothing here yet."}</div>;
  if (body == null) return <div className="muted pad">Loading…</div>;
  if (readOnly) {
    return (
      <div className="content-doc">
        <MarkdownReport markdown={body} />
      </div>
    );
  }
  return (
    <div className="content-doc">
      <DocEditor url={url} body={body} onSaved={setBody}>
        <MarkdownReport markdown={body} />
      </DocEditor>
    </div>
  );
}
