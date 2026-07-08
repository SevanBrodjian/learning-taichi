import { useEffect, useMemo, useState } from "react";
import { fetchTraining, fetchText } from "../api.js";
import MarkdownReport from "./MarkdownReport.jsx";
import DocEditor from "./DocEditor.jsx";

// Per-device read tracking for the "New" tag (#10). A section is New if never opened on this device,
// or edited (mtime advanced) since it was last opened. Opening it clears the tag.
const READ_KEY = "lt_training_read";
const ACTIVE_KEY = "lt_training_active";
const loadRead = () => {
  try { return JSON.parse(localStorage.getItem(READ_KEY) || "{}"); } catch { return {}; }
};

// The standalone textbook: a left TOC (prerequisites split from core) and the selected section.
// Cross-references (wiki-links) navigate within this view. `onRead` lets the parent recompute the
// nav's "New" badge the moment a section is opened (its read state changes here).
export default function TrainingView({ onRead }) {
  const [toc, setToc] = useState(null);
  const [err, setErr] = useState(null);
  const [activeId, setActiveId] = useState(() => localStorage.getItem(ACTIVE_KEY) || null);
  const [body, setBody] = useState(null);
  const [readMap, setReadMap] = useState(loadRead);

  const loadToc = () => fetchTraining().then(setToc).catch((e) => setErr(String(e)));
  useEffect(() => { loadToc(); }, []);

  const sections = useMemo(() => (toc?.groups || []).flatMap((g) => g.sections), [toc]);
  useEffect(() => {
    if (sections.length && !sections.find((s) => s.id === activeId)) setActiveId(sections[0].id);
  }, [sections]); // eslint-disable-line react-hooks/exhaustive-deps

  const active = sections.find((s) => s.id === activeId);

  // Persist place + clear the New tag when a section is opened.
  useEffect(() => {
    if (!active) return;
    localStorage.setItem(ACTIVE_KEY, active.id);
    if (active.mtime) {
      setReadMap((prev) => {
        if (prev[active.id] === active.mtime) return prev;
        const next = { ...prev, [active.id]: active.mtime };
        localStorage.setItem(READ_KEY, JSON.stringify(next));
        return next;
      });
    }
  }, [active?.id, active?.mtime]);

  // Tell the parent whenever read state changes, so the nav's "New" badge recomputes immediately.
  useEffect(() => { onRead && onRead(); }, [readMap]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!active) {
      setBody(null);
      return;
    }
    let alive = true;
    setBody(null);
    fetchText(active.url)
      .then((t) => alive && setBody(t))
      .catch(() => alive && setBody("*Section not written yet.*"));
    return () => {
      alive = false;
    };
  }, [active?.url]); // eslint-disable-line react-hooks/exhaustive-deps

  const isNew = (s) => s.mtime && (!(s.id in readMap) || s.mtime > readMap[s.id]);

  if (err) return <div className="error">{err}</div>;
  if (!toc) return <div className="muted pad">Loading textbook…</div>;
  if (sections.length === 0) return <div className="muted pad">No training sections yet.</div>;

  return (
    <div className="doc-layout">
      <nav className="toc">
        {toc.title && <div className="toc-title">{toc.title}</div>}
        {(toc.groups || []).map((g) => (
          <div className="toc-group" key={g.id}>
            <div className="toc-group-title">{g.title}</div>
            {g.sections.map((s) => (
              <button
                key={s.id}
                className={`toc-item ${s.id === activeId ? "active" : ""}`}
                onClick={() => setActiveId(s.id)}
              >
                <span className="toc-item-title">{s.title}</span>
                {isNew(s) && s.id !== activeId && <span className="new-tag">New</span>}
              </button>
            ))}
          </div>
        ))}
      </nav>
      <article className="doc-body">
        <DocEditor url={active?.url} body={body} onSaved={(t) => { setBody(t); loadToc(); }}>
          <MarkdownReport markdown={body} sections={sections} onNavigate={setActiveId} />
        </DocEditor>
      </article>
    </div>
  );
}
