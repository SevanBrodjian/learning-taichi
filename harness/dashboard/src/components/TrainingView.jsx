import { useEffect, useMemo, useState } from "react";
import { fetchTraining, fetchText } from "../api.js";
import MarkdownReport from "./MarkdownReport.jsx";

// The standalone textbook: a left TOC (prerequisites split from core) and the selected section.
// Cross-references (wiki-links) navigate within this view.
export default function TrainingView() {
  const [toc, setToc] = useState(null);
  const [err, setErr] = useState(null);
  const [activeId, setActiveId] = useState(null);
  const [body, setBody] = useState(null);

  useEffect(() => {
    fetchTraining().then(setToc).catch((e) => setErr(String(e)));
  }, []);

  const sections = useMemo(() => (toc?.groups || []).flatMap((g) => g.sections), [toc]);
  useEffect(() => {
    if (sections.length && !sections.find((s) => s.id === activeId)) setActiveId(sections[0].id);
  }, [sections]); // eslint-disable-line react-hooks/exhaustive-deps

  const active = sections.find((s) => s.id === activeId);
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
                {s.title}
              </button>
            ))}
          </div>
        ))}
      </nav>
      <article className="doc-body">
        <MarkdownReport markdown={body} sections={sections} onNavigate={setActiveId} />
      </article>
    </div>
  );
}
