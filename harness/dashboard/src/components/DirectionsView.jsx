import { useEffect, useState } from "react";
import { fetchDirections } from "../api.js";
import DocView from "./DocView.jsx";

// The research-directions backlog (queued vs proposed) — the roadmap, read-only for now.
export default function DirectionsView() {
  const [url, setUrl] = useState(undefined);
  useEffect(() => {
    fetchDirections()
      .then((d) => setUrl(d.url))
      .catch(() => setUrl(null));
  }, []);
  if (url === undefined) return <div className="muted pad">Loading…</div>;
  return <DocView url={url} empty="No research_directions.md found." />;
}
