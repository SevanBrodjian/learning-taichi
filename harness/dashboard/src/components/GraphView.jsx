import { useEffect, useMemo, useRef, useState } from "react";

// The task graph — the project as a Zettelkasten. Nodes are tasks; directed edges are follow-up links
// (a task -> the tasks it followed up on). A research "direction" is just a connected region of this
// graph, and tags color/filter it. Self-contained SVG (no libs): layered layout (roots left, follow-ups
// flow right), pan + zoom, tag filtering, hover lineage highlight, click to open.

const NODE_W = 186;
const NODE_H = 44;
const COL_W = 250;
const ROW_H = 66;
const MARGIN = 48;

// status -> accent. has_artifact nodes get a glow ring.
const STATUS = {
  proposed: { fill: "#141b25", stroke: "#3a4658", text: "#9fb0c0", label: "Proposed" },
  queued:   { fill: "#0f2230", stroke: "#4cc2ff", text: "#bfe6ff", label: "Queued" },
  active:   { fill: "#11261a", stroke: "#7ee787", text: "#c7f5cf", label: "Active" },
  done:     { fill: "#12201a", stroke: "#2f6f4e", text: "#8fc7a8", label: "Done" },
};
const st = (s) => STATUS[s] || STATUS.proposed;

// A small stable palette for tag chips.
const TAG_COLORS = ["#4cc2ff", "#7ee787", "#ffb037", "#c98bff", "#ff7b9c", "#5ee0c8", "#e6a23c", "#8fa8ff"];

function parentsOf(t) {
  const f = t.follow_up_of;
  if (!f) return [];
  return Array.isArray(f) ? f : [f];
}

export default function GraphView({ overview, onOpenTask, onOpenRef }) {
  const dirs = overview?.directions || [];

  // A stable content signature so the 4s board poll (which hands us a fresh overview object every tick)
  // does not recompute the whole layout and make the graph flicker/jump. Recompute only when the task
  // set, links, statuses, or tags actually change.
  const sig = dirs.map((d) =>
    d.tasks.map((t) =>
      `${d.id}/${t.id}:${t.status}:${t.has_artifact ? 1 : 0}:${(Array.isArray(t.follow_up_of) ? t.follow_up_of.join(",") : t.follow_up_of) || ""}:${(t.tags || []).join(",")}`
    ).join("|")
  ).join("||");

  // ---- build nodes + edges, and a layered layout (stable across polls via `sig`) ----
  const { nodes, edges, tags, width, height } = useMemo(() => {
    const nodes = [];
    const byKey = new Map();
    for (const d of dirs) {
      for (const t of d.tasks) {
        const key = `${d.id}/${t.id}`;
        const tagList = [...new Set([...(t.tags || []), d.id])]; // direction id is an implicit tag
        const n = {
          key, id: t.id, direction: d.id, directionName: d.name, title: t.title,
          status: t.status, has: t.has_artifact, detail: t.detail, tags: tagList,
          parents: parentsOf(t).map((p) => `${d.id}/${p}`), // links are within a direction
        };
        nodes.push(n);
        byKey.set(key, n);
      }
    }
    // depth = longest chain of parents (roots at 0)
    const depthMemo = new Map();
    const depth = (n, seen = new Set()) => {
      if (depthMemo.has(n.key)) return depthMemo.get(n.key);
      if (seen.has(n.key)) return 0;
      seen.add(n.key);
      const ps = n.parents.map((k) => byKey.get(k)).filter(Boolean);
      const dv = ps.length ? 1 + Math.max(...ps.map((p) => depth(p, seen))) : 0;
      depthMemo.set(n.key, dv);
      return dv;
    };
    nodes.forEach((n) => (n.depth = depth(n)));

    // group by depth, order within a column by parents' barycenter (one pass) to reduce crossings
    const cols = new Map();
    nodes.forEach((n) => { if (!cols.has(n.depth)) cols.set(n.depth, []); cols.get(n.depth).push(n); });
    const sortedDepths = [...cols.keys()].sort((a, b) => a - b);
    const slotOf = new Map();
    for (const d of sortedDepths) {
      const col = cols.get(d);
      col.sort((a, b) => {
        const ba = a.parents.reduce((s, k) => s + (slotOf.get(k) ?? 0), 0) / (a.parents.length || 1);
        const bb = b.parents.reduce((s, k) => s + (slotOf.get(k) ?? 0), 0) / (b.parents.length || 1);
        if (ba !== bb) return ba - bb;
        return (a.directionName + a.title).localeCompare(b.directionName + b.title);
      });
      col.forEach((n, i) => {
        n.x = MARGIN + d * COL_W;
        n.y = MARGIN + i * ROW_H;
        slotOf.set(n.key, i);
      });
    }
    const edges = [];
    nodes.forEach((n) => n.parents.forEach((pk) => {
      const p = byKey.get(pk);
      if (p) edges.push({ from: p, to: n, key: `${pk}->${n.key}` });
    }));
    const tags = [...new Set(nodes.flatMap((n) => n.tags))].sort();
    const width = MARGIN * 2 + (Math.max(0, ...nodes.map((n) => n.depth)) + 1) * COL_W;
    const height = MARGIN * 2 + Math.max(1, ...sortedDepths.map((d) => cols.get(d).length)) * ROW_H;
    return { nodes, edges, tags, width, height };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  // ---- pan / zoom ----
  const wrapRef = useRef(null);
  const [view, setView] = useState({ x: 20, y: 20, k: 0.9 });
  const drag = useRef(null);
  const [activeTag, setActiveTag] = useState(null);
  const [hover, setHover] = useState(null);
  const [mounted, setMounted] = useState(false);
  useEffect(() => { const t = setTimeout(() => setMounted(true), 30); return () => clearTimeout(t); }, []);

  const fit = () => {
    const el = wrapRef.current;
    if (!el) return;
    const k = Math.min(1, Math.max(0.25, Math.min(el.clientWidth / width, el.clientHeight / height) * 0.95));
    setView({ x: (el.clientWidth - width * k) / 2, y: 24, k });
  };
  useEffect(() => { fit(); /* eslint-disable-next-line */ }, [width, height]);

  const onPointerDown = (e) => {
    if (e.target.closest(".gnode")) return; // let node handle its click
    drag.current = { sx: e.clientX, sy: e.clientY, ox: view.x, oy: view.y, moved: false };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.sx, dy = e.clientY - drag.current.sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
    setView((v) => ({ ...v, x: drag.current.ox + dx, y: drag.current.oy + dy }));
  };
  const onPointerUp = () => { drag.current = null; };
  const zoom = (factor, cx, cy) => setView((v) => {
    const k = Math.min(2.2, Math.max(0.2, v.k * factor));
    const el = wrapRef.current;
    const px = cx ?? (el ? el.clientWidth / 2 : 0), py = cy ?? (el ? el.clientHeight / 2 : 0);
    // keep the point under the cursor fixed
    return { k, x: px - (px - v.x) * (k / v.k), y: py - (py - v.y) * (k / v.k) };
  });
  const onWheel = (e) => { e.preventDefault(); zoom(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.nativeEvent.offsetX, e.nativeEvent.offsetY); };

  // lineage highlight: the hovered node + all its ancestors and descendants
  const lineage = useMemo(() => {
    if (!hover) return null;
    const set = new Set([hover]);
    const up = (k) => { const n = nodes.find((x) => x.key === k); n?.parents.forEach((p) => { if (!set.has(p)) { set.add(p); up(p); } }); };
    const down = (k) => edges.forEach((e) => { if (e.from.key === k && !set.has(e.to.key)) { set.add(e.to.key); down(e.to.key); } });
    up(hover); down(hover);
    return set;
  }, [hover, nodes, edges]);

  const dim = (n) => {
    if (activeTag && !n.tags.includes(activeTag)) return true;
    if (lineage && !lineage.has(n.key)) return true;
    return false;
  };
  const edgeDim = (e) => dim(e.from) || dim(e.to);

  const openNode = (n) => {
    if (drag.current?.moved) return;
    if (n.has) onOpenTask?.({ ...n, key: n.key });
    else onOpenRef?.(n.direction, n.id, false);
  };

  const tagColor = (t) => TAG_COLORS[[...tags].indexOf(t) % TAG_COLORS.length];

  if (!overview) return <div className="muted pad">Loading…</div>;
  if (nodes.length === 0) return <div className="muted pad">No tasks yet — the graph fills in as you add them.</div>;

  return (
    <div className="graphview">
      <div className="graph-toolbar">
        <span className="graph-title">Task map <span className="muted">· {nodes.length} tasks, {edges.length} links</span></span>
        <span className="graph-tags">
          {tags.map((t) => (
            <button key={t} className={`gtag ${activeTag === t ? "active" : ""}`}
                    style={{ "--tc": tagColor(t) }}
                    onClick={() => setActiveTag((a) => (a === t ? null : t))}>{t}</button>
          ))}
        </span>
        <span className="graph-zoom">
          <button className="act-btn" onClick={() => zoom(1 / 1.2)} aria-label="zoom out">−</button>
          <button className="act-btn" onClick={() => zoom(1.2)} aria-label="zoom in">+</button>
          <button className="act-btn" onClick={fit}>Fit</button>
        </span>
      </div>
      <div className="graph-canvas" ref={wrapRef}
           onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
           onPointerLeave={onPointerUp} onWheel={onWheel}>
        <svg className="graph-svg" width="100%" height="100%">
          <defs>
            <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L7,3 L0,6 Z" fill="#3a4658" />
            </marker>
            <marker id="arrowlit" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L7,3 L0,6 Z" fill="#4cc2ff" />
            </marker>
          </defs>
          <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}
             style={{ opacity: mounted ? 1 : 0, transition: "opacity .5s ease" }}>
            {edges.map((e) => {
              const x1 = e.from.x + NODE_W, y1 = e.from.y + NODE_H / 2;
              const x2 = e.to.x, y2 = e.to.y + NODE_H / 2;
              const mx = (x1 + x2) / 2;
              const lit = lineage && lineage.has(e.from.key) && lineage.has(e.to.key);
              return (
                <path key={e.key} className="gedge"
                      d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                      fill="none" stroke={lit ? "#4cc2ff" : "#2b3a4d"}
                      strokeWidth={lit ? 2 : 1.3} opacity={edgeDim(e) ? 0.12 : lit ? 0.95 : 0.5}
                      markerEnd={`url(#${lit ? "arrowlit" : "arrow"})`} />
              );
            })}
            {nodes.map((n, i) => {
              const c = st(n.status);
              const d = dim(n);
              const pt = n.tags[0];
              return (
                <g key={n.key} className="gnode"
                   transform={`translate(${n.x},${n.y})`}
                   style={{ cursor: "pointer", opacity: d ? 0.22 : 1,
                            transition: `opacity .3s ease, transform .4s cubic-bezier(.2,.8,.2,1)`,
                            transitionDelay: mounted ? "0s" : `${Math.min(i * 12, 400)}ms` }}
                   onMouseEnter={() => setHover(n.key)} onMouseLeave={() => setHover(null)}
                   onClick={() => openNode(n)}>
                  {n.has && <rect x={-3} y={-3} width={NODE_W + 6} height={NODE_H + 6} rx={9}
                                  fill="none" stroke={c.stroke} strokeWidth={1} opacity={0.35} />}
                  <rect width={NODE_W} height={NODE_H} rx={7} fill={c.fill} stroke={c.stroke}
                        strokeWidth={hover === n.key ? 2 : 1.3} />
                  <rect width={4} height={NODE_H} rx={2} fill={tagColor(pt)} opacity={0.9} />
                  <text x={13} y={18} className="gnode-title" fill={c.text}>{trunc(n.title, 26)}</text>
                  <text x={13} y={33} className="gnode-sub" fill="#6f7d8f">{n.directionName}</text>
                  {n.status === "active" && <circle cx={NODE_W - 12} cy={14} r={3.5} fill="#7ee787" className="gpulse" />}
                </g>
              );
            })}
          </g>
        </svg>
        {hover && (() => {
          const n = nodes.find((x) => x.key === hover);
          return n && n.title.length > 26 ? <div className="graph-tip" style={{ left: 16, bottom: 16 }}>{n.title}</div> : null;
        })()}
      </div>
      <div className="graph-legend">
        {Object.entries(STATUS).map(([k, val]) => (
          <span key={k} className="glegend"><span className="gdot" style={{ background: val.stroke }} />{val.label}</span>
        ))}
        <span className="glegend"><span className="gdot ring" />has result</span>
        <span className="muted">drag to pan · scroll or ± to zoom · click a task to open</span>
      </div>
    </div>
  );
}

function trunc(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
