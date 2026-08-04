import { useEffect, useMemo, useRef, useState } from "react";

/**
 * The Map — the research lineage, not a hairball.
 *
 * Rebuilt for B3. What was wrong with the old one (all visible in one screenshot): twelve orphan tasks
 * stacked in a dead left column, 11 links for 21 tasks, long edges routed straight THROUGH other nodes,
 * titles truncated to unreadability, half the canvas empty, and "tags" that were really direction names
 * and filtered nothing useful.
 *
 * What changed:
 *  - EDGES HAVE KINDS (extends / re-does / refutes / applies / prerequisite-of), drawn differently and
 *    labelled on hover. A follow-up that overturned its parent no longer looks like one that built on it.
 *  - ORTHOGONAL ROUTING with per-edge lanes, so an edge spanning several columns travels in the gutter
 *    between them instead of crossing node bodies.
 *  - LANE-PACKED LAYOUT: nodes are placed by depth, then packed so a column only uses the rows it needs;
 *    the canvas is sized to content instead of leaving two-thirds empty.
 *  - WIDER NODES + two-line titles, so a task is readable without hovering.
 *  - TAGS FILTER FOR REAL, and the four canonical tags carry fixed colours.
 *  - A STORY RAIL: hovering any node lights its full ancestry and descent, and the header names the
 *    lineage you are looking at.
 *
 * CRASH FIX (Sevan: "it crashes the whole PWA when I scroll on that page sometimes"):
 *  - onPointerMove read `drag.current.ox` INSIDE a setState updater. React can run that updater after
 *    pointerup/pointerleave has already set `drag.current = null`, throwing a TypeError from inside
 *    render — which on an iPad PWA takes the whole web view down. The drag origin is now captured into
 *    locals before the updater, and the updater touches no refs.
 *  - setPointerCapture is wrapped: it throws if the pointer was already released.
 *  - The wheel handler is attached natively with {passive:false} instead of via React's onWheel, which
 *    is registered passive — so preventDefault() actually works instead of throwing and letting the
 *    gesture fall through to the shell.
 *  - Wheel zoom is rAF-coalesced, so a trackpad/touch flick cannot queue hundreds of re-renders.
 */

const NODE_W = 210;
const NODE_H = 52;
const COL_W = 290;          // node + gutter; the gutter is where edges are allowed to travel
const ROW_H = 74;
const MARGIN = 56;
const GUTTER = COL_W - NODE_W;

const STATUS = {
  proposed: { fill: "#141b25", stroke: "#3a4658", text: "#9fb0c0", label: "Proposed" },
  queued:   { fill: "#0f2230", stroke: "#4cc2ff", text: "#bfe6ff", label: "Queued" },
  active:   { fill: "#11261a", stroke: "#7ee787", text: "#c7f5cf", label: "Active" },
  done:     { fill: "#12201a", stroke: "#2f6f4e", text: "#8fc7a8", label: "Done" },
};
const st = (s) => STATUS[s] || STATUS.proposed;

// The four canonical tags (spec: gradients / materials / learned / rendering) get stable colours.
const TAG_COLORS = {
  gradients: "#4cc2ff", materials: "#ffb037", learned: "#c98bff", rendering: "#5ee0c8",
};
const FALLBACK = ["#7ee787", "#ff7b9c", "#8fa8ff", "#e6a23c"];

// How each edge kind draws. `dash` null = solid. These are semantic, not decorative.
const KIND = {
  "extends":         { color: "#3f5468", label: "extends" },
  "re-does":         { color: "#ffb037", dash: "7 4", label: "re-does" },
  "refutes":         { color: "#ff7b9c", dash: "2 4", label: "refutes" },
  "applies":         { color: "#5ee0c8", dash: "1 5", label: "applies" },
  "prerequisite-of": { color: "#8fa8ff", dash: "10 4", label: "prerequisite of" },
};
const kindOf = (k) => KIND[k] || KIND["extends"];

const parentsOf = (t) =>
  (Array.isArray(t.follow_up_of) ? t.follow_up_of : t.follow_up_of ? [t.follow_up_of] : [])
    .map((p) => (typeof p === "string" ? { id: p, dir: null, kind: "extends" } : p))
    .filter((p) => p && p.id);

export default function GraphView({ overview, onOpenTask, onOpenRef }) {
  const dirs = overview?.directions || [];

  // Stable signature so the 4s board poll doesn't relayout and make the map twitch.
  const sig = dirs.map((d) =>
    d.tasks.map((t) =>
      `${d.id}/${t.id}:${t.status}:${t.has_artifact ? 1 : 0}:` +
      parentsOf(t).map((p) => `${p.dir || d.id}/${p.id}/${p.kind}`).join(",") +
      `:${(t.tags || []).join(",")}`
    ).join("|")
  ).join("||");

  const { nodes, edges, tags, width, height, byKey } = useMemo(() => {
    const nodes = [];
    const byKey = new Map();
    const byId = new Map();
    for (const d of dirs) {
      for (const t of d.tasks) {
        const key = `${d.id}/${t.id}`;
        const n = {
          key, id: t.id, direction: d.id, directionName: d.name, title: t.title,
          status: t.status, has: t.has_artifact, detail: t.detail,
          tags: t.tags && t.tags.length ? t.tags : [d.id],
          rawParents: parentsOf(t),
        };
        nodes.push(n); byKey.set(key, n);
        if (!byId.has(t.id)) byId.set(t.id, key);
      }
    }
    // Resolve parents across directions.
    for (const n of nodes) {
      n.parents = n.rawParents
        .map((p) => (p.dir && byKey.has(`${p.dir}/${p.id}`) ? `${p.dir}/${p.id}` : byId.get(p.id)))
        .filter((k) => k && k !== n.key);
      n.parentKind = {};
      n.rawParents.forEach((p) => {
        const k = p.dir && byKey.has(`${p.dir}/${p.id}`) ? `${p.dir}/${p.id}` : byId.get(p.id);
        if (k) n.parentKind[k] = p.kind || "extends";
      });
    }

    // depth = longest ancestor chain (cycle-safe)
    const memo = new Map();
    const depth = (key, stack = new Set()) => {
      if (memo.has(key)) return memo.get(key);
      if (stack.has(key)) return 0;
      stack.add(key);
      const n = byKey.get(key);
      const ps = (n?.parents || []).filter((k) => byKey.has(k));
      const v = ps.length ? 1 + Math.max(...ps.map((k) => depth(k, stack))) : 0;
      stack.delete(key);
      memo.set(key, v);
      return v;
    };
    nodes.forEach((n) => (n.depth = depth(n.key)));

    // Column packing: order each column by its parents' mean row to reduce crossings.
    const cols = new Map();
    nodes.forEach((n) => { if (!cols.has(n.depth)) cols.set(n.depth, []); cols.get(n.depth).push(n); });
    const depths = [...cols.keys()].sort((a, b) => a - b);
    const rowOf = new Map();
    for (const d of depths) {
      const col = cols.get(d);
      col.sort((a, b) => {
        const ra = a.parents.length ? a.parents.reduce((s, k) => s + (rowOf.get(k) ?? 0), 0) / a.parents.length : 1e6;
        const rb = b.parents.length ? b.parents.reduce((s, k) => s + (rowOf.get(k) ?? 0), 0) / b.parents.length : 1e6;
        if (ra !== rb) return ra - rb;
        return a.title.localeCompare(b.title);
      });
      col.forEach((n, i) => {
        n.x = MARGIN + d * COL_W;
        n.y = MARGIN + i * ROW_H;
        rowOf.set(n.key, i);
      });
    }

    // Edges, with a lane per edge so long spans don't stack on the same gutter line.
    const edges = [];
    const laneUse = new Map();
    nodes.forEach((n) => n.parents.forEach((pk) => {
      const p = byKey.get(pk);
      if (!p) return;
      const span = n.depth - p.depth;
      const gk = p.depth;                            // gutter immediately right of the parent column
      const used = laneUse.get(gk) || 0;
      laneUse.set(gk, used + 1);
      edges.push({
        from: p, to: n, key: `${pk}->${n.key}`,
        kind: n.parentKind[pk] || "extends",
        span, lane: used,
      });
    }));

    const tagSet = [...new Set(nodes.flatMap((n) => n.tags))].sort();
    const maxRows = Math.max(1, ...depths.map((d) => cols.get(d).length));
    const width = MARGIN * 2 + (Math.max(0, ...nodes.map((n) => n.depth)) + 1) * COL_W;
    const height = MARGIN * 2 + maxRows * ROW_H;
    return { nodes, edges, tags: tagSet, width, height, byKey };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

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
    const k = Math.min(1, Math.max(0.25, Math.min(el.clientWidth / width, el.clientHeight / height) * 0.94));
    setView({ x: Math.max(12, (el.clientWidth - width * k) / 2), y: 20, k });
  };
  useEffect(() => { fit(); /* eslint-disable-next-line */ }, [width, height]);

  const onPointerDown = (e) => {
    if (e.target.closest(".gnode")) return;
    drag.current = { sx: e.clientX, sy: e.clientY, ox: view.x, oy: view.y, moved: false };
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* already released */ }
  };
  const onPointerMove = (e) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.sx, dy = e.clientY - d.sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    // CRASH FIX: capture the origin into locals. The updater must not dereference drag.current, which
    // can be nulled by pointerup/pointerleave before React runs it.
    const nx = d.ox + dx, ny = d.oy + dy;
    setView((v) => ({ ...v, x: nx, y: ny }));
  };
  const onPointerUp = () => { drag.current = null; };

  const zoom = (factor, cx, cy) => setView((v) => {
    const k = Math.min(2.2, Math.max(0.2, v.k * factor));
    const el = wrapRef.current;
    const px = cx ?? (el ? el.clientWidth / 2 : 0), py = cy ?? (el ? el.clientHeight / 2 : 0);
    return { k, x: px - (px - v.x) * (k / v.k), y: py - (py - v.y) * (k / v.k) };
  });

  // Native non-passive wheel + rAF coalescing. React's onWheel is passive, so preventDefault() there
  // does nothing and the gesture escapes to the shell; and an uncoalesced handler queues a re-render
  // per wheel tick, which is the other half of the scroll-crash story.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    let pending = null, raf = 0;
    const flush = () => {
      raf = 0;
      if (!pending) return;
      const { dy, x, y } = pending; pending = null;
      zoom(dy < 0 ? 1.1 : 1 / 1.1, x, y);
    };
    const onWheel = (e) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      pending = { dy: e.deltaY, x: e.clientX - r.left, y: e.clientY - r.top };
      if (!raf) raf = requestAnimationFrame(flush);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => { el.removeEventListener("wheel", onWheel); if (raf) cancelAnimationFrame(raf); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Lineage: the hovered node plus its whole ancestry and descent — the "story" of that task.
  const lineage = useMemo(() => {
    if (!hover) return null;
    const set = new Set([hover]);
    const up = (k) => (byKey.get(k)?.parents || []).forEach((p) => { if (!set.has(p)) { set.add(p); up(p); } });
    const down = (k) => edges.forEach((e) => { if (e.from.key === k && !set.has(e.to.key)) { set.add(e.to.key); down(e.to.key); } });
    up(hover); down(hover);
    return set;
  }, [hover, edges, byKey]);

  const dim = (n) => (activeTag && !n.tags.includes(activeTag)) || (lineage && !lineage.has(n.key));
  const edgeDim = (e) => dim(e.from) || dim(e.to);
  const tagColor = (t) => TAG_COLORS[t] || FALLBACK[[...tags].indexOf(t) % FALLBACK.length];

  const openNode = (n) => {
    if (drag.current?.moved) return;
    if (n.has) onOpenTask?.({ ...n, key: n.key });
    else onOpenRef?.(n.direction, n.id, false);
  };

  // Orthogonal routing that never crosses a node body.
  //
  // The only node-free space is (a) the vertical gutters between columns and (b) the horizontal band
  // between two rows: nodes occupy ROW_H*r .. ROW_H*r + NODE_H, leaving ROW_H - NODE_H clear all the way
  // across every column. So a long edge drops into its parent's gutter, crosses in an inter-row CHANNEL,
  // climbs the gutter immediately left of its target, and steps in. An adjacent-column edge needs only
  // the single gutter between them and can stay a simple S.
  const rowIdx = (n) => Math.round((n.y - MARGIN) / ROW_H);
  const corner = (x, y, dx, dy, r) => `Q${x},${y} ${x + dx * r},${y + dy * r}`;

  const path = (e) => {
    const x1 = e.from.x + NODE_W, y1 = e.from.y + NODE_H / 2;
    const x2 = e.to.x, y2 = e.to.y + NODE_H / 2;
    const r = 8;

    if (Math.abs(y1 - y2) < 1 && e.span <= 1) return `M${x1},${y1} L${x2},${y2}`;

    const exitX = x1 + 12 + (e.lane % 4) * ((GUTTER - 30) / 4);

    if (e.span <= 1) {
      // one gutter, nothing in between to hit
      const d = y2 > y1 ? 1 : -1;
      return [`M${x1},${y1}`, `L${exitX - r},${y1}`, corner(exitX, y1, 0, d, r),
              `L${exitX},${y2 - r * d}`, corner(exitX, y2, 1, 0, r), `L${x2},${y2}`].join(" ");
    }

    // Long span: cross in the clear band under the upper of the two rows.
    const band = Math.min(rowIdx(e.from), rowIdx(e.to));
    const gap = ROW_H - NODE_H;
    const chY = MARGIN + band * ROW_H + NODE_H + gap / 2 + ((e.lane % 3) - 1) * (gap / 5);
    const entryX = x2 - 16 - (e.lane % 3) * 5;
    const d1 = chY > y1 ? 1 : -1;
    const d2 = y2 > chY ? 1 : -1;
    return [
      `M${x1},${y1}`,
      `L${exitX - r},${y1}`, corner(exitX, y1, 0, d1, r),
      `L${exitX},${chY - r * d1}`, corner(exitX, chY, 1, 0, r),
      `L${entryX - r},${chY}`, corner(entryX, chY, 0, d2, r),
      `L${entryX},${y2 - r * d2}`, corner(entryX, y2, 1, 0, r),
      `L${x2},${y2}`,
    ].join(" ");
  };

  if (!overview) return <div className="muted pad">Loading…</div>;
  if (nodes.length === 0) return <div className="muted pad">No tasks yet — the graph fills in as you add them.</div>;

  const hoverNode = hover ? byKey.get(hover) : null;
  const usedKinds = [...new Set(edges.map((e) => e.kind))];

  return (
    <div className="graphview">
      <div className="graph-toolbar">
        <span className="graph-title">
          Task map <span className="muted">· {nodes.length} tasks · {edges.length} links</span>
        </span>
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
           onPointerDown={onPointerDown} onPointerMove={onPointerMove}
           onPointerUp={onPointerUp} onPointerLeave={onPointerUp}>
        <svg className="graph-svg" width="100%" height="100%">
          <defs>
            {Object.entries(KIND).map(([k, v]) => (
              <marker key={k} id={`ar-${k}`} markerWidth="8" markerHeight="8" refX="6.5" refY="3"
                      orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L6.5,3 L0,6 Z" fill={v.color} />
              </marker>
            ))}
          </defs>
          <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}
             style={{ opacity: mounted ? 1 : 0, transition: "opacity .45s ease" }}>
            {edges.map((e) => {
              const kv = kindOf(e.kind);
              const lit = lineage && lineage.has(e.from.key) && lineage.has(e.to.key);
              return (
                <path key={e.key} className="gedge" d={path(e)} fill="none"
                      stroke={kv.color} strokeDasharray={kv.dash || undefined}
                      strokeWidth={lit ? 2.2 : 1.4}
                      opacity={edgeDim(e) ? 0.08 : lit ? 1 : 0.45}
                      markerEnd={`url(#ar-${e.kind in KIND ? e.kind : "extends"})`} />
              );
            })}
            {nodes.map((n, i) => {
              const c = st(n.status);
              const d = dim(n);
              return (
                <g key={n.key} className="gnode" transform={`translate(${n.x},${n.y})`}
                   style={{ cursor: "pointer", opacity: d ? 0.16 : 1,
                            transition: "opacity .28s ease",
                            transitionDelay: mounted ? "0s" : `${Math.min(i * 10, 320)}ms` }}
                   onMouseEnter={() => setHover(n.key)} onMouseLeave={() => setHover(null)}
                   onClick={() => openNode(n)}>
                  <rect width={NODE_W} height={NODE_H} rx={4} fill={c.fill} stroke={c.stroke}
                        strokeWidth={hover === n.key ? 2 : 1.2} />
                  {n.tags.slice(0, 3).map((tg, j) => (
                    <rect key={tg} x={0} y={j * (NODE_H / Math.min(n.tags.length, 3))}
                          width={3.5} height={NODE_H / Math.min(n.tags.length, 3)}
                          fill={tagColor(tg)} opacity={0.95} />
                  ))}
                  {wrap(n.title, 30, 2).map((line, li) => (
                    <text key={li} x={13} y={19 + li * 13} className="gnode-title" fill={c.text}>{line}</text>
                  ))}
                  {n.has && <circle cx={NODE_W - 11} cy={NODE_H - 11} r={2.6} fill={c.stroke} opacity={0.8} />}
                  {n.status === "active" && <circle cx={NODE_W - 11} cy={12} r={3.5} fill="#7ee787" className="gpulse" />}
                </g>
              );
            })}
          </g>
        </svg>

        {hoverNode && (
          <div className="graph-tip">
            <b>{hoverNode.title}</b>
            <span className="gt-meta">{hoverNode.directionName} · {st(hoverNode.status).label}</span>
            {hoverNode.parents.length > 0 && (
              <span className="gt-rel">
                {hoverNode.rawParents.map((p) => {
                  const k = p.dir && byKey.has(`${p.dir}/${p.id}`) ? `${p.dir}/${p.id}` : null;
                  const pn = k ? byKey.get(k) : nodes.find((x) => x.id === p.id);
                  return pn ? (
                    <span key={p.id} className="gt-edge">
                      <em style={{ color: kindOf(p.kind).color }}>{kindOf(p.kind).label}</em> {pn.title}
                    </span>
                  ) : null;
                })}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="graph-legend">
        {usedKinds.map((k) => (
          <span key={k} className="glegend">
            <svg width="26" height="8" style={{ overflow: "visible" }}>
              <line x1="0" y1="4" x2="22" y2="4" stroke={kindOf(k).color} strokeWidth="1.8"
                    strokeDasharray={kindOf(k).dash || undefined} />
            </svg>
            {kindOf(k).label}
          </span>
        ))}
        <span className="muted">drag to pan · scroll or ± to zoom · hover for the lineage · click to open</span>
      </div>
    </div>
  );
}

// Wrap a title into at most `max` lines of ~`n` chars, so tasks are readable without hovering.
function wrap(s, n, max) {
  const words = String(s).split(/\s+/);
  const lines = [];
  let cur = "";
  for (const w of words) {
    if (!cur.length) cur = w;
    else if ((cur + " " + w).length <= n) cur += " " + w;
    else { lines.push(cur); cur = w; if (lines.length === max) break; }
  }
  if (lines.length < max && cur) lines.push(cur);
  if (lines.length === max) {
    const consumed = lines.join(" ").length;
    if (consumed < String(s).length - 1) lines[max - 1] = lines[max - 1].replace(/.{0,2}$/, "…");
  }
  return lines;
}
