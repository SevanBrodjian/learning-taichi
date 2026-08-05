import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * The Map — a relaxation-solved task graph.
 *
 * WHY THIS SHAPE. The previous version was a layered DAG: columns by depth, left to right. With 21 tasks
 * at depth 7 it smeared into a ~2000px strip four rows tall — an aspect ratio no screen wants, unrelated
 * tasks forced into the same column, and long edges that had to be routed through whatever gaps were
 * left. Sevan: "one long horizontal stretch does not visualize well ... the arrows are all crazy
 * overlapping ... tasks should be less fixed and gridlocked -- maybe they can move fluidly and adapt to
 * use space effectively."
 *
 * So the layout is a small force solve. Nodes repel, edges pull, tasks sharing a tag attract, and a weak
 * gravity keeps it centred. Related work clusters on its own, the graph fills two dimensions instead of
 * one, and edges become short straight lines between well-separated nodes — which reads far better than
 * orthogonal routing through a dense grid.
 *
 * RELIABLE AND FAST, NOT A TOY. The solve is deterministic (positions seed from a hash of the task id)
 * and runs to completion SYNCHRONOUSLY before first paint — a fixed iteration budget, then frozen. It
 * does not jitter, does not spin the CPU, and looks identical on every reload. rAF is used only to relax
 * the graph while a node is being dragged, and it stops as soon as the energy decays.
 *
 * TOUCH IS A FIRST-CLASS TARGET (it is an iPad PWA):
 *  - SELECTION, not hover, drives the lineage highlight. Tapping a node pins its story; hover is a
 *    desktop-only convenience layered on top. There is no hover on an iPad.
 *  - Pinch-to-zoom and one-finger pan via pointer events, with `touch-action: none` so the browser does
 *    not steal the gesture. Panning previously fought the browser and felt broken.
 *  - The wheel listener is attached once the canvas actually exists. The old one was registered in a
 *    mount-only effect that ran while the component was still rendering its "Loading…" branch, so the
 *    ref was null, it bailed, and it never re-ran — which is why scroll-zoom silently stopped working.
 */

const NODE_W = 172;
const NODE_H = 46;

const STATUS = {
  proposed: { fill: "#141b25", stroke: "#3a4658", text: "#9fb0c0", label: "Proposed" },
  queued:   { fill: "#0f2230", stroke: "#4cc2ff", text: "#bfe6ff", label: "Queued" },
  active:   { fill: "#11261a", stroke: "#7ee787", text: "#c7f5cf", label: "Active" },
  done:     { fill: "#12201a", stroke: "#2f6f4e", text: "#8fc7a8", label: "Done" },
};
const st = (s) => STATUS[s] || STATUS.proposed;

const TAG_COLORS = { gradients: "#4cc2ff", materials: "#ffb037", learned: "#c98bff", rendering: "#5ee0c8" };
const FALLBACK = ["#7ee787", "#ff7b9c", "#8fa8ff", "#e6a23c"];

const KIND = {
  "extends":         { color: "#4a6076", label: "extends" },
  "re-does":         { color: "#ffb037", dash: "7 5", label: "re-does" },
  "refutes":         { color: "#ff7b9c", dash: "2 5", label: "refutes" },
  "applies":         { color: "#5ee0c8", dash: "1 6", label: "applies" },
  "prerequisite-of": { color: "#8fa8ff", dash: "10 5", label: "prerequisite of" },
};
const kindOf = (k) => KIND[k] || KIND["extends"];

const parentsOf = (t) =>
  (Array.isArray(t.follow_up_of) ? t.follow_up_of : t.follow_up_of ? [t.follow_up_of] : [])
    .map((p) => (typeof p === "string" ? { id: p, dir: null, kind: "extends" } : p))
    .filter((p) => p && p.id);

// Deterministic per-key pseudo-random in [0,1). Same layout on every reload, no Math.random.
function hash01(s, salt) {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return ((h >>> 0) % 100000) / 100000;
}

export default function GraphView({ overview, onOpenTask, onOpenRef }) {
  const dirs = overview?.directions || [];

  const sig = dirs.map((d) =>
    d.tasks.map((t) =>
      `${d.id}/${t.id}:${t.status}:${t.has_artifact ? 1 : 0}:` +
      parentsOf(t).map((p) => `${p.dir || d.id}/${p.id}/${p.kind}`).join(",") +
      `:${(t.tags || []).join(",")}`
    ).join("|")
  ).join("||");

  // ── build + solve ──────────────────────────────────────────────────────────────────────────────
  const solved = useMemo(() => {
    const nodes = [];
    const byKey = new Map();
    const byId = new Map();
    for (const d of dirs) {
      for (const t of d.tasks) {
        const key = `${d.id}/${t.id}`;
        const n = {
          key, id: t.id, direction: d.id, directionName: d.name, title: t.title,
          status: t.status, has: t.has_artifact, detail: t.detail,
          tags: (t.tags && t.tags.length ? t.tags : [d.id]),
          raw: parentsOf(t),
        };
        nodes.push(n); byKey.set(key, n);
        if (!byId.has(t.id)) byId.set(t.id, key);
      }
    }
    const resolve = (p) => (p.dir && byKey.has(`${p.dir}/${p.id}`) ? `${p.dir}/${p.id}` : byId.get(p.id));
    for (const n of nodes) {
      n.parents = n.raw.map(resolve).filter((k) => k && k !== n.key);
      n.kindOfParent = {};
      n.raw.forEach((p) => { const k = resolve(p); if (k) n.kindOfParent[k] = p.kind || "extends"; });
    }
    const edges = [];
    for (const n of nodes) {
      for (const pk of n.parents) {
        const p = byKey.get(pk);
        if (p) edges.push({ from: p, to: n, key: `${pk}->${n.key}`, kind: n.kindOfParent[pk] || "extends" });
      }
    }

    // generation = longest ancestor chain; used only as a gentle left-to-right bias so the story still
    // reads in reading order without the layout being locked to a grid.
    const memo = new Map();
    const gen = (k, stack = new Set()) => {
      if (memo.has(k)) return memo.get(k);
      if (stack.has(k)) return 0;
      stack.add(k);
      const ps = (byKey.get(k)?.parents || []).filter((x) => byKey.has(x));
      const v = ps.length ? 1 + Math.max(...ps.map((x) => gen(x, stack))) : 0;
      stack.delete(k); memo.set(k, v);
      return v;
    };
    nodes.forEach((n) => (n.gen = gen(n.key)));
    const maxGen = Math.max(1, ...nodes.map((n) => n.gen));

    // deterministic seed: spread on a spiral, nudged right by generation
    const W = 1180, H = 720;
    nodes.forEach((n, i) => {
      const a = hash01(n.key, 1) * Math.PI * 2;
      const r = 120 + hash01(n.key, 2) * 260;
      n.x = W / 2 + Math.cos(a) * r + (n.gen / maxGen - 0.5) * 420;
      n.y = H / 2 + Math.sin(a) * r;
      n.vx = 0; n.vy = 0;
    });

    // tag centroids give same-tag work a place to gather
    const tagList = [...new Set(nodes.flatMap((n) => n.tags))].sort();
    const tagAnchor = new Map();
    tagList.forEach((t, i) => {
      const a = (i / Math.max(1, tagList.length)) * Math.PI * 2 - Math.PI / 2;
      tagAnchor.set(t, { x: W / 2 + Math.cos(a) * 300, y: H / 2 + Math.sin(a) * 190 });
    });

    const LINK_LEN = 232, LINK_K = 0.05;
    const REPEL = 44000, DAMP = 0.82;
    const step = (iter) => {
      const cool = Math.max(0.12, 1 - iter / 640);
      for (const n of nodes) { n.fx = 0; n.fy = 0; }
      // repulsion (21 nodes -> 210 pairs; no quadtree needed)
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          let dx = a.x - b.x, dy = (a.y - b.y) * 1.35;   // vertical bias: keep rows readable
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) { dx = (hash01(a.key, 3) - 0.5) || 0.1; dy = (hash01(b.key, 4) - 0.5) || 0.1; d2 = 1; }
          const f = Math.min(REPEL / d2, 24);
          const d = Math.sqrt(d2);
          a.fx += (dx / d) * f; a.fy += (dy / d) * f;
          b.fx -= (dx / d) * f; b.fy -= (dy / d) * f;
        }
      }
      // link springs
      for (const e of edges) {
        const dx = e.to.x - e.from.x, dy = e.to.y - e.from.y;
        const d = Math.max(1, Math.hypot(dx, dy));
        const f = (d - LINK_LEN) * LINK_K;
        e.from.fx += (dx / d) * f; e.from.fy += (dy / d) * f;
        e.to.fx -= (dx / d) * f; e.to.fy -= (dy / d) * f;
      }
      // tag cohesion + centre gravity + generational left-to-right bias
      for (const n of nodes) {
        for (const t of n.tags) {
          const a = tagAnchor.get(t);
          if (a) { n.fx += (a.x - n.x) * 0.0055; n.fy += (a.y - n.y) * 0.0055; }
        }
        n.fx += (W / 2 - n.x) * 0.004;
        n.fy += (H / 2 - n.y) * 0.006;
        const targetX = 150 + (n.gen / maxGen) * (W - 300);
        n.fx += (targetX - n.x) * 0.010;
      }
      // EDGE CLEARANCE. A force layout draws straight edges, and in a dense field those cut straight
      // through unrelated boxes — the same "lines overlap blocks" complaint in a new form. So once the
      // graph has roughly settled, push any node that is sitting on top of an edge off to the side.
      // Cheap: 24 edges x 21 nodes.
      if (iter > 260) {
        for (const e of edges) {
          const ax = e.from.x + NODE_W / 2, ay = e.from.y + NODE_H / 2;
          const bx = e.to.x + NODE_W / 2, by = e.to.y + NODE_H / 2;
          const ex = bx - ax, ey = by - ay;
          const len2 = ex * ex + ey * ey;
          if (len2 < 1) continue;
          for (const n of nodes) {
            if (n === e.from || n === e.to) continue;
            const cx = n.x + NODE_W / 2, cy = n.y + NODE_H / 2;
            let t = ((cx - ax) * ex + (cy - ay) * ey) / len2;
            if (t <= 0.02 || t >= 0.98) continue;          // only the middle of the span matters
            const px = ax + ex * t, py = ay + ey * t;
            // scale the offset into node-space so a wide box needs more horizontal clearance
            const dx = (cx - px) / (NODE_W / 2 + 16), dy = (cy - py) / (NODE_H / 2 + 14);
            const d = Math.hypot(dx, dy);
            if (d >= 1) continue;                          // already clear
            const push = (1 - d) * 11 * cool;
            const nx = d > 0.01 ? dx / d : (hash01(n.key, 5) - 0.5);
            const ny = d > 0.01 ? dy / d : 1;
            n.fx += nx * push; n.fy += ny * push * 1.4;
          }
        }
      }

      // rectangle separation so labels never overlap
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const ox = (NODE_W + 30) - Math.abs(a.x - b.x);
          const oy = (NODE_H + 30) - Math.abs(a.y - b.y);
          if (ox > 0 && oy > 0) {
            if (ox < oy) { const s = (a.x < b.x ? -1 : 1) * ox * 0.5; a.x += s; b.x -= s; }
            else { const s = (a.y < b.y ? -1 : 1) * oy * 0.5; a.y += s; b.y -= s; }
          }
        }
      }
      for (const n of nodes) {
        n.vx = (n.vx + n.fx) * DAMP; n.vy = (n.vy + n.fy) * DAMP;
        const sp = Math.hypot(n.vx, n.vy), cap = 26 * cool;
        if (sp > cap) { n.vx = (n.vx / sp) * cap; n.vy = (n.vy / sp) * cap; }
        n.x += n.vx; n.y += n.vy;
      }
    };
    for (let i = 0; i < 700; i++) step(i);          // solved before first paint, then frozen

    // normalize into a tidy box
    const pad = 70;
    const minX = Math.min(...nodes.map((n) => n.x)), maxX = Math.max(...nodes.map((n) => n.x));
    const minY = Math.min(...nodes.map((n) => n.y)), maxY = Math.max(...nodes.map((n) => n.y));
    nodes.forEach((n) => { n.x = n.x - minX + pad; n.y = n.y - minY + pad; });
    const width = (maxX - minX) + pad * 2 + NODE_W;
    const height = (maxY - minY) + pad * 2 + NODE_H;
    return { nodes, edges, byKey, tags: tagList, width, height, step };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const { nodes, edges, byKey, tags, width, height } = solved;

  // ── view + interaction ─────────────────────────────────────────────────────────────────────────
  const wrapRef = useRef(null);
  const [view, setView] = useState({ x: 0, y: 0, k: 0.85 });
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);
  const [activeTag, setActiveTag] = useState(null);
  const [, force] = useState(0);
  const pointers = useRef(new Map());
  const gesture = useRef(null);
  const nodeDrag = useRef(null);
  const relaxRaf = useRef(0);

  const fit = useCallback(() => {
    const el = wrapRef.current;
    if (!el || !width || !height) return;
    const k = Math.min(1.1, Math.max(0.2, Math.min(el.clientWidth / width, el.clientHeight / height) * 0.92));
    setView({ x: (el.clientWidth - width * k) / 2, y: (el.clientHeight - height * k) / 2, k });
  }, [width, height]);
  useEffect(() => { fit(); }, [fit]);

  const toWorld = (cx, cy) => {
    const r = wrapRef.current.getBoundingClientRect();
    return { x: (cx - r.left - view.x) / view.k, y: (cy - r.top - view.y) / view.k };
  };

  // Relax the graph for a short while after a node is dragged, then stop. This is the only animation.
  const relax = useCallback(() => {
    if (relaxRaf.current) return;
    let n = 0;
    const tick = () => {
      for (let i = 0; i < 2; i++) solved.step(300);
      if (nodeDrag.current) {
        const nd = nodeDrag.current;
        nd.node.x = nd.x; nd.node.y = nd.y; nd.node.vx = 0; nd.node.vy = 0;
      }
      force((v) => v + 1);
      n++;
      relaxRaf.current = n < 90 || nodeDrag.current ? requestAnimationFrame(tick) : 0;
    };
    relaxRaf.current = requestAnimationFrame(tick);
  }, [solved]);
  useEffect(() => () => { if (relaxRaf.current) cancelAnimationFrame(relaxRaf.current); }, []);

  const onPointerDown = (e) => {
    const el = wrapRef.current;
    // Must not throw: setPointerCapture rejects ids that are not an active pointer, and an exception here
    // aborts the handler before any gesture state is set — which silently kills tap-to-select.
    try { el.setPointerCapture?.(e.pointerId); } catch { /* not capturable; gestures still work */ }
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const hitKey = e.target.closest?.(".gnode")?.dataset?.key;
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      gesture.current = { d: Math.hypot(a.x - b.x, a.y - b.y), k: view.k,
                          cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, x: view.x, y: view.y };
      nodeDrag.current = null;
      return;
    }
    if (hitKey) {
      const n = byKey.get(hitKey);
      const w = toWorld(e.clientX, e.clientY);
      nodeDrag.current = { node: n, x: n.x, y: n.y, ox: w.x - n.x, oy: w.y - n.y, moved: false };
    } else {
      gesture.current = { pan: true, sx: e.clientX, sy: e.clientY, x: view.x, y: view.y, moved: false };
    }
  };

  const onPointerMove = (e) => {
    if (!pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (pointers.current.size === 2 && gesture.current && !gesture.current.pan) {
      const g = gesture.current;
      const [a, b] = [...pointers.current.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      const k = Math.min(2.4, Math.max(0.18, g.k * (d / Math.max(1, g.d))));
      const r = wrapRef.current.getBoundingClientRect();
      const px = g.cx - r.left, py = g.cy - r.top;
      setView({ k, x: px - (px - g.x) * (k / g.k), y: py - (py - g.y) * (k / g.k) });
      return;
    }
    const nd = nodeDrag.current;
    if (nd) {
      const w = toWorld(e.clientX, e.clientY);
      nd.x = w.x - nd.ox; nd.y = w.y - nd.oy; nd.moved = true;
      force((v) => v + 1);   // repaint on the event itself, so dragging still works if rAF is throttled
      relax();               // rAF only settles the NEIGHBOURS; it is never load-bearing for the drag
      return;
    }
    const g = gesture.current;
    if (g?.pan) {
      const dx = e.clientX - g.sx, dy = e.clientY - g.sy;
      if (Math.abs(dx) + Math.abs(dy) > 3) g.moved = true;
      setView((v) => ({ ...v, x: g.x + dx, y: g.y + dy }));
    }
  };

  const onPointerUp = (e) => {
    pointers.current.delete(e.pointerId);
    const nd = nodeDrag.current, g = gesture.current;
    if (nd && !nd.moved) setSelected((s) => (s === nd.node.key ? null : nd.node.key));
    else if (g?.pan && !g.moved && !nd) setSelected(null);
    nodeDrag.current = null;
    if (pointers.current.size === 0) gesture.current = null;
  };

  // Wheel: attached to the live canvas (not on mount, when it may not exist yet) and rAF-coalesced.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // Throttle by CLOCK, not by rAF. An rAF-coalesced handler latches: the first event schedules a frame,
    // and if that frame never arrives (backgrounded tab, throttled PWA, low-power mode) the pending flag
    // is never cleared and every later wheel event is swallowed — zoom silently dies until reload.
    // A timestamp gate degrades to "applies every event" instead of "applies none".
    let last = 0, accum = 0;
    const onWheel = (ev) => {
      ev.preventDefault();
      accum += ev.deltaY;
      const now = ev.timeStamp || Date.now();
      if (now - last < 16) return;
      last = now;
      const dy = accum; accum = 0;
      if (!dy) return;
      const r = el.getBoundingClientRect();
      const px = ev.clientX - r.left, py = ev.clientY - r.top;
      setView((v) => {
        const k = Math.min(2.4, Math.max(0.18, v.k * (dy < 0 ? 1.12 : 1 / 1.12)));
        return { k, x: px - (px - v.x) * (k / v.k), y: py - (py - v.y) * (k / v.k) };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [nodes.length]);   // re-attach once the canvas actually exists

  const focusKey = selected || hover;
  const lineage = useMemo(() => {
    if (!focusKey) return null;
    const set = new Set([focusKey]);
    const up = (k) => (byKey.get(k)?.parents || []).forEach((p) => { if (!set.has(p)) { set.add(p); up(p); } });
    const down = (k) => edges.forEach((e) => { if (e.from.key === k && !set.has(e.to.key)) { set.add(e.to.key); down(e.to.key); } });
    up(focusKey); down(focusKey);
    return set;
  }, [focusKey, edges, byKey]);

  const dim = (n) => (activeTag && !n.tags.includes(activeTag)) || (lineage && !lineage.has(n.key));
  const tagColor = (t) => TAG_COLORS[t] || FALLBACK[tags.indexOf(t) % FALLBACK.length];

  if (!overview) return <div className="muted pad">Loading…</div>;
  if (!nodes.length) return <div className="muted pad">No tasks yet — the graph fills in as you add them.</div>;

  // Trim an edge to the node borders so the arrowhead lands on the edge of the box, not under it.
  const seg = (e) => {
    const p = nodeDrag.current?.node === e.from ? nodeDrag.current : null;
    const q = nodeDrag.current?.node === e.to ? nodeDrag.current : null;
    const ax = (p ? p.x : e.from.x) + NODE_W / 2, ay = (p ? p.y : e.from.y) + NODE_H / 2;
    const bx = (q ? q.x : e.to.x) + NODE_W / 2, by = (q ? q.y : e.to.y) + NODE_H / 2;
    const clip = (cx, cy, tx, ty) => {
      const dx = tx - cx, dy = ty - cy;
      if (!dx && !dy) return [cx, cy];
      const sx = (NODE_W / 2 + 6) / Math.abs(dx || 1e-6), sy = (NODE_H / 2 + 6) / Math.abs(dy || 1e-6);
      const s = Math.min(sx, sy, 1e6);
      return [cx + dx * Math.min(s, 1), cy + dy * Math.min(s, 1)];
    };
    const [x1, y1] = clip(ax, ay, bx, by);
    const [x2, y2] = clip(bx, by, ax, ay);
    return { x1, y1, x2, y2 };
  };

  const sel = focusKey ? byKey.get(focusKey) : null;
  const usedKinds = [...new Set(edges.map((e) => e.kind))];
  const pos = (n) => (nodeDrag.current?.node === n ? nodeDrag.current : n);

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
          <button className="act-btn" onClick={() => setView((v) => ({ ...v, k: Math.max(0.18, v.k / 1.2) }))}>−</button>
          <button className="act-btn" onClick={() => setView((v) => ({ ...v, k: Math.min(2.4, v.k * 1.2) }))}>+</button>
          <button className="act-btn" onClick={fit}>Fit</button>
        </span>
      </div>

      <div className="graph-canvas" ref={wrapRef}
           onPointerDown={onPointerDown} onPointerMove={onPointerMove}
           onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
        <svg className="graph-svg" width="100%" height="100%">
          <defs>
            {Object.entries(KIND).map(([k, v]) => (
              <marker key={k} id={`ar-${k}`} markerWidth="7" markerHeight="7" refX="6" refY="2.5"
                      orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L6,2.5 L0,5 Z" fill={v.color} />
              </marker>
            ))}
          </defs>
          <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
            {edges.map((e) => {
              const kv = kindOf(e.kind);
              const lit = lineage && lineage.has(e.from.key) && lineage.has(e.to.key);
              const d = dim(e.from) || dim(e.to);
              const s = seg(e);
              return (
                <line key={e.key} className="gedge" x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                      stroke={kv.color} strokeDasharray={kv.dash || undefined}
                      strokeWidth={lit ? 2.6 : 1.2} opacity={d ? 0.05 : lit ? 1 : focusKey ? 0.05 : 0.19}
                      markerEnd={`url(#ar-${e.kind in KIND ? e.kind : "extends"})`} />
              );
            })}
            {nodes.map((n) => {
              const c = st(n.status);
              const p = pos(n);
              const isSel = selected === n.key;
              return (
                <g key={n.key} className="gnode" data-key={n.key}
                   transform={`translate(${p.x},${p.y})`}
                   style={{ cursor: "pointer", opacity: dim(n) ? 0.13 : 1 }}
                   onMouseEnter={() => setHover(n.key)} onMouseLeave={() => setHover(null)}
                   onDoubleClick={() => (n.has ? onOpenTask?.({ ...n, key: n.key }) : onOpenRef?.(n.direction, n.id, false))}>
                  <rect width={NODE_W} height={NODE_H} rx={4} fill={c.fill} stroke={isSel ? "#dfe6ee" : c.stroke}
                        strokeWidth={isSel ? 2 : 1.2} />
                  {n.tags.slice(0, 3).map((tg, j, arr) => (
                    <rect key={tg} x={0} y={(j * NODE_H) / arr.length} width={3} height={NODE_H / arr.length}
                          fill={tagColor(tg)} />
                  ))}
                  {wrap(n.title, 26, 2).map((line, li) => (
                    <text key={li} x={11} y={18 + li * 13} className="gnode-title" fill={c.text}>{line}</text>
                  ))}
                  {n.has && <circle cx={NODE_W - 9} cy={NODE_H - 9} r={2.4} fill={c.stroke} opacity={0.85} />}
                  {n.status === "active" && <circle cx={NODE_W - 9} cy={10} r={3.2} fill="#7ee787" className="gpulse" />}
                </g>
              );
            })}
          </g>
        </svg>

        {sel && (
          <div className="graph-card">
            <b>{sel.title}</b>
            <span className="gt-meta">{sel.directionName} · {st(sel.status).label}
              {sel.tags.map((t) => <i key={t} style={{ background: tagColor(t) }} />)}
            </span>
            {sel.raw.length > 0 && (
              <span className="gt-rel">
                {sel.raw.map((p) => {
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
            {sel.has && (
              <button className="act-btn primary gt-open"
                      onClick={() => onOpenTask?.({ ...sel, key: sel.key })}>Open task →</button>
            )}
          </div>
        )}
      </div>

      <div className="graph-legend">
        {usedKinds.map((k) => (
          <span key={k} className="glegend">
            <svg width="24" height="7"><line x1="0" y1="3.5" x2="20" y2="3.5" stroke={kindOf(k).color}
                  strokeWidth="1.8" strokeDasharray={kindOf(k).dash || undefined} /></svg>
            {kindOf(k).label}
          </span>
        ))}
        <span className="muted glegend-hint">tap a task for its lineage · drag to pan · pinch or scroll to zoom · double-tap to open</span>
      </div>
    </div>
  );
}

function wrap(s, n, max) {
  const words = String(s).split(/\s+/);
  const lines = []; let cur = "";
  for (const w of words) {
    if (!cur) cur = w;
    else if ((cur + " " + w).length <= n) cur += " " + w;
    else { lines.push(cur); cur = w; if (lines.length === max) break; }
  }
  if (lines.length < max && cur) lines.push(cur);
  if (lines.length === max && lines.join(" ").length < String(s).length - 1)
    lines[max - 1] = lines[max - 1].replace(/.{0,1}$/, "…");
  return lines;
}
