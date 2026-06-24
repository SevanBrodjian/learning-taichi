// Dependency-free SVG loss curve (log-y by default). Plain SVG so it ports anywhere and stays
// themeable via CSS. Sized for the task tiles: integer-power y ticks, labeled x axis, legible fonts.
export default function LossChart({ series, width = 680, height = 340, log = true }) {
  if (!series || series.length === 0) return <div className="muted chart-empty">No metrics yet.</div>;

  const pad = { l: 78, r: 24, t: 24, b: 54 };
  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;

  const xs = series.map((d) => d.iter);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);

  const t = (v) => (log ? Math.log10(Math.max(v, 1e-12)) : v);
  const tvals = series.map((d) => t(d.loss));
  let tmin = Math.min(...tvals);
  let tmax = Math.max(...tvals);
  if (tmin === tmax) { tmin -= 1; tmax += 1; }

  const X = (x) => pad.l + iw * ((x - xmin) / (xmax - xmin || 1));
  const Y = (v) => pad.t + ih * (1 - (t(v) - tmin) / (tmax - tmin || 1));

  const d = series.map((p, i) => `${i ? "L" : "M"}${X(p.iter).toFixed(1)} ${Y(p.loss).toFixed(1)}`).join(" ");

  // y ticks at integer powers of 10 (log) so labels read "1e-4", not "1e-4.3".
  let ylines;
  if (log) {
    const lo = Math.ceil(tmin);
    const hi = Math.floor(tmax);
    const exps = [];
    for (let e = lo; e <= hi; e++) exps.push(e);
    const src = exps.length >= 2 ? exps : [Math.round(tmin), Math.round(tmax)];
    ylines = src.map((e) => ({ y: pad.t + ih * (1 - (e - tmin) / (tmax - tmin || 1)), label: `1e${e}` }));
  } else {
    ylines = Array.from({ length: 5 }, (_, i) => ({
      y: pad.t + ih * (1 - i / 4),
      label: (tmin + (i / 4) * (tmax - tmin)).toPrecision(2),
    }));
  }

  const xlabels = Array.from({ length: 5 }, (_, i) => {
    const xv = xmin + (i / 4) * (xmax - xmin);
    return { x: X(xv), label: Math.round(xv) };
  });

  const last = series[series.length - 1];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="loss-chart"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Optimization loss curve"
    >
      {ylines.map((l, i) => (
        <g key={i}>
          <line x1={pad.l} y1={l.y} x2={width - pad.r} y2={l.y} className="grid" />
          <text x={pad.l - 12} y={l.y + 5} className="tick" textAnchor="end">{l.label}</text>
        </g>
      ))}
      {xlabels.map((l, i) => (
        <text key={i} x={l.x} y={height - pad.b + 26} className="tick" textAnchor="middle">{l.label}</text>
      ))}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} className="axis" />
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} className="axis" />
      <text x={pad.l + iw / 2} y={height - 12} className="axis-label" textAnchor="middle">iteration</text>
      <text className="axis-label" textAnchor="middle" transform={`translate(22 ${pad.t + ih / 2}) rotate(-90)`}>
        {log ? "loss (log scale)" : "loss"}
      </text>
      <path d={d} className="loss-line" fill="none" />
      <circle cx={X(last.iter)} cy={Y(last.loss)} r="4" className="loss-dot" />
    </svg>
  );
}
