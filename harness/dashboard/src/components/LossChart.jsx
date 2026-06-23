// Dependency-free SVG loss curve (log-y by default). Kept as plain SVG so it ports anywhere and stays
// fully themeable via CSS.
export default function LossChart({ series, width = 680, height = 300, log = true }) {
  if (!series || series.length === 0) return <div className="muted chart-empty">No metrics yet.</div>;

  const pad = { l: 64, r: 18, t: 18, b: 42 };
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

  const d = series
    .map((p, i) => `${i ? "L" : "M"}${X(p.iter).toFixed(1)} ${Y(p.loss).toFixed(1)}`)
    .join(" ");

  const ticks = 4;
  const ylines = Array.from({ length: ticks + 1 }, (_, i) => {
    const tv = tmin + (i / ticks) * (tmax - tmin);
    const y = pad.t + ih * (1 - i / ticks);
    return { y, label: log ? `1e${tv.toFixed(1)}` : tv.toPrecision(2) };
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
          <text x={pad.l - 10} y={l.y + 4} className="tick" textAnchor="end">{l.label}</text>
        </g>
      ))}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} className="axis" />
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} className="axis" />
      <text x={pad.l + iw / 2} y={height - 8} className="axis-label" textAnchor="middle">iteration</text>
      <text className="axis-label" textAnchor="middle"
            transform={`translate(16 ${pad.t + ih / 2}) rotate(-90)`}>{log ? "loss (log)" : "loss"}</text>
      <path d={d} className="loss-line" fill="none" />
      <circle cx={X(last.iter)} cy={Y(last.loss)} r="3.5" className="loss-dot" />
      <text x={X(last.iter) - 8} y={Y(last.loss) - 8} className="tick" textAnchor="end">
        {Number(last.loss).toExponential(2)}
      </text>
    </svg>
  );
}
