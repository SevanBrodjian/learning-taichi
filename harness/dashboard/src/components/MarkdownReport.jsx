import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import VideoPlayer from "./VideoPlayer.jsx";

// Resolve [[target|label]] wiki-links. If `target` is a known section id, turn it into an
// in-app cross-reference (#train:id) the Training view intercepts; otherwise drop the brackets
// to plain emphasis so unresolved links never render as raw [[...]].
function resolveWikiLinks(md, ids) {
  return md.replace(/\[\[([^\]]+)\]\]/g, (_, inner) => {
    const [rawTarget, rawLabel] = inner.split("|");
    const target = rawTarget.trim();
    const label = (rawLabel ?? rawTarget).trim();
    if (ids && ids.has(target)) return `[${label}](#train:${target})`;
    return `*${label}*`;
  });
}

// Same rendering pipeline the personal site can adopt: GFM + LaTeX math via KaTeX, plus the
// lightweight wiki-links above for the hyperlinked-textbook structure.
// Memoized: unrelated App state (the 4s board poll) re-renders ancestors, and without this the
// react-markdown subtree — including any embedded <video> — would re-render and could reset playback.
// Memo keeps a paused training video paused, since equal props skip the re-render entirely.
// react-markdown does not render raw HTML (no rehype-raw), so an HTML COMMENT does not disappear — it
// leaks into the page as body text. That is wrong everywhere (the decisions carry an `auto_run_at`
// comment the server parses; the notebook template opens with a note to its writer) and it is worst on a
// document whose author expects `<!-- ... -->` to be invisible. Strip them, but only outside fenced code
// blocks and inline code spans, or a page documenting HTML would have its examples eaten.
const MARK = "\uE000";   // a private-use char, so a placeholder can never collide with real text
function stripHtmlComments(md) {
  if (!md.includes("<!--")) return md;
  const out = [];
  let fence = null;      // the ``` or ~~~ run that opened the current code block
  let inComment = false; // a comment that spans lines
  for (const line of md.split("\n")) {
    const f = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
    if (!inComment && f) {
      if (fence && line.trim().startsWith(fence)) fence = null;
      else if (!fence) fence = f[1];
      out.push(line);
      continue;
    }
    if (fence) { out.push(line); continue; }
    let s = line;
    if (inComment) {
      const end = s.indexOf("-->");
      if (end < 0) continue;                 // still inside the comment: drop the whole line
      s = s.slice(end + 3);
      inComment = false;
    }
    // Protect inline code spans, then remove complete comments, then note an unterminated one.
    const spans = [];
    s = s.replace(/`[^`]*`/g, (m) => { spans.push(m); return MARK + (spans.length - 1) + MARK; });
    s = s.replace(/<!--[\s\S]*?-->/g, "");
    const open = s.indexOf("<!--");
    if (open >= 0) { s = s.slice(0, open); inComment = true; }
    s = s.replace(/\uE000(\d+)\uE000/g, (_, i) => spans[Number(i)]);
    if (s.trim() === "" && line.trim() !== "") continue;  // the line was only a comment
    out.push(s);
  }
  return out.join("\n");
}

// A doc that lives on disk can reference a sibling file the way any markdown file does — the notebook
// writes `![](media/sketch.jpg)`. The browser would resolve that against the dashboard's own URL, so a
// doc served from /api/data/... passes `baseUrl` and relative sources resolve against the doc instead.
function resolveSrc(src, baseUrl) {
  if (!src || !baseUrl) return src;
  if (/^([a-z]+:|\/\/|\/)/i.test(src)) return src;      // absolute, protocol-relative, or rooted
  return baseUrl.replace(/\/?$/, "/") + src.replace(/^\.\//, "");
}

function MarkdownReport({ markdown, sections, onNavigate, baseUrl }) {
  if (!markdown) return null;
  const ids = sections ? new Set(sections.map((s) => s.id)) : null;
  const text = resolveWikiLinks(stripHtmlComments(markdown), ids);

  const components = {
    // Markdown image syntax pointing at a video renders the same clean autoplay player as the task
    // pages (play/pause that holds, minimal controls), so training pages can embed grid/heatmap demos
    // rather than static diagrams (#13). Alt text becomes a visible caption (spans, not <figure>, so it
    // stays valid inside the <p> react-markdown wraps a lone image in). Captions are plain text, so write
    // them without $math$.
    img({ src: rawSrc, alt, ...props }) {
      const src = resolveSrc(rawSrc, baseUrl);
      const isVideo = src && /\.(mp4|webm|mov)(\?|$)/i.test(src);
      const media = isVideo
        ? <VideoPlayer src={src} className="md-video" />
        : <img src={src} alt={alt} loading="lazy" {...props} />;
      if (!alt) return media;
      return (
        <span className="md-figure">
          {media}
          <span className="md-figcaption">{alt}</span>
        </span>
      );
    },
    a({ href, children, ...props }) {
      if (href && href.startsWith("#train:")) {
        const id = href.slice("#train:".length);
        return (
          <a
            href={href}
            className="xref"
            onClick={(e) => {
              e.preventDefault();
              if (onNavigate) onNavigate(id);
            }}
          >
            {children}
          </a>
        );
      }
      const ext = href && /^https?:/.test(href);
      return (
        <a href={href} {...(ext ? { target: "_blank", rel: "noreferrer" } : {})} {...props}>
          {children}
        </a>
      );
    },
  };

  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

export default memo(MarkdownReport);
