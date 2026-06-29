import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

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
export default function MarkdownReport({ markdown, sections, onNavigate }) {
  if (!markdown) return null;
  const ids = sections ? new Set(sections.map((s) => s.id)) : null;
  const text = resolveWikiLinks(markdown, ids);

  const components = {
    // Markdown image syntax pointing at a video renders a real looping <video> so training pages can
    // embed grid/heatmap demos, not just static diagrams (#13).
    img({ src, alt, ...props }) {
      if (src && /\.(mp4|webm|mov)(\?|$)/i.test(src)) {
        return <video className="md-video" src={src} muted loop autoPlay playsInline controls />;
      }
      return <img src={src} alt={alt} loading="lazy" {...props} />;
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
