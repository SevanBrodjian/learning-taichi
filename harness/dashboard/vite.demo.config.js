import { defineConfig } from "vite";

// Standalone WebGPU demo (`npm run build:demo`): demo/index.html + the dashboard's own mpm/ bundle,
// emitted as ONE self-contained HTML file (script inlined), so it can be dropped anywhere and iframed
// without an assets folder travelling with it.
const inlineEntry = {
  name: "inline-entry-into-html",
  enforce: "post",
  generateBundle(_options, bundle) {
    const html = Object.values(bundle).find((f) => f.type === "asset" && f.fileName.endsWith(".html"));
    if (!html) return;
    let src = String(html.source);
    for (const [name, chunk] of Object.entries(bundle)) {
      if (chunk.type !== "chunk") continue;
      const tag = new RegExp(`<script type="module"[^>]*src="[^"]*${chunk.fileName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"[^>]*></script>`);
      if (!tag.test(src)) continue;
      // `</script` inside the code would end the inline element early.
      const code = chunk.code.replace(/<\/script/gi, "<\\/script");
      src = src.replace(tag, () => `<script type="module">\n${code}</script>`);
      delete bundle[name];
    }
    html.source = src;
  },
};

export default defineConfig({
  root: "demo",
  base: "./",
  publicDir: false,
  plugins: [inlineEntry],
  build: {
    outDir: "../dist-demo",
    emptyOutDir: true,
    modulePreload: false,
    assetsInlineLimit: 100000000,
    cssCodeSplit: false,
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
