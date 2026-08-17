"""Assemble the standalone, fully self-contained demo page from the parts in this folder.

No CDN, no fetch, no external font: everything is inlined, so the output file drops straight into
a sandboxed iframe or onto a personal website with no build step and no server.

    .venv/Scripts/python.exe runs/material-variants/interactive-simulation-of-one-material/web/build.py
"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

CSS = (HERE / "demo.css").read_text(encoding="utf-8")
PARAMS = (HERE / "params.js").read_text(encoding="utf-8")
MPM = (HERE / "mpm-elastic.js").read_text(encoding="utf-8")
DEMO = (HERE / "demo.js").read_text(encoding="utf-8")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Elastic in a browser -- MLS-MPM, real time</title>
<style>
:root{color-scheme:dark}
html,body{margin:0;background:#0a0e14}
body{padding:22px 20px 30px;font-family:-apple-system,BlinkMacSystemFont,system-ui,"Segoe UI",sans-serif}
.wrap{max-width:820px;margin:0 auto}
h1{font-size:17px;color:#dfe6ee;margin:0 0 4px;font-weight:600}
p.sub{font-size:13px;color:#7f8ea3;margin:0 0 16px;max-width:70ch;line-height:1.55}
%s
</style></head>
<body><div class="wrap">
<h1>Elastic block, MLS-MPM, in the browser</h1>
<p class="sub">Same parameters as the reference GPU simulator, same constitutive law, one CPU thread.
Drag the material to grab it.</p>
<div id="demo"></div>
</div>
<script>%s</script>
<script>%s</script>
<script>%s</script>
<script>MPMDemo.mount(document.getElementById('demo'));</script>
</body></html>
""" % (CSS, PARAMS, MPM, DEMO)

dst = HERE / "demo.html"
dst.write_text(PAGE, encoding="utf-8")
print("wrote", dst, len(PAGE), "bytes")
