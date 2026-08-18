"""Tiny localhost file server for the four-material WebGPU verification harness.

Two reasons it exists:
  1. `navigator.gpu` is only exposed in a SECURE CONTEXT. `file://` is not one and a plain-HTTP LAN
     origin is not one either -- but `http://localhost` IS, so serving the harness from localhost is
     the whole trick. (This is the trap that made an earlier task conclude "this device has no
     WebGPU".)
  2. The harness produces megabytes of float32 trajectory that has to reach Python. It POSTs each
     artifact here and this writes it straight to disk.

    .venv/Scripts/python.exe runs/material-variants/the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/verify/serve.py [port]
"""
import http.server
import pathlib
import socketserver
import sys

RUN = pathlib.Path(__file__).resolve().parents[1]
OUT = RUN / "verify" / "out"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8741
ALIAS = "/api/data/learning-taichi/runs/material-variants/" \
        "the-demo-mvp-four-materials-on-webgpu-live-on-the-demo-page/"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(RUN), **kw)

    def log_message(self, fmt, *args):        # keep the console readable
        if "POST" in (args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def translate_path(self, path):
        # so bespoke_page.html can be previewed EXACTLY as the dashboard serves it: its media use
        # absolute /api/data/... URLs, and a page reviewed with broken <video> tags is not reviewed
        if path.startswith(ALIAS):
            path = "/" + path[len(ALIAS):]
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_POST(self):
        name = self.path.split("?name=")[-1]
        name = "".join(c for c in name if c.isalnum() or c in "._-")
        if not name:
            self.send_error(400, "bad name")
            return
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_bytes(data)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        print("serving %s at http://localhost:%d/" % (RUN, PORT), flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
