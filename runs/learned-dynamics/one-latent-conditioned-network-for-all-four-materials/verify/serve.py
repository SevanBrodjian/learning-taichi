"""Tiny localhost file server for the WebGPU harness.

`navigator.gpu` is only exposed in a SECURE CONTEXT. `file://` is not one and a plain-HTTP LAN origin
is not one either -- but `http://localhost` IS, so serving from localhost is the whole trick. (This
is the trap that once made a task in this project conclude "this device has no WebGPU".)

The harness also produces megabytes of float32 that have to reach Python, so it POSTs each artifact
here and this writes it straight to disk.

    .venv/Scripts/python.exe runs/.../verify/serve.py [port]
"""
import http.server
import pathlib
import socketserver
import sys

RUN = pathlib.Path(__file__).resolve().parents[1]
OUT = RUN / "verify" / "out"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8752
ALIAS = "/api/data/learning-taichi/runs/learned-dynamics/" \
        "one-latent-conditioned-network-for-all-four-materials/"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(RUN), **kw)

    def log_message(self, fmt, *args):
        if args and "POST" in str(args[0]):
            sys.stderr.write("%s\n" % (fmt % args))

    def translate_path(self, path):
        # so the bespoke page can be previewed exactly as the dashboard serves it
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
