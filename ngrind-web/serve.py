import functools
import http.server
import sys

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
    "worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)

class NoStore(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
root = sys.argv[2] if len(sys.argv) > 2 else "target/web"
# Binds to localhost so a locally served build is not reachable from the
# network; pass an address as the third argument to expose it deliberately.
address = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
handler = functools.partial(NoStore, directory=root)
http.server.ThreadingHTTPServer((address, port), handler).serve_forever()
